"""Grad-CAM and Grad-CAM++ visual explanations.

Differences from the straightforward notebook implementation, all of which
matter once this runs inside a long-lived server:

* Hooks are registered and removed through a context manager, so repeated calls
  cannot stack duplicate hooks or leak activations between requests.
* Gradients are enabled explicitly, so Grad-CAM still works when the caller sits
  inside ``torch.inference_mode()``.
* Grad-CAM++ is available, which localises better when a class appears in
  several places in the frame.
* The resulting map is scored against the lesion mask, giving an *attention
  alignment* number that answers a question a reviewer will always ask: did the
  network look at the lesion, or at a hair, a ruler or the vignette?
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Method = Literal["gradcam", "gradcam++"]

COLORMAPS = {
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
}


@dataclass
class CAMResult:
    """A class-activation map and its derived interpretability statistics."""

    cam: np.ndarray  # float32 (H, W) in [0, 1]
    class_index: int
    method: Method
    peak_xy: tuple[int, int]
    concentration: float  # fraction of total activation in the top 10% of pixels

    def to_dict(self) -> dict:
        return {
            "class_index": self.class_index,
            "method": self.method,
            "peak": list(self.peak_xy),
            "concentration": round(self.concentration, 3),
        }


class GradCAM:
    """Gradient-weighted class activation mapping for a single target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

    # ------------------------------------------------------------------ hooks
    def _forward_hook(self, _module, _inputs, output) -> None:
        self._activations = output

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0]

    @contextmanager
    def _hooked(self) -> Iterator[None]:
        handles = [
            self.target_layer.register_forward_hook(self._forward_hook),
            self.target_layer.register_full_backward_hook(self._backward_hook),
        ]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            self._activations = None
            self._gradients = None

    # ---------------------------------------------------------------- compute
    def __call__(
        self,
        batch: torch.Tensor,
        class_index: int | None = None,
        *,
        method: Method = "gradcam",
        output_size: tuple[int, int] | None = None,
    ) -> tuple[CAMResult, torch.Tensor]:
        """Generate a CAM for ``batch`` (a single-image batch).

        ``output_size`` is ``(height, width)``; defaults to the input spatial
        size. Returns the CAM result and the raw logits so callers do not need a
        second forward pass.
        """
        if batch.dim() != 4 or batch.shape[0] != 1:
            raise ValueError("Grad-CAM expects a batch of exactly one image.")

        was_training = self.model.training
        self.model.eval()

        # Two nested guards are needed, not one. A caller may already be inside
        # torch.inference_mode(), and tensors created there are "inference
        # tensors" that autograd refuses outright - enable_grad() alone does not
        # lift that. inference_mode(False) re-enters normal mode so the clone
        # below is a regular tensor that can carry a graph.
        with self._hooked(), torch.inference_mode(False), torch.enable_grad():
            inputs = batch.detach().clone().requires_grad_(True)
            logits = self.model(inputs)

            if class_index is None:
                class_index = int(logits.argmax(dim=1).item())

            self.model.zero_grad(set_to_none=True)
            logits[0, class_index].backward(retain_graph=False)

            if self._activations is None or self._gradients is None:
                raise RuntimeError(
                    "Grad-CAM captured no activations. The target layer is "
                    "probably not part of the forward graph."
                )

            activations = self._activations[0].detach()  # (C, h, w)
            gradients = self._gradients[0].detach()  # (C, h, w)
            detached_logits = logits.detach()

        weights = (
            self._gradcam_pp_weights(activations, gradients)
            if method == "gradcam++"
            else gradients.mean(dim=(1, 2))
        )

        cam = torch.einsum("c,chw->hw", weights, activations)
        cam = F.relu(cam)

        target_hw = output_size or (batch.shape[2], batch.shape[3])
        cam = F.interpolate(
            cam[None, None], size=target_hw, mode="bilinear", align_corners=False
        )[0, 0]

        cam_np = cam.cpu().numpy().astype(np.float32)
        span = float(cam_np.max() - cam_np.min())
        cam_np = (
            (cam_np - cam_np.min()) / span
            if span > 1e-8
            else np.zeros_like(cam_np)
        )

        if was_training:
            self.model.train()

        peak_index = int(np.argmax(cam_np))
        peak_xy = (peak_index % cam_np.shape[1], peak_index // cam_np.shape[1])

        return (
            CAMResult(
                cam=cam_np,
                class_index=class_index,
                method=method,
                peak_xy=peak_xy,
                concentration=_concentration(cam_np),
            ),
            detached_logits,
        )

    @staticmethod
    def _gradcam_pp_weights(
        activations: torch.Tensor, gradients: torch.Tensor
    ) -> torch.Tensor:
        """Grad-CAM++ channel weights from second- and third-order gradients.

        Uses the closed form of Chattopadhyay et al. (2018), where the ReLU'd
        first-order gradient stands in for the exponential factor.
        """
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        sum_activations = activations.sum(dim=(1, 2), keepdim=True)
        denominator = 2.0 * grad_2 + sum_activations * grad_3
        denominator = torch.where(
            denominator.abs() > 1e-8, denominator, torch.ones_like(denominator)
        )
        alpha = grad_2 / denominator
        return (alpha * F.relu(gradients)).sum(dim=(1, 2))


def _concentration(cam: np.ndarray, top_fraction: float = 0.10) -> float:
    """Share of total activation held by the hottest ``top_fraction`` of pixels.

    A diffuse map (low concentration) means the evidence is spread over the whole
    frame, which usually means the prediction is not anchored to the lesion.
    """
    total = float(cam.sum())
    if total <= 1e-8:
        return 0.0
    k = max(1, int(cam.size * top_fraction))
    top = np.partition(cam.ravel(), -k)[-k:]
    return float(top.sum() / total)


def attention_alignment(cam: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Measure how much of the CAM's evidence falls inside the lesion mask.

    ``inside_ratio`` is the share of total activation within the lesion.
    ``lift`` compares mean activation inside versus outside; a lift near or below
    1.0 means the network is not preferentially attending to the lesion at all.
    """
    if cam.shape != mask.shape:
        mask = cv2.resize(
            mask, (cam.shape[1], cam.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    inside = mask > 0
    outside = ~inside
    total = float(cam.sum())
    if total <= 1e-8 or not inside.any():
        return {"inside_ratio": 0.0, "lift": 0.0, "verdict": 0.0}

    inside_ratio = float(cam[inside].sum() / total)
    mean_inside = float(cam[inside].mean())
    mean_outside = float(cam[outside].mean()) if outside.any() else 0.0
    lift = mean_inside / mean_outside if mean_outside > 1e-8 else float("inf")
    lift = min(lift, 10.0)

    # Blend the two into a single 0-1 trust value for the UI.
    verdict = float(np.clip(0.6 * inside_ratio + 0.4 * min(lift / 3.0, 1.0), 0.0, 1.0))
    return {"inside_ratio": inside_ratio, "lift": lift, "verdict": verdict}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def colorize(cam: np.ndarray, colormap: str = "jet") -> np.ndarray:
    """Turn a ``[0, 1]`` CAM into an RGB heatmap."""
    code = COLORMAPS.get(colormap, cv2.COLORMAP_JET)
    heatmap = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), code)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def overlay(
    image: np.ndarray,
    cam: np.ndarray,
    *,
    alpha: float = 0.45,
    colormap: str = "jet",
    threshold: float = 0.25,
) -> np.ndarray:
    """Blend a heatmap over the image, leaving cold regions untinted.

    Masking below ``threshold`` keeps the original skin texture visible where the
    model found nothing, which reads far better clinically than a uniform wash.
    """
    if cam.shape[:2] != image.shape[:2]:
        cam = cv2.resize(cam, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    heatmap = colorize(cam, colormap).astype(np.float32)
    base = image.astype(np.float32)
    weight = (np.clip((cam - threshold) / max(1e-6, 1.0 - threshold), 0, 1) * alpha)[
        ..., None
    ]
    blended = base * (1 - weight) + heatmap * weight
    return np.clip(blended, 0, 255).astype(np.uint8)


def contour_overlay(
    image: np.ndarray, cam: np.ndarray, *, level: float = 0.6
) -> np.ndarray:
    """Draw an iso-contour of the CAM: a cleaner read than a full heatmap."""
    if cam.shape[:2] != image.shape[:2]:
        cam = cv2.resize(cam, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    canvas = image.copy()
    binary = (cam >= level).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (255, 64, 64), 2)
    return canvas


__all__ = [
    "CAMResult",
    "GradCAM",
    "Method",
    "attention_alignment",
    "colorize",
    "contour_overlay",
    "overlay",
]
