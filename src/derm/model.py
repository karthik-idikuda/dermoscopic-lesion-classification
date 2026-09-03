"""Model construction, checkpoint loading and the shared inference bundle.

The service must start even when ``models/best_model.pth`` is absent, because
the trained weights are produced on Kaggle and are too large to commit. When no
checkpoint is found the model is still built, but it is flagged
``weights_status="untrained"`` and every downstream consumer surfaces a loud
warning: the geometric, colour and quality features remain fully valid, only the
neural classification is meaningless.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .config import CLASS_CODES, SETTINGS, ModelConfig

logger = logging.getLogger(__name__)

WeightsStatus = Literal["trained", "imagenet", "untrained"]


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick a compute device.

    ``auto`` resolves to CUDA when present and CPU otherwise. Apple MPS is not
    selected automatically because a few of the backward ops Grad-CAM relies on
    still fall back inconsistently there; set ``DERM_DEVICE=mps`` to opt in.
    """
    preference = (preference or "auto").lower()
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if preference == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(preference)


# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #


def build_model(
    config: ModelConfig | None = None, *, pretrained: bool = False
) -> nn.Module:
    """Instantiate the backbone with a 7-way head.

    Prefers ``timm`` (what the training notebook used, so checkpoint keys match)
    and falls back to torchvision if timm is not installed.
    """
    config = config or SETTINGS.model
    try:
        import timm

        model = timm.create_model(
            config.architecture,
            pretrained=pretrained,
            num_classes=config.num_classes,
            drop_rate=config.dropout,
        )
        return model
    except ImportError:  # pragma: no cover - timm is a declared dependency
        logger.warning("timm unavailable, falling back to torchvision.")
    except Exception as exc:  # noqa: BLE001 - offline weight download etc.
        if pretrained:
            logger.warning("Pretrained download failed (%s); using random init.", exc)
            import timm

            return timm.create_model(
                config.architecture,
                pretrained=False,
                num_classes=config.num_classes,
                drop_rate=config.dropout,
            )
        raise

    from torchvision import models as tv_models

    factory = getattr(tv_models, config.architecture, None)
    if factory is None:
        raise ValueError(f"Unknown architecture: {config.architecture}")
    model = factory(weights="DEFAULT" if pretrained else None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, config.num_classes)
    return model


def find_module(model: nn.Module, dotted_path: str) -> nn.Module:
    """Resolve a dotted module path such as ``features.7.2`` or ``conv_head``."""
    target: Any = model
    for part in dotted_path.split("."):
        target = target[int(part)] if part.isdigit() else getattr(target, part)
    return target


def resolve_gradcam_layer(model: nn.Module, config: ModelConfig) -> nn.Module:
    """Find the last convolutional feature map, tolerating architecture swaps."""
    for candidate in (config.gradcam_layer, "conv_head", "features", "blocks"):
        try:
            module = find_module(model, candidate)
        except (AttributeError, IndexError, ValueError):
            continue
        if isinstance(module, nn.Sequential) and len(module):
            return module[-1]
        return module

    # Last resort: the final Conv2d in module order.
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not conv_layers:
        raise RuntimeError("No convolutional layer found for Grad-CAM.")
    return conv_layers[-1]


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #


def build_eval_transform(config: ModelConfig | None = None) -> transforms.Compose:
    """Deterministic transform used for validation, test and serving."""
    config = config or SETTINGS.model
    return transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(list(config.mean), list(config.std)),
        ]
    )


def build_train_transform(config: ModelConfig | None = None) -> transforms.Compose:
    """Augmentation pipeline.

    Dermoscopic images have no canonical orientation, so full dihedral
    augmentation is safe and effective. Affine jitter simulates the small
    translation and scale differences between captures.
    """
    config = config or SETTINGS.model
    return transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(25),
            transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.9, 1.1)),
            transforms.ColorJitter(
                brightness=0.20, contrast=0.20, saturation=0.20, hue=0.02
            ),
            transforms.ToTensor(),
            transforms.Normalize(list(config.mean), list(config.std)),
            transforms.RandomErasing(p=0.20, scale=(0.02, 0.10)),
        ]
    )


def denormalize(tensor: torch.Tensor, config: ModelConfig | None = None) -> np.ndarray:
    """Invert normalisation to recover a displayable ``uint8`` RGB array."""
    config = config or SETTINGS.model
    array = tensor.detach().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    array = array.transpose(1, 2, 0)
    array = np.array(config.std) * array + np.array(config.mean)
    return (np.clip(array, 0, 1) * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Checkpoint IO
# --------------------------------------------------------------------------- #

_PREFIXES = ("module.", "_orig_mod.", "model.")


def _clean_state_dict(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Strip DataParallel / torch.compile / wrapper prefixes from keys."""
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in _PREFIXES:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load a checkpoint saved either as a bare ``state_dict`` or a dict wrapper.

    The training notebook wrote ``torch.save(model.state_dict(), ...)``, while
    :mod:`derm.train` writes a richer dict including class order and calibration
    temperature. Both are accepted.
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)
    metadata: dict[str, Any] = {}

    if isinstance(raw, dict) and any(
        key in raw for key in ("state_dict", "model_state_dict")
    ):
        state = raw.get("state_dict") or raw["model_state_dict"]
        metadata = {k: v for k, v in raw.items() if k not in {"state_dict", "model_state_dict"}}
    elif isinstance(raw, dict):
        state = raw
    else:  # a pickled nn.Module
        state = raw.state_dict()

    return _clean_state_dict(state), metadata


# --------------------------------------------------------------------------- #
# Inference bundle
# --------------------------------------------------------------------------- #


@dataclass
class ModelBundle:
    """Everything inference needs, loaded once and shared across requests."""

    model: nn.Module
    device: torch.device
    config: ModelConfig
    class_codes: tuple[str, ...]
    weights_status: WeightsStatus
    checkpoint_path: Path | None
    temperature: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def is_trained(self) -> bool:
        return self.weights_status == "trained"

    @property
    def transform(self) -> transforms.Compose:
        return build_eval_transform(self.config)

    def prepare(self, image: Image.Image) -> torch.Tensor:
        """PIL image -> normalised ``(1, 3, H, W)`` batch on the model's device."""
        return self.transform(image).unsqueeze(0).to(self.device)

    def ensure_eval(self) -> None:
        """Force deterministic inference mode. Call after any stochastic pass."""
        self.model.eval()

    @torch.inference_mode()
    def logits(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass, serialised so concurrent requests cannot interleave.

        Deliberately does *not* call ``model.eval()``: MC dropout needs to leave
        the model in a stochastic state across repeated calls. The model is put
        in eval mode at construction, and every stochastic helper restores it in
        a ``finally`` block.
        """
        with self._lock:
            return self.model(batch.to(self.device))

    def probabilities(self, batch: torch.Tensor) -> torch.Tensor:
        """Temperature-scaled softmax probabilities."""
        logits = self.logits(batch)
        return torch.softmax(logits / max(self.temperature, 1e-3), dim=1)

    def describe(self) -> dict[str, Any]:
        return {
            "architecture": self.config.architecture,
            "image_size": self.config.image_size,
            "num_classes": len(self.class_codes),
            "class_codes": list(self.class_codes),
            "device": str(self.device),
            "weights_status": self.weights_status,
            "is_trained": self.is_trained,
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
            "temperature": round(self.temperature, 4),
            "warnings": list(self.warnings),
            "metrics": {
                k: v
                for k, v in self.metadata.items()
                if k in {"test_accuracy", "macro_f1", "epoch", "trained_at"}
            },
        }


def create_bundle(
    checkpoint_path: Path | None = None,
    *,
    device: str | None = None,
    allow_pretrained_download: bool = False,
) -> ModelBundle:
    """Build a :class:`ModelBundle`, loading weights if a checkpoint exists."""
    config = SETTINGS.model
    target_device = resolve_device(device or SETTINGS.device)
    path = Path(checkpoint_path) if checkpoint_path else SETTINGS.checkpoint_path

    warnings: list[str] = []
    metadata: dict[str, Any] = {}
    temperature = SETTINGS.inference.temperature
    status: WeightsStatus = "untrained"
    used_path: Path | None = None

    # Read the checkpoint *before* building the model. A checkpoint is the
    # authority on what weights it actually contains, so its declared
    # architecture and image size must win over the ambient settings —
    # otherwise a b0 checkpoint silently fails to populate a b3 graph and the
    # model reports "untrained" for no visible reason.
    state: dict[str, torch.Tensor] | None = None
    if path.exists():
        try:
            state, metadata = load_checkpoint(path)
        except Exception as exc:  # noqa: BLE001 - a bad file must not kill startup
            warnings.append(f"Failed to read checkpoint {path.name}: {exc}")
            logger.exception("Checkpoint read failed")
            state, metadata = None, {}

    declared_arch = metadata.get("architecture")
    if isinstance(declared_arch, str) and declared_arch != config.architecture:
        warnings.append(
            f"Checkpoint declares architecture '{declared_arch}'; building that "
            f"instead of the configured '{config.architecture}'."
        )
        config = replace(config, architecture=declared_arch)

    declared_size = metadata.get("image_size")
    if isinstance(declared_size, int) and declared_size != config.image_size:
        warnings.append(
            f"Checkpoint was trained at {declared_size}px; using that instead of "
            f"the configured {config.image_size}px."
        )
        config = replace(config, image_size=declared_size)

    model = build_model(config, pretrained=False)

    if state is not None:
        try:
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                warnings.append(
                    f"{len(missing)} parameter(s) missing from the checkpoint "
                    f"(first: {missing[0]}); they keep their initial values."
                )
            if unexpected:
                warnings.append(
                    f"{len(unexpected)} unexpected key(s) in the checkpoint were ignored."
                )
            head_loaded = not any("classifier" in key for key in missing)
            status = "trained" if head_loaded else "untrained"
            temperature = float(metadata.get("temperature", temperature))
            used_path = path
            logger.info("Loaded checkpoint from %s", path)
        except Exception as exc:  # noqa: BLE001 - a bad file must not kill startup
            warnings.append(f"Failed to load checkpoint {path.name}: {exc}")
            logger.exception("Checkpoint load failed")
    elif not path.exists():
        warnings.append(
            f"No checkpoint at {path}. The classifier is running with untrained "
            "weights, so its class probabilities are meaningless. Train with "
            "`python -m derm.train` or copy your Kaggle best_model.pth into "
            "models/. Segmentation, ABCD morphometry, image-quality checks and "
            "lesion tracking are unaffected."
        )
        if allow_pretrained_download:
            try:
                model = build_model(config, pretrained=True)
                status = "imagenet"
                warnings.append(
                    "Loaded ImageNet features; the 7-way head is still random."
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not fetch ImageNet weights: %s", exc)

    model.to(target_device).eval()

    class_codes = tuple(metadata.get("class_codes") or CLASS_CODES)
    if len(class_codes) != config.num_classes:
        warnings.append(
            f"Checkpoint declares {len(class_codes)} classes but the model has "
            f"{config.num_classes}; falling back to the default taxonomy."
        )
        class_codes = CLASS_CODES

    return ModelBundle(
        model=model,
        device=target_device,
        config=config,
        class_codes=class_codes,
        weights_status=status,
        checkpoint_path=used_path,
        temperature=temperature,
        metadata=metadata,
        warnings=warnings,
    )


_BUNDLE: ModelBundle | None = None
_BUNDLE_LOCK = threading.Lock()


def get_bundle(*, reload: bool = False, **kwargs) -> ModelBundle:
    """Return the process-wide bundle, constructing it on first use."""
    global _BUNDLE
    if _BUNDLE is None or reload:
        with _BUNDLE_LOCK:
            if _BUNDLE is None or reload:
                _BUNDLE = create_bundle(**kwargs)
    return _BUNDLE


def bundle_loaded() -> bool:
    """Whether the shared bundle has already been constructed.

    Lets a liveness probe report readiness without *forcing* the heavy model
    load, which matters on memory-constrained hosts where eagerly loading torch
    plus the checkpoint during a health check can OOM-kill the container.
    """
    return _BUNDLE is not None


__all__ = [
    "ModelBundle",
    "WeightsStatus",
    "build_eval_transform",
    "build_model",
    "build_train_transform",
    "create_bundle",
    "bundle_loaded",
    "denormalize",
    "find_module",
    "get_bundle",
    "load_checkpoint",
    "resolve_device",
    "resolve_gradcam_layer",
]
