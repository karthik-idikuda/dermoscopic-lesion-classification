"""End-to-end analysis pipeline.

One call takes raw uploaded bytes and returns everything the UI and the PDF
report need. The ordering matters:

1. **Quality assessment** on the untouched frame, so defects are reported
   against what the user actually submitted.
2. **Vignette crop** to establish a shared coordinate frame for every overlay.
3. **Restoration** (hair inpainting, colour constancy) into a *separate* image
   used only for geometry. The classifier is deliberately fed the un-restored
   frame, because that is the distribution HAM10000 was trained on - feeding it
   colour-normalised input would be a silent train/serve skew.
4. **Segmentation and ABCD morphometry** on the restored image, where hair would
   otherwise wreck border measurements.
5. **Classification with uncertainty**, then **Grad-CAM**, then severity grading,
   then the narrative report.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

from . import gradcam as gradcam_module
from . import morphology as morphology_module
from . import preprocessing, quality, report as report_module, segmentation, severity
from . import uncertainty as uncertainty_module
from .config import LESION_CLASSES, MEDICAL_DISCLAIMER, SETTINGS
from .model import ModelBundle, get_bundle, resolve_gradcam_layer

# Rendered-overlay resolution. Overridable so a memory-constrained deployment
# can shrink the transient RGB/PNG buffers (a 512px frame is ~0.75MB per copy
# and several are held at once during rendering). Set DERM_DISPLAY_SIZE=384 on
# a small host to trim peak memory with only a minor drop in overlay detail.
import os as _os

DISPLAY_SIZE = int(_os.environ.get("DERM_DISPLAY_SIZE", "512"))


@dataclass
class AnalysisOptions:
    """Per-request switches. Defaults are the full pipeline."""

    hair_removal: bool = True
    color_constancy: bool = True
    vignette_crop: bool = True
    segmentation: bool = True
    morphometry: bool = True
    gradcam: bool = True
    gradcam_method: gradcam_module.Method = "gradcam++"
    colormap: str = "jet"
    tta: bool = True
    mc_dropout: bool = True
    include_images: bool = True
    narrative: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "AnalysisOptions":
        if not data:
            return cls()
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in valid and v is not None})


@dataclass
class Prediction:
    """One class's probability, enriched with its clinical descriptor."""

    code: str
    name: str
    short_name: str
    probability: float
    malignancy: str
    color: str

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "name": self.name,
            "short_name": self.short_name,
            "probability": round(self.probability, 6),
            "percentage": round(self.probability * 100, 2),
            "malignancy": self.malignancy,
            "color": self.color,
        }


@dataclass
class AnalysisResult:
    """Complete result of one image analysis."""

    case_id: str
    created_at: str
    predictions: list[Prediction]
    severity: severity.SeverityAssessment
    quality: quality.QualityReport
    uncertainty: uncertainty_module.UncertaintyReport | None
    morphology: morphology_module.MorphologyFeatures | None
    segmentation: segmentation.Segmentation | None
    cam: gradcam_module.CAMResult | None
    attention: dict[str, float] | None
    preprocessing_steps: list[str]
    narrative: report_module.ClinicalNarrative | None
    images: dict[str, str] = field(default_factory=dict)
    model_info: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    filename: str | None = None

    @property
    def top(self) -> Prediction:
        return self.predictions[0]

    def to_dict(self, *, include_images: bool = True) -> dict:
        return {
            "case_id": self.case_id,
            "created_at": self.created_at,
            "filename": self.filename,
            # Exact identity of the bytes analyzed. The frontend verifies this
            # digest before rendering so a late response can never be shown for
            # a newer file selection.
            "source": self.source,
            "prediction": {
                "code": self.top.code,
                "name": self.top.name,
                "short_name": self.top.short_name,
                "confidence": round(self.top.probability, 6),
                "percentage": round(self.top.probability * 100, 2),
                "malignancy": self.top.malignancy,
                "description": LESION_CLASSES[self.top.code].description,
                "management": LESION_CLASSES[self.top.code].management,
            },
            "probabilities": [p.to_dict() for p in self.predictions],
            "severity": self.severity.to_dict(),
            "quality": self.quality.to_dict(),
            "uncertainty": self.uncertainty.to_dict() if self.uncertainty else None,
            "morphology": self.morphology.to_dict() if self.morphology else None,
            "segmentation": self.segmentation.to_dict() if self.segmentation else None,
            "explanation": {
                **(self.cam.to_dict() if self.cam else {}),
                "attention": (
                    {k: (None if v == float("inf") else round(v, 3))
                     for k, v in self.attention.items()}
                    if self.attention
                    else None
                ),
            },
            "preprocessing": self.preprocessing_steps,
            "narrative": self.narrative.to_dict() if self.narrative else None,
            "images": self.images if include_images else {},
            "model": self.model_info,
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
            "disclaimer": MEDICAL_DISCLAIMER,
        }


def _resize_for_display(array: np.ndarray, size: int = DISPLAY_SIZE) -> np.ndarray:
    import cv2

    height, width = array.shape[:2]
    if max(height, width) <= size:
        return array
    scale = size / max(height, width)
    return cv2.resize(
        array,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def analyze_image(
    data: bytes,
    *,
    options: AnalysisOptions | None = None,
    bundle: ModelBundle | None = None,
    filename: str | None = None,
) -> AnalysisResult:
    """Run the full pipeline on raw image bytes."""
    options = options or AnalysisOptions()
    bundle = bundle or get_bundle()
    timings: dict[str, float] = {}
    started = time.perf_counter()

    def mark(name: str, since: float) -> float:
        now = time.perf_counter()
        timings[name] = (now - since) * 1000.0
        return now

    # -- 1. decode and assess quality -------------------------------------- #
    stage = time.perf_counter()
    image = preprocessing.load_image(data)
    base = preprocessing.to_array(image)
    quality_report = quality.assess(base)
    stage = mark("decode_and_quality", stage)

    # -- 2. establish the shared analysis frame ---------------------------- #
    frame = base
    steps: list[str] = []
    if options.vignette_crop and preprocessing.detect_vignette(frame) > 0.35:
        cropped = preprocessing.crop_vignette(frame)
        if cropped.shape != frame.shape:
            frame = cropped
            steps.append("cropped lens vignette")

    # -- 3. restoration for geometry only ---------------------------------- #
    geometry_image = frame
    hair_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if options.hair_removal:
        hair_mask, hair_ratio = preprocessing.detect_hair(frame)
        if np.any(hair_mask):
            geometry_image = preprocessing.remove_hair(geometry_image, hair_mask)
            steps.append(f"inpainted hair ({hair_ratio * 100:.1f}% of pixels)")
    if options.color_constancy:
        geometry_image = preprocessing.shades_of_gray(geometry_image)
        steps.append("shades-of-gray colour constancy")
    stage = mark("preprocess", stage)

    # -- 4. segmentation and morphometry ----------------------------------- #
    seg = segmentation.segment(geometry_image) if options.segmentation else None
    stage = mark("segmentation", stage)

    morph = (
        morphology_module.analyze(geometry_image, seg)
        if (options.morphometry and seg is not None)
        else None
    )
    stage = mark("morphometry", stage)

    # -- 5. classification with uncertainty -------------------------------- #
    model_frame = preprocessing.to_pil(frame)
    batch = bundle.prepare(model_frame)

    uncertainty_report = uncertainty_module.estimate(
        bundle,
        batch,
        use_tta=options.tta,
        use_mc_dropout=options.mc_dropout,
        n_tta=SETTINGS.inference.tta_transforms,
        mc_passes=SETTINGS.inference.mc_dropout_passes,
        low_confidence=SETTINGS.inference.low_confidence_threshold,
        high_entropy=SETTINGS.inference.high_entropy_threshold,
    )
    probabilities = uncertainty_report.probabilities
    stage = mark("classification", stage)

    predictions = _build_predictions(probabilities, bundle.class_codes)

    # -- 6. Grad-CAM -------------------------------------------------------- #
    display = _resize_for_display(frame)
    cam_result: gradcam_module.CAMResult | None = None
    attention: dict[str, float] | None = None
    cam_map: np.ndarray | None = None

    if options.gradcam:
        try:
            explainer = gradcam_module.GradCAM(
                bundle.model, resolve_gradcam_layer(bundle.model, bundle.config)
            )
            cam_result, _ = explainer(
                batch,
                int(probabilities.argmax()),
                method=options.gradcam_method,
                output_size=(display.shape[0], display.shape[1]),
            )
            cam_map = cam_result.cam
            if seg is not None:
                attention = gradcam_module.attention_alignment(cam_map, seg.mask)
        except Exception as exc:  # noqa: BLE001 - explanation is best-effort
            steps.append(f"Grad-CAM unavailable: {exc}")
    stage = mark("gradcam", stage)

    # -- 7. severity grading ------------------------------------------------ #
    assessment = severity.grade(
        probabilities,
        bundle.class_codes,
        morphology=morph,
        uncertainty=uncertainty_report,
        quality=quality_report,
        attention_verdict=attention.get("verdict") if attention else None,
        model_is_trained=bundle.is_trained,
    )
    stage = mark("severity", stage)

    # -- 8. rendered images ------------------------------------------------- #
    images: dict[str, str] = {}
    if options.include_images:
        images["original"] = preprocessing.encode_png(display)
        if steps and geometry_image is not frame:
            images["restored"] = preprocessing.encode_png(
                _resize_for_display(geometry_image)
            )
        if np.any(hair_mask):
            images["hair_mask"] = preprocessing.encode_png(
                _resize_for_display(np.dstack([hair_mask] * 3))
            )
        if seg is not None:
            images["segmentation"] = preprocessing.encode_png(
                _resize_for_display(
                    segmentation.mask_preview(geometry_image, seg)
                )
            )
            images["contour"] = preprocessing.encode_png(
                _resize_for_display(segmentation.overlay_contour(geometry_image, seg))
            )
        if cam_map is not None:
            images["heatmap"] = preprocessing.encode_png(
                gradcam_module.colorize(cam_map, options.colormap)
            )
            images["overlay"] = preprocessing.encode_png(
                gradcam_module.overlay(display, cam_map, colormap=options.colormap)
            )
            images["cam_contour"] = preprocessing.encode_png(
                gradcam_module.contour_overlay(display, cam_map)
            )
    stage = mark("rendering", stage)

    # -- 9. narrative ------------------------------------------------------- #
    narrative = None
    if options.narrative:
        narrative = report_module.generate_narrative(
            predictions=[p.to_dict() for p in predictions],
            assessment=assessment,
            morphology=morph,
            quality_report=quality_report,
            uncertainty_report=uncertainty_report,
            attention=attention,
            model_is_trained=assessment.neural_usable,
            preprocessing_steps=steps,
        )
    mark("narrative", stage)

    timings["total"] = (time.perf_counter() - started) * 1000.0

    return AnalysisResult(
        case_id=uuid.uuid4().hex[:12],
        created_at=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        predictions=predictions,
        severity=assessment,
        quality=quality_report,
        uncertainty=uncertainty_report,
        morphology=morph,
        segmentation=seg,
        cam=cam_result,
        attention=attention,
        preprocessing_steps=steps,
        narrative=narrative,
        images=images,
        model_info=bundle.describe(),
        timings_ms=timings,
        source={
            "filename": filename,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "width": image.width,
            "height": image.height,
        },
        filename=filename,
    )


def _build_predictions(
    probabilities: np.ndarray, class_codes: tuple[str, ...]
) -> list[Prediction]:
    """Sort probabilities descending and attach clinical metadata."""
    items: list[Prediction] = []
    for index, code in enumerate(class_codes):
        if index >= len(probabilities):
            break
        lesion = LESION_CLASSES[code]
        items.append(
            Prediction(
                code=code,
                name=lesion.name,
                short_name=lesion.short_name,
                probability=float(probabilities[index]),
                malignancy=lesion.malignancy,
                color=lesion.color,
            )
        )
    items.sort(key=lambda p: p.probability, reverse=True)
    return items


@torch.inference_mode()
def quick_probabilities(
    data: bytes, bundle: ModelBundle | None = None
) -> dict[str, float]:
    """Classification only - used by the batch endpoint and by evaluation code."""
    bundle = bundle or get_bundle()
    image = preprocessing.load_image(data)
    batch = bundle.prepare(image)
    probabilities = bundle.probabilities(batch)[0].detach().cpu().numpy()
    return {
        code: float(probabilities[i])
        for i, code in enumerate(bundle.class_codes)
        if i < len(probabilities)
    }


__all__ = [
    "AnalysisOptions",
    "AnalysisResult",
    "Prediction",
    "analyze_image",
    "quick_probabilities",
]
