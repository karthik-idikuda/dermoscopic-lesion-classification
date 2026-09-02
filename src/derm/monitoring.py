"""Longitudinal lesion tracking - the "E" (evolution) of ABCDE.

Change over time is one of the strongest predictors of melanoma, and it is the
one thing a single-image classifier structurally cannot see. This module compares
a baseline capture against a follow-up and reports what moved.

An honest caveat drives the whole design: without a physical scale marker in both
frames, absolute growth in millimetres is unknowable, because a change in
apparent size is indistinguishable from the camera moving closer. So every size
metric here is reported *relative to the frame*, flagged as such, and the caller
can supply a known field-of-view width to convert to millimetres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import cv2
import numpy as np

from . import morphology as morphology_module
from . import preprocessing, segmentation


@dataclass
class ChangeMetric:
    """One measured difference between the two captures."""

    name: str
    baseline: float
    followup: float
    delta: float
    relative_change: float  # signed fraction, e.g. 0.25 == +25%
    significant: bool
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "baseline": round(self.baseline, 4),
            "followup": round(self.followup, 4),
            "delta": round(self.delta, 4),
            "relative_change": round(self.relative_change, 4),
            "percent_change": round(self.relative_change * 100, 1),
            "significant": self.significant,
            "note": self.note,
        }


@dataclass
class ChangeReport:
    """Verdict on how a lesion has evolved between two captures."""

    verdict: str  # "stable" | "minor_change" | "significant_change"
    change_score: float  # 0-100
    headline: str
    metrics: list[ChangeMetric] = field(default_factory=list)
    new_colors: list[str] = field(default_factory=list)
    lost_colors: list[str] = field(default_factory=list)
    new_structures: list[str] = field(default_factory=list)
    structural_similarity: float = 0.0
    days_between: int | None = None
    growth_per_month: float | None = None
    recommendation: str = ""
    caveats: list[str] = field(default_factory=list)
    images: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "change_score": round(self.change_score, 1),
            "headline": self.headline,
            "metrics": [m.to_dict() for m in self.metrics],
            "new_colors": self.new_colors,
            "lost_colors": self.lost_colors,
            "new_structures": self.new_structures,
            "structural_similarity": round(self.structural_similarity, 3),
            "days_between": self.days_between,
            "growth_per_month": (
                None if self.growth_per_month is None else round(self.growth_per_month, 4)
            ),
            "recommendation": self.recommendation,
            "caveats": self.caveats,
            "images": self.images,
        }


def _lesion_crop(image: np.ndarray, seg: segmentation.Segmentation, size: int = 192) -> np.ndarray:
    """Crop tightly to the lesion and resize, so shape is compared scale-free."""
    x, y, w, h = seg.bbox
    if w < 4 or h < 4:
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    pad = int(0.12 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image.shape[1], x + w + pad)
    y1 = min(image.shape[0], y + h + pad)
    return cv2.resize(image[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)


def _structural_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM between the two lesion crops, in ``[0, 1]``."""
    try:
        from skimage.metrics import structural_similarity as ssim

        gray_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        value = ssim(gray_a, gray_b, data_range=255)
        return float(np.clip(value, 0.0, 1.0))
    except Exception:  # noqa: BLE001 - fall back to correlation
        gray_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()
        gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()
        if gray_a.std() < 1e-6 or gray_b.std() < 1e-6:
            return 0.0
        return float(np.clip(np.corrcoef(gray_a, gray_b)[0, 1], 0.0, 1.0))


def _color_histogram_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya distance between hue-saturation histograms, in ``[0, 1]``."""
    hsv_a = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    hsv_b = cv2.cvtColor(b, cv2.COLOR_RGB2HSV)
    hist_a = cv2.calcHist([hsv_a], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_b, hist_b, 0, 1, cv2.NORM_MINMAX)
    return float(np.clip(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA), 0, 1))


def _difference_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Heatmap of where the two registered crops differ."""
    diff = cv2.absdiff(
        cv2.cvtColor(a, cv2.COLOR_RGB2LAB), cv2.cvtColor(b, cv2.COLOR_RGB2LAB)
    )
    magnitude = diff.astype(np.float32).sum(axis=2)
    span = magnitude.max() - magnitude.min()
    normalised = (
        (magnitude - magnitude.min()) / span if span > 1e-6 else np.zeros_like(magnitude)
    )
    heatmap = cv2.applyColorMap((normalised * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def _analyze_single(data: bytes) -> tuple[np.ndarray, segmentation.Segmentation, morphology_module.MorphologyFeatures]:
    array = preprocessing.to_array(preprocessing.load_image(data))
    result = preprocessing.preprocess(array)
    seg = segmentation.segment(result.image)
    morph = morphology_module.analyze(result.image, seg)
    return result.image, seg, morph


def compare(
    baseline: bytes,
    followup: bytes,
    *,
    baseline_date: date | str | None = None,
    followup_date: date | str | None = None,
    frame_width_mm: float | None = None,
    include_images: bool = True,
) -> ChangeReport:
    """Compare two captures of the same lesion.

    ``frame_width_mm`` is the real-world width of the field of view. Supply it
    (identically for both captures) to get millimetre measurements; without it
    all sizes stay relative to the frame.
    """
    base_image, base_seg, base_morph = _analyze_single(baseline)
    follow_image, follow_seg, follow_morph = _analyze_single(followup)

    days = _days_between(baseline_date, followup_date)

    metrics: list[ChangeMetric] = []

    # --- size ------------------------------------------------------------- #
    base_diameter = base_morph.diameter_fraction
    follow_diameter = follow_morph.diameter_fraction
    diameter_change = _relative(base_diameter, follow_diameter)
    metrics.append(
        ChangeMetric(
            name="Diameter (fraction of frame width)",
            baseline=base_diameter,
            followup=follow_diameter,
            delta=follow_diameter - base_diameter,
            relative_change=diameter_change,
            significant=abs(diameter_change) >= 0.20,
            note="Relative to frame; not comparable unless both photos used the same magnification.",
        )
    )

    base_area = base_seg.area_ratio
    follow_area = follow_seg.area_ratio
    area_change = _relative(base_area, follow_area)
    metrics.append(
        ChangeMetric(
            name="Area (fraction of frame)",
            baseline=base_area,
            followup=follow_area,
            delta=follow_area - base_area,
            relative_change=area_change,
            significant=abs(area_change) >= 0.30,
        )
    )

    # --- ABCD components -------------------------------------------------- #
    for label, base_value, follow_value, threshold in (
        ("ABCD total dermoscopy score", base_morph.abcd.tds, follow_morph.abcd.tds, 0.15),
        ("Asymmetry (A)", base_morph.abcd.asymmetry, follow_morph.abcd.asymmetry, 0.01),
        ("Border irregularity (B)", base_morph.abcd.border, follow_morph.abcd.border, 0.25),
        ("Colour count (C)", base_morph.abcd.colors, follow_morph.abcd.colors, 0.01),
        ("Colour variance", base_morph.color_variance, follow_morph.color_variance, 0.25),
    ):
        relative = _relative(float(base_value), float(follow_value))
        metrics.append(
            ChangeMetric(
                name=label,
                baseline=float(base_value),
                followup=float(follow_value),
                delta=float(follow_value) - float(base_value),
                relative_change=relative,
                significant=abs(relative) >= threshold and follow_value != base_value,
            )
        )

    # --- appearance ------------------------------------------------------- #
    base_crop = _lesion_crop(base_image, base_seg)
    follow_crop = _lesion_crop(follow_image, follow_seg)
    similarity = _structural_similarity(base_crop, follow_crop)
    color_distance = _color_histogram_distance(base_crop, follow_crop)

    metrics.append(
        ChangeMetric(
            name="Colour distribution distance",
            baseline=0.0,
            followup=color_distance,
            delta=color_distance,
            relative_change=color_distance,
            significant=color_distance >= 0.35,
            note="Bhattacharyya distance between hue-saturation histograms (0 = identical).",
        )
    )

    base_colors = set(base_morph.abcd.colors_present)
    follow_colors = set(follow_morph.abcd.colors_present)
    new_colors = sorted(follow_colors - base_colors)
    lost_colors = sorted(base_colors - follow_colors)
    new_structures = sorted(
        set(follow_morph.abcd.structures_present) - set(base_morph.abcd.structures_present)
    )

    # --- composite score --------------------------------------------------- #
    score = 0.0
    score += min(abs(diameter_change) / 0.5, 1.0) * 28
    score += min(abs(area_change) / 0.8, 1.0) * 18
    score += min(abs(follow_morph.abcd.tds - base_morph.abcd.tds) / 2.0, 1.0) * 22
    score += (1.0 - similarity) * 14
    score += min(color_distance / 0.6, 1.0) * 10
    score += min(len(new_colors) / 2.0, 1.0) * 8
    if "blue_white_veil" in new_structures:
        score += 10
    score = float(np.clip(score, 0.0, 100.0))

    if score >= 55:
        verdict, recommendation = (
            "significant_change",
            "Documented change of this magnitude warrants prompt in-person "
            "dermatological review, ideally within 2-4 weeks, regardless of the "
            "single-image classification.",
        )
    elif score >= 28:
        verdict, recommendation = (
            "minor_change",
            "Some measurable change. Re-photograph in 4-8 weeks under matched "
            "conditions, and seek review if the trend continues.",
        )
    else:
        verdict, recommendation = (
            "stable",
            "No meaningful change detected. Continue routine self-monitoring.",
        )

    growth_per_month = None
    if days and days > 0 and base_diameter > 1e-6:
        growth_per_month = float(diameter_change / (days / 30.44))

    headline_bits = [f"{verdict.replace('_', ' ').capitalize()} ({score:.0f}/100)"]
    if abs(diameter_change) >= 0.05:
        direction = "larger" if diameter_change > 0 else "smaller"
        headline_bits.append(f"{abs(diameter_change) * 100:.0f}% {direction} in frame")
    if new_colors:
        headline_bits.append(f"{len(new_colors)} new colour(s)")
    headline = "; ".join(headline_bits)

    caveats = [
        "Apparent size change is only meaningful if both photographs were taken at "
        "the same magnification and distance. Include a ruler or a fixed-diameter "
        "dermatoscope aperture in both captures for reliable measurement.",
        "Lighting differences shift colour metrics. Colour constancy is applied, but "
        "it cannot fully compensate for a different light source.",
    ]
    if frame_width_mm:
        base_mm = base_diameter * frame_width_mm
        follow_mm = follow_diameter * frame_width_mm
        metrics.insert(
            0,
            ChangeMetric(
                name="Diameter (mm, from supplied field of view)",
                baseline=base_mm,
                followup=follow_mm,
                delta=follow_mm - base_mm,
                relative_change=diameter_change,
                significant=abs(follow_mm - base_mm) >= 1.0,
                note=f"Assumes a {frame_width_mm:.1f} mm field of view in both captures.",
            ),
        )
    else:
        caveats.append(
            "No field-of-view width was supplied, so no absolute millimetre "
            "measurement is reported."
        )
    if not (base_seg.is_reliable and follow_seg.is_reliable):
        caveats.append(
            "Segmentation was unreliable in at least one capture; geometric "
            "comparisons are approximate."
        )
    if days is None:
        caveats.append("No capture dates supplied, so growth rate is not computed.")

    images: dict[str, str] = {}
    if include_images:
        images["baseline"] = preprocessing.encode_png(base_crop, max_size=256)
        images["followup"] = preprocessing.encode_png(follow_crop, max_size=256)
        images["difference"] = preprocessing.encode_png(
            _difference_map(base_crop, follow_crop), max_size=256
        )
        images["baseline_contour"] = preprocessing.encode_png(
            segmentation.overlay_contour(base_image, base_seg), max_size=256
        )
        images["followup_contour"] = preprocessing.encode_png(
            segmentation.overlay_contour(follow_image, follow_seg), max_size=256
        )

    return ChangeReport(
        verdict=verdict,
        change_score=score,
        headline=headline,
        metrics=metrics,
        new_colors=new_colors,
        lost_colors=lost_colors,
        new_structures=new_structures,
        structural_similarity=similarity,
        days_between=days,
        growth_per_month=growth_per_month,
        recommendation=recommendation,
        caveats=caveats,
        images=images,
    )


def _relative(baseline: float, followup: float) -> float:
    """Signed relative change, guarding against a zero baseline."""
    if abs(baseline) < 1e-9:
        return 0.0 if abs(followup) < 1e-9 else 1.0
    return float((followup - baseline) / abs(baseline))


def _days_between(start: Any, end: Any) -> int | None:
    parsed = []
    for value in (start, end):
        if value is None or value == "":
            return None
        if isinstance(value, date):
            parsed.append(value)
            continue
        try:
            parsed.append(date.fromisoformat(str(value)[:10]))
        except ValueError:
            return None
    delta = (parsed[1] - parsed[0]).days
    return delta if delta >= 0 else None


__all__ = ["ChangeMetric", "ChangeReport", "compare"]
