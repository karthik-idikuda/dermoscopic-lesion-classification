"""Automatic lesion segmentation.

The ABCD morphometry and the lesion-change tracker both need to know which
pixels belong to the lesion. A learned segmenter would be better, but there are
no lesion masks in HAM10000, so this uses a classical pipeline that is
deterministic, fast and needs no extra training data:

    contrast enhancement -> Otsu threshold on a lesion-enhanced channel ->
    morphological cleanup -> hole filling -> best central component

When thresholding clearly fails (mask covers almost everything or almost
nothing) the module falls back to a centred ellipse and lowers its reported
confidence, so downstream code can discount the geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WORKING_SIZE = 384


@dataclass
class Segmentation:
    """Binary lesion mask plus descriptive geometry."""

    mask: np.ndarray  # uint8 {0, 255}, same H/W as the input image
    contour: np.ndarray | None  # (N, 1, 2) largest external contour
    area_ratio: float  # lesion pixels / total pixels
    centroid: tuple[float, float]  # (x, y) in pixels
    equivalent_diameter: float  # px, diameter of a circle of equal area
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float  # 0-1 trust in this mask
    method: str  # "otsu" | "adaptive" | "fallback_ellipse"

    @property
    def is_reliable(self) -> bool:
        return self.confidence >= 0.5

    def to_dict(self) -> dict:
        return {
            "area_ratio": round(self.area_ratio, 4),
            "equivalent_diameter_px": round(self.equivalent_diameter, 1),
            "centroid": [round(self.centroid[0], 1), round(self.centroid[1], 1)],
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "reliable": self.is_reliable,
        }


def _lesion_channel(array: np.ndarray) -> np.ndarray:
    """Build a single channel in which the lesion is bright and skin is dark.

    Lesions are darker than skin in luminance but also shift in the blue-yellow
    (LAB ``b``) direction. Combining inverted luminance with the inverted blue
    channel handles both pigmented and erythematous lesions better than either
    alone.
    """
    lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
    luminance = lab[..., 0].astype(np.float32)
    blue = array[..., 2].astype(np.float32)

    def _norm(channel: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(channel, (1, 99))
        if hi - lo < 1e-3:
            return np.zeros_like(channel)
        return np.clip((channel - lo) / (hi - lo), 0, 1)

    combined = 0.65 * (1.0 - _norm(luminance)) + 0.35 * (1.0 - _norm(blue))
    enhanced = (combined * 255).astype(np.uint8)
    return cv2.GaussianBlur(enhanced, (7, 7), 0)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes by re-drawing every external contour filled."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _pick_component(mask: np.ndarray) -> np.ndarray:
    """Keep the component that best balances size against being central.

    Dermoscopic framing centres the lesion, so a large blob at the edge is far
    more likely to be shadow or a ruler than the lesion of interest.
    """
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if count <= 1:
        return mask

    height, width = mask.shape
    centre = np.array([width / 2.0, height / 2.0])
    max_distance = np.linalg.norm(centre)

    best_label, best_score = 0, -1.0
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 0.002 * mask.size:
            continue
        distance = float(np.linalg.norm(np.array(centroids[label]) - centre))
        centrality = 1.0 - min(1.0, distance / max_distance)
        score = (area / mask.size) * (0.35 + 0.65 * centrality)
        if score > best_score:
            best_label, best_score = label, score

    if best_label == 0:
        return mask
    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def _fallback_ellipse(shape: tuple[int, int]) -> np.ndarray:
    """Centred ellipse covering ~35% of the frame, used when Otsu fails."""
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (width // 2, height // 2),
        (int(width * 0.33), int(height * 0.33)),
        0,
        0,
        360,
        255,
        thickness=cv2.FILLED,
    )
    return mask


def segment(array: np.ndarray) -> Segmentation:
    """Segment the dominant lesion in an RGB image."""
    original_shape = array.shape[:2]
    scale = min(1.0, WORKING_SIZE / max(original_shape))
    working = (
        cv2.resize(array, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else array
    )

    channel = _lesion_channel(working)
    method = "otsu"

    _, mask = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = _fill_holes(mask)
    mask = _pick_component(mask)

    ratio = float(np.count_nonzero(mask)) / mask.size
    confidence = 0.85

    # Otsu splits a bimodal histogram; on a uniform lesion-free patch it splits
    # noise instead, which shows up as an absurd coverage ratio.
    if ratio < 0.01 or ratio > 0.92:
        method = "adaptive"
        blurred = cv2.medianBlur(channel, 9)
        mask = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -6
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = _fill_holes(mask)
        mask = _pick_component(mask)
        ratio = float(np.count_nonzero(mask)) / mask.size
        confidence = 0.55

    if ratio < 0.005 or ratio > 0.95:
        method = "fallback_ellipse"
        mask = _fallback_ellipse(mask.shape)
        ratio = float(np.count_nonzero(mask)) / mask.size
        confidence = 0.15

    # A mask hugging the frame edge usually means the lesion is cropped.
    border = np.concatenate(
        [mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]]
    )
    border_touch = float(np.count_nonzero(border)) / border.size
    if border_touch > 0.4 and method != "fallback_ellipse":
        confidence *= 0.6

    if scale < 1.0:
        mask = cv2.resize(
            mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST
        )
        mask = (mask > 127).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea) if contours else None

    area = float(np.count_nonzero(mask))
    if area > 0:
        moments = cv2.moments(mask, binaryImage=True)
        centroid = (
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        )
    else:
        centroid = (original_shape[1] / 2.0, original_shape[0] / 2.0)

    bbox = cv2.boundingRect(contour) if contour is not None else (0, 0, 0, 0)

    return Segmentation(
        mask=mask,
        contour=contour,
        area_ratio=area / mask.size,
        centroid=centroid,
        equivalent_diameter=float(np.sqrt(4.0 * area / np.pi)),
        bbox=bbox,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        method=method,
    )


def overlay_contour(
    array: np.ndarray,
    segmentation: Segmentation,
    *,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw the lesion outline and its bounding box on a copy of the image."""
    canvas = array.copy()
    if segmentation.contour is not None:
        cv2.drawContours(canvas, [segmentation.contour], -1, color, thickness)
        x, y, w, h = segmentation.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 255), 1)
    return canvas


def mask_preview(array: np.ndarray, segmentation: Segmentation) -> np.ndarray:
    """Tint the lesion region so the mask is legible over the original image."""
    canvas = array.copy().astype(np.float32)
    tint = np.zeros_like(canvas)
    tint[..., 1] = 255.0  # green
    alpha = (segmentation.mask > 0).astype(np.float32)[..., None] * 0.35
    blended = canvas * (1 - alpha) + tint * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


__all__ = ["Segmentation", "mask_preview", "overlay_contour", "segment"]
