"""Dermoscopy-specific image preprocessing.

Dermoscopic photographs carry artefacts that classical and deep models both
trip over: terminal hairs crossing the lesion, the strong colour cast of
whatever light source the dermatoscope used, and black circular vignetting from
the lens barrel. This module removes those before segmentation and morphometry.

The neural classifier is deliberately fed the *un*-restored image by default
(it was trained on raw HAM10000 crops) while the geometric ABCD analysis runs on
the cleaned image, where hair would otherwise destroy border measurements.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps

# --------------------------------------------------------------------------- #
# Basic conversions
# --------------------------------------------------------------------------- #


def load_image(data: bytes) -> Image.Image:
    """Decode uploaded bytes into an upright RGB :class:`PIL.Image.Image`.

    EXIF orientation is applied so phone photographs are not silently rotated,
    and any alpha channel is flattened onto white.
    """
    if not data:
        raise ValueError("Empty image payload.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # noqa: BLE001 - surface a clean API error
        raise ValueError(f"Unsupported or corrupt image file: {exc}") from exc

    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA", "P"}:
        background = Image.new("RGB", image.size, (255, 255, 255))
        converted = image.convert("RGBA")
        background.paste(converted, mask=converted.split()[-1])
        image = background
    return image.convert("RGB")


def to_array(image: Image.Image) -> np.ndarray:
    """RGB :class:`PIL.Image` -> ``uint8`` ``(H, W, 3)`` array."""
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def to_pil(array: np.ndarray) -> Image.Image:
    """``uint8``/float array -> RGB :class:`PIL.Image`."""
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(array, mode="RGB")


def encode_png(array: np.ndarray, *, max_size: int = 512) -> str:
    """Encode an array as a base64 ``data:`` URI for direct use in an ``<img>``."""
    image = to_pil(array)
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def center_square_crop(array: np.ndarray) -> np.ndarray:
    """Crop to the largest centred square, preserving lesion aspect ratio."""
    height, width = array.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return array[top : top + side, left : left + side]


# --------------------------------------------------------------------------- #
# Artefact handling
# --------------------------------------------------------------------------- #


@dataclass
class PreprocessResult:
    """Cleaned image plus a record of what was actually changed."""

    image: np.ndarray
    hair_mask: np.ndarray
    hair_ratio: float
    hair_removed: bool
    color_constancy_applied: bool
    vignette_cropped: bool
    steps: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return ", ".join(self.steps) if self.steps else "no restoration required"


_HAIR_KERNEL_SIZE = 17
_LINE_KERNEL_ANGLES = (0, 30, 60, 90, 120, 150)


def _line_kernel(size: int, angle_degrees: float) -> np.ndarray:
    """A binary structuring element containing a single line through the centre."""
    kernel = np.zeros((size, size), dtype=np.uint8)
    centre = size // 2
    theta = np.deg2rad(angle_degrees)
    dx, dy = np.cos(theta), np.sin(theta)
    start = (
        int(round(centre - dx * centre)),
        int(round(centre - dy * centre)),
    )
    end = (
        int(round(centre + dx * centre)),
        int(round(centre + dy * centre)),
    )
    cv2.line(kernel, start, end, 1, thickness=1)
    return kernel


def detect_hair(array: np.ndarray, *, min_ratio: float = 0.005) -> tuple[np.ndarray, float]:
    """Locate hair-like structures with a morphological black-hat filter.

    Hairs are thin, elongated and darker than the surrounding skin, so a
    black-hat transform with directional line kernels isolates them while
    leaving the (broad, blob-shaped) lesion untouched.

    Returns the binary mask and the fraction of pixels it covers.
    """
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    # A 5x5 blur suppresses sensor noise without erasing 1-3 px hair shafts.
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Directional line kernels catch hairs at any angle. Each is drawn as an
    # actual line through the kernel centre; rotating a flat rectangular kernel
    # with warpAffine silently produces empty kernels for most angles.
    responses = []
    for angle in _LINE_KERNEL_ANGLES:
        kernel = _line_kernel(_HAIR_KERNEL_SIZE, angle)
        if kernel.sum() == 0:  # defensive; should not happen
            continue
        responses.append(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel))

    if not responses:
        return np.zeros(gray.shape, dtype=np.uint8), 0.0

    blackhat = np.maximum.reduce(responses)
    # Otsu adapts to however dark the hairs are in this particular image.
    threshold, mask = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if threshold < 8:  # essentially flat response -> no hair present
        return np.zeros(gray.shape, dtype=np.uint8), 0.0

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    # Keep only genuinely elongated components; drop compact blobs, which are
    # much more likely to be globules or dots belonging to the lesion itself.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        extent = max(w, h)
        if area < 12 or extent < 9:
            # Too small to be a hair shaft; almost certainly sensor noise or a
            # pigment dot belonging to the lesion.
            continue
        elongation = extent / max(1.0, min(w, h))
        fill = area / max(1.0, float(w * h))
        if elongation >= 2.0 and fill <= 0.7:
            filtered[labels == label] = 255

    ratio = float(np.count_nonzero(filtered)) / filtered.size
    if ratio < min_ratio:
        return np.zeros_like(filtered), ratio
    return filtered, ratio


def remove_hair(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint hair pixels (the DullRazor idea, with Telea inpainting)."""
    if not np.any(mask):
        return array.copy()
    dilated = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )
    bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    restored = cv2.inpaint(bgr, dilated, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)


def shades_of_gray(array: np.ndarray, power: int = 6) -> np.ndarray:
    """Shades-of-Gray colour constancy.

    Normalises the illuminant so that colour-variegation scores are comparable
    between images captured on different dermatoscopes. ``power=6`` is the
    setting reported to work best on dermoscopic data.
    """
    image = array.astype(np.float32)
    flat = image.reshape(-1, 3)
    norm = np.power(np.power(flat, power).mean(axis=0), 1.0 / power)
    norm_total = np.sqrt(np.sum(norm**2))
    if norm_total <= 1e-6:
        return array.copy()
    gains = (norm_total / np.sqrt(3.0)) / np.maximum(norm, 1e-6)
    corrected = image * gains
    return np.clip(corrected, 0, 255).astype(np.uint8)


def detect_vignette(array: np.ndarray, dark_threshold: int = 28) -> float:
    """Fraction of the image border that is near-black (lens barrel shadow)."""
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    band = max(2, min(height, width) // 12)
    border = np.concatenate(
        [
            gray[:band, :].ravel(),
            gray[-band:, :].ravel(),
            gray[:, :band].ravel(),
            gray[:, -band:].ravel(),
        ]
    )
    return float(np.mean(border < dark_threshold))


def crop_vignette(array: np.ndarray, dark_threshold: int = 28) -> np.ndarray:
    """Crop away a black circular frame so it cannot be mistaken for lesion."""
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    mask = (gray >= dark_threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return array
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[largest]
    if area < 0.25 * gray.size or w < 32 or h < 32:
        return array
    # Inset slightly so the residual soft edge of the barrel is excluded too.
    inset = int(0.02 * min(w, h))
    x, y = x + inset, y + inset
    w, h = max(16, w - 2 * inset), max(16, h - 2 * inset)
    return array[y : y + h, x : x + w]


def preprocess(
    array: np.ndarray,
    *,
    do_hair_removal: bool = True,
    do_color_constancy: bool = True,
    do_vignette_crop: bool = True,
) -> PreprocessResult:
    """Run the full restoration chain and report what it did."""
    steps: list[str] = []
    working = array.copy()

    vignette_cropped = False
    if do_vignette_crop and detect_vignette(working) > 0.35:
        cropped = crop_vignette(working)
        if cropped.shape != working.shape:
            working = cropped
            vignette_cropped = True
            steps.append("cropped lens vignette")

    hair_mask, hair_ratio = detect_hair(working)
    hair_removed = False
    if do_hair_removal and np.any(hair_mask):
        working = remove_hair(working, hair_mask)
        hair_removed = True
        steps.append(f"inpainted hair ({hair_ratio * 100:.1f}% of pixels)")

    color_applied = False
    if do_color_constancy:
        working = shades_of_gray(working)
        color_applied = True
        steps.append("shades-of-gray colour constancy")

    return PreprocessResult(
        image=working,
        hair_mask=hair_mask,
        hair_ratio=hair_ratio,
        hair_removed=hair_removed,
        color_constancy_applied=color_applied,
        vignette_cropped=vignette_cropped,
        steps=steps,
    )


__all__ = [
    "PreprocessResult",
    "center_square_crop",
    "crop_vignette",
    "detect_hair",
    "detect_vignette",
    "encode_png",
    "load_image",
    "preprocess",
    "remove_hair",
    "shades_of_gray",
    "to_array",
    "to_pil",
]
