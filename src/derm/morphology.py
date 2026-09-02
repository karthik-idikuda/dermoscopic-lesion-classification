"""Quantitative ABCD morphometry from the lesion mask.

This is the interpretable half of the system. The network gives a probability
distribution; these measurements give a clinician-legible reason: how asymmetric
the lesion is, how ragged its border is, how many distinct colours it contains.

The scoring follows the Stolz ABCD rule for dermoscopy:

    TDS = 1.3*A + 0.1*B + 0.5*C + 0.5*D

with A in 0-2, B in 0-8, C in 0-6, D in 0-5, and the conventional cut-points
TDS < 4.76 benign, 4.76-5.45 suspicious, > 5.45 highly suspicious for melanoma.

Important caveat, stated plainly because it matters for how the numbers are
read: A and B are computed geometrically and are faithful to the original rule,
while C is a colour-quantisation approximation and D (differential structures)
is approximated by texture-pattern detectors rather than expert annotation.
D in particular should be treated as a weak signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .segmentation import Segmentation

# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #


@dataclass
class ABCDScore:
    """Stolz ABCD components and the resulting total dermoscopy score."""

    asymmetry: int  # 0-2
    border: int  # 0-8
    colors: int  # 0-6
    structures: int  # 0-5
    tds: float
    interpretation: str  # "benign" | "suspicious" | "highly_suspicious"
    colors_present: list[str] = field(default_factory=list)
    structures_present: list[str] = field(default_factory=list)

    @property
    def max_tds(self) -> float:
        return 1.3 * 2 + 0.1 * 8 + 0.5 * 6 + 0.5 * 5  # 8.9

    def to_dict(self) -> dict:
        return {
            "asymmetry": self.asymmetry,
            "border": self.border,
            "colors": self.colors,
            "structures": self.structures,
            "tds": round(self.tds, 2),
            "tds_max": self.max_tds,
            "interpretation": self.interpretation,
            "colors_present": self.colors_present,
            "structures_present": self.structures_present,
        }


@dataclass
class MorphologyFeatures:
    """ABCD score plus continuous shape/colour descriptors."""

    abcd: ABCDScore
    asymmetry_index: float  # 0-1, mean XOR overlap error over both axes
    border_irregularity: float  # 0-1, normalised radial-distance variation
    circularity: float  # 1.0 = perfect circle
    compactness: float  # perimeter^2 / (4*pi*area)
    eccentricity: float  # 0 = circle, ->1 = elongated
    solidity: float  # area / convex-hull area
    color_variance: float  # mean per-channel std inside the lesion
    color_asymmetry: float  # 0-1, pigment-intensity asymmetry
    diameter_px: float
    diameter_fraction: float  # lesion diameter / image width
    lesion_skin_contrast: float  # 0-1
    blue_white_veil: float  # fraction of lesion with veil-like colour
    reliable: bool

    def to_dict(self) -> dict:
        return {
            "abcd": self.abcd.to_dict(),
            "shape": {
                "asymmetry_index": round(self.asymmetry_index, 3),
                "border_irregularity": round(self.border_irregularity, 3),
                "circularity": round(self.circularity, 3),
                "compactness": round(self.compactness, 3),
                "eccentricity": round(self.eccentricity, 3),
                "solidity": round(self.solidity, 3),
                "diameter_px": round(self.diameter_px, 1),
                "diameter_fraction": round(self.diameter_fraction, 3),
            },
            "color": {
                "variance": round(self.color_variance, 2),
                "asymmetry": round(self.color_asymmetry, 3),
                "lesion_skin_contrast": round(self.lesion_skin_contrast, 3),
                "blue_white_veil": round(self.blue_white_veil, 3),
            },
            "reliable": self.reliable,
        }


# --------------------------------------------------------------------------- #
# A - Asymmetry
# --------------------------------------------------------------------------- #


def _align_to_principal_axes(
    mask: np.ndarray, image: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Rotate mask (and image) so the lesion's major axis is horizontal.

    Asymmetry must be measured about the lesion's own axes, otherwise a
    perfectly symmetric but tilted oval scores as asymmetric.
    """
    points = cv2.findNonZero(mask)
    if points is None or len(points) < 5:
        return mask, image

    (cx, cy), (_, _), angle = cv2.minAreaRect(points)
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    height, width = mask.shape
    rotated_mask = cv2.warpAffine(
        mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0
    )
    rotated_image = (
        cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR)
        if image is not None
        else None
    )
    return rotated_mask, rotated_image


def _crop_to_mask(mask: np.ndarray, image: np.ndarray | None = None):
    x, y, w, h = cv2.boundingRect(mask)
    if w == 0 or h == 0:
        return mask, image
    cropped_mask = mask[y : y + h, x : x + w]
    cropped_image = image[y : y + h, x : x + w] if image is not None else None
    return cropped_mask, cropped_image


def asymmetry(
    mask: np.ndarray, image: np.ndarray | None = None
) -> tuple[int, float, float]:
    """Score asymmetry about both principal axes.

    Returns ``(score 0-2, shape asymmetry index, colour asymmetry index)``.
    A sector counts as asymmetric when the mismatch between the lesion and its
    mirror image exceeds 12% of the lesion area, the threshold commonly used in
    automated ABCD implementations.
    """
    aligned_mask, aligned_image = _align_to_principal_axes(mask, image)
    aligned_mask, aligned_image = _crop_to_mask(aligned_mask, aligned_image)

    area = float(np.count_nonzero(aligned_mask))
    if area == 0:
        return 0, 0.0, 0.0

    binary = (aligned_mask > 0).astype(np.uint8)
    horizontal = np.fliplr(binary)
    vertical = np.flipud(binary)

    shape_h = float(np.count_nonzero(cv2.bitwise_xor(binary, horizontal))) / (2 * area)
    shape_v = float(np.count_nonzero(cv2.bitwise_xor(binary, vertical))) / (2 * area)

    color_h = color_v = 0.0
    if aligned_image is not None:
        gray = cv2.cvtColor(aligned_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray *= binary  # zero outside the lesion
        total = gray.sum()
        if total > 0:
            color_h = float(np.abs(gray - np.fliplr(gray)).sum()) / (2 * total)
            color_v = float(np.abs(gray - np.flipud(gray)).sum()) / (2 * total)

    threshold = 0.12
    axis_h = max(shape_h, color_h)
    axis_v = max(shape_v, color_v)
    score = int(axis_h > threshold) + int(axis_v > threshold)

    shape_index = float(np.clip((shape_h + shape_v) / 2.0, 0.0, 1.0))
    color_index = float(np.clip((color_h + color_v) / 2.0, 0.0, 1.0))
    return score, shape_index, color_index


# --------------------------------------------------------------------------- #
# B - Border
# --------------------------------------------------------------------------- #


def border_score(mask: np.ndarray, contour: np.ndarray | None) -> tuple[int, float]:
    """Score border irregularity in eight sectors around the centroid.

    Stolz awards one point per eighth of the periphery showing an abrupt pigment
    cut-off. Here each sector is scored on the local variability of the radial
    distance from the centroid, normalised by the mean radius, which is the
    standard geometric proxy for a ragged or notched edge.

    Returns ``(score 0-8, continuous irregularity index 0-1)``.
    """
    if contour is None or len(contour) < 8:
        return 0, 0.0

    points = contour.reshape(-1, 2).astype(np.float32)
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] == 0:
        return 0, 0.0
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    radii = np.sqrt(dx**2 + dy**2)
    angles = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)
    mean_radius = float(radii.mean())
    if mean_radius <= 1e-6:
        return 0, 0.0

    sector_size = 2 * np.pi / 8
    score = 0
    variations: list[float] = []
    for sector in range(8):
        lo = sector * sector_size
        hi = lo + sector_size
        selected = radii[(angles >= lo) & (angles < hi)]
        if selected.size < 3:
            continue
        # Local roughness relative to this sector's own scale.
        variation = float(selected.std()) / mean_radius
        variations.append(variation)
        if variation > 0.12:
            score += 1

    irregularity = float(np.clip(np.mean(variations) / 0.30, 0.0, 1.0)) if variations else 0.0
    return score, irregularity


# --------------------------------------------------------------------------- #
# C - Colour
# --------------------------------------------------------------------------- #

# Each entry maps a clinical colour name to a predicate over (L, a, b, H, S, V).
COLOR_NAMES = (
    "white",
    "red",
    "light_brown",
    "dark_brown",
    "blue_gray",
    "black",
)


def color_score(
    image: np.ndarray, mask: np.ndarray, *, min_fraction: float = 0.05
) -> tuple[int, list[str]]:
    """Count how many of the six diagnostic colours occupy >=5% of the lesion.

    Thresholds are defined in LAB (perceptually uniform lightness) combined with
    HSV hue, applied only to pixels inside the lesion mask.
    """
    inside = mask > 0
    total = int(np.count_nonzero(inside))
    if total == 0:
        return 0, []

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lightness = lab[..., 0].astype(np.float32) * (100.0 / 255.0)  # -> 0-100
    a_star = lab[..., 1].astype(np.float32) - 128.0
    b_star = lab[..., 2].astype(np.float32) - 128.0
    hue = hsv[..., 0].astype(np.float32) * 2.0  # OpenCV packs hue as 0-179
    saturation = hsv[..., 1].astype(np.float32) / 255.0

    predicates = {
        # Depigmented / regression areas: very light, barely chromatic.
        "white": (lightness > 70) & (saturation < 0.18),
        # Erythema or vascular structures: strongly positive a*, reddish hue.
        "red": (a_star > 18) & (saturation > 0.30) & ((hue < 25) | (hue > 335)),
        "light_brown": (lightness >= 45) & (lightness <= 75) & (b_star > 8) & (a_star > 3),
        "dark_brown": (lightness >= 20) & (lightness < 45) & (b_star > 5),
        # Blue-gray granularity / veil: negative b*, bluish hue.
        "blue_gray": (b_star < -3) & (lightness > 20) & (lightness < 70),
        "black": lightness < 20,
    }

    present: list[str] = []
    for name in COLOR_NAMES:
        fraction = float(np.count_nonzero(predicates[name] & inside)) / total
        if fraction >= min_fraction:
            present.append(name)

    return len(present), present


# --------------------------------------------------------------------------- #
# D - Differential structures (approximated)
# --------------------------------------------------------------------------- #


def structure_score(image: np.ndarray, mask: np.ndarray) -> tuple[int, list[str]]:
    """Approximate the count of distinct dermoscopic structures present.

    Five structure families are probed with classical filters. These are
    proxies, not expert annotations, so the score is intentionally conservative:

    * ``structureless_area`` - a broad low-variance plateau
    * ``dots_globules``      - small dark blobs found by a Laplacian-of-Gaussian
    * ``pigment_network``    - reticular ridges found by a black-hat + skeleton
    * ``streaks``            - elongated high-gradient structures at the periphery
    * ``blue_white_veil``    - confluent bluish region with a whitish overlay
    """
    inside = mask > 0
    total = int(np.count_nonzero(inside))
    if total < 100:
        return 0, []

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    present: list[str] = []

    # 1. Structureless / homogeneous plateau -------------------------------- #
    local_std = cv2.GaussianBlur(gray.astype(np.float32) ** 2, (15, 15), 0) - (
        cv2.GaussianBlur(gray.astype(np.float32), (15, 15), 0) ** 2
    )
    local_std = np.sqrt(np.maximum(local_std, 0))
    flat_fraction = float(np.count_nonzero((local_std < 6.0) & inside)) / total
    if flat_fraction > 0.30:
        present.append("structureless_area")

    # 2. Dots and globules -------------------------------------------------- #
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.6)
    log = cv2.Laplacian(blurred, cv2.CV_32F, ksize=5)
    blobs = (log > np.percentile(log[inside], 97)) & inside
    blobs = cv2.morphologyEx(
        blobs.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(blobs, connectivity=8)
    globules = sum(
        1
        for label in range(1, count)
        if 6 <= stats[label, cv2.CC_STAT_AREA] <= max(40, total * 0.01)
    )
    if globules >= 6:
        present.append("dots_globules")

    # 3. Pigment network ---------------------------------------------------- #
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    if np.any(inside):
        ridge_threshold = np.percentile(blackhat[inside], 92)
        ridges = ((blackhat >= max(ridge_threshold, 6)) & inside).astype(np.uint8)
        ridges = cv2.morphologyEx(
            ridges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        ridge_fraction = float(np.count_nonzero(ridges)) / total
        # A network is a connected mesh: many ridge pixels in few components.
        n_comp, _, comp_stats, _ = cv2.connectedComponentsWithStats(ridges, connectivity=8)
        largest = (
            float(comp_stats[1:, cv2.CC_STAT_AREA].max()) if n_comp > 1 else 0.0
        )
        mesh_ratio = largest / max(1.0, float(np.count_nonzero(ridges)))
        if ridge_fraction > 0.04 and mesh_ratio > 0.25:
            present.append("pigment_network")

    # 4. Peripheral streaks ------------------------------------------------- #
    eroded = cv2.erode(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1
    )
    rim = (mask > 0) & (eroded == 0)
    if np.count_nonzero(rim) > 50:
        gradient = cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
        )
        rim_gradient = gradient[rim]
        core_gradient = gradient[eroded > 0] if np.any(eroded > 0) else rim_gradient
        if rim_gradient.mean() > 1.35 * max(core_gradient.mean(), 1e-6):
            present.append("streaks")

    # 5. Blue-white veil ---------------------------------------------------- #
    if veil_fraction(image, mask) > 0.08:
        present.append("blue_white_veil")

    return min(len(present), 5), present


def veil_fraction(image: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of the lesion showing blue-white veil colouring."""
    inside = mask > 0
    total = int(np.count_nonzero(inside))
    if total == 0:
        return 0.0
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness = lab[..., 0].astype(np.float32) * (100.0 / 255.0)
    b_star = lab[..., 2].astype(np.float32) - 128.0
    veil = (b_star < -4) & (lightness > 35) & (lightness < 75) & inside
    return float(np.count_nonzero(veil)) / total


# --------------------------------------------------------------------------- #
# Continuous descriptors
# --------------------------------------------------------------------------- #


def _shape_descriptors(mask: np.ndarray, contour: np.ndarray | None) -> dict[str, float]:
    area = float(np.count_nonzero(mask))
    if contour is None or area <= 0:
        return {
            "circularity": 0.0,
            "compactness": 0.0,
            "eccentricity": 0.0,
            "solidity": 0.0,
        }

    perimeter = float(cv2.arcLength(contour, True))
    contour_area = max(float(cv2.contourArea(contour)), 1.0)
    circularity = (
        float(np.clip(4 * np.pi * contour_area / (perimeter**2), 0.0, 1.0))
        if perimeter > 0
        else 0.0
    )
    compactness = (perimeter**2) / (4 * np.pi * contour_area) if contour_area > 0 else 0.0

    hull = cv2.convexHull(contour)
    hull_area = max(float(cv2.contourArea(hull)), 1.0)
    solidity = float(np.clip(contour_area / hull_area, 0.0, 1.0))

    eccentricity = 0.0
    if len(contour) >= 5:
        (_, _), (major, minor), _ = cv2.fitEllipse(contour)
        major, minor = max(major, minor), min(major, minor)
        if major > 1e-6:
            ratio = (minor / major) ** 2
            eccentricity = float(np.sqrt(max(0.0, 1.0 - ratio)))

    return {
        "circularity": circularity,
        "compactness": float(compactness),
        "eccentricity": eccentricity,
        "solidity": solidity,
    }


def _contrast(image: np.ndarray, mask: np.ndarray) -> float:
    """Normalised lightness difference between lesion and surrounding skin."""
    inside = mask > 0
    ring = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)), iterations=1
    )
    outside = (ring > 0) & ~inside
    if not np.any(inside) or not np.any(outside):
        return 0.0
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    return float(np.clip(abs(lab[outside].mean() - lab[inside].mean()) / 255.0 * 4, 0, 1))


def interpret_tds(tds: float) -> str:
    if tds < 4.76:
        return "benign"
    if tds <= 5.45:
        return "suspicious"
    return "highly_suspicious"


def analyze(
    image: np.ndarray, segmentation: Segmentation
) -> MorphologyFeatures:
    """Compute the full ABCD score and continuous descriptors for one lesion."""
    mask = segmentation.mask
    contour = segmentation.contour

    a_score, shape_asym, color_asym = asymmetry(mask, image)
    b_score, irregularity = border_score(mask, contour)
    c_score, colors_present = color_score(image, mask)
    d_score, structures_present = structure_score(image, mask)

    tds = 1.3 * a_score + 0.1 * b_score + 0.5 * c_score + 0.5 * d_score

    abcd = ABCDScore(
        asymmetry=a_score,
        border=b_score,
        colors=c_score,
        structures=d_score,
        tds=tds,
        interpretation=interpret_tds(tds),
        colors_present=colors_present,
        structures_present=structures_present,
    )

    shape = _shape_descriptors(mask, contour)
    inside = mask > 0
    color_variance = (
        float(np.mean([image[..., c][inside].std() for c in range(3)]))
        if np.any(inside)
        else 0.0
    )

    return MorphologyFeatures(
        abcd=abcd,
        asymmetry_index=shape_asym,
        border_irregularity=irregularity,
        circularity=shape["circularity"],
        compactness=shape["compactness"],
        eccentricity=shape["eccentricity"],
        solidity=shape["solidity"],
        color_variance=color_variance,
        color_asymmetry=color_asym,
        diameter_px=segmentation.equivalent_diameter,
        diameter_fraction=float(
            segmentation.equivalent_diameter / max(1, mask.shape[1])
        ),
        lesion_skin_contrast=_contrast(image, mask),
        blue_white_veil=veil_fraction(image, mask),
        reliable=segmentation.is_reliable,
    )


__all__ = [
    "ABCDScore",
    "MorphologyFeatures",
    "analyze",
    "asymmetry",
    "border_score",
    "color_score",
    "interpret_tds",
    "structure_score",
    "veil_fraction",
]
