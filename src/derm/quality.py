"""Input quality assessment and out-of-distribution gating.

A classifier trained on HAM10000 will confidently label a photo of a keyboard as
a melanocytic nevus, because softmax has no way to say "this is not skin". Two
cheap safeguards go a long way:

1. **Technical quality** - resolution, focus, exposure, contrast, specular glare.
   A blurred or blown-out photograph gets flagged before the network runs.
2. **Domain plausibility** - a skin-chromaticity heuristic plus colourfulness and
   radial-symmetry checks estimate whether the frame even looks like a
   dermoscopic close-up of skin.

Both feed the severity engine, which downgrades trust in the prediction rather
than hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

Severity = Literal["info", "warning", "critical"]


@dataclass
class QualityIssue:
    """One detected problem with the submitted image."""

    code: str
    severity: Severity
    message: str
    value: float | None = None


@dataclass
class QualityReport:
    """Aggregate technical-quality and domain-plausibility verdict."""

    score: float  # 0-100, higher is better
    verdict: str  # "good" | "acceptable" | "poor"
    is_skin_like: bool
    skin_fraction: float
    sharpness: float
    brightness: float
    contrast: float
    colorfulness: float
    glare_fraction: float
    texture_heterogeneity: float
    width: int
    height: int
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def blocking(self) -> bool:
        """True when at least one issue is severe enough to distrust the result."""
        return any(issue.severity == "critical" for issue in self.issues)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 1),
            "verdict": self.verdict,
            "is_skin_like": self.is_skin_like,
            "blocking": self.blocking,
            "metrics": {
                "skin_fraction": round(self.skin_fraction, 3),
                "sharpness": round(self.sharpness, 1),
                "brightness": round(self.brightness, 1),
                "contrast": round(self.contrast, 1),
                "colorfulness": round(self.colorfulness, 1),
                "glare_fraction": round(self.glare_fraction, 3),
                "texture_heterogeneity": round(self.texture_heterogeneity, 3),
                "resolution": f"{self.width}x{self.height}",
            },
            "issues": [
                {
                    "code": issue.code,
                    "severity": issue.severity,
                    "message": issue.message,
                    "value": None if issue.value is None else round(issue.value, 3),
                }
                for issue in self.issues
            ],
        }


# --------------------------------------------------------------------------- #
# Individual metrics
# --------------------------------------------------------------------------- #


def laplacian_sharpness(array: np.ndarray) -> float:
    """Variance of the Laplacian: a standard, fast focus measure."""
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    # Normalise for image size so a large sharp image is not penalised.
    if max(gray.shape) > 512:
        scale = 512 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def colorfulness(array: np.ndarray) -> float:
    """Hasler-Susstrunk colourfulness metric.

    Near zero for a greyscale or monochrome frame, which is a strong hint that
    the upload is not a dermoscopic photograph.
    """
    b, g, r = cv2.split(array.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std + 0.3 * mean)


def skin_fraction(array: np.ndarray) -> float:
    """Fraction of pixels inside a broad skin-chromaticity envelope.

    Uses the union of an HSV rule and a YCrCb rule. The bounds are deliberately
    permissive so that dark skin tones, erythema and heavily pigmented melanoma
    all still register as skin; the aim is only to reject obvious non-skin input.
    """
    hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(array, cv2.COLOR_RGB2YCrCb)

    hsv_mask = cv2.inRange(hsv, np.array([0, 15, 30]), np.array([28, 255, 255]))
    hsv_mask |= cv2.inRange(hsv, np.array([160, 15, 30]), np.array([180, 255, 255]))
    ycrcb_mask = cv2.inRange(
        ycrcb, np.array([0, 130, 74]), np.array([255, 183, 130])
    )

    combined = cv2.bitwise_or(hsv_mask, ycrcb_mask)
    return float(np.count_nonzero(combined)) / combined.size


def texture_heterogeneity(array: np.ndarray) -> float:
    """Spread of local intensity variance across the frame.

    Dermoscopic photographs always carry some mix of skin texture, pigment
    structure and lesion edges, so how *much* local contrast varies from patch
    to patch is itself informative. A flat, gradient or uniformly-patterned
    surface (a wall, sand, an out-of-focus object, a screenshot) has almost the
    same local variance everywhere and scores near zero here, which is a strong
    hint the frame is not a photograph of a lesion - independent of whether its
    average colour happens to fall inside the skin-tone envelope.
    """
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if max(gray.shape) > 512:
        scale = 512 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mean = cv2.blur(gray, (9, 9))
    sqmean = cv2.blur(gray * gray, (9, 9))
    local_variance = np.maximum(sqmean - mean**2, 0)
    local_std = np.sqrt(local_variance)
    return float(local_std.std())


def glare_fraction(array: np.ndarray) -> float:
    """Fraction of near-saturated pixels (immersion-fluid specular highlights)."""
    hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
    bright = (hsv[..., 2] > 245) & (hsv[..., 1] < 30)
    return float(np.count_nonzero(bright)) / bright.size


# --------------------------------------------------------------------------- #
# Aggregate assessment
# --------------------------------------------------------------------------- #

MIN_DIMENSION = 64
RECOMMENDED_DIMENSION = 224
SHARPNESS_POOR = 25.0
SHARPNESS_SOFT = 80.0
SKIN_FRACTION_MIN = 0.30
COLORFULNESS_MIN = 4.0
TEXTURE_MIN = 0.50


def assess(array: np.ndarray) -> QualityReport:
    """Score an RGB image for suitability as dermoscopic model input."""
    height, width = array.shape[:2]
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

    sharpness = laplacian_sharpness(array)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    colour = colorfulness(array)
    skin = skin_fraction(array)
    glare = glare_fraction(array)
    texture = texture_heterogeneity(array)

    issues: list[QualityIssue] = []
    penalty = 0.0

    # --- resolution ------------------------------------------------------- #
    if min(height, width) < MIN_DIMENSION:
        issues.append(
            QualityIssue(
                "resolution_too_low",
                "critical",
                f"Image is only {width}x{height} px. At least "
                f"{MIN_DIMENSION}x{MIN_DIMENSION} is required for a meaningful result.",
                float(min(height, width)),
            )
        )
        penalty += 40
    elif min(height, width) < RECOMMENDED_DIMENSION:
        issues.append(
            QualityIssue(
                "resolution_low",
                "warning",
                f"Image is {width}x{height} px and will be upscaled to "
                f"{RECOMMENDED_DIMENSION}px, which loses fine dermoscopic detail.",
                float(min(height, width)),
            )
        )
        penalty += 12

    # --- focus ------------------------------------------------------------ #
    if sharpness < SHARPNESS_POOR:
        issues.append(
            QualityIssue(
                "out_of_focus",
                "critical",
                "The image is heavily blurred. Border and texture measurements "
                "will be unreliable; retake with the dermatoscope in contact.",
                sharpness,
            )
        )
        penalty += 30
    elif sharpness < SHARPNESS_SOFT:
        issues.append(
            QualityIssue(
                "soft_focus",
                "warning",
                "The image is slightly soft. Fine structures such as dots and "
                "globules may be missed.",
                sharpness,
            )
        )
        penalty += 10

    # --- exposure --------------------------------------------------------- #
    if brightness < 45:
        issues.append(
            QualityIssue(
                "underexposed",
                "warning",
                "The image is very dark, which exaggerates apparent pigmentation.",
                brightness,
            )
        )
        penalty += 12
    elif brightness > 215:
        issues.append(
            QualityIssue(
                "overexposed",
                "warning",
                "The image is washed out; pigment network detail is likely lost.",
                brightness,
            )
        )
        penalty += 12

    if contrast < 18:
        issues.append(
            QualityIssue(
                "low_contrast",
                "warning",
                "Very low contrast between lesion and surrounding skin.",
                contrast,
            )
        )
        penalty += 10

    # --- glare ------------------------------------------------------------ #
    if glare > 0.12:
        issues.append(
            QualityIssue(
                "specular_glare",
                "warning",
                f"{glare * 100:.0f}% of the frame is specular glare. Add immersion "
                "fluid or use a polarised dermatoscope.",
                glare,
            )
        )
        penalty += 10

    # --- domain plausibility ---------------------------------------------- #
    # These three checks are independent failure modes (flat, greyscale, wrong
    # hue), so each is reported and penalised on its own rather than as an
    # elif chain - a flat grey image should surface both problems, not just
    # whichever is checked first.
    is_skin_like = (
        skin >= SKIN_FRACTION_MIN
        and colour >= COLORFULNESS_MIN
        and texture >= TEXTURE_MIN
    )
    if texture < TEXTURE_MIN:
        issues.append(
            QualityIssue(
                "no_dermoscopic_structure",
                "critical",
                "The image has almost no local texture variation - it looks flat, "
                "blank or uniformly patterned rather than a close-up photograph of "
                "skin or a lesion. The classification below is not trustworthy.",
                texture,
            )
        )
        penalty += 35
    if colour < COLORFULNESS_MIN:
        issues.append(
            QualityIssue(
                "not_color_image",
                "critical",
                "The image is effectively greyscale. This model expects a colour "
                "dermoscopic photograph.",
                colour,
            )
        )
        penalty += 30
    elif skin < SKIN_FRACTION_MIN:
        issues.append(
            QualityIssue(
                "out_of_distribution",
                "critical",
                f"Only {skin * 100:.0f}% of the frame matches skin tones, so this "
                "probably is not a dermoscopic image of skin. The classification "
                "below is not trustworthy.",
                skin,
            )
        )
        penalty += 35

    score = max(0.0, 100.0 - penalty)
    verdict = "good" if score >= 80 else "acceptable" if score >= 55 else "poor"

    return QualityReport(
        score=score,
        verdict=verdict,
        is_skin_like=is_skin_like,
        skin_fraction=skin,
        sharpness=sharpness,
        brightness=brightness,
        contrast=contrast,
        colorfulness=colour,
        glare_fraction=glare,
        texture_heterogeneity=texture,
        width=width,
        height=height,
        issues=issues,
    )


__all__ = ["QualityIssue", "QualityReport", "assess", "colorfulness", "glare_fraction",
           "laplacian_sharpness", "skin_fraction", "texture_heterogeneity"]
