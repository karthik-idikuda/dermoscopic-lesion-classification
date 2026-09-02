"""Automated severity grading.

The notebook graded severity from the predicted class and its softmax confidence
alone. That works, and its "low confidence escalates to high risk" safety net is
the right instinct, but it throws away three things the pipeline already knows:
the lesion's geometry, how stable the prediction is under augmentation, and
whether the photograph was even usable.

This engine combines four evidence streams into a single 0-100 score with an
explicit, inspectable audit trail:

    neural risk        clinically weighted malignancy probability
    morphology risk    ABCD total dermoscopy score
    uncertainty risk   entropy and augmentation disagreement
    quality risk       technical defects and out-of-distribution input

A set of hard overrides then guarantees that no melanoma-suspicious case can be
graded low, whatever the weighted arithmetic says. Overrides can only ever raise
the tier, never lower it - a deliberately asymmetric design, because the cost of
a missed melanoma is not symmetric with the cost of an unnecessary referral.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .config import LESION_CLASSES
from .morphology import MorphologyFeatures
from .quality import QualityReport
from .uncertainty import UncertaintyReport

Tier = Literal["LOW", "MODERATE", "HIGH", "CRITICAL", "INDETERMINATE"]

TIER_ORDER: dict[str, int] = {
    "INDETERMINATE": 0,
    "LOW": 1,
    "MODERATE": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}

# Clinical weighting of each class's contribution to malignancy risk. Melanoma
# dominates because it is the only class in HAM10000 that reliably kills.
CLASS_RISK_WEIGHT: dict[str, float] = {
    "mel": 1.00,
    "bcc": 0.70,
    "akiec": 0.55,
    "bkl": 0.10,
    "df": 0.05,
    "vasc": 0.05,
    "nv": 0.02,
}

TIER_GUIDANCE: dict[str, dict[str, str]] = {
    "CRITICAL": {
        "timeframe": "within 1-2 weeks",
        "action": "Urgent dermatology referral for excision biopsy",
        "color": "#b3202b",
    },
    "HIGH": {
        "timeframe": "within 2-4 weeks",
        "action": "Dermatology referral for specialist assessment and likely biopsy",
        "color": "#e74c3c",
    },
    "MODERATE": {
        "timeframe": "within 4-8 weeks",
        "action": "Dermatology consultation advised; photograph and monitor meanwhile",
        "color": "#f39c12",
    },
    "LOW": {
        "timeframe": "routine",
        "action": "Routine self-monitoring; re-present if the lesion changes",
        "color": "#2ecc71",
    },
    "INDETERMINATE": {
        "timeframe": "repeat assessment",
        "action": "Retake the image or seek in-person review; automated result unusable",
        "color": "#7f8c8d",
    },
}


@dataclass
class RiskDriver:
    """One named contribution to the final score, for the audit trail."""

    label: str
    detail: str
    direction: Literal["increases", "decreases", "neutral"]
    contribution: float  # points on the 0-100 scale

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "detail": self.detail,
            "direction": self.direction,
            "contribution": round(self.contribution, 1),
        }


@dataclass
class SeverityAssessment:
    """Final graded output."""

    tier: Tier
    score: float  # 0-100
    malignancy_probability: float  # summed probability of malignant classes
    headline: str
    recommendation: str
    timeframe: str
    color: str
    requires_human_review: bool
    review_reasons: list[str] = field(default_factory=list)
    drivers: list[RiskDriver] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)
    overrides_applied: list[str] = field(default_factory=list)
    neural_usable: bool = True

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "score": round(self.score, 1),
            "malignancy_probability": round(self.malignancy_probability, 4),
            "headline": self.headline,
            "recommendation": self.recommendation,
            "timeframe": self.timeframe,
            "color": self.color,
            "requires_human_review": self.requires_human_review,
            "review_reasons": self.review_reasons,
            "drivers": [d.to_dict() for d in self.drivers],
            "components": {k: round(v, 1) for k, v in self.components.items()},
            "overrides_applied": self.overrides_applied,
            # False whenever the classifier's output was excluded from grading
            # (untrained weights, or input that is not a skin photo at all).
            # Consumers should not present a named diagnosis or probability
            # ranking as meaningful when this is False.
            "neural_usable": self.neural_usable,
        }


def _tier_from_score(score: float) -> Tier:
    if score >= 72:
        return "CRITICAL"
    if score >= 48:
        return "HIGH"
    if score >= 24:
        return "MODERATE"
    return "LOW"


def _escalate(current: Tier, minimum: Tier) -> Tier:
    """Raise ``current`` to at least ``minimum``; never lower it."""
    return minimum if TIER_ORDER[minimum] > TIER_ORDER[current] else current


def neural_risk(probabilities: np.ndarray, class_codes: tuple[str, ...]) -> float:
    """Clinically weighted malignancy risk in ``[0, 100]``."""
    total = 0.0
    for index, code in enumerate(class_codes):
        if index < len(probabilities):
            total += CLASS_RISK_WEIGHT.get(code, 0.1) * float(probabilities[index])
    return float(np.clip(total * 100.0, 0.0, 100.0))


def morphology_risk(morphology: MorphologyFeatures | None) -> float:
    """Map the ABCD total dermoscopy score onto ``[0, 100]``.

    TDS 4.76 is the classical benign/suspicious cut-point and TDS 5.45 the
    highly-suspicious one; the mapping is scaled so those land near 50 and 70.
    """
    if morphology is None:
        return 0.0
    if not morphology.reliable:
        # Segmentation was a fallback ellipse: the geometry means little, so
        # contribute a neutral mid-low value rather than a confident zero.
        return 20.0
    tds = morphology.abcd.tds
    return float(np.clip((tds / 6.8) * 100.0, 0.0, 100.0))


def uncertainty_risk(uncertainty: UncertaintyReport | None) -> float:
    """Uncertainty expressed as risk: an unsure model needs a human."""
    if uncertainty is None:
        return 0.0
    entropy_part = uncertainty.entropy
    disagreement_part = 1.0 - uncertainty.tta_agreement
    margin_part = 1.0 - float(np.clip(uncertainty.margin, 0.0, 1.0))
    combined = 0.45 * entropy_part + 0.30 * disagreement_part + 0.25 * margin_part
    return float(np.clip(combined * 100.0, 0.0, 100.0))


def quality_risk(quality: QualityReport | None) -> float:
    """Poor input quality raises risk because the result cannot be trusted."""
    if quality is None:
        return 0.0
    return float(np.clip(100.0 - quality.score, 0.0, 100.0))


def grade(
    probabilities: np.ndarray,
    class_codes: tuple[str, ...],
    *,
    morphology: MorphologyFeatures | None = None,
    uncertainty: UncertaintyReport | None = None,
    quality: QualityReport | None = None,
    attention_verdict: float | None = None,
    model_is_trained: bool = True,
) -> SeverityAssessment:
    """Produce the composite severity assessment."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted_index = int(probabilities.argmax())
    predicted_code = (
        class_codes[predicted_index] if predicted_index < len(class_codes) else "nv"
    )
    lesion = LESION_CLASSES.get(predicted_code)
    confidence = float(probabilities.max())

    # An out-of-distribution input (not a dermoscopic photo of skin) makes the
    # classifier's softmax output meaningless in exactly the same way untrained
    # weights do: well-formed numbers with no relationship to the image. Both
    # conditions get the same treatment - neural output is excluded from the
    # score, from the driver list, and from the melanoma safety-net overrides -
    # so a photo of a brick wall can never be graded HIGH because the network
    # happened to assign it 33% "melanoma".
    non_skin_input = quality is not None and not quality.is_skin_like
    neural_usable = model_is_trained and not non_skin_input

    malignant_probability = (
        float(
            sum(
                probabilities[i]
                for i, code in enumerate(class_codes)
                if i < len(probabilities) and LESION_CLASSES[code].is_malignant
            )
        )
        if neural_usable
        else 0.0
    )

    components = {
        "neural": neural_risk(probabilities, class_codes) if neural_usable else 0.0,
        "morphology": morphology_risk(morphology),
        "uncertainty": uncertainty_risk(uncertainty),
        "quality": quality_risk(quality),
    }

    drivers: list[RiskDriver] = []
    overrides: list[str] = []
    review_reasons: list[str] = []

    if neural_usable:
        weights = {"neural": 0.52, "morphology": 0.24, "uncertainty": 0.16, "quality": 0.08}
    else:
        # With untrained weights, or input that is not skin at all, the softmax
        # is noise. Grade on the geometry and the image quality only, and
        # refuse to claim a real tier.
        weights = {"neural": 0.0, "morphology": 0.70, "uncertainty": 0.0, "quality": 0.30}
        if not model_is_trained:
            review_reasons.append(
                "The classifier has no trained weights, so the neural prediction was "
                "excluded from grading. Only image morphometry was used."
            )
        else:
            review_reasons.append(
                "This image does not appear to be a dermoscopic photograph of skin, "
                "so the neural prediction was excluded from grading rather than "
                "shown as a diagnosis."
            )

    score = sum(components[key] * weight for key, weight in weights.items())
    tier: Tier = _tier_from_score(score)

    # ------------------------------------------------------------------ drivers
    if neural_usable:
        drivers.append(
            RiskDriver(
                label="Neural classification",
                detail=(
                    f"{lesion.short_name if lesion else predicted_code} at "
                    f"{confidence * 100:.1f}% confidence; combined malignant "
                    f"probability {malignant_probability * 100:.1f}%"
                ),
                direction="increases" if components["neural"] >= 30 else "decreases",
                contribution=components["neural"] * weights["neural"],
            )
        )

    if morphology is not None:
        abcd = morphology.abcd
        drivers.append(
            RiskDriver(
                label="ABCD morphometry",
                detail=(
                    f"TDS {abcd.tds:.2f} ({abcd.interpretation.replace('_', ' ')}): "
                    f"A={abcd.asymmetry}/2, B={abcd.border}/8, C={abcd.colors}/6, "
                    f"D={abcd.structures}/5"
                ),
                direction="increases" if abcd.tds >= 4.76 else "decreases",
                contribution=components["morphology"] * weights["morphology"],
            )
        )

    if uncertainty is not None and model_is_trained:
        drivers.append(
            RiskDriver(
                label="Prediction stability",
                detail=(
                    f"{uncertainty.verdict}: entropy {uncertainty.entropy:.2f}, "
                    f"{uncertainty.tta_agreement * 100:.0f}% of augmented views agree"
                ),
                direction="increases" if uncertainty.verdict != "confident" else "decreases",
                contribution=components["uncertainty"] * weights["uncertainty"],
            )
        )

    if quality is not None:
        drivers.append(
            RiskDriver(
                label="Image quality",
                detail=f"{quality.verdict} ({quality.score:.0f}/100)",
                direction="increases" if quality.score < 70 else "decreases",
                contribution=components["quality"] * weights["quality"],
            )
        )

    if attention_verdict is not None and model_is_trained:
        aligned = attention_verdict >= 0.5
        drivers.append(
            RiskDriver(
                label="Explanation alignment",
                detail=(
                    "Grad-CAM evidence is concentrated on the segmented lesion"
                    if aligned
                    else "Grad-CAM evidence falls largely outside the lesion, so the "
                    "prediction may be driven by background artefacts"
                ),
                direction="decreases" if aligned else "increases",
                contribution=0.0,
            )
        )
        if not aligned:
            review_reasons.append(
                "The model's visual evidence does not overlap the detected lesion."
            )

    # ---------------------------------------------------------------- overrides
    if not neural_usable:
        tier = "INDETERMINATE"
        if not model_is_trained:
            overrides.append("No trained weights: tier forced to INDETERMINATE.")
        else:
            overrides.append(
                "Input does not appear to be a dermoscopic image of skin: tier "
                "forced to INDETERMINATE."
            )
    else:
        mel_index = class_codes.index("mel") if "mel" in class_codes else None
        mel_probability = (
            float(probabilities[mel_index])
            if mel_index is not None and mel_index < len(probabilities)
            else 0.0
        )

        if predicted_code == "mel":
            tier = _escalate(tier, "CRITICAL" if confidence >= 0.70 else "HIGH")
            overrides.append(
                f"Melanoma is the top prediction ({confidence * 100:.1f}%)."
            )
        elif mel_probability >= 0.25:
            tier = _escalate(tier, "HIGH")
            overrides.append(
                f"Melanoma probability {mel_probability * 100:.1f}% exceeds the 25% "
                "safety-net threshold even though it is not the top class."
            )
        elif mel_probability >= 0.10:
            tier = _escalate(tier, "MODERATE")
            overrides.append(
                f"Non-trivial melanoma probability ({mel_probability * 100:.1f}%)."
            )

        if lesion is not None and lesion.is_malignant:
            tier = _escalate(tier, "MODERATE")
            if predicted_code != "mel":
                overrides.append(
                    f"{lesion.short_name} is a malignant or premalignant diagnosis."
                )

        if confidence < 0.50:
            tier = _escalate(tier, "MODERATE")
            overrides.append(
                f"Top-class confidence is only {confidence * 100:.1f}%, below the 50% "
                "threshold for an actionable automated result."
            )
            review_reasons.append("Model confidence is below 50%.")

        if uncertainty is not None and uncertainty.verdict == "uncertain":
            tier = _escalate(tier, "MODERATE")
            review_reasons.append(
                "The prediction is unstable across augmented views of the same image."
            )

        if morphology is not None and morphology.abcd.tds > 5.45:
            tier = _escalate(tier, "HIGH")
            overrides.append(
                f"ABCD total dermoscopy score {morphology.abcd.tds:.2f} is above the "
                "5.45 highly-suspicious cut-point."
            )

    if quality is not None and quality.blocking:
        review_reasons.extend(
            issue.message for issue in quality.issues if issue.severity == "critical"
        )
        # non_skin_input already forced INDETERMINATE and excluded the neural
        # component above; this only needs to catch other critical quality
        # failures (e.g. resolution too low) on an otherwise skin-like image.
        if not quality.is_skin_like:
            tier = "INDETERMINATE"

    if morphology is not None and not morphology.reliable:
        review_reasons.append(
            "Lesion segmentation was unreliable, so ABCD geometry is approximate."
        )

    guidance = TIER_GUIDANCE[tier]
    headline = _headline(
        tier, predicted_code, confidence, model_is_trained, non_skin=non_skin_input
    )

    return SeverityAssessment(
        tier=tier,
        score=float(np.clip(score, 0.0, 100.0)),
        malignancy_probability=malignant_probability,
        headline=headline,
        recommendation=guidance["action"],
        timeframe=guidance["timeframe"],
        color=guidance["color"],
        requires_human_review=bool(review_reasons) or tier in {"HIGH", "CRITICAL", "INDETERMINATE"},
        review_reasons=list(dict.fromkeys(review_reasons)),
        drivers=drivers,
        components=components,
        overrides_applied=overrides,
        neural_usable=neural_usable,
    )


def _headline(
    tier: Tier, code: str, confidence: float, trained: bool, *, non_skin: bool = False
) -> str:
    lesion = LESION_CLASSES.get(code)
    name = lesion.short_name if lesion else code
    if tier == "INDETERMINATE":
        if non_skin:
            return "Indeterminate - image does not appear to be a dermoscopic photo of skin"
        if not trained:
            return "Indeterminate - classifier not trained, morphometry only"
        return "Indeterminate - input unsuitable for automated assessment"
    if tier == "CRITICAL":
        return f"Critical concern - {name} at {confidence * 100:.0f}% confidence"
    if tier == "HIGH":
        return f"High concern - {name} at {confidence * 100:.0f}% confidence"
    if tier == "MODERATE":
        return f"Moderate concern - {name} at {confidence * 100:.0f}% confidence"
    return f"Low concern - {name} at {confidence * 100:.0f}% confidence"


__all__ = [
    "CLASS_RISK_WEIGHT",
    "RiskDriver",
    "SeverityAssessment",
    "TIER_GUIDANCE",
    "TIER_ORDER",
    "Tier",
    "grade",
    "morphology_risk",
    "neural_risk",
    "quality_risk",
    "uncertainty_risk",
]
