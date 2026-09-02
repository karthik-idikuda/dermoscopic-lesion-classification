"""Uncertainty quantification for the classifier.

A softmax score is not a probability of being correct. A network trained with
class weights on a 67%-``nv`` dataset is especially prone to confident mistakes,
so three complementary signals are computed:

* **Predictive entropy** - how flat the distribution is (aleatoric-ish).
* **Test-time augmentation spread** - does the prediction survive flipping and
  rotating the image? Dermoscopic images have no canonical orientation, so a
  prediction that flips when the image does is not a real one.
* **MC dropout** - keeping dropout active at inference approximates sampling
  from the posterior over weights, giving an epistemic estimate plus the BALD
  mutual-information score.

The combined verdict drives escalation in :mod:`derm.severity`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .model import ModelBundle


@dataclass
class UncertaintyReport:
    """All uncertainty signals for one image."""

    probabilities: np.ndarray  # final (TTA-averaged when enabled) distribution
    entropy: float  # normalised 0-1
    margin: float  # top1 - top2 probability
    tta_agreement: float  # 0-1 share of augmentations agreeing with top-1
    tta_std: float  # std of the top-1 probability across augmentations
    mc_std: float  # mean per-class std across MC dropout passes
    mutual_information: float  # BALD score, normalised 0-1
    verdict: str  # "confident" | "borderline" | "uncertain"
    n_tta: int = 0
    n_mc: int = 0
    per_augmentation: list[int] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        return float(self.probabilities.max())

    @property
    def predicted_index(self) -> int:
        return int(self.probabilities.argmax())

    def to_dict(self) -> dict:
        # Spread and mutual information are reported at 6 decimals rather than 4.
        # They are genuinely small quantities - an untrained head emits logits of
        # order 1e-5, so the MC spread is ~1e-6 - and rounding at 4 places
        # displays a real non-zero value as a misleading exact 0.0.
        return {
            "entropy": round(self.entropy, 4),
            "margin": round(self.margin, 6),
            "tta_agreement": round(self.tta_agreement, 3),
            "tta_std": round(self.tta_std, 6),
            "mc_std": round(self.mc_std, 6),
            "mutual_information": round(self.mutual_information, 6),
            "verdict": self.verdict,
            "n_tta": self.n_tta,
            "n_mc": self.n_mc,
        }


# --------------------------------------------------------------------------- #
# Scalar measures
# --------------------------------------------------------------------------- #


def normalized_entropy(probabilities: np.ndarray) -> float:
    """Shannon entropy scaled to ``[0, 1]`` by dividing by ``log(num_classes)``."""
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    probs = probs / probs.sum()
    entropy = float(-(probs * np.log(probs)).sum())
    return float(np.clip(entropy / math.log(len(probs)), 0.0, 1.0))


def top_margin(probabilities: np.ndarray) -> float:
    """Gap between the top two probabilities; small means the model is torn."""
    ordered = np.sort(np.asarray(probabilities, dtype=np.float64))[::-1]
    return float(ordered[0] - ordered[1]) if ordered.size >= 2 else float(ordered[0])


# --------------------------------------------------------------------------- #
# Test-time augmentation
# --------------------------------------------------------------------------- #


def _dihedral(batch: torch.Tensor, index: int) -> torch.Tensor:
    """The eight symmetries of the square, indexed 0-7."""
    result = batch
    if index >= 4:
        result = torch.flip(result, dims=[3])
    rotations = index % 4
    if rotations:
        result = torch.rot90(result, k=rotations, dims=[2, 3])
    return result


def tta_predict(
    bundle: ModelBundle, batch: torch.Tensor, n_augmentations: int = 5
) -> tuple[np.ndarray, float, float, list[int]]:
    """Average predictions over dihedral augmentations.

    Returns ``(mean probabilities, agreement, std of top-1 prob, per-aug argmax)``.
    """
    n_augmentations = int(np.clip(n_augmentations, 1, 8))
    distributions: list[np.ndarray] = []

    for index in range(n_augmentations):
        augmented = _dihedral(batch, index)
        probs = bundle.probabilities(augmented)[0].detach().cpu().numpy()
        distributions.append(probs.astype(np.float64))

    stacked = np.stack(distributions)
    mean = stacked.mean(axis=0)
    mean = mean / max(mean.sum(), 1e-12)

    argmaxes = [int(dist.argmax()) for dist in distributions]
    consensus = int(mean.argmax())
    agreement = float(np.mean([a == consensus for a in argmaxes]))
    top_std = float(stacked[:, consensus].std())
    return mean, agreement, top_std, argmaxes


# --------------------------------------------------------------------------- #
# MC dropout
# --------------------------------------------------------------------------- #


_DROPOUT_TYPES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)

_NORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


def _activate_stochasticity(model: nn.Module) -> bool:
    """Turn dropout back on for inference. Returns whether anything was enabled.

    Two mechanisms have to be handled, because timm does not use the one most
    MC-dropout snippets assume:

    * **Module dropout** - ``nn.Dropout`` / ``DropPath`` instances. These can be
      switched to train mode individually, which is the clean case.
    * **Functional dropout** - timm's EfficientNet head calls
      ``F.dropout(x, p=self.drop_rate, training=self.training)`` and contains no
      ``nn.Dropout`` module at all. Scanning for modules finds nothing, so MC
      dropout silently degenerates to repeated identical passes. The only way to
      activate it is to put the model in train mode, which then also switches
      BatchNorm to batch statistics - fatal on a batch of one. So normalisation
      layers are explicitly pushed back to eval afterwards.
    """
    modules = [
        module
        for module in model.modules()
        if isinstance(module, _DROPOUT_TYPES) and getattr(module, "p", 0.0) > 0.0
    ]
    drop_paths = [
        module
        for module in model.modules()
        if module.__class__.__name__ in {"DropPath", "StochasticDepth"}
        and getattr(module, "drop_prob", 0.0) > 0.0
    ]

    if modules or drop_paths:
        for module in modules + drop_paths:
            module.train()
        return True

    if float(getattr(model, "drop_rate", 0.0) or 0.0) > 0.0:
        model.train()
        for module in model.modules():
            if isinstance(module, _NORM_TYPES):
                module.eval()
        return True

    return False


def mc_dropout_predict(
    bundle: ModelBundle, batch: torch.Tensor, passes: int = 10
) -> tuple[np.ndarray, float, float, int]:
    """Sample the predictive distribution with dropout left on.

    Returns ``(mean probabilities, mean per-class std, normalised BALD, passes)``.
    If the architecture has no active dropout, this degrades to a single
    deterministic pass with zero spread, reported honestly via ``passes=0`` so the
    UI can say "not available" instead of implying perfect certainty.
    """
    if passes < 2:
        probabilities = (
            bundle.probabilities(batch)[0].detach().cpu().numpy().astype(np.float64)
        )
        return probabilities, 0.0, 0.0, 0

    try:
        if not _activate_stochasticity(bundle.model):
            probabilities = (
                bundle.probabilities(batch)[0].detach().cpu().numpy().astype(np.float64)
            )
            return probabilities, 0.0, 0.0, 0

        distributions = [
            bundle.probabilities(batch)[0].detach().cpu().numpy().astype(np.float64)
            for _ in range(int(passes))
        ]
    finally:
        bundle.ensure_eval()  # always restore, even if a pass raises

    stacked = np.stack(distributions)
    mean = stacked.mean(axis=0)
    mean = mean / max(mean.sum(), 1e-12)
    spread = float(stacked.std(axis=0).mean())

    # BALD: entropy of the mean minus the mean of the entropies.
    total_entropy = normalized_entropy(mean)
    expected_entropy = float(np.mean([normalized_entropy(d) for d in stacked]))
    mutual_information = float(np.clip(total_entropy - expected_entropy, 0.0, 1.0))
    return mean, spread, mutual_information, len(distributions)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def classify_verdict(
    confidence: float,
    entropy: float,
    agreement: float,
    *,
    low_confidence: float = 0.50,
    high_entropy: float = 0.55,
) -> str:
    """Reduce the signals to one of three labels used across the UI."""
    if confidence < low_confidence or entropy > high_entropy or agreement < 0.6:
        return "uncertain"
    if confidence < 0.70 or entropy > 0.40 or agreement < 0.8:
        return "borderline"
    return "confident"


def estimate(
    bundle: ModelBundle,
    batch: torch.Tensor,
    *,
    use_tta: bool = True,
    use_mc_dropout: bool = True,
    n_tta: int = 5,
    mc_passes: int = 10,
    low_confidence: float = 0.50,
    high_entropy: float = 0.55,
) -> UncertaintyReport:
    """Run the enabled uncertainty estimators and assemble the report."""
    agreement, tta_std, argmaxes, n_tta_used = 1.0, 0.0, [], 0

    if use_tta and n_tta > 1:
        probabilities, agreement, tta_std, argmaxes = tta_predict(bundle, batch, n_tta)
        n_tta_used = min(int(n_tta), 8)
    else:
        probabilities = (
            bundle.probabilities(batch)[0].detach().cpu().numpy().astype(np.float64)
        )

    mc_std, mutual_information, n_mc = 0.0, 0.0, 0
    if use_mc_dropout and mc_passes >= 2:
        _, mc_std, mutual_information, n_mc = mc_dropout_predict(
            bundle, batch, mc_passes
        )

    entropy = normalized_entropy(probabilities)
    confidence = float(probabilities.max())

    return UncertaintyReport(
        probabilities=probabilities,
        entropy=entropy,
        margin=top_margin(probabilities),
        tta_agreement=agreement,
        tta_std=tta_std,
        mc_std=mc_std,
        mutual_information=mutual_information,
        verdict=classify_verdict(
            confidence,
            entropy,
            agreement,
            low_confidence=low_confidence,
            high_entropy=high_entropy,
        ),
        n_tta=n_tta_used,
        n_mc=n_mc,
        per_augmentation=argmaxes,
    )


__all__ = [
    "UncertaintyReport",
    "classify_verdict",
    "estimate",
    "mc_dropout_predict",
    "normalized_entropy",
    "top_margin",
    "tta_predict",
]
