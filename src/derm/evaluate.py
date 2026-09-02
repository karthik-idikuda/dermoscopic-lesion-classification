"""Evaluation entry point.

    python -m derm.evaluate --checkpoint models/best_model.pth

Produces the numbers a report needs and the ones a reviewer will ask for:

* per-class precision / recall / F1 with support, and macro / weighted averages
* confusion matrix, both raw counts and row-normalised
* one-vs-rest ROC-AUC per class
* **melanoma-specific operating analysis** - sensitivity and specificity across
  thresholds, because at the default argmax threshold a 67%-``nv`` model will
  always under-call melanoma
* **calibration** - expected calibration error and a reliability diagram, with and
  without the fitted temperature
* **severity safety-net audit** - what fraction of true melanomas the grading
  engine escalates to HIGH or CRITICAL, which is the metric that actually matters
  for a triage tool
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import CLASS_CODES, DOCS_DIR, FIGURES_DIR, LESION_CLASSES, SETTINGS
from .data import load_metadata, make_splits, SkinLesionDataset
from .model import build_eval_transform, create_bundle
from .severity import grade

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def per_class_metrics(targets: np.ndarray, predictions: np.ndarray) -> list[dict]:
    """Precision, recall, F1 and support for every class."""
    rows: list[dict] = []
    for index, code in enumerate(CLASS_CODES):
        tp = float(np.sum((predictions == index) & (targets == index)))
        fp = float(np.sum((predictions == index) & (targets != index)))
        fn = float(np.sum((predictions != index) & (targets == index)))
        tn = float(np.sum((predictions != index) & (targets != index)))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
        rows.append(
            {
                "code": code,
                "name": LESION_CLASSES[code].name,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "specificity": round(specificity, 4),
                "f1": round(f1, 4),
                "support": int(tp + fn),
            }
        )
    return rows


def expected_calibration_error(
    probabilities: np.ndarray, targets: np.ndarray, bins: int = 15
) -> tuple[float, list[dict]]:
    """ECE plus the per-bin data needed for a reliability diagram."""
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == targets).astype(np.float64)

    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    detail: list[dict] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        count = int(mask.sum())
        if count == 0:
            detail.append({"bin": [round(lower, 3), round(upper, 3)], "count": 0,
                           "accuracy": None, "confidence": None})
            continue
        accuracy = float(correct[mask].mean())
        confidence = float(confidences[mask].mean())
        error += (count / len(targets)) * abs(accuracy - confidence)
        detail.append(
            {
                "bin": [round(lower, 3), round(upper, 3)],
                "count": count,
                "accuracy": round(accuracy, 4),
                "confidence": round(confidence, 4),
            }
        )
    return float(error), detail


def melanoma_operating_curve(
    probabilities: np.ndarray, targets: np.ndarray
) -> list[dict]:
    """Sensitivity/specificity for melanoma across probability thresholds."""
    mel_index = CLASS_CODES.index("mel")
    scores = probabilities[:, mel_index]
    is_melanoma = targets == mel_index
    rows: list[dict] = []
    for threshold in np.arange(0.05, 1.0, 0.05):
        flagged = scores >= threshold
        tp = float(np.sum(flagged & is_melanoma))
        fn = float(np.sum(~flagged & is_melanoma))
        fp = float(np.sum(flagged & ~is_melanoma))
        tn = float(np.sum(~flagged & ~is_melanoma))
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "sensitivity": round(tp / (tp + fn), 4) if tp + fn else 0.0,
                "specificity": round(tn / (tn + fp), 4) if tn + fp else 0.0,
                "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
                "flagged": int(flagged.sum()),
            }
        )
    return rows


def safety_net_audit(
    probabilities: np.ndarray, targets: np.ndarray, *, trained: bool = True
) -> dict:
    """How often does severity grading escalate a true melanoma?

    This is the headline number for a triage tool: at argmax the classifier will
    miss melanomas, but the grading engine's overrides are designed to catch them
    anyway. Measured here on the classifier signal alone (no morphometry), which
    is the conservative case.
    """
    mel_index = CLASS_CODES.index("mel")
    tiers: list[str] = []
    for row in probabilities:
        assessment = grade(row, CLASS_CODES, model_is_trained=trained)
        tiers.append(assessment.tier)

    tiers_array = np.array(tiers)
    is_melanoma = targets == mel_index
    escalated = np.isin(tiers_array, ["HIGH", "CRITICAL"])

    total_melanoma = int(is_melanoma.sum())
    caught = int((is_melanoma & escalated).sum())
    benign = ~np.isin(
        targets, [CLASS_CODES.index(c) for c in CLASS_CODES if LESION_CLASSES[c].is_malignant]
    )

    return {
        "true_melanoma": total_melanoma,
        "melanoma_escalated": caught,
        "melanoma_catch_rate": round(caught / total_melanoma, 4) if total_melanoma else 0.0,
        "benign_escalated": int((benign & escalated).sum()),
        "benign_total": int(benign.sum()),
        "benign_escalation_rate": (
            round(float((benign & escalated).sum() / benign.sum()), 4)
            if benign.sum()
            else 0.0
        ),
        "tier_distribution": {
            tier: int((tiers_array == tier).sum()) for tier in sorted(set(tiers))
        },
    }


def roc_auc_ovr(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """One-vs-rest ROC-AUC per class, plus the macro average."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:  # pragma: no cover
        return {}

    scores: dict[str, float] = {}
    for index, code in enumerate(CLASS_CODES):
        binary = (targets == index).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            continue
        scores[code] = round(float(roc_auc_score(binary, probabilities[:, index])), 4)
    if scores:
        scores["macro"] = round(float(np.mean(list(scores.values()))), 4)
    return scores


# --------------------------------------------------------------------------- #
# Inference over the test split
# --------------------------------------------------------------------------- #


@torch.no_grad()
def collect_logits(bundle, loader, device) -> tuple[np.ndarray, np.ndarray]:
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    bundle.model.eval()
    for images, labels in loader:
        logits = bundle.model(images.to(device))
        all_logits.append(logits.float().cpu().numpy())
        all_targets.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_targets)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / max(temperature, 1e-6)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def evaluate(
    checkpoint: Path | None = None,
    *,
    dataset_root: Path | None = None,
    batch_size: int = 32,
    num_workers: int = 2,
    group_by_lesion: bool = True,
    output_dir: Path | None = None,
) -> dict:
    """Evaluate a checkpoint on the held-out test split."""
    output_dir = output_dir or DOCS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    # JSON stays at the top of output_dir; plots go into its figures/ subfolder,
    # which is what the API serves. Honour DERM_FIGURES_DIR for the default case.
    figures_dir = FIGURES_DIR if output_dir == DOCS_DIR else output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    bundle = create_bundle(checkpoint)
    if not bundle.is_trained:
        logger.warning(
            "No trained weights loaded - every metric below describes an "
            "untrained network and is meaningless. %s",
            "; ".join(bundle.warnings),
        )

    frame = load_metadata(dataset_root)
    splits = make_splits(
        frame,
        val_size=SETTINGS.train.val_size,
        test_size=SETTINGS.train.test_size,
        seed=SETTINGS.train.seed,
        group_by_lesion=group_by_lesion,
    )
    dataset = SkinLesionDataset(splits.test, build_eval_transform())
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    logger.info("Evaluating on %d test images", len(dataset))

    logits, targets = collect_logits(bundle, loader, bundle.device)
    raw_probabilities = softmax(logits, 1.0)
    calibrated = softmax(logits, bundle.temperature)
    predictions = calibrated.argmax(axis=1)

    accuracy = float((predictions == targets).mean())
    class_rows = per_class_metrics(targets, predictions)
    macro_f1 = float(np.mean([row["f1"] for row in class_rows]))
    supports = np.array([row["support"] for row in class_rows], dtype=np.float64)
    weighted_f1 = (
        float(np.average([row["f1"] for row in class_rows], weights=supports))
        if supports.sum()
        else 0.0
    )

    ece_raw, _ = expected_calibration_error(raw_probabilities, targets)
    ece_calibrated, reliability = expected_calibration_error(calibrated, targets)

    confusion = np.zeros((len(CLASS_CODES), len(CLASS_CODES)), dtype=int)
    for true_index, predicted_index in zip(targets, predictions):
        confusion[true_index, predicted_index] += 1

    results = {
        "checkpoint": str(bundle.checkpoint_path) if bundle.checkpoint_path else None,
        "weights_status": bundle.weights_status,
        # Carried through from the checkpoint so the report can never present a
        # linear probe as if it were an end-to-end fine-tuned model.
        "training_method": bundle.metadata.get("training_method", "unknown"),
        "temperature": round(bundle.temperature, 4),
        "grouped_by_lesion": group_by_lesion,
        "test_images": int(len(dataset)),
        "test_lesions": int(splits.test["lesion_id"].nunique()),
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(
            float(np.mean([row["recall"] for row in class_rows])), 4
        ),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": class_rows,
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_normalized": np.round(
            confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1), 4
        ).tolist(),
        "roc_auc": roc_auc_ovr(calibrated, targets),
        "calibration": {
            "ece_uncalibrated": round(ece_raw, 4),
            "ece_calibrated": round(ece_calibrated, 4),
            "reliability": reliability,
        },
        "melanoma_operating_curve": melanoma_operating_curve(calibrated, targets),
        "safety_net": safety_net_audit(calibrated, targets, trained=bundle.is_trained),
        "splits": splits.describe(),
    }

    (output_dir / "evaluation.json").write_text(json.dumps(results, indent=2))
    _plot_confusion(confusion, figures_dir / "efficientnet_confusion_matrix.png")
    _plot_reliability(reliability, ece_calibrated, figures_dir / "calibration.png")
    _update_comparison(results, output_dir / "model_comparison.json")

    _print_summary(results)
    return results


def _print_summary(results: dict) -> None:
    print("\n" + "=" * 74)
    print(f"Test images {results['test_images']} across {results['test_lesions']} lesions")
    print(f"Lesion-grouped split : {results['grouped_by_lesion']}")
    print(f"Accuracy             : {results['accuracy'] * 100:.2f}%")
    print(f"Balanced accuracy    : {results['balanced_accuracy'] * 100:.2f}%")
    print(f"Macro F1             : {results['macro_f1']:.4f}")
    print(f"ECE (calibrated)     : {results['calibration']['ece_calibrated']:.4f}"
          f"  (raw {results['calibration']['ece_uncalibrated']:.4f})")
    print("-" * 74)
    print(f"{'class':<8}{'precision':>11}{'recall':>9}{'spec':>9}{'f1':>9}{'support':>9}")
    for row in results["per_class"]:
        print(
            f"{row['code']:<8}{row['precision']:>11.3f}{row['recall']:>9.3f}"
            f"{row['specificity']:>9.3f}{row['f1']:>9.3f}{row['support']:>9d}"
        )
    net = results["safety_net"]
    print("-" * 74)
    print(
        f"Melanoma safety net: {net['melanoma_escalated']}/{net['true_melanoma']} "
        f"true melanomas escalated to HIGH or CRITICAL "
        f"({net['melanoma_catch_rate'] * 100:.1f}%)"
    )
    print(
        f"Benign escalation cost: {net['benign_escalated']}/{net['benign_total']} "
        f"({net['benign_escalation_rate'] * 100:.1f}%)"
    )
    print("=" * 74)


def _plot_confusion(confusion: np.ndarray, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:  # pragma: no cover
        return

    normalized = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    figure, axes = plt.subplots(1, 2, figsize=(19, 8))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Greens", ax=axes[0],
                xticklabels=CLASS_CODES, yticklabels=CLASS_CODES, cbar=False)
    axes[0].set_title("Counts", fontweight="bold")
    sns.heatmap(normalized, annot=True, fmt=".2f", cmap="Blues", ax=axes[1], vmin=0, vmax=1,
                xticklabels=CLASS_CODES, yticklabels=CLASS_CODES, cbar=False)
    axes[1].set_title("Row-normalised (per-class recall on the diagonal)", fontweight="bold")
    for axis in axes:
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
    figure.suptitle(
        f"{SETTINGS.model.architecture} - test set confusion matrix",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_reliability(reliability: list[dict], ece: float, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return

    points = [b for b in reliability if b["count"] and b["accuracy"] is not None]
    if not points:
        return
    confidences = [b["confidence"] for b in points]
    accuracies = [b["accuracy"] for b in points]

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.plot([0, 1], [0, 1], "--", color="#888888", label="perfect calibration")
    axis.plot(confidences, accuracies, "o-", color="#c0392b", label="model")
    axis.set_xlabel("Mean predicted confidence")
    axis.set_ylabel("Observed accuracy")
    axis.set_title(f"Reliability diagram (ECE = {ece:.4f})", fontweight="bold")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _update_comparison(results: dict, path: Path) -> None:
    """Merge the new numbers into ``model_comparison.json``, keeping history."""
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}

    per_class = {row["code"]: row for row in results["per_class"]}
    # Name the entry after how the weights were produced, not just the
    # architecture: a linear probe and a fine-tuned model are not comparable,
    # and collapsing them under one key invites exactly that mistake.
    method = results.get("training_method") or "measured"
    suffix = {
        "linear_probe_on_frozen_imagenet_features": "_linear_probe",
        "finetuned": "_finetuned",
    }.get(method, "")
    key = f"{SETTINGS.model.architecture}{suffix}"
    existing[key] = {
        "verified": "measured",
        "training_method": method,
        "accuracy": round(results["accuracy"] * 100, 2),
        "balanced_accuracy": round(results["balanced_accuracy"] * 100, 2),
        "macro_f1": results["macro_f1"],
        "melanoma_recall": round(per_class.get("mel", {}).get("recall", 0.0) * 100, 2),
        "vasc_recall": round(per_class.get("vasc", {}).get("recall", 0.0) * 100, 2),
        "df_recall": round(per_class.get("df", {}).get("recall", 0.0) * 100, 2),
        "roc_auc_macro": results["roc_auc"].get("macro"),
        "ece": results["calibration"]["ece_calibrated"],
        "melanoma_safety_net_catch_rate": round(
            results["safety_net"]["melanoma_catch_rate"] * 100, 2
        ),
        "grouped_by_lesion": results["grouped_by_lesion"],
        "test_images": results["test_images"],
    }
    path.write_text(json.dumps(existing, indent=4))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on the HAM10000 test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--image-wise-split", action="store_true",
                        help="Evaluate on the original leaky split for comparison")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    args = build_parser().parse_args(argv)
    evaluate(
        args.checkpoint,
        dataset_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        group_by_lesion=not args.image_wise_split,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
