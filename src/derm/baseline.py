"""Reproducible SVM baseline (HOG + colour histogram features).

    python -m derm.baseline --limit 3000

This is notebook 02 turned into a script, with two changes: it uses the same
lesion-grouped split as the deep model so the comparison is like-for-like, and
feature extraction is parallelised because 10,015 images is slow in a single
process.

The baseline exists to make one point precisely: 73% accuracy sounds respectable
until you look at per-class recall, where the same model scores 0% on vascular
lesions. That gap is the argument for the CNN.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .config import CLASS_CODES, DOCS_DIR, FIGURES_DIR, LESION_CLASSES, MODELS_DIR, SETTINGS
from .data import load_metadata, make_splits

logger = logging.getLogger(__name__)

FEATURE_SIZE = 64
HOG_ORIENTATIONS = 8
HISTOGRAM_BINS = 32


def extract_features(image_path: str) -> np.ndarray:
    """HOG (shape and edges) concatenated with RGB histograms (colour).

    1,568 HOG + 96 histogram = 1,664 features per image at 64x64.
    """
    from PIL import Image
    from skimage.color import rgb2gray
    from skimage.feature import hog

    with Image.open(image_path) as handle:
        image = handle.convert("RGB").resize((FEATURE_SIZE, FEATURE_SIZE))
    array = np.asarray(image, dtype=np.uint8)

    hog_features = hog(
        rgb2gray(array),
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
    )
    histograms = [
        np.histogram(array[..., channel], bins=HISTOGRAM_BINS, range=(0, 256))[0]
        for channel in range(3)
    ]
    return np.concatenate([hog_features, np.concatenate(histograms)]).astype(np.float32)


def _extract_many(paths: list[str], workers: int) -> np.ndarray:
    if workers <= 1:
        return np.stack([extract_features(p) for p in paths])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return np.stack(list(pool.map(extract_features, paths, chunksize=32)))


def run(
    *,
    dataset_root: Path | None = None,
    limit: int | None = None,
    workers: int = 4,
    output_dir: Path | None = None,
    save_model: bool = True,
) -> dict:
    """Train and evaluate the SVM baseline."""
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    output_dir = output_dir or DOCS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    # Plots go to figures/ so the API can serve them; JSON stays alongside.
    figures_dir = FIGURES_DIR if output_dir == DOCS_DIR else output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    frame = load_metadata(dataset_root)
    splits = make_splits(
        frame,
        val_size=SETTINGS.train.val_size,
        test_size=SETTINGS.train.test_size,
        seed=SETTINGS.train.seed,
        group_by_lesion=True,
    )

    train_frame, test_frame = splits.train, splits.test
    if limit:
        # Sample per class so a quick run still sees every class.
        per_class = max(1, limit // len(CLASS_CODES))
        train_frame = (
            train_frame.groupby("dx", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), per_class), random_state=42))
            .reset_index(drop=True)
        )
        logger.info("Limited training set to %d images", len(train_frame))

    logger.info("Extracting features for %d training images...", len(train_frame))
    x_train = _extract_many(train_frame["image_path"].tolist(), workers)
    logger.info("Extracting features for %d test images...", len(test_frame))
    x_test = _extract_many(test_frame["image_path"].tolist(), workers)

    y_train = train_frame["label"].to_numpy()
    y_test = test_frame["label"].to_numpy()
    logger.info("Feature matrix: %s", x_train.shape)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    logger.info("Training SVM (RBF kernel, C=10, balanced class weights)...")
    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        random_state=42,
        probability=False,
    )
    model.fit(x_train_scaled, y_train)

    predictions = model.predict(x_test_scaled)
    accuracy = float((predictions == y_test).mean())

    labels = list(range(len(CLASS_CODES)))
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=list(CLASS_CODES),
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(y_test, predictions, labels=labels)

    per_class = {
        code: {
            "precision": round(report[code]["precision"], 4),
            "recall": round(report[code]["recall"], 4),
            "f1": round(report[code]["f1-score"], 4),
            "support": int(report[code]["support"]),
        }
        for code in CLASS_CODES
        if code in report
    }

    results = {
        "model": "SVM (HOG + colour histogram)",
        "features": int(x_train.shape[1]),
        "train_images": int(len(train_frame)),
        "test_images": int(len(test_frame)),
        "grouped_by_lesion": True,
        "accuracy": round(accuracy * 100, 2),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "per_class": per_class,
        "melanoma_recall": round(per_class.get("mel", {}).get("recall", 0.0) * 100, 2),
        "vasc_recall": round(per_class.get("vasc", {}).get("recall", 0.0) * 100, 2),
        "df_recall": round(per_class.get("df", {}).get("recall", 0.0) * 100, 2),
        "confusion_matrix": confusion.tolist(),
    }

    (output_dir / "svm_baseline_results.json").write_text(json.dumps(results, indent=2))
    _plot_confusion(confusion, figures_dir / "svm_confusion_matrix.png")
    _merge_comparison(results, output_dir / "model_comparison.json")

    if save_model:
        try:
            import joblib

            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"model": model, "scaler": scaler, "class_codes": list(CLASS_CODES)},
                MODELS_DIR / "svm_baseline.joblib",
            )
        except ImportError:
            logger.warning("joblib not installed; skipping model export.")

    print(f"\nSVM baseline accuracy {results['accuracy']:.2f}%, "
          f"macro-F1 {results['macro_f1']:.4f}")
    for code, row in per_class.items():
        print(f"  {code:<6} recall {row['recall'] * 100:5.1f}%  "
              f"f1 {row['f1']:.3f}  (n={row['support']})  "
              f"{LESION_CLASSES[code].short_name}")
    return results


def _plot_confusion(confusion: np.ndarray, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:  # pragma: no cover
        return
    figure, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", ax=axis,
                xticklabels=CLASS_CODES, yticklabels=CLASS_CODES)
    axis.set_title("SVM baseline - confusion matrix", fontsize=13, fontweight="bold")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _merge_comparison(results: dict, path: Path) -> None:
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing["SVM_baseline"] = {
        "accuracy": results["accuracy"],
        "macro_f1": results["macro_f1"],
        "melanoma_recall": results["melanoma_recall"],
        "vasc_recall": results["vasc_recall"],
        "df_recall": results["df_recall"],
        "grouped_by_lesion": True,
        "test_images": results["test_images"],
    }
    path.write_text(json.dumps(existing, indent=4))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(
        description="Train and evaluate the classical SVM baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the training set for a fast run")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DOCS_DIR)
    args = parser.parse_args(argv)

    run(
        dataset_root=args.data_root,
        limit=args.limit,
        workers=args.workers,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
