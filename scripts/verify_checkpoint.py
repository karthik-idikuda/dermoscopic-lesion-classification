"""Validate a trained checkpoint before trusting it.

    python scripts/verify_checkpoint.py models/best_model.pth

A checkpoint that fails loudly is harmless. The dangerous case is one that loads
cleanly but is subtly wrong — a mismatched class order, a head that never
trained, or different preprocessing than the serving code applies — because it
then produces confident, plausible, wrong diagnoses. These checks target exactly
that.

With ``--images DIR`` the script also runs the checkpoint over real labelled
images and reports accuracy, which is the only check that truly proves the
class order is right.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from derm.config import CLASS_CODES, LESION_CLASSES, SETTINGS  # noqa: E402
from derm.model import build_model, create_bundle, load_checkpoint  # noqa: E402
from derm.uncertainty import normalized_entropy  # noqa: E402

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results: list[tuple[str, str, str]] = []


def check(name: str, status: str, detail: str = "") -> None:
    results.append((status, name, detail))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?",
                        default=SETTINGS.checkpoint_path)
    parser.add_argument("--images", type=Path, default=None,
                        help="HAM10000 root, to measure real accuracy on the test split")
    parser.add_argument("--limit", type=int, default=300,
                        help="How many test images to score with --images")
    args = parser.parse_args()

    path = args.checkpoint
    print(f"\nVerifying {path}\n{'=' * 72}\n")

    # ---- 1. exists and loads --------------------------------------------- #
    print("1. File")
    if not path.exists():
        check("checkpoint exists", FAIL, f"nothing at {path}")
        print(
            "\nNo checkpoint to verify. Either train one:\n"
            "    python -m derm.train --device mps\n"
            "or copy the best_model.pth produced by notebook 03 into models/.\n"
        )
        return 1
    size_mb = path.stat().st_size / 1e6
    check("checkpoint exists", PASS, f"{size_mb:.1f} MB")
    if size_mb < 5:
        check("plausible size", WARN, "unusually small for EfficientNet-B3 (~49 MB)")
    else:
        check("plausible size", PASS)

    try:
        state, metadata = load_checkpoint(path)
    except Exception as exc:  # noqa: BLE001
        check("loads with torch.load", FAIL, str(exc))
        return 1
    check("loads with torch.load", PASS, f"{len(state)} tensors")

    # ---- 2. architecture match ------------------------------------------- #
    print("\n2. Architecture")
    reference = build_model(SETTINGS.model, pretrained=False)
    reference_state = reference.state_dict()

    # load_state_dict(strict=False) tolerates missing and unexpected keys but
    # still raises on shape mismatches, so incompatible tensors are separated
    # out first and reported as a finding rather than an exception.
    mismatched = {
        key: (tuple(tensor.shape), tuple(reference_state[key].shape))
        for key, tensor in state.items()
        if key in reference_state and tensor.shape != reference_state[key].shape
    }
    if mismatched:
        for key, (got, want) in list(mismatched.items())[:4]:
            check(f"shape of {key}", FAIL, f"checkpoint {got}, model expects {want}")
        if len(mismatched) > 4:
            check("shape mismatches", FAIL, f"{len(mismatched)} tensors in total")
    else:
        check("all tensor shapes match", PASS)

    loadable = {k: v for k, v in state.items() if k not in mismatched}
    missing, unexpected = reference.load_state_dict(loadable, strict=False)
    missing = [k for k in missing if k not in mismatched]

    if missing:
        check("all parameters present", FAIL,
              f"{len(missing)} missing, first: {missing[0]}")
    else:
        check("all parameters present", PASS)

    if unexpected:
        check("no unexpected keys", WARN,
              f"{len(unexpected)} ignored, first: {unexpected[0]}")
    else:
        check("no unexpected keys", PASS)

    head_missing = [k for k in missing if "classifier" in k]
    head_mismatched = [k for k in mismatched if "classifier" in k]
    if head_missing or head_mismatched:
        check("classification head loaded", FAIL,
              "the head was NOT loaded — predictions would be random")
    else:
        check("classification head loaded", PASS)

    if mismatched:
        print("\nThis checkpoint is not compatible with the serving architecture "
              f"({SETTINGS.model.architecture}, {len(CLASS_CODES)} classes). Stopping.")
        return 1

    weight_key = next((k for k in state if k.endswith("classifier.weight")), None)
    if weight_key is None:
        check("head shape", WARN, "no classifier.weight found")
    else:
        shape = tuple(state[weight_key].shape)
        expected_classes = len(CLASS_CODES)
        if shape[0] == expected_classes:
            check("head shape", PASS, f"{shape} → {expected_classes} classes")
        else:
            check("head shape", FAIL,
                  f"{shape} outputs {shape[0]} classes, expected {expected_classes}")

    # ---- 3. class order --------------------------------------------------- #
    print("\n3. Class order")
    declared = metadata.get("class_codes")
    if declared:
        if list(declared) == list(CLASS_CODES):
            check("class order recorded and matches", PASS, str(list(declared)))
        else:
            check("class order recorded and matches", FAIL,
                  f"checkpoint says {list(declared)}, serving expects {list(CLASS_CODES)}")
    else:
        check(
            "class order recorded", WARN,
            "bare state_dict with no class list. Serving assumes alphabetical "
            f"{list(CLASS_CODES)}, which is what sklearn LabelEncoder produces. "
            "Verify with --images.",
        )

    # ---- 4. metadata ------------------------------------------------------ #
    print("\n4. Recorded metadata")
    if metadata:
        for key in ("architecture", "image_size", "temperature", "epoch",
                    "test_accuracy", "test_macro_f1", "trained_at",
                    "grouped_by_lesion"):
            if key in metadata:
                check(key, PASS, str(metadata[key]))
        if "grouped_by_lesion" in metadata and not metadata["grouped_by_lesion"]:
            check("leak-free split", WARN,
                  "trained on an image-wise split; reported accuracy is inflated")
    else:
        check("metadata present", WARN,
              "bare state_dict — no metrics, temperature or provenance recorded")

    # ---- 5. behaviour ----------------------------------------------------- #
    print("\n5. Behaviour")
    bundle = create_bundle(path, device="cpu")
    if not bundle.is_trained:
        check("bundle reports trained", FAIL, "; ".join(bundle.warnings))
        return 1
    check("bundle reports trained", PASS, f"weights_status={bundle.weights_status}")

    generator = torch.Generator().manual_seed(0)
    batch = torch.randn(8, 3, SETTINGS.model.image_size, SETTINGS.model.image_size,
                        generator=generator)

    first = bundle.probabilities(batch).numpy()
    second = bundle.probabilities(batch).numpy()
    if np.allclose(first, second, atol=1e-6):
        check("deterministic in eval mode", PASS)
    else:
        check("deterministic in eval mode", FAIL,
              "repeated passes differ — dropout may be active at inference")

    if np.allclose(first.sum(axis=1), 1.0, atol=1e-4):
        check("probabilities normalised", PASS)
    else:
        check("probabilities normalised", FAIL, f"row sums {first.sum(axis=1)}")

    # An untrained head maps everything to ~uniform (entropy ≈ 1.0). A trained
    # one should differentiate, even on noise.
    entropies = [normalized_entropy(row) for row in first]
    mean_entropy = float(np.mean(entropies))
    if mean_entropy > 0.985:
        check("head is discriminative", FAIL,
              f"mean normalised entropy {mean_entropy:.4f} — output is essentially "
              "uniform, so the head looks untrained")
    elif mean_entropy > 0.93:
        check("head is discriminative", WARN,
              f"mean normalised entropy {mean_entropy:.4f} — unusually flat")
    else:
        check("head is discriminative", PASS,
              f"mean normalised entropy {mean_entropy:.4f}")

    predicted = {CLASS_CODES[i] for i in first.argmax(axis=1)}
    check("predicted classes on noise", PASS, f"{sorted(predicted)}")

    # ---- 6. real accuracy ------------------------------------------------- #
    if args.images:
        print("\n6. Real accuracy on the held-out test split")
        try:
            code = measure_accuracy(bundle, args.images, args.limit)
            if code != 0:
                return code
        except Exception as exc:  # noqa: BLE001
            check("accuracy measurement", FAIL, str(exc))
    else:
        print("\n6. Real accuracy")
        check(
            "measured on real images", WARN,
            "skipped. Pass --images data/ham10000 to confirm the class order is "
            "genuinely right; every check above can pass with a permuted head.",
        )

    # ---- summary ---------------------------------------------------------- #
    failures = [r for r in results if r[0] == FAIL]
    warnings = [r for r in results if r[0] == WARN]
    print(f"\n{'=' * 72}")
    print(f"{len(results) - len(failures) - len(warnings)} passed, "
          f"{len(warnings)} warning(s), {len(failures)} failure(s)")
    if failures:
        print("\nFAILURES:")
        for _, name, detail in failures:
            print(f"  - {name}: {detail}")
        print("\nThis checkpoint should not be served.")
        return 1
    if warnings:
        print("\nWarnings:")
        for _, name, detail in warnings:
            print(f"  - {name}: {detail}")
    print("\nCheckpoint looks usable.")
    return 0


def measure_accuracy(bundle, root: Path, limit: int) -> int:
    """Score the checkpoint on the lesion-grouped test split."""
    from torch.utils.data import DataLoader

    from derm.data import SkinLesionDataset, load_metadata, make_splits
    from derm.model import build_eval_transform

    frame = load_metadata(root)
    splits = make_splits(
        frame,
        val_size=SETTINGS.train.val_size,
        test_size=SETTINGS.train.test_size,
        seed=SETTINGS.train.seed,
        group_by_lesion=True,
    )
    test = splits.test
    if limit and len(test) > limit:
        # Sample per class so every class is represented in a quick run.
        per_class = max(1, limit // len(CLASS_CODES))
        test = (
            test.groupby("dx", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), per_class), random_state=0))
            .reset_index(drop=True)
        )

    loader = DataLoader(
        SkinLesionDataset(test, build_eval_transform()),
        batch_size=16, shuffle=False, num_workers=0,
    )

    correct = 0
    total = 0
    per_class_correct = dict.fromkeys(CLASS_CODES, 0)
    per_class_total = dict.fromkeys(CLASS_CODES, 0)

    for images, labels in loader:
        probs = bundle.probabilities(images).numpy()
        predictions = probs.argmax(axis=1)
        for prediction, label in zip(predictions, labels.numpy()):
            code = CLASS_CODES[label]
            per_class_total[code] += 1
            if prediction == label:
                correct += 1
                per_class_correct[code] += 1
            total += 1

    accuracy = correct / max(total, 1)
    check("images scored", PASS, f"{total} from the lesion-grouped test split")

    # Chance is ~14.3%; the nv-only baseline is ~67%. Anything at or below
    # chance means the head or the class order is wrong.
    if accuracy < 0.2:
        check("accuracy above chance", FAIL,
              f"{accuracy * 100:.1f}% — at or below the 14.3% chance level, so the "
              "class order is almost certainly permuted")
    elif accuracy < 0.5:
        check("accuracy above chance", WARN, f"{accuracy * 100:.1f}% — low")
    else:
        check("accuracy above chance", PASS, f"{accuracy * 100:.1f}%")

    print("\n    per-class recall:")
    recalls = []
    for code in CLASS_CODES:
        n = per_class_total[code]
        if not n:
            continue
        recall = per_class_correct[code] / n
        recalls.append(recall)
        print(f"      {code:<6} {recall * 100:>5.1f}%  (n={n:>3})  "
              f"{LESION_CLASSES[code].short_name}")

    # A model that only ever predicts `nv` scores ~67% accuracy but has near-zero
    # recall elsewhere. Balanced recall catches that collapse.
    balanced = float(np.mean(recalls)) if recalls else 0.0
    if balanced < 0.25:
        check("balanced recall", FAIL,
              f"{balanced * 100:.1f}% — the model has collapsed onto the majority class")
    elif balanced < 0.45:
        check("balanced recall", WARN, f"{balanced * 100:.1f}%")
    else:
        check("balanced recall", PASS, f"{balanced * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
