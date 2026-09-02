"""Quantify data leakage in the image-wise HAM10000 split.

    python scripts/audit_leakage.py

Needs only ``HAM10000_metadata.csv`` (700 KB) — no images, no GPU, no training.

HAM10000 contains 10,015 images of 7,470 distinct lesions: 2,545 images are
repeat photographs of a lesion that already appears elsewhere in the dataset.
Splitting on images therefore places near-duplicate photographs of the same
physical lesion into both the training and the test set.

This script reproduces the exact split used in notebook 03
(``train_test_split(test_size=0.3, random_state=42, stratify=label)`` followed by
a 50/50 division of the remainder) and measures how many test images have a
same-lesion twin in the training set. It then does the same for the
lesion-grouped split used by :mod:`derm.data`, which should be exactly zero.

Output is written to ``docs/split_audit.json`` and is fully reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from derm.config import CLASS_CODES, DOCS_DIR, LESION_CLASSES  # noqa: E402


def load_metadata_only(path: Path) -> pd.DataFrame:
    """Read the metadata CSV without requiring the image files to exist."""
    frame = pd.read_csv(path)
    missing = {"lesion_id", "image_id", "dx"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["label"] = frame["dx"].map({c: i for i, c in enumerate(CLASS_CODES)}).astype(int)
    return frame


def notebook_split(frame: pd.DataFrame, seed: int = 42):
    """Reproduce the original image-wise split from notebook 03."""
    from sklearn.model_selection import train_test_split

    train, temp = train_test_split(
        frame, test_size=0.3, random_state=seed, stratify=frame["label"]
    )
    val, test = train_test_split(
        temp, test_size=0.5, random_state=seed, stratify=temp["label"]
    )
    return train, val, test


def measure(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Count images in val/test whose lesion also appears in train."""
    train_lesions = set(train["lesion_id"])

    def leaked(frame: pd.DataFrame) -> dict:
        mask = frame["lesion_id"].isin(train_lesions)
        by_class = {}
        for code in CLASS_CODES:
            subset = frame[frame["dx"] == code]
            if len(subset) == 0:
                continue
            n_leaked = int(subset["lesion_id"].isin(train_lesions).sum())
            by_class[code] = {
                "images": int(len(subset)),
                "leaked": n_leaked,
                "leaked_pct": round(n_leaked / len(subset) * 100, 2),
            }
        return {
            "images": int(len(frame)),
            "lesions": int(frame["lesion_id"].nunique()),
            "leaked_images": int(mask.sum()),
            "leaked_pct": round(float(mask.mean() * 100), 2),
            "by_class": by_class,
        }

    return {"val": leaked(val), "test": leaked(test)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "data" / "ham10000" / "HAM10000_metadata.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DOCS_DIR / "split_audit.json")
    args = parser.parse_args()

    if not args.metadata.exists():
        print(
            f"Metadata not found at {args.metadata}.\n"
            "Fetch it with: python scripts/prepare_data.py  (or pass --metadata)",
            file=sys.stderr,
        )
        return 1

    frame = load_metadata_only(args.metadata)

    total_images = len(frame)
    total_lesions = frame["lesion_id"].nunique()
    per_lesion = frame.groupby("lesion_id").size()
    duplicated_images = int(total_images - total_lesions)

    print("=" * 72)
    print("HAM10000 dataset composition")
    print("=" * 72)
    print(f"  images                       : {total_images:,}")
    print(f"  distinct lesions             : {total_lesions:,}")
    print(f"  repeat images of same lesion : {duplicated_images:,} "
          f"({duplicated_images / total_images * 100:.1f}% of the dataset)")
    print(f"  lesions with >1 image        : {int((per_lesion > 1).sum()):,}")
    print(f"  max images of one lesion     : {int(per_lesion.max())}")

    # ---- image-wise split (the original notebook behaviour) --------------- #
    train, val, test = notebook_split(frame, args.seed)
    image_wise = measure(train, val, test)

    print()
    print("=" * 72)
    print(f"Image-wise split (notebook 03: random_state={args.seed}, stratified)")
    print("=" * 72)
    print(f"  train {len(train):,} · val {len(val):,} · test {len(test):,}")
    print(f"  TEST images whose lesion is also in TRAIN : "
          f"{image_wise['test']['leaked_images']:,} / {image_wise['test']['images']:,} "
          f"({image_wise['test']['leaked_pct']}%)")
    print(f"  VAL  images whose lesion is also in TRAIN : "
          f"{image_wise['val']['leaked_images']:,} / {image_wise['val']['images']:,} "
          f"({image_wise['val']['leaked_pct']}%)")
    print("\n  Per-class leakage in the test set:")
    print(f"    {'class':<8}{'images':>8}{'leaked':>8}{'%':>8}   diagnosis")
    for code, row in image_wise["test"]["by_class"].items():
        print(f"    {code:<8}{row['images']:>8}{row['leaked']:>8}{row['leaked_pct']:>8.1f}   "
              f"{LESION_CLASSES[code].short_name}")

    # ---- lesion-grouped split (derm.data default) ------------------------- #
    from derm.data import make_splits

    splits = make_splits(frame, seed=args.seed, group_by_lesion=True)
    grouped = measure(splits.train, splits.val, splits.test)

    print()
    print("=" * 72)
    print("Lesion-grouped split (derm.data.make_splits, group_by_lesion=True)")
    print("=" * 72)
    print(f"  train {len(splits.train):,} · val {len(splits.val):,} · test {len(splits.test):,}")
    print(f"  TEST images whose lesion is also in TRAIN : "
          f"{grouped['test']['leaked_images']} ({grouped['test']['leaked_pct']}%)")
    print(f"  VAL  images whose lesion is also in TRAIN : "
          f"{grouped['val']['leaked_images']} ({grouped['val']['leaked_pct']}%)")

    print("\n  Class balance preserved by the grouped split:")
    print(f"    {'class':<8}{'overall %':>11}{'train %':>10}{'test %':>10}")
    for code in CLASS_CODES:
        overall = (frame["dx"] == code).mean() * 100
        tr = (splits.train["dx"] == code).mean() * 100
        te = (splits.test["dx"] == code).mean() * 100
        print(f"    {code:<8}{overall:>11.2f}{tr:>10.2f}{te:>10.2f}")

    verdict = (
        "The image-wise split leaks "
        f"{image_wise['test']['leaked_pct']}% of test images, so any accuracy "
        "measured on it is optimistically biased. The lesion-grouped split leaks "
        f"{grouped['test']['leaked_pct']}%."
    )
    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    for line in (verdict[i:i + 70] for i in range(0, len(verdict), 70)):
        print(f"  {line}")

    payload = {
        "source": str(args.metadata),
        "seed": args.seed,
        "reproducible": True,
        "dataset": {
            "images": int(total_images),
            "lesions": int(total_lesions),
            "repeat_images": duplicated_images,
            "repeat_images_pct": round(duplicated_images / total_images * 100, 2),
            "lesions_with_multiple_images": int((per_lesion > 1).sum()),
            "max_images_per_lesion": int(per_lesion.max()),
            "class_counts": {c: int((frame["dx"] == c).sum()) for c in CLASS_CODES},
        },
        "image_wise_split": {
            "description": "notebook 03: train_test_split(test_size=0.3, random_state=42, stratify=dx) then 50/50",
            "sizes": {"train": len(train), "val": len(val), "test": len(test)},
            **image_wise,
        },
        "lesion_grouped_split": {
            "description": "derm.data.make_splits(group_by_lesion=True)",
            "sizes": {
                "train": len(splits.train),
                "val": len(splits.val),
                "test": len(splits.test),
            },
            **grouped,
        },
        "verdict": verdict,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
