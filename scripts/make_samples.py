#!/usr/bin/env python3
"""Pick a small set of demo images, one folder per diagnosis.

    python scripts/make_samples.py --per-class 3

Writes ``samples/<code>-<name>/<image_id>.jpg`` plus ``samples/INDEX.md``.

Selection is deterministic (seeded) and drawn only from the **test** split of
the lesion-grouped partition, so nothing here was seen while the classifier head
was fitted. Demoing on training images would be a quiet way of overstating how
well the system works.

Images are only ever copied, never modified, so what the reviewer sees is what
HAM10000 actually contains.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from derm.config import CLASS_CODES, DATA_DIR, LESION_CLASSES  # noqa: E402
from derm.data import load_metadata, make_splits  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=DATA_DIR / "ham10000")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "samples")
    p.add_argument("--per-class", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    frame = load_metadata(args.data_root)
    splits = make_splits(frame, group_by_lesion=True)
    test = splits.test

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    lines = [
        "# Sample dermoscopic images",
        "",
        "Drawn from the **held-out test split** of the lesion-grouped partition, so none",
        "of these images influenced the classifier head. Ground truth is in the folder",
        "name and in the table below — useful for checking whether a prediction is right.",
        "",
        "> These are real patient images from HAM10000 (CC BY-NC 4.0). Cite Tschandl et al.",
        "> 2018 if you reproduce them. Research and educational use only.",
        "",
        "| File | True diagnosis | Class | Localisation | Age | Sex |",
        "|---|---|---|---|---|---|",
    ]

    total = 0
    print(f"Sampling {args.per_class} test image(s) per class\n")
    for code in CLASS_CODES:
        meta = LESION_CLASSES[code]
        subset = test[test["dx"] == code]
        if subset.empty:
            print(f"  {code}: none in test split, skipped")
            continue

        take = subset.sample(n=min(args.per_class, len(subset)),
                             random_state=args.seed)
        folder = args.out / f"{code}-{meta.short_name.lower().replace(' ', '-')}"
        folder.mkdir(parents=True, exist_ok=True)

        for _, row in take.iterrows():
            src = Path(row["image_path"])
            if not src.exists():
                continue
            dst = folder / f"{row['image_id']}.jpg"
            shutil.copy2(src, dst)
            total += 1
            age = "—" if str(row["age"]) in {"nan", "None", ""} else f"{float(row['age']):.0f}"
            lines.append(
                f"| `{folder.name}/{dst.name}` | {meta.name} | {meta.malignancy} "
                f"| {row['localization']} | {age} | {row['sex']} |"
            )
        print(f"  {code:6s} {meta.short_name:24s} {len(take)} image(s) -> {folder.name}/")

    lines += [
        "",
        "## How to use these",
        "",
        "1. Start the app: `uvicorn app.main:app --reload`",
        "2. Open <http://127.0.0.1:8000> and drag one of these files onto the dropzone.",
        "3. Compare the predicted class against the true diagnosis above.",
        "",
        "Good ones to demonstrate with:",
        "",
        "- a **melanoma** (`mel-melanoma/`) to show the severity override escalating the case",
        "- a **melanocytic nevus** (`nv-melanocytic-nevus/`) as the benign contrast",
        "- a **vascular lesion** (`vasc-vascular-lesion/`) to show ABCD colour scoring",
        "",
        "For the *Track change* view, upload two different images of the same class as a",
        "stand-in for two visits — real longitudinal pairs are not part of HAM10000.",
    ]

    (args.out / "INDEX.md").write_text("\n".join(lines) + "\n")
    print(f"\n{total} images -> {args.out.relative_to(PROJECT_ROOT)}/")
    print(f"Index written to {(args.out / 'INDEX.md').relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
