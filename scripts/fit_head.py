#!/usr/bin/env python3
"""Produce a real, usable checkpoint without full-network training.

    python scripts/fit_head.py

Why this exists
---------------
Fine-tuning all 12M parameters of EfficientNet-B3 needs a GPU: on an 8 GB Apple
M2 it drives swap past 13 GB and stalls the machine. But the project does not
need full fine-tuning to be *functional* — it needs a classifier head that has
genuinely learned something, so that probabilities, Grad-CAM, uncertainty and
severity grading all produce meaningful output instead of a uniform 1/7.

So this script does the cheap half of transfer learning:

1. **Feature extraction (forward only).** The ImageNet-pretrained backbone is
   frozen and run under ``torch.inference_mode()`` in small batches. No
   gradients, no optimiser state, no activation graph is retained, so peak RAM
   stays near the size of one batch. Features are cached to disk, so this is
   paid once.
2. **Head fitting.** Only ``classifier`` (1536 -> 7, ~10.7k parameters) is
   trained, on the cached 1536-dimensional vectors. That is multinomial logistic
   regression: seconds on CPU, no image decoding, no memory pressure.
3. **Temperature scaling** on the validation split, so the confidence the UI
   displays is calibrated rather than arbitrary.

What this is and is not
-----------------------
This is a **linear probe**, not a fine-tuned model, and the saved checkpoint
records that in its metadata so no downstream report can misrepresent it. A
linear probe on frozen ImageNet features is meaningfully weaker than end-to-end
fine-tuning — expect noticeably lower macro-F1 and melanoma recall. It is
reported honestly rather than dressed up.

Splitting is lesion-grouped by default, so the resulting metrics are not
inflated by the 36% near-duplicate leakage that an image-wise split produces.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from derm.config import (  # noqa: E402
    CLASS_CODES,
    DATA_DIR,
    DOCS_DIR,
    MODELS_DIR,
    SETTINGS,
)
from derm.data import load_metadata, make_splits  # noqa: E402
from derm.model import build_eval_transform, build_model, resolve_device  # noqa: E402


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #


class ImageOnly(Dataset):
    """Yield transformed tensors for a dataframe of image paths."""

    def __init__(self, frame, transform):
        self.paths = frame["image_path"].tolist()
        self.labels = frame["label"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            tensor = self.transform(img.convert("RGB"))
        return tensor, self.labels[i]


@torch.inference_mode()
def extract(model, frame, transform, device, *, batch_size, workers, label):
    """Forward-only feature extraction. Returns (features, labels).

    ``model.forward_features`` + pooling is used rather than the full forward so
    we capture the 1536-d penultimate representation the classifier consumes.
    """
    loader = DataLoader(
        ImageOnly(frame, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
    )
    feats: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    total = len(loader)
    started = time.time()

    for i, (x, y) in enumerate(loader, 1):
        x = x.to(device, non_blocking=False)
        f = model.forward_features(x)
        f = model.global_pool(f) if hasattr(model, "global_pool") else f.mean((2, 3))
        if f.ndim > 2:
            f = torch.flatten(f, 1)
        feats.append(f.float().cpu().numpy())
        ys.append(y.numpy())

        if i % 20 == 0 or i == total:
            done = i / total
            elapsed = time.time() - started
            eta = elapsed / done - elapsed
            print(f"    {label}: {i}/{total} batches  "
                  f"{done * 100:5.1f}%  eta {eta / 60:4.1f} min", flush=True)

    return np.concatenate(feats), np.concatenate(ys)


def cached_features(model, splits, transform, device, cache: Path, *,
                    batch_size, workers):
    """Extract features for all three splits, memoised on disk."""
    if cache.exists():
        print(f"  reusing cached features at {cache.relative_to(PROJECT_ROOT)}")
        blob = np.load(cache)
        return {s: (blob[f"{s}_x"], blob[f"{s}_y"]) for s in ("train", "val", "test")}

    out = {}
    for name, frame in (("train", splits.train), ("val", splits.val),
                        ("test", splits.test)):
        print(f"  extracting {name} ({len(frame)} images)")
        out[name] = extract(model, frame, transform, device, batch_size=batch_size,
                            workers=workers, label=name)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        **{f"{s}_x": out[s][0] for s in out},
        **{f"{s}_y": out[s][1] for s in out},
    )
    print(f"  cached to {cache.relative_to(PROJECT_ROOT)}")
    return out


# --------------------------------------------------------------------------- #
# Head fitting
# --------------------------------------------------------------------------- #


def fit_head(train, val, *, epochs, lr, weight_decay, device, seed=42):
    """Fit a linear classifier on cached features, selecting on macro-F1.

    Class-weighted loss is essential here: 67% of HAM10000 is ``nv``, and an
    unweighted fit produces a head that essentially never predicts melanoma.
    """
    from sklearn.metrics import f1_score

    torch.manual_seed(seed)
    xtr = torch.from_numpy(train[0]).float()
    ytr = torch.from_numpy(train[1]).long()
    xva = torch.from_numpy(val[0]).float()
    yva = torch.from_numpy(val[1]).long()

    # Standardise: linear models on unnormalised CNN features converge poorly.
    mean = xtr.mean(0, keepdim=True)
    std = xtr.std(0, keepdim=True).clamp_min(1e-6)
    xtr_n, xva_n = (xtr - mean) / std, (xva - mean) / std

    counts = np.bincount(train[1], minlength=len(CLASS_CODES)).astype(np.float64)
    weights = torch.from_numpy((counts.sum() / (len(counts) * counts))).float()
    print("  class weights: " + ", ".join(
        f"{c}={w:.2f}" for c, w in zip(CLASS_CODES, weights.tolist())))

    head = nn.Linear(xtr.shape[1], len(CLASS_CODES))
    optim = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    best = {"f1": -1.0, "state": None, "epoch": -1}
    order = torch.randperm(len(xtr_n))
    batch = 512

    for epoch in range(1, epochs + 1):
        head.train()
        order = order[torch.randperm(len(order))]
        running = 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            optim.zero_grad(set_to_none=True)
            loss = loss_fn(head(xtr_n[idx]), ytr[idx])
            loss.backward()
            optim.step()
            running += loss.item() * len(idx)
        sched.step()

        head.eval()
        with torch.no_grad():
            pred = head(xva_n).argmax(1).numpy()
        f1 = f1_score(yva.numpy(), pred, average="macro", zero_division=0)
        if f1 > best["f1"]:
            best = {"f1": f1, "state": {k: v.clone() for k, v in head.state_dict().items()},
                    "epoch": epoch}
        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d}  loss {running / len(order):.4f}  "
                  f"val macro-F1 {f1:.4f}", flush=True)

    print(f"  best val macro-F1 {best['f1']:.4f} at epoch {best['epoch']}")
    head.load_state_dict(best["state"])
    return head, mean, std, best["f1"]


def fit_temperature(head, val, mean, std):
    """Single-parameter temperature scaling on the validation split."""
    x = (torch.from_numpy(val[0]).float() - mean) / std
    y = torch.from_numpy(val[1]).long()
    with torch.no_grad():
        logits = head(x)

    log_t = torch.zeros(1, requires_grad=True)
    optim = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60)
    nll = nn.CrossEntropyLoss()

    def closure():
        optim.zero_grad()
        loss = nll(logits / log_t.exp(), y)
        loss.backward()
        return loss

    optim.step(closure)
    t = float(log_t.exp().item())
    print(f"  calibrated temperature: {t:.4f}")
    return max(0.05, min(t, 10.0))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=DATA_DIR / "ham10000")
    p.add_argument("--out", type=Path, default=MODELS_DIR / "best_model.pth")
    p.add_argument("--cache", type=Path, default=DOCS_DIR / "features_b3.npz")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Forward-only, so this stays small and memory-safe.")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default=SETTINGS.device)
    p.add_argument("--image-wise-split", action="store_true",
                   help="Reproduce the leaky split for comparison only.")
    args = p.parse_args()

    device = resolve_device(args.device)
    print(f"\nFitting classifier head on frozen EfficientNet-B3 features")
    print(f"  device: {device}   (backbone frozen, forward passes only)\n")

    print("1. Metadata and lesion-grouped split")
    frame = load_metadata(args.data_root)
    splits = make_splits(frame, group_by_lesion=not args.image_wise_split)
    print(f"  train {len(splits.train)}  val {len(splits.val)}  test {len(splits.test)}"
          f"   grouped_by_lesion={not args.image_wise_split}")

    print("\n2. Feature extraction (frozen backbone, no gradients)")
    model = build_model(pretrained=True).to(device).eval()
    transform = build_eval_transform()
    feats = cached_features(model, splits, transform, device, args.cache,
                            batch_size=args.batch_size, workers=args.workers)

    print("\n3. Head fitting on cached features")
    head, mean, std, val_f1 = fit_head(
        feats["train"], feats["val"], epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, device=device)

    print("\n4. Temperature scaling")
    temperature = fit_temperature(head, feats["val"], mean, std)

    print("\n5. Assembling a full checkpoint")
    # Fold the feature standardisation into the linear layer so the saved
    # weights slot straight into the unmodified architecture:
    #   W'x + b'  where  W' = W/std  and  b' = b - (W/std)·mean
    W = head.weight.detach() / std
    b = head.bias.detach() - (W @ mean.squeeze(0))

    full = build_model(pretrained=True)
    with torch.no_grad():
        full.classifier.weight.copy_(W)
        full.classifier.bias.copy_(b)

    # Key names deliberately mirror what derm.train writes, so both paths produce
    # interchangeable checkpoints and verify_checkpoint.py can read either.
    payload = {
        "state_dict": full.state_dict(),
        "class_codes": list(CLASS_CODES),
        "architecture": SETTINGS.model.architecture,
        "image_size": SETTINGS.model.image_size,
        "mean": list(SETTINGS.model.mean),
        "std": list(SETTINGS.model.std),
        "temperature": temperature,
        "training_method": "linear_probe_on_frozen_imagenet_features",
        "grouped_by_lesion": not args.image_wise_split,
        "val_macro_f1": round(float(val_f1), 4),
        "notes": (
            "Classifier head fitted on frozen ImageNet-pretrained EfficientNet-B3 "
            "features (linear probe). The backbone was NOT fine-tuned, so this is "
            "weaker than end-to-end training and should be reported as a linear "
            "probe, not as a fine-tuned model."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    size = args.out.stat().st_size / 1e6
    print(f"  wrote {args.out.relative_to(PROJECT_ROOT)}  ({size:.0f} MB)")

    # Quick honest test-split read-out so the number is visible immediately.
    print("\n6. Held-out test performance (linear probe)")
    from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

    xte = (torch.from_numpy(feats["test"][0]).float() - mean) / std
    yte = feats["test"][1]
    with torch.no_grad():
        pred = head(xte).argmax(1).numpy()

    acc = float((pred == yte).mean())
    bal = float(balanced_accuracy_score(yte, pred))
    macro = float(f1_score(yte, pred, average="macro", zero_division=0))
    per_class = recall_score(yte, pred, average=None, zero_division=0,
                             labels=list(range(len(CLASS_CODES))))
    mel = float(per_class[CLASS_CODES.index("mel")])

    print(f"  accuracy           {acc * 100:5.2f}%")
    print(f"  balanced accuracy  {bal * 100:5.2f}%")
    print(f"  macro F1           {macro:.4f}")
    print(f"  melanoma recall    {mel * 100:5.2f}%")
    print("  per-class recall:  " + ", ".join(
        f"{c}={r * 100:.1f}%" for c, r in zip(CLASS_CODES, per_class)))

    summary = {
        "training_method": "linear_probe_on_frozen_imagenet_features",
        "grouped_by_lesion": not args.image_wise_split,
        "accuracy": round(acc * 100, 2),
        "balanced_accuracy": round(bal * 100, 2),
        "macro_f1": round(macro, 4),
        "melanoma_recall": round(mel * 100, 2),
        "per_class_recall": {c: round(float(r) * 100, 2)
                             for c, r in zip(CLASS_CODES, per_class)},
        "temperature": round(temperature, 4),
        "verified": "measured",
    }
    (DOCS_DIR / "linear_probe_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  summary -> docs/linear_probe_results.json")
    print("\nNext: python scripts/verify_checkpoint.py models/best_model.pth\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
