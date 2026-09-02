"""Training entry point.

    python -m derm.train --epochs 20 --batch-size 32

Improvements over the notebook loop:

* **Group-aware splits** (see :mod:`derm.data`) so the reported test score is not
  inflated by multiple photographs of the same lesion.
* **Checkpoint selection on macro-F1**, not accuracy. On a dataset that is 67%
  ``nv``, accuracy peaks on epochs that quietly ignore the rare classes.
* **Optional focal loss** as an alternative to class weighting for the extreme
  imbalance between ``nv`` (6,705) and ``df`` (115).
* **Mixed precision and gradient clipping** for stability and speed on CUDA.
* **Early stopping** so a 15-epoch run does not have to be babysat.
* **Temperature scaling** fitted on the validation set after training, so served
  confidences are calibrated instead of systematically overconfident.
* **Rich checkpoint metadata** - class order, metrics, calibration temperature -
  which is what lets the serving layer verify it loaded the right thing.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import CLASS_CODES, DOCS_DIR, FIGURES_DIR, MODELS_DIR, SETTINGS, TrainConfig
from .data import class_weights, load_metadata, make_loaders, make_splits
from .model import build_model, build_eval_transform, build_train_transform, resolve_device

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Class-weighted focal loss.

    Down-weights the easy, abundant ``nv`` examples so the gradient budget goes to
    the hard minority classes. ``gamma=2`` is the standard setting.
    """

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else torch.tensor([]))
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.weight if self.weight.numel() else None
        ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def macro_f1(targets: np.ndarray, predictions: np.ndarray, num_classes: int) -> float:
    """Unweighted mean F1 - the metric that actually reflects rare-class skill."""
    scores = []
    for index in range(num_classes):
        tp = float(np.sum((predictions == index) & (targets == index)))
        fp = float(np.sum((predictions == index) & (targets != index)))
        fn = float(np.sum((predictions != index) & (targets == index)))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator > 0 else 0.0)
    return float(np.mean(scores))


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Return loss, accuracy, macro-F1 plus raw logits and targets."""
    model.eval()
    total_loss, seen = 0.0, 0
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * labels.size(0)
        seen += labels.size(0)
        all_logits.append(logits.detach().float().cpu().numpy())
        all_targets.append(labels.detach().cpu().numpy())

    logits_array = np.concatenate(all_logits) if all_logits else np.zeros((0, len(CLASS_CODES)))
    targets_array = np.concatenate(all_targets) if all_targets else np.zeros((0,), dtype=int)
    predictions = logits_array.argmax(axis=1) if len(logits_array) else np.zeros((0,), dtype=int)

    accuracy = float((predictions == targets_array).mean() * 100) if seen else 0.0
    return (
        total_loss / max(seen, 1),
        accuracy,
        macro_f1(targets_array, predictions, len(CLASS_CODES)) if seen else 0.0,
        logits_array,
        targets_array,
    )


def fit_temperature(
    logits: np.ndarray, targets: np.ndarray, *, max_iter: int = 200
) -> float:
    """Fit a single temperature by minimising validation NLL (Guo et al., 2017).

    Values above 1.0 mean the raw model was overconfident, which is the usual
    outcome for a network trained with class weights.
    """
    if len(logits) == 0:
        return 1.0
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = F.cross_entropy(logits_tensor / temperature, targets_tensor)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception as exc:  # noqa: BLE001 - calibration must never break training
        logger.warning("Temperature calibration failed (%s); using T=1.0", exc)
        return 1.0
    return float(np.clip(float(torch.exp(log_temperature).item()), 0.25, 10.0))


def train(
    config: TrainConfig,
    *,
    dataset_root: Path | None = None,
    output_dir: Path | None = None,
    pretrained: bool = True,
    group_by_lesion: bool = True,
    balanced_sampler: bool = False,
    device_preference: str | None = None,
) -> dict:
    """Run training and write ``best_model.pth`` plus a history JSON."""
    output_dir = output_dir or MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = resolve_device(device_preference or SETTINGS.device)
    logger.info("Training on %s", device)

    frame = load_metadata(dataset_root)
    splits = make_splits(
        frame,
        val_size=config.val_size,
        test_size=config.test_size,
        seed=config.seed,
        group_by_lesion=group_by_lesion,
    )
    split_summary = splits.describe()
    logger.info("Split summary: %s", json.dumps(split_summary["train"]))
    if split_summary["train_eval_lesion_overlap"]:
        logger.error(
            "Lesion overlap between train and eval splits: %d",
            split_summary["train_eval_lesion_overlap"],
        )

    train_loader, val_loader, test_loader = make_loaders(
        splits,
        build_train_transform(),
        build_eval_transform(),
        config,
        balanced_sampler=balanced_sampler,
    )

    model = build_model(SETTINGS.model, pretrained=pretrained).to(device)

    weights = torch.tensor(class_weights(splits.train), dtype=torch.float32, device=device)
    if config.use_focal_loss:
        criterion: nn.Module = FocalLoss(weight=weights).to(device)
    else:
        criterion = nn.CrossEntropyLoss(
            weight=weights, label_smoothing=config.label_smoothing
        )
    eval_criterion = nn.CrossEntropyLoss(weight=weights)

    # Separate parameter groups: the randomly initialised head needs a much
    # larger learning rate than the pretrained backbone.
    head_names = {name for name, _ in model.named_parameters() if "classifier" in name}
    head_params = [p for n, p in model.named_parameters() if n in head_names]
    backbone_params = [p for n, p in model.named_parameters() if n not in head_names]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": config.backbone_lr},
            {"params": head_params, "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs), eta_min=1e-6
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict] = []
    best_metric = -np.inf
    best_state: dict | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = output_dir / "best_model.pth"

    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        model.train()
        running_loss, correct, seen = 0.0, 0, 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * labels.size(0)
            correct += int((logits.argmax(1) == labels).sum().item())
            seen += labels.size(0)

        scheduler.step()

        train_loss = running_loss / max(seen, 1)
        train_accuracy = correct / max(seen, 1) * 100
        val_loss, val_accuracy, val_f1, _, _ = evaluate_split(
            model, val_loader, eval_criterion, device
        )

        monitored = val_f1 if config.monitor == "macro_f1" else val_accuracy / 100.0
        improved = monitored > best_metric
        if improved:
            best_metric = monitored
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_accuracy, 2),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_accuracy, 2),
            "val_macro_f1": round(val_f1, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
            "seconds": round(time.perf_counter() - started, 1),
            "best": improved,
        }
        history.append(entry)
        logger.info(
            "Epoch %2d/%d  train %.4f/%.2f%%  val %.4f/%.2f%%  macroF1 %.4f  %s",
            epoch,
            config.epochs,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
            val_f1,
            "*" if improved else "",
        )

        if epochs_without_improvement >= config.early_stopping_patience:
            logger.info("Early stopping after %d epochs without improvement.", epochs_without_improvement)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- calibration on validation, then final test evaluation ------------ #
    _, _, _, val_logits, val_targets = evaluate_split(
        model, val_loader, eval_criterion, device
    )
    temperature = fit_temperature(val_logits, val_targets)
    logger.info("Calibrated temperature: %.4f", temperature)

    test_loss, test_accuracy, test_f1, test_logits, test_targets = evaluate_split(
        model, test_loader, eval_criterion, device
    )
    logger.info(
        "TEST  loss %.4f  accuracy %.2f%%  macro-F1 %.4f", test_loss, test_accuracy, test_f1
    )

    metadata = {
        "state_dict": model.state_dict(),
        "class_codes": list(CLASS_CODES),
        "architecture": SETTINGS.model.architecture,
        "image_size": SETTINGS.model.image_size,
        "mean": list(SETTINGS.model.mean),
        "std": list(SETTINGS.model.std),
        "temperature": temperature,
        "epoch": best_epoch,
        "test_accuracy": round(test_accuracy, 2),
        "test_macro_f1": round(test_f1, 4),
        "macro_f1": round(test_f1, 4),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_config": asdict(config),
        "splits": split_summary,
        "grouped_by_lesion": group_by_lesion,
    }
    torch.save(metadata, checkpoint_path)
    logger.info("Saved checkpoint to %s", checkpoint_path)

    summary = {
        key: value for key, value in metadata.items() if key != "state_dict"
    }
    summary["history"] = history
    summary["test_loss"] = round(test_loss, 4)

    history_path = DOCS_DIR / "training_history.json"
    history_path.write_text(json.dumps(summary, indent=2))

    np.savez_compressed(
        DOCS_DIR / "test_logits.npz", logits=test_logits, targets=test_targets
    )
    _plot_curves(history, FIGURES_DIR / "training_curves.png")
    return summary


def _plot_curves(history: list[dict], path: Path) -> None:
    """Write loss / accuracy / macro-F1 curves."""
    if not history:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return

    epochs = [entry["epoch"] for entry in history]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, [e["train_loss"] for e in history], "o-", label="train")
    axes[0].plot(epochs, [e["val_loss"] for e in history], "o-", label="validation")
    axes[0].set_title("Loss", fontweight="bold")

    axes[1].plot(epochs, [e["train_accuracy"] for e in history], "o-", label="train")
    axes[1].plot(epochs, [e["val_accuracy"] for e in history], "o-", label="validation")
    axes[1].set_title("Accuracy (%)", fontweight="bold")

    axes[2].plot(epochs, [e["val_macro_f1"] for e in history], "o-", color="#8e44ad")
    axes[2].set_title("Validation macro-F1", fontweight="bold")

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
        if axis.get_legend_handles_labels()[0]:
            axis.legend()

    figure.suptitle(
        f"{SETTINGS.model.architecture} training", fontsize=14, fontweight="bold"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the dermoscopic lesion classifier on HAM10000.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    defaults = TrainConfig()
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Directory containing HAM10000_metadata.csv")
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--backbone-lr", type=float, default=defaults.backbone_lr)
    parser.add_argument("--head-lr", type=float, default=defaults.head_lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--patience", type=int, default=defaults.early_stopping_patience)
    parser.add_argument("--monitor", choices=["macro_f1", "accuracy"], default=defaults.monitor)
    parser.add_argument("--focal-loss", action="store_true",
                        help="Use focal loss instead of weighted cross-entropy")
    parser.add_argument("--balanced-sampler", action="store_true",
                        help="Oversample minority classes instead of weighting the loss")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Train from random init (requires no network access)")
    parser.add_argument("--image-wise-split", action="store_true",
                        help="Reproduce the original leaky split for comparison")
    parser.add_argument("--device", default=None, help="cuda | cpu | mps | auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )
    args = build_parser().parse_args(argv)

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        early_stopping_patience=args.patience,
        monitor=args.monitor,
        use_focal_loss=args.focal_loss,
    )

    if args.image_wise_split:
        logger.warning(
            "Using an image-wise split. Multiple photographs of the same lesion "
            "will span train and test, so the reported test score will be "
            "optimistically biased."
        )

    summary = train(
        config,
        dataset_root=args.data_root,
        output_dir=args.output_dir,
        pretrained=not args.no_pretrained,
        group_by_lesion=not args.image_wise_split,
        balanced_sampler=args.balanced_sampler,
        device_preference=args.device,
    )
    print(
        f"\nDone. Best epoch {summary['epoch']}, "
        f"test accuracy {summary['test_accuracy']}%, "
        f"test macro-F1 {summary['test_macro_f1']}."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
