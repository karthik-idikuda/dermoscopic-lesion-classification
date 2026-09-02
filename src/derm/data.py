"""HAM10000 dataset discovery, splitting and loading.

One correctness issue is fixed here relative to the original notebooks, and it
materially affects the reported numbers.

HAM10000 contains 10,015 images but only 7,470 distinct lesions: many lesions
were photographed several times, sometimes at different magnifications. Splitting
on *images* therefore puts near-duplicate photographs of the same physical lesion
into both the training and the test set, and the model can score well by
recognising the individual lesion rather than the diagnosis. This is textbook
data leakage and it inflates test accuracy.

:func:`make_splits` defaults to grouping on ``lesion_id`` so that every image of a
given lesion lands in exactly one split. Expect the honest test accuracy to come
out a few points *below* the 80.17% from the image-wise split; that lower number
is the trustworthy one. Pass ``group_by_lesion=False`` to reproduce the original
behaviour for comparison.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import CLASS_CODES, CLASS_INDEX, DATA_DIR, PROJECT_ROOT, TrainConfig

logger = logging.getLogger(__name__)

METADATA_FILENAME = "HAM10000_metadata.csv"

SEARCH_ROOTS = (
    DATA_DIR,
    PROJECT_ROOT / "data",
    Path("/kaggle/input"),
    Path.home() / "Downloads",
    Path.home() / ".cache" / "ham10000",
)


class DatasetNotFoundError(FileNotFoundError):
    """Raised when HAM10000 cannot be located on disk."""


def find_metadata(root: Path | str | None = None) -> Path:
    """Locate ``HAM10000_metadata.csv``.

    Checks ``DERM_HAM10000_DIR`` first, then an explicit ``root``, then a handful
    of conventional locations, recursing a few levels into each.
    """
    candidates: list[Path] = []
    env_root = os.environ.get("DERM_HAM10000_DIR")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    if root:
        candidates.append(Path(root).expanduser())
    candidates.extend(SEARCH_ROOTS)

    for candidate in candidates:
        if not candidate.exists():
            continue
        direct = candidate / METADATA_FILENAME
        if direct.is_file():
            return direct
        try:
            for depth in ("*", "*/*", "*/*/*"):
                for match in candidate.glob(f"{depth}/{METADATA_FILENAME}"):
                    return match
        except OSError:  # permission problems on a scanned directory
            continue

    searched = "\n  ".join(str(c) for c in candidates)
    raise DatasetNotFoundError(
        "Could not find HAM10000_metadata.csv. Download the dataset from "
        "https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 and "
        "either place it under data/ham10000/ or set DERM_HAM10000_DIR.\n"
        f"Searched:\n  {searched}"
    )


def index_images(dataset_root: Path) -> dict[str, Path]:
    """Map ``image_id`` to file path across the ``part_1``/``part_2`` folders."""
    index: dict[str, Path] = {}
    for pattern in ("**/*.jpg", "**/*.jpeg", "**/*.png"):
        for path in dataset_root.glob(pattern):
            index.setdefault(path.stem, path)
    return index


def load_metadata(root: Path | str | None = None) -> pd.DataFrame:
    """Load the metadata CSV, resolve image paths and encode labels."""
    metadata_path = find_metadata(root)
    dataset_root = metadata_path.parent
    frame = pd.read_csv(metadata_path)

    required = {"image_id", "dx", "lesion_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{metadata_path} is missing columns: {sorted(missing)}")

    index = index_images(dataset_root)
    if not index:
        raise DatasetNotFoundError(
            f"Found {metadata_path} but no image files beneath {dataset_root}."
        )

    frame["image_path"] = frame["image_id"].map(lambda i: index.get(i))
    unresolved = int(frame["image_path"].isna().sum())
    if unresolved:
        logger.warning("Dropping %d row(s) with no matching image file.", unresolved)
        frame = frame.dropna(subset=["image_path"]).reset_index(drop=True)

    unknown = set(frame["dx"].unique()) - set(CLASS_CODES)
    if unknown:
        raise ValueError(f"Unexpected diagnosis codes in metadata: {sorted(unknown)}")

    frame["label"] = frame["dx"].map(CLASS_INDEX).astype(int)
    frame["image_path"] = frame["image_path"].map(str)
    return frame


@dataclass
class Splits:
    """Train / validation / test frames plus the leakage-relevant metadata."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    grouped_by_lesion: bool

    def describe(self) -> dict:
        def summary(frame: pd.DataFrame) -> dict:
            return {
                "images": int(len(frame)),
                "lesions": int(frame["lesion_id"].nunique()),
                "class_counts": {
                    code: int((frame["dx"] == code).sum()) for code in CLASS_CODES
                },
            }

        overlap = 0
        if self.grouped_by_lesion:
            train_lesions = set(self.train["lesion_id"])
            overlap = len(
                train_lesions
                & (set(self.val["lesion_id"]) | set(self.test["lesion_id"]))
            )
        return {
            "grouped_by_lesion": self.grouped_by_lesion,
            "train": summary(self.train),
            "val": summary(self.val),
            "test": summary(self.test),
            "train_eval_lesion_overlap": overlap,
        }


def make_splits(
    frame: pd.DataFrame,
    *,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    group_by_lesion: bool = True,
) -> Splits:
    """Split into train/val/test, grouping by lesion to prevent leakage.

    Group-aware splitting cannot perfectly preserve class proportions, so the
    grouped path stratifies *within each class* by assigning whole lesions of that
    class to splits. That keeps the class balance close to the original while
    still guaranteeing zero lesion overlap between splits.
    """
    if not group_by_lesion:
        from sklearn.model_selection import train_test_split

        train, temp = train_test_split(
            frame,
            test_size=val_size + test_size,
            random_state=seed,
            stratify=frame["label"],
        )
        relative = test_size / (val_size + test_size)
        val, test = train_test_split(
            temp, test_size=relative, random_state=seed, stratify=temp["label"]
        )
        return Splits(
            train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True),
            grouped_by_lesion=False,
        )

    rng = np.random.default_rng(seed)
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    # One lesion can only carry one diagnosis in HAM10000, so grouping by
    # (dx, lesion_id) is unambiguous.
    for _, group in frame.groupby("dx", sort=True):
        lesions = group["lesion_id"].unique()
        rng.shuffle(lesions)
        n = len(lesions)
        n_test = max(1, int(round(n * test_size))) if n > 2 else 0
        n_val = max(1, int(round(n * val_size))) if n > 2 else 0
        if n_test + n_val >= n:  # tiny class: keep at least one lesion for training
            n_test = min(n_test, max(0, n - 2))
            n_val = min(n_val, max(0, n - 1 - n_test))
        test_ids.extend(lesions[:n_test])
        val_ids.extend(lesions[n_test : n_test + n_val])
        train_ids.extend(lesions[n_test + n_val :])

    def subset(ids: list[str]) -> pd.DataFrame:
        return frame[frame["lesion_id"].isin(set(ids))].reset_index(drop=True)

    return Splits(subset(train_ids), subset(val_ids), subset(test_ids), True)


class SkinLesionDataset(Dataset):
    """Image/label dataset reading directly from disk.

    Unreadable files raise a clear error rather than returning a silently blank
    tensor, because a corrupt file that trains as a black image is very hard to
    notice later.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        transform=None,
        *,
        return_path: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = row["image_path"]
        try:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to read image {path}: {exc}") from exc

        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label"])
        if self.return_path:
            return image, label, str(path)
        return image, label


def class_weights(frame: pd.DataFrame) -> np.ndarray:
    """Inverse-frequency weights aligned to :data:`CLASS_CODES` order."""
    counts = np.array(
        [max(1, int((frame["label"] == index).sum())) for index in range(len(CLASS_CODES))],
        dtype=np.float64,
    )
    weights = counts.sum() / (len(counts) * counts)
    return weights


def make_sampler(frame: pd.DataFrame) -> WeightedRandomSampler:
    """Balanced sampler, an alternative to weighting the loss.

    Oversampling the 115-image ``df`` class up to ``nv`` levels risks
    memorisation, so this is offered as an option rather than the default; the
    default path weights the loss instead.
    """
    weights = class_weights(frame)
    sample_weights = weights[frame["label"].to_numpy()]
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(frame),
        replacement=True,
    )


def make_loaders(
    splits: Splits,
    train_transform,
    eval_transform,
    config: TrainConfig | None = None,
    *,
    balanced_sampler: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build the three dataloaders."""
    config = config or TrainConfig()
    train_dataset = SkinLesionDataset(splits.train, train_transform)
    val_dataset = SkinLesionDataset(splits.val, eval_transform)
    test_dataset = SkinLesionDataset(splits.test, eval_transform)

    sampler = make_sampler(splits.train) if balanced_sampler else None
    common = {
        "num_workers": config.num_workers,
        "pin_memory": False,
        "persistent_workers": config.num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=len(train_dataset) > config.batch_size,
        **common,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, **common
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, **common
    )
    return train_loader, val_loader, test_loader


__all__ = [
    "DatasetNotFoundError",
    "METADATA_FILENAME",
    "SkinLesionDataset",
    "Splits",
    "class_weights",
    "find_metadata",
    "index_images",
    "load_metadata",
    "make_loaders",
    "make_sampler",
    "make_splits",
]
