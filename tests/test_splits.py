"""Split-integrity tests against the real HAM10000 metadata.

These run on the authoritative metadata CSV (700 KB, no images needed). If it is
not present the module skips, so the suite still works on a clean checkout.

    python scripts/prepare_data.py   # fetches the metadata

The zero-leakage property of the grouped split is the single most important
correctness invariant in the training code: if it regresses, every reported
metric silently becomes optimistic.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from derm.config import CLASS_CODES, PROJECT_ROOT
from derm.data import make_splits

METADATA = PROJECT_ROOT / "data" / "ham10000" / "HAM10000_metadata.csv"

pytestmark = pytest.mark.skipif(
    not METADATA.exists(),
    reason="HAM10000 metadata not present; run scripts/prepare_data.py",
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    data = pd.read_csv(METADATA)
    data["label"] = data["dx"].map({c: i for i, c in enumerate(CLASS_CODES)}).astype(int)
    return data


class TestMetadataIntegrity:
    def test_expected_dataset_size(self, frame):
        assert len(frame) == 10015
        assert frame["lesion_id"].nunique() == 7470

    def test_all_classes_present_with_known_counts(self, frame):
        # Counts from the published dataset; these also validate config.py.
        expected = {
            "nv": 6705, "mel": 1113, "bkl": 1099, "bcc": 514,
            "akiec": 327, "vasc": 142, "df": 115,
        }
        actual = frame["dx"].value_counts().to_dict()
        assert actual == expected

    def test_config_counts_match_metadata(self, frame):
        """LESION_CLASSES.ham10000_count must match the real data."""
        from derm.config import LESION_CLASSES

        actual = frame["dx"].value_counts().to_dict()
        for code, meta in LESION_CLASSES.items():
            assert meta.ham10000_count == actual[code], (
                f"config says {code}={meta.ham10000_count}, metadata says {actual[code]}"
            )

    def test_one_diagnosis_per_lesion(self, frame):
        """Grouping by lesion is only sound if a lesion has a single diagnosis."""
        per_lesion = frame.groupby("lesion_id")["dx"].nunique()
        assert per_lesion.max() == 1

    def test_dataset_has_repeat_images(self, frame):
        """The premise of the leakage argument: many lesions are re-photographed."""
        repeats = len(frame) - frame["lesion_id"].nunique()
        assert repeats == 2545
        assert frame.groupby("lesion_id").size().max() == 6


class TestGroupedSplit:
    @pytest.fixture(scope="class")
    def splits(self, frame):
        return make_splits(frame, seed=42, group_by_lesion=True)

    def test_zero_lesion_overlap(self, splits):
        """The core invariant. A regression here inflates every reported metric."""
        train = set(splits.train["lesion_id"])
        val = set(splits.val["lesion_id"])
        test = set(splits.test["lesion_id"])
        assert train & val == set()
        assert train & test == set()
        assert val & test == set()

    def test_no_image_appears_twice(self, splits):
        ids = pd.concat([splits.train, splits.val, splits.test])["image_id"]
        assert ids.duplicated().sum() == 0

    def test_every_image_is_allocated(self, frame, splits):
        total = len(splits.train) + len(splits.val) + len(splits.test)
        assert total == len(frame)

    def test_all_classes_in_every_split(self, splits):
        for name, part in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
            present = set(part["dx"].unique())
            assert present == set(CLASS_CODES), f"{name} is missing {set(CLASS_CODES) - present}"

    def test_class_balance_is_approximately_preserved(self, frame, splits):
        """Grouping cannot stratify exactly, but it should stay close."""
        for code in CLASS_CODES:
            overall = (frame["dx"] == code).mean()
            for part in (splits.train, splits.test):
                share = (part["dx"] == code).mean()
                assert abs(share - overall) < 0.02, (
                    f"{code}: overall {overall:.4f} vs split {share:.4f}"
                )

    def test_split_proportions_are_close_to_target(self, frame, splits):
        assert 0.65 <= len(splits.train) / len(frame) <= 0.75
        assert 0.10 <= len(splits.val) / len(frame) <= 0.20
        assert 0.10 <= len(splits.test) / len(frame) <= 0.20

    def test_is_deterministic(self, frame):
        a = make_splits(frame, seed=42, group_by_lesion=True)
        b = make_splits(frame, seed=42, group_by_lesion=True)
        assert list(a.test["image_id"]) == list(b.test["image_id"])

    def test_different_seed_gives_different_split(self, frame):
        a = make_splits(frame, seed=42, group_by_lesion=True)
        b = make_splits(frame, seed=7, group_by_lesion=True)
        assert list(a.test["image_id"]) != list(b.test["image_id"])

    def test_describe_reports_zero_overlap(self, splits):
        assert splits.describe()["train_eval_lesion_overlap"] == 0
        assert splits.describe()["grouped_by_lesion"] is True


class TestImageWiseSplitLeaks:
    """Documents the flaw in the original notebook split, so it stays visible."""

    @pytest.fixture(scope="class")
    def splits(self, frame):
        return make_splits(frame, seed=42, group_by_lesion=False)

    def test_image_wise_split_does_leak(self, splits):
        train = set(splits.train["lesion_id"])
        leaked = splits.test["lesion_id"].isin(train)
        # Measured at 36.13% for seed 42; assert it is substantial, not exact,
        # so a scikit-learn RNG change does not break the suite.
        assert leaked.mean() > 0.25, (
            "expected substantial leakage from an image-wise split; "
            f"got {leaked.mean() * 100:.1f}%"
        )

    def test_leakage_is_worse_for_melanoma_than_nevi(self, frame, splits):
        """Leakage concentrates in the classes that matter most clinically."""
        train = set(splits.train["lesion_id"])
        test = splits.test

        def rate(code: str) -> float:
            subset = test[test["dx"] == code]
            return float(subset["lesion_id"].isin(train).mean())

        assert rate("mel") > rate("nv"), (
            f"melanoma leakage {rate('mel'):.3f} should exceed nevus leakage {rate('nv'):.3f}"
        )

    def test_grouped_split_fixes_it(self, frame):
        grouped = make_splits(frame, seed=42, group_by_lesion=True)
        train = set(grouped.train["lesion_id"])
        assert grouped.test["lesion_id"].isin(train).sum() == 0
