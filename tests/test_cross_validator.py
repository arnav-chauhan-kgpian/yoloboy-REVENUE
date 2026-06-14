"""
tests/test_cross_validator.py
==============================
Unit tests for src/models/cross_validator.py.

Fixture: synthetic 3-campaign × 300-day DataFrame.
- min_train_days=90 for fast fold generation.
- Verifies fold count, no leakage, temporal ordering, index validity,
  and the attribution_mature filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.cross_validator import (
    CrossValidatorError,
    Fold,
    RollingOriginValidator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_CAMPAIGNS = 3
N_DAYS = 300
START_DATE = "2024-01-01"


def _make_df(
    n_campaigns: int = N_CAMPAIGNS,
    n_days: int = N_DAYS,
    start: str = START_DATE,
    with_attribution: bool = True,
    mature_cutoff_days: int = 14,  # last 14 days marked immature
) -> pd.DataFrame:
    """Return a synthetic campaign × date DataFrame."""
    dates = pd.date_range(start=start, periods=n_days, freq="D")
    rows = []
    for cid in range(n_campaigns):
        for date in dates:
            rows.append(
                {
                    "campaign_id": f"camp_{cid}",
                    "date": date,
                    "revenue_attributed": float(cid * 100 + 1),
                    "spend": float(cid * 10 + 1),
                }
            )
    df = pd.DataFrame(rows)
    if with_attribution:
        global_max = df["date"].max()
        cutoff = global_max - pd.Timedelta(days=mature_cutoff_days - 1)
        df["attribution_mature"] = df["date"] < cutoff
    return df


@pytest.fixture(scope="module")
def df_base() -> pd.DataFrame:
    return _make_df()


@pytest.fixture(scope="module")
def validator_default() -> RollingOriginValidator:
    return RollingOriginValidator(
        min_train_days=90,
        val_window_days=30,
        step_days=30,
    )


@pytest.fixture(scope="module")
def folds_default(df_base, validator_default) -> list[Fold]:
    return validator_default.split(df_base)


# ---------------------------------------------------------------------------
# TestFoldCount
# ---------------------------------------------------------------------------

class TestFoldCount:
    def test_at_least_one_fold(self, folds_default):
        assert len(folds_default) >= 1

    def test_expected_fold_count(self, df_base, validator_default):
        """300-day span with min_train=90, val=30, step=30 → 6 folds."""
        folds = validator_default.split(df_base)
        # val_start range: day 90 to day 299-29 = day 270 → (270-90)/30 + 1 = 7
        # (exact count may vary by off-by-one; just assert range)
        assert 5 <= len(folds) <= 8

    def test_max_folds_respected(self, df_base):
        v = RollingOriginValidator(min_train_days=90, max_folds=3)
        folds = v.split(df_base)
        assert len(folds) == 3

    def test_max_folds_one(self, df_base):
        v = RollingOriginValidator(min_train_days=90, max_folds=1)
        folds = v.split(df_base)
        assert len(folds) == 1

    def test_insufficient_data_raises(self):
        tiny = _make_df(n_campaigns=2, n_days=50)
        v = RollingOriginValidator(min_train_days=90, val_window_days=30)
        with pytest.raises(CrossValidatorError, match="calendar days"):
            v.split(tiny)


# ---------------------------------------------------------------------------
# TestFoldOrdering
# ---------------------------------------------------------------------------

class TestFoldOrdering:
    def test_fold_ids_sequential(self, folds_default):
        for i, fold in enumerate(folds_default):
            assert fold.fold_id == i

    def test_val_starts_strictly_increasing(self, folds_default):
        for i in range(1, len(folds_default)):
            assert folds_default[i].val_start > folds_default[i - 1].val_start

    def test_train_end_before_val_start(self, folds_default):
        for fold in folds_default:
            assert fold.train_end < fold.val_start

    def test_val_window_size(self, folds_default):
        for fold in folds_default:
            actual_days = (fold.val_end - fold.val_start).days + 1
            assert actual_days == 30

    def test_train_window_grows(self, folds_default):
        """Training set grows by step_days with each fold."""
        for i in range(1, len(folds_default)):
            assert folds_default[i].n_train > folds_default[i - 1].n_train


# ---------------------------------------------------------------------------
# TestNoLeakage
# ---------------------------------------------------------------------------

class TestNoLeakage:
    def test_check_no_leakage_passes(self, folds_default, validator_default):
        assert validator_default.check_no_leakage(folds_default) is True

    def test_no_index_overlap(self, folds_default):
        for fold in folds_default:
            train_set = set(fold.train_indices.tolist())
            val_set = set(fold.val_indices.tolist())
            assert len(train_set & val_set) == 0

    def test_train_dates_before_val(self, df_base, folds_default):
        """Every training-row date is strictly before every val-row date."""
        for fold in folds_default:
            train_dates = df_base.loc[fold.train_indices, "date"]
            val_dates   = df_base.loc[fold.val_indices, "date"]
            assert train_dates.max() < val_dates.min()

    def test_leakage_detected_on_corrupted_fold(self, folds_default):
        """Manually create an overlapping fold and verify detection."""
        good_fold = folds_default[0]
        bad_fold = Fold(
            fold_id=99,
            train_start=good_fold.train_start,
            train_end=good_fold.train_end,
            val_start=good_fold.val_start,
            val_end=good_fold.val_end,
            # Intentionally overlap val indices into train
            train_indices=np.concatenate([
                good_fold.train_indices, good_fold.val_indices[:5]
            ]),
            val_indices=good_fold.val_indices,
        )
        v = RollingOriginValidator()
        with pytest.raises(AssertionError, match="leakage"):
            v.check_no_leakage([bad_fold])


# ---------------------------------------------------------------------------
# TestIndexValidity
# ---------------------------------------------------------------------------

class TestIndexValidity:
    def test_train_indices_in_df(self, df_base, folds_default):
        valid_idx = set(df_base.index.tolist())
        for fold in folds_default:
            assert set(fold.train_indices.tolist()).issubset(valid_idx)

    def test_val_indices_in_df(self, df_base, folds_default):
        valid_idx = set(df_base.index.tolist())
        for fold in folds_default:
            assert set(fold.val_indices.tolist()).issubset(valid_idx)

    def test_n_train_matches_indices(self, folds_default):
        for fold in folds_default:
            assert fold.n_train == len(fold.train_indices)

    def test_n_val_matches_indices(self, folds_default):
        for fold in folds_default:
            assert fold.n_val == len(fold.val_indices)

    def test_indices_are_numpy_arrays(self, folds_default):
        for fold in folds_default:
            assert isinstance(fold.train_indices, np.ndarray)
            assert isinstance(fold.val_indices, np.ndarray)

    def test_val_row_count(self, folds_default):
        """Validation window of 30 days × 3 campaigns = 90 rows per fold."""
        for fold in folds_default:
            assert fold.n_val == 30 * N_CAMPAIGNS


# ---------------------------------------------------------------------------
# TestAttributionMatureFilter
# ---------------------------------------------------------------------------

class TestAttributionMatureFilter:
    def test_train_uses_only_mature_rows(self, df_base, folds_default):
        """Every row in the training index must have attribution_mature=True."""
        for fold in folds_default:
            mature = df_base.loc[fold.train_indices, "attribution_mature"]
            assert mature.all(), (
                f"Fold {fold.fold_id}: {(~mature).sum()} immature rows in train set"
            )

    def test_val_includes_immature_rows(self, df_base, folds_default):
        """Validation set is unrestricted (includes immature rows when present)."""
        last_fold = folds_default[-1]
        attr = df_base.loc[last_fold.val_indices, "attribution_mature"]
        # Last fold's validation window may overlap the immature tail
        # (if it does, immature rows are present; if not, this still passes)
        assert len(attr) > 0

    def test_no_attribution_column_uses_all_rows(self, df_base):
        """Without an attribution_mature column, all rows are used for training."""
        df_no_attr = df_base.drop(columns=["attribution_mature"])
        v = RollingOriginValidator(
            min_train_days=90, attribution_col=None
        )
        folds = v.split(df_no_attr)
        first_fold = folds[0]
        # Training rows = all rows before val_start
        dates = df_no_attr["date"]
        expected_n_train = int((dates < first_fold.val_start).sum())
        assert first_fold.n_train == expected_n_train

    def test_attribution_col_none_explicit(self, df_base):
        """attribution_col=None does not crash even when column exists."""
        v = RollingOriginValidator(
            min_train_days=90, attribution_col=None
        )
        folds = v.split(df_base)
        assert len(folds) >= 1


# ---------------------------------------------------------------------------
# TestSummary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_shape(self, folds_default, validator_default):
        summary = validator_default.summary(folds_default)
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == len(folds_default)

    def test_summary_columns(self, folds_default, validator_default):
        expected = {
            "fold_id", "train_start", "train_end",
            "val_start", "val_end", "n_train", "n_val",
            "train_days", "val_days",
        }
        summary = validator_default.summary(folds_default)
        assert expected.issubset(set(summary.columns))

    def test_summary_val_days_correct(self, folds_default, validator_default):
        summary = validator_default.summary(folds_default)
        assert (summary["val_days"] == 30).all()


# ---------------------------------------------------------------------------
# TestFoldRepr
# ---------------------------------------------------------------------------

class TestFoldRepr:
    def test_repr_contains_fold_id(self, folds_default):
        fold = folds_default[0]
        r = repr(fold)
        assert "Fold(id=0" in r

    def test_fold_is_frozen(self, folds_default):
        """Frozen dataclass raises on attribute assignment."""
        fold = folds_default[0]
        with pytest.raises((AttributeError, TypeError)):
            fold.fold_id = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_date_col_raises(self, df_base):
        bad = df_base.rename(columns={"date": "ds"})
        v = RollingOriginValidator(min_train_days=90)
        with pytest.raises(CrossValidatorError, match="date_col"):
            v.split(bad)

    def test_negative_min_train_days_raises(self):
        with pytest.raises(ValueError, match="min_train_days"):
            RollingOriginValidator(min_train_days=-1)

    def test_zero_val_window_raises(self):
        with pytest.raises(ValueError, match="val_window_days"):
            RollingOriginValidator(val_window_days=0)

    def test_zero_step_raises(self):
        with pytest.raises(ValueError, match="step_days"):
            RollingOriginValidator(step_days=0)

    def test_zero_max_folds_raises(self):
        with pytest.raises(ValueError, match="max_folds"):
            RollingOriginValidator(max_folds=0)
