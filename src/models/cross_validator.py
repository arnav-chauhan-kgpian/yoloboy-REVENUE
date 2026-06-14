"""
src/models/cross_validator.py
==============================
Rolling-origin (walk-forward) temporal cross-validator for campaign revenue
forecasting.

Design invariants
-----------------
- Training set: all rows whose date is strictly before the validation window.
  Restricted to ``attribution_mature=True`` rows so immature targets do not
  contaminate training loss.
- Validation set: all rows within the 30-day validation window (regardless of
  maturity — the trainer filters for mature rows before computing metrics).
- No row appears in both train and validation index sets.
- Folds are strictly ordered; each fold's val_start > previous fold's val_start.
- No gap is inserted between train_end and val_start; the attribution_mature
  filter serves as the guard against near-term data leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

MIN_TRAIN_DAYS_DEFAULT: Final[int] = 180
VAL_WINDOW_DAYS_DEFAULT: Final[int] = 30
STEP_DAYS_DEFAULT: Final[int] = 30


# ---------------------------------------------------------------------------
# Fold container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fold:
    """Immutable descriptor for one rolling-origin fold.

    Parameters
    ----------
    fold_id : int
        Sequential 0-based fold identifier.
    train_start, train_end : pd.Timestamp
        Inclusive date bounds of the training window.
    val_start, val_end : pd.Timestamp
        Inclusive date bounds of the validation window.
    train_indices : np.ndarray
        Integer label indices into the source DataFrame for training rows.
    val_indices : np.ndarray
        Integer label indices into the source DataFrame for validation rows.
    """

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    train_indices: np.ndarray
    val_indices: np.ndarray

    @property
    def n_train(self) -> int:
        return len(self.train_indices)

    @property
    def n_val(self) -> int:
        return len(self.val_indices)

    def __repr__(self) -> str:
        return (
            f"Fold(id={self.fold_id}, "
            f"train={self.train_start.date()}→{self.train_end.date()} "
            f"[{self.n_train} rows], "
            f"val={self.val_start.date()}→{self.val_end.date()} "
            f"[{self.n_val} rows])"
        )


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class CrossValidatorError(Exception):
    pass


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class RollingOriginValidator:
    """Walk-forward temporal cross-validator.

    Generates folds where the training window grows by *step_days* on each
    iteration and the validation window slides forward by the same amount.
    Training data never contains any date that falls within a validation
    window.

    Parameters
    ----------
    min_train_days : int
        Minimum calendar days required in the training window before the
        first fold is generated.  Default: 180 (≈ 6 months).
    val_window_days : int
        Number of calendar days in each validation window.  Default: 30.
    step_days : int
        Days to advance between consecutive folds.  Default: 30.
    max_folds : int | None
        Cap on the number of folds generated.  ``None`` = unlimited.
    date_col : str
        Name of the date column.  Default: ``"date"``.
    attribution_col : str | None
        If provided, training indices are restricted to rows where this
        column is True (excludes attribution-immature targets from
        training).  Default: ``"attribution_mature"``.
    """

    def __init__(
        self,
        min_train_days: int = MIN_TRAIN_DAYS_DEFAULT,
        val_window_days: int = VAL_WINDOW_DAYS_DEFAULT,
        step_days: int = STEP_DAYS_DEFAULT,
        max_folds: int | None = None,
        date_col: str = "date",
        attribution_col: str | None = "attribution_mature",
    ) -> None:
        if min_train_days <= 0:
            raise ValueError(f"min_train_days must be positive, got {min_train_days}")
        if val_window_days <= 0:
            raise ValueError(f"val_window_days must be positive, got {val_window_days}")
        if step_days <= 0:
            raise ValueError(f"step_days must be positive, got {step_days}")
        if max_folds is not None and max_folds <= 0:
            raise ValueError(f"max_folds must be positive or None, got {max_folds}")

        self.min_train_days = min_train_days
        self.val_window_days = val_window_days
        self.step_days = step_days
        self.max_folds = max_folds
        self.date_col = date_col
        self.attribution_col = attribution_col

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, df: pd.DataFrame) -> list[Fold]:
        """Generate rolling-origin folds for *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Feature store DataFrame.  Must be sorted by ``(campaign_id, date)``
            (the output of :func:`~src.features.feature_store.build_feature_store`
            satisfies this).

        Returns
        -------
        list[Fold]
            Chronologically ordered list of :class:`Fold` objects.

        Raises
        ------
        CrossValidatorError
            If *df* lacks the date column or has insufficient date coverage.
        """
        if self.date_col not in df.columns:
            raise CrossValidatorError(
                f"date_col '{self.date_col}' not found in DataFrame columns"
            )

        dates = pd.to_datetime(df[self.date_col])
        global_min: pd.Timestamp = dates.min()
        global_max: pd.Timestamp = dates.max()

        total_days = (global_max - global_min).days + 1
        required = self.min_train_days + self.val_window_days
        if total_days < required:
            raise CrossValidatorError(
                f"Dataset spans {total_days} calendar days but "
                f"min_train_days + val_window_days = {required}. "
                "Reduce min_train_days or val_window_days."
            )

        # Attribution maturity mask for training rows
        has_attr = (
            self.attribution_col is not None
            and self.attribution_col in df.columns
        )
        if has_attr:
            attr_mask = df[self.attribution_col].fillna(False).astype(bool)
        else:
            attr_mask = pd.Series(True, index=df.index)

        folds: list[Fold] = []
        val_start = global_min + pd.Timedelta(days=self.min_train_days)

        while True:
            val_end = val_start + pd.Timedelta(days=self.val_window_days - 1)

            if val_end > global_max:
                break

            train_end = val_start - pd.Timedelta(days=1)
            train_start = global_min

            train_mask = (dates <= train_end) & attr_mask
            val_mask = (dates >= val_start) & (dates <= val_end)

            train_idx = df.index[train_mask].to_numpy()
            val_idx = df.index[val_mask].to_numpy()

            if len(train_idx) == 0 or len(val_idx) == 0:
                logger.debug(
                    "Fold skipped at val_start=%s: empty train=%d val=%d",
                    val_start.date(),
                    len(train_idx),
                    len(val_idx),
                )
            else:
                folds.append(
                    Fold(
                        fold_id=len(folds),
                        train_start=train_start,
                        train_end=train_end,
                        val_start=val_start,
                        val_end=val_end,
                        train_indices=train_idx,
                        val_indices=val_idx,
                    )
                )
                if self.max_folds is not None and len(folds) >= self.max_folds:
                    break

            val_start += pd.Timedelta(days=self.step_days)

        if not folds:
            raise CrossValidatorError(
                "No valid folds could be generated. "
                "Check min_train_days and the dataset date coverage."
            )

        logger.info(
            "Generated %d folds | min_train=%d d | val=%d d | step=%d d",
            len(folds),
            self.min_train_days,
            self.val_window_days,
            self.step_days,
        )
        return folds

    def summary(self, folds: list[Fold]) -> pd.DataFrame:
        """Return a DataFrame of fold boundaries and row counts.

        Parameters
        ----------
        folds : list[Fold]
            Output of :meth:`split`.

        Returns
        -------
        pd.DataFrame
            One row per fold with columns:
            fold_id, train_start, train_end, val_start, val_end,
            n_train, n_val, train_days, val_days.
        """
        rows = [
            {
                "fold_id":    f.fold_id,
                "train_start": f.train_start.date(),
                "train_end":  f.train_end.date(),
                "val_start":  f.val_start.date(),
                "val_end":    f.val_end.date(),
                "n_train":    f.n_train,
                "n_val":      f.n_val,
                "train_days": (f.train_end - f.train_start).days + 1,
                "val_days":   (f.val_end - f.val_start).days + 1,
            }
            for f in folds
        ]
        return pd.DataFrame(rows)

    def check_no_leakage(self, folds: list[Fold]) -> bool:
        """Assert that no train index appears in a val set in any fold.

        Also asserts that train dates are strictly before val dates in each fold.

        Returns
        -------
        bool
            True if all invariants hold.

        Raises
        ------
        AssertionError
            On any leakage violation.
        """
        for fold in folds:
            train_set = set(fold.train_indices.tolist())
            val_set = set(fold.val_indices.tolist())
            overlap = train_set & val_set
            assert not overlap, (
                f"Fold {fold.fold_id}: {len(overlap)} indices appear in both "
                "train and val sets — data leakage detected."
            )
            assert fold.train_end < fold.val_start, (
                f"Fold {fold.fold_id}: train_end ({fold.train_end.date()}) "
                f">= val_start ({fold.val_start.date()}) — temporal leakage."
            )
        return True
