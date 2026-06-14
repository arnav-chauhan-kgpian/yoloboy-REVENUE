"""
src/features/rolling_features.py
==================================
Campaign-level rolling window and momentum feature generator.

All windows are strictly historical: the current day is NEVER included.
Implementation uses shift(1) before each rolling aggregation to guarantee
no data leakage.

Input: DataFrame that already contains lag features from lag_features.py
       (momentum features depend on {prefix}_lag_1 columns).
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rolling specification
# ---------------------------------------------------------------------------

# source column → {aggregation: (window1, window2, ...)}
ROLLING_SPEC: Final[dict[str, dict[str, tuple[int, ...]]]] = {
    "revenue_attributed": {
        "mean": (7, 14, 28),
        "std":  (7, 14, 28),
        "min":  (7,),
        "max":  (7,),
    },
    "spend": {
        "mean": (7, 14, 28),
        "std":  (7, 14, 28),
    },
    "clicks": {
        "mean": (7, 14, 28),
    },
    "impressions": {
        "mean": (7, 14, 28),
    },
}

# source column → output column prefix
_PREFIX: Final[dict[str, str]] = {
    "revenue_attributed": "revenue",
    "spend":              "spend",
    "clicks":             "clicks",
    "impressions":        "impressions",
}

# momentum: {prefix: (window1, window2, ...)}
MOMENTUM_SPEC: Final[dict[str, tuple[int, ...]]] = {
    "revenue": (7, 14, 28),
    "spend":   (7, 14, 28),
}

# ---------------------------------------------------------------------------
# Derived column name lists
# ---------------------------------------------------------------------------

def _rolling_col(prefix: str, agg: str, window: int) -> str:
    return f"{prefix}_roll_{agg}_{window}"


ROLLING_COLUMNS: Final[tuple[str, ...]] = tuple(
    _rolling_col(_PREFIX[col], agg, w)
    for col, aggs in ROLLING_SPEC.items()
    for agg, windows in aggs.items()
    for w in windows
)

MOMENTUM_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"{prefix}_momentum_{w}"
    for prefix, windows in MOMENTUM_SPEC.items()
    for w in windows
)

ALL_ROLLING_COLUMNS: Final[tuple[str, ...]] = ROLLING_COLUMNS + MOMENTUM_COLUMNS

# min_periods per aggregation type
_MIN_PERIODS: Final[dict[str, int]] = {
    "mean": 1,
    "std":  2,
    "min":  1,
    "max":  1,
}


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class RollingFeatureError(Exception):
    pass


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame) -> None:
    required_base = {"campaign_id", "date"} | set(ROLLING_SPEC.keys())
    missing_base = required_base - set(df.columns)
    if missing_base:
        raise RollingFeatureError(f"Missing required columns: {sorted(missing_base)}")

    # momentum requires lag_1 columns
    required_lag1 = {f"{prefix}_lag_1" for prefix in MOMENTUM_SPEC}
    missing_lag1 = required_lag1 - set(df.columns)
    if missing_lag1:
        raise RollingFeatureError(
            f"Momentum features require lag_1 columns missing from input: "
            f"{sorted(missing_lag1)}.  Run add_lag_features() first."
        )

    if df.empty:
        raise RollingFeatureError("Input DataFrame is empty")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: pd.DataFrame,
    rolling_spec: dict[str, dict[str, tuple[int, ...]]] | None = None,
    momentum_spec: dict[str, tuple[int, ...]] | None = None,
) -> pd.DataFrame:
    """Add strictly-historical rolling window and momentum features.

    Each rolling feature is computed by first applying ``shift(1)`` to the
    source series (within each campaign), then applying ``rolling(window)``.
    This guarantees the current day's value is never included.

    Momentum formula::

        {prefix}_momentum_{w} = {prefix}_lag_1 / {prefix}_roll_mean_{w}

    Zero denominators are replaced with ``NaN`` to avoid division-by-zero.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`~src.features.lag_features.add_lag_features`.
        Must contain ``campaign_id``, ``date``, all source columns in
        *rolling_spec*, and ``{prefix}_lag_1`` columns for momentum.
    rolling_spec : dict | None
        Custom rolling specification.  Defaults to :data:`ROLLING_SPEC`.
    momentum_spec : dict | None
        Custom momentum specification.  Defaults to :data:`MOMENTUM_SPEC`.

    Returns
    -------
    pd.DataFrame
        Copy of *df* sorted by ``(campaign_id, date)`` with rolling and
        momentum columns appended.

    Raises
    ------
    RollingFeatureError
        If required columns are absent or the DataFrame is empty.
    """
    rspec = ROLLING_SPEC if rolling_spec is None else rolling_spec
    mspec = MOMENTUM_SPEC if momentum_spec is None else momentum_spec

    _validate(df)

    out = df.copy()
    out = out.sort_values(["campaign_id", "date"]).reset_index(drop=True)

    grouped = out.groupby("campaign_id", sort=False)

    n_roll = 0
    for source_col, aggs in rspec.items():
        prefix = _PREFIX.get(source_col, source_col)
        series_group = grouped[source_col]
        for agg, windows in aggs.items():
            min_p = _MIN_PERIODS.get(agg, 1)
            for w in windows:
                col_name = _rolling_col(prefix, agg, w)
                out[col_name] = series_group.transform(
                    lambda x, _w=w, _mp=min_p, _agg=agg: (
                        getattr(x.shift(1).rolling(_w, min_periods=_mp), _agg)()
                    )
                )
                n_roll += 1

    # Momentum features
    n_mom = 0
    for prefix, windows in mspec.items():
        lag1_col = f"{prefix}_lag_1"
        for w in windows:
            roll_mean_col = _rolling_col(prefix, "mean", w)
            mom_col = f"{prefix}_momentum_{w}"
            denom = out[roll_mean_col].replace(0.0, np.nan)
            out[mom_col] = out[lag1_col] / denom
            n_mom += 1

    logger.info(
        "Rolling features: added %d rolling + %d momentum columns for %d campaigns",
        n_roll,
        n_mom,
        out["campaign_id"].nunique(),
    )
    return out
