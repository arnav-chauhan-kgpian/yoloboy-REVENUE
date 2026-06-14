"""
src/features/lag_features.py
=============================
Campaign-level lag feature generator for the AIgnition forecasting engine.

All lags are computed strictly within each campaign_id, sorted by date.
First N rows per campaign will be NaN for lag-N features (no imputation).
"""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default lag specifications
# ---------------------------------------------------------------------------

REVENUE_LAGS: Final[tuple[int, ...]] = (1, 3, 7, 14, 28)
SPEND_LAGS: Final[tuple[int, ...]] = (1, 3, 7, 14, 28)
CLICKS_LAGS: Final[tuple[int, ...]] = (1, 7, 14, 28)
IMPRESSIONS_LAGS: Final[tuple[int, ...]] = (1, 7, 14, 28)
CONVERSIONS_LAGS: Final[tuple[int, ...]] = (1, 7, 14, 28)

# source column → lag window sizes
LAG_SPEC: Final[dict[str, tuple[int, ...]]] = {
    "revenue_attributed": REVENUE_LAGS,
    "spend": SPEND_LAGS,
    "clicks": CLICKS_LAGS,
    "impressions": IMPRESSIONS_LAGS,
    "conversions": CONVERSIONS_LAGS,
}

# source column → output column prefix
_PREFIX: Final[dict[str, str]] = {
    "revenue_attributed": "revenue",
    "spend": "spend",
    "clicks": "clicks",
    "impressions": "impressions",
    "conversions": "conversions",
}

LAG_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"{_PREFIX[col]}_lag_{n}"
    for col, lags in LAG_SPEC.items()
    for n in lags
)


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class LagFeatureError(Exception):
    pass


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame, spec: dict[str, tuple[int, ...]]) -> None:
    required = {"campaign_id", "date"} | set(spec.keys())
    missing = required - set(df.columns)
    if missing:
        raise LagFeatureError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise LagFeatureError("Input DataFrame is empty")
    if df["date"].isnull().any():
        raise LagFeatureError("date column must not contain NaN")
    if df["campaign_id"].isnull().any():
        raise LagFeatureError("campaign_id column must not contain NaN")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_lag_features(
    df: pd.DataFrame,
    lag_spec: dict[str, tuple[int, ...]] | None = None,
) -> pd.DataFrame:
    """Add campaign-level lag features to the canonical DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical DataFrame produced by harmonizer.  Must contain columns
        ``campaign_id``, ``date``, and all source columns in *lag_spec*.
    lag_spec : dict[str, tuple[int, ...]] | None
        Mapping from source column name to tuple of lag windows in days.
        Defaults to :data:`LAG_SPEC`.

    Returns
    -------
    pd.DataFrame
        Copy of *df* sorted by ``(campaign_id, date)`` with lag columns
        appended.  Lag columns are named ``{prefix}_lag_{n}``.
        The first *n* rows per campaign will be ``NaN`` for lag-*n* features.

    Raises
    ------
    LagFeatureError
        If required columns are absent or the DataFrame is empty.
    """
    spec = LAG_SPEC if lag_spec is None else lag_spec
    _validate(df, spec)

    out = df.copy()
    out = out.sort_values(["campaign_id", "date"]).reset_index(drop=True)

    grouped = out.groupby("campaign_id", sort=False)

    n_cols = 0
    for source_col, lags in spec.items():
        prefix = _PREFIX.get(source_col, source_col)
        series_group = grouped[source_col]
        for n in lags:
            out[f"{prefix}_lag_{n}"] = series_group.shift(n)
            n_cols += 1

    logger.info(
        "Lag features: added %d columns for %d campaigns (%d rows)",
        n_cols,
        out["campaign_id"].nunique(),
        len(out),
    )
    return out
