"""
src/features/feature_store.py
==============================
Final training feature matrix builder for the meridian forecasting engine.

Joins all feature sources into a single flat DataFrame ready for LightGBM:

    canonical (harmonizer)
        → lag features (lag_features)
        → rolling + momentum features (rolling_features)
        → taxonomy attributes (taxonomy_parser)
        → holiday calendar (holiday_calendar)
        → calendar cyclical features
        → budget utilisation
        → ROAS lag features
        → validation

Output grain: one row per (campaign_id, date).
Target column: revenue_attributed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.data.harmonizer import harmonize_from_dir
from src.data.taxonomy_parser import parse_taxonomy
from src.features.holiday_calendar import build_holiday_calendar
from src.features.lag_features import add_lag_features
from src.features.rolling_features import add_rolling_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature column manifests
# ---------------------------------------------------------------------------

CALENDAR_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_weekend",
)

TAXONOMY_JOIN_COLUMNS: Final[tuple[str, ...]] = (
    "format",
    "audience_strategy",
    "funnel_stage",
    "ad_product_type",
    "strategy_key",
    "is_brand",
    "is_non_brand",
    "is_upper_funnel",
    "cross_engine_pair_flag",
)

HOLIDAY_JOIN_COLUMNS: Final[tuple[str, ...]] = (
    "is_holiday",
    "is_bfcm",
    "is_cyber_week",
    "is_holiday_season",
    "days_to_black_friday",
    "days_since_black_friday",
    "days_to_thanksgiving",
    "days_since_thanksgiving",
    "holiday_intensity_score",
)

BUDGET_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "budget_utilization",
    "budget_headroom",
)

ROAS_LAG_COLUMNS: Final[tuple[str, ...]] = (
    "roas_lag_7",
    "roas_lag_14",
    "roas_lag_28",
)

TARGET_COLUMN: Final[str] = "revenue_attributed"


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class FeatureStoreError(Exception):
    pass


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar cyclical features from the ``date`` column."""
    dt = df["date"].dt
    df = df.copy()
    df["day_of_week"]  = dt.dayofweek.astype("int8")
    df["day_of_month"] = dt.day.astype("int8")
    df["week_of_year"] = dt.isocalendar().week.astype("int8")
    df["month"]        = dt.month.astype("int8")
    df["quarter"]      = dt.quarter.astype("int8")
    df["year"]         = dt.year.astype("int16")
    df["is_weekend"]   = (dt.dayofweek >= 5).astype(bool)
    return df


def _add_budget_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add budget utilisation and headroom features.

    Both features are NaN where ``daily_budget`` is NaN (21 known rows in
    the production dataset: 14 Google + 7 Meta with missing budget data).
    """
    df = df.copy()
    budget = df["daily_budget"].replace(0.0, np.nan)
    df["budget_utilization"] = df["spend"] / budget
    df["budget_headroom"]    = df["daily_budget"] - df["spend"]
    return df


def _add_roas_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged ROAS features (revenue / spend) using lagged values only.

    Zero spend denominators become NaN to avoid division-by-zero.
    """
    df = df.copy()
    for w in (7, 14, 28):
        rev_col   = f"revenue_lag_{w}"
        spend_col = f"spend_lag_{w}"
        denom = df[spend_col].replace(0.0, np.nan)
        df[f"roas_lag_{w}"] = df[rev_col] / denom
    return df


def _merge_taxonomy(
    df: pd.DataFrame,
    taxonomy: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join taxonomy campaign attributes onto the feature store."""
    keep = list(TAXONOMY_JOIN_COLUMNS)
    tax_slim = taxonomy[["platform", "campaign_id"] + keep].drop_duplicates(
        subset=["platform", "campaign_id"]
    )
    merged = df.merge(tax_slim, on=["platform", "campaign_id"], how="left")

    # strategy_key is always non-null for matched rows (unlike 'format' which is
    # legitimately None for Meta campaigns)
    n_missing = merged["strategy_key"].isnull().sum()
    if n_missing > 0:
        logger.warning(
            "Taxonomy join: %d rows have no taxonomy match "
            "(platform/campaign_id not found in taxonomy).",
            n_missing,
        )
    return merged


def _merge_holiday(
    df: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join holiday calendar features onto the feature store by date."""
    keep = list(HOLIDAY_JOIN_COLUMNS)
    # Normalise both sides to date-only (midnight) before joining
    cal_slim = calendar[["date"] + keep].copy()
    cal_slim["date"] = pd.to_datetime(cal_slim["date"]).dt.normalize()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    merged = df.merge(cal_slim, on="date", how="left")

    n_missing = merged[keep[0]].isnull().sum()
    if n_missing > 0:
        logger.warning(
            "Holiday join: %d rows have no calendar match.  "
            "Extend the holiday calendar date range to cover all input dates.",
            n_missing,
        )
    return merged


def _validate_feature_store(df: pd.DataFrame) -> None:
    """Post-build validation: duplicates, grain, required columns."""
    # Grain check
    dupes = df.duplicated(subset=["campaign_id", "date"]).sum()
    if dupes > 0:
        raise FeatureStoreError(
            f"Feature store contains {dupes} duplicate (campaign_id, date) rows."
        )

    # Target must be present
    if TARGET_COLUMN not in df.columns:
        raise FeatureStoreError(f"Target column '{TARGET_COLUMN}' is missing.")

    # Essential feature groups must be present
    for col in CALENDAR_FEATURE_COLUMNS:
        if col not in df.columns:
            raise FeatureStoreError(f"Calendar feature '{col}' is missing.")

    for col in BUDGET_FEATURE_COLUMNS:
        if col not in df.columns:
            raise FeatureStoreError(f"Budget feature '{col}' is missing.")

    for col in ROAS_LAG_COLUMNS:
        if col not in df.columns:
            raise FeatureStoreError(f"ROAS lag feature '{col}' is missing.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_store(
    canon: pd.DataFrame | None = None,
    taxonomy: pd.DataFrame | None = None,
    calendar: pd.DataFrame | None = None,
    data_dir: str | Path = "dataset",
) -> pd.DataFrame:
    """Build the complete training feature matrix.

    Parameters
    ----------
    canon : pd.DataFrame | None
        Canonical DataFrame from :func:`~src.data.harmonizer.harmonize`.
        If *None*, loaded via :func:`~src.data.harmonizer.harmonize_from_dir`
        using *data_dir*.
    taxonomy : pd.DataFrame | None
        Campaign taxonomy from :func:`~src.data.taxonomy_parser.parse_taxonomy`.
        If *None*, derived from *canon*.
    calendar : pd.DataFrame | None
        Holiday calendar from
        :func:`~src.features.holiday_calendar.build_holiday_calendar`.
        If *None*, built for years 2024–2027.
    data_dir : str | Path
        Directory containing raw CSV files.  Used only when *canon* is *None*.

    Returns
    -------
    pd.DataFrame
        One row per ``(campaign_id, date)``, sorted by ``(campaign_id, date)``.
        Contains all lag, rolling, calendar, taxonomy, holiday, budget, and
        ROAS lag features plus the target column ``revenue_attributed``.

    Raises
    ------
    FeatureStoreError
        If post-build validation fails (duplicates, missing columns).
    """
    # --- Load canonical if not provided ---
    if canon is None:
        logger.info("Loading canonical DataFrame from %s", data_dir)
        canon = harmonize_from_dir(data_dir)

    # --- Derive taxonomy if not provided ---
    if taxonomy is None:
        logger.info("Deriving taxonomy from canonical DataFrame")
        taxonomy = parse_taxonomy(canon)

    # --- Build holiday calendar if not provided ---
    if calendar is None:
        min_year = canon["date"].dt.year.min()
        max_year = canon["date"].dt.year.max()
        logger.info(
            "Building holiday calendar %d–%d", min_year, max_year
        )
        calendar = build_holiday_calendar(
            start_year=int(min_year),
            end_year=int(max_year),
        )

    n_input = len(canon)
    logger.info("Feature store build starting: %d input rows", n_input)

    # --- Lag features ---
    df = add_lag_features(canon)

    # --- Rolling + momentum features ---
    df = add_rolling_features(df)

    # --- Taxonomy join ---
    df = _merge_taxonomy(df, taxonomy)

    # --- Holiday calendar join ---
    df = _merge_holiday(df, calendar)

    # --- Calendar cyclical features ---
    df = _add_calendar_features(df)

    # --- Budget utilisation ---
    df = _add_budget_features(df)

    # --- ROAS lags ---
    df = _add_roas_lags(df)

    # --- Sort and reset index ---
    df = df.sort_values(["campaign_id", "date"]).reset_index(drop=True)

    # --- Validate ---
    _validate_feature_store(df)

    logger.info(
        "Feature store complete: %d rows × %d columns",
        len(df),
        len(df.columns),
    )
    return df
