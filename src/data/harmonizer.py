"""
src/data/harmonizer.py
======================
Schema harmonization layer for the AIgnition 2026 marketing mix engine.

Purpose
-------
Accept the three raw platform DataFrames produced by loader.py and transform
them into a single canonical DataFrame with a unified column schema, normalized
values, consistent units, and the attribution-maturity flag.

Inputs
------
A ``RawDataset`` (from ``src.data.loader.load_raw_data``):
  dataset.google — Google Ads raw DataFrame (11 cols, ~19 272 rows)
  dataset.meta   — Meta Ads raw DataFrame   (12 cols,  ~3 417 rows)
  dataset.bing   — Bing Ads raw DataFrame   (10 cols,  ~2 873 rows)

Transformations (in order)
--------------------------
1. Google ``metrics_cost_micros`` / 1 000 000 → ``spend`` (currency).
2. Platform revenue columns → ``revenue_attributed``:
     Google ``metrics_conversions_value``
     Meta   ``conversion``  (this is a REVENUE VALUE, not an event count)
     Bing   ``Revenue``
3. Date columns → ``date``:
     Google ``segments_date`` | Meta ``date_start`` | Bing ``TimePeriod``
4. Campaign ID columns → ``campaign_id``:
     Google/Meta already correct; Bing ``CampaignId`` renamed.
5. Campaign name columns → ``campaign_name``:
     Google/Meta already correct; Bing ``CampaignName`` renamed.
6. ``platform`` column added: literal "google" / "meta" / "bing".
7. ``channel_format`` normalized from platform-reported channel types:
     Google PERFORMANCE_MAX → PMax  |  SEARCH → Search
            SHOPPING        → Shopping
            VIDEO / DEMAND_GEN / DISPLAY → UpperFunnel
            (unknown)       → Other
     Bing   PerformanceMax → PMax   |  Search → Search
            Shopping       → Shopping | Audience → UpperFunnel
     Meta   (no channel_type column — inferred from campaign_name):
            "Adv_Plus"    → PMax        (Advantage+ = Meta's PMax format)
            "DPA"         → Shopping    (Dynamic Product Ads)
            "Brand"       → Brand       (overrides Prospecting/Remarketing)
            "Remarketing" → Remarketing
            "Prospecting" → Prospecting
            (none match)  → Other
8. ``reach`` preserved from Meta; Google/Bing rows receive NaN.
9. ``video_views`` preserved from Google; Meta/Bing rows receive NaN.
10. ``attribution_mature`` flag: False for the last 14 calendar days of each
    platform's data (computed from that platform's own max date), True for all
    earlier rows. Each platform's cutoff advances automatically on data refresh.

Outputs
-------
Single ``pd.DataFrame`` with exactly CANONICAL_COLUMNS in declared order.
Row count equals sum of all three platform row counts (25 562 expected).
dtype summary:
  platform, channel_format → category
  campaign_id              → int64
  campaign_name            → object
  date                     → datetime64[ns]
  spend, revenue_attributed, clicks, impressions,
  conversions, daily_budget, reach, video_views  → float64
  attribution_mature                              → bool

Validation (post-concat, raises HarmonizeError on failure)
----------------------------------------------------------
  - spend >= 0 for all rows
  - revenue_attributed >= 0 for all rows
  - no duplicate (platform, campaign_id, date) composite key

Design axioms
-------------
- NEVER sum revenue across platforms. This module produces per-platform rows;
  cross-platform reconciliation belongs in a later reconciliation layer.
- Meta ``conversion`` is REVENUE VALUE (mean ≈ $485, max ≈ $26 539), not an
  event count. Renaming it ``revenue_attributed`` is the primary semantic fix
  this module exists to apply.
- Google micros conversion happens here and ONLY here. The raw loader leaves
  micros intact as an audit trail.
- Attribution-maturity cutoffs are computed per-platform, not globally, so that
  platforms with different max dates each mask the correct trailing window.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.data.loader import RawDataset, load_raw_data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ATTRIBUTION_MATURITY_DAYS: Final[int] = 14

CANONICAL_COLUMNS: Final[tuple[str, ...]] = (
    "platform",
    "campaign_id",
    "campaign_name",
    "date",
    "spend",
    "revenue_attributed",
    "clicks",
    "impressions",
    "conversions",
    "daily_budget",
    "channel_format",
    "reach",
    "video_views",
    "attribution_mature",
)

CANONICAL_CHANNEL_FORMATS: Final[frozenset[str]] = frozenset({
    "Search",
    "PMax",
    "Shopping",
    "UpperFunnel",
    "Prospecting",
    "Remarketing",
    "Brand",
    "Other",
})

# Google campaign_advertising_channel_type → canonical channel_format
_GOOGLE_CHANNEL_MAP: Final[dict[str, str]] = {
    "SEARCH":          "Search",
    "PERFORMANCE_MAX": "PMax",
    "SHOPPING":        "Shopping",
    "VIDEO":           "UpperFunnel",
    "DEMAND_GEN":      "UpperFunnel",
    "DISPLAY":         "UpperFunnel",
}

# Bing CampaignType → canonical channel_format
_BING_CHANNEL_MAP: Final[dict[str, str]] = {
    "Search":         "Search",
    "PerformanceMax": "PMax",
    "Shopping":       "Shopping",
    "Audience":       "UpperFunnel",
}

# Canonical dtype map applied after concat to enforce consistency
_CANONICAL_DTYPES: Final[dict[str, str]] = {
    "campaign_id":        "int64",
    "spend":              "float64",
    "revenue_attributed": "float64",
    "clicks":             "float64",
    "impressions":        "float64",
    "conversions":        "float64",
    "daily_budget":       "float64",
    "reach":              "float64",
    "video_views":        "float64",
    "attribution_mature": "bool",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class HarmonizeError(Exception):
    """
    Raised when post-concat validation detects a data integrity violation.

    Attributes
    ----------
    details : str
        Sample rows or context for debugging.
    """

    def __init__(self, message: str, details: str = "") -> None:
        super().__init__(message)
        self.details = details

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base}\n  details: {self.details}" if self.details else base


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _maturity_cutoff(date_series: pd.Series) -> pd.Timestamp:
    """
    Return the last date considered attribution-mature for a given platform.

    Any row with date > cutoff is in the trailing 14-day immature window.
    Cutoff = platform_max_date - 14 days (so the 14 most recent days are
    immature, everything before is mature).

    Parameters
    ----------
    date_series : pd.Series[datetime64[ns]]
        The date column for a single platform.

    Returns
    -------
    pd.Timestamp
        Cutoff date (inclusive on the mature side).
    """
    return date_series.max() - pd.Timedelta(days=ATTRIBUTION_MATURITY_DAYS)


def _infer_meta_channel_format(names: pd.Series) -> pd.Series:
    """
    Infer ``channel_format`` for Meta from campaign_name strings.

    Meta has no platform-reported channel type, so format is derived from
    naming conventions observed across all 16 Meta campaigns in the dataset.

    Priority (highest overwrites lower — applied via sequential assignment):
      "Adv_Plus"    → PMax        (Advantage+ is Meta's equivalent of PMax)
      "DPA"         → Shopping    (Dynamic Product Ads)
      "Brand"       → Brand
      "Remarketing" → Remarketing
      "Prospecting" → Prospecting
      (none match)  → Other

    Examples
    --------
    "Prospecting_Adv_Plus_Campaign_01" → PMax   (Adv_Plus wins)
    "Prospecting_DPA_Campaign_01"      → Shopping (DPA wins over Prospecting)
    "Prospecting_Brand_Campaign_01"    → Brand  (Brand wins over Prospecting)
    "Remarketing_DPA_Campaign_01"      → Shopping (DPA wins over Remarketing)
    "Remarketing_Brand_Campaign_01"    → Brand  (Brand wins over Remarketing)
    "Generic_Campaign_01"              → Other

    Parameters
    ----------
    names : pd.Series[str]
        ``campaign_name`` column from the Meta DataFrame.

    Returns
    -------
    pd.Series[str]
        Inferred channel_format for each row.
    """
    result = pd.Series("Other", index=names.index, dtype="object")

    # Apply in ascending priority; each overwrites the previous assignment.
    result[names.str.contains("Prospecting", na=False)] = "Prospecting"
    result[names.str.contains("Remarketing", na=False)] = "Remarketing"
    result[names.str.contains("Brand",       na=False)] = "Brand"
    result[names.str.contains("DPA",         na=False)] = "Shopping"
    result[names.str.contains("Adv_Plus",    na=False)] = "PMax"  # highest priority

    return result


# ---------------------------------------------------------------------------
# Per-platform normalizers (private — tested via harmonize())
# ---------------------------------------------------------------------------

def _normalize_google(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Google Ads DataFrame into the canonical schema.

    Parameters
    ----------
    df : pd.DataFrame
        Output of DataLoader.load_google().  Must contain the full Google
        column set with native naming.

    Returns
    -------
    pd.DataFrame
        Rows in canonical column order.  Same row count as input.
        Google-specific: ``reach`` = NaN; ``video_views`` populated.
    """
    cutoff = _maturity_cutoff(df["segments_date"])
    logger.debug("[Google] Attribution maturity cutoff: %s", cutoff.date())

    out = pd.DataFrame(index=df.index)
    out["platform"]            = "google"
    out["campaign_id"]         = df["campaign_id"]
    out["campaign_name"]       = df["campaign_name"]
    out["date"]                = df["segments_date"]
    out["spend"]               = df["metrics_cost_micros"] / 1_000_000.0
    out["revenue_attributed"]  = df["metrics_conversions_value"]
    out["clicks"]              = df["metrics_clicks"].astype("float64")
    out["impressions"]         = df["metrics_impressions"].astype("float64")
    out["conversions"]         = df["metrics_conversions"]
    out["daily_budget"]        = df["campaign_budget_amount"]
    out["channel_format"]      = (
        df["campaign_advertising_channel_type"]
        .map(_GOOGLE_CHANNEL_MAP)
        .fillna("Other")
    )
    out["reach"]               = np.nan
    out["video_views"]         = df["metrics_video_views"].astype("float64")
    out["attribution_mature"]  = df["segments_date"] <= cutoff
    return out


def _normalize_meta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Meta Ads DataFrame into the canonical schema.

    Parameters
    ----------
    df : pd.DataFrame
        Output of DataLoader.load_meta().  Must contain the full Meta column
        set with native naming.

    Returns
    -------
    pd.DataFrame
        Rows in canonical column order.  Same row count as input.
        Meta-specific: ``reach`` populated; ``video_views`` and ``conversions``
        (count) = NaN (not reported in this dataset).
        ``conversion`` (Meta revenue value) renamed to ``revenue_attributed``.
    """
    cutoff = _maturity_cutoff(df["date_start"])
    logger.debug("[Meta] Attribution maturity cutoff: %s", cutoff.date())

    out = pd.DataFrame(index=df.index)
    out["platform"]            = "meta"
    out["campaign_id"]         = df["campaign_id"]
    out["campaign_name"]       = df["campaign_name"]
    out["date"]                = df["date_start"]
    out["spend"]               = df["spend"]
    out["revenue_attributed"]  = df["conversion"]   # revenue VALUE — not an event count
    out["clicks"]              = df["clicks"]
    out["impressions"]         = df["impressions"]
    out["conversions"]         = np.nan             # conversion event count not in dataset
    out["daily_budget"]        = df["daily_budget"]
    out["channel_format"]      = _infer_meta_channel_format(df["campaign_name"])
    out["reach"]               = df["reach"]
    out["video_views"]         = np.nan             # not in dataset
    out["attribution_mature"]  = df["date_start"] <= cutoff
    return out


def _normalize_bing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Bing Ads DataFrame into the canonical schema.

    Parameters
    ----------
    df : pd.DataFrame
        Output of DataLoader.load_bing().  Must contain the full Bing column
        set with PascalCase native naming.

    Returns
    -------
    pd.DataFrame
        Rows in canonical column order.  Same row count as input.
        Bing-specific: ``reach`` and ``video_views`` = NaN (not in dataset).
    """
    cutoff = _maturity_cutoff(df["TimePeriod"])
    logger.debug("[Bing] Attribution maturity cutoff: %s", cutoff.date())

    out = pd.DataFrame(index=df.index)
    out["platform"]            = "bing"
    out["campaign_id"]         = df["CampaignId"]
    out["campaign_name"]       = df["CampaignName"]
    out["date"]                = df["TimePeriod"]
    out["spend"]               = df["Spend"]
    out["revenue_attributed"]  = df["Revenue"]
    out["clicks"]              = df["Clicks"].astype("float64")
    out["impressions"]         = df["Impressions"].astype("float64")
    out["conversions"]         = df["Conversions"]
    out["daily_budget"]        = df["DailyBudget"]
    out["channel_format"]      = (
        df["CampaignType"]
        .map(_BING_CHANNEL_MAP)
        .fillna("Other")
    )
    out["reach"]               = np.nan
    out["video_views"]         = np.nan
    out["attribution_mature"]  = df["TimePeriod"] <= cutoff
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_canonical(df: pd.DataFrame) -> None:
    """
    Run post-concat data integrity checks on the canonical DataFrame.

    Checks
    ------
    1. ``spend`` >= 0 for every row.
    2. ``revenue_attributed`` >= 0 for every row.
    3. No duplicate (platform, campaign_id, date) composite key.

    Parameters
    ----------
    df : pd.DataFrame
        Fully assembled canonical DataFrame.

    Raises
    ------
    HarmonizeError
        On the first failing check, with a description and sample of bad rows.
    """
    mask_neg_spend = df["spend"] < 0
    if mask_neg_spend.any():
        sample = (
            df.loc[mask_neg_spend, ["platform", "campaign_id", "date", "spend"]]
            .head(5)
            .to_string(index=False)
        )
        raise HarmonizeError(
            f"{mask_neg_spend.sum()} rows have negative spend.",
            details=sample,
        )

    mask_neg_rev = df["revenue_attributed"] < 0
    if mask_neg_rev.any():
        sample = (
            df.loc[mask_neg_rev, ["platform", "campaign_id", "date", "revenue_attributed"]]
            .head(5)
            .to_string(index=False)
        )
        raise HarmonizeError(
            f"{mask_neg_rev.sum()} rows have negative revenue_attributed.",
            details=sample,
        )

    dup_mask = df.duplicated(subset=["platform", "campaign_id", "date"], keep=False)
    if dup_mask.any():
        sample = (
            df.loc[dup_mask, ["platform", "campaign_id", "date"]]
            .head(5)
            .to_string(index=False)
        )
        raise HarmonizeError(
            f"{dup_mask.sum()} duplicate rows on (platform, campaign_id, date).",
            details=sample,
        )

    logger.info(
        "Canonical validation passed: %d rows, %d columns", len(df), df.shape[1]
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def harmonize(dataset: RawDataset) -> pd.DataFrame:
    """
    Harmonize three raw platform DataFrames into a single canonical DataFrame.

    Parameters
    ----------
    dataset : RawDataset
        Output of ``src.data.loader.load_raw_data()`` or
        ``DataLoader.load_all()``.

    Returns
    -------
    pd.DataFrame
        Canonical DataFrame with CANONICAL_COLUMNS in declared order.
        Row count = sum of all three platform row counts (25 562 expected).
        Dtypes: see module docstring.

    Raises
    ------
    HarmonizeError
        If any post-concat validation check fails (negative spend/revenue,
        duplicate composite key).

    Examples
    --------
    >>> from src.data.loader import load_raw_data
    >>> from src.data.harmonizer import harmonize, CANONICAL_COLUMNS
    >>> ds = load_raw_data()
    >>> canon = harmonize(ds)
    >>> canon.shape
    (25562, 14)
    >>> list(canon.columns) == list(CANONICAL_COLUMNS)
    True
    >>> canon["platform"].dtype.name
    'category'
    >>> canon["attribution_mature"].dtype
    dtype('bool')
    """
    logger.info("=== Harmonizer: start ===")

    google_norm = _normalize_google(dataset.google)
    logger.info("[Google] Normalized: %d rows", len(google_norm))

    meta_norm = _normalize_meta(dataset.meta)
    logger.info("[Meta]   Normalized: %d rows", len(meta_norm))

    bing_norm = _normalize_bing(dataset.bing)
    logger.info("[Bing]   Normalized: %d rows", len(bing_norm))

    combined = pd.concat([google_norm, meta_norm, bing_norm], ignore_index=True)

    # Enforce canonical column order
    combined = combined[list(CANONICAL_COLUMNS)]

    # Enforce dtypes — category columns handled separately (astype dict skips them)
    combined = combined.astype(_CANONICAL_DTYPES)
    # Explicit datetime64[ns] cast: pandas 2.0+ pd.to_datetime() may return
    # datetime64[us] or datetime64[s] depending on resolution inference, which
    # breaks tests that assert dtype("datetime64[ns]").
    combined["date"]           = combined["date"].astype("datetime64[ns]")
    combined["platform"]       = combined["platform"].astype("category")
    combined["channel_format"] = combined["channel_format"].astype("category")

    _validate_canonical(combined)

    n_mature   = int(combined["attribution_mature"].sum())
    n_immature = int((~combined["attribution_mature"]).sum())
    logger.info(
        "=== Harmonizer complete: %d rows | %d cols | "
        "%d mature | %d immature (trailing %d days per platform) ===",
        len(combined),
        combined.shape[1],
        n_mature,
        n_immature,
        ATTRIBUTION_MATURITY_DAYS,
    )

    return combined


def harmonize_from_dir(data_dir: str | Path = "dataset") -> pd.DataFrame:
    """
    Load raw data from ``data_dir`` and harmonize in one call.

    Convenience wrapper around ``harmonize(load_raw_data(data_dir))``.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing the three platform CSV files.

    Returns
    -------
    pd.DataFrame
        Canonical harmonized DataFrame.

    Raises
    ------
    FileNotFoundError
        If ``data_dir`` or any platform file is missing.
    HarmonizeError
        If post-concat validation fails.
    """
    return harmonize(load_raw_data(data_dir))
