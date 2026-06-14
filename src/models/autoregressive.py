"""
src/models/autoregressive.py
==============================
Autoregressive future revenue forecasting.

At each horizon step k (k = 1 .. n_future_days):

  1.  Revenue lag features (revenue_lag_{n}) are populated from a buffer that
      combines actual historical values with P50 predictions from all prior
      steps (k-1, k-2, …, 1).
  2.  Rolling revenue statistics (revenue_roll_{mean,std,min,max}_{w}) are
      recomputed over the same mixed actual+predicted window so that windows
      gradually fill with predictions as the horizon extends.
  3.  Revenue momentum and ROAS lag features derived from the above are
      recomputed for internal consistency.
  4.  Calendar features are updated for the future date.
  5.  Spend, clicks, impressions, taxonomy, and holiday features remain fixed
      at their last known values (spend is a decision variable; clicks and
      impressions are assumed proportional to spend which is held constant).
  6.  All campaigns are predicted in a single batched model.predict() call per
      horizon step.
  7.  The P50 prediction for each campaign is stored in the prediction buffer
      before moving to step k+1.

Uncertainty grows with horizon because prediction error accumulates through
lag inputs: by step 14, the 7-day rolling window is entirely filled with
prior predictions, compounding any systematic bias.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants matching lag_features.py and rolling_features.py
# These MUST stay in sync with the feature-engineering definitions.
# ---------------------------------------------------------------------------

# lag_features.REVENUE_LAGS
_REVENUE_LAGS: tuple[int, ...] = (1, 3, 7, 14, 28)

# rolling_features.ROLLING_SPEC["revenue_attributed"]
_ROLL_MEAN_WINDOWS: tuple[int, ...] = (7, 14, 28)
_ROLL_STD_WINDOWS:  tuple[int, ...] = (7, 14, 28)
_ROLL_MIN_WINDOWS:  tuple[int, ...] = (7,)
_ROLL_MAX_WINDOWS:  tuple[int, ...] = (7,)

# rolling_features.MOMENTUM_SPEC["revenue"]
_MOMENTUM_WINDOWS: tuple[int, ...] = (7, 14, 28)

# feature_store.ROAS_LAG_COLUMNS
_ROAS_LAG_WINDOWS: tuple[int, ...] = (7, 14, 28)

# Buffer depth: must cover max(rolling window, lag) + headroom
# Longest rolling window = 28; longest lag = 28; total = 56 → use 60
_BUFFER_DAYS: int = 60


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_actual_buffer(
    fs: pd.DataFrame,
    cid: str,
    last_date: pd.Timestamp,
) -> dict[pd.Timestamp, float]:
    """Return a date → revenue_attributed dict for the last _BUFFER_DAYS actuals."""
    start = last_date - pd.Timedelta(days=_BUFFER_DAYS - 1)
    camp = (
        fs[(fs["campaign_id"] == cid) & (fs["date"] >= start)]
        .set_index("date")["revenue_attributed"]
    )
    return {pd.Timestamp(d): float(v) for d, v in camp.items()}


def _lookup(
    date: pd.Timestamp,
    actual: dict[pd.Timestamp, float],
    predicted: dict[pd.Timestamp, float],
) -> float:
    """Return revenue at *date*: predicted P50 takes precedence over actual."""
    if date in predicted:
        return predicted[date]
    return actual.get(date, np.nan)


def _window_values(
    future_date: pd.Timestamp,
    window: int,
    actual: dict[pd.Timestamp, float],
    predicted: dict[pd.Timestamp, float],
) -> list[float]:
    """Collect non-NaN revenue values for the rolling window ending at future_date-1.

    Reproduces the shift(1).rolling(window) logic from rolling_features.py:
    values used are those at future_date-1, future_date-2, …, future_date-window.
    """
    values = []
    for lag in range(1, window + 1):
        v = _lookup(future_date - pd.Timedelta(days=lag), actual, predicted)
        if not np.isnan(v):
            values.append(v)
    return values


def _update_revenue_features(
    row: pd.Series,
    future_date: pd.Timestamp,
    actual: dict[pd.Timestamp, float],
    predicted: dict[pd.Timestamp, float],
) -> pd.Series:
    """Overwrite all revenue-derived lag/rolling features in *row* for *future_date*.

    Only touches columns that are present in *row* (i.e. were in the training
    feature set).  All other columns are left unchanged.
    """
    row = row.copy()
    idx = row.index  # fast membership test

    # ------------------------------------------------------------------ #
    # 1. Revenue lags                                                      #
    # ------------------------------------------------------------------ #
    for lag_n in _REVENUE_LAGS:
        col = f"revenue_lag_{lag_n}"
        if col in idx:
            target = future_date - pd.Timedelta(days=lag_n)
            row[col] = _lookup(target, actual, predicted)

    # ------------------------------------------------------------------ #
    # 2. Rolling mean                                                      #
    # ------------------------------------------------------------------ #
    for w in _ROLL_MEAN_WINDOWS:
        col = f"revenue_roll_mean_{w}"
        if col in idx:
            vals = _window_values(future_date, w, actual, predicted)
            row[col] = float(np.mean(vals)) if vals else np.nan

    # ------------------------------------------------------------------ #
    # 3. Rolling std                                                       #
    # ------------------------------------------------------------------ #
    for w in _ROLL_STD_WINDOWS:
        col = f"revenue_roll_std_{w}"
        if col in idx:
            vals = _window_values(future_date, w, actual, predicted)
            row[col] = float(np.std(vals, ddof=1)) if len(vals) >= 2 else np.nan

    # ------------------------------------------------------------------ #
    # 4. Rolling min                                                       #
    # ------------------------------------------------------------------ #
    for w in _ROLL_MIN_WINDOWS:
        col = f"revenue_roll_min_{w}"
        if col in idx:
            vals = _window_values(future_date, w, actual, predicted)
            row[col] = float(np.min(vals)) if vals else np.nan

    # ------------------------------------------------------------------ #
    # 5. Rolling max                                                       #
    # ------------------------------------------------------------------ #
    for w in _ROLL_MAX_WINDOWS:
        col = f"revenue_roll_max_{w}"
        if col in idx:
            vals = _window_values(future_date, w, actual, predicted)
            row[col] = float(np.max(vals)) if vals else np.nan

    # ------------------------------------------------------------------ #
    # 6. Revenue momentum  =  revenue_lag_1 / revenue_roll_mean_w         #
    # ------------------------------------------------------------------ #
    lag1_col = "revenue_lag_1"
    lag1 = row[lag1_col] if lag1_col in idx else np.nan
    for w in _MOMENTUM_WINDOWS:
        col = f"revenue_momentum_{w}"
        if col not in idx:
            continue
        roll_mean_col = f"revenue_roll_mean_{w}"
        roll_mean = row[roll_mean_col] if roll_mean_col in idx else np.nan
        if not np.isnan(roll_mean) and roll_mean != 0.0:
            row[col] = lag1 / roll_mean
        else:
            row[col] = np.nan

    # ------------------------------------------------------------------ #
    # 7. ROAS lags  =  revenue_lag_w / spend_lag_w                        #
    # ------------------------------------------------------------------ #
    for w in _ROAS_LAG_WINDOWS:
        col = f"roas_lag_{w}"
        if col not in idx:
            continue
        rev_lag_col = f"revenue_lag_{w}"
        spd_lag_col = f"spend_lag_{w}"
        rev_lag = row[rev_lag_col] if rev_lag_col in idx else np.nan
        spd_lag = row[spd_lag_col] if spd_lag_col in idx else np.nan
        if not np.isnan(spd_lag) and spd_lag != 0.0:
            row[col] = rev_lag / spd_lag
        else:
            row[col] = np.nan

    return row


def _update_calendar_features(row: pd.Series, date: pd.Timestamp) -> pd.Series:
    """Overwrite calendar fields in *row* for *date*."""
    row = row.copy()
    idx = row.index
    if "date"         in idx: row["date"]         = date
    if "day_of_week"  in idx: row["day_of_week"]  = date.dayofweek
    if "day_of_month" in idx: row["day_of_month"] = date.day
    if "week_of_year" in idx: row["week_of_year"] = int(date.isocalendar()[1])
    if "month"        in idx: row["month"]         = date.month
    if "quarter"      in idx: row["quarter"]       = (date.month - 1) // 3 + 1
    if "year"         in idx: row["year"]          = date.year
    if "is_weekend"   in idx: row["is_weekend"]    = int(date.dayofweek >= 5)
    return row


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_future_forecasts(
    fs: pd.DataFrame,
    model: Any,
    feature_cols: list[str],
    n_future_days: int = 14,
) -> pd.DataFrame:
    """Autoregressive P10/P50/P90 projection for *n_future_days* beyond the feature store.

    Parameters
    ----------
    fs : pd.DataFrame
        Feature store produced by build_feature_store().  Must contain at least
        the last _BUFFER_DAYS (60) days of data before the forecast origin.
    model : RevenueQuantileModel
        Trained quantile model with a .predict(X) method returning a DataFrame
        with columns 'p10', 'p50', 'p90'.
    feature_cols : list[str]
        Ordered list of feature columns (output of get_feature_columns()).
        Only columns present in both this list and the batch DataFrame are
        passed to model.predict().
    n_future_days : int
        Number of calendar days to forecast beyond the last date in *fs*.

    Returns
    -------
    pd.DataFrame
        One row per (campaign, future day).  Columns:
            campaign_id, campaign_name, platform, date,
            revenue_attributed (always NaN), p10, p50, p90, is_future (True).
        Row count: n_campaigns × n_future_days.
    """
    if fs is None or fs.empty or n_future_days < 1:
        return pd.DataFrame()

    last_date = pd.Timestamp(fs["date"].max())
    campaign_ids: list[str] = sorted(fs["campaign_id"].unique())

    if not campaign_ids:
        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Build per-campaign state                                             #
    # ------------------------------------------------------------------ #
    actual_buffers:    dict[str, dict[pd.Timestamp, float]] = {}
    predicted_buffers: dict[str, dict[pd.Timestamp, float]] = {
        cid: {} for cid in campaign_ids
    }
    last_rows: dict[str, pd.Series] = {}

    for cid in campaign_ids:
        camp_df = fs[fs["campaign_id"] == cid].sort_values("date")
        actual_buffers[cid] = _build_actual_buffer(fs, cid, last_date)
        last_rows[cid] = camp_df.iloc[-1].copy()

    # ------------------------------------------------------------------ #
    # Autoregressive rollout                                               #
    # ------------------------------------------------------------------ #
    results: list[dict] = []

    for day_offset in range(1, n_future_days + 1):
        future_date = last_date + pd.Timedelta(days=day_offset)

        batch_rows: list[pd.Series] = []
        for cid in campaign_ids:
            row = last_rows[cid].copy()
            row = _update_calendar_features(row, future_date)
            row = _update_revenue_features(
                row,
                future_date,
                actual_buffers[cid],
                predicted_buffers[cid],
            )
            batch_rows.append(row)

        # Single batched predict — one call covers all campaigns for this day
        batch_df  = pd.DataFrame(batch_rows)
        X_cols    = [c for c in feature_cols if c in batch_df.columns]
        preds     = model.predict(batch_df[X_cols])

        for i, cid in enumerate(campaign_ids):
            p50 = float(preds["p50"].iloc[i])
            predicted_buffers[cid][future_date] = p50

            base = last_rows[cid]
            results.append({
                "campaign_id":        cid,
                "campaign_name":      base.get("campaign_name", cid),
                "platform":           base.get("platform", ""),
                "date":               future_date,
                "revenue_attributed": np.nan,
                "p10":                float(preds["p10"].iloc[i]),
                "p50":                p50,
                "p90":                float(preds["p90"].iloc[i]),
                "is_future":          True,
            })

        logger.debug(
            "Autoregressive step %d/%d (date=%s): %d campaigns predicted",
            day_offset, n_future_days, future_date.date(), len(campaign_ids),
        )

    return pd.DataFrame(results) if results else pd.DataFrame()
