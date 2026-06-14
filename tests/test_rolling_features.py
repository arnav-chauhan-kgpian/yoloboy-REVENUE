"""
tests/test_rolling_features.py
================================
Pytest suite for src/features/rolling_features.py.

Uses a synthetic canonical DataFrame so every rolling and momentum value
can be verified algebraically.

Revenue for camp_A: day i → (i+1)*100.0  (strictly increasing by 100 each day)
Spend for camp_A:   day i → (i+1)*10.0
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
import pytest

from src.features.lag_features import add_lag_features
from src.features.rolling_features import (
    ALL_ROLLING_COLUMNS,
    MOMENTUM_COLUMNS,
    ROLLING_COLUMNS,
    ROLLING_SPEC,
    RollingFeatureError,
    add_rolling_features,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_DAYS: Final[int] = 35
CAMP_A: Final[str] = "camp_A"
CAMP_B: Final[str] = "camp_B"
START_DATE: Final[pd.Timestamp] = pd.Timestamp("2024-01-01")

CAMP_A_REV_FACTOR: Final[float] = 100.0   # revenue = (day+1) * 100
CAMP_A_SPEND_FACTOR: Final[float] = 10.0  # spend   = (day+1) * 10


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_canonical(n_days: int = N_DAYS) -> pd.DataFrame:
    rows = []
    for cid, rev_f, sp_f in [
        (CAMP_A, CAMP_A_REV_FACTOR, CAMP_A_SPEND_FACTOR),
        (CAMP_B, CAMP_A_REV_FACTOR * 2, CAMP_A_SPEND_FACTOR * 2),
    ]:
        for i in range(n_days):
            rows.append(
                {
                    "platform":           "google" if cid == CAMP_A else "meta",
                    "campaign_id":        cid,
                    "campaign_name":      f"{cid}_name",
                    "date":               START_DATE + pd.Timedelta(days=i),
                    "spend":              float(i + 1) * sp_f,
                    "revenue_attributed": float(i + 1) * rev_f,
                    "clicks":             float(i + 1),
                    "impressions":        float(i + 1) * 100.0,
                    "conversions":        float(i + 1) * 0.1,
                    "daily_budget":       1000.0,
                    "channel_format":     "Search",
                    "reach":              0.0,
                    "video_views":        0.0,
                    "attribution_mature": True,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def lagged() -> pd.DataFrame:
    return add_lag_features(_make_canonical())


@pytest.fixture(scope="module")
def rolled(lagged: pd.DataFrame) -> pd.DataFrame:
    return add_rolling_features(lagged)


@pytest.fixture(scope="module")
def camp_a(rolled: pd.DataFrame) -> pd.DataFrame:
    return rolled[rolled["campaign_id"] == CAMP_A].sort_values("date").reset_index(drop=True)


@pytest.fixture(scope="module")
def camp_b(rolled: pd.DataFrame) -> pd.DataFrame:
    return rolled[rolled["campaign_id"] == CAMP_B].sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arithmetic_mean(n: int, factor: float, end_inclusive: int) -> float:
    """Mean of factor*[end_inclusive-n+1 … end_inclusive]."""
    start = end_inclusive - n + 1
    values = [factor * d for d in range(start, end_inclusive + 1)]
    return sum(values) / len(values)


# ===========================================================================
# TestOutputSchema
# ===========================================================================

class TestOutputSchema:
    """All expected rolling and momentum columns are present."""

    def test_all_rolling_columns_present(self, rolled: pd.DataFrame) -> None:
        for col in ROLLING_COLUMNS:
            assert col in rolled.columns, f"Missing: {col}"

    def test_all_momentum_columns_present(self, rolled: pd.DataFrame) -> None:
        for col in MOMENTUM_COLUMNS:
            assert col in rolled.columns, f"Missing: {col}"

    def test_rolling_cols_are_float(self, rolled: pd.DataFrame) -> None:
        for col in ALL_ROLLING_COLUMNS:
            assert pd.api.types.is_float_dtype(rolled[col]), f"{col} must be float"

    def test_row_count_preserved(self, lagged: pd.DataFrame, rolled: pd.DataFrame) -> None:
        assert len(rolled) == len(lagged)

    def test_original_columns_preserved(self, lagged: pd.DataFrame, rolled: pd.DataFrame) -> None:
        for col in lagged.columns:
            assert col in rolled.columns

    def test_sorted_by_campaign_then_date(self, rolled: pd.DataFrame) -> None:
        for cid, grp in rolled.groupby("campaign_id"):
            assert grp["date"].is_monotonic_increasing


# ===========================================================================
# TestRollingMeanCorrectness
# ===========================================================================

class TestRollingMeanCorrectness:
    """
    Rolling mean at row i = mean of revenue values for the 'window' days
    ending at day i-1 (inclusive).

    Revenue for camp_A: day j → (j+1)*100.  So mean of window W ending at
    day d-1 = mean of days [d-W, …, d-1] → mean of [(d-W+1)*100 … d*100].
    """

    def test_revenue_roll_mean_7_at_row_7(self, camp_a: pd.DataFrame) -> None:
        # Window of 7 days: days 0-6 → revenues 100,200,...,700 → mean = 400
        assert camp_a.loc[7, "revenue_roll_mean_7"] == pytest.approx(400.0)

    def test_revenue_roll_mean_7_at_row_10(self, camp_a: pd.DataFrame) -> None:
        # Days 3-9 → revenues 400,500,...,1000 → mean = 700
        assert camp_a.loc[10, "revenue_roll_mean_7"] == pytest.approx(700.0)

    def test_revenue_roll_mean_14_at_row_14(self, camp_a: pd.DataFrame) -> None:
        # Days 0-13 → revenues 100,...,1400 → mean = 750
        assert camp_a.loc[14, "revenue_roll_mean_14"] == pytest.approx(750.0)

    def test_revenue_roll_mean_28_at_row_28(self, camp_a: pd.DataFrame) -> None:
        # Days 0-27 → revenues 100,...,2800 → mean = 1450
        assert camp_a.loc[28, "revenue_roll_mean_28"] == pytest.approx(1450.0)

    def test_spend_roll_mean_7_at_row_7(self, camp_a: pd.DataFrame) -> None:
        # Days 0-6 → spend 10,...,70 → mean = 40
        assert camp_a.loc[7, "spend_roll_mean_7"] == pytest.approx(40.0)

    def test_clicks_roll_mean_7_at_row_7(self, camp_a: pd.DataFrame) -> None:
        # Days 0-6 → clicks 1,...,7 → mean = 4
        assert camp_a.loc[7, "clicks_roll_mean_7"] == pytest.approx(4.0)

    def test_impressions_roll_mean_7_at_row_7(self, camp_a: pd.DataFrame) -> None:
        # Days 0-6 → impressions 100,...,700 → mean = 400
        assert camp_a.loc[7, "impressions_roll_mean_7"] == pytest.approx(400.0)


# ===========================================================================
# TestRollingMinMaxCorrectness
# ===========================================================================

class TestRollingMinMaxCorrectness:

    def test_revenue_roll_min_7_at_row_7(self, camp_a: pd.DataFrame) -> None:
        # Window days 0-6: min revenue = day 0 = 100
        assert camp_a.loc[7, "revenue_roll_min_7"] == pytest.approx(100.0)

    def test_revenue_roll_max_7_at_row_7(self, camp_a: pd.DataFrame) -> None:
        # Window days 0-6: max revenue = day 6 = 700
        assert camp_a.loc[7, "revenue_roll_max_7"] == pytest.approx(700.0)

    def test_revenue_roll_min_7_at_row_14(self, camp_a: pd.DataFrame) -> None:
        # Window days 7-13: min = day 7 = 800
        assert camp_a.loc[14, "revenue_roll_min_7"] == pytest.approx(800.0)

    def test_revenue_roll_max_7_at_row_14(self, camp_a: pd.DataFrame) -> None:
        # Window days 7-13: max = day 13 = 1400
        assert camp_a.loc[14, "revenue_roll_max_7"] == pytest.approx(1400.0)


# ===========================================================================
# TestRollingStdCorrectness
# ===========================================================================

class TestRollingStdCorrectness:
    """Rolling std uses ddof=1 and min_periods=2."""

    def test_revenue_roll_std_7_at_row_7(self, camp_a: pd.DataFrame) -> None:
        # std([100,200,...,700]) with ddof=1
        values = [i * CAMP_A_REV_FACTOR for i in range(1, 8)]
        expected = pd.Series(values).std(ddof=1)
        assert camp_a.loc[7, "revenue_roll_std_7"] == pytest.approx(expected, rel=1e-4)

    def test_revenue_roll_std_7_at_row_1_is_nan(self, camp_a: pd.DataFrame) -> None:
        # Only 1 value in window after shift: std requires min_periods=2 → NaN
        assert pd.isna(camp_a.loc[1, "revenue_roll_std_7"])

    def test_spend_roll_std_7_at_row_7(self, camp_a: pd.DataFrame) -> None:
        values = [i * CAMP_A_SPEND_FACTOR for i in range(1, 8)]
        expected = pd.Series(values).std(ddof=1)
        assert camp_a.loc[7, "spend_roll_std_7"] == pytest.approx(expected, rel=1e-4)


# ===========================================================================
# TestNaNBehavior
# ===========================================================================

class TestNaNBehavior:
    """Rolling features at row 0 must be NaN (no historical data after shift)."""

    def test_revenue_roll_mean_7_row_0_is_nan(self, camp_a: pd.DataFrame) -> None:
        assert pd.isna(camp_a.loc[0, "revenue_roll_mean_7"])

    def test_revenue_roll_mean_14_row_0_is_nan(self, camp_a: pd.DataFrame) -> None:
        assert pd.isna(camp_a.loc[0, "revenue_roll_mean_14"])

    def test_revenue_roll_std_7_row_0_and_1_are_nan(self, camp_a: pd.DataFrame) -> None:
        assert pd.isna(camp_a.loc[0, "revenue_roll_std_7"])
        assert pd.isna(camp_a.loc[1, "revenue_roll_std_7"])

    def test_momentum_row_0_is_nan(self, camp_a: pd.DataFrame) -> None:
        # lag_1 is NaN at row 0 → momentum NaN
        assert pd.isna(camp_a.loc[0, "revenue_momentum_7"])


# ===========================================================================
# TestShiftedWindowsNoLeakage
# ===========================================================================

class TestShiftedWindowsNoLeakage:
    """Current day's revenue must never appear in any rolling feature."""

    def test_roll_mean_7_does_not_include_today(self, camp_a: pd.DataFrame) -> None:
        # Revenue increases monotonically; roll_mean_7 must be < today's revenue
        # for all rows where a full 7-day window is available
        valid = camp_a.iloc[7:].copy()
        assert (valid["revenue_roll_mean_7"] < valid["revenue_attributed"]).all(), (
            "revenue_roll_mean_7 includes or exceeds today's value — possible leakage"
        )

    def test_roll_mean_14_does_not_include_today(self, camp_a: pd.DataFrame) -> None:
        valid = camp_a.iloc[14:].copy()
        assert (valid["revenue_roll_mean_14"] < valid["revenue_attributed"]).all()

    def test_roll_mean_28_does_not_include_today(self, camp_a: pd.DataFrame) -> None:
        valid = camp_a.iloc[28:].copy()
        assert (valid["revenue_roll_mean_28"] < valid["revenue_attributed"]).all()

    def test_roll_max_7_does_not_equal_today(self, camp_a: pd.DataFrame) -> None:
        # max of days [i-7 … i-1] for increasing series: max = day i-1 < day i
        valid = camp_a.iloc[7:].copy()
        assert (valid["revenue_roll_max_7"] < valid["revenue_attributed"]).all()

    def test_roll_mean_uses_yesterday_not_today(self, camp_a: pd.DataFrame) -> None:
        # At row 8: today = 900, yesterday = 800, roll_mean_7 includes days 1-7 = 400
        # Verify roll_mean_7 at row 8 = mean(200,...,800) = 500
        assert camp_a.loc[8, "revenue_roll_mean_7"] == pytest.approx(500.0)


# ===========================================================================
# TestCampaignIsolation
# ===========================================================================

class TestCampaignIsolation:
    """Rolling features must be computed per campaign."""

    def test_camp_b_roll_mean_7_row_0_is_nan(self, camp_b: pd.DataFrame) -> None:
        assert pd.isna(camp_b.loc[0, "revenue_roll_mean_7"])

    def test_camp_b_values_double_camp_a(
        self, camp_a: pd.DataFrame, camp_b: pd.DataFrame
    ) -> None:
        # camp_B revenues = 2× camp_A → roll_mean should be 2× as well
        a_mean = camp_a.loc[7, "revenue_roll_mean_7"]
        b_mean = camp_b.loc[7, "revenue_roll_mean_7"]
        assert b_mean == pytest.approx(a_mean * 2.0, rel=1e-6)

    def test_camp_a_and_b_have_same_row_count(
        self, camp_a: pd.DataFrame, camp_b: pd.DataFrame
    ) -> None:
        assert len(camp_a) == len(camp_b)


# ===========================================================================
# TestMomentumCorrectness
# ===========================================================================

class TestMomentumCorrectness:
    """Momentum = lag_1 / roll_mean."""

    def test_revenue_momentum_7_at_row_7(self, camp_a: pd.DataFrame) -> None:
        # lag_1 = day 6 revenue = 700; roll_mean_7 = 400 → momentum = 700/400 = 1.75
        assert camp_a.loc[7, "revenue_momentum_7"] == pytest.approx(1.75, rel=1e-5)

    def test_spend_momentum_7_at_row_7(self, camp_a: pd.DataFrame) -> None:
        # lag_1 = day 6 spend = 70; roll_mean_7 = 40 → momentum = 70/40 = 1.75
        assert camp_a.loc[7, "spend_momentum_7"] == pytest.approx(1.75, rel=1e-5)

    def test_revenue_momentum_14_at_row_14(self, camp_a: pd.DataFrame) -> None:
        # lag_1 = day 13 = 1400; roll_mean_14 = 750 → momentum = 1400/750
        assert camp_a.loc[14, "revenue_momentum_14"] == pytest.approx(1400.0 / 750.0, rel=1e-5)

    def test_momentum_gt_one_for_increasing_series(self, camp_a: pd.DataFrame) -> None:
        # With monotonically increasing revenue, lag_1 > roll_mean always
        valid = camp_a.iloc[14:][["revenue_momentum_7", "revenue_momentum_14"]].dropna()
        assert (valid > 1.0).all().all()

    def test_momentum_nan_when_roll_mean_is_zero(self) -> None:
        """Zero denominator must produce NaN, not inf."""
        # Create a campaign with zero revenue for first 7 days
        rows = []
        for i in range(14):
            rows.append(
                {
                    "campaign_id":        "zero_camp",
                    "date":               pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                    "spend":              0.0,
                    "revenue_attributed": 0.0,
                    "clicks":             0.0,
                    "impressions":        0.0,
                    "conversions":        0.0,
                    "daily_budget":       1000.0,
                    "channel_format":     "Search",
                    "reach":              0.0,
                    "video_views":        0.0,
                    "attribution_mature": True,
                    "platform":           "google",
                    "campaign_name":      "zero_camp_name",
                }
            )
        df = add_lag_features(pd.DataFrame(rows))
        result = add_rolling_features(df)
        # Where roll_mean = 0 (all zeros), momentum should be NaN not inf
        zero_denom = result[result["revenue_roll_mean_7"] == 0.0]
        if len(zero_denom) > 0:
            assert zero_denom["revenue_momentum_7"].isna().all()

    def test_momentum_row_0_is_nan(self, camp_a: pd.DataFrame) -> None:
        assert pd.isna(camp_a.loc[0, "revenue_momentum_7"])


# ===========================================================================
# TestValidation
# ===========================================================================

class TestValidation:
    """RollingFeatureError raised for bad inputs."""

    def test_missing_campaign_id_raises(self, lagged: pd.DataFrame) -> None:
        bad = lagged.drop(columns=["campaign_id"])
        with pytest.raises(RollingFeatureError):
            add_rolling_features(bad)

    def test_missing_lag1_raises(self, lagged: pd.DataFrame) -> None:
        bad = lagged.drop(columns=["revenue_lag_1"])
        with pytest.raises(RollingFeatureError, match="lag_1"):
            add_rolling_features(bad)

    def test_missing_source_column_raises(self, lagged: pd.DataFrame) -> None:
        bad = lagged.drop(columns=["revenue_attributed"])
        with pytest.raises(RollingFeatureError):
            add_rolling_features(bad)

    def test_empty_dataframe_raises(self, lagged: pd.DataFrame) -> None:
        empty = lagged.iloc[0:0]
        with pytest.raises(RollingFeatureError, match="empty"):
            add_rolling_features(empty)
