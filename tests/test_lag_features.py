"""
tests/test_lag_features.py
===========================
Pytest suite for src/features/lag_features.py.

Uses a small synthetic canonical DataFrame (2 campaigns × 35 days) so every
lag value can be verified algebraically without loading real data.
"""

from __future__ import annotations

from datetime import date
from typing import Final

import numpy as np
import pandas as pd
import pytest

from src.features.lag_features import (
    LAG_COLUMNS,
    LAG_SPEC,
    LagFeatureError,
    add_lag_features,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_DAYS: Final[int] = 35          # enough for all 28-day lags
CAMP_A: Final[str] = "camp_A"
CAMP_B: Final[str] = "camp_B"
START_DATE: Final[pd.Timestamp] = pd.Timestamp("2024-01-01")

# Camp A revenue: day i → (i+1)*100.0   (1-indexed, monotonically increasing)
# Camp B revenue: day i → (i+1)*200.0   (2× camp A, never confused)
CAMP_A_FACTOR: Final[float] = 100.0
CAMP_B_FACTOR: Final[float] = 200.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_canonical(n_days: int = N_DAYS) -> pd.DataFrame:
    """Two campaigns, *n_days* days each, deterministic values."""
    rows = []
    for cid, rev_factor, sp_factor in [
        (CAMP_A, CAMP_A_FACTOR, 10.0),
        (CAMP_B, CAMP_B_FACTOR, 20.0),
    ]:
        for i in range(n_days):
            rows.append(
                {
                    "platform":            "google" if cid == CAMP_A else "meta",
                    "campaign_id":         cid,
                    "campaign_name":       f"{cid}_name",
                    "date":                START_DATE + pd.Timedelta(days=i),
                    "spend":               float(i + 1) * 10.0,
                    "revenue_attributed":  float(i + 1) * rev_factor,
                    "clicks":              float(i + 1),
                    "impressions":         float(i + 1) * 100.0,
                    "conversions":         float(i + 1) * 0.1,
                    "daily_budget":        1000.0,
                    "channel_format":      "Search",
                    "reach":               0.0,
                    "video_views":         0.0,
                    "attribution_mature":  True,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def canon() -> pd.DataFrame:
    return _make_canonical()


@pytest.fixture(scope="module")
def lagged(canon: pd.DataFrame) -> pd.DataFrame:
    return add_lag_features(canon)


@pytest.fixture(scope="module")
def camp_a(lagged: pd.DataFrame) -> pd.DataFrame:
    return lagged[lagged["campaign_id"] == CAMP_A].sort_values("date").reset_index(drop=True)


@pytest.fixture(scope="module")
def camp_b(lagged: pd.DataFrame) -> pd.DataFrame:
    return lagged[lagged["campaign_id"] == CAMP_B].sort_values("date").reset_index(drop=True)


# ===========================================================================
# TestOutputSchema
# ===========================================================================

class TestOutputSchema:
    """All expected lag columns are present with correct types."""

    def test_all_lag_columns_present(self, lagged: pd.DataFrame) -> None:
        for col in LAG_COLUMNS:
            assert col in lagged.columns, f"Missing lag column: {col}"

    def test_lag_columns_are_float(self, lagged: pd.DataFrame) -> None:
        for col in LAG_COLUMNS:
            assert pd.api.types.is_float_dtype(lagged[col]), (
                f"{col} must be float dtype"
            )

    def test_row_count_preserved(self, canon: pd.DataFrame, lagged: pd.DataFrame) -> None:
        assert len(lagged) == len(canon)

    def test_original_columns_preserved(self, canon: pd.DataFrame, lagged: pd.DataFrame) -> None:
        for col in canon.columns:
            assert col in lagged.columns

    def test_sorted_by_campaign_then_date(self, lagged: pd.DataFrame) -> None:
        for cid, grp in lagged.groupby("campaign_id"):
            assert grp["date"].is_monotonic_increasing, (
                f"Campaign {cid} dates not sorted"
            )


# ===========================================================================
# TestRevenueLagCorrectness
# ===========================================================================

class TestRevenueLagCorrectness:
    """revenue_lag_n at row i equals revenue_attributed at row i-n."""

    def test_revenue_lag_1_at_row_1(self, camp_a: pd.DataFrame) -> None:
        # Row 1: today is day 1 (revenue 200), yesterday is day 0 (revenue 100)
        assert camp_a.loc[1, "revenue_lag_1"] == pytest.approx(1 * CAMP_A_FACTOR)

    def test_revenue_lag_3_at_row_3(self, camp_a: pd.DataFrame) -> None:
        # Row 3: day 3 → lag_3 = day 0 → revenue = 1 * factor
        assert camp_a.loc[3, "revenue_lag_3"] == pytest.approx(1 * CAMP_A_FACTOR)

    def test_revenue_lag_7_at_row_7(self, camp_a: pd.DataFrame) -> None:
        # Row 7: lag_7 = day 0 → revenue = 1 * factor
        assert camp_a.loc[7, "revenue_lag_7"] == pytest.approx(1 * CAMP_A_FACTOR)

    def test_revenue_lag_14_at_row_14(self, camp_a: pd.DataFrame) -> None:
        assert camp_a.loc[14, "revenue_lag_14"] == pytest.approx(1 * CAMP_A_FACTOR)

    def test_revenue_lag_28_at_row_28(self, camp_a: pd.DataFrame) -> None:
        assert camp_a.loc[28, "revenue_lag_28"] == pytest.approx(1 * CAMP_A_FACTOR)

    def test_revenue_lag_1_at_row_10(self, camp_a: pd.DataFrame) -> None:
        # Row 10: today is day 10, lag_1 = day 9 → revenue = 10 * factor
        assert camp_a.loc[10, "revenue_lag_1"] == pytest.approx(10 * CAMP_A_FACTOR)

    def test_revenue_lag_7_at_row_10(self, camp_a: pd.DataFrame) -> None:
        # Row 10: lag_7 = day 3 → revenue = 4 * factor
        assert camp_a.loc[10, "revenue_lag_7"] == pytest.approx(4 * CAMP_A_FACTOR)

    def test_revenue_lag_14_at_row_20(self, camp_a: pd.DataFrame) -> None:
        # Row 20: lag_14 = day 6 → revenue = 7 * factor
        assert camp_a.loc[20, "revenue_lag_14"] == pytest.approx(7 * CAMP_A_FACTOR)

    def test_revenue_lag_28_at_row_28(self, camp_a: pd.DataFrame) -> None:
        assert camp_a.loc[28, "revenue_lag_28"] == pytest.approx(1 * CAMP_A_FACTOR)


# ===========================================================================
# TestSpendLagCorrectness
# ===========================================================================

class TestSpendLagCorrectness:
    """spend_lag_n at row i equals spend at row i-n."""

    def test_spend_lag_1_at_row_1(self, camp_a: pd.DataFrame) -> None:
        assert camp_a.loc[1, "spend_lag_1"] == pytest.approx(10.0)  # day 0: 1*10

    def test_spend_lag_7_at_row_7(self, camp_a: pd.DataFrame) -> None:
        assert camp_a.loc[7, "spend_lag_7"] == pytest.approx(10.0)

    def test_spend_lag_28_at_row_28(self, camp_a: pd.DataFrame) -> None:
        assert camp_a.loc[28, "spend_lag_28"] == pytest.approx(10.0)


# ===========================================================================
# TestNaNBehavior
# ===========================================================================

class TestNaNBehavior:
    """First n rows per campaign must be NaN for lag-n features."""

    @pytest.mark.parametrize("n", [1, 3, 7, 14, 28])
    def test_revenue_lag_n_nan_for_first_n_rows(
        self, camp_a: pd.DataFrame, n: int
    ) -> None:
        assert camp_a.loc[: n - 1, f"revenue_lag_{n}"].isna().all(), (
            f"Expected first {n} rows of revenue_lag_{n} to be NaN"
        )

    @pytest.mark.parametrize("n", [1, 3, 7, 14, 28])
    def test_revenue_lag_n_not_nan_at_row_n(
        self, camp_a: pd.DataFrame, n: int
    ) -> None:
        assert not pd.isna(camp_a.loc[n, f"revenue_lag_{n}"]), (
            f"revenue_lag_{n} at row {n} should not be NaN"
        )

    @pytest.mark.parametrize("n", [1, 7, 14, 28])
    def test_clicks_lag_n_nan_for_first_n_rows(
        self, camp_a: pd.DataFrame, n: int
    ) -> None:
        assert camp_a.loc[: n - 1, f"clicks_lag_{n}"].isna().all()

    def test_row_0_all_lags_nan(self, camp_a: pd.DataFrame) -> None:
        """All lag features at row 0 (first day) must be NaN."""
        for col in LAG_COLUMNS:
            assert pd.isna(camp_a.loc[0, col]), (
                f"Expected {col} at row 0 to be NaN, got {camp_a.loc[0, col]}"
            )


# ===========================================================================
# TestCampaignIsolation
# ===========================================================================

class TestCampaignIsolation:
    """Lag values from camp_A must never bleed into camp_B and vice versa."""

    def test_camp_b_row_0_all_lags_nan(self, camp_b: pd.DataFrame) -> None:
        """First row of camp_B must be NaN regardless of camp_A data."""
        for col in LAG_COLUMNS:
            assert pd.isna(camp_b.loc[0, col]), (
                f"{col} at camp_B row 0 must be NaN, not camp_A's last value"
            )

    def test_camp_b_lag1_does_not_equal_camp_a_last_value(
        self, camp_a: pd.DataFrame, camp_b: pd.DataFrame
    ) -> None:
        # camp_A last revenue = 35 * 100 = 3500
        # camp_B row 0 revenue_lag_1 should be NaN, not 3500
        assert pd.isna(camp_b.loc[0, "revenue_lag_1"])

    def test_camp_a_values_distinct_from_camp_b(
        self, camp_a: pd.DataFrame, camp_b: pd.DataFrame
    ) -> None:
        # camp_B revenues are 2× camp_A; non-NaN lag_1 values should differ
        a_lag1 = camp_a["revenue_lag_1"].dropna()
        b_lag1 = camp_b["revenue_lag_1"].dropna()
        # camp_B revenue is 2× camp_A for the same day, so lag values differ
        for idx in a_lag1.index:
            if idx < len(b_lag1):
                assert a_lag1.iloc[idx] != b_lag1.iloc[idx]

    def test_two_campaigns_present(self, lagged: pd.DataFrame) -> None:
        assert lagged["campaign_id"].nunique() == 2

    def test_campaign_groups_have_correct_length(
        self, lagged: pd.DataFrame
    ) -> None:
        for cid, grp in lagged.groupby("campaign_id"):
            assert len(grp) == N_DAYS


# ===========================================================================
# TestNoFutureLeakage
# ===========================================================================

class TestNoFutureLeakage:
    """Lag features must never include same-day or future values."""

    def test_revenue_lag_1_never_equals_same_day_revenue(
        self, camp_a: pd.DataFrame
    ) -> None:
        # revenue_attributed increases monotonically; lag_1 is always smaller
        valid = camp_a.dropna(subset=["revenue_lag_1"])
        assert (valid["revenue_lag_1"] < valid["revenue_attributed"]).all(), (
            "revenue_lag_1 should always be < revenue_attributed for increasing series"
        )

    def test_spend_lag_1_never_equals_same_day_spend(
        self, camp_a: pd.DataFrame
    ) -> None:
        valid = camp_a.dropna(subset=["spend_lag_1"])
        assert (valid["spend_lag_1"] < valid["spend"]).all()

    def test_revenue_lag_7_at_row_6_is_nan(self, camp_a: pd.DataFrame) -> None:
        assert pd.isna(camp_a.loc[6, "revenue_lag_7"])

    def test_revenue_lag_28_at_row_27_is_nan(self, camp_a: pd.DataFrame) -> None:
        assert pd.isna(camp_a.loc[27, "revenue_lag_28"])


# ===========================================================================
# TestCustomLagSpec
# ===========================================================================

class TestCustomLagSpec:
    """add_lag_features supports arbitrary custom lag specifications."""

    def test_custom_spec_single_metric(self, canon: pd.DataFrame) -> None:
        custom = {"revenue_attributed": (1, 2)}
        result = add_lag_features(canon, lag_spec=custom)
        assert "revenue_lag_1" in result.columns
        assert "revenue_lag_2" in result.columns
        assert "revenue_lag_3" not in result.columns

    def test_custom_spec_produces_correct_values(self, canon: pd.DataFrame) -> None:
        custom = {"spend": (2,)}
        result = add_lag_features(canon, lag_spec=custom)
        camp = result[result["campaign_id"] == CAMP_A].sort_values("date")
        # Row 2: lag_2 = day 0 = 1*10 = 10.0
        assert camp.iloc[2]["spend_lag_2"] == pytest.approx(10.0)


# ===========================================================================
# TestValidation
# ===========================================================================

class TestValidation:
    """Input validation raises LagFeatureError for bad inputs."""

    def test_missing_campaign_id_raises(self, canon: pd.DataFrame) -> None:
        bad = canon.drop(columns=["campaign_id"])
        with pytest.raises(LagFeatureError, match="campaign_id"):
            add_lag_features(bad)

    def test_missing_date_raises(self, canon: pd.DataFrame) -> None:
        bad = canon.drop(columns=["date"])
        with pytest.raises(LagFeatureError, match="date"):
            add_lag_features(bad)

    def test_missing_source_column_raises(self, canon: pd.DataFrame) -> None:
        bad = canon.drop(columns=["revenue_attributed"])
        with pytest.raises(LagFeatureError):
            add_lag_features(bad)

    def test_empty_dataframe_raises(self) -> None:
        empty = _make_canonical().iloc[0:0]  # correct columns, zero rows
        with pytest.raises(LagFeatureError, match="empty"):
            add_lag_features(empty)
