"""
tests/test_feature_store.py
============================
Pytest suite for src/features/feature_store.py.

Two fixture tiers:
 - Synthetic: small deterministic DataFrames for unit tests of each feature.
 - Integration: real data loaded from dataset/ once per module (slow path).
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
import pytest

from src.data.harmonizer import harmonize_from_dir
from src.data.taxonomy_parser import parse_taxonomy
from src.features.feature_store import (
    BUDGET_FEATURE_COLUMNS,
    CALENDAR_FEATURE_COLUMNS,
    HOLIDAY_JOIN_COLUMNS,
    ROAS_LAG_COLUMNS,
    TAXONOMY_JOIN_COLUMNS,
    TARGET_COLUMN,
    FeatureStoreError,
    build_feature_store,
)
from src.features.holiday_calendar import build_holiday_calendar

# ---------------------------------------------------------------------------
# Forensic constants (from test_data_layer.py)
# ---------------------------------------------------------------------------

TOTAL_CANONICAL_ROWS: Final[int] = 25_562
GOOGLE_BUDGET_NULLS: Final[int] = 14
META_BUDGET_NULLS: Final[int] = 7
TOTAL_BUDGET_NULLS: Final[int] = GOOGLE_BUDGET_NULLS + META_BUDGET_NULLS

# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

N_DAYS: Final[int] = 35
START_DATE: Final[pd.Timestamp] = pd.Timestamp("2024-01-01")


def _make_synthetic_canon(n_days: int = N_DAYS) -> pd.DataFrame:
    """Two campaigns × n_days, deterministic revenue/spend."""
    rows = []
    for cid, platform, rev_f, sp_f in [
        ("camp_A", "google", 100.0, 10.0),
        ("camp_B", "meta",   200.0, 20.0),
    ]:
        for i in range(n_days):
            rows.append(
                {
                    "platform":           platform,
                    "campaign_id":        cid,
                    "campaign_name":      f"{cid}_Campaign_1",
                    "date":               START_DATE + pd.Timedelta(days=i),
                    "spend":              float(i + 1) * sp_f,
                    "revenue_attributed": float(i + 1) * rev_f,
                    "clicks":             float(i + 1),
                    "impressions":        float(i + 1) * 100.0,
                    "conversions":        float(i + 1) * 0.1,
                    "daily_budget":       500.0,
                    "channel_format":     "Search",
                    "reach":              0.0,
                    "video_views":        0.0,
                    "attribution_mature": True,
                }
            )
    return pd.DataFrame(rows)


def _make_synthetic_taxonomy(canon: pd.DataFrame) -> pd.DataFrame:
    """Minimal taxonomy matching the synthetic campaigns."""
    rows = []
    for _, row in canon[["platform", "campaign_id", "campaign_name", "channel_format"]].drop_duplicates().iterrows():
        rows.append(
            {
                "platform":              row["platform"],
                "campaign_id":           row["campaign_id"],
                "campaign_name":         row["campaign_name"],
                "channel_format":        row["channel_format"],
                "format":                "Search",
                "audience_strategy":     "NTM",
                "funnel_stage":          None,
                "ad_product_type":       None,
                "campaign_number":       1,
                "strategy_key":          f"{row['platform']}_Search_NTM",
                "cross_engine_pair_flag": False,
                "is_brand":              False,
                "is_non_brand":          True,
                "is_upper_funnel":       False,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def syn_canon() -> pd.DataFrame:
    return _make_synthetic_canon()


@pytest.fixture(scope="module")
def syn_taxonomy(syn_canon: pd.DataFrame) -> pd.DataFrame:
    return _make_synthetic_taxonomy(syn_canon)


@pytest.fixture(scope="module")
def syn_calendar() -> pd.DataFrame:
    return build_holiday_calendar(2024, 2024)


@pytest.fixture(scope="module")
def syn_fs(syn_canon, syn_taxonomy, syn_calendar) -> pd.DataFrame:
    return build_feature_store(
        canon=syn_canon,
        taxonomy=syn_taxonomy,
        calendar=syn_calendar,
    )


# Integration fixtures — load real data once
@pytest.fixture(scope="module")
def real_canon() -> pd.DataFrame:
    return harmonize_from_dir("dataset")


@pytest.fixture(scope="module")
def real_taxonomy(real_canon: pd.DataFrame) -> pd.DataFrame:
    return parse_taxonomy(real_canon)


@pytest.fixture(scope="module")
def real_calendar() -> pd.DataFrame:
    return build_holiday_calendar(2024, 2027)


@pytest.fixture(scope="module")
def real_fs(real_canon, real_taxonomy, real_calendar) -> pd.DataFrame:
    return build_feature_store(
        canon=real_canon,
        taxonomy=real_taxonomy,
        calendar=real_calendar,
    )


# ===========================================================================
# TestOutputSchema
# ===========================================================================

class TestOutputSchema:
    """Feature store must have expected shape and column set."""

    def test_row_count_equals_input_rows(
        self, syn_canon: pd.DataFrame, syn_fs: pd.DataFrame
    ) -> None:
        assert len(syn_fs) == len(syn_canon)

    def test_calendar_feature_columns_present(self, syn_fs: pd.DataFrame) -> None:
        for col in CALENDAR_FEATURE_COLUMNS:
            assert col in syn_fs.columns, f"Missing calendar feature: {col}"

    def test_taxonomy_join_columns_present(self, syn_fs: pd.DataFrame) -> None:
        for col in TAXONOMY_JOIN_COLUMNS:
            assert col in syn_fs.columns, f"Missing taxonomy feature: {col}"

    def test_holiday_join_columns_present(self, syn_fs: pd.DataFrame) -> None:
        for col in HOLIDAY_JOIN_COLUMNS:
            assert col in syn_fs.columns, f"Missing holiday feature: {col}"

    def test_budget_feature_columns_present(self, syn_fs: pd.DataFrame) -> None:
        for col in BUDGET_FEATURE_COLUMNS:
            assert col in syn_fs.columns, f"Missing budget feature: {col}"

    def test_roas_lag_columns_present(self, syn_fs: pd.DataFrame) -> None:
        for col in ROAS_LAG_COLUMNS:
            assert col in syn_fs.columns, f"Missing ROAS lag: {col}"

    def test_target_column_present(self, syn_fs: pd.DataFrame) -> None:
        assert TARGET_COLUMN in syn_fs.columns

    def test_no_null_campaign_id(self, syn_fs: pd.DataFrame) -> None:
        assert syn_fs["campaign_id"].notna().all()

    def test_no_null_date(self, syn_fs: pd.DataFrame) -> None:
        assert syn_fs["date"].notna().all()

    def test_sorted_by_campaign_then_date(self, syn_fs: pd.DataFrame) -> None:
        for cid, grp in syn_fs.groupby("campaign_id"):
            assert grp["date"].is_monotonic_increasing


# ===========================================================================
# TestNoDuplicateRows
# ===========================================================================

class TestNoDuplicateRows:
    """One row per (campaign_id, date) — no duplicates."""

    def test_no_duplicate_rows_synthetic(self, syn_fs: pd.DataFrame) -> None:
        dupes = syn_fs.duplicated(subset=["campaign_id", "date"]).sum()
        assert dupes == 0, f"Found {dupes} duplicate (campaign_id, date) pairs"

    def test_no_duplicate_rows_real(self, real_fs: pd.DataFrame) -> None:
        dupes = real_fs.duplicated(subset=["campaign_id", "date"]).sum()
        assert dupes == 0, f"Found {dupes} duplicate (campaign_id, date) pairs"


# ===========================================================================
# TestRealDataRowCount
# ===========================================================================

class TestRealDataRowCount:
    """Feature store must have exactly the same number of rows as canonical."""

    def test_real_row_count(
        self, real_canon: pd.DataFrame, real_fs: pd.DataFrame
    ) -> None:
        assert len(real_fs) == len(real_canon) == TOTAL_CANONICAL_ROWS


# ===========================================================================
# TestCalendarFeatures
# ===========================================================================

class TestCalendarFeatures:

    def test_day_of_week_range(self, syn_fs: pd.DataFrame) -> None:
        assert syn_fs["day_of_week"].between(0, 6).all()

    def test_day_of_month_range(self, syn_fs: pd.DataFrame) -> None:
        assert syn_fs["day_of_month"].between(1, 31).all()

    def test_month_range(self, syn_fs: pd.DataFrame) -> None:
        assert syn_fs["month"].between(1, 12).all()

    def test_quarter_range(self, syn_fs: pd.DataFrame) -> None:
        assert syn_fs["quarter"].between(1, 4).all()

    def test_week_of_year_range(self, syn_fs: pd.DataFrame) -> None:
        assert syn_fs["week_of_year"].between(1, 53).all()

    def test_is_weekend_is_bool(self, syn_fs: pd.DataFrame) -> None:
        assert syn_fs["is_weekend"].dtype == bool

    def test_is_weekend_correct_for_known_date(self, syn_fs: pd.DataFrame) -> None:
        # 2024-01-01 is a Monday → is_weekend = False
        row = syn_fs[syn_fs["date"] == pd.Timestamp("2024-01-01")]
        assert row["is_weekend"].iloc[0] == False

    def test_is_weekend_correct_for_saturday(self, syn_fs: pd.DataFrame) -> None:
        # 2024-01-06 is a Saturday → is_weekend = True
        row = syn_fs[syn_fs["date"] == pd.Timestamp("2024-01-06")]
        if len(row) > 0:
            assert row["is_weekend"].iloc[0] == True

    def test_day_of_week_correct_for_known_date(self, syn_fs: pd.DataFrame) -> None:
        # 2024-01-01 is Monday → dayofweek = 0
        row = syn_fs[syn_fs["date"] == pd.Timestamp("2024-01-01")]
        assert row["day_of_week"].iloc[0] == 0

    def test_year_correct(self, syn_fs: pd.DataFrame) -> None:
        assert (syn_fs["year"] == 2024).all()


# ===========================================================================
# TestBudgetUtilizationFeatures
# ===========================================================================

class TestBudgetUtilizationFeatures:

    def test_budget_utilization_correctness(self, syn_fs: pd.DataFrame) -> None:
        """budget_utilization = spend / daily_budget."""
        camp = syn_fs[syn_fs["campaign_id"] == "camp_A"].sort_values("date")
        expected = camp["spend"] / camp["daily_budget"]
        pd.testing.assert_series_equal(
            camp["budget_utilization"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_budget_headroom_correctness(self, syn_fs: pd.DataFrame) -> None:
        """budget_headroom = daily_budget - spend."""
        camp = syn_fs[syn_fs["campaign_id"] == "camp_A"].sort_values("date")
        expected = camp["daily_budget"] - camp["spend"]
        pd.testing.assert_series_equal(
            camp["budget_headroom"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_budget_utilization_day_0_camp_a(self, syn_fs: pd.DataFrame) -> None:
        # Day 0: spend = 10, daily_budget = 500 → utilization = 0.02
        row = syn_fs[
            (syn_fs["campaign_id"] == "camp_A") &
            (syn_fs["date"] == pd.Timestamp("2024-01-01"))
        ]
        assert row["budget_utilization"].iloc[0] == pytest.approx(10.0 / 500.0)

    def test_budget_nan_rows_in_real_data(self, real_fs: pd.DataFrame) -> None:
        """Known NaN budget rows produce NaN utilization, not errors."""
        n_nan_util = real_fs["budget_utilization"].isna().sum()
        # Must have at least as many NaN as known NaN budgets
        assert n_nan_util >= TOTAL_BUDGET_NULLS

    def test_budget_utilization_non_negative(self, syn_fs: pd.DataFrame) -> None:
        valid = syn_fs["budget_utilization"].dropna()
        assert (valid >= 0).all()

    def test_budget_headroom_finite(self, syn_fs: pd.DataFrame) -> None:
        valid = syn_fs["budget_headroom"].dropna()
        assert np.isfinite(valid).all()


# ===========================================================================
# TestHolidayMerge
# ===========================================================================

class TestHolidayMerge:

    def test_is_holiday_is_bool_type(self, syn_fs: pd.DataFrame) -> None:
        assert syn_fs["is_holiday"].dtype == bool

    def test_is_bfcm_is_bool_type(self, syn_fs: pd.DataFrame) -> None:
        assert syn_fs["is_bfcm"].dtype == bool

    def test_is_holiday_season_is_bool_type(self, syn_fs: pd.DataFrame) -> None:
        assert syn_fs["is_holiday_season"].dtype == bool

    def test_days_to_black_friday_nonnegative(self, syn_fs: pd.DataFrame) -> None:
        assert (syn_fs["days_to_black_friday"] >= 0).all()

    def test_days_since_black_friday_nonnegative(self, syn_fs: pd.DataFrame) -> None:
        assert (syn_fs["days_since_black_friday"] >= 0).all()

    def test_holiday_intensity_score_bounded(self, syn_fs: pd.DataFrame) -> None:
        assert (syn_fs["holiday_intensity_score"] >= 0.0).all()
        assert (syn_fs["holiday_intensity_score"] <= 1.0).all()

    def test_new_years_day_is_holiday_in_feature_store(
        self, syn_fs: pd.DataFrame
    ) -> None:
        row = syn_fs[syn_fs["date"] == pd.Timestamp("2024-01-01")]
        assert row["is_holiday"].iloc[0] == True

    def test_regular_day_not_holiday(self, syn_fs: pd.DataFrame) -> None:
        # 2024-01-03 is a Wednesday, not a federal holiday
        row = syn_fs[syn_fs["date"] == pd.Timestamp("2024-01-03")]
        assert row["is_holiday"].iloc[0] == False

    def test_no_null_holiday_features_after_merge(self, syn_fs: pd.DataFrame) -> None:
        for col in HOLIDAY_JOIN_COLUMNS:
            assert syn_fs[col].notna().all(), f"{col} has unexpected NaN after merge"

    def test_holiday_merge_correct_in_real_data(self, real_fs: pd.DataFrame) -> None:
        """No NaN in holiday columns for the real dataset."""
        for col in HOLIDAY_JOIN_COLUMNS:
            assert real_fs[col].notna().all(), f"Real data: {col} has NaN"


# ===========================================================================
# TestTaxonomyMerge
# ===========================================================================

class TestTaxonomyMerge:

    def test_taxonomy_columns_present(self, syn_fs: pd.DataFrame) -> None:
        for col in TAXONOMY_JOIN_COLUMNS:
            assert col in syn_fs.columns

    def test_is_brand_dtype_bool(self, syn_fs: pd.DataFrame) -> None:
        assert syn_fs["is_brand"].dtype == bool

    def test_is_non_brand_dtype_bool(self, syn_fs: pd.DataFrame) -> None:
        assert syn_fs["is_non_brand"].dtype == bool

    def test_is_upper_funnel_dtype_bool(self, syn_fs: pd.DataFrame) -> None:
        assert syn_fs["is_upper_funnel"].dtype == bool

    def test_strategy_key_not_null_after_merge(self, syn_fs: pd.DataFrame) -> None:
        assert syn_fs["strategy_key"].notna().all()

    def test_format_is_string(self, syn_fs: pd.DataFrame) -> None:
        assert syn_fs["format"].notna().all()

    def test_taxonomy_merge_real_data(self, real_fs: pd.DataFrame) -> None:
        """All 136 campaigns should get a taxonomy match."""
        missing = real_fs["strategy_key"].isna().sum()
        assert missing == 0, f"{missing} rows have no taxonomy match in real data"

    def test_cross_engine_pair_flag_dtype_bool_real(
        self, real_fs: pd.DataFrame
    ) -> None:
        assert real_fs["cross_engine_pair_flag"].dtype == bool

    def test_brand_campaigns_have_is_brand_true_real(
        self, real_fs: pd.DataFrame
    ) -> None:
        brand_rows = real_fs[real_fs["is_brand"] == True]
        assert len(brand_rows) > 0

    def test_non_brand_campaigns_have_is_non_brand_true_real(
        self, real_fs: pd.DataFrame
    ) -> None:
        nb_rows = real_fs[real_fs["is_non_brand"] == True]
        assert len(nb_rows) > 0


# ===========================================================================
# TestROASLagFeatures
# ===========================================================================

class TestROASLagFeatures:

    def test_roas_lag_7_at_row_7_camp_a(self, syn_fs: pd.DataFrame) -> None:
        # Row 7, camp_A:
        #   revenue_lag_7 = day 0 = 100.0, spend_lag_7 = day 0 = 10.0
        #   roas_lag_7 = 100/10 = 10.0
        camp = syn_fs[syn_fs["campaign_id"] == "camp_A"].sort_values("date").reset_index(drop=True)
        assert camp.loc[7, "roas_lag_7"] == pytest.approx(10.0)

    def test_roas_lag_columns_are_float(self, syn_fs: pd.DataFrame) -> None:
        for col in ROAS_LAG_COLUMNS:
            assert pd.api.types.is_float_dtype(syn_fs[col])

    def test_roas_lag_7_nan_at_row_6(self, syn_fs: pd.DataFrame) -> None:
        camp = syn_fs[syn_fs["campaign_id"] == "camp_A"].sort_values("date").reset_index(drop=True)
        assert pd.isna(camp.loc[6, "roas_lag_7"])

    def test_roas_uses_lagged_values_only(self, syn_fs: pd.DataFrame) -> None:
        # roas_lag_7 uses revenue_lag_7 and spend_lag_7, not today's values
        camp = syn_fs[syn_fs["campaign_id"] == "camp_A"].sort_values("date").reset_index(drop=True)
        for idx in range(7, N_DAYS):
            expected = camp.loc[idx, "revenue_lag_7"] / camp.loc[idx, "spend_lag_7"]
            assert camp.loc[idx, "roas_lag_7"] == pytest.approx(expected, rel=1e-5)

    def test_roas_zero_spend_produces_nan(self) -> None:
        """Zero lagged spend must yield NaN, not inf."""
        rows = [
            {
                "campaign_id": "z", "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "spend": 0.0, "revenue_attributed": float(i) * 100,
                "clicks": 1.0, "impressions": 100.0, "conversions": 0.1,
                "daily_budget": 500.0, "channel_format": "Search",
                "reach": 0.0, "video_views": 0.0, "attribution_mature": True,
                "platform": "google", "campaign_name": "z_Campaign_1",
            }
            for i in range(35)
        ]
        canon = pd.DataFrame(rows)
        tax = _make_synthetic_taxonomy(canon)
        cal = build_holiday_calendar(2024, 2024)
        fs = build_feature_store(canon=canon, taxonomy=tax, calendar=cal)
        roas_vals = fs["roas_lag_7"].dropna()
        assert np.isfinite(roas_vals).all() or len(roas_vals) == 0


# ===========================================================================
# TestNoLeakage
# ===========================================================================

class TestNoLeakage:
    """Revenue target must not appear in any feature value."""

    def test_lag_1_never_equals_same_day_revenue(
        self, syn_fs: pd.DataFrame
    ) -> None:
        camp = syn_fs[syn_fs["campaign_id"] == "camp_A"].sort_values("date")
        valid = camp.dropna(subset=["revenue_lag_1"])
        assert (valid["revenue_lag_1"] < valid["revenue_attributed"]).all()

    def test_roll_mean_7_never_equals_same_day_revenue(
        self, syn_fs: pd.DataFrame
    ) -> None:
        camp = syn_fs[syn_fs["campaign_id"] == "camp_A"].sort_values("date")
        valid = camp.iloc[7:].dropna(subset=["revenue_roll_mean_7"])
        assert (valid["revenue_roll_mean_7"] < valid["revenue_attributed"]).all()

    def test_roas_lag_uses_past_values_only(self, syn_fs: pd.DataFrame) -> None:
        # Verify roas_lag_7 equals revenue_lag_7 / spend_lag_7 (not today's values)
        camp = syn_fs[syn_fs["campaign_id"] == "camp_A"].sort_values("date").reset_index(drop=True)
        for idx in range(7, N_DAYS):
            roas = camp.loc[idx, "roas_lag_7"]
            rev_lag = camp.loc[idx, "revenue_lag_7"]
            sp_lag = camp.loc[idx, "spend_lag_7"]
            if not pd.isna(roas) and sp_lag != 0:
                assert roas == pytest.approx(rev_lag / sp_lag, rel=1e-5)


# ===========================================================================
# TestValidation
# ===========================================================================

class TestValidation:

    def test_build_feature_store_returns_dataframe(self, syn_fs: pd.DataFrame) -> None:
        assert isinstance(syn_fs, pd.DataFrame)

    def test_real_feature_store_returns_dataframe(self, real_fs: pd.DataFrame) -> None:
        assert isinstance(real_fs, pd.DataFrame)

    def test_real_feature_store_column_count(self, real_fs: pd.DataFrame) -> None:
        # Must have substantially more columns than the canonical 14
        assert len(real_fs.columns) > 50

    def test_real_feature_store_has_target(self, real_fs: pd.DataFrame) -> None:
        assert TARGET_COLUMN in real_fs.columns
        assert real_fs[TARGET_COLUMN].notna().any()
