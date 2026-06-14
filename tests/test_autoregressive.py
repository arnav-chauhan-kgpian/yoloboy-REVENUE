"""
tests/test_autoregressive.py
==============================
Verifies the autoregressive future forecast rollout in
src/models/autoregressive.py.

Key properties tested
---------------------
1.  Output shape — n_campaigns × n_future_days rows.
2.  All is_future = True.
3.  All revenue_attributed is NaN.
4.  P10 ≤ P50 ≤ P90 for every row (quantile ordering).
5.  The lag cascade — revenue_lag_1 on day T+2 equals the P50 returned for
    day T+1 (the most important correctness property).
6.  Lag-7 cascade — revenue_lag_7 on day T+8 equals the P50 for day T+1.
7.  Rolling mean update — revenue_roll_mean_7 on day T+8 is the mean of the
    seven P50 predictions from T+1..T+7 (all predicted, no actuals in window).
8.  Uncertainty growth — the aggregate interval width on day 14 is at least
    as wide as on day 1 (predictions compound; uncertainty cannot shrink
    monotonically when features degrade).
9.  Calendar features are updated correctly for each future date.
10. Empty/degenerate input handling.
11. Smoke test with a realistic-looking feature store.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.autoregressive import (
    _build_actual_buffer,
    _lookup,
    _window_values,
    _update_calendar_features,
    _update_revenue_features,
    generate_future_forecasts,
)


# ---------------------------------------------------------------------------
# Test utilities / mocks
# ---------------------------------------------------------------------------

class _ConstantModel:
    """Returns constant P10/P50/P90 regardless of input features."""

    def __init__(self, p10: float = 80.0, p50: float = 100.0, p90: float = 120.0):
        self.p10 = p10
        self.p50 = p50
        self.p90 = p90

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        n = len(X)
        return pd.DataFrame(
            {"p10": np.full(n, self.p10), "p50": np.full(n, self.p50), "p90": np.full(n, self.p90)},
        )


class _RecordingModel:
    """Returns constant predictions and records every feature DataFrame it receives."""

    def __init__(self, p50: float = 500.0, spread_pct: float = 0.10):
        self._p50 = p50
        self._spread = spread_pct
        self.calls: list[pd.DataFrame] = []

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        self.calls.append(X.copy())
        n = len(X)
        return pd.DataFrame({
            "p10": np.full(n, self._p50 * (1.0 - self._spread)),
            "p50": np.full(n, self._p50),
            "p90": np.full(n, self._p50 * (1.0 + self._spread)),
        })


class _GrowingUncertaintyModel:
    """Spreads widen with call index — simulates compounding error."""

    def __init__(self):
        self.call_index = 0

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        self.call_index += 1
        n = len(X)
        spread = 50.0 + self.call_index * 10.0
        return pd.DataFrame({
            "p10": np.full(n, 500.0 - spread),
            "p50": np.full(n, 500.0),
            "p90": np.full(n, 500.0 + spread),
        })


RNG = np.random.default_rng(31415)


def _make_feature_store(
    n_campaigns: int = 3,
    n_days: int = 90,
    base_revenue: float = 300.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a minimal but structurally correct feature store for testing."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for cid in range(n_campaigns):
        platform = ["google", "meta", "bing"][cid % 3]
        for i, date in enumerate(dates):
            rev = base_revenue + cid * 50.0 + RNG.normal(0, 20)
            rev = max(rev, 0.0)
            spend = rev / 3.0 + RNG.normal(0, 10)
            spend = max(spend, 1.0)
            rows.append({
                "campaign_id":             f"camp_{cid}",
                "campaign_name":           f"Campaign {cid}",
                "platform":                platform,
                "date":                    date,
                "spend":                   spend,
                "revenue_attributed":      rev,
                "attribution_mature":      i < n_days - 14,
                # Calendar
                "day_of_week":             date.dayofweek,
                "day_of_month":            date.day,
                "week_of_year":            date.isocalendar()[1],
                "month":                   date.month,
                "quarter":                 date.quarter,
                "year":                    date.year,
                "is_weekend":              int(date.dayofweek >= 5),
                # Revenue lags
                "revenue_lag_1":           rev * 0.98 + RNG.normal(0, 5),
                "revenue_lag_3":           rev * 0.96 + RNG.normal(0, 5),
                "revenue_lag_7":           rev * 0.94 + RNG.normal(0, 5),
                "revenue_lag_14":          rev * 0.91 + RNG.normal(0, 5),
                "revenue_lag_28":          rev * 0.88 + RNG.normal(0, 5),
                # Spend lags
                "spend_lag_1":             spend * 0.98,
                "spend_lag_3":             spend * 0.96,
                "spend_lag_7":             spend * 0.94,
                "spend_lag_14":            spend * 0.91,
                "spend_lag_28":            spend * 0.88,
                # Rolling revenue
                "revenue_roll_mean_7":     rev * 0.97,
                "revenue_roll_mean_14":    rev * 0.96,
                "revenue_roll_mean_28":    rev * 0.95,
                "revenue_roll_std_7":      rev * 0.10,
                "revenue_roll_std_14":     rev * 0.11,
                "revenue_roll_std_28":     rev * 0.12,
                "revenue_roll_min_7":      rev * 0.80,
                "revenue_roll_max_7":      rev * 1.20,
                # Rolling spend
                "spend_roll_mean_7":       spend * 0.97,
                "spend_roll_mean_14":      spend * 0.96,
                "spend_roll_mean_28":      spend * 0.95,
                "spend_roll_std_7":        spend * 0.08,
                "spend_roll_std_14":       spend * 0.09,
                "spend_roll_std_28":       spend * 0.10,
                # Momentum
                "revenue_momentum_7":      1.01,
                "revenue_momentum_14":     1.00,
                "revenue_momentum_28":     0.99,
                "spend_momentum_7":        1.01,
                "spend_momentum_14":       1.00,
                "spend_momentum_28":       0.99,
                # ROAS lags
                "roas_lag_7":              rev / max(spend, 1.0),
                "roas_lag_14":             rev / max(spend, 1.0) * 0.98,
                "roas_lag_28":             rev / max(spend, 1.0) * 0.96,
            })

    fs = pd.DataFrame(rows)
    exclude = {
        "revenue_attributed", "date", "campaign_id", "campaign_name",
        "platform", "attribution_mature",
    }
    feature_cols = [c for c in fs.columns if c not in exclude]
    return fs, feature_cols


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------

class TestLookup:
    def test_predicted_takes_precedence(self):
        ts = pd.Timestamp("2024-03-01")
        actual    = {ts: 100.0}
        predicted = {ts: 200.0}
        assert _lookup(ts, actual, predicted) == 200.0

    def test_falls_back_to_actual(self):
        ts = pd.Timestamp("2024-03-01")
        actual    = {ts: 100.0}
        predicted = {}
        assert _lookup(ts, actual, predicted) == 100.0

    def test_returns_nan_when_absent(self):
        ts = pd.Timestamp("2024-03-01")
        assert np.isnan(_lookup(ts, {}, {}))


class TestWindowValues:
    def test_collects_lookback_values(self):
        origin = pd.Timestamp("2024-03-08")
        actual = {
            pd.Timestamp("2024-03-07"): 1.0,
            pd.Timestamp("2024-03-06"): 2.0,
            pd.Timestamp("2024-03-05"): 3.0,
        }
        vals = _window_values(origin, window=3, actual=actual, predicted={})
        assert vals == [1.0, 2.0, 3.0]

    def test_predicted_replaces_actual_in_window(self):
        origin = pd.Timestamp("2024-03-08")
        actual    = {pd.Timestamp("2024-03-07"): 1.0}
        predicted = {pd.Timestamp("2024-03-07"): 99.0}
        vals = _window_values(origin, window=1, actual=actual, predicted=predicted)
        assert vals == [99.0]

    def test_missing_dates_excluded(self):
        origin = pd.Timestamp("2024-03-08")
        actual = {pd.Timestamp("2024-03-06"): 5.0}   # only day -2 present
        vals = _window_values(origin, window=3, actual=actual, predicted={})
        assert vals == [5.0]   # day -1 and day -3 are NaN → excluded


class TestUpdateCalendarFeatures:
    def test_all_calendar_fields_updated(self):
        row = pd.Series({
            "date": pd.Timestamp("2024-01-01"),
            "day_of_week": 0, "day_of_month": 1, "week_of_year": 1,
            "month": 1, "quarter": 1, "year": 2024, "is_weekend": 0,
        })
        new_date = pd.Timestamp("2024-03-15")  # Friday
        updated = _update_calendar_features(row, new_date)

        assert updated["date"]         == new_date
        assert updated["day_of_week"]  == 4       # Friday
        assert updated["day_of_month"] == 15
        assert updated["month"]        == 3
        assert updated["quarter"]      == 1
        assert updated["year"]         == 2024
        assert updated["is_weekend"]   == 0

    def test_weekend_flag_set_correctly(self):
        row = pd.Series({"is_weekend": 0, "day_of_week": 0, "day_of_month": 1,
                         "week_of_year": 1, "month": 1, "quarter": 1, "year": 2024})
        saturday = pd.Timestamp("2024-03-16")
        updated = _update_calendar_features(row, saturday)
        assert updated["is_weekend"] == 1

    def test_missing_calendar_cols_ignored(self):
        row = pd.Series({"some_other_col": 42.0})
        updated = _update_calendar_features(row, pd.Timestamp("2024-03-15"))
        assert "some_other_col" in updated.index


class TestUpdateRevenueFeatures:
    def _base_row(self) -> pd.Series:
        """A row with all revenue-derived columns present."""
        return pd.Series({
            "revenue_lag_1":       300.0,
            "revenue_lag_3":       295.0,
            "revenue_lag_7":       290.0,
            "revenue_lag_14":      285.0,
            "revenue_lag_28":      280.0,
            "revenue_roll_mean_7": 298.0,
            "revenue_roll_mean_14":296.0,
            "revenue_roll_mean_28":294.0,
            "revenue_roll_std_7":  20.0,
            "revenue_roll_std_14": 21.0,
            "revenue_roll_std_28": 22.0,
            "revenue_roll_min_7":  270.0,
            "revenue_roll_max_7":  330.0,
            "revenue_momentum_7":  1.01,
            "revenue_momentum_14": 1.00,
            "revenue_momentum_28": 0.99,
            "spend_lag_7":         100.0,
            "spend_lag_14":        98.0,
            "spend_lag_28":        96.0,
            "roas_lag_7":          3.0,
            "roas_lag_14":         2.9,
            "roas_lag_28":         2.8,
        })

    def test_lag_1_updated_from_predicted(self):
        row       = self._base_row()
        origin    = pd.Timestamp("2024-03-08")
        predicted = {pd.Timestamp("2024-03-07"): 999.0}
        updated   = _update_revenue_features(row, origin, {}, predicted)
        assert updated["revenue_lag_1"] == pytest.approx(999.0)

    def test_lag_7_updated_from_predicted(self):
        row       = self._base_row()
        origin    = pd.Timestamp("2024-03-15")  # T+7 relative to T=2024-03-08
        predicted = {pd.Timestamp("2024-03-08"): 777.0}
        updated   = _update_revenue_features(row, origin, {}, predicted)
        assert updated["revenue_lag_7"] == pytest.approx(777.0)

    def test_roll_mean_7_uses_window(self):
        row    = self._base_row()
        origin = pd.Timestamp("2024-03-15")
        actual = {
            pd.Timestamp("2024-03-14"): 100.0,
            pd.Timestamp("2024-03-13"): 100.0,
            pd.Timestamp("2024-03-12"): 100.0,
            pd.Timestamp("2024-03-11"): 100.0,
            pd.Timestamp("2024-03-10"): 100.0,
            pd.Timestamp("2024-03-09"): 100.0,
            pd.Timestamp("2024-03-08"): 100.0,
        }
        updated = _update_revenue_features(row, origin, actual, {})
        assert updated["revenue_roll_mean_7"] == pytest.approx(100.0)

    def test_momentum_computed_from_updated_lag1_and_roll_mean(self):
        row    = self._base_row()
        origin = pd.Timestamp("2024-03-08")
        # Set lag_1 = 200, roll_mean_7 = 100 → momentum_7 should be 2.0
        actual    = {pd.Timestamp("2024-03-07"): 200.0}
        predicted = {}
        # Fill window with 100.0 for 7 dates
        for delta in range(1, 8):
            d = origin - pd.Timedelta(days=delta)
            actual[d] = 100.0
        actual[pd.Timestamp("2024-03-07")] = 200.0  # lag_1
        updated = _update_revenue_features(row, origin, actual, predicted)
        # lag_1 = 200; roll_mean_7 = mean of [200, 100, 100, 100, 100, 100, 100] = 100/7*6 + 200/7
        # Actually: window is dates at offset 1..7 from origin = 2024-03-07..2024-03-01
        # lag_1 is date at offset 1 = 2024-03-07 = 200
        # roll_mean_7 uses offsets 1..7, so also includes 200
        expected_roll_mean = np.mean([200, 100, 100, 100, 100, 100, 100])
        assert updated["revenue_roll_mean_7"] == pytest.approx(expected_roll_mean, rel=1e-5)
        assert updated["revenue_momentum_7"] == pytest.approx(200.0 / expected_roll_mean, rel=1e-5)

    def test_roas_lag_updated_from_new_revenue_lag(self):
        row    = self._base_row()
        origin = pd.Timestamp("2024-03-15")
        # Set revenue at T-7 = 700, spend_lag_7 stays at 100 → roas_lag_7 = 7.0
        predicted = {pd.Timestamp("2024-03-08"): 700.0}
        updated   = _update_revenue_features(row, origin, {}, predicted)
        assert updated["revenue_lag_7"]  == pytest.approx(700.0)
        assert updated["roas_lag_7"]     == pytest.approx(7.0)   # 700 / spend_lag_7(100)

    def test_roll_std_nan_with_single_value(self):
        row    = self._base_row()
        origin = pd.Timestamp("2024-03-08")
        # Only one value in the window
        actual = {pd.Timestamp("2024-03-07"): 300.0}
        updated = _update_revenue_features(row, origin, actual, {})
        # std needs at least 2 values (ddof=1)
        assert np.isnan(updated["revenue_roll_std_7"])

    def test_roll_std_computed_with_two_values(self):
        row    = self._base_row()
        origin = pd.Timestamp("2024-03-08")
        actual = {
            pd.Timestamp("2024-03-07"): 100.0,
            pd.Timestamp("2024-03-06"): 200.0,
        }
        updated = _update_revenue_features(row, origin, actual, {})
        expected_std = float(np.std([100.0, 200.0], ddof=1))
        assert updated["revenue_roll_std_7"] == pytest.approx(expected_std, rel=1e-5)

    def test_columns_not_in_row_are_not_created(self):
        row = pd.Series({"some_col": 1.0})  # no revenue cols
        origin = pd.Timestamp("2024-03-08")
        updated = _update_revenue_features(row, origin, {}, {})
        assert "revenue_lag_1" not in updated.index


# ---------------------------------------------------------------------------
# Integration tests for generate_future_forecasts
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_row_count(self):
        fs, fcols = _make_feature_store(n_campaigns=3)
        result = generate_future_forecasts(fs, _ConstantModel(), fcols, n_future_days=14)
        assert len(result) == 3 * 14

    def test_one_campaign(self):
        fs, fcols = _make_feature_store(n_campaigns=1)
        result = generate_future_forecasts(fs, _ConstantModel(), fcols, n_future_days=7)
        assert len(result) == 7

    def test_n_future_days_respected(self):
        fs, fcols = _make_feature_store(n_campaigns=2)
        for n in (1, 7, 14, 30):
            result = generate_future_forecasts(fs, _ConstantModel(), fcols, n_future_days=n)
            assert len(result) == 2 * n, f"Expected {2*n} rows for n_future_days={n}"

    def test_required_columns_present(self):
        fs, fcols = _make_feature_store()
        result = generate_future_forecasts(fs, _ConstantModel(), fcols)
        for col in ("campaign_id", "campaign_name", "platform", "date",
                    "revenue_attributed", "p10", "p50", "p90", "is_future"):
            assert col in result.columns, f"Missing column: {col}"


class TestOutputValues:
    def test_is_future_all_true(self):
        fs, fcols = _make_feature_store()
        result = generate_future_forecasts(fs, _ConstantModel(), fcols)
        assert result["is_future"].all()

    def test_revenue_attributed_all_nan(self):
        fs, fcols = _make_feature_store()
        result = generate_future_forecasts(fs, _ConstantModel(), fcols)
        assert result["revenue_attributed"].isna().all()

    def test_quantile_ordering(self):
        fs, fcols = _make_feature_store()
        result = generate_future_forecasts(fs, _ConstantModel(), fcols)
        assert (result["p10"] <= result["p50"]).all()
        assert (result["p50"] <= result["p90"]).all()

    def test_p50_nonnegative(self):
        fs, fcols = _make_feature_store()
        result = generate_future_forecasts(fs, _ConstantModel(p10=10, p50=100, p90=200), fcols)
        assert (result["p50"] >= 0).all()

    def test_dates_are_future(self):
        fs, fcols = _make_feature_store()
        last_date = fs["date"].max()
        result = generate_future_forecasts(fs, _ConstantModel(), fcols, n_future_days=14)
        assert (result["date"] > last_date).all()

    def test_date_range_correct(self):
        fs, fcols = _make_feature_store()
        last_date = pd.Timestamp(fs["date"].max())
        result = generate_future_forecasts(fs, _ConstantModel(), fcols, n_future_days=7)
        dates = sorted(result["date"].unique())
        expected = [last_date + pd.Timedelta(days=d) for d in range(1, 8)]
        assert dates == expected


class TestAutoregessiveCascade:
    """The most important tests — verify that lag features are updated from prior predictions."""

    def test_lag1_on_day2_equals_p50_of_day1(self):
        """revenue_lag_1 passed to the model on day T+2 must equal P50(T+1)."""
        fs, fcols = _make_feature_store(n_campaigns=1)
        model = _RecordingModel(p50=777.0)
        generate_future_forecasts(fs, model, fcols, n_future_days=3)

        # model.calls[0] = features for day T+1
        # model.calls[1] = features for day T+2 → revenue_lag_1 must be 777.0
        assert len(model.calls) >= 2
        if "revenue_lag_1" in model.calls[1].columns:
            actual_lag1 = float(model.calls[1]["revenue_lag_1"].iloc[0])
            assert actual_lag1 == pytest.approx(777.0, rel=1e-9), (
                f"Expected revenue_lag_1=777.0 on day T+2, got {actual_lag1}"
            )

    def test_lag7_on_day8_equals_p50_of_day1(self):
        """revenue_lag_7 passed on day T+8 must equal the P50 prediction for day T+1."""
        fs, fcols = _make_feature_store(n_campaigns=1)
        model = _RecordingModel(p50=555.0)
        generate_future_forecasts(fs, model, fcols, n_future_days=8)

        # model.calls[7] = features for day T+8
        # revenue_lag_7 at T+8 → looks 7 days back → lands on T+1 → P50=555.0
        assert len(model.calls) == 8
        if "revenue_lag_7" in model.calls[7].columns:
            actual_lag7 = float(model.calls[7]["revenue_lag_7"].iloc[0])
            assert actual_lag7 == pytest.approx(555.0, rel=1e-9), (
                f"Expected revenue_lag_7=555.0 on day T+8, got {actual_lag7}"
            )

    def test_roll_mean7_on_day8_uses_only_predictions(self):
        """revenue_roll_mean_7 on day T+8 must equal mean of P50(T+1)..P50(T+7).

        At day T+8, the 7-day window (T+7, T+6, ..., T+1) contains only
        predicted values.  With a constant-P50 model, roll_mean_7 == P50.
        """
        fs, fcols = _make_feature_store(n_campaigns=1)
        constant_p50 = 444.0
        model = _RecordingModel(p50=constant_p50)
        generate_future_forecasts(fs, model, fcols, n_future_days=8)

        assert len(model.calls) == 8
        if "revenue_roll_mean_7" in model.calls[7].columns:
            roll_mean = float(model.calls[7]["revenue_roll_mean_7"].iloc[0])
            assert roll_mean == pytest.approx(constant_p50, rel=1e-9), (
                f"Expected roll_mean_7={constant_p50:.1f} on day T+8 (all-predicted window), got {roll_mean:.4f}"
            )

    def test_day1_still_uses_actuals_for_lag7(self):
        """On day T+1, revenue_lag_7 must come from actual history, not predictions."""
        fs, fcols = _make_feature_store(n_campaigns=1)
        model = _RecordingModel(p50=999.0)

        # The actual revenue_lag_7 on the last row of the feature store
        last_actual_lag7 = float(
            fs[fs["campaign_id"] == "camp_0"].sort_values("date").iloc[-1].get(
                "revenue_lag_7", np.nan
            )
        )

        generate_future_forecasts(fs, model, fcols, n_future_days=1)

        # model.calls[0] = features for day T+1
        if "revenue_lag_7" in model.calls[0].columns:
            fed_lag7 = float(model.calls[0]["revenue_lag_7"].iloc[0])
            # lag_7 on T+1 → looks at T-6 (7 days before T+1 = day T-6)
            # This is a real historical value, not the fresh p50=999
            assert fed_lag7 != pytest.approx(999.0), (
                "Day T+1 should not use predicted values for lag_7 (no predictions exist yet)"
            )

    def test_multiple_campaigns_independent_buffers(self):
        """Each campaign's prediction buffer must be independent."""
        fs, fcols = _make_feature_store(n_campaigns=3, base_revenue=100.0)
        model = _RecordingModel(p50=200.0)
        result = generate_future_forecasts(fs, model, fcols, n_future_days=5)

        for cid in fs["campaign_id"].unique():
            camp_result = result[result["campaign_id"] == cid]
            assert len(camp_result) == 5
            assert np.allclose(camp_result["p50"].values, 200.0)


class TestUncertaintyGrowth:
    """Interval width should not systematically shrink as horizon grows."""

    def test_interval_width_non_decreasing_on_average(self):
        """With a model that widens predictions step-by-step, the output must reflect it."""
        fs, fcols = _make_feature_store(n_campaigns=2)
        model = _GrowingUncertaintyModel()
        result = generate_future_forecasts(fs, model, fcols, n_future_days=14)

        widths = (result["p90"] - result["p10"]).values
        # Day 1 width < day 14 width (model explicitly widens each call)
        day1_widths  = widths[:len(fs["campaign_id"].unique())]
        day14_widths = widths[-len(fs["campaign_id"].unique()):]
        assert day14_widths.mean() > day1_widths.mean(), (
            "Interval should be wider on day 14 than on day 1 when model widens each step"
        )

    def test_constant_model_produces_constant_width(self):
        """A constant-output model should produce the same width every day."""
        fs, fcols = _make_feature_store(n_campaigns=2)
        model = _ConstantModel(p10=80, p50=100, p90=120)
        result = generate_future_forecasts(fs, model, fcols, n_future_days=14)

        widths = (result["p90"] - result["p10"]).round(6)
        assert widths.nunique() == 1, "Constant model should produce identical widths every day"


class TestEdgeCases:
    def test_empty_feature_store_returns_empty(self):
        empty_fs = pd.DataFrame()
        result = generate_future_forecasts(empty_fs, _ConstantModel(), [])
        assert result.empty

    def test_n_future_days_zero_returns_empty(self):
        fs, fcols = _make_feature_store()
        result = generate_future_forecasts(fs, _ConstantModel(), fcols, n_future_days=0)
        assert result.empty

    def test_single_day_feature_store(self):
        fs_single, fcols = _make_feature_store(n_campaigns=2, n_days=1)
        result = generate_future_forecasts(fs_single, _ConstantModel(), fcols, n_future_days=3)
        assert len(result) == 2 * 3

    def test_feature_cols_subset_does_not_crash(self):
        """If feature_cols contains columns not in the feature store, they are silently skipped."""
        fs, fcols = _make_feature_store()
        extended_cols = fcols + ["nonexistent_col_xyz"]
        result = generate_future_forecasts(fs, _ConstantModel(), extended_cols)
        assert not result.empty

    def test_n_future_days_1_returns_single_step(self):
        fs, fcols = _make_feature_store(n_campaigns=4)
        result = generate_future_forecasts(fs, _ConstantModel(), fcols, n_future_days=1)
        assert len(result) == 4


class TestCalendarUpdates:
    def test_future_dates_have_correct_day_of_week(self):
        """day_of_week fed to the model must match the actual future calendar date."""
        fs, fcols = _make_feature_store(n_campaigns=1)
        model = _RecordingModel()
        last_date = pd.Timestamp(fs["date"].max())
        generate_future_forecasts(fs, model, fcols, n_future_days=7)

        if "day_of_week" in model.calls[0].columns:
            for i, call in enumerate(model.calls):
                expected_date = last_date + pd.Timedelta(days=i + 1)
                fed_dow = int(call["day_of_week"].iloc[0])
                assert fed_dow == expected_date.dayofweek, (
                    f"Step {i+1}: expected day_of_week={expected_date.dayofweek}, got {fed_dow}"
                )

    def test_weekend_flag_correct(self):
        """is_weekend must be 1 on Saturday/Sunday and 0 otherwise."""
        fs, fcols = _make_feature_store(n_campaigns=1)
        model = _RecordingModel()
        last_date = pd.Timestamp(fs["date"].max())
        generate_future_forecasts(fs, model, fcols, n_future_days=7)

        if "is_weekend" in model.calls[0].columns:
            for i, call in enumerate(model.calls):
                expected_date = last_date + pd.Timedelta(days=i + 1)
                expected_flag = int(expected_date.dayofweek >= 5)
                fed_flag = int(call["is_weekend"].iloc[0])
                assert fed_flag == expected_flag
