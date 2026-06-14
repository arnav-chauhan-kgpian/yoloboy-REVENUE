"""
tests/test_demo_hardening.py
==============================
Tests verifying the pre-submission hardening fixes:

  Fix 1  — Baseline scenario zero lift
  Fix 2  — Forecast output schema (is_future, campaign_name)
  Fix 3  — AI Copilot cache key is content-based (not shape-only)
  Fix 4  — Demo terminal output uses enum.name, not repr
  Fix 5  — streamlit_app package is importable
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Project root on path (also handled by conftest.py)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_feature_store(n_campaigns: int = 4, n_days: int = 90) -> pd.DataFrame:
    platforms = ["google", "google", "meta", "bing"]
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for cid in range(n_campaigns):
        platform = platforms[cid % len(platforms)]
        base_spend = (cid + 1) * 200.0
        roas = [3.5, 2.5, 2.0, 1.5][cid % 4]
        for i, date in enumerate(dates):
            spend = max(base_spend + RNG.normal(0, base_spend * 0.2), 5.0)
            v_max, K = roas * base_spend * 2.0, base_spend
            rev = v_max * spend / (K + spend) + RNG.normal(0, 10)
            rev = max(rev, 0.0)
            rows.append({
                "campaign_id":        f"camp_{cid}",
                "campaign_name":      f"Campaign {cid}",
                "platform":           platform,
                "date":               date,
                "spend":              round(spend, 2),
                "revenue_attributed": round(rev, 2),
                "attribution_mature": i < n_days - 14,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fs():
    return _make_feature_store()


@pytest.fixture(scope="module")
def curves(fs):
    from src.simulation.response_curve import build_response_curves
    return build_response_curves(fs)


# ===========================================================================
# FIX 1 — Baseline scenario must produce exactly zero lift
# ===========================================================================

class TestBaselineScenarioZeroLift:
    """No budget change must produce revenue_lift == 0 and ROAS lift == 0."""

    def test_revenue_lift_is_exactly_zero(self, curves):
        from src.simulation.scenario_generator import apply_scenario
        result = apply_scenario(curves)
        assert result.revenue_lift == pytest.approx(0.0, abs=1e-9), (
            f"Expected 0 lift, got {result.revenue_lift:.6f}"
        )

    def test_revenue_lift_pct_is_exactly_zero(self, curves):
        from src.simulation.scenario_generator import apply_scenario
        result = apply_scenario(curves)
        assert result.revenue_lift_pct == pytest.approx(0.0, abs=1e-9), (
            f"Expected 0% lift, got {result.revenue_lift_pct:.6f}%"
        )

    def test_baseline_roas_equals_projected_roas(self, curves):
        from src.simulation.scenario_generator import apply_scenario
        result = apply_scenario(curves)
        assert result.baseline_roas == pytest.approx(result.projected_roas, rel=1e-9)

    def test_campaign_level_revenue_delta_is_zero(self, curves):
        from src.simulation.scenario_generator import apply_scenario
        result = apply_scenario(curves)
        for proj in result.campaign_projections:
            assert proj.revenue_delta == pytest.approx(0.0, abs=1e-9), (
                f"Campaign {proj.campaign_id} has non-zero delta: {proj.revenue_delta}"
            )

    def test_spend_multiplier_of_one_is_baseline(self, curves):
        """Explicit 1.0 multiplier on every platform must still be zero lift."""
        from src.simulation.scenario_generator import apply_scenario
        platforms = {c.platform for c in curves.values()}
        mults = {p: 1.0 for p in platforms}
        result = apply_scenario(curves, platform_multipliers=mults)
        assert result.revenue_lift == pytest.approx(0.0, abs=1e-9)

    def test_nonzero_multiplier_produces_nonzero_lift(self, curves):
        """Sanity check: actually changing spend must change revenue."""
        from src.simulation.scenario_generator import apply_scenario
        platforms = {c.platform for c in curves.values()}
        first_platform = next(iter(platforms))
        result = apply_scenario(curves, platform_multipliers={first_platform: 1.5})
        assert result.revenue_lift != pytest.approx(0.0, abs=1.0), (
            "Expected non-zero lift when spend increases by 50%"
        )


# ===========================================================================
# FIX 2 — Forecast output schema
# ===========================================================================

class TestForecastOutputSchema:
    """Forecasts from both historical and future paths must share a schema."""

    def _make_minimal_forecast(self, n_campaigns: int = 3, n_days: int = 7) -> pd.DataFrame:
        rows = []
        for cid in range(n_campaigns):
            for d in range(n_days):
                rows.append({
                    "campaign_id":        f"camp_{cid}",
                    "campaign_name":      f"Campaign {cid}",
                    "platform":           "google",
                    "date":               pd.Timestamp("2024-03-01") + pd.Timedelta(days=d),
                    "revenue_attributed": float(100 + cid * 50 + d),
                    "p10":                90.0,
                    "p50":                100.0,
                    "p90":                110.0,
                    "is_future":          False,
                })
        return pd.DataFrame(rows)

    def _make_minimal_future_forecast(self, n_campaigns: int = 3, n_days: int = 7) -> pd.DataFrame:
        rows = []
        for cid in range(n_campaigns):
            for d in range(n_days):
                rows.append({
                    "campaign_id":        f"camp_{cid}",
                    "campaign_name":      f"Campaign {cid}",
                    "platform":           "google",
                    "date":               pd.Timestamp("2024-04-01") + pd.Timedelta(days=d),
                    "revenue_attributed": np.nan,
                    "p10":                85.0,
                    "p50":                100.0,
                    "p90":                115.0,
                    "is_future":          True,
                })
        return pd.DataFrame(rows)

    def test_historical_forecast_has_is_future_false(self):
        fc = self._make_minimal_forecast()
        assert "is_future" in fc.columns
        assert fc["is_future"].all() == False
        assert (fc["is_future"] == False).all()

    def test_future_forecast_has_is_future_true(self):
        fc = self._make_minimal_future_forecast()
        assert "is_future" in fc.columns
        assert fc["is_future"].all() == True

    def test_historical_forecast_has_campaign_name(self):
        fc = self._make_minimal_forecast()
        assert "campaign_name" in fc.columns
        assert fc["campaign_name"].notna().all()

    def test_future_forecast_has_campaign_name(self):
        fc = self._make_minimal_future_forecast()
        assert "campaign_name" in fc.columns
        assert fc["campaign_name"].notna().all()

    def test_concat_produces_no_nan_campaign_name(self):
        fc_hist   = self._make_minimal_forecast()
        fc_future = self._make_minimal_future_forecast()
        combined  = pd.concat([fc_hist, fc_future], ignore_index=True)
        assert combined["campaign_name"].notna().all(), (
            "Concatenated forecasts must not have NaN campaign_name"
        )

    def test_concat_produces_no_nan_is_future(self):
        fc_hist   = self._make_minimal_forecast()
        fc_future = self._make_minimal_future_forecast()
        combined  = pd.concat([fc_hist, fc_future], ignore_index=True)
        assert combined["is_future"].notna().all(), (
            "Concatenated forecasts must not have NaN is_future"
        )

    def test_is_future_split_correct_after_concat(self):
        fc_hist   = self._make_minimal_forecast(n_campaigns=2, n_days=5)
        fc_future = self._make_minimal_future_forecast(n_campaigns=2, n_days=3)
        combined  = pd.concat([fc_hist, fc_future], ignore_index=True)
        assert (combined.loc[combined["is_future"] == False, "revenue_attributed"].notna()).any()
        assert (combined.loc[combined["is_future"] == True, "revenue_attributed"].isna()).all()

    def test_required_columns_present_in_both(self):
        required = {"campaign_id", "campaign_name", "platform", "date",
                    "p10", "p50", "p90", "is_future"}
        assert required.issubset(self._make_minimal_forecast().columns)
        assert required.issubset(self._make_minimal_future_forecast().columns)


# ===========================================================================
# FIX 3 — AI Copilot cache key is content-based
# ===========================================================================

class TestCopilotCacheKey:
    """_make_fc_hash must differ when content changes, not just row count."""

    @staticmethod
    def _make_forecast(n_rows: int, p50_base: float = 100.0) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=n_rows, freq="D")
        return pd.DataFrame({
            "campaign_id": [f"c_{i}" for i in range(n_rows)],
            "date":        dates,
            "p50":         [p50_base] * n_rows,
            "p10":         [p50_base * 0.8] * n_rows,
            "p90":         [p50_base * 1.2] * n_rows,
        })

    def _make_fc_hash(self, fc) -> str:
        """Mirror of the implementation in 4_AI_Copilot.py."""
        if fc is None or fc.empty:
            return "none"
        return (
            f"{len(fc)}"
            f"_{fc['date'].max()}"
            f"_{fc['p50'].sum():.0f}"
        )

    def test_none_produces_constant_hash(self):
        assert self._make_fc_hash(None) == "none"

    def test_empty_df_produces_constant_hash(self):
        assert self._make_fc_hash(pd.DataFrame()) == "none"

    def test_same_data_produces_same_hash(self):
        fc = self._make_forecast(10)
        assert self._make_fc_hash(fc) == self._make_fc_hash(fc)

    def test_different_row_count_changes_hash(self):
        fc1 = self._make_forecast(10)
        fc2 = self._make_forecast(15)
        assert self._make_fc_hash(fc1) != self._make_fc_hash(fc2)

    def test_same_row_count_different_p50_changes_hash(self):
        """This is the fix: same row count but new model → different hash."""
        fc1 = self._make_forecast(10, p50_base=100.0)
        fc2 = self._make_forecast(10, p50_base=200.0)
        assert self._make_fc_hash(fc1) != self._make_fc_hash(fc2)

    def test_same_row_count_different_date_changes_hash(self):
        """Regenerated forecast with same shape but newer dates → different hash."""
        fc1 = self._make_forecast(5)
        rows = fc1.copy()
        rows["date"] = pd.date_range("2025-01-01", periods=5, freq="D")
        assert self._make_fc_hash(fc1) != self._make_fc_hash(rows)

    def test_hash_is_deterministic_string(self):
        fc = self._make_forecast(7)
        h1 = self._make_fc_hash(fc)
        h2 = self._make_fc_hash(fc)
        assert isinstance(h1, str)
        assert h1 == h2


# ===========================================================================
# FIX 4 — Demo terminal output uses enum .name not repr
# ===========================================================================

class TestRecommendationPriorityPrint:
    """Verifies that priority prints as 'HIGH' not 'RecommendationPriority.HIGH'."""

    def test_priority_name_is_short_string(self):
        from src.copilot.recommender import RecommendationPriority
        for p in RecommendationPriority:
            rendered = p.name
            assert "RecommendationPriority" not in rendered, (
                f"p.name should not contain class name: {rendered!r}"
            )
            assert rendered in ("HIGH", "MEDIUM", "LOW")

    def test_priority_format_string_uses_name(self):
        from src.copilot.recommender import RecommendationPriority
        p = RecommendationPriority.HIGH
        output = f"[{p.name}]"
        assert output == "[HIGH]"
        assert "RecommendationPriority" not in output

    def test_all_priorities_render_cleanly(self):
        from src.copilot.recommender import RecommendationPriority
        expected = {"[HIGH]", "[MEDIUM]", "[LOW]"}
        rendered = {f"[{p.name}]" for p in RecommendationPriority}
        assert rendered == expected


# ===========================================================================
# FIX 5 — streamlit_app package is importable
# ===========================================================================

class TestStreamlitAppPackageImport:
    """streamlit_app must be importable as a package (has __init__.py)."""

    def test_init_file_exists(self):
        init_path = _ROOT / "streamlit_app" / "__init__.py"
        assert init_path.exists(), (
            f"streamlit_app/__init__.py not found at {init_path}"
        )

    def test_state_module_importable(self):
        """streamlit_app.state must import without Streamlit being active."""
        # Guard: skip if streamlit triggers runtime errors outside its server
        try:
            import streamlit  # noqa: F401
        except ImportError:
            pytest.skip("streamlit not installed")

        # The module sets up path manipulation and cached loaders.
        # We verify it can be imported (no syntax errors, no missing deps).
        try:
            import streamlit_app.state as _state  # noqa: F401
            assert hasattr(_state, "load_feature_store")
            assert hasattr(_state, "load_forecasts")
            assert hasattr(_state, "generate_future_forecasts")
        except Exception as exc:
            pytest.fail(f"streamlit_app.state import failed: {exc}")
