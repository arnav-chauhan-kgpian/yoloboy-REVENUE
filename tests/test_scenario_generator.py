"""
tests/test_scenario_generator.py
==================================
Unit tests for src/simulation/scenario_generator.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.simulation.scenario_generator import (
    ScenarioResult,
    ScenarioError,
    apply_scenario,
    compare_scenarios,
    generate_standard_scenarios,
)
from src.simulation.response_curve import build_response_curves
import pandas as pd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(11)


def _make_feature_store(n_campaigns=6, n_days=80) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    platforms = ["google", "google", "meta", "meta", "bing", "bing"]
    for cid in range(n_campaigns):
        platform = platforms[cid % len(platforms)]
        base_spend = (cid + 1) * 100.0
        roas = [3.0, 2.5, 2.0, 1.8, 1.2, 1.0][cid % 6]
        for i, date in enumerate(dates):
            spend_noise = RNG.normal(0, base_spend * 0.25)
            spend = max(base_spend + spend_noise + i * 0.3, 1.0)
            v_max, K = roas * base_spend * 2.0, base_spend
            rev = v_max * spend / (K + spend) + RNG.normal(0, 8)
            rev = max(rev, 0.0)
            rows.append({
                "campaign_id":       f"camp_{cid}",
                "campaign_name":     f"Campaign {cid}",
                "platform":          platform,
                "date":              date,
                "spend":             spend,
                "revenue_attributed": rev,
                "attribution_mature": i < n_days - 14,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def curves():
    fs = _make_feature_store()
    return build_response_curves(fs)


@pytest.fixture(scope="module")
def baseline_result(curves):
    return apply_scenario(curves, scenario_name="Baseline")


# ---------------------------------------------------------------------------
# TestApplyScenario
# ---------------------------------------------------------------------------

class TestApplyScenario:
    def test_returns_scenario_result(self, curves):
        r = apply_scenario(curves)
        assert isinstance(r, ScenarioResult)

    def test_baseline_lift_is_exactly_zero(self, baseline_result):
        """No budget changes → revenue lift must be exactly 0 (spend_delta == 0 guard)."""
        assert baseline_result.revenue_lift == pytest.approx(0.0, abs=1e-9)
        assert baseline_result.revenue_lift_pct == pytest.approx(0.0, abs=1e-9)

    def test_baseline_roas_equals_projected_roas(self, baseline_result):
        """No budget changes → baseline and projected ROAS must be identical."""
        assert baseline_result.baseline_roas == pytest.approx(baseline_result.projected_roas, rel=1e-9)

    def test_increase_spend_increases_revenue(self, curves):
        r = apply_scenario(curves, platform_multipliers={"google": 1.5})
        # Google campaigns get more spend → more revenue
        google_proj = [p for p in r.campaign_projections if p.platform == "google"]
        for p in google_proj:
            assert p.new_daily_spend >= p.baseline_daily_spend

    def test_decrease_spend_decreases_revenue(self, curves):
        r = apply_scenario(curves, platform_multipliers={"google": 0.5})
        google_proj = [p for p in r.campaign_projections if p.platform == "google"]
        for p in google_proj:
            assert p.new_daily_spend <= p.baseline_daily_spend

    def test_platform_multiplier_doesnt_affect_other_platforms(self, curves):
        baseline = apply_scenario(curves)
        changed  = apply_scenario(curves, platform_multipliers={"google": 1.5})
        for p_b, p_c in zip(
            [p for p in baseline.campaign_projections if p.platform != "google"],
            [p for p in changed.campaign_projections  if p.platform != "google"],
        ):
            assert p_b.new_daily_spend == pytest.approx(p_c.new_daily_spend, rel=1e-9)

    def test_campaign_override_takes_precedence(self, curves):
        cid = next(iter(curves.keys()))
        override_spend = 9999.0
        r = apply_scenario(
            curves,
            platform_multipliers={"google": 2.0},
            campaign_overrides={cid: override_spend},
        )
        proj = next(p for p in r.campaign_projections if p.campaign_id == cid)
        assert proj.new_daily_spend == pytest.approx(override_spend)

    def test_revenue_lift_sign_matches_spend_direction(self, curves):
        r_up   = apply_scenario(curves, platform_multipliers={"google": 1.5})
        r_down = apply_scenario(curves, platform_multipliers={"google": 0.5})
        google_up_lift   = sum(p.revenue_delta for p in r_up.campaign_projections   if p.platform == "google")
        google_down_lift = sum(p.revenue_delta for p in r_down.campaign_projections if p.platform == "google")
        assert google_up_lift   >= 0, "Increased spend should increase revenue"
        assert google_down_lift <= 0, "Decreased spend should decrease revenue"

    def test_risk_score_bounded(self, curves):
        r = apply_scenario(curves, platform_multipliers={"google": 1.2})
        assert 0.0 <= r.risk_score <= 1.0

    def test_confidence_interval_ordering(self, curves):
        r = apply_scenario(curves)
        assert r.ci_low <= r.projected_total_revenue <= r.ci_high

    def test_empty_curves_raises(self):
        with pytest.raises(ScenarioError):
            apply_scenario({})

    def test_campaign_projections_count(self, curves, baseline_result):
        assert len(baseline_result.campaign_projections) == len(curves)

    def test_total_spend_computed_correctly(self, curves, baseline_result):
        expected = sum(c.avg_daily_spend for c in curves.values())
        assert baseline_result.baseline_total_spend == pytest.approx(expected, rel=1e-5)

    def test_roas_positive_when_spend_positive(self, curves, baseline_result):
        if baseline_result.baseline_total_spend > 0:
            assert baseline_result.baseline_roas >= 0

    def test_scenario_name_preserved(self, curves):
        r = apply_scenario(curves, scenario_name="My Test Scenario")
        assert r.scenario_name == "My Test Scenario"


# ---------------------------------------------------------------------------
# TestCompareScenarios
# ---------------------------------------------------------------------------

class TestCompareScenarios:
    def test_returns_dataframe(self, curves):
        r1 = apply_scenario(curves, scenario_name="A")
        r2 = apply_scenario(curves, platform_multipliers={"google": 1.2}, scenario_name="B")
        df = compare_scenarios([r1, r2])
        assert len(df) == 2

    def test_columns_present(self, curves):
        r = apply_scenario(curves)
        df = compare_scenarios([r])
        expected = {"scenario", "revenue_lift_pct", "projected_revenue", "risk_score"}
        assert expected.issubset(df.columns)

    def test_empty_list(self):
        df = compare_scenarios([])
        assert len(df) == 0


# ---------------------------------------------------------------------------
# TestGenerateStandardScenarios
# ---------------------------------------------------------------------------

class TestGenerateStandardScenarios:
    def test_returns_list(self, curves):
        scenarios = generate_standard_scenarios(curves)
        assert isinstance(scenarios, list)

    def test_at_least_three_scenarios(self, curves):
        scenarios = generate_standard_scenarios(curves)
        assert len(scenarios) >= 3

    def test_baseline_has_exactly_zero_lift(self, curves):
        scenarios = generate_standard_scenarios(curves)
        baseline = next((s for s in scenarios if s.scenario_name == "Baseline"), None)
        assert baseline is not None
        assert baseline.revenue_lift == pytest.approx(0.0, abs=1e-9)
        assert baseline.revenue_lift_pct == pytest.approx(0.0, abs=1e-9)

    def test_all_have_risk_scores(self, curves):
        for s in generate_standard_scenarios(curves):
            assert 0.0 <= s.risk_score <= 1.0
