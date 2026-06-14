"""
tests/test_optimizer.py
========================
Unit tests for src/simulation/optimizer.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.simulation.optimizer import (
    OptimizationConstraint,
    OptimizationResult,
    CampaignAllocation,
    OptimizerError,
    optimize_budget,
)
from src.simulation.response_curve import build_response_curves
import pandas as pd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_feature_store(n_campaigns=6, n_days=80) -> pd.DataFrame:
    platforms = ["google", "google", "meta", "meta", "bing", "bing"]
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for cid in range(n_campaigns):
        platform = platforms[cid % len(platforms)]
        base_spend = (cid + 1) * 120.0
        roas = [3.5, 2.8, 2.0, 1.7, 1.1, 0.9][cid % 6]
        for i, date in enumerate(dates):
            spend_noise = RNG.normal(0, base_spend * 0.3)
            spend = max(base_spend + spend_noise + i * 0.2, 1.0)
            v_max, K = roas * base_spend * 2.5, base_spend
            rev = v_max * spend / (K + spend) + RNG.normal(0, 10)
            rev = max(rev, 0.0)
            rows.append({
                "campaign_id":        f"camp_{cid}",
                "campaign_name":      f"Campaign {cid}",
                "platform":           platform,
                "date":               date,
                "spend":              spend,
                "revenue_attributed": rev,
                "attribution_mature": i < n_days - 14,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def curves():
    fs = _make_feature_store()
    return build_response_curves(fs)


@pytest.fixture(scope="module")
def total_budget(curves):
    return sum(c.avg_daily_spend for c in curves.values())


@pytest.fixture(scope="module")
def opt_result(curves, total_budget):
    return optimize_budget(curves, total_budget=total_budget)


# ---------------------------------------------------------------------------
# TestOptimizeBudget
# ---------------------------------------------------------------------------

class TestOptimizeBudget:
    def test_returns_optimization_result(self, curves, total_budget):
        result = optimize_budget(curves, total_budget=total_budget)
        assert isinstance(result, OptimizationResult)

    def test_budget_conservation(self, opt_result, total_budget):
        """Sum of optimal spend must equal total_budget within tolerance."""
        allocated = sum(a.optimal_spend for a in opt_result.optimal_allocations)
        assert allocated == pytest.approx(total_budget, rel=1e-3)

    def test_budget_error_small(self, opt_result, total_budget):
        assert abs(opt_result.budget_error) < total_budget * 0.01

    def test_optimal_revenue_gte_baseline(self, opt_result):
        """Optimized allocation should not worsen revenue."""
        assert opt_result.optimal_total_revenue >= opt_result.baseline_total_revenue * 0.95

    def test_revenue_lift_pct_consistent(self, opt_result):
        if opt_result.baseline_total_revenue > 0:
            expected = (
                (opt_result.optimal_total_revenue - opt_result.baseline_total_revenue)
                / opt_result.baseline_total_revenue * 100
            )
            assert opt_result.revenue_lift_pct == pytest.approx(expected, rel=1e-4)

    def test_risk_score_bounded(self, opt_result):
        assert 0.0 <= opt_result.risk_score <= 1.0

    def test_all_campaigns_have_allocations(self, curves, opt_result):
        alloc_ids = {a.campaign_id for a in opt_result.optimal_allocations}
        assert alloc_ids == set(curves.keys())

    def test_allocations_are_positive(self, opt_result):
        for a in opt_result.optimal_allocations:
            assert a.optimal_spend >= 0.0

    def test_empty_curves_raises(self):
        with pytest.raises(OptimizerError):
            optimize_budget({}, total_budget=1000.0)

    def test_zero_budget_raises(self, curves):
        with pytest.raises(OptimizerError):
            optimize_budget(curves, total_budget=0.0)

    def test_negative_budget_raises(self, curves):
        with pytest.raises(OptimizerError):
            optimize_budget(curves, total_budget=-500.0)

    def test_n_iterations_positive(self, opt_result):
        assert opt_result.n_iterations >= 0

    def test_solver_message_is_string(self, opt_result):
        assert isinstance(opt_result.solver_message, str)

    def test_allocation_dataclass_fields(self, opt_result):
        for alloc in opt_result.optimal_allocations:
            assert isinstance(alloc, CampaignAllocation)
            assert alloc.baseline_spend >= 0
            assert alloc.optimal_spend >= 0
            assert isinstance(alloc.spend_delta, float)
            assert isinstance(alloc.spend_delta_pct, float)

    def test_saturation_at_optimal_bounded(self, opt_result):
        for alloc in opt_result.optimal_allocations:
            assert 0.0 <= alloc.saturation_at_optimal <= 1.0


# ---------------------------------------------------------------------------
# TestConstraints
# ---------------------------------------------------------------------------

class TestConstraints:
    def test_constraint_respected_max(self, curves, total_budget):
        campaign_ids = list(curves.keys())
        target_id = campaign_ids[0]
        constraints = [
            OptimizationConstraint(
                campaign_id=target_id,
                max_daily_spend=10.0,
            )
        ]
        result = optimize_budget(curves, total_budget=total_budget, constraints=constraints)
        a = next(x for x in result.optimal_allocations if x.campaign_id == target_id)
        assert a.optimal_spend <= 10.0 + 1e-3

    def test_constraint_respected_min(self, curves, total_budget):
        campaign_ids = list(curves.keys())
        target_id = campaign_ids[0]
        min_spend = 5.0
        constraints = [
            OptimizationConstraint(
                campaign_id=target_id,
                min_daily_spend=min_spend,
            )
        ]
        result = optimize_budget(curves, total_budget=total_budget, constraints=constraints)
        a = next(x for x in result.optimal_allocations if x.campaign_id == target_id)
        assert a.optimal_spend >= min_spend - 1e-3


# ---------------------------------------------------------------------------
# TestOptimizationResult
# ---------------------------------------------------------------------------

class TestOptimizationResult:
    def test_baseline_roas_positive(self, opt_result):
        if opt_result.baseline_total_revenue > 0:
            assert opt_result.baseline_roas >= 0

    def test_optimal_roas_positive(self, opt_result):
        if opt_result.optimal_total_revenue > 0:
            assert opt_result.optimal_roas >= 0

    def test_budget_allocated_matches_total(self, opt_result, total_budget):
        assert opt_result.budget_allocated == pytest.approx(
            opt_result.total_budget, rel=1e-3
        )
