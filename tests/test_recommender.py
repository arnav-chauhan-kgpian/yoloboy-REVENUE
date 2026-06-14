"""
tests/test_recommender.py
==========================
Unit tests for src/copilot/recommender.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import pandas as pd

from src.copilot.recommender import (
    Recommendation,
    RecommendationType,
    RecommendationPriority,
    from_optimizer_result,
    generate_quick_recommendations,
)
from src.simulation.optimizer import optimize_budget
from src.simulation.response_curve import build_response_curves


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(55)


def _make_feature_store(n_campaigns=6, n_days=80) -> pd.DataFrame:
    platforms = ["google", "google", "meta", "meta", "bing", "bing"]
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for cid in range(n_campaigns):
        platform = platforms[cid % len(platforms)]
        base_spend = (cid + 1) * 100.0
        roas = [4.0, 3.0, 2.0, 1.5, 0.8, 0.5][cid % 6]
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
    return build_response_curves(_make_feature_store())


@pytest.fixture(scope="module")
def opt_result(curves):
    total = sum(c.avg_daily_spend for c in curves.values())
    return optimize_budget(curves, total_budget=total)


@pytest.fixture(scope="module")
def recommendations_from_opt(opt_result, curves):
    return from_optimizer_result(opt_result, curves)


@pytest.fixture(scope="module")
def quick_recommendations(curves):
    return generate_quick_recommendations(curves)


# ---------------------------------------------------------------------------
# TestFromOptimizerResult
# ---------------------------------------------------------------------------

class TestFromOptimizerResult:
    def test_returns_list(self, recommendations_from_opt):
        assert isinstance(recommendations_from_opt, list)

    def test_all_recommendation_instances(self, recommendations_from_opt):
        for r in recommendations_from_opt:
            assert isinstance(r, Recommendation)

    def test_recommendation_has_required_fields(self, recommendations_from_opt):
        for r in recommendations_from_opt:
            assert isinstance(r.title, str) and len(r.title) > 0
            assert isinstance(r.rationale, str) and len(r.rationale) > 0
            assert r.priority in RecommendationPriority
            assert r.type in RecommendationType

    def test_priority_ordering(self, recommendations_from_opt):
        """Recommendations should be ordered by priority (HIGH → MEDIUM → LOW)."""
        priority_order = {RecommendationPriority.HIGH: 0, RecommendationPriority.MEDIUM: 1, RecommendationPriority.LOW: 2}
        priorities = [priority_order[r.priority] for r in recommendations_from_opt]
        assert priorities == sorted(priorities)

    def test_expected_revenue_lift_finite(self, recommendations_from_opt):
        for r in recommendations_from_opt:
            assert np.isfinite(r.expected_revenue_lift)

    def test_spend_change_sign_matches_type(self, recommendations_from_opt):
        for r in recommendations_from_opt:
            if r.type == RecommendationType.INCREASE_BUDGET:
                assert r.spend_change >= 0, f"INCREASE_BUDGET should have positive spend_change: {r.title}"
            elif r.type == RecommendationType.DECREASE_BUDGET:
                assert r.spend_change <= 0, f"DECREASE_BUDGET should have negative spend_change: {r.title}"

    def test_reallocate_has_source_and_target(self, recommendations_from_opt):
        for r in recommendations_from_opt:
            if r.type == RecommendationType.REALLOCATE:
                assert r.source_campaign_id is not None
                assert r.target_campaign_id is not None

    def test_respects_top_n(self, opt_result, curves):
        recs = from_optimizer_result(opt_result, curves, top_n=2)
        assert len(recs) <= 2

    def test_top_n_zero_returns_empty(self, opt_result, curves):
        recs = from_optimizer_result(opt_result, curves, top_n=0)
        assert recs == []


# ---------------------------------------------------------------------------
# TestQuickRecommendations
# ---------------------------------------------------------------------------

class TestQuickRecommendations:
    def test_returns_list(self, quick_recommendations):
        assert isinstance(quick_recommendations, list)

    def test_all_recommendation_instances(self, quick_recommendations):
        for r in quick_recommendations:
            assert isinstance(r, Recommendation)

    def test_nonempty_for_valid_curves(self, quick_recommendations):
        assert len(quick_recommendations) >= 1

    def test_top_n_respected(self, curves):
        recs = generate_quick_recommendations(curves, top_n=2)
        assert len(recs) <= 2

    def test_expected_lift_finite(self, quick_recommendations):
        for r in quick_recommendations:
            assert np.isfinite(r.expected_revenue_lift)

    def test_spend_changes_have_correct_sign(self, quick_recommendations):
        for r in quick_recommendations:
            if r.type == RecommendationType.INCREASE_BUDGET:
                assert r.spend_change >= 0
            elif r.type == RecommendationType.DECREASE_BUDGET:
                assert r.spend_change <= 0

    def test_empty_curves_returns_empty(self):
        result = generate_quick_recommendations({})
        assert result == []
