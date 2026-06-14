"""
src/copilot/recommender.py
============================
Budget and campaign recommendations, all grounded in optimizer and curve data.

Every recommendation states:
- The specific source campaign and target campaign (where applicable)
- The exact spend amounts and projected revenue impact
- The marginal ROAS differential that justifies the move
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Final

import pandas as pd

from src.simulation.optimizer import OptimizationResult, CampaignAllocation
from src.simulation.response_curve import CampaignResponseCurve

logger = logging.getLogger(__name__)

MIN_MOVE_ABS: Final[float] = 10.0       # minimum daily $ to bother recommending
MIN_MOVE_PCT: Final[float] = 0.05       # minimum 5% change to bother recommending
MIN_MROAS_DIFFERENTIAL: Final[float] = 0.2   # marginal ROAS must differ by at least 0.2x


class RecommendationType(str, Enum):
    INCREASE_BUDGET      = "increase_budget"
    DECREASE_BUDGET      = "decrease_budget"
    REALLOCATE           = "reallocate"
    INVESTIGATE          = "investigate"
    MAINTAIN             = "maintain"


class RecommendationPriority(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


@dataclass
class Recommendation:
    type: RecommendationType
    priority: RecommendationPriority
    title: str
    rationale: str
    expected_revenue_lift: float    # daily $
    expected_revenue_lift_pct: float
    source_campaign_id: str | None   = None  # for REALLOCATE: take from here
    target_campaign_id: str | None   = None  # for REALLOCATE: give to here
    spend_change: float              = 0.0   # daily $ change (positive = increase)
    baseline_marginal_roas: float    = 0.0
    target_marginal_roas: float      = 0.0


class RecommenderError(Exception):
    pass


# ---------------------------------------------------------------------------
# Builder from optimizer output
# ---------------------------------------------------------------------------

def from_optimizer_result(
    opt: OptimizationResult,
    curves: dict[str, CampaignResponseCurve],
    top_n: int = 5,
) -> list[Recommendation]:
    """Convert an OptimizationResult into a ranked list of Recommendations.

    Returns at most *top_n* recommendations, ordered by expected revenue lift.
    """
    recommendations: list[Recommendation] = []

    for alloc in opt.optimal_allocations:
        spend_delta = alloc.spend_delta
        if abs(spend_delta) < MIN_MOVE_ABS:
            continue
        if alloc.baseline_spend > 0 and abs(spend_delta / alloc.baseline_spend) < MIN_MOVE_PCT:
            continue

        if spend_delta > 0:
            rtype    = RecommendationType.INCREASE_BUDGET
            priority = (
                RecommendationPriority.HIGH
                if alloc.revenue_delta_pct > 5.0
                else RecommendationPriority.MEDIUM
            )
            title = (
                f"Increase {alloc.platform.title()} · {alloc.campaign_id} "
                f"spend by ${spend_delta:,.0f}/day (+{alloc.spend_delta_pct:.0f}%)"
            )
            rationale = (
                f"The optimizer projects +${alloc.revenue_delta:,.0f}/day (+{alloc.revenue_delta_pct:.1f}%) "
                f"revenue at marginal ROAS ${alloc.optimal_marginal_roas:.2f}x. "
                f"Saturation at optimal spend: {alloc.saturation_at_optimal:.0%}."
            )
        else:
            rtype    = RecommendationType.DECREASE_BUDGET
            priority = RecommendationPriority.MEDIUM
            title = (
                f"Reduce {alloc.platform.title()} · {alloc.campaign_id} "
                f"spend by ${abs(spend_delta):,.0f}/day ({alloc.spend_delta_pct:.0f}%)"
            )
            rationale = (
                f"Campaign is {'saturated' if alloc.saturation_at_optimal > 0.75 else 'over-budgeted'} "
                f"at current spend. Reducing by ${abs(spend_delta):,.0f}/day frees budget "
                f"for higher-return campaigns. Revenue impact: {alloc.revenue_delta_pct:+.1f}%/day."
            )

        recommendations.append(
            Recommendation(
                type=rtype,
                priority=priority,
                title=title,
                rationale=rationale,
                expected_revenue_lift=alloc.revenue_delta,
                expected_revenue_lift_pct=alloc.revenue_delta_pct,
                target_campaign_id=alloc.campaign_id,
                spend_change=spend_delta,
                baseline_marginal_roas=alloc.baseline_marginal_roas,
                target_marginal_roas=alloc.optimal_marginal_roas,
            )
        )

    # Generate explicit reallocation recommendations
    increases = sorted(
        [r for r in recommendations if r.type == RecommendationType.INCREASE_BUDGET],
        key=lambda r: r.target_marginal_roas,
        reverse=True,
    )
    decreases = sorted(
        [r for r in recommendations if r.type == RecommendationType.DECREASE_BUDGET],
        key=lambda r: abs(r.spend_change),
        reverse=True,
    )

    for src, tgt in zip(decreases[:3], increases[:3]):
        mroas_diff = tgt.target_marginal_roas - src.baseline_marginal_roas
        if mroas_diff < MIN_MROAS_DIFFERENTIAL:
            continue
        amount = min(abs(src.spend_change), tgt.spend_change)
        if amount < MIN_MOVE_ABS:
            continue
        recommendations.append(
            Recommendation(
                type=RecommendationType.REALLOCATE,
                priority=RecommendationPriority.HIGH,
                title=(
                    f"Move ${amount:,.0f}/day from {src.target_campaign_id} "
                    f"→ {tgt.target_campaign_id}"
                ),
                rationale=(
                    f"Marginal ROAS differential: {tgt.target_campaign_id} earns "
                    f"${tgt.target_marginal_roas:.2f}x vs {src.target_campaign_id} earns "
                    f"${src.baseline_marginal_roas:.2f}x. "
                    f"Moving ${amount:,.0f}/day is projected to lift daily revenue by "
                    f"${tgt.expected_revenue_lift + abs(src.expected_revenue_lift):.0f}."
                ),
                expected_revenue_lift=tgt.expected_revenue_lift + abs(src.expected_revenue_lift),
                expected_revenue_lift_pct=(
                    tgt.expected_revenue_lift_pct + abs(src.expected_revenue_lift_pct)
                ) / 2.0,
                source_campaign_id=src.target_campaign_id,
                target_campaign_id=tgt.target_campaign_id,
                spend_change=amount,
                baseline_marginal_roas=src.baseline_marginal_roas,
                target_marginal_roas=tgt.target_marginal_roas,
            )
        )

    # Sort: REALLOCATE first, then by expected lift descending
    priority_order = {
        RecommendationPriority.HIGH:   0,
        RecommendationPriority.MEDIUM: 1,
        RecommendationPriority.LOW:    2,
    }
    recommendations.sort(
        key=lambda r: (priority_order.get(r.priority, 9), -r.expected_revenue_lift)
    )
    logger.info("Recommender generated %d recommendations.", len(recommendations[:top_n]))
    return recommendations[:top_n]


# ---------------------------------------------------------------------------
# Builder from curves alone (no optimizer output)
# ---------------------------------------------------------------------------

def generate_quick_recommendations(
    curves: dict[str, CampaignResponseCurve],
    top_n: int = 5,
) -> list[Recommendation]:
    """Fast heuristic recommendations based solely on marginal ROAS differentials.

    Does not require running the optimizer.
    """
    recommendations: list[Recommendation] = []
    if not curves:
        return recommendations

    sorted_by_mroas = sorted(
        curves.values(),
        key=lambda c: c.current_marginal_roas,
        reverse=True,
    )

    total_spend = sum(c.avg_daily_spend for c in curves.values())
    if total_spend <= 0:
        return recommendations

    top_mroas    = sorted_by_mroas[:3]
    bottom_mroas = sorted_by_mroas[-3:][::-1]

    for c in top_mroas:
        if c.saturation_score > 0.80:
            continue
        increase_amt = c.avg_daily_spend * 0.15
        if increase_amt < MIN_MOVE_ABS:
            continue
        proj_lift, lift_pct = c.project(c.avg_daily_spend + increase_amt)
        revenue_lift = proj_lift - c.avg_daily_revenue
        recommendations.append(
            Recommendation(
                type=RecommendationType.INCREASE_BUDGET,
                priority=RecommendationPriority.HIGH,
                title=(
                    f"Boost {c.platform.title()} · {c.campaign_id} "
                    f"by ${increase_amt:,.0f}/day (+15%)"
                ),
                rationale=(
                    f"Highest marginal ROAS in portfolio: ${c.current_marginal_roas:.2f}x. "
                    f"Saturation at {c.saturation_score:.0%} — still room to grow. "
                    f"Projected daily revenue lift: +${revenue_lift:,.0f}."
                ),
                expected_revenue_lift=revenue_lift,
                expected_revenue_lift_pct=lift_pct,
                target_campaign_id=c.campaign_id,
                spend_change=increase_amt,
                baseline_marginal_roas=c.current_marginal_roas,
                target_marginal_roas=c.hill.marginal_roas(c.avg_daily_spend + increase_amt),
            )
        )

    for c in bottom_mroas:
        if c.avg_daily_spend < MIN_MOVE_ABS * 2:
            continue
        decrease_amt = c.avg_daily_spend * 0.15
        if decrease_amt < MIN_MOVE_ABS:
            continue
        proj_rev, _ = c.project(c.avg_daily_spend - decrease_amt)
        revenue_impact = proj_rev - c.avg_daily_revenue  # negative
        recommendations.append(
            Recommendation(
                type=RecommendationType.DECREASE_BUDGET,
                priority=RecommendationPriority.MEDIUM,
                title=(
                    f"Reduce {c.platform.title()} · {c.campaign_id} "
                    f"by ${decrease_amt:,.0f}/day (-15%)"
                ),
                rationale=(
                    f"Lowest marginal ROAS in portfolio: ${c.current_marginal_roas:.2f}x. "
                    f"Saturation at {c.saturation_score:.0%}. "
                    f"Revenue impact of cut: ${revenue_impact:,.0f}/day."
                ),
                expected_revenue_lift=revenue_impact,
                expected_revenue_lift_pct=(
                    revenue_impact / c.avg_daily_revenue * 100.0
                    if c.avg_daily_revenue > 0 else 0.0
                ),
                source_campaign_id=c.campaign_id,
                spend_change=-decrease_amt,
                baseline_marginal_roas=c.current_marginal_roas,
            )
        )

    recommendations.sort(
        key=lambda r: (0 if r.type == RecommendationType.REALLOCATE else 1, -r.expected_revenue_lift)
    )
    return recommendations[:top_n]
