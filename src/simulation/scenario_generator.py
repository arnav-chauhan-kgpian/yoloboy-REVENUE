"""
src/simulation/scenario_generator.py
======================================
Budget scenario generation — apply spend changes and project revenue outcomes.

Supported change types
----------------------
- Platform-level percentage change: ``{platform: 1.20}`` means +20% for all
  campaigns on that platform.
- Campaign-level absolute spend: ``{campaign_id: 500.0}`` sets daily spend directly.
- Mixed: platform changes applied first, then campaign overrides on top.

All projections use the fitted :class:`~src.simulation.hill_curve.HillCurve`
for each campaign.  Confidence intervals are derived from the Hill curve's R²:
lower R² → wider interval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from src.simulation.response_curve import CampaignResponseCurve

logger = logging.getLogger(__name__)

# Uncertainty multiplier: 1 - R² * factor → fraction of revenue as ±CI half-width
_CI_UNCERTAINTY_SCALE: Final[float] = 0.35


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CampaignProjection:
    campaign_id: str
    campaign_name: str
    platform: str
    baseline_daily_spend: float
    new_daily_spend: float
    spend_delta: float          # new - baseline
    spend_delta_pct: float      # (new - baseline) / baseline * 100
    baseline_daily_revenue: float
    projected_daily_revenue: float
    revenue_delta: float
    revenue_delta_pct: float
    baseline_marginal_roas: float
    new_marginal_roas: float


@dataclass
class ScenarioResult:
    scenario_name: str
    description: str

    # Totals (daily)
    baseline_total_spend: float
    new_total_spend: float
    baseline_total_revenue: float
    projected_total_revenue: float

    # Lifts
    revenue_lift: float              # projected - baseline (daily)
    revenue_lift_pct: float          # lift / baseline * 100

    # ROAS
    baseline_roas: float
    projected_roas: float

    # Confidence interval (P10, P90) on projected daily revenue
    ci_low: float
    ci_high: float

    # Risk score 0-1
    risk_score: float

    # Per-campaign details
    campaign_projections: list[CampaignProjection] = field(default_factory=list)


class ScenarioError(Exception):
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _confidence_interval(
    curves: dict[str, CampaignResponseCurve],
    new_spend: dict[str, float],
    projected_revenue: float,
) -> tuple[float, float]:
    """Estimate (P10, P90) for the projected total revenue.

    Campaigns with unreliable curves (low R²) contribute more uncertainty.
    """
    total_spend = sum(new_spend.values())
    if total_spend <= 0:
        return projected_revenue, projected_revenue

    # Weighted uncertainty: each campaign weighted by its share of new spend
    uncertainty = 0.0
    for cid, spend in new_spend.items():
        if cid not in curves or spend <= 0:
            continue
        c = curves[cid]
        weight = spend / total_spend
        # More uncertainty for unreliable fits
        r2 = c.r_squared if c.is_reliable else 0.0
        uncertainty += weight * (1.0 - r2) * _CI_UNCERTAINTY_SCALE

    half_width = projected_revenue * uncertainty
    ci_low  = max(0.0, projected_revenue - half_width * 1.28)
    ci_high = projected_revenue + half_width * 1.28
    return ci_low, ci_high


def _risk_score(
    curves: dict[str, CampaignResponseCurve],
    new_spend: dict[str, float],
    total_spend: float,
) -> float:
    """Aggregate risk score 0-1.

    Factors:
    - Saturation: spending past the knee raises risk.
    - Unreliable fits: can't predict response → risk.
    - Budget concentration: large share in one campaign → risk.
    """
    if not curves or total_spend <= 0:
        return 0.0

    saturation_risk  = 0.0
    reliability_risk = 0.0
    concentration_scores = []

    for cid, spend in new_spend.items():
        if cid not in curves or spend <= 0:
            continue
        c = curves[cid]
        share = spend / total_spend
        concentration_scores.append(share)

        sat = c.hill.saturation_score(spend)
        saturation_risk += share * sat

        reliability_risk += share * (0.0 if c.is_reliable else 1.0)

    # Herfindahl concentration index (0=diverse, 1=one campaign)
    concentration = sum(s**2 for s in concentration_scores) if concentration_scores else 0.0

    risk = 0.5 * saturation_risk + 0.35 * reliability_risk + 0.15 * concentration
    return float(np.clip(risk, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Core application functions
# ---------------------------------------------------------------------------

def apply_scenario(
    curves: dict[str, CampaignResponseCurve],
    platform_multipliers: dict[str, float] | None = None,
    campaign_overrides: dict[str, float] | None = None,
    scenario_name: str = "Custom Scenario",
    description: str = "",
) -> ScenarioResult:
    """Apply budget changes and return projected ScenarioResult.

    Parameters
    ----------
    curves :
        Campaign response curves (from :func:`~src.simulation.response_curve.build_response_curves`).
    platform_multipliers :
        ``{platform: multiplier}`` — all campaigns on that platform are scaled
        by this factor.  E.g. ``{"google": 1.20}`` → +20% spend on Google.
    campaign_overrides :
        ``{campaign_id: new_daily_spend}`` — hard-set specific campaign spend.
        Applied after platform multipliers.
    scenario_name, description :
        Labels for reporting.

    Returns
    -------
    ScenarioResult
    """
    platform_multipliers = platform_multipliers or {}
    campaign_overrides   = campaign_overrides or {}

    if not curves:
        raise ScenarioError("No response curves provided.")

    baseline_spend:  dict[str, float] = {}
    new_spend:       dict[str, float] = {}

    for cid, c in curves.items():
        bs = c.avg_daily_spend
        baseline_spend[cid] = bs

        # Step 1: apply platform multiplier
        mult = platform_multipliers.get(c.platform, 1.0)
        ns = bs * mult

        # Step 2: campaign override takes precedence
        if cid in campaign_overrides:
            ns = campaign_overrides[cid]

        new_spend[cid] = max(ns, 0.0)

    # Project revenue for each campaign
    projections: list[CampaignProjection] = []
    total_baseline_rev = 0.0
    total_projected_rev = 0.0

    for cid, c in curves.items():
        bs = baseline_spend[cid]
        ns = new_spend[cid]
        baseline_rev = c.avg_daily_revenue

        # When spend is unchanged, keep revenue exactly at baseline to avoid
        # Hill-curve residual noise creating a phantom lift at zero spend delta.
        if abs(ns - bs) < 1e-9:
            projected_rev = baseline_rev
        else:
            projected_rev, _ = c.project(ns)

        total_baseline_rev  += baseline_rev
        total_projected_rev += projected_rev

        bs_margs = c.hill.marginal_roas(bs)
        ns_margs = c.hill.marginal_roas(ns)

        spend_delta = ns - bs
        spend_delta_pct = (spend_delta / bs * 100.0) if bs > 0 else 0.0
        rev_delta = projected_rev - baseline_rev
        rev_delta_pct = (rev_delta / baseline_rev * 100.0) if baseline_rev > 0 else 0.0

        projections.append(
            CampaignProjection(
                campaign_id=cid,
                campaign_name=c.campaign_name,
                platform=c.platform,
                baseline_daily_spend=bs,
                new_daily_spend=ns,
                spend_delta=spend_delta,
                spend_delta_pct=spend_delta_pct,
                baseline_daily_revenue=baseline_rev,
                projected_daily_revenue=projected_rev,
                revenue_delta=rev_delta,
                revenue_delta_pct=rev_delta_pct,
                baseline_marginal_roas=bs_margs,
                new_marginal_roas=ns_margs,
            )
        )

    total_baseline_spend = sum(baseline_spend.values())
    total_new_spend      = sum(new_spend.values())

    revenue_lift     = total_projected_rev - total_baseline_rev
    revenue_lift_pct = (revenue_lift / total_baseline_rev * 100.0) if total_baseline_rev > 0 else 0.0

    baseline_roas  = (total_baseline_rev / total_baseline_spend) if total_baseline_spend > 0 else 0.0
    projected_roas = (total_projected_rev / total_new_spend)     if total_new_spend > 0     else 0.0

    ci_low, ci_high = _confidence_interval(curves, new_spend, total_projected_rev)
    risk = _risk_score(curves, new_spend, total_new_spend)

    return ScenarioResult(
        scenario_name=scenario_name,
        description=description,
        baseline_total_spend=total_baseline_spend,
        new_total_spend=total_new_spend,
        baseline_total_revenue=total_baseline_rev,
        projected_total_revenue=total_projected_rev,
        revenue_lift=revenue_lift,
        revenue_lift_pct=revenue_lift_pct,
        baseline_roas=baseline_roas,
        projected_roas=projected_roas,
        ci_low=ci_low,
        ci_high=ci_high,
        risk_score=risk,
        campaign_projections=projections,
    )


def compare_scenarios(results: list[ScenarioResult]) -> pd.DataFrame:
    """Return a summary DataFrame comparing multiple ScenarioResults."""
    rows = []
    for r in results:
        rows.append(
            {
                "scenario":              r.scenario_name,
                "baseline_revenue":      r.baseline_total_revenue,
                "projected_revenue":     r.projected_total_revenue,
                "revenue_lift":          r.revenue_lift,
                "revenue_lift_pct":      r.revenue_lift_pct,
                "baseline_spend":        r.baseline_total_spend,
                "new_spend":             r.new_total_spend,
                "baseline_roas":         r.baseline_roas,
                "projected_roas":        r.projected_roas,
                "ci_low":                r.ci_low,
                "ci_high":               r.ci_high,
                "risk_score":            r.risk_score,
            }
        )
    return pd.DataFrame(rows)


def generate_standard_scenarios(
    curves: dict[str, CampaignResponseCurve],
) -> list[ScenarioResult]:
    """Generate a default set of scenarios for dashboard display.

    Scenarios:
    - Baseline (no change)
    - Google +20% / Meta -10%
    - Google -20% / Meta +20%
    - All platforms +10%
    - All platforms -10%
    """
    scenarios = [
        (
            "Baseline",
            "No budget change.",
            {},
            {},
        ),
        (
            "Google Boost",
            "Increase Google spend by 20%, reduce Meta by 10%.",
            {"google": 1.20, "meta": 0.90},
            {},
        ),
        (
            "Meta Boost",
            "Reduce Google by 20%, increase Meta by 20%.",
            {"google": 0.80, "meta": 1.20},
            {},
        ),
        (
            "Scale Up",
            "Increase all platforms by 10%.",
            {"google": 1.10, "meta": 1.10, "bing": 1.10},
            {},
        ),
        (
            "Scale Down",
            "Reduce all platforms by 10%.",
            {"google": 0.90, "meta": 0.90, "bing": 0.90},
            {},
        ),
    ]

    results = []
    for name, desc, platform_mults, campaign_ovr in scenarios:
        try:
            r = apply_scenario(
                curves,
                platform_multipliers=platform_mults,
                campaign_overrides=campaign_ovr,
                scenario_name=name,
                description=desc,
            )
            results.append(r)
        except ScenarioError as exc:
            logger.warning("Scenario '%s' failed: %s", name, exc)

    return results
