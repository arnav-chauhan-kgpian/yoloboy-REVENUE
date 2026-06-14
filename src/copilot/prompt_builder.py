"""
src/copilot/prompt_builder.py
==============================
Build grounded prompts for the LLM copilot.

All numbers embedded in prompts are sourced from actual data objects passed in.
The system prompt explicitly instructs the model not to invent numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from src.copilot.insight_engine import Insight
from src.copilot.risk_detector import Risk
from src.copilot.recommender import Recommendation
from src.simulation.optimizer import OptimizationResult
from src.simulation.response_curve import CampaignResponseCurve

_SYSTEM_PROMPT = """\
You are meridian, an AI revenue forecasting and budget optimization copilot for ecommerce advertisers.

CRITICAL RULES:
1. ONLY use numbers, metrics, and values that appear in the CONTEXT section below.
2. Do NOT invent, estimate, or hallucinate any revenue figures, percentages, or campaign names.
3. If a question cannot be answered from the provided context, say "I don't have data on that."
4. Be specific — reference exact dollar amounts and percentages from the context.
5. Be actionable — every recommendation must have a clear reason grounded in the data.

OUTPUT FORMAT: Return valid JSON matching this schema:
{
  "summary": "2-3 sentence executive summary with specific numbers from context",
  "risks": ["risk explanation with specific metric values", ...],
  "opportunities": ["opportunity with specific numbers", ...],
  "recommendations": ["actionable recommendation with rationale", ...],
  "confidence": 0.0  // 0-1, your confidence the analysis is correct given available data
}
"""


@dataclass
class PromptContext:
    """All data needed to build a grounded LLM prompt."""

    # Forecast metrics
    forecast_period_days: int = 7
    total_p50_revenue: float  = 0.0
    total_p10_revenue: float  = 0.0
    total_p90_revenue: float  = 0.0
    revenue_wow_change_pct: float = 0.0

    # Spend context
    total_daily_spend: float  = 0.0
    current_roas: float       = 0.0

    # Platform breakdown (platform → daily revenue)
    platform_revenues: dict[str, float] = field(default_factory=dict)
    platform_spends:   dict[str, float] = field(default_factory=dict)

    # Top campaigns by revenue
    top_campaigns: list[dict[str, Any]] = field(default_factory=list)

    # Optimization result summary
    optimization_revenue_lift_pct: float = 0.0
    optimization_revenue_lift_abs: float = 0.0
    optimization_risk_score: float       = 0.0

    # Pre-generated insights, risks, recommendations
    insights: list[Insight]         = field(default_factory=list)
    risks: list[Risk]               = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)

    # Saturation context
    saturated_campaigns: list[str]    = field(default_factory=list)
    headroom_campaigns: list[str]     = field(default_factory=list)
    low_utilization_campaigns: list[str] = field(default_factory=list)


def build_context_from_data(
    forecasts: "pd.DataFrame | None" = None,
    fs: "pd.DataFrame | None"        = None,
    curves: dict[str, CampaignResponseCurve] | None = None,
    opt_result: OptimizationResult | None = None,
    insights: list[Insight] | None    = None,
    risks: list[Risk] | None          = None,
    recommendations: list[Recommendation] | None = None,
    window_days: int = 7,
) -> PromptContext:
    """Populate a PromptContext from available data objects."""
    import numpy as np
    import pandas as pd

    ctx = PromptContext(forecast_period_days=window_days)

    # --- Forecast metrics ---
    if forecasts is not None and not forecasts.empty:
        cutoff = forecasts["date"].max() - pd.Timedelta(days=window_days - 1)
        recent = forecasts[forecasts["date"] >= cutoff]

        ctx.total_p50_revenue = float(recent["p50"].sum()) if "p50" in recent.columns else 0.0
        ctx.total_p10_revenue = float(recent["p10"].sum()) if "p10" in recent.columns else 0.0
        ctx.total_p90_revenue = float(recent["p90"].sum()) if "p90" in recent.columns else 0.0

        # WoW change
        prior_cutoff = cutoff - pd.Timedelta(days=window_days)
        prior = forecasts[(forecasts["date"] >= prior_cutoff) & (forecasts["date"] < cutoff)]
        prior_rev = float(prior["p50"].sum()) if not prior.empty and "p50" in prior.columns else 0.0
        if prior_rev > 0:
            ctx.revenue_wow_change_pct = (ctx.total_p50_revenue - prior_rev) / prior_rev * 100.0

        # Platform breakdown
        if "platform" in recent.columns and "p50" in recent.columns:
            ctx.platform_revenues = {
                str(p): float(g["p50"].sum())
                for p, g in recent.groupby("platform")
            }

        # Top campaigns
        if "campaign_id" in recent.columns and "p50" in recent.columns:
            top = (
                recent.groupby("campaign_id")["p50"].sum()
                .sort_values(ascending=False)
                .head(5)
            )
            total = ctx.total_p50_revenue or 1.0
            ctx.top_campaigns = [
                {"campaign_id": str(cid), "revenue": float(rev), "share_pct": float(rev / total * 100)}
                for cid, rev in top.items()
            ]

    # --- Spend context ---
    if curves:
        ctx.total_daily_spend = sum(c.avg_daily_spend for c in curves.values())
        total_rev = sum(c.avg_daily_revenue for c in curves.values())
        ctx.current_roas = total_rev / ctx.total_daily_spend if ctx.total_daily_spend > 0 else 0.0

        # Platform spend
        from collections import defaultdict
        pspend: dict[str, float] = defaultdict(float)
        for c in curves.values():
            pspend[c.platform] += c.avg_daily_spend
        ctx.platform_spends = dict(pspend)

        # Saturation buckets
        ctx.saturated_campaigns    = [cid for cid, c in curves.items() if c.saturation_score >= 0.75]
        ctx.headroom_campaigns     = [cid for cid, c in curves.items() if c.saturation_score < 0.40]
        ctx.low_utilization_campaigns = [
            cid for cid, c in curves.items()
            if c.avg_daily_spend < 0.3 * (ctx.total_daily_spend / max(len(curves), 1))
        ]

    # --- Optimization summary ---
    if opt_result is not None:
        ctx.optimization_revenue_lift_pct = opt_result.revenue_lift_pct
        ctx.optimization_revenue_lift_abs  = opt_result.revenue_lift
        ctx.optimization_risk_score        = opt_result.risk_score

    ctx.insights        = insights or []
    ctx.risks           = risks or []
    ctx.recommendations = recommendations or []

    return ctx


def build_context_json(ctx: PromptContext) -> str:
    """Serialise PromptContext to a compact JSON string for embedding in prompts."""
    data = {
        "forecast": {
            "period_days":      ctx.forecast_period_days,
            "total_p50":        round(ctx.total_p50_revenue, 2),
            "total_p10":        round(ctx.total_p10_revenue, 2),
            "total_p90":        round(ctx.total_p90_revenue, 2),
            "wow_change_pct":   round(ctx.revenue_wow_change_pct, 2),
        },
        "spend": {
            "total_daily":      round(ctx.total_daily_spend, 2),
            "current_roas":     round(ctx.current_roas, 3),
            "by_platform":      {p: round(v, 2) for p, v in ctx.platform_spends.items()},
        },
        "revenue_by_platform": {p: round(v, 2) for p, v in ctx.platform_revenues.items()},
        "top_campaigns":        ctx.top_campaigns,
        "optimization": {
            "lift_pct":    round(ctx.optimization_revenue_lift_pct, 2),
            "lift_abs":    round(ctx.optimization_revenue_lift_abs, 2),
            "risk_score":  round(ctx.optimization_risk_score, 3),
        },
        "saturation": {
            "saturated":       ctx.saturated_campaigns[:10],
            "has_headroom":    ctx.headroom_campaigns[:10],
            "low_utilization": ctx.low_utilization_campaigns[:10],
        },
        "insights": [
            {
                "type":        i.type,
                "severity":    i.severity,
                "title":       i.title,
                "metric":      i.metric_name,
                "value":       round(i.metric_value, 3),
                "unit":        i.metric_unit,
            }
            for i in ctx.insights[:10]
        ],
        "risks": [
            {
                "type":       r.type,
                "severity":   r.severity,
                "title":      r.title,
                "metric":     r.metric_name,
                "value":      round(r.metric_value, 3),
                "threshold":  round(r.threshold, 3),
                "campaign":   r.campaign_id,
                "platform":   r.platform,
            }
            for r in ctx.risks[:10]
        ],
        "recommendations": [
            {
                "type":           rec.type,
                "priority":       rec.priority,
                "title":          rec.title,
                "revenue_lift":   round(rec.expected_revenue_lift, 2),
                "lift_pct":       round(rec.expected_revenue_lift_pct, 2),
                "spend_change":   round(rec.spend_change, 2),
                "from_campaign":  rec.source_campaign_id,
                "to_campaign":    rec.target_campaign_id,
            }
            for rec in ctx.recommendations[:5]
        ],
    }
    return json.dumps(data, indent=2)


def build_analysis_prompt(ctx: PromptContext, question: str | None = None) -> str:
    """Build the complete user-turn prompt with embedded context."""
    context_json = build_context_json(ctx)

    base = f"""CONTEXT (all numbers below are actual data — use ONLY these):
{context_json}

TASK: Based strictly on the CONTEXT above, generate an analysis with:
1. A 2-3 sentence executive summary mentioning the most important numbers.
2. The top 3 risks (referencing specific metric values from context).
3. The top 3 opportunities (referencing specific metrics and campaigns).
4. 3-5 prioritised, actionable budget recommendations with expected impact.
5. A confidence score (0.0-1.0) for your analysis.

Return ONLY valid JSON matching the required schema. Do not add commentary outside the JSON."""

    if question:
        base += f"""

USER QUESTION: {question}
Answer the user question as part of the "summary" field. Keep all other fields populated."""

    return base


def build_system_prompt() -> str:
    return _SYSTEM_PROMPT
