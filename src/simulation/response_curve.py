"""
src/simulation/response_curve.py
==================================
Campaign-level spend→revenue response curves built from the feature store.

Each campaign gets a :class:`CampaignResponseCurve` that wraps a fitted
:class:`~src.simulation.hill_curve.HillCurve`.  The curve is fit on
attribution-mature daily (spend, revenue) pairs from the feature store.

Current spend is the trailing-30-day average over attribution-mature rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from src.simulation.hill_curve import HillCurve

logger = logging.getLogger(__name__)

CURRENT_SPEND_WINDOW_DAYS: Final[int] = 30
MIN_ROWS_FOR_CURVE: Final[int] = 15


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CampaignResponseCurve:
    """Spend→revenue response model for a single campaign."""

    campaign_id: str
    campaign_name: str
    platform: str
    hill: HillCurve

    # Current operating point
    avg_daily_spend: float       # trailing-30-day average
    avg_daily_revenue: float     # trailing-30-day average
    saturation_score: float      # 0=headroom, 1=saturated
    current_marginal_roas: float # d(revenue)/d(spend) at avg_daily_spend
    current_roas: float          # avg_daily_revenue / avg_daily_spend
    n_training_rows: int

    # Metadata
    is_reliable: bool            # False = linear fallback or poor fit
    r_squared: float

    def project(self, new_daily_spend: float) -> tuple[float, float]:
        """Project (projected_daily_revenue, lift_pct) at new_daily_spend.

        Parameters
        ----------
        new_daily_spend : float
            Proposed new average daily spend.

        Returns
        -------
        (projected_revenue, lift_pct)
            projected_revenue: expected daily revenue
            lift_pct: relative change vs current (negative = decline)
        """
        projected = float(self.hill.evaluate(new_daily_spend))
        baseline = self.avg_daily_revenue
        if baseline > 0:
            lift_pct = (projected - baseline) / baseline * 100.0
        else:
            lift_pct = 0.0 if projected <= 0 else float("inf")
        return projected, lift_pct

    def marginal_roas_at(self, spend: float) -> float:
        return self.hill.marginal_roas(spend)

    def saturation_at(self, spend: float) -> float:
        return self.hill.saturation_score(spend)


class ResponseCurveError(Exception):
    pass


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _current_spend_revenue(
    campaign_df: pd.DataFrame,
    window_days: int = CURRENT_SPEND_WINDOW_DAYS,
) -> tuple[float, float]:
    """Return trailing-window average daily (spend, revenue) for a campaign."""
    mature = campaign_df[campaign_df["attribution_mature"]] \
        if "attribution_mature" in campaign_df.columns \
        else campaign_df

    if mature.empty:
        mature = campaign_df  # fallback: use all rows

    recent = mature.sort_values("date").tail(window_days)
    avg_spend   = float(recent["spend"].mean()) if not recent.empty else 0.0
    avg_revenue = float(recent["revenue_attributed"].mean()) if not recent.empty else 0.0
    return max(avg_spend, 0.0), max(avg_revenue, 0.0)


def _fit_campaign_curve(
    campaign_df: pd.DataFrame,
    campaign_id: str,
    campaign_name: str,
    platform: str,
) -> CampaignResponseCurve:
    """Fit a Hill curve for one campaign from daily feature store rows."""
    # Use only attribution-mature rows for fitting
    if "attribution_mature" in campaign_df.columns:
        fit_df = campaign_df[campaign_df["attribution_mature"].fillna(False)]
    else:
        fit_df = campaign_df

    spend   = fit_df["spend"].values.astype(float)
    revenue = fit_df["revenue_attributed"].values.astype(float)

    hill = HillCurve.fit(spend, revenue)

    n_rows = len(fit_df)
    avg_spend, avg_revenue = _current_spend_revenue(campaign_df)

    saturation = hill.saturation_score(avg_spend)
    marginal   = hill.marginal_roas(avg_spend)
    roas       = (avg_revenue / avg_spend) if avg_spend > 0 else 0.0

    return CampaignResponseCurve(
        campaign_id=campaign_id,
        campaign_name=campaign_name,
        platform=platform,
        hill=hill,
        avg_daily_spend=avg_spend,
        avg_daily_revenue=avg_revenue,
        saturation_score=saturation,
        current_marginal_roas=marginal,
        current_roas=roas,
        n_training_rows=n_rows,
        is_reliable=hill.is_reliable,
        r_squared=hill.r_squared,
    )


def build_response_curves(
    fs: pd.DataFrame,
) -> dict[str, CampaignResponseCurve]:
    """Build campaign-level response curves from the feature store.

    Parameters
    ----------
    fs : pd.DataFrame
        Feature store produced by :func:`~src.features.feature_store.build_feature_store`.
        Must contain: campaign_id, campaign_name (or derived), platform, date,
        spend, revenue_attributed, attribution_mature.

    Returns
    -------
    dict[str, CampaignResponseCurve]
        Keys are campaign_id strings.
    """
    required = {"campaign_id", "platform", "date", "spend", "revenue_attributed"}
    missing = required - set(fs.columns)
    if missing:
        raise ResponseCurveError(f"Feature store missing columns: {missing}")

    if "campaign_name" not in fs.columns:
        fs = fs.copy()
        fs["campaign_name"] = fs["campaign_id"]

    curves: dict[str, CampaignResponseCurve] = {}
    for (cid, platform), grp in fs.groupby(["campaign_id", "platform"], sort=False):
        name = grp["campaign_name"].iloc[0] if "campaign_name" in grp.columns else cid
        if len(grp) < MIN_ROWS_FOR_CURVE:
            logger.debug(
                "Campaign %s has only %d rows — using linear fallback.", cid, len(grp)
            )
        curve = _fit_campaign_curve(grp, str(cid), str(name), str(platform))
        curves[str(cid)] = curve
        logger.debug(
            "Campaign %s | R²=%.3f | reliable=%s | saturation=%.2f",
            cid, curve.r_squared, curve.is_reliable, curve.saturation_score
        )

    logger.info(
        "Built response curves for %d campaigns | %d reliable",
        len(curves),
        sum(1 for c in curves.values() if c.is_reliable),
    )
    return curves


def compute_current_spend(
    curves: dict[str, CampaignResponseCurve],
) -> dict[str, float]:
    """Return {campaign_id: avg_daily_spend} from fitted curves."""
    return {cid: c.avg_daily_spend for cid, c in curves.items()}


def platform_summary(
    curves: dict[str, CampaignResponseCurve],
) -> pd.DataFrame:
    """Return a per-platform summary DataFrame with aggregated metrics."""
    rows = []
    for cid, c in curves.items():
        rows.append(
            {
                "campaign_id":     cid,
                "campaign_name":   c.campaign_name,
                "platform":        c.platform,
                "avg_daily_spend": c.avg_daily_spend,
                "avg_daily_revenue": c.avg_daily_revenue,
                "current_roas":    c.current_roas,
                "marginal_roas":   c.current_marginal_roas,
                "saturation_score": c.saturation_score,
                "is_reliable":     c.is_reliable,
                "r_squared":       c.r_squared,
            }
        )
    return pd.DataFrame(rows)
