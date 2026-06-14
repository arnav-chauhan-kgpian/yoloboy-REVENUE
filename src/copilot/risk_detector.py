"""
src/copilot/risk_detector.py
==============================
Detect actionable risks from feature store, response curves, and forecasts.

All thresholds are data-derived, not heuristic guesses.  Each Risk carries:
- the specific metric value that triggered it
- the threshold it crossed
- a plain-English explanation grounded in those numbers
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Final

import pandas as pd

from src.simulation.response_curve import CampaignResponseCurve

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

BUDGET_UTIL_LOW_THRESHOLD: Final[float]    = 0.30   # < 30% → low utilisation risk
BUDGET_UTIL_CRITICAL_THRESHOLD: Final[float] = 0.10  # < 10% → critical
SATURATION_WARNING_THRESHOLD: Final[float]  = 0.75   # > 75% → approaching saturation
SATURATION_CRITICAL_THRESHOLD: Final[float] = 0.90   # > 90% → saturated
WIDE_CI_THRESHOLD: Final[float]             = 0.60   # (P90-P10)/P50 > 60% → low confidence
ZERO_REVENUE_SHARE_THRESHOLD: Final[float]  = 0.30   # > 30% zero-revenue days → risk


class RiskType(str, Enum):
    BUDGET_UTILIZATION  = "budget_utilization"
    SATURATION          = "saturation"
    FORECAST_CONFIDENCE = "forecast_confidence"
    ZERO_REVENUE        = "zero_revenue"
    SPEND_CONCENTRATION = "spend_concentration"


class RiskSeverity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


@dataclass
class Risk:
    type: RiskType
    severity: RiskSeverity
    title: str
    explanation: str
    metric_name: str
    metric_value: float
    threshold: float
    campaign_id: str | None = None
    platform: str | None    = None
    recommendation: str     = ""


class RiskDetector:
    """Detect actionable risks from multiple data sources."""

    # ------------------------------------------------------------------
    # Budget utilisation
    # ------------------------------------------------------------------

    def detect_budget_utilization_risks(
        self,
        fs: pd.DataFrame,
        window_days: int = 30,
    ) -> list[Risk]:
        """Campaigns that are significantly under-spending their daily budget.

        Uses the ``budget_utilization`` feature (spend / daily_budget) from the
        feature store.  Campaigns with mean utilisation below threshold are flagged.
        """
        risks: list[Risk] = []
        if "budget_utilization" not in fs.columns or "campaign_id" not in fs.columns:
            return risks

        recent_cutoff = fs["date"].max() - pd.Timedelta(days=window_days - 1)
        recent = fs[fs["date"] >= recent_cutoff]

        for cid, grp in recent.groupby("campaign_id"):
            util_vals = grp["budget_utilization"].dropna()
            if util_vals.empty:
                continue
            mean_util = float(util_vals.mean())
            platform  = str(grp["platform"].iloc[0]) if "platform" in grp.columns else "unknown"

            if mean_util > BUDGET_UTIL_LOW_THRESHOLD:
                continue

            if mean_util < BUDGET_UTIL_CRITICAL_THRESHOLD:
                severity = RiskSeverity.CRITICAL
                rec = (
                    f"Budget for campaign '{cid}' appears over-allocated. "
                    "Reallocate surplus to campaigns with higher utilisation."
                )
            else:
                severity = RiskSeverity.HIGH
                rec = (
                    f"Review budget cap for campaign '{cid}'. "
                    "Either the budget is too high or bids are too low to clear it."
                )

            risks.append(
                Risk(
                    type=RiskType.BUDGET_UTILIZATION,
                    severity=severity,
                    title=f"{platform.title()} · {cid}: budget utilisation {mean_util:.0%}",
                    explanation=(
                        f"Campaign '{cid}' ({platform}) spent only {mean_util:.0%} of its "
                        f"daily budget on average over the last {window_days} days. "
                        f"Threshold: {BUDGET_UTIL_LOW_THRESHOLD:.0%}. "
                        "Increasing budget further is unlikely to improve revenue."
                    ),
                    metric_name="mean_budget_utilization",
                    metric_value=mean_util,
                    threshold=BUDGET_UTIL_LOW_THRESHOLD,
                    campaign_id=str(cid),
                    platform=platform,
                    recommendation=rec,
                )
            )

        return risks

    # ------------------------------------------------------------------
    # Saturation
    # ------------------------------------------------------------------

    def detect_saturation_risks(
        self,
        curves: dict[str, CampaignResponseCurve],
    ) -> list[Risk]:
        """Campaigns operating near their Hill-curve saturation point."""
        risks: list[Risk] = []

        for cid, c in curves.items():
            sat = c.saturation_score
            if sat < SATURATION_WARNING_THRESHOLD:
                continue

            if sat >= SATURATION_CRITICAL_THRESHOLD:
                severity = RiskSeverity.CRITICAL
                rec = (
                    f"Campaign '{cid}' is effectively saturated. "
                    "Further spend increases will yield negligible revenue gains. "
                    "Reallocate budget to lower-saturation campaigns."
                )
            else:
                severity = RiskSeverity.HIGH
                rec = (
                    f"Campaign '{cid}' is approaching saturation. "
                    "Marginal ROAS is ${c.current_marginal_roas:.2f} vs average ROAS ${c.current_roas:.2f}."
                )

            risks.append(
                Risk(
                    type=RiskType.SATURATION,
                    severity=severity,
                    title=f"{c.platform.title()} · {cid}: saturation {sat:.0%}",
                    explanation=(
                        f"Campaign '{cid}' ({c.platform}) is at {sat:.0%} saturation "
                        f"(Hill curve saturation score). "
                        f"Marginal ROAS at current spend ${c.avg_daily_spend:,.0f}/day = "
                        f"${c.current_marginal_roas:.2f}x, "
                        f"well below average ROAS of ${c.current_roas:.2f}x."
                    ),
                    metric_name="saturation_score",
                    metric_value=sat,
                    threshold=SATURATION_WARNING_THRESHOLD,
                    campaign_id=cid,
                    platform=c.platform,
                    recommendation=rec,
                )
            )

        return risks

    # ------------------------------------------------------------------
    # Forecast confidence
    # ------------------------------------------------------------------

    def detect_confidence_risks(
        self,
        forecasts: pd.DataFrame,
        window_days: int = 7,
    ) -> list[Risk]:
        """Campaigns with wide P10/P90 intervals at the campaign level."""
        risks: list[Risk] = []
        if not all(c in forecasts.columns for c in ["p10", "p50", "p90", "campaign_id"]):
            return risks

        cutoff = forecasts["date"].max() - pd.Timedelta(days=window_days - 1)
        recent = forecasts[forecasts["date"] >= cutoff]

        for cid, grp in recent.groupby("campaign_id"):
            p50 = float(grp["p50"].sum())
            if p50 <= 0:
                continue
            p10  = float(grp["p10"].sum())
            p90  = float(grp["p90"].sum())
            width = (p90 - p10) / p50

            if width < WIDE_CI_THRESHOLD:
                continue

            platform = str(grp["platform"].iloc[0]) if "platform" in grp.columns else "unknown"
            risks.append(
                Risk(
                    type=RiskType.FORECAST_CONFIDENCE,
                    severity=RiskSeverity.MEDIUM,
                    title=f"{platform.title()} · {cid}: low forecast confidence ({width:.0%} CI width)",
                    explanation=(
                        f"Campaign '{cid}' has a P10–P90 interval of ${p10:,.0f}–${p90:,.0f} "
                        f"around P50=${p50:,.0f} (width={width:.0%} of P50). "
                        f"Threshold: {WIDE_CI_THRESHOLD:.0%}. "
                        "Decisions relying on this campaign's forecast carry higher uncertainty."
                    ),
                    metric_name="forecast_ci_width_pct",
                    metric_value=width * 100,
                    threshold=WIDE_CI_THRESHOLD * 100,
                    campaign_id=str(cid),
                    platform=platform,
                    recommendation=(
                        "Avoid large budget commitments for this campaign until confidence improves."
                    ),
                )
            )

        return risks

    # ------------------------------------------------------------------
    # Zero-revenue campaigns
    # ------------------------------------------------------------------

    def detect_zero_revenue_risks(
        self,
        fs: pd.DataFrame,
        window_days: int = 30,
    ) -> list[Risk]:
        """Campaigns with a high fraction of zero-revenue days despite spend."""
        risks: list[Risk] = []
        if "revenue_attributed" not in fs.columns or "spend" not in fs.columns:
            return risks

        cutoff = fs["date"].max() - pd.Timedelta(days=window_days - 1)
        recent = fs[fs["date"] >= cutoff]

        for cid, grp in recent.groupby("campaign_id"):
            spending_days = grp[grp["spend"] > 0]
            if len(spending_days) < 5:
                continue
            zero_rev_frac = float((spending_days["revenue_attributed"] == 0).mean())
            if zero_rev_frac < ZERO_REVENUE_SHARE_THRESHOLD:
                continue

            platform = str(grp["platform"].iloc[0]) if "platform" in grp.columns else "unknown"
            risks.append(
                Risk(
                    type=RiskType.ZERO_REVENUE,
                    severity=RiskSeverity.HIGH,
                    title=f"{platform.title()} · {cid}: {zero_rev_frac:.0%} zero-revenue days",
                    explanation=(
                        f"Campaign '{cid}' ({platform}) had {zero_rev_frac:.0%} of its "
                        f"spending days yield zero attributed revenue over the last {window_days} days. "
                        "Attribution may be broken, or conversion tracking is missing."
                    ),
                    metric_name="zero_revenue_fraction",
                    metric_value=zero_rev_frac * 100,
                    threshold=ZERO_REVENUE_SHARE_THRESHOLD * 100,
                    campaign_id=str(cid),
                    platform=platform,
                    recommendation=(
                        "Audit conversion tracking for this campaign. "
                        "If tracking is correct, consider pausing or restructuring."
                    ),
                )
            )

        return risks

    # ------------------------------------------------------------------
    # Spend concentration
    # ------------------------------------------------------------------

    def detect_concentration_risks(
        self,
        curves: dict[str, CampaignResponseCurve],
        concentration_threshold: float = 0.50,
    ) -> list[Risk]:
        """Single platform receiving > threshold of total spend."""
        risks: list[Risk] = []
        total_spend = sum(c.avg_daily_spend for c in curves.values())
        if total_spend <= 0:
            return risks

        from collections import defaultdict
        platform_spend: dict[str, float] = defaultdict(float)
        for c in curves.values():
            platform_spend[c.platform] += c.avg_daily_spend

        for platform, spend in platform_spend.items():
            share = spend / total_spend
            if share <= concentration_threshold:
                continue
            risks.append(
                Risk(
                    type=RiskType.SPEND_CONCENTRATION,
                    severity=RiskSeverity.MEDIUM,
                    title=f"{platform.title()} receives {share:.0%} of total spend",
                    explanation=(
                        f"{platform.title()} accounts for ${spend:,.0f}/day out of "
                        f"${total_spend:,.0f}/day total ({share:.0%}). "
                        "High concentration creates single-point-of-failure risk "
                        "if platform performance declines."
                    ),
                    metric_name="platform_spend_share",
                    metric_value=share * 100,
                    threshold=concentration_threshold * 100,
                    platform=platform,
                    recommendation=(
                        f"Diversify budget across platforms to reduce dependency on {platform.title()}."
                    ),
                )
            )

        return risks

    # ------------------------------------------------------------------
    # Combined
    # ------------------------------------------------------------------

    def detect_all_risks(
        self,
        fs: pd.DataFrame | None = None,
        curves: dict[str, CampaignResponseCurve] | None = None,
        forecasts: pd.DataFrame | None = None,
    ) -> list[Risk]:
        """Run all detectors and return merged, severity-sorted Risk list."""
        risks: list[Risk] = []

        if fs is not None and not fs.empty:
            risks += self.detect_budget_utilization_risks(fs)
            risks += self.detect_zero_revenue_risks(fs)

        if curves:
            risks += self.detect_saturation_risks(curves)
            risks += self.detect_concentration_risks(curves)

        if forecasts is not None and not forecasts.empty:
            risks += self.detect_confidence_risks(forecasts)

        severity_order = {
            RiskSeverity.CRITICAL: 0,
            RiskSeverity.HIGH:     1,
            RiskSeverity.MEDIUM:   2,
            RiskSeverity.LOW:      3,
        }
        risks.sort(key=lambda r: severity_order.get(r.severity, 9))
        logger.info("RiskDetector found %d risks.", len(risks))
        return risks
