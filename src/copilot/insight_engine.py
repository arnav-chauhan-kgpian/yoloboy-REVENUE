"""
src/copilot/insight_engine.py
==============================
Generates data-grounded insights from forecasts, feature store, and simulations.

Every insight traces back to a specific metric with a specific value extracted
from the input DataFrames.  No values are invented.

Insight types
-------------
REVENUE_TREND       — week-over-week revenue momentum
PLATFORM_SHARE      — platform's share of total revenue changing
FORECAST_CONFIDENCE — wide P10/P90 interval → low confidence
SPEND_MOMENTUM      — spend acceleration or deceleration
HOLIDAY_SIGNAL      — upcoming holiday impacting forecast
SATURATION          — campaign approaching saturation
BUDGET_UTILIZATION  — campaign under-spending its budget
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BUDGET_UTIL_LOW_THRESHOLD:   Final[float] = 0.30   # below 30% util → risk
CONFIDENCE_WIDE_THRESHOLD:   Final[float] = 0.60   # (P90-P10)/P50 > 60% → wide CI
SATURATION_HIGH_THRESHOLD:   Final[float] = 0.75   # saturation > 75% → at risk
SPEND_MOMENTUM_THRESHOLD:    Final[float] = 0.10   # |spend change| > 10% → notable
REVENUE_TREND_THRESHOLD:     Final[float] = 0.05   # |revenue change| > 5% → notable


class InsightType(str, Enum):
    REVENUE_TREND       = "revenue_trend"
    PLATFORM_SHARE      = "platform_share"
    FORECAST_CONFIDENCE = "forecast_confidence"
    SPEND_MOMENTUM      = "spend_momentum"
    HOLIDAY_SIGNAL      = "holiday_signal"
    SATURATION          = "saturation"
    BUDGET_UTILIZATION  = "budget_utilization"
    TOP_CAMPAIGN        = "top_campaign"


class InsightSeverity(str, Enum):
    INFO     = "info"
    POSITIVE = "positive"
    WARNING  = "warning"
    CRITICAL = "critical"


@dataclass
class Insight:
    type: InsightType
    severity: InsightSeverity
    title: str
    explanation: str
    metric_name: str
    metric_value: float
    metric_unit: str           # "$", "%", "x", "days", ""
    campaign_id: str | None = None
    platform: str | None    = None
    confidence: float        = 1.0    # 0-1, from data availability


class InsightEngineError(Exception):
    pass


def _latest_window(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Return last *days* rows by date from df (must have a 'date' column)."""
    cutoff = df["date"].max() - pd.Timedelta(days=days - 1)
    return df[df["date"] >= cutoff]


def _prior_window(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Return the *days* rows before the most recent *days*."""
    latest_max = df["date"].max()
    prior_end   = latest_max - pd.Timedelta(days=days)
    prior_start = prior_end - pd.Timedelta(days=days - 1)
    return df[(df["date"] >= prior_start) & (df["date"] <= prior_end)]


class InsightEngine:
    """Extract data-grounded insights from forecasts and the feature store."""

    # ------------------------------------------------------------------
    # Revenue trend
    # ------------------------------------------------------------------

    def revenue_trend_insights(
        self,
        forecasts: pd.DataFrame,
        window_days: int = 7,
    ) -> list[Insight]:
        """Week-over-week revenue momentum from P50 forecasts."""
        insights: list[Insight] = []
        if forecasts.empty or "p50" not in forecasts.columns:
            return insights

        recent = _latest_window(forecasts, window_days)
        prior  = _prior_window(forecasts, window_days)

        if recent.empty or prior.empty:
            return insights

        recent_rev = float(recent["p50"].sum())
        prior_rev  = float(prior["p50"].sum())

        if prior_rev <= 0:
            return insights

        change_pct = (recent_rev - prior_rev) / prior_rev * 100.0

        if abs(change_pct) < REVENUE_TREND_THRESHOLD * 100:
            return insights

        if change_pct >= 0:
            severity  = InsightSeverity.POSITIVE
            direction = "up"
        else:
            severity  = InsightSeverity.WARNING
            direction = "down"

        insights.append(
            Insight(
                type=InsightType.REVENUE_TREND,
                severity=severity,
                title=f"Revenue trending {direction} {abs(change_pct):.1f}% week-over-week",
                explanation=(
                    f"The last {window_days} days generated ${recent_rev:,.0f} in P50 revenue, "
                    f"compared with ${prior_rev:,.0f} in the prior {window_days}-day period "
                    f"— a change of {change_pct:+.1f}%."
                ),
                metric_name="wow_revenue_change_pct",
                metric_value=change_pct,
                metric_unit="%",
                confidence=0.9,
            )
        )
        return insights

    # ------------------------------------------------------------------
    # Platform share
    # ------------------------------------------------------------------

    def platform_share_insights(
        self,
        forecasts: pd.DataFrame,
        window_days: int = 7,
    ) -> list[Insight]:
        """Flag platform revenue shifts."""
        insights: list[Insight] = []
        if forecasts.empty or "p50" not in forecasts.columns or "platform" not in forecasts.columns:
            return insights

        recent = _latest_window(forecasts, window_days)
        prior  = _prior_window(forecasts, window_days)
        if recent.empty or prior.empty:
            return insights

        recent_total = float(recent["p50"].sum())
        prior_total  = float(prior["p50"].sum())
        if recent_total <= 0 or prior_total <= 0:
            return insights

        for platform in recent["platform"].unique():
            r_rev = float(recent[recent["platform"] == platform]["p50"].sum())
            p_rev = float(prior[prior["platform"] == platform]["p50"].sum())

            r_share = r_rev / recent_total * 100.0
            p_share = p_rev / prior_total  * 100.0 if prior_total > 0 else 0.0
            share_delta = r_share - p_share

            if abs(share_delta) < 3.0:  # less than 3pp shift → not notable
                continue

            direction = "gaining" if share_delta > 0 else "losing"
            severity  = InsightSeverity.INFO if abs(share_delta) < 10 else InsightSeverity.WARNING

            insights.append(
                Insight(
                    type=InsightType.PLATFORM_SHARE,
                    severity=severity,
                    title=f"{platform.title()} {direction} share ({share_delta:+.1f}pp)",
                    explanation=(
                        f"{platform.title()} now accounts for {r_share:.1f}% of forecast revenue "
                        f"(${r_rev:,.0f}), versus {p_share:.1f}% (${p_rev:,.0f}) last period."
                    ),
                    metric_name="platform_share_delta_pp",
                    metric_value=share_delta,
                    metric_unit="pp",
                    platform=platform,
                    confidence=0.85,
                )
            )
        return insights

    # ------------------------------------------------------------------
    # Forecast confidence
    # ------------------------------------------------------------------

    def forecast_confidence_insights(
        self,
        forecasts: pd.DataFrame,
        window_days: int = 7,
    ) -> list[Insight]:
        """Detect wide P10/P90 intervals indicating low forecast confidence."""
        insights: list[Insight] = []
        if not all(c in forecasts.columns for c in ["p10", "p50", "p90"]):
            return insights

        recent = _latest_window(forecasts, window_days)
        if recent.empty:
            return insights

        p50 = float(recent["p50"].sum())
        if p50 <= 0:
            return insights

        p10 = float(recent["p10"].sum())
        p90 = float(recent["p90"].sum())
        width_pct = (p90 - p10) / p50 * 100.0

        if width_pct > CONFIDENCE_WIDE_THRESHOLD * 100:
            insights.append(
                Insight(
                    type=InsightType.FORECAST_CONFIDENCE,
                    severity=InsightSeverity.WARNING,
                    title=f"Forecast confidence interval is wide ({width_pct:.0f}%)",
                    explanation=(
                        f"The P10/P90 interval spans ${p10:,.0f} to ${p90:,.0f} "
                        f"around a P50 of ${p50:,.0f} — a range of {width_pct:.0f}% of P50. "
                        "Interpret point forecasts with caution."
                    ),
                    metric_name="forecast_width_pct",
                    metric_value=width_pct,
                    metric_unit="%",
                    confidence=0.95,
                )
            )
        return insights

    # ------------------------------------------------------------------
    # Spend momentum
    # ------------------------------------------------------------------

    def spend_momentum_insights(
        self,
        fs: pd.DataFrame,
        window_days: int = 7,
    ) -> list[Insight]:
        """Detect significant recent changes in platform-level spend."""
        insights: list[Insight] = []
        if "spend" not in fs.columns or "platform" not in fs.columns:
            return insights

        recent = _latest_window(fs, window_days)
        prior  = _prior_window(fs, window_days)
        if recent.empty or prior.empty:
            return insights

        for platform in recent["platform"].unique():
            r_spend = float(recent[recent["platform"] == platform]["spend"].sum())
            p_spend = float(prior[prior["platform"] == platform]["spend"].sum())
            if p_spend <= 0:
                continue
            change_pct = (r_spend - p_spend) / p_spend * 100.0
            if abs(change_pct) < SPEND_MOMENTUM_THRESHOLD * 100:
                continue

            direction = "accelerating" if change_pct > 0 else "decelerating"
            severity  = InsightSeverity.INFO

            insights.append(
                Insight(
                    type=InsightType.SPEND_MOMENTUM,
                    severity=severity,
                    title=f"{platform.title()} spend {direction} ({change_pct:+.1f}%)",
                    explanation=(
                        f"{platform.title()} spend over the last {window_days} days "
                        f"was ${r_spend:,.0f}, {change_pct:+.1f}% vs prior period "
                        f"(${p_spend:,.0f})."
                    ),
                    metric_name="spend_change_pct",
                    metric_value=change_pct,
                    metric_unit="%",
                    platform=platform,
                    confidence=0.9,
                )
            )
        return insights

    # ------------------------------------------------------------------
    # Holiday signal
    # ------------------------------------------------------------------

    def holiday_insights(
        self,
        fs: pd.DataFrame,
        lookahead_days: int = 14,
    ) -> list[Insight]:
        """Detect upcoming high-intensity holiday periods in the feature store."""
        insights: list[Insight] = []
        if "holiday_intensity_score" not in fs.columns:
            return insights

        latest_date = fs["date"].max()
        future_cutoff = latest_date + pd.Timedelta(days=lookahead_days)

        # Use data near end of known period as proxy for upcoming intensity
        tail = fs[fs["date"] >= latest_date - pd.Timedelta(days=7)]
        avg_intensity = float(tail["holiday_intensity_score"].mean())

        if avg_intensity > 0.3:
            insights.append(
                Insight(
                    type=InsightType.HOLIDAY_SIGNAL,
                    severity=InsightSeverity.POSITIVE,
                    title=f"Elevated holiday intensity detected (score={avg_intensity:.2f})",
                    explanation=(
                        f"The average holiday intensity score over the last 7 days is "
                        f"{avg_intensity:.2f} (scale 0–1), indicating seasonal demand uplift. "
                        "Forecasts may underestimate peak revenue if not extrapolating this signal."
                    ),
                    metric_name="holiday_intensity_score",
                    metric_value=avg_intensity,
                    metric_unit="",
                    confidence=0.7,
                )
            )
        return insights

    # ------------------------------------------------------------------
    # Top campaign
    # ------------------------------------------------------------------

    def top_campaign_insights(
        self,
        forecasts: pd.DataFrame,
        top_n: int = 3,
        window_days: int = 7,
    ) -> list[Insight]:
        """Identify top-contributing campaigns in the forecast window."""
        insights: list[Insight] = []
        if forecasts.empty or "p50" not in forecasts.columns or "campaign_id" not in forecasts.columns:
            return insights

        recent = _latest_window(forecasts, window_days)
        if recent.empty:
            return insights

        total_rev = float(recent["p50"].sum())
        if total_rev <= 0:
            return insights

        by_camp = (
            recent.groupby("campaign_id")["p50"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )

        for cid, rev in by_camp.items():
            share = rev / total_rev * 100.0
            insights.append(
                Insight(
                    type=InsightType.TOP_CAMPAIGN,
                    severity=InsightSeverity.INFO,
                    title=f"Top contributor: {cid} ({share:.1f}% of revenue)",
                    explanation=(
                        f"Campaign '{cid}' generated ${rev:,.0f} in P50 forecast revenue "
                        f"over the last {window_days} days, representing {share:.1f}% of total."
                    ),
                    metric_name="campaign_revenue_share_pct",
                    metric_value=share,
                    metric_unit="%",
                    campaign_id=str(cid),
                    confidence=0.9,
                )
            )
        return insights

    # ------------------------------------------------------------------
    # Combined entry point
    # ------------------------------------------------------------------

    def generate_all_insights(
        self,
        forecasts: pd.DataFrame | None = None,
        fs: pd.DataFrame | None = None,
        window_days: int = 7,
    ) -> list[Insight]:
        """Generate all insight types from available data.

        Parameters
        ----------
        forecasts : pd.DataFrame
            Must have columns: campaign_id, platform, date, p10, p50, p90.
        fs : pd.DataFrame
            Feature store with spend, revenue_attributed, holiday columns.
        window_days : int
            Lookback window for trend computation.

        Returns
        -------
        list[Insight]
            Sorted by severity (critical → info).
        """
        insights: list[Insight] = []

        if forecasts is not None and not forecasts.empty:
            insights += self.revenue_trend_insights(forecasts, window_days)
            insights += self.platform_share_insights(forecasts, window_days)
            insights += self.forecast_confidence_insights(forecasts, window_days)
            insights += self.top_campaign_insights(forecasts, window_days=window_days)

        if fs is not None and not fs.empty:
            insights += self.spend_momentum_insights(fs, window_days)
            insights += self.holiday_insights(fs)

        severity_order = {
            InsightSeverity.CRITICAL: 0,
            InsightSeverity.WARNING:  1,
            InsightSeverity.POSITIVE: 2,
            InsightSeverity.INFO:     3,
        }
        insights.sort(key=lambda i: severity_order.get(i.severity, 9))
        logger.info("InsightEngine generated %d insights.", len(insights))
        return insights
