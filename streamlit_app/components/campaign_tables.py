"""
streamlit_app/components/campaign_tables.py
============================================
DataFrame builders for the Campaign Analysis page.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.simulation.response_curve import CampaignResponseCurve, platform_summary


def build_campaign_table(
    curves: dict[str, CampaignResponseCurve],
    forecasts: pd.DataFrame | None = None,
    window_days: int = 7,
) -> pd.DataFrame:
    """Build a styled campaign analysis DataFrame.

    Merges response-curve metrics with forecast-period revenue.
    """
    df = platform_summary(curves)

    # Add forecast revenue if available
    if forecasts is not None and not forecasts.empty:
        cutoff = forecasts["date"].max() - pd.Timedelta(days=window_days - 1)
        recent_rev = (
            forecasts[forecasts["date"] >= cutoff]
            .groupby("campaign_id")["p50"]
            .sum()
            .reset_index()
            .rename(columns={"p50": f"forecast_rev_{window_days}d"})
        )
        df = df.merge(recent_rev, on="campaign_id", how="left")

    # Format saturation score as %
    df["saturation_pct"] = (df["saturation_score"] * 100).round(1)

    # TM vs NTM classification from strategy_key or campaign_id pattern
    if "campaign_id" in df.columns:
        df["segment"] = df["campaign_id"].apply(
            lambda x: "TM" if "_TM_" in str(x).upper() or "BRAND" in str(x).upper() else "NTM"
        )

    # Round numeric columns for display
    numeric_cols = [
        "avg_daily_spend", "avg_daily_revenue", "current_roas",
        "marginal_roas", "saturation_pct", "r_squared",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].round(2)

    # Rename for display
    rename = {
        "campaign_id":       "Campaign ID",
        "campaign_name":     "Campaign Name",
        "platform":          "Platform",
        "avg_daily_spend":   "Avg Daily Spend ($)",
        "avg_daily_revenue": "Avg Daily Revenue ($)",
        "current_roas":      "Current ROAS",
        "marginal_roas":     "Marginal ROAS",
        "saturation_pct":    "Saturation (%)",
        "is_reliable":       "Reliable Curve",
        "r_squared":         "R²",
    }
    return df.rename(columns=rename)


def saturation_color(val: float) -> str:
    """Return a background color string for saturation % values."""
    if val >= 75:
        return "background-color: #FECACA"   # red
    if val >= 50:
        return "background-color: #FEF08A"   # yellow
    return "background-color: #BBF7D0"       # green


def platform_comparison_table(
    curves: dict[str, CampaignResponseCurve],
) -> pd.DataFrame:
    """Aggregated per-platform metrics."""
    rows = []
    platform_data: dict[str, list] = {}
    for c in curves.values():
        if c.platform not in platform_data:
            platform_data[c.platform] = []
        platform_data[c.platform].append(c)

    for platform, camp_list in platform_data.items():
        total_spend   = sum(c.avg_daily_spend for c in camp_list)
        total_rev     = sum(c.avg_daily_revenue for c in camp_list)
        avg_sat       = sum(c.saturation_score for c in camp_list) / len(camp_list)
        avg_mroas     = sum(c.current_marginal_roas for c in camp_list) / len(camp_list)
        n_campaigns   = len(camp_list)
        n_reliable    = sum(1 for c in camp_list if c.is_reliable)

        rows.append(
            {
                "Platform":           platform.title(),
                "Campaigns":          n_campaigns,
                "Reliable Curves":    n_reliable,
                "Daily Spend ($)":    round(total_spend, 0),
                "Daily Revenue ($)":  round(total_rev, 0),
                "ROAS":               round(total_rev / total_spend, 2) if total_spend > 0 else 0,
                "Avg Saturation (%)": round(avg_sat * 100, 1),
                "Avg Marginal ROAS":  round(avg_mroas, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("Daily Revenue ($)", ascending=False)


def tm_ntm_table(
    curves: dict[str, CampaignResponseCurve],
) -> pd.DataFrame:
    """TM vs NTM comparison across platforms."""
    rows = []
    for cid, c in curves.items():
        segment = (
            "TM"
            if "_TM_" in cid.upper() or "BRAND" in cid.upper()
            else "NTM"
        )
        rows.append(
            {
                "campaign_id":   cid,
                "platform":      c.platform,
                "segment":       segment,
                "daily_spend":   c.avg_daily_spend,
                "daily_revenue": c.avg_daily_revenue,
                "roas":          c.current_roas,
                "marginal_roas": c.current_marginal_roas,
                "saturation":    c.saturation_score,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    agg = (
        df.groupby(["platform", "segment"])
        .agg(
            n_campaigns=("campaign_id", "count"),
            total_daily_spend=("daily_spend", "sum"),
            total_daily_revenue=("daily_revenue", "sum"),
            avg_roas=("roas", "mean"),
            avg_marginal_roas=("marginal_roas", "mean"),
            avg_saturation=("saturation", "mean"),
        )
        .reset_index()
    )

    for col in ["total_daily_spend", "total_daily_revenue", "avg_roas",
                "avg_marginal_roas", "avg_saturation"]:
        agg[col] = agg[col].round(2)

    return agg.rename(
        columns={
            "platform":           "Platform",
            "segment":            "Segment",
            "n_campaigns":        "Campaigns",
            "total_daily_spend":  "Daily Spend ($)",
            "total_daily_revenue":"Daily Revenue ($)",
            "avg_roas":           "Avg ROAS",
            "avg_marginal_roas":  "Avg Marginal ROAS",
            "avg_saturation":     "Avg Saturation",
        }
    )
