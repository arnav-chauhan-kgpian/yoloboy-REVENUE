"""
streamlit_app/components/forecast_charts.py
============================================
Plotly chart builders for the Forecast page.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def fan_chart(
    forecasts: pd.DataFrame,
    title: str = "Revenue Forecast Fan Chart",
    group_by: str = "date",
) -> go.Figure:
    """P10/P50/P90 fan chart aggregated by date.

    Parameters
    ----------
    forecasts : pd.DataFrame
        Columns required: date, p10, p50, p90.
        Optional: revenue_attributed (actuals), is_future.
    """
    df = forecasts.copy()
    df["date"] = pd.to_datetime(df["date"])

    agg = df.groupby("date").agg(
        p10=("p10", "sum"),
        p50=("p50", "sum"),
        p90=("p90", "sum"),
    ).reset_index()

    actuals_agg = None
    if "revenue_attributed" in df.columns:
        actuals_agg = (
            df[df["revenue_attributed"].notna()]
            .groupby("date")["revenue_attributed"]
            .sum()
            .reset_index()
        )

    fig = go.Figure()

    # P10/P90 shaded band
    fig.add_trace(
        go.Scatter(
            x=list(agg["date"]) + list(agg["date"][::-1]),
            y=list(agg["p90"])  + list(agg["p10"][::-1]),
            fill="toself",
            fillcolor="rgba(99,110,250,0.15)",
            line={"color": "rgba(255,255,255,0)"},
            name="P10–P90 Interval",
            hoverinfo="skip",
        )
    )

    # P50 line
    fig.add_trace(
        go.Scatter(
            x=agg["date"],
            y=agg["p50"],
            mode="lines",
            name="P50 Forecast",
            line={"color": "#636EFA", "width": 2},
        )
    )

    # P10 / P90 dashed lines
    for col, name, dash in [("p10", "P10", "dot"), ("p90", "P90", "dot")]:
        fig.add_trace(
            go.Scatter(
                x=agg["date"],
                y=agg[col],
                mode="lines",
                name=name,
                line={"color": "#636EFA", "width": 1, "dash": dash},
                opacity=0.6,
            )
        )

    # Actuals
    if actuals_agg is not None and not actuals_agg.empty:
        fig.add_trace(
            go.Scatter(
                x=actuals_agg["date"],
                y=actuals_agg["revenue_attributed"],
                mode="markers",
                name="Actuals",
                marker={"color": "#EF553B", "size": 5},
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Revenue ($)",
        yaxis_tickformat="$,.0f",
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "y": -0.15},
        height=420,
    )
    return fig


def platform_contribution_chart(
    forecasts: pd.DataFrame,
    window_days: int = 7,
) -> go.Figure:
    """Stacked area chart of P50 revenue by platform over time."""
    df = forecasts.copy()
    df["date"] = pd.to_datetime(df["date"])

    if "platform" not in df.columns:
        return go.Figure()

    agg = (
        df.groupby(["date", "platform"])["p50"]
        .sum()
        .reset_index()
        .pivot(index="date", columns="platform", values="p50")
        .fillna(0)
        .reset_index()
    )

    platforms = [c for c in agg.columns if c != "date"]
    colors    = px.colors.qualitative.Set2

    fig = go.Figure()
    for i, platform in enumerate(platforms):
        fig.add_trace(
            go.Scatter(
                x=agg["date"],
                y=agg[platform],
                mode="lines",
                name=platform.title(),
                stackgroup="one",
                fillcolor=colors[i % len(colors)],
                line={"color": colors[i % len(colors)]},
            )
        )

    fig.update_layout(
        title="Platform Revenue Contribution (P50)",
        xaxis_title="Date",
        yaxis_title="Revenue ($)",
        yaxis_tickformat="$,.0f",
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "y": -0.15},
        height=380,
    )
    return fig


def campaign_leaderboard_chart(
    forecasts: pd.DataFrame,
    top_n: int = 10,
) -> go.Figure:
    """Horizontal bar chart of top-N campaigns by P50 forecast revenue."""
    if "campaign_id" not in forecasts.columns or "p50" not in forecasts.columns:
        return go.Figure()

    top = (
        forecasts.groupby("campaign_id")["p50"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    fig = px.bar(
        top,
        x="p50",
        y="campaign_id",
        orientation="h",
        labels={"p50": "Forecast Revenue ($)", "campaign_id": "Campaign"},
        color="p50",
        color_continuous_scale="Blues",
        title=f"Top {top_n} Campaigns by P50 Revenue",
    )
    fig.update_layout(
        height=max(300, top_n * 30),
        template="plotly_white",
        yaxis={"autorange": "reversed"},
        coloraxis_showscale=False,
        xaxis_tickformat="$,.0f",
    )
    return fig
