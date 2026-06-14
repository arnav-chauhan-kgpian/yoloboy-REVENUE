"""
streamlit_app/components/budget_charts.py
==========================================
Plotly chart builders for the Budget Simulator page.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.simulation.scenario_generator import ScenarioResult


def before_after_chart(result: ScenarioResult) -> go.Figure:
    """Grouped bar chart comparing baseline vs projected revenue by platform."""
    if not result.campaign_projections:
        return go.Figure()

    df = pd.DataFrame(
        [
            {
                "platform": p.platform,
                "baseline":   p.baseline_daily_revenue,
                "projected":  p.projected_daily_revenue,
            }
            for p in result.campaign_projections
        ]
    )
    agg = df.groupby("platform")[["baseline", "projected"]].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Baseline",
            x=agg["platform"].str.title(),
            y=agg["baseline"],
            marker_color="#636EFA",
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Projected",
            x=agg["platform"].str.title(),
            y=agg["projected"],
            marker_color="#00CC96",
        )
    )
    fig.update_layout(
        barmode="group",
        title="Baseline vs Projected Revenue by Platform",
        yaxis_title="Daily Revenue ($)",
        yaxis_tickformat="$,.0f",
        template="plotly_white",
        legend={"orientation": "h", "y": -0.15},
        height=360,
    )
    return fig


def allocation_waterfall_chart(result: ScenarioResult) -> go.Figure:
    """Waterfall chart showing spend changes by platform.

    When all budget sliders are at baseline (no change applied), falls back
    to a grouped bar showing current spend allocation per platform so the
    chart always contains meaningful information.
    """
    if not result.campaign_projections:
        return go.Figure()

    df = pd.DataFrame(
        [
            {
                "platform":   p.platform,
                "spend_delta": p.spend_delta,
                "baseline":   p.baseline_daily_spend,
                "new_spend":  p.new_daily_spend,
            }
            for p in result.campaign_projections
        ]
    )
    agg = df.groupby("platform").agg(
        spend_delta=("spend_delta", "sum"),
        baseline=("baseline", "sum"),
        new_spend=("new_spend", "sum"),
    ).reset_index()
    agg["label"] = agg["platform"].str.title()

    total_abs_delta = agg["spend_delta"].abs().sum()

    # ------------------------------------------------------------------ #
    # Baseline state: no budget change — show current spend distribution  #
    # ------------------------------------------------------------------ #
    if total_abs_delta < 1.0:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=agg["label"],
                y=agg["baseline"],
                marker_color="#636EFA",
                text=[f"${v:,.0f}" for v in agg["baseline"]],
                textposition="outside",
                name="Current Daily Spend",
            )
        )
        fig.update_layout(
            title="Current Daily Spend Allocation (No Change Applied)",
            yaxis_title="Daily Spend ($)",
            yaxis_tickformat="$,.0f",
            template="plotly_white",
            height=340,
            annotations=[{
                "text": "Adjust sliders to see reallocation impact",
                "xref": "paper", "yref": "paper",
                "x": 0.5, "y": -0.18,
                "showarrow": False,
                "font": {"size": 12, "color": "gray"},
            }],
        )
        return fig

    # ------------------------------------------------------------------ #
    # Non-baseline: show spend delta waterfall                            #
    # ------------------------------------------------------------------ #
    agg_list   = agg.to_dict("records")
    total_delta = agg["spend_delta"].sum()

    measures = ["relative"] * len(agg_list) + ["total"]
    x_labels = [r["label"] for r in agg_list] + ["Total Change"]
    y_values = [r["spend_delta"] for r in agg_list] + [total_delta]

    fig = go.Figure(
        go.Waterfall(
            x=x_labels,
            y=y_values,
            measure=measures,
            connector={"line": {"color": "rgb(63,63,63)"}},
            increasing={"marker": {"color": "#00CC96"}},
            decreasing={"marker": {"color": "#EF553B"}},
            totals={"marker": {"color": "#636EFA"}},
            text=[f"${v:+,.0f}" for v in y_values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Budget Reallocation Waterfall (Daily $)",
        yaxis_title="Spend Change ($)",
        yaxis_tickformat="$,.0f",
        template="plotly_white",
        height=340,
    )
    return fig


def marginal_roas_chart(
    result: ScenarioResult,
    top_n: int = 15,
) -> go.Figure:
    """Scatter of baseline vs projected marginal ROAS per campaign.

    Bubble size encodes the magnitude of spend change.  At baseline
    (no change applied) all spend deltas are zero — dots are shown at
    a fixed minimum size so campaigns are always visible, with the
    diagonal indicating «no change».  Campaigns above the diagonal
    gained marginal ROAS; below lost it.
    """
    if not result.campaign_projections:
        return go.Figure()

    df = pd.DataFrame(
        [
            {
                "campaign_id":     p.campaign_id,
                "platform":        p.platform,
                "baseline_mroas":  p.baseline_marginal_roas,
                "new_mroas":       p.new_marginal_roas,
                "spend_delta_pct": p.spend_delta_pct,
            }
            for p in result.campaign_projections
        ]
    )

    df["abs_delta"] = df["spend_delta_pct"].abs()
    df = df.sort_values("abs_delta", ascending=False).head(top_n)

    # Clamp to a minimum visible size — at baseline all deltas are 0
    # which would make every dot invisible without this guard.
    MIN_BUBBLE = 6.0
    df["bubble_size"] = df["abs_delta"].clip(lower=MIN_BUBBLE)

    no_change = df["abs_delta"].max() < 0.5   # baseline state

    fig = px.scatter(
        df,
        x="baseline_mroas",
        y="new_mroas",
        color="platform",
        size="bubble_size",
        size_max=30,
        hover_name="campaign_id",
        hover_data={
            "platform":        True,
            "spend_delta_pct": ":.1f",
            "bubble_size":     False,
        },
        title="Marginal ROAS by Campaign" + (" (Baseline)" if no_change else ""),
        labels={
            "baseline_mroas": "Baseline Marginal ROAS ($/$ spend)",
            "new_mroas":      "Projected Marginal ROAS ($/$ spend)",
        },
    )

    # Diagonal reference line (no change)
    max_val = max(float(df[["baseline_mroas", "new_mroas"]].max().max()), 1.0)
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line={"color": "gray", "dash": "dash", "width": 1},
            name="No change",
            showlegend=False,
        )
    )

    if no_change:
        fig.add_annotation(
            text="Adjust sliders to see marginal ROAS shifts",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font={"size": 12, "color": "gray"},
        )

    fig.update_layout(template="plotly_white", height=400)
    return fig


def scenario_comparison_chart(scenarios_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing multiple scenarios by revenue lift %."""
    if scenarios_df.empty or "scenario" not in scenarios_df.columns:
        return go.Figure()

    colors = [
        "#636EFA" if r >= 0 else "#EF553B"
        for r in scenarios_df["revenue_lift_pct"]
    ]

    fig = go.Figure(
        go.Bar(
            x=scenarios_df["scenario"],
            y=scenarios_df["revenue_lift_pct"],
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in scenarios_df["revenue_lift_pct"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Revenue Lift by Scenario",
        yaxis_title="Revenue Lift (%)",
        template="plotly_white",
        height=360,
    )
    return fig
