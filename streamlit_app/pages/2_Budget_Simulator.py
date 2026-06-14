"""
streamlit_app/pages/2_Budget_Simulator.py
==========================================
Budget Simulator — drag sliders to project revenue impact.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd

from streamlit_app.state import (
    init_session_state,
    load_curves,
    load_opt_result,
    show_not_ready_message,
)
from streamlit_app.components.budget_charts import (
    before_after_chart,
    allocation_waterfall_chart,
    marginal_roas_chart,
    scenario_comparison_chart,
)
from src.simulation.scenario_generator import apply_scenario, compare_scenarios, generate_standard_scenarios
from src.simulation.optimizer import optimize_budget

st.set_page_config(page_title="Budget Simulator — AIgnition", layout="wide")
init_session_state()

st.title("💰 Budget Simulator")
st.caption("Adjust platform budgets and see projected revenue impact in real time.")

curves   = load_curves()
opt_base = load_opt_result()

if curves is None:
    show_not_ready_message()
    st.stop()

# --------------------------------------------------------------------------
# Sidebar sliders
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Platform Budget Controls")
    st.caption("Adjust as % of current spend")

    all_platforms = sorted({c.platform for c in curves.values()})

    platform_pcts: dict[str, int] = {}
    for platform in all_platforms:
        platform_pcts[platform] = st.slider(
            f"{platform.title()} budget %",
            min_value=50,
            max_value=200,
            value=100,
            step=5,
            key=f"{platform}_budget_pct",
        )

    st.divider()
    st.subheader("Campaign Overrides (optional)")
    st.caption("Set specific daily spend for individual campaigns")

    override_campaign = st.selectbox(
        "Campaign",
        options=["None"] + sorted(curves.keys()),
        index=0,
    )
    override_spend = None
    if override_campaign != "None":
        current_spend = curves[override_campaign].avg_daily_spend
        override_spend = st.number_input(
            f"Daily spend for {override_campaign}",
            min_value=0.0,
            value=float(current_spend),
            step=10.0,
        )
        if st.button("Apply Override"):
            st.session_state.campaign_overrides[override_campaign] = override_spend

    if st.session_state.campaign_overrides:
        st.caption(f"Active overrides: {len(st.session_state.campaign_overrides)}")
        if st.button("Clear All Overrides"):
            st.session_state.campaign_overrides = {}

    st.divider()
    if st.button("Run Optimizer", type="primary", help="Find optimal spend allocation"):
        total_budget = sum(c.avg_daily_spend for c in curves.values())
        with st.spinner("Optimizing..."):
            result = optimize_budget(curves, total_budget)
            st.session_state.scenario_result = result
        st.success(f"Done! Lift: +{result.revenue_lift_pct:.1f}%")

# --------------------------------------------------------------------------
# Build scenario from sliders
# --------------------------------------------------------------------------
platform_multipliers = {p: pct / 100.0 for p, pct in platform_pcts.items()}
campaign_overrides   = dict(st.session_state.campaign_overrides)

scenario_result = apply_scenario(
    curves,
    platform_multipliers=platform_multipliers,
    campaign_overrides=campaign_overrides,
    scenario_name="Custom Scenario",
    description="User-configured budget allocation",
)

# --------------------------------------------------------------------------
# KPIs
# --------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Baseline Revenue/day", f"${scenario_result.baseline_total_revenue:,.0f}")
with col2:
    st.metric(
        "Projected Revenue/day",
        f"${scenario_result.projected_total_revenue:,.0f}",
        f"{scenario_result.revenue_lift_pct:+.1f}%",
    )
with col3:
    st.metric("Baseline ROAS", f"{scenario_result.baseline_roas:.2f}x")
with col4:
    st.metric(
        "Projected ROAS",
        f"{scenario_result.projected_roas:.2f}x",
        f"{scenario_result.projected_roas - scenario_result.baseline_roas:+.2f}x",
    )
with col5:
    st.metric(
        "Confidence Interval",
        f"${scenario_result.ci_low:,.0f}–${scenario_result.ci_high:,.0f}",
        help="P10–P90 range for projected revenue",
    )

risk_color = "🟢" if scenario_result.risk_score < 0.3 else "🟡" if scenario_result.risk_score < 0.6 else "🔴"
st.caption(f"Risk score: {risk_color} {scenario_result.risk_score:.2f} (0=low, 1=high)")

st.divider()

# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Before/After", "Waterfall", "Marginal ROAS", "Standard Scenarios"
])

with tab1:
    st.plotly_chart(before_after_chart(scenario_result), use_container_width=True)

with tab2:
    st.plotly_chart(allocation_waterfall_chart(scenario_result), use_container_width=True)

with tab3:
    st.plotly_chart(marginal_roas_chart(scenario_result), use_container_width=True)

with tab4:
    with st.spinner("Generating standard scenarios..."):
        std_scenarios = generate_standard_scenarios(curves)
        scenarios_df  = compare_scenarios(std_scenarios)

    st.plotly_chart(scenario_comparison_chart(scenarios_df), use_container_width=True)
    st.dataframe(
        scenarios_df[[
            "scenario", "revenue_lift_pct", "projected_revenue",
            "projected_roas", "risk_score",
        ]].rename(columns={
            "scenario":          "Scenario",
            "revenue_lift_pct":  "Lift (%)",
            "projected_revenue": "Projected Revenue ($)",
            "projected_roas":    "Projected ROAS",
            "risk_score":        "Risk",
        }),
        use_container_width=True,
    )

# --------------------------------------------------------------------------
# Optimizer result (if available)
# --------------------------------------------------------------------------
if st.session_state.get("scenario_result") is not None:
    opt_result = st.session_state.scenario_result
    st.divider()
    st.subheader("Optimizer Result")

    ocol1, ocol2, ocol3 = st.columns(3)
    with ocol1:
        st.metric("Revenue Lift", f"+${opt_result.revenue_lift:,.0f}/day", f"+{opt_result.revenue_lift_pct:.1f}%")
    with ocol2:
        st.metric("Converged", "Yes" if opt_result.converged else "No")
    with ocol3:
        st.metric("Budget Accuracy", f"${opt_result.budget_error:,.2f} error")

    opt_df = pd.DataFrame(
        [
            {
                "Campaign":     a.campaign_id,
                "Platform":     a.platform.title(),
                "Current ($)":  round(a.baseline_spend, 0),
                "Optimal ($)":  round(a.optimal_spend, 0),
                "Change ($)":   round(a.spend_delta, 0),
                "Change (%)":   round(a.spend_delta_pct, 1),
                "Rev Lift ($)": round(a.revenue_delta, 0),
                "Saturation":   f"{a.saturation_at_optimal:.0%}",
            }
            for a in opt_result.optimal_allocations
            if abs(a.spend_delta) > 0.5
        ]
    ).sort_values("Rev Lift ($)", ascending=False)

    st.dataframe(opt_df, use_container_width=True)

# --------------------------------------------------------------------------
# Campaign projections table
# --------------------------------------------------------------------------
with st.expander("Campaign-Level Projections"):
    proj_df = pd.DataFrame(
        [
            {
                "Campaign":       p.campaign_id,
                "Platform":       p.platform.title(),
                "Baseline ($)":   round(p.baseline_daily_spend, 0),
                "New Spend ($)":  round(p.new_daily_spend, 0),
                "Baseline Rev":   round(p.baseline_daily_revenue, 0),
                "Projected Rev":  round(p.projected_daily_revenue, 0),
                "Rev Lift (%)":   round(p.revenue_delta_pct, 1),
            }
            for p in scenario_result.campaign_projections
        ]
    ).sort_values("Rev Lift (%)", ascending=False)
    st.dataframe(proj_df, use_container_width=True)
