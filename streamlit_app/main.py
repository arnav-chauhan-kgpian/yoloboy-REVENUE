"""
streamlit_app/main.py
======================
AIgnition Streamlit app entry point.

Run with:  streamlit run streamlit_app/main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from streamlit_app.state import (
    init_session_state,
    load_feature_store,
    load_forecasts,
    load_curves,
    load_opt_result,
    show_not_ready_message,
)

st.set_page_config(
    page_title="AIgnition — Revenue Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()

# --------------------------------------------------------------------------
# Sidebar — global info
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🚀 AIgnition")
    st.caption("Ecommerce Revenue Intelligence")
    st.divider()

    fs        = load_feature_store()
    forecasts = load_forecasts()
    curves    = load_curves()
    opt       = load_opt_result()

    ready = fs is not None and curves is not None

    if ready:
        st.success("Data loaded ✓")
        st.caption(f"Feature store: {len(fs):,} rows")
        if forecasts is not None:
            st.caption(f"Forecasts: {len(forecasts):,} rows")
        st.caption(f"Campaigns: {len(curves):,}")
        n_platforms = len({c.platform for c in curves.values()})
        st.caption(f"Platforms: {n_platforms}")
    else:
        st.error("Run `python demo.py` first")

    st.divider()
    st.caption("Navigate using the sidebar pages →")

# --------------------------------------------------------------------------
# Homepage content
# --------------------------------------------------------------------------
st.title("🚀 AIgnition — Revenue Intelligence Platform")
st.markdown(
    "**AI-powered ecommerce revenue forecasting and budget optimization "
    "across Google, Meta, and Bing.**"
)

if not ready:
    show_not_ready_message()
    st.stop()

st.divider()

# KPI row
col1, col2, col3, col4 = st.columns(4)

total_daily_rev   = sum(c.avg_daily_revenue for c in curves.values())
total_daily_spend = sum(c.avg_daily_spend for c in curves.values())
blended_roas      = total_daily_rev / total_daily_spend if total_daily_spend > 0 else 0
n_saturated       = sum(1 for c in curves.values() if c.saturation_score >= 0.75)

with col1:
    st.metric("Daily Revenue (avg)", f"${total_daily_rev:,.0f}")
with col2:
    st.metric("Daily Spend (avg)", f"${total_daily_spend:,.0f}")
with col3:
    st.metric("Blended ROAS", f"{blended_roas:.2f}x")
with col4:
    st.metric("Saturated Campaigns", n_saturated, help="Campaigns at >75% saturation")

st.divider()

if opt is not None:
    st.subheader("Optimization Opportunity")
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        st.metric(
            "Revenue Lift Available",
            f"+${opt.revenue_lift:,.0f}/day",
            f"+{opt.revenue_lift_pct:.1f}%",
        )
    with oc2:
        st.metric("Current ROAS", f"{opt.baseline_roas:.2f}x")
    with oc3:
        st.metric("Optimised ROAS", f"{opt.optimal_roas:.2f}x")

    st.info(
        "ℹ️ This reallocation can be achieved **without changing total spend** — "
        "only by redistributing budget across campaigns.",
        icon="ℹ️",
    )

st.divider()
st.markdown(
    "**Use the sidebar to navigate:**\n"
    "- 📈 **Forecast** — Revenue fan chart and campaign rankings\n"
    "- 💰 **Budget Simulator** — Drag sliders, see projected impact\n"
    "- 📊 **Campaign Analysis** — Saturation, utilisation, TM/NTM\n"
    "- 🤖 **AI Copilot** — Ask questions, get grounded recommendations"
)
