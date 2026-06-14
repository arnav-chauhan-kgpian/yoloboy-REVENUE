"""
streamlit_app/pages/1_Forecast.py
===================================
Revenue Forecast page — fan chart, platform contribution, campaign leaderboard.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from streamlit_app.state import (
    init_session_state,
    load_feature_store,
    load_forecasts,
    load_curves,
    load_model,
    show_not_ready_message,
    generate_forecasts,
    generate_future_forecasts,
    MODEL_DIR,
)
from streamlit_app.components.forecast_charts import (
    fan_chart,
    platform_contribution_chart,
    campaign_leaderboard_chart,
)
from src.models.trainer import get_feature_columns

st.set_page_config(page_title="Forecast — AIgnition", layout="wide")
init_session_state()

st.title("📈 Revenue Forecast")

fs        = load_feature_store()
forecasts = load_forecasts()
curves    = load_curves()
model     = load_model()

if fs is None or curves is None:
    show_not_ready_message()
    st.stop()

# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Forecast Controls")

    horizon = st.selectbox(
        "Lookback window",
        options=[7, 14, 30, 60],
        index=0,
        format_func=lambda x: f"Last {x} days",
        key="forecast_horizon",
    )

    all_platforms = sorted({c.platform for c in curves.values()})
    selected_platforms = st.multiselect(
        "Filter by platform",
        options=all_platforms,
        default=all_platforms,
        key="selected_platforms",
    )

    all_campaigns = sorted(curves.keys())
    selected_campaigns = st.multiselect(
        "Filter by campaign (optional)",
        options=all_campaigns,
        default=[],
        key="selected_campaigns",
    )

    show_future = st.checkbox("Show 14-day forward projection", value=False)

# --------------------------------------------------------------------------
# Build forecast data
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Generating forecasts...")
def _get_forecasts(horizon: int):
    if forecasts is not None:
        return forecasts
    if model is not None and fs is not None:
        feature_cols = get_feature_columns(fs)
        return generate_forecasts(fs, model, feature_cols, n_days=horizon)
    return None

fc = _get_forecasts(horizon)

if fc is None:
    st.warning("No forecasts available. Ensure the model is trained.")
    st.stop()

# Apply filters
if selected_platforms:
    fc = fc[fc["platform"].isin(selected_platforms)]
if selected_campaigns:
    fc = fc[fc["campaign_id"].isin(selected_campaigns)]

# Optionally append future
if show_future and model is not None:
    @st.cache_data(show_spinner="Projecting future...")
    def _get_future():
        feature_cols = get_feature_columns(fs)
        return generate_future_forecasts(fs, model, feature_cols, n_future_days=14)

    future_fc = _get_future()
    if selected_platforms:
        future_fc = future_fc[future_fc["platform"].isin(selected_platforms)]
    fc = pd.concat([fc, future_fc], ignore_index=True)

# --------------------------------------------------------------------------
# KPIs
# --------------------------------------------------------------------------
cutoff = fc["date"].max() - pd.Timedelta(days=horizon - 1)
recent_fc = fc[fc["date"] >= cutoff]

col1, col2, col3, col4 = st.columns(4)
total_p50 = recent_fc["p50"].sum() if "p50" in recent_fc.columns else 0
total_p10 = recent_fc["p10"].sum() if "p10" in recent_fc.columns else 0
total_p90 = recent_fc["p90"].sum() if "p90" in recent_fc.columns else 0

with col1:
    st.metric(f"P50 Revenue ({horizon}d)", f"${total_p50:,.0f}")
with col2:
    st.metric("P10 (pessimistic)", f"${total_p10:,.0f}")
with col3:
    st.metric("P90 (optimistic)", f"${total_p90:,.0f}")
with col4:
    width_pct = (total_p90 - total_p10) / total_p50 * 100 if total_p50 > 0 else 0
    st.metric("Interval Width", f"{width_pct:.0f}%", help="(P90-P10)/P50")

st.divider()

# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
st.plotly_chart(
    fan_chart(fc, title=f"Revenue Forecast — Last {horizon} Days"),
    use_container_width=True,
)

col_left, col_right = st.columns([1, 1])
with col_left:
    st.plotly_chart(
        platform_contribution_chart(fc, window_days=horizon),
        use_container_width=True,
    )
with col_right:
    st.plotly_chart(
        campaign_leaderboard_chart(fc, top_n=10),
        use_container_width=True,
    )

# --------------------------------------------------------------------------
# Raw data table
# --------------------------------------------------------------------------
with st.expander("Raw Forecast Data"):
    st.dataframe(
        fc.sort_values(["date", "campaign_id"], ascending=[False, True]),
        use_container_width=True,
    )
