"""
streamlit_app/pages/3_Campaign_Analysis.py
===========================================
Campaign Analysis — saturation, utilisation, TM/NTM, cross-engine.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from streamlit_app.state import (
    init_session_state,
    load_feature_store,
    load_forecasts,
    load_curves,
    show_not_ready_message,
)
from streamlit_app.components.campaign_tables import (
    build_campaign_table,
    platform_comparison_table,
    tm_ntm_table,
    saturation_color,
)

st.set_page_config(page_title="Campaign Analysis — AIgnition", layout="wide")
init_session_state()

st.title("📊 Campaign Analysis")

fs        = load_feature_store()
forecasts = load_forecasts()
curves    = load_curves()

if curves is None:
    show_not_ready_message()
    st.stop()

# --------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    all_platforms = sorted({c.platform for c in curves.values()})
    sel_platforms = st.multiselect(
        "Platform", options=all_platforms, default=all_platforms
    )
    sort_by = st.selectbox(
        "Sort by",
        ["Avg Daily Revenue ($)", "Saturation (%)", "Marginal ROAS", "Current ROAS", "Avg Daily Spend ($)"],
        index=0,
    )
    ascending = st.checkbox("Ascending", value=False)

# --------------------------------------------------------------------------
# KPIs
# --------------------------------------------------------------------------
filtered_curves = {
    cid: c for cid, c in curves.items() if c.platform in sel_platforms
}

total_daily_rev   = sum(c.avg_daily_revenue for c in filtered_curves.values())
total_daily_spend = sum(c.avg_daily_spend for c in filtered_curves.values())
blended_roas      = total_daily_rev / total_daily_spend if total_daily_spend > 0 else 0
n_saturated       = sum(1 for c in filtered_curves.values() if c.saturation_score >= 0.75)
n_headroom        = sum(1 for c in filtered_curves.values() if c.saturation_score < 0.40)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Campaigns", len(filtered_curves))
with col2:
    st.metric("Daily Revenue", f"${total_daily_rev:,.0f}")
with col3:
    st.metric("ROAS", f"{blended_roas:.2f}x")
with col4:
    st.metric("Saturated (>75%)", n_saturated, delta=f"{n_saturated/max(len(filtered_curves),1)*100:.0f}%")
with col5:
    st.metric("Has Headroom (<40%)", n_headroom)

st.divider()

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Campaign Ranking", "Saturation Map", "TM vs NTM", "Platform Comparison"
])

with tab1:
    df = build_campaign_table(filtered_curves, forecasts)
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)

    # Apply color to Saturation column
    if "Saturation (%)" in df.columns:
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Saturation (%)": st.column_config.ProgressColumn(
                    "Saturation (%)",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Current ROAS": st.column_config.NumberColumn(format="%.2f"),
                "Marginal ROAS": st.column_config.NumberColumn(format="%.2f"),
                "Avg Daily Spend ($)": st.column_config.NumberColumn(format="$%.0f"),
                "Avg Daily Revenue ($)": st.column_config.NumberColumn(format="$%.0f"),
                "Reliable Curve": st.column_config.CheckboxColumn(),
            },
        )
    else:
        st.dataframe(df, use_container_width=True)

with tab2:
    sat_df = pd.DataFrame(
        [
            {
                "campaign_id": cid,
                "platform":    c.platform,
                "saturation":  c.saturation_score * 100,
                "daily_spend": c.avg_daily_spend,
                "daily_rev":   c.avg_daily_revenue,
                "marginal_roas": c.current_marginal_roas,
            }
            for cid, c in filtered_curves.items()
        ]
    )

    fig = px.scatter(
        sat_df,
        x="daily_spend",
        y="saturation",
        color="platform",
        size="daily_rev",
        hover_name="campaign_id",
        hover_data={"platform": True, "marginal_roas": ":.2f", "daily_rev": ":,.0f"},
        labels={
            "daily_spend": "Avg Daily Spend ($)",
            "saturation":  "Saturation Score (%)",
        },
        title="Saturation Score vs Daily Spend",
    )
    fig.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="Saturation warning (75%)")
    fig.add_hline(y=40, line_dash="dash", line_color="green", annotation_text="Headroom zone (40%)")
    fig.update_layout(template="plotly_white", height=480)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "**Red zone (>75%):** Campaigns approaching saturation — marginal ROAS declining rapidly.  "
        "**Green zone (<40%):** Strong budget headroom — additional spend likely to yield good returns."
    )

with tab3:
    tm_df = tm_ntm_table(filtered_curves)
    if not tm_df.empty:
        st.dataframe(tm_df, use_container_width=True)

        # TM vs NTM ROAS bar chart
        if "Segment" in tm_df.columns and "Platform" in tm_df.columns:
            fig2 = px.bar(
                tm_df,
                x="Platform",
                y="Avg ROAS",
                color="Segment",
                barmode="group",
                title="Average ROAS: TM vs NTM by Platform",
            )
            fig2.update_layout(template="plotly_white", height=360)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No TM/NTM classification available.")

with tab4:
    plat_df = platform_comparison_table(filtered_curves)
    st.dataframe(
        plat_df,
        use_container_width=True,
        column_config={
            "Daily Spend ($)":    st.column_config.NumberColumn(format="$%.0f"),
            "Daily Revenue ($)":  st.column_config.NumberColumn(format="$%.0f"),
            "ROAS":               st.column_config.NumberColumn(format="%.2f"),
            "Avg Saturation (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
        },
    )

    # Cross-engine ROAS comparison
    fig3 = px.bar(
        plat_df,
        x="Platform",
        y="ROAS",
        color="Platform",
        title="ROAS by Platform",
        text="ROAS",
    )
    fig3.update_traces(texttemplate="%{text:.2f}x", textposition="outside")
    fig3.update_layout(template="plotly_white", height=360, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
