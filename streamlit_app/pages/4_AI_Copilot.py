"""
streamlit_app/pages/4_AI_Copilot.py
======================================
AI Copilot — executive summary, recommendations, risks, chat interface.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from streamlit_app.state import (
    init_session_state,
    load_feature_store,
    load_forecasts,
    load_curves,
    load_opt_result,
    show_not_ready_message,
)
from streamlit_app.components.copilot_panels import (
    executive_summary_panel,
    risks_panel,
    opportunities_panel,
    recommendations_panel,
    chat_message_panel,
)
from src.copilot.insight_engine import InsightEngine
from src.copilot.risk_detector import RiskDetector
from src.copilot.recommender import from_optimizer_result, generate_quick_recommendations
from src.copilot.llm_client import LLMClient

st.set_page_config(page_title="AI Copilot — meridian", layout="wide")
init_session_state()

st.title("🤖 AI Copilot")
st.caption("Grounded revenue intelligence — every number traces back to your actual data.")

fs        = load_feature_store()
forecasts = load_forecasts()
curves    = load_curves()
opt       = load_opt_result()

if curves is None:
    show_not_ready_message()
    st.stop()

# --------------------------------------------------------------------------
# Generate data-grounded insights (cached per session)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Analysing data...")
def _generate_analysis(_curves_hash, _fc_hash):
    engine   = InsightEngine()
    detector = RiskDetector()

    insights = engine.generate_all_insights(forecasts=forecasts, fs=fs)
    risks    = detector.detect_all_risks(fs=fs, curves=curves, forecasts=forecasts)

    if opt is not None:
        recs = from_optimizer_result(opt, curves)
    else:
        recs = generate_quick_recommendations(curves)

    return insights, risks, recs

def _make_fc_hash(fc) -> str:
    """Content-based cache key for the forecasts DataFrame.

    Incorporates row count, latest date, and total P50 sum so any
    regeneration with new data busts the cached AI analysis.
    """
    if fc is None or fc.empty:
        return "none"
    return (
        f"{len(fc)}"
        f"_{fc['date'].max()}"
        f"_{fc['p50'].sum():.0f}"
    )


fc_hash     = _make_fc_hash(forecasts)
curves_hash = len(curves)

insights, risks, recs = _generate_analysis(curves_hash, fc_hash)

# --------------------------------------------------------------------------
# LLM client (initialise once per session; re-init if API key appears later)
# --------------------------------------------------------------------------
import os as _os
_api_key_now = _os.environ.get("GROQ_API_KEY", "")
_cached_client: LLMClient | None = st.session_state.get("llm_client")

if _cached_client is None or (not _cached_client.is_llm_available and _api_key_now):
    st.session_state.llm_client = LLMClient()

llm: LLMClient = st.session_state.llm_client

# Generate initial analysis
@st.cache_data(show_spinner="AI is thinking...")
def _initial_analysis(_curves_hash, _fc_hash):
    return llm.analyse(
        forecasts=forecasts,
        fs=fs,
        curves=curves,
        opt_result=opt,
        insights=insights,
        risks=risks,
        recommendations=recs,
    )

if st.session_state.copilot_output is None:
    st.session_state.copilot_output = _initial_analysis(curves_hash, fc_hash)

output = st.session_state.copilot_output

# --------------------------------------------------------------------------
# LLM badge
# --------------------------------------------------------------------------
if llm.is_llm_available:
    st.success("🤖 Groq AI connected — responses use LLM reasoning over your data.", icon="✅")
else:
    st.info(
        "ℹ️ Running in rule-based mode. Set `GROQ_API_KEY` to enable Groq LLM.",
        icon="ℹ️",
    )

st.divider()

# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    executive_summary_panel(output)
    st.divider()
    risks_panel(output.risks, risks)

with col_right:
    opportunities_panel(output.opportunities)
    st.divider()
    recommendations_panel(output.recommendations, recs)

st.divider()

# --------------------------------------------------------------------------
# Chat interface
# --------------------------------------------------------------------------
st.subheader("💬 Ask the Copilot")
st.caption(
    "Example questions:\n"
    "- *What happens if I increase Google spend by 20%?*\n"
    "- *Which campaign should receive more budget?*\n"
    "- *Why is forecast confidence lower this month?*"
)

# Display chat history
chat_message_panel(st.session_state.copilot_messages)

# Chat input
if prompt := st.chat_input("Ask about your campaigns and forecasts..."):
    # Display user message
    st.session_state.copilot_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Analysing..."):
            response = llm.analyse(
                forecasts=forecasts,
                fs=fs,
                curves=curves,
                opt_result=opt,
                insights=insights,
                risks=risks,
                recommendations=recs,
                question=prompt,
            )
        st.markdown(response.summary)

        if response.recommendations:
            st.markdown("**Recommendations:**")
            for r in response.recommendations[:3]:
                st.markdown(f"- {r}")

    st.session_state.copilot_messages.append(
        {"role": "assistant", "content": response.summary}
    )

# Reset chat button
if st.session_state.copilot_messages:
    if st.button("Clear chat history"):
        st.session_state.copilot_messages = []
        st.rerun()

# --------------------------------------------------------------------------
# Raw insights / risks expanders
# --------------------------------------------------------------------------
with st.expander("Raw Data Insights"):
    for i in insights:
        st.markdown(f"**{i.type.value}** ({i.severity.value}): {i.explanation}")
        st.caption(f"Metric: {i.metric_name} = {i.metric_value:.3f} {i.metric_unit}")

with st.expander("Detected Risks (detail)"):
    if not risks:
        st.write("No risks detected.")
    for r in risks:
        st.markdown(f"**{r.severity.value.upper()}** — {r.title}")
        st.write(r.explanation)
        if r.recommendation:
            st.caption(f"Recommendation: {r.recommendation}")
        st.divider()
