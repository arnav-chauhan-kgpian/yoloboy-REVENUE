"""
streamlit_app/components/copilot_panels.py
==========================================
Streamlit panel components for the AI Copilot page.
"""

from __future__ import annotations

import streamlit as st

from src.copilot.llm_client import CopilotOutput
from src.copilot.risk_detector import Risk, RiskSeverity
from src.copilot.recommender import Recommendation, RecommendationType


_SEVERITY_ICONS = {
    RiskSeverity.CRITICAL: "🔴",
    RiskSeverity.HIGH:     "🟠",
    RiskSeverity.MEDIUM:   "🟡",
    RiskSeverity.LOW:      "🟢",
}

_TYPE_ICONS = {
    RecommendationType.INCREASE_BUDGET:  "⬆️",
    RecommendationType.DECREASE_BUDGET:  "⬇️",
    RecommendationType.REALLOCATE:       "↔️",
    RecommendationType.INVESTIGATE:      "🔍",
    RecommendationType.MAINTAIN:         "✅",
}


def executive_summary_panel(output: CopilotOutput) -> None:
    """Render the executive summary card."""
    with st.container(border=True):
        st.subheader("Executive Summary")
        badge = "🤖 AI-powered" if output.source == "llm" else "📊 Data-driven"
        confidence_pct = int(output.confidence * 100)
        st.caption(f"{badge} · Confidence: {confidence_pct}%")
        st.markdown(output.summary)


def risks_panel(risks_text: list[str], risks_data: list[Risk] | None = None) -> None:
    """Render detected risks with severity icons."""
    st.subheader("Risks Detected")
    if not risks_text:
        st.success("No significant risks detected.")
        return

    for i, risk_text in enumerate(risks_text):
        # Try to get severity from data if available
        severity_icon = "🟡"
        if risks_data and i < len(risks_data):
            severity_icon = _SEVERITY_ICONS.get(risks_data[i].severity, "🟡")

        st.markdown(f"{severity_icon} {risk_text}")


def opportunities_panel(opportunities: list[str]) -> None:
    """Render opportunities list."""
    st.subheader("Opportunities")
    if not opportunities:
        st.info("No clear opportunities identified.")
        return

    for opp in opportunities:
        st.markdown(f"✨ {opp}")


def recommendations_panel(
    recommendations_text: list[str],
    recommendations_data: list[Recommendation] | None = None,
) -> None:
    """Render budget recommendations with type icons and expected lift."""
    st.subheader("Budget Recommendations")
    if not recommendations_text:
        st.info("No recommendations at this time.")
        return

    for i, rec_text in enumerate(recommendations_text):
        icon = "💡"
        if recommendations_data and i < len(recommendations_data):
            icon = _TYPE_ICONS.get(recommendations_data[i].type, "💡")
            rec_obj = recommendations_data[i]
            with st.expander(f"{icon} {rec_obj.title}", expanded=(i == 0)):
                st.write(rec_obj.rationale)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Expected Lift",
                        f"${rec_obj.expected_revenue_lift:,.0f}/day",
                        f"{rec_obj.expected_revenue_lift_pct:+.1f}%",
                    )
                with col2:
                    st.metric(
                        "Spend Change",
                        f"${rec_obj.spend_change:+,.0f}/day",
                    )
                with col3:
                    st.metric(
                        "Marginal ROAS",
                        f"${rec_obj.target_marginal_roas:.2f}x",
                    )
        else:
            st.markdown(f"{icon} {rec_text}")


def chat_message_panel(messages: list[dict]) -> None:
    """Render the chat history."""
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
