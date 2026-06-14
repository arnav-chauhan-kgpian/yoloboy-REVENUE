"""
src/copilot/llm_client.py
==========================
LLM client for the meridian AI copilot.

Uses the Groq API (llama-3.3-70b-versatile by default — fast, free tier
available) via the OpenAI-compatible Groq Python SDK.

Falls back to a rule-based engine when the API key is absent or the call
fails.  The rule-based fallback produces the same CopilotOutput structure
from the pre-computed insights, risks, and recommendations — no LLM required.

Setup:
    pip install groq
    export GROQ_API_KEY=gsk_...   # get one free at console.groq.com
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from src.copilot.insight_engine import Insight
from src.copilot.prompt_builder import (
    PromptContext,
    build_analysis_prompt,
    build_context_from_data,
    build_system_prompt,
)
from src.copilot.recommender import Recommendation
from src.copilot.risk_detector import Risk

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "llama-3.3-70b-versatile"
_MAX_TOKENS    = 1500


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

@dataclass
class CopilotOutput:
    summary: str
    risks: list[str]
    opportunities: list[str]
    recommendations: list[str]
    confidence: float
    source: str = "llm"   # "llm" | "rule_based"


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

class _RuleBasedCopilot:
    """Generates CopilotOutput from pre-computed data without LLM."""

    # Keywords that signal the user is asking about the assistant itself,
    # not about the marketing data.
    _META_KEYWORDS = frozenset([
        "your name", "who are you", "what are you", "who r u",
        "what is your", "what's your", "introduce yourself",
        "are you", "tell me about yourself",
    ])

    def _is_meta_question(self, q: str) -> bool:
        low = q.lower()
        return any(kw in low for kw in self._META_KEYWORDS)

    def generate(self, ctx: PromptContext, question: str | None = None) -> CopilotOutput:
        # Handle meta / identity questions before touching any data context.
        if question and self._is_meta_question(question):
            return CopilotOutput(
                summary=(
                    "I'm meridian's AI Copilot — a rule-based analytics assistant "
                    "(Groq LLM mode off). I analyse your marketing data and surface "
                    "revenue forecasts, budget recommendations, and campaign risks. "
                    "Set GROQ_API_KEY to enable full conversational AI answers."
                ),
                risks=[],
                opportunities=[],
                recommendations=["Set GROQ_API_KEY in .env to enable LLM-powered Q&A."],
                confidence=1.0,
                source="rule_based",
            )

        # --- Summary ---
        summary_parts = []
        if ctx.total_p50_revenue > 0:
            summary_parts.append(
                f"Revenue forecast for the next {ctx.forecast_period_days} days: "
                f"${ctx.total_p50_revenue:,.0f} P50 "
                f"(${ctx.total_p10_revenue:,.0f}–${ctx.total_p90_revenue:,.0f} P10/P90 range)."
            )
        if ctx.revenue_wow_change_pct != 0:
            direction = "up" if ctx.revenue_wow_change_pct > 0 else "down"
            summary_parts.append(
                f"Revenue is trending {direction} {abs(ctx.revenue_wow_change_pct):.1f}% week-over-week."
            )
        if ctx.optimization_revenue_lift_pct > 0:
            summary_parts.append(
                f"Budget reallocation could lift revenue by "
                f"${ctx.optimization_revenue_lift_abs:,.0f}/day "
                f"(+{ctx.optimization_revenue_lift_pct:.1f}%)."
            )

        # Prepend the question prefix WITHOUT duplicating the first sentence.
        # (The old code used .insert() which left the original item in place
        # and caused the first sentence to appear twice in the joined string.)
        if question:
            prefix = f'(Re: "{question}") '
            if summary_parts:
                summary_parts[0] = prefix + summary_parts[0]
            else:
                summary_parts.append(
                    prefix + "No forecast data available — run demo.py first."
                )

        summary = " ".join(summary_parts) or "Insufficient data to generate summary."

        # --- Risks ---
        risk_texts = [r.title + ": " + r.explanation for r in ctx.risks[:5]]
        if not risk_texts:
            risk_texts = ["No significant risks detected in current data."]

        # --- Opportunities ---
        opps = []
        if ctx.headroom_campaigns:
            opps.append(
                f"{len(ctx.headroom_campaigns)} campaigns have low saturation (<40%) "
                f"and room for increased spend: {', '.join(ctx.headroom_campaigns[:3])}."
            )
        if ctx.optimization_revenue_lift_pct > 1.0:
            opps.append(
                f"Budget optimizer found a reallocation opportunity worth "
                f"+{ctx.optimization_revenue_lift_pct:.1f}% revenue "
                f"(+${ctx.optimization_revenue_lift_abs:,.0f}/day) "
                f"without changing total spend."
            )
        for i in ctx.insights:
            if i.severity.value in ("positive", "info") and len(opps) < 5:
                opps.append(i.explanation)
        if not opps:
            opps = ["No clear opportunities detected in current data."]

        # --- Recommendations ---
        rec_texts = [rec.title + " — " + rec.rationale for rec in ctx.recommendations[:5]]
        if not rec_texts:
            rec_texts = ["Review campaign saturation and budget utilisation metrics."]

        return CopilotOutput(
            summary=summary,
            risks=risk_texts,
            opportunities=opps,
            recommendations=rec_texts,
            confidence=0.70,
            source="rule_based",
        )


_rule_based = _RuleBasedCopilot()


# ---------------------------------------------------------------------------
# LLM client — Groq backend
# ---------------------------------------------------------------------------

class LLMClient:
    """Groq-backed AI copilot with rule-based fallback.

    Parameters
    ----------
    model : str
        Groq model ID.  Defaults to ``llama-3.3-70b-versatile``.
    api_key : str | None
        Groq API key.  If None, reads from ``GROQ_API_KEY`` env var.
        If neither is set, the rule-based fallback is used automatically.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> None:
        self.model    = model

        # Load .env so the key is available even when the caller hasn't
        # explicitly called load_dotenv (e.g. Streamlit pages that import
        # this module before state.py has run).
        if not os.environ.get("GROQ_API_KEY"):
            try:
                from pathlib import Path
                from dotenv import load_dotenv
                _root = Path(__file__).resolve().parent.parent.parent
                load_dotenv(_root / ".env", override=False)
            except ImportError:
                pass  # python-dotenv not installed; fall through to rule-based

        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._client  = None
        self._available = False

        if self._api_key:
            try:
                from groq import Groq
                self._client    = Groq(api_key=self._api_key)
                self._available = True
                logger.info("LLMClient: Groq API available (model=%s).", model)
            except ImportError:
                logger.warning(
                    "groq package not installed. "
                    "Run: pip install groq. Using rule-based fallback."
                )
        else:
            logger.info("No GROQ_API_KEY set. Using rule-based copilot.")

    @property
    def is_llm_available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        ctx: PromptContext,
        question: str | None = None,
    ) -> CopilotOutput:
        """Generate a CopilotOutput from PromptContext.

        Tries Groq first; falls back to rule-based on any error.
        """
        if not self._available:
            return _rule_based.generate(ctx, question)

        try:
            return self._llm_generate(ctx, question)
        except Exception as exc:
            logger.warning("Groq call failed (%s). Falling back to rule-based.", exc)
            return _rule_based.generate(ctx, question)

    def _llm_generate(self, ctx: PromptContext, question: str | None) -> CopilotOutput:
        user_prompt   = build_analysis_prompt(ctx, question)
        system_prompt = build_system_prompt()

        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rsplit("```", 1)[0].strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Groq returned non-JSON (%s). Falling back.", exc)
            return _rule_based.generate(ctx, question)

        return CopilotOutput(
            summary=str(data.get("summary", "")),
            risks=list(data.get("risks", [])),
            opportunities=list(data.get("opportunities", [])),
            recommendations=list(data.get("recommendations", [])),
            confidence=float(data.get("confidence", 0.75)),
            source="llm",
        )

    # ------------------------------------------------------------------
    # Convenience: build context and generate in one call
    # ------------------------------------------------------------------

    def analyse(
        self,
        forecasts: "pd.DataFrame | None" = None,
        fs: "pd.DataFrame | None"        = None,
        curves: "dict | None"            = None,
        opt_result: "Any | None"         = None,
        insights: list[Insight] | None   = None,
        risks: list[Risk] | None         = None,
        recommendations: list[Recommendation] | None = None,
        question: str | None             = None,
        window_days: int                 = 7,
    ) -> CopilotOutput:
        """All-in-one: build context → generate → return CopilotOutput."""
        ctx = build_context_from_data(
            forecasts=forecasts,
            fs=fs,
            curves=curves,
            opt_result=opt_result,
            insights=insights,
            risks=risks,
            recommendations=recommendations,
            window_days=window_days,
        )
        return self.generate(ctx, question)
