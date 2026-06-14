# 5-Minute Judge Demo Script

## Setup (before judges arrive)

```bash
# Ensure artifacts are built and Streamlit is running
python demo.py --demo
# App opens at http://localhost:8501
```

---

## Minute 1 — Home Page (KPIs + Optimization Lift)

**Say:** "This is meridian — it forecasts revenue and optimises budget allocation across Google, Meta, and Bing."

**Show:**
- Daily Revenue metric (e.g. $13,200/day)
- Blended ROAS metric
- **Optimization Opportunity** section: "+$X/day (+Y%) available without changing total spend"

**Key point:** "The optimizer found that reallocating the same total budget across campaigns could increase revenue by Y% — zero additional spend required."

---

## Minute 2 — Forecast Page

**Navigate:** Sidebar → Forecast

**Show:**
- The P10/P50/P90 fan chart across all campaigns
- Tick "Show 14-day forward projection" → the chart extends into the future with widening uncertainty bands

**Key point:** "These aren't point estimates. We show calibrated uncertainty — P10/P90 form an 80% prediction interval. On held-out data the actual coverage is above 78%."

- Use the campaign selector to show a single campaign's fan chart
- Point out the confidence narrowing near recent history and widening into the future

---

## Minute 3 — Budget Simulator

**Navigate:** Sidebar → Budget Simulator

**Show the baseline state:**
- All sliders at 100% → revenue lift = $0 (correct: no change)

**Demonstrate a reallocation:**
- Move Google to 120%, Bing to 80%
- Revenue lift updates in real time: "+$X/day (+Y%)"
- Budget waterfall chart shows which campaigns gained/lost
- Marginal ROAS chart shows which campaigns have the highest return on the marginal dollar

**Key point:** "This is powered by the Hill saturation curves we fit per campaign. Campaigns with low saturation have steep marginal ROAS — that's where extra budget works hardest."

---

## Minute 4 — Campaign Analysis

**Navigate:** Sidebar → Campaign Analysis

**Show:**
- Saturation score bar chart — highlight any campaigns > 75% saturation
- Hill curve scatter: X = avg spend, Y = saturation score
- Budget utilization heatmap

**Key point:** "Camp_X is at 90% saturation — doubling spend there gives almost no additional revenue. Camp_Y is at 25% — that's where we should reallocate."

---

## Minute 5 — AI Copilot

**Navigate:** Sidebar → AI Copilot

**Show initial analysis:**
- Executive Summary (Groq LLM or rule-based)
- Risks panel: "Which campaigns are at risk?"
- Recommendations: ranked by expected revenue lift

**Demo the chat:**
- Type: "Which campaign should receive more budget?"
- Show the response grounding specific campaign names and revenue numbers

**Type:** "What happens if I increase Google spend by 20%?"
- Show the response incorporating the simulation result

**Key point:** "Every number the AI cites traces back to the actual data — no hallucinated figures. If the LLM key is missing, the rule-based engine produces the same structured output."

---

## Close

"meridian turns raw ad platform exports into an end-to-end revenue intelligence system: forecast uncertainty, saturation-aware optimization, real-time simulation, and grounded AI explanations — deployed in one command."

```bash
python demo.py        # full pipeline
python demo.py --demo # instant reload from saved artifacts
```
