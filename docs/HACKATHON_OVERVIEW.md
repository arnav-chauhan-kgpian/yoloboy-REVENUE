# meridian — NetElixir AIgnition 3.0 Submission

## The Problem

Ecommerce brands running ads across Google, Meta, and Bing have no unified view of where their budget is working hardest. Campaign managers guess at ROAS, manually tweak budgets, and lack confidence intervals. Mis-allocation costs thousands per day.

## The Solution

meridian is a full-stack revenue intelligence platform that:

1. **Forecasts** daily revenue per campaign with calibrated uncertainty (P10/P50/P90) using LightGBM quantile regression
2. **Identifies** which campaigns are at saturation and which have headroom using Hill curve fitting
3. **Optimises** budget allocation across platforms using SLSQP — finding the highest-revenue split at the same total spend
4. **Simulates** the revenue impact of any budget change in real time via an interactive slider interface
5. **Answers questions** about the data in natural language using Groq `llama-3.3-70b-versatile`, grounded in live campaign signals

## Key Numbers

| Metric | Value |
|---|---|
| Campaigns modelled | 12 (Google · Meta · Bing) |
| Forecast horizon | 14-day autoregressive per campaign |
| Uncertainty bands | P10 / P50 / P90 |
| Typical MAE | < $120/day per campaign |
| Typical P10–P90 coverage | > 78% of actuals |
| Budget optimizer | SLSQP, Hill saturation, 0.5× reallocation cap |
| Typical lift from reallocation | +3–8% revenue at same spend |
| Tests | 1050 passing |
| Lines of code | ~4,200 |

## Technical Stack

| Component | Technology |
|---|---|
| Forecasting model | LightGBM (quantile regression, 3× independent models) |
| Saturation curves | Hill model (`revenue = V_max × spend / (K + spend)`) |
| Optimizer | SciPy SLSQP with equality constraint |
| Autoregressive rollout | Day-by-day lag buffer, per-campaign |
| AI Copilot | Groq llama-3.3-70b-versatile (rule-based fallback) |
| Dashboard | Streamlit multi-page app |
| Deployment | Render / Docker / Heroku |

## What Makes It Different

- **Calibrated uncertainty**: P10/P50/P90 quantile bands, not point estimates
- **Grounded AI**: every copilot statement traces to a real number from the data
- **Zero spend required for the optimizer**: reallocates existing budget, doesn't ask for more
- **Works without an LLM key**: rule-based fallback produces the same analytical output
- **One-command startup**: `python demo.py` handles everything end-to-end
