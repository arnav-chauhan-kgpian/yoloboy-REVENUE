# Changelog

All notable changes to this project are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

---

## [Unreleased]

---

## [1.0.0] — 2026-06-14

### Added
- LightGBM P10/P50/P90 quantile revenue forecasting with rolling-origin cross-validation
- Autoregressive 14-day forward projection with per-campaign lag buffer
- Hill saturation curve fitting (V_max, K) for 12 campaigns across Google · Meta · Bing
- SLSQP budget optimizer — maximises revenue without changing total spend
- Interactive budget simulator with real-time revenue impact
- Groq `llama-3.3-70b-versatile` AI Copilot with rule-based fallback
- Insight engine, risk detector, and recommendation engine
- 5-page Streamlit dashboard (Home · Forecast · Budget Simulator · Campaign Analysis · AI Copilot)
- `demo.py` one-command pipeline with `--demo` fast-load mode
- `run_training.py` standalone training entrypoint
- `run_inference.py` standalone inference entrypoint
- `build_artifacts.py` headless artifact builder for Docker/Render
- Render deployment via `render.yaml`
- Docker deployment via `Dockerfile`
- GitHub Actions CI (test + lint)
- 1050 unit tests across 19 test modules

### Fixed
- Baseline scenario zero-lift bug — guard against `abs(new_spend - baseline_spend) < 1e-9`
- Forecast artifact covered only 3/12 campaigns — replaced `.tail(N)` with date-based cutoff
- Rule-based copilot duplicate first sentence — replaced `insert()` with in-place replace
- Budget waterfall chart showed ±$1 at baseline — shows spend distribution instead
- Marginal ROAS chart invisible at baseline — enforces minimum bubble size of 6px
- `train()` called with invalid kwargs — moved `max_folds`/`min_train_days` to `validator_kwargs`
- `.env` not loaded by Streamlit — `load_dotenv()` now called in `LLMClient.__init__` and `state.py`
