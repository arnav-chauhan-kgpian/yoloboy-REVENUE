# Project Structure

```
aignition/
│
├── demo.py                     One-command pipeline launcher
│                               python demo.py          # full pipeline + Streamlit
│                               python demo.py --demo   # load artifacts + Streamlit
│
├── run_training.py             Standalone training entrypoint
├── run_inference.py            Standalone inference entrypoint
├── build_artifacts.py          Headless artifact builder (Docker / Render build step)
│
├── requirements.txt            Python dependencies
├── runtime.txt                 Python version pin for cloud platforms
├── Procfile                    Heroku / Railway startup command
├── render.yaml                 Render deployment configuration
├── Dockerfile                  Docker image definition
│
├── .env.example                Environment variable template
├── .gitignore
│
├── src/                        Source packages
│   ├── data/
│   │   ├── loader.py           Raw CSV loading (Google · Meta · Bing)
│   │   ├── harmonizer.py       Canonical DataFrame builder
│   │   └── taxonomy_parser.py  Campaign classification (format, audience, funnel)
│   │
│   ├── features/
│   │   ├── feature_store.py    build_feature_store() — joins all feature sources
│   │   ├── lag_features.py     1/3/7/14/28-day lags for spend and revenue
│   │   ├── rolling_features.py 7/14/30-day rolling mean and std
│   │   └── holiday_calendar.py BFCM, Cyber Week, holiday intensity signals
│   │
│   ├── models/
│   │   ├── lgbm_quantile.py    RevenueQuantileModel — P10/P50/P90 LightGBM wrapper
│   │   ├── trainer.py          train() — rolling-origin CV, model persistence
│   │   ├── cross_validator.py  RollingOriginValidator — temporal fold generation
│   │   └── autoregressive.py   generate_future_forecasts() — 14-day AR rollout
│   │
│   ├── simulation/
│   │   ├── hill_curve.py       Hill saturation: revenue = V_max·x / (K + x)
│   │   ├── response_curve.py   CampaignResponseCurve, build_response_curves()
│   │   ├── optimizer.py        optimize_budget() — SLSQP revenue maximisation
│   │   └── scenario_generator.py  ScenarioGenerator — slider → ΔRevenue
│   │
│   └── copilot/
│       ├── insight_engine.py   InsightEngine — data-grounded signals
│       ├── risk_detector.py    RiskDetector — saturation / utilization / concentration
│       ├── recommender.py      Recommendation generation from optimizer output
│       ├── prompt_builder.py   LLM prompt construction (system + user)
│       └── llm_client.py       LLMClient — Groq backend + rule-based fallback
│
├── streamlit_app/              Multi-page Streamlit dashboard
│   ├── main.py                 Entry point + Home page (KPIs, optimization summary)
│   ├── state.py                Session state init, cached data loaders
│   ├── pages/
│   │   ├── 1_Forecast.py       P10/P50/P90 fan chart + 14-day AR projection
│   │   ├── 2_Budget_Simulator.py  Interactive budget sliders + waterfall + ROAS
│   │   ├── 3_Campaign_Analysis.py Saturation curves, utilization, TM/NTM breakdown
│   │   └── 4_AI_Copilot.py    Executive summary, risks, recs, conversational chat
│   └── components/
│       ├── budget_charts.py    Budget waterfall + marginal ROAS scatter
│       ├── campaign_tables.py  Campaign projection and saturation tables
│       ├── copilot_panels.py   Summary, risk, opportunity, recommendation panels
│       └── forecast_charts.py  P10/P50/P90 fan chart, campaign rankings
│
├── tests/                      1050 unit tests, 19 modules
│   ├── conftest.py
│   ├── test_trainer.py
│   ├── test_lgbm_quantile.py
│   ├── test_cross_validator.py
│   ├── test_feature_store.py
│   ├── test_autoregressive.py
│   ├── test_optimizer.py
│   ├── test_scenario_generator.py
│   └── ...
│
├── docs/                       Hackathon documentation
│   ├── HACKATHON_OVERVIEW.md
│   ├── TECHNICAL_ARCHITECTURE.md
│   ├── DEMO_SCRIPT.md
│   └── JUDGE_GUIDE.md
│
├── dataset/                    Generated artifacts (git-ignored)
│   ├── feature_store.parquet
│   ├── forecasts.parquet
│   ├── curves.pkl
│   └── opt_result.pkl
│
└── models/                     Trained model files (git-ignored)
    ├── p10.pkl
    ├── p50.pkl
    ├── p90.pkl
    └── model_meta.pkl
```
