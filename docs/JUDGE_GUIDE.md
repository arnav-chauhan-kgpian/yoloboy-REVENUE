# Judge Evaluation Guide

This guide helps you evaluate AIgnition in under 5 minutes.

---

## 1-Minute Setup

```bash
git clone https://github.com/your-org/aignition.git
cd aignition
pip install -r requirements.txt
python demo.py --demo    # loads pre-built artifacts, opens Streamlit
```

If no artifacts exist yet (fresh clone without committed artifacts):
```bash
python demo.py           # trains model and launches Streamlit (~3 min)
```

---

## Evaluation Checklist

### Forecasting

- [ ] Navigate to **Forecast** page
- [ ] Verify P10/P50/P90 bands display for all 12 campaigns
- [ ] Tick "Show 14-day forward projection" — verify uncertainty widens with horizon
- [ ] Select individual campaign from dropdown — verify per-campaign detail

### Optimization

- [ ] Home page shows a non-zero "Revenue Lift Available" metric
- [ ] Navigate to **Budget Simulator**
- [ ] Move any slider away from 100% — verify revenue lift updates in real time
- [ ] Verify waterfall chart updates to show per-campaign changes
- [ ] Return sliders to 100% — verify lift returns to $0

### Campaign Analysis

- [ ] Navigate to **Campaign Analysis**
- [ ] Verify saturation scores display (some should be > 0.75)
- [ ] Hill curve chart shows realistic shape (concave, saturating)

### AI Copilot

- [ ] Navigate to **AI Copilot**
- [ ] Verify "Groq AI connected" badge (if GROQ_API_KEY is set) or "rule-based mode" badge
- [ ] Executive summary references actual dollar amounts from the data
- [ ] Type a question in the chat — verify response references specific campaigns
- [ ] Risks and recommendations panels are non-empty

### Tests

```bash
pytest tests/ -q
# Expected: 1050 passed
```

### Deployment

```bash
# Docker
docker build -t aignition . && docker run -p 8501:8501 aignition
# Open http://localhost:8501
```

---

## What to Look For

| Criterion | What "good" looks like |
|---|---|
| **Technical depth** | Quantile regression (not point forecasts), rolling-origin CV, Hill saturation, SLSQP |
| **Uncertainty quantification** | P10/P90 coverage > 70% of actuals |
| **Optimization correctness** | Lift > 0 only when sliders deviate from baseline |
| **AI grounding** | Every AI statement references a real number from the data |
| **Code quality** | 19 test files, type hints, clear module separation |
| **Demo reliability** | `--demo` mode loads instantly without retraining |

---

## Architecture in 30 Seconds

```
Raw CSVs → Feature Store (45 cols) → LightGBM P10/P50/P90
                                   → Hill Curves → SLSQP Optimizer
                                   → Autoregressive 14-day Rollout
                                   → AI Copilot (Groq + rule-based)
                                   → Streamlit Dashboard
```

---

## Key Files

| File | Purpose |
|---|---|
| `demo.py` | Full pipeline entrypoint |
| `src/models/lgbm_quantile.py` | LightGBM P10/P50/P90 model |
| `src/models/trainer.py` | Training pipeline with rolling-origin CV |
| `src/simulation/optimizer.py` | SLSQP budget optimization |
| `src/simulation/response_curve.py` | Hill saturation curve fitting |
| `src/copilot/llm_client.py` | Groq + rule-based AI copilot |
| `streamlit_app/main.py` | Streamlit entry point |
