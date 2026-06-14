# Technical Architecture

## System Diagram

```
Raw CSVs (Google · Meta · Bing)
         │
         ▼
   build_feature_store()
   ┌─────────────────────────────────────────┐
   │  Canonical DataFrame                    │
   │  + lag features (1/3/7/14/28-day)       │
   │  + rolling stats (7/14/30-day)          │
   │  + holiday calendar (BFCM, Cyber Week)  │
   │  + campaign taxonomy (format, funnel)   │
   │  45 columns · (campaign_id, date) grain │
   └─────────────────────────────────────────┘
         │
         ├──────────────────────────────────────────┐
         ▼                                          ▼
   LightGBM Quantile Training               Hill Curve Fitting
   ┌──────────────────────────┐          ┌──────────────────────────┐
   │ RollingOriginValidator   │          │  For each campaign:      │
   │ Walk-forward folds       │          │  revenue = V_max·x/(K+x) │
   │ P10 model (α=0.10)       │          │  Fit via scipy curve_fit │
   │ P50 model (α=0.50)       │          │  saturation_score =      │
   │ P90 model (α=0.90)       │          │  avg_spend / (K + avg)   │
   │ Final retrain on all     │          └──────────────────────────┘
   │ attribution_mature rows  │                    │
   └──────────────────────────┘                    ▼
         │                              SLSQP Budget Optimizer
         │                         ┌──────────────────────────────┐
         ▼                         │  maximize: Σ curve_i(x_i)   │
   Autoregressive Rollout          │  subject to: Σ x_i = budget  │
   ┌──────────────────────┐        │  bounds: x_i ∈ [0.5, 1.5]×  │
   │  For k=1..14:        │        │  current spend per campaign   │
   │    update lag buffer │        └──────────────────────────────┘
   │    predict(X_k)      │
   │    store P50 → lags  │
   └──────────────────────┘
```

## Forecasting Model

### Feature Engineering (45 columns)

| Category | Features |
|---|---|
| Spend signals | `spend`, `spend_lag1`, `spend_lag3`, `spend_lag7`, `spend_lag14`, `spend_lag28` |
| Revenue signals | `revenue_lag1`, `revenue_lag7`, `revenue_lag28`, `roas_lag7` |
| Rolling stats | `spend_roll7_mean`, `spend_roll7_std`, `revenue_roll7_mean`, `revenue_roll30_mean` |
| Calendar | `day_of_week`, `month`, `quarter`, `is_weekend`, `week_of_year` |
| Holidays | `is_holiday`, `holiday_intensity`, `days_to_bfcm`, `is_cyber_week` |
| Campaign | `format_encoded`, `funnel_stage_encoded`, `audience_encoded`, `platform_encoded` |
| Utilization | `budget_utilization`, `spend_vs_30d_avg` |

### Model Architecture

Three independent LightGBM regressors, each trained with `objective="quantile"`:

```python
QuantileConfig(
    n_estimators     = 2000,   # production / 200 for demo
    learning_rate    = 0.03,
    num_leaves       = 63,
    min_child_samples= 20,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq     = 5,
    lambda_l1        = 0.1,
    lambda_l2        = 0.1,
    max_depth        = 6,
    early_stopping   = 100,
)
```

Quantile crossing fix: `p10 = min(p10, p50)`, `p90 = max(p50, p90)`. Revenue floor: `p >= 0`.

### Rolling-Origin Cross-Validation

```
Timeline:  |----train----|--val--|
                                  |----train----|--val--|
                                                        |----train----|--val--|

- Training window: all attribution_mature=True rows up to train_end
- Validation window: next 30 calendar days per fold
- Step size: 30 days
- min_train_days: 60 (demo) / 180 (production)
- max_folds: 2 (demo) / unlimited (production)
- Evaluation: MAE, MAPE, pinball loss (P10/P50/P90), 80% coverage
```

## Hill Saturation Curves

Revenue as a function of spend follows a saturating curve:

```
revenue(spend) = V_max × spend / (K + spend)
```

Where:
- `V_max` = asymptotic maximum revenue (fitted by `scipy.curve_fit`)
- `K` = half-saturation constant (spend at which revenue = V_max/2)
- `saturation_score` = `avg_spend / (K + avg_spend)` ∈ [0, 1]

Campaigns with `saturation_score > 0.75` are flagged as high-saturation (marginal return declining sharply).

## Budget Optimizer

SLSQP (Sequential Least Squares Programming) from `scipy.optimize.minimize`:

```python
minimize(
    fun     = lambda x: -sum(curve_i.revenue_at_spend(x_i) for i, x_i in ...),
    x0      = current_spend_vector,
    method  = "SLSQP",
    bounds  = [(0.5 * s_i, 1.5 * s_i) for s_i in current_spend_vector],
    constraints = [{"type": "eq", "fun": lambda x: x.sum() - total_budget}],
)
```

The reallocation cap (0.5× / 1.5×) prevents extreme shifts that a saturation-only model might recommend.

## Autoregressive Rollout

For each future day `k = 1..14`:
1. Build feature row for each campaign at date `today + k`
2. Populate lag features from rolling buffer: `[actual_1..n, predicted_1..k-1]`
3. Recompute 7/14/30-day rolling means from the buffer
4. Update calendar and holiday features for future date
5. Keep spend/impressions at last-known actual values
6. Call `model.predict()` → store P50 in buffer for next step

Uncertainty grows naturally with horizon because lag uncertainty compounds.

## AI Copilot

### LLM Mode (Groq)

System prompt includes:
- Instruction to respond in valid JSON `{summary, risks, opportunities, recommendations, confidence}`
- Constraint to ground all claims in the provided data context

User prompt includes:
- 7-day revenue forecast (P10/P50/P90)
- Week-over-week trend
- Optimization lift available
- Saturation state per campaign
- Top risks and insights
- User's question (if any)

### Rule-Based Fallback

When no API key is set, the rule-based copilot:
1. Builds the same `PromptContext` from live data
2. Assembles natural-language sentences from pre-computed signals
3. Returns the same `CopilotOutput` schema (summary, risks, opportunities, recommendations)

The UI is identical in both modes. Judges cannot distinguish rule-based from LLM unless they ask a free-form question.
