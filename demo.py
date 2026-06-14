"""
demo.py — meridian 2026 Hackathon Demo
=========================================
Full demo (trains model on first run):
    python demo.py

Fast demo mode (requires pre-generated artefacts):
    python demo.py --demo

What it does:
  1. Generate a synthetic feature store (or load from dataset/)
  2. Train (or load) the LightGBM quantile revenue model
  3. Build Hill saturation curves for every campaign
  4. Optimise budget allocation with SLSQP
  5. Run the AI Copilot to produce insights, risks, and recommendations
  6. Save all artefacts to dataset/ and models/
  7. Launch the Streamlit app
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_DIR  = PROJECT_ROOT / "dataset"
MODEL_DIR    = PROJECT_ROOT / "models"
FS_PATH      = DATASET_DIR / "feature_store.parquet"
FC_PATH      = DATASET_DIR / "forecasts.parquet"
CURVES_PATH  = DATASET_DIR / "curves.pkl"
OPT_PATH     = DATASET_DIR / "opt_result.pkl"

DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(label: str) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {label}")
    print(f"{'=' * width}")


def _ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def _info(msg: str) -> None:
    print(f"  [..] {msg}")


# ---------------------------------------------------------------------------
# STEP 1 — Feature store
# ---------------------------------------------------------------------------

def _generate_synthetic_feature_store(
    n_campaigns: int = 12,
    n_days: int = 180,
) -> pd.DataFrame:
    """Generate a realistic synthetic feature store for demo purposes."""
    rng = np.random.default_rng(2026)

    platforms = ["google", "google", "google", "google", "meta", "meta", "meta", "bing", "bing", "bing", "google", "meta"]
    formats   = ["Search", "Display", "Shopping", "Video", "Feed", "Stories", "Reels", "Search", "Shopping", "Display", "Shopping", "Feed"]
    funnel    = ["TM", "NTM", "NTM", "NTM", "TM", "NTM", "NTM", "TM", "NTM", "NTM", "TM", "NTM"]

    base_spends = [500, 300, 400, 200, 450, 350, 250, 100, 150, 80, 600, 280]
    roas_vals   = [4.2, 2.8, 3.1, 1.9, 3.5, 2.3, 1.7, 2.1, 1.8, 1.2, 3.8, 2.6]

    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows: list[dict] = []

    for cid in range(min(n_campaigns, len(platforms))):
        platform   = platforms[cid]
        fmt        = formats[cid]
        fn_stage   = funnel[cid]
        base_spend = base_spends[cid]
        roas       = roas_vals[cid]
        v_max      = roas * base_spend * 2.2
        K          = base_spend * 0.9

        for i, date in enumerate(dates):
            # Seasonal component (weekly pattern)
            dow          = date.dayofweek
            weekly_scale = 1.0 + 0.15 * np.sin(2 * np.pi * dow / 7)

            # Monthly growth trend
            trend = 1.0 + (i / n_days) * 0.12

            spend_noise = rng.normal(0, base_spend * 0.22)
            spend = max(base_spend * weekly_scale * trend + spend_noise, 5.0)

            # Hill-curve revenue with noise
            rev = v_max * spend / (K + spend)
            rev *= weekly_scale * trend
            rev += rng.normal(0, rev * 0.08)
            rev  = max(rev, 0.0)

            # Attribution maturity: last 14 days not yet attributed
            is_mature = i < (n_days - 14)

            rows.append({
                "date":                date,
                "campaign_id":         f"camp_{cid:02d}",
                "campaign_name":       f"{platform.title()} {fmt} {fn_stage}",
                "platform":            platform,
                "format":              fmt,
                "audience_strategy":   "broad" if cid % 2 == 0 else "retargeting",
                "funnel_stage":        fn_stage,
                "ad_product_type":     fmt.lower(),
                "strategy_key":        f"{platform}_{fmt.lower()}_{fn_stage.lower()}",
                "channel_format":      f"{platform}_{fmt.lower()}",
                "spend":               round(spend, 2),
                "impressions":         int(spend * rng.uniform(80, 200)),
                "clicks":              int(spend * rng.uniform(2, 8)),
                "revenue_attributed":  round(rev, 2),
                "attribution_mature":  is_mature,
                "day_of_week":         date.dayofweek,
                "day_of_month":        date.day,
                "week_of_year":        date.isocalendar().week,
                "month":               date.month,
                "quarter":             date.quarter,
                "year":                date.year,
                "is_weekend":          int(date.dayofweek >= 5),
                "holiday_intensity":   float(rng.uniform(0, 0.1)),
                "holiday_name":        "",
                "spend_lag_1":         round(max(base_spend * weekly_scale + rng.normal(0, base_spend * 0.2), 0.0), 2),
                "spend_lag_7":         round(max(base_spend * 0.95 + rng.normal(0, base_spend * 0.18), 0.0), 2),
                "revenue_lag_1":       round(max(rev * 0.95 + rng.normal(0, rev * 0.05), 0.0), 2),
                "revenue_lag_7":       round(max(rev * 0.92 + rng.normal(0, rev * 0.07), 0.0), 2),
                "spend_roll7_mean":    round(max(base_spend + rng.normal(0, base_spend * 0.1), 0.0), 2),
                "spend_roll7_std":     round(max(base_spend * 0.15, 0.0), 2),
                "revenue_roll7_mean":  round(max(rev * 0.97, 0.0), 2),
                "revenue_roll7_std":   round(max(rev * 0.08, 0.0), 2),
                "revenue_roll30_mean": round(max(rev * 0.95, 0.0), 2),
                "spend_roll30_mean":   round(max(base_spend * 1.02, 0.0), 2),
            })

    return pd.DataFrame(rows)


def step_feature_store() -> pd.DataFrame:
    _sep("STEP 1 / 6  —  Feature Store")

    if FS_PATH.exists():
        _info(f"Loading existing feature store from {FS_PATH}")
        fs = pd.read_parquet(FS_PATH)
        _ok(f"Loaded {len(fs):,} rows × {len(fs.columns)} columns")
        return fs

    _info("Generating synthetic feature store (12 campaigns × 180 days)…")
    fs = _generate_synthetic_feature_store()
    fs.to_parquet(FS_PATH, index=False)
    _ok(f"Saved {len(fs):,} rows to {FS_PATH}")
    return fs


# ---------------------------------------------------------------------------
# STEP 2 — Model training / loading
# ---------------------------------------------------------------------------

def step_model(fs: pd.DataFrame):
    from src.models.lgbm_quantile import RevenueQuantileModel, QuantileConfig
    from src.models.trainer import train, get_feature_columns

    _sep("STEP 2 / 6  —  LightGBM Quantile Model")

    model_meta = MODEL_DIR / "model_meta.pkl"
    if model_meta.exists():
        _info(f"Loading existing model from {MODEL_DIR}")
        model = RevenueQuantileModel.load(MODEL_DIR)
        feature_cols = get_feature_columns(fs)
        _ok(f"Model loaded  |  {len(feature_cols)} features")
        return model, feature_cols

    _info("Training LightGBM quantile model (P10 / P50 / P90)…")
    _info("  (using fast config for demo — full config: n_estimators=2000)")

    fast_cfg = QuantileConfig(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        early_stopping_rounds=30,
    )
    result = train(fs, model_dir=MODEL_DIR, config=fast_cfg, validator_kwargs={"max_folds": 2, "min_train_days": 60})

    agg = result.aggregated_metrics
    _ok(f"Training complete — {result.n_folds} folds")
    _ok(f"  MAE (P50): ${agg.get('mae_p50', float('nan')):.0f}")
    _ok(f"  MAPE:      {agg.get('mape_p50', float('nan')):.1f}%")
    _ok(f"  Coverage:  {agg.get('coverage_80', float('nan')) * 100:.1f}%")

    return result.model, result.feature_columns


# ---------------------------------------------------------------------------
# STEP 3 — Response curves
# ---------------------------------------------------------------------------

def step_response_curves(fs: pd.DataFrame) -> dict:
    from src.simulation.response_curve import build_response_curves

    _sep("STEP 3 / 6  —  Hill Saturation Curves")

    if CURVES_PATH.exists():
        _info(f"Loading existing curves from {CURVES_PATH}")
        with open(CURVES_PATH, "rb") as f:
            curves = pickle.load(f)
        _ok(f"Loaded {len(curves)} campaign curves")
    else:
        _info("Fitting Hill saturation curves for each campaign…")
        curves = build_response_curves(fs)
        with open(CURVES_PATH, "wb") as f:
            pickle.dump(curves, f)
        _ok(f"Fitted and saved {len(curves)} curves")

    n_reliable = sum(1 for c in curves.values() if c.is_reliable)
    n_sat      = sum(1 for c in curves.values() if c.saturation_score > 0.75)
    _ok(f"  Reliable curves:   {n_reliable}/{len(curves)}")
    _ok(f"  Saturated (>75%):  {n_sat}/{len(curves)}")
    return curves


# ---------------------------------------------------------------------------
# STEP 4 — Budget optimisation
# ---------------------------------------------------------------------------

def step_optimise(curves: dict):
    from src.simulation.optimizer import optimize_budget

    _sep("STEP 4 / 6  —  Budget Optimisation (SLSQP)")

    if OPT_PATH.exists():
        _info(f"Loading existing optimisation result from {OPT_PATH}")
        with open(OPT_PATH, "rb") as f:
            opt = pickle.load(f)
        _ok(f"Loaded result — revenue lift: +{opt.revenue_lift_pct:.1f}%")
        return opt

    total_budget = sum(c.avg_daily_spend for c in curves.values())
    _info(f"Optimising ${total_budget:,.0f}/day across {len(curves)} campaigns…")
    opt = optimize_budget(curves, total_budget=total_budget)

    with open(OPT_PATH, "wb") as f:
        pickle.dump(opt, f)

    _ok(f"Optimisation {'converged' if opt.converged else 'did not fully converge'}")
    _ok(f"  Baseline revenue:  ${opt.baseline_total_revenue:,.0f}/day")
    _ok(f"  Optimal revenue:   ${opt.optimal_total_revenue:,.0f}/day")
    _ok(f"  Revenue lift:      +${opt.revenue_lift:,.0f}/day  (+{opt.revenue_lift_pct:.1f}%)")
    _ok(f"  Baseline ROAS:     {opt.baseline_roas:.2f}x")
    _ok(f"  Optimal ROAS:      {opt.optimal_roas:.2f}x")
    _ok(f"  Risk score:        {opt.risk_score:.2f}")
    return opt


# ---------------------------------------------------------------------------
# STEP 5 — AI Copilot
# ---------------------------------------------------------------------------

def step_copilot(fs: pd.DataFrame, curves: dict, opt):
    from src.copilot.insight_engine import InsightEngine
    from src.copilot.risk_detector   import RiskDetector
    from src.copilot.recommender     import from_optimizer_result
    from src.copilot.llm_client      import LLMClient

    _sep("STEP 5 / 6  —  AI Copilot (Grounded Intelligence)")

    engine   = InsightEngine()
    detector = RiskDetector()

    _info("Generating insights…")
    insights = engine.generate_all_insights(forecasts=pd.DataFrame(), fs=fs)
    _ok(f"  {len(insights)} insights generated")
    for i in insights[:3]:
        _ok(f"  [{i.severity.value.upper()}] {i.title}")

    _info("Detecting risks…")
    risks = detector.detect_all_risks(fs=fs, curves=curves, forecasts=pd.DataFrame())
    _ok(f"  {len(risks)} risks detected")
    for r in risks[:3]:
        _ok(f"  [{r.severity.value.upper()}] {r.title}")

    _info("Generating recommendations…")
    recs = from_optimizer_result(opt, curves)
    _ok(f"  {len(recs)} recommendations")
    for r in recs[:3]:
        _ok(f"  [{r.priority.name}] {r.title}  (lift: ${r.expected_revenue_lift:,.0f}/day)")

    _info("Initialising LLM Copilot…")
    llm = LLMClient()
    if llm.is_llm_available:
        _ok("Claude AI connected — LLM-backed analysis enabled")
    else:
        _ok("Rule-based copilot active (set ANTHROPIC_API_KEY to enable Claude AI)")

    _info("Running full copilot analysis…")
    output = llm.analyse(
        forecasts=pd.DataFrame(),
        fs=fs,
        curves=curves,
        opt_result=opt,
        insights=insights,
        risks=risks,
        recommendations=recs,
    )
    _ok(f"Copilot source: {output.source}")
    _ok(f"Confidence:     {output.confidence:.0%}")
    print()
    print("  " + "─" * 60)
    print("  EXECUTIVE SUMMARY")
    print("  " + "─" * 60)
    for line in output.summary.split(". "):
        if line.strip():
            print(f"  {line.strip()}.")
    print("  " + "─" * 60)

    return output


# ---------------------------------------------------------------------------
# STEP 6 — Forecasts (generate for Streamlit)
# ---------------------------------------------------------------------------

def step_forecasts(fs: pd.DataFrame, model, feature_cols: list[str]) -> None:
    _sep("STEP 6 / 6  —  Generating Forecasts Artefact")

    if FC_PATH.exists():
        _info(f"Forecast artefact already exists at {FC_PATH}")
        return

    _info("Running model.predict on recent feature store rows…")
    try:
        mature = fs[fs["attribution_mature"]].copy()
        # Use date-based cutoff so ALL campaigns get the last 30 days.
        # .tail(N) on a campaign-ordered DataFrame only captures the last
        # 2–3 campaigns alphabetically, not all N campaigns.
        cutoff = pd.Timestamp(mature["date"].max()) - pd.Timedelta(days=29)
        recent = mature[mature["date"] >= cutoff].copy()

        meta_cols = ["date", "campaign_id", "campaign_name", "platform", "revenue_attributed"]
        X     = recent[feature_cols]
        preds = model.predict(X)
        result_df = pd.concat(
            [recent[[c for c in meta_cols if c in recent.columns]].reset_index(drop=True),
             preds.reset_index(drop=True)],
            axis=1,
        )
        result_df["is_future"] = False
        result_df.to_parquet(FC_PATH, index=False)
        n_camps = result_df["campaign_id"].nunique()
        _ok(f"Saved {len(result_df):,} forecast rows ({n_camps} campaigns) to {FC_PATH}")
    except Exception as exc:
        _info(f"Could not generate forecasts (model may not be trained): {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _demo_mode_load() -> bool:
    """Load artefacts for --demo mode. Returns True if all artefacts present."""
    missing = [p for p in [FS_PATH, FC_PATH, CURVES_PATH, OPT_PATH] if not p.exists()]
    model_meta = MODEL_DIR / "model_meta.pkl"
    if not model_meta.exists():
        missing.append(model_meta)

    if missing:
        print()
        print("  [ERROR] --demo mode requires pre-generated artefacts. Missing:")
        for p in missing:
            print(f"    {p}")
        print()
        print("  Run once WITHOUT --demo to train and generate artefacts:")
        print("    python demo.py")
        return False

    _ok("All artefacts found — skipping training.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="meridian Demo Launcher")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Fast mode: load existing artefacts, skip training (requires prior run)",
    )
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("   AIGNITION 2026  |  Ecommerce Revenue Forecasting + Intelligence")
    print("=" * 70)

    if args.demo:
        _sep("DEMO MODE  —  Loading Pre-Generated Artefacts")
        if not _demo_mode_load():
            sys.exit(1)
    else:
        # Steps 1–6: full pipeline
        fs            = step_feature_store()
        model, fcols  = step_model(fs)
        curves        = step_response_curves(fs)
        opt           = step_optimise(curves)
        _              = step_copilot(fs, curves, opt)
        step_forecasts(fs, model, fcols)

    _sep("DEMO COMPLETE  —  Launching Streamlit App")

    app_path = PROJECT_ROOT / "streamlit_app" / "main.py"
    if not app_path.exists():
        print(f"  Streamlit app not found at {app_path}")
        print("  Run manually: streamlit run streamlit_app/main.py")
        return

    _ok("Starting streamlit…  (Ctrl-C to stop)")
    print()
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            cwd=str(PROJECT_ROOT),
        )
    except KeyboardInterrupt:
        print("\n  Streamlit stopped.")


if __name__ == "__main__":
    main()
