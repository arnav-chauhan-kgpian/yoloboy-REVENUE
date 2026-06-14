"""
run_training.py
===============
Standalone training entrypoint for AIgnition.

Builds the feature store, trains LightGBM P10/P50/P90 quantile models,
fits Hill saturation curves, runs the budget optimizer, and saves all
artifacts to disk.

Usage:
    python run_training.py
    python run_training.py --data-dir dataset --model-dir models
    python run_training.py --fast          # n_estimators=200, 2 folds
    python run_training.py --production    # n_estimators=2000, full CV

Output directories:
    models/             p10.pkl  p50.pkl  p90.pkl  model_meta.pkl
    dataset/            feature_store.parquet  forecasts.parquet
                        curves.pkl  opt_result.pkl
    evaluation_report.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AIgnition revenue forecasting models")
    p.add_argument("--data-dir",  default="dataset", help="Directory with raw CSV files")
    p.add_argument("--model-dir", default="models",  help="Output directory for model artifacts")
    p.add_argument("--report",    default="evaluation_report.json")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--fast",       action="store_true",
                   help="Fast config: n_estimators=200, 2 CV folds (~1 min)")
    g.add_argument("--production", action="store_true",
                   help="Production config: n_estimators=2000, full CV (~10 min)")
    return p.parse_args()


def main() -> None:
    args = parse_args = _parse_args()

    data_dir  = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    from src.models.lgbm_quantile import QuantileConfig, RevenueQuantileModel
    from src.models.trainer import train, get_feature_columns
    from src.simulation.response_curve import build_response_curves
    from src.simulation.optimizer import optimize_budget

    # Config
    if args.production:
        config = QuantileConfig(n_estimators=2000, learning_rate=0.03)
        validator_kwargs = None
    elif args.fast:
        config = QuantileConfig(n_estimators=200, learning_rate=0.05,
                                num_leaves=31, early_stopping_rounds=30)
        validator_kwargs = {"max_folds": 2, "min_train_days": 60}
    else:
        config = None
        validator_kwargs = None

    t0 = time.time()
    print()
    print("=" * 60)
    print("  AIgnition Training Pipeline")
    mode = "production" if args.production else ("fast" if args.fast else "default")
    print(f"  Mode: {mode}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 — Feature store
    # ------------------------------------------------------------------
    print("\n[1/5] Building feature store...")
    fs_path = data_dir / "feature_store.parquet"

    if fs_path.exists():
        print(f"  Loading existing feature store from {fs_path}")
        fs = pd.read_parquet(fs_path)
    else:
        # Try real data first; fall back to synthetic
        try:
            from src.features.feature_store import build_feature_store
            fs = build_feature_store(data_dir=data_dir)
            print("  Built from real CSV data")
        except Exception:
            from demo import _generate_synthetic_feature_store
            fs = _generate_synthetic_feature_store()
            print("  Generated synthetic feature store (no CSVs found)")

        fs.to_parquet(fs_path, index=False)

    print(f"  {len(fs):,} rows × {len(fs.columns)} columns → {fs_path}")

    # ------------------------------------------------------------------
    # Step 2 — Train quantile models
    # ------------------------------------------------------------------
    print("\n[2/5] Training LightGBM quantile models (P10 / P50 / P90)...")
    result = train(
        fs=fs,
        model_dir=model_dir,
        report_path=args.report,
        config=config,
        validator_kwargs=validator_kwargs,
    )

    agg = result.aggregated_metrics
    print(f"  Folds:        {result.n_folds}")
    print(f"  Train rows:   {result.n_total_train_rows:,}")
    print(f"  Features:     {result.n_features}")
    print(f"  MAE  (P50):   ${agg.get('mae_p50', float('nan')):.0f}")
    print(f"  MAPE (P50):   {agg.get('mape_p50', float('nan')):.1f}%")
    print(f"  Coverage 80%: {agg.get('coverage_80', float('nan')) * 100:.1f}%")

    # ------------------------------------------------------------------
    # Step 3 — Hill saturation curves
    # ------------------------------------------------------------------
    print("\n[3/5] Fitting Hill saturation curves...")
    curves_path = data_dir / "curves.pkl"
    curves = build_response_curves(fs)
    with open(curves_path, "wb") as f:
        pickle.dump(curves, f)

    n_reliable = sum(1 for c in curves.values() if c.is_reliable)
    n_sat      = sum(1 for c in curves.values() if c.saturation_score > 0.75)
    print(f"  {len(curves)} campaigns  |  {n_reliable} reliable  |  {n_sat} saturated")

    # ------------------------------------------------------------------
    # Step 4 — Budget optimization
    # ------------------------------------------------------------------
    print("\n[4/5] Running SLSQP budget optimization...")
    opt_path     = data_dir / "opt_result.pkl"
    total_budget = sum(c.avg_daily_spend for c in curves.values())
    opt          = optimize_budget(curves, total_budget=total_budget)

    with open(opt_path, "wb") as f:
        pickle.dump(opt, f)

    print(f"  Converged:       {opt.converged}")
    print(f"  Revenue lift:    +${opt.revenue_lift:,.0f}/day  (+{opt.revenue_lift_pct:.1f}%)")
    print(f"  Baseline ROAS:   {opt.baseline_roas:.2f}x")
    print(f"  Optimised ROAS:  {opt.optimal_roas:.2f}x")

    # ------------------------------------------------------------------
    # Step 5 — Forecasts
    # ------------------------------------------------------------------
    print("\n[5/5] Generating P10/P50/P90 forecasts...")
    fc_path      = data_dir / "forecasts.parquet"
    feature_cols = get_feature_columns(fs)
    mature       = fs[fs["attribution_mature"]].copy()
    cutoff       = pd.Timestamp(mature["date"].max()) - pd.Timedelta(days=29)
    recent       = mature[mature["date"] >= cutoff].copy()
    X            = recent[feature_cols]
    preds        = result.model.predict(X)

    meta_cols = ["date", "campaign_id", "campaign_name", "platform", "revenue_attributed"]
    fc = pd.concat(
        [recent[[c for c in meta_cols if c in recent.columns]].reset_index(drop=True),
         preds.reset_index(drop=True)],
        axis=1,
    )
    fc["is_future"] = False
    fc.to_parquet(fc_path, index=False)

    n_campaigns = fc["campaign_id"].nunique()
    print(f"  {len(fc):,} rows  ·  {n_campaigns} campaigns  →  {fc_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"  Training complete in {elapsed:.0f}s")
    print(f"  Models  → {model_dir}/")
    print(f"  Report  → {args.report}")
    print()
    print("  Next steps:")
    print("    streamlit run streamlit_app/main.py")
    print("    python run_inference.py  --future-days 14")
    print("=" * 60)


if __name__ == "__main__":
    main()
