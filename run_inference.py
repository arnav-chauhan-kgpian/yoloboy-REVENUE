"""
run_inference.py
================
Standalone inference entrypoint for AIgnition.

Loads pre-trained models, generates P10/P50/P90 forecasts for the most
recent N days of every campaign, and optionally appends an autoregressive
forward projection.

Usage:
    python run_inference.py
    python run_inference.py --model-dir models --data-dir dataset
    python run_inference.py --days 30 --future-days 14
    python run_inference.py --output my_forecasts.parquet

Output:
    dataset/forecasts.parquet   (default)
    Schema: date · campaign_id · campaign_name · platform ·
            revenue_attributed · p10 · p50 · p90 · is_future

Prerequisites:
    Run python run_training.py (or python demo.py) first to train the models.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate AIgnition revenue forecasts")
    p.add_argument("--model-dir",    default="models",
                   help="Directory containing trained model artifacts")
    p.add_argument("--data-dir",     default="dataset",
                   help="Directory containing feature_store.parquet")
    p.add_argument("--output",       default=None,
                   help="Output parquet path (default: <data-dir>/forecasts.parquet)")
    p.add_argument("--days",         type=int, default=30,
                   help="Historical days to include (per campaign)")
    p.add_argument("--future-days",  type=int, default=0,
                   help="Autoregressive future horizon in days (0 = skip)")
    p.add_argument("--validate",     action="store_true",
                   help="Print schema validation summary after writing")
    return p.parse_args()


def _validate_schema(df: pd.DataFrame, path: Path) -> None:
    required = {"date", "campaign_id", "campaign_name", "platform",
                "revenue_attributed", "p10", "p50", "p90", "is_future"}
    missing = required - set(df.columns)
    if missing:
        print(f"  [WARN] Missing columns: {missing}")
    else:
        print("  Schema OK — all required columns present")

    print(f"  Rows:       {len(df):,}")
    print(f"  Campaigns:  {df['campaign_id'].nunique()}")
    print(f"  Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"  is_future:  {df['is_future'].value_counts().to_dict()}")
    crossing = (df["p10"] > df["p50"]).sum() + (df["p50"] > df["p90"]).sum()
    print(f"  Quantile crossings: {crossing}")
    print(f"  Negative P50: {(df['p50'] < 0).sum()}")


def main() -> None:
    args      = _parse_args()
    model_dir = Path(args.model_dir)
    data_dir  = Path(args.data_dir)
    out_path  = Path(args.output) if args.output else data_dir / "forecasts.parquet"

    from src.models.lgbm_quantile import RevenueQuantileModel
    from src.models.trainer import get_feature_columns

    t0 = time.time()
    print()
    print("=" * 60)
    print("  AIgnition Inference Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("\n[1/3] Loading model...")
    if not (model_dir / "model_meta.pkl").exists():
        print(f"  ERROR: No trained model found at {model_dir}/")
        print("  Run: python run_training.py")
        sys.exit(1)

    model = RevenueQuantileModel.load(model_dir)
    print(f"  Loaded from {model_dir}/")

    # ------------------------------------------------------------------
    # Load feature store
    # ------------------------------------------------------------------
    print("\n[2/3] Loading feature store...")
    fs_path = data_dir / "feature_store.parquet"
    if not fs_path.exists():
        print(f"  ERROR: Feature store not found at {fs_path}")
        print("  Run: python run_training.py")
        sys.exit(1)

    fs           = pd.read_parquet(fs_path)
    feature_cols = get_feature_columns(fs)
    print(f"  {len(fs):,} rows  ·  {len(feature_cols)} features")

    # ------------------------------------------------------------------
    # Generate historical quantile forecasts
    # ------------------------------------------------------------------
    print(f"\n[3/3] Generating {args.days}-day historical forecasts...")
    mature = fs[fs["attribution_mature"]].copy()
    cutoff = pd.Timestamp(mature["date"].max()) - pd.Timedelta(days=args.days - 1)
    recent = mature[mature["date"] >= cutoff].copy()

    X     = recent[feature_cols]
    preds = model.predict(X)

    meta_cols = ["date", "campaign_id", "campaign_name", "platform", "revenue_attributed"]
    fc = pd.concat(
        [recent[[c for c in meta_cols if c in recent.columns]].reset_index(drop=True),
         preds.reset_index(drop=True)],
        axis=1,
    )
    fc["is_future"] = False

    n_campaigns = fc["campaign_id"].nunique()
    print(f"  {len(fc):,} rows  ·  {n_campaigns} campaigns")

    # ------------------------------------------------------------------
    # Autoregressive future projection (optional)
    # ------------------------------------------------------------------
    if args.future_days > 0:
        print(f"\n  Generating {args.future_days}-day autoregressive projection...")
        try:
            from src.models.autoregressive import generate_future_forecasts
            future = generate_future_forecasts(fs, model, feature_cols, args.future_days)
            future["is_future"] = True
            fc = pd.concat([fc, future], ignore_index=True)
            print(f"  Added {len(future):,} future rows")
        except Exception as exc:
            print(f"  [WARN] Future forecast failed: {exc}  (skipped)")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(out_path, index=False)

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"  Forecasts saved → {out_path}")
    print(f"  Elapsed: {elapsed:.1f}s")

    if args.validate:
        print()
        _validate_schema(fc, out_path)

    print("=" * 60)


if __name__ == "__main__":
    main()
