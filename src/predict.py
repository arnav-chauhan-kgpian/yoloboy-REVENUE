"""
src/predict.py
==============
Inference step for the meridian scoring pipeline.

Loads the pickled RevenueQuantileModel from MODEL_PATH, generates
P10/P50/P90 revenue forecasts from the feature parquet produced by
generate_features.py, and writes predictions.csv to OUTPUT_PATH.

Usage:
    python src/predict.py \
        --features features.parquet \
        --model    ./pickle/model.pkl \
        --output   ./output/predictions.csv
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.trainer import get_feature_columns


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate meridian revenue predictions from a pickled model"
    )
    p.add_argument(
        "--features",
        default="features.parquet",
        help="Features parquet produced by generate_features.py",
    )
    p.add_argument(
        "--model",
        default="./pickle/model.pkl",
        help="Path to pickled RevenueQuantileModel",
    )
    p.add_argument(
        "--output",
        default="./output/predictions.csv",
        help="Output path for predictions CSV",
    )
    args = p.parse_args()

    features_path = Path(args.features)
    model_path    = Path(args.model)
    output_path   = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load features -------------------------------------------------------
    print(f"[predict] Loading features from: {features_path.resolve()}")
    fs = pd.read_parquet(features_path)
    print(f"[predict] Feature matrix: {len(fs):,} rows × {fs.shape[1]} cols")

    # --- Load model ----------------------------------------------------------
    print(f"[predict] Loading model from: {model_path.resolve()}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # --- Predict -------------------------------------------------------------
    feature_cols = get_feature_columns(fs)
    X = fs[feature_cols]
    print(f"[predict] Running inference ({len(feature_cols)} features)...")
    preds = model.predict(X)   # returns DataFrame with p10, p50, p90 columns

    # --- Assemble output -----------------------------------------------------
    meta_cols = ["date", "campaign_id", "campaign_name", "platform"]
    available = [c for c in meta_cols if c in fs.columns]
    result = pd.concat(
        [fs[available].reset_index(drop=True), preds.reset_index(drop=True)],
        axis=1,
    )

    # Write fresh — never append
    result.to_csv(output_path, index=False)
    print(
        f"[predict] Wrote {len(result):,} predictions to {output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
