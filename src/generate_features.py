"""
src/generate_features.py
========================
Feature generation step for the meridian scoring pipeline.

Reads the three raw advertising CSVs from DATA_DIR, runs the full
feature-engineering pipeline (harmonize → taxonomy → lags → holidays),
and writes the result to a parquet file consumed by predict.py.

Usage:
    python src/generate_features.py --data-dir ./data --out features.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.feature_store import build_feature_store


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate meridian feature store from raw advertising CSVs"
    )
    p.add_argument(
        "--data-dir",
        default="./data",
        help="Directory containing google_ads_campaign_stats.csv, "
             "meta_ads_campaign_stats.csv, bing_campaign_stats.csv",
    )
    p.add_argument(
        "--out",
        default="features.parquet",
        help="Output path for the feature parquet file",
    )
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[generate_features] Reading CSVs from: {data_dir.resolve()}")
    fs = build_feature_store(data_dir=data_dir)
    fs.to_parquet(out_path, index=False)
    print(
        f"[generate_features] Wrote {len(fs):,} rows × {fs.shape[1]} cols "
        f"to {out_path.resolve()}"
    )


if __name__ == "__main__":
    main()
