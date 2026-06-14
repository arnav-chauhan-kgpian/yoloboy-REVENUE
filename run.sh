#!/usr/bin/env bash
set -euo pipefail

# meridian — NetElixir AIgnition 3.0 submission entry point
#
# Usage:
#   ./run.sh [DATA_DIR] [MODEL_PATH] [OUTPUT_PATH]
#
# Defaults (for local development):
#   DATA_DIR    = ./data
#   MODEL_PATH  = ./pickle/model.pkl
#   OUTPUT_PATH = ./output/predictions.csv
#
# At test time the pipeline calls:
#   ./run.sh ./data ./pickle/model.pkl ./output/predictions.csv

DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "=================================================="
echo "  meridian — Revenue Forecasting Pipeline"
echo "=================================================="
echo "  Data dir : $DATA_DIR"
echo "  Model    : $MODEL_PATH"
echo "  Output   : $OUTPUT_PATH"
echo ""

# Step 1: Generate features from raw CSVs
echo "[1/2] Generating features..."
python src/generate_features.py \
    --data-dir "$DATA_DIR" \
    --out features.parquet

# Step 2: Load model and produce predictions
echo "[2/2] Running inference..."
python src/predict.py \
    --features features.parquet \
    --model    "$MODEL_PATH" \
    --output   "$OUTPUT_PATH"

echo ""
echo "Done. Predictions written to $OUTPUT_PATH"
