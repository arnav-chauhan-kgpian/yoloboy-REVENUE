"""
build_artifacts.py
==================
Pre-generate all ML artifacts without launching Streamlit.

Used as the Docker / Render build step so the running container starts
instantly with pre-built models and forecasts.

Usage:
    python build_artifacts.py          # full pipeline (2-5 min)
    python build_artifacts.py --fast   # fast config (< 60s, demo quality)
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Import demo.py as a module (main() is guarded by __name__ == "__main__"
# so importing it is side-effect-free).
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location("_demo_module", ROOT / "demo.py")
_demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_demo)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build meridian artifacts for deployment")
    parser.add_argument("--fast", action="store_true",
                        help="Fast training config (n_estimators=200, 2 folds)")
    args = parser.parse_args()

    if args.fast:
        from src.models.lgbm_quantile import QuantileConfig
        from src.models.trainer import get_feature_columns

        _orig_step_model = _demo.step_model

        def _fast_step_model(fs):
            from src.models.lgbm_quantile import QuantileConfig
            from src.models.trainer import train
            from src.models.lgbm_quantile import RevenueQuantileModel

            model_meta = _demo.MODEL_DIR / "model_meta.pkl"
            if model_meta.exists():
                model = RevenueQuantileModel.load(_demo.MODEL_DIR)
                feature_cols = get_feature_columns(fs)
                return model, feature_cols

            fast_cfg = QuantileConfig(n_estimators=200, learning_rate=0.05,
                                      num_leaves=31, early_stopping_rounds=30)
            result = train(fs, model_dir=_demo.MODEL_DIR, config=fast_cfg,
                           validator_kwargs={"max_folds": 2, "min_train_days": 60})
            return result.model, result.feature_columns

        _demo.step_model = _fast_step_model

    t0 = time.time()
    print()
    print("=" * 60)
    print("  meridian — Artifact Builder")
    print("=" * 60)

    fs           = _demo.step_feature_store()
    model, fcols = _demo.step_model(fs)
    curves       = _demo.step_response_curves(fs)
    opt          = _demo.step_optimise(curves)
    _demo.step_copilot(fs, curves, opt)
    _demo.step_forecasts(fs, model, fcols)

    # --- Train scoring model on the REAL feature pipeline ---
    # The app model (models/) is trained on synthetic data.
    # pickle/model.pkl must be trained on build_feature_store() output
    # so that generate_features.py + predict.py share the same schema.
    import tempfile
    from src.features.feature_store import build_feature_store
    from src.models.lgbm_quantile import QuantileConfig
    from src.models.trainer import train as _train

    print()
    print("  [Scoring] Building real feature store from data/...")
    real_fs = build_feature_store(data_dir=ROOT / "data")
    scoring_cfg = QuantileConfig(
        n_estimators=200, learning_rate=0.05,
        num_leaves=31, early_stopping_rounds=30,
    )
    print("  [Scoring] Training scoring model (n_estimators=200)...")
    with tempfile.TemporaryDirectory() as tmp:
        result = _train(
            real_fs,
            model_dir=Path(tmp),
            config=scoring_cfg,
            validator_kwargs={"max_folds": 2, "min_train_days": 60},
        )
        scoring_model = result.model

    pickle_dir = ROOT / "pickle"
    pickle_dir.mkdir(exist_ok=True)
    pickle_path = pickle_dir / "model.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(scoring_model, f, protocol=4)
    print(f"  pickle/model.pkl → {pickle_path}")

    # --- Copy sample CSVs to data/ (scoring pipeline drops in test data here) ---
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    dataset_dir = ROOT / "dataset"
    for csv_name in [
        "google_ads_campaign_stats.csv",
        "meta_ads_campaign_stats.csv",
        "bing_campaign_stats.csv",
    ]:
        src = dataset_dir / csv_name
        dst = data_dir / csv_name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"  data/{csv_name} → copied")

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"  All artifacts built in {elapsed:.0f}s")
    print(f"  Models  → {_demo.MODEL_DIR}/")
    print(f"  Dataset → {_demo.DATASET_DIR}/")
    print(f"  Pickle  → {pickle_dir}/")
    print(f"  Data    → {data_dir}/")
    print()
    print("  Start the app with:")
    print("    streamlit run streamlit_app/main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
