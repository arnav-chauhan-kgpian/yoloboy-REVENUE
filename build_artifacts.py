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

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"  All artifacts built in {elapsed:.0f}s")
    print(f"  Models  → {_demo.MODEL_DIR}/")
    print(f"  Dataset → {_demo.DATASET_DIR}/")
    print()
    print("  Start the app with:")
    print("    streamlit run streamlit_app/main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
