"""
streamlit_app/state.py
========================
Shared state initialisation and cached data loaders for the Streamlit app.

All pages import from here.  st.cache_resource is used for the model (not
serialisable by Streamlit's data cache) and st.cache_data for DataFrames.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path when Streamlit loads pages
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env so GROQ_API_KEY (and any other secrets) are available to all pages.
# dotenv only sets vars that aren't already in the environment, so shell-level
# overrides still win.
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env", override=False)
except ImportError:
    pass  # dotenv optional; user can set vars in their shell instead

import streamlit as st

# Default artifact paths (produced by demo.py)
FEATURE_STORE_PATH   = _ROOT / "dataset" / "feature_store.parquet"
FORECASTS_PATH       = _ROOT / "dataset" / "forecasts.parquet"
CURVES_PATH          = _ROOT / "dataset" / "curves.pkl"
OPT_RESULT_PATH      = _ROOT / "dataset" / "opt_result.pkl"
MODEL_DIR            = _ROOT / "models"

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    from src.models.lgbm_quantile import RevenueQuantileModel
    if not (MODEL_DIR / "model_meta.pkl").exists():
        return None
    try:
        return RevenueQuantileModel.load(MODEL_DIR)
    except Exception:
        return None


@st.cache_data(show_spinner="Loading feature store...")
def load_feature_store() -> pd.DataFrame | None:
    if not FEATURE_STORE_PATH.exists():
        return None
    return pd.read_parquet(FEATURE_STORE_PATH)


@st.cache_data(show_spinner="Loading forecasts...")
def load_forecasts() -> pd.DataFrame | None:
    if not FORECASTS_PATH.exists():
        return None
    return pd.read_parquet(FORECASTS_PATH)


@st.cache_data(show_spinner="Loading response curves...")
def load_curves() -> dict | None:
    if not CURVES_PATH.exists():
        return None
    with open(CURVES_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner="Loading optimisation result...")
def load_opt_result():
    if not OPT_RESULT_PATH.exists():
        return None
    with open(OPT_RESULT_PATH, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Session-state init
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Initialise all session-state keys on first page load."""
    defaults = {
        "google_budget_pct": 100,
        "meta_budget_pct":   100,
        "bing_budget_pct":   100,
        "campaign_overrides": {},
        "scenario_result":   None,
        "copilot_messages":  [],
        "copilot_output":    None,
        "selected_platforms": [],
        "selected_campaigns": [],
        "forecast_horizon":   7,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Forecast generation
# ---------------------------------------------------------------------------

def generate_forecasts(
    fs: pd.DataFrame,
    model,
    feature_cols: list[str],
    n_days: int = 30,
) -> pd.DataFrame:
    """Generate P10/P50/P90 for the last *n_days* of the feature store."""
    cutoff = fs["date"].max() - pd.Timedelta(days=n_days - 1)
    recent = fs[fs["date"] >= cutoff].copy()
    X = recent[[c for c in feature_cols if c in recent.columns]]
    preds = model.predict(X)
    result = recent[["campaign_id", "campaign_name", "platform", "date", "revenue_attributed"]].copy()
    result["p10"] = preds["p10"].values
    result["p50"] = preds["p50"].values
    result["p90"] = preds["p90"].values
    result["is_future"] = False
    return result


def generate_future_forecasts(
    fs: pd.DataFrame,
    model,
    feature_cols: list[str],
    n_future_days: int = 14,
) -> pd.DataFrame:
    """Autoregressive P10/P50/P90 projection for *n_future_days* beyond the feature store.

    Delegates to src.models.autoregressive.generate_future_forecasts which
    implements a proper day-by-day rollout: each step's revenue lag and
    rolling features are updated from the P50 predictions of prior steps,
    so uncertainty grows naturally with the forecast horizon.
    """
    from src.models.autoregressive import generate_future_forecasts as _ar_forecast
    return _ar_forecast(fs, model, feature_cols, n_future_days=n_future_days)


# ---------------------------------------------------------------------------
# Data-not-ready helper
# ---------------------------------------------------------------------------

def show_not_ready_message() -> None:
    st.error(
        "**Demo data not found.**\n\n"
        "Run `python demo.py` from the project root to train the model "
        "and generate all required artifacts.",
        icon="🚫",
    )
    st.code("python demo.py", language="bash")
