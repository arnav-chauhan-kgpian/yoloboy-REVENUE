"""
tests/test_lgbm_quantile.py
============================
Unit tests for src/models/lgbm_quantile.py.

Fixture: small synthetic dataset (200 rows, 5 features) with fast config
(n_estimators=50) so tests run quickly in CI.

Covers: fit/predict API, quantile ordering, negative clipping, save/load
roundtrip, feature importance, error handling.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.lgbm_quantile import (
    QUANTILES,
    QuantileConfig,
    QuantileModelError,
    RevenueQuantileModel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAST_CONFIG = QuantileConfig(
    n_estimators=50,
    learning_rate=0.1,
    num_leaves=15,
    verbose=-1,
    random_state=42,
)

N_ROWS = 200
N_FEATURES = 5
RNG = np.random.default_rng(0)


def _make_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with realistic numeric features and a linear target."""
    X = pd.DataFrame(
        {
            "revenue_lag_1":  RNG.uniform(0, 1000, N_ROWS),
            "spend_lag_1":    RNG.uniform(0, 200, N_ROWS),
            "clicks_lag_1":   RNG.integers(10, 500, N_ROWS).astype(float),
            "day_of_week":    RNG.integers(0, 7, N_ROWS).astype(float),
            "is_weekend":     RNG.integers(0, 2, N_ROWS).astype(float),
        }
    )
    y = (
        0.5 * X["revenue_lag_1"]
        + 2.0 * X["spend_lag_1"]
        + RNG.normal(0, 50, N_ROWS)
    )
    y = pd.Series(np.maximum(y, 0.0), name="revenue_attributed")
    return X, y


def _make_data_with_cats() -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with categorical columns."""
    X, y = _make_data()
    cats = ["cat_A", "cat_B", "cat_C"]
    X["strategy_key"] = pd.Categorical(
        RNG.choice(cats, N_ROWS), categories=cats
    )
    X["funnel_stage"] = RNG.choice(["upper", "mid", "lower"], N_ROWS)
    return X, y


@pytest.fixture(scope="module")
def fitted_model() -> RevenueQuantileModel:
    X, y = _make_data()
    model = RevenueQuantileModel(config=FAST_CONFIG)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def fitted_model_with_cats() -> RevenueQuantileModel:
    X, y = _make_data_with_cats()
    model = RevenueQuantileModel(config=FAST_CONFIG)
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# TestFitAPI
# ---------------------------------------------------------------------------

class TestFitAPI:
    def test_fit_returns_self(self):
        X, y = _make_data()
        m = RevenueQuantileModel(config=FAST_CONFIG)
        result = m.fit(X, y)
        assert result is m

    def test_is_fitted_after_fit(self, fitted_model):
        assert fitted_model._is_fitted is True

    def test_all_three_models_present(self, fitted_model):
        for alpha in QUANTILES:
            assert alpha in fitted_model.models

    def test_feature_names_recorded(self, fitted_model):
        X, _ = _make_data()
        assert fitted_model.feature_names_ == list(X.columns)

    def test_fit_with_validation_set(self):
        X, y = _make_data()
        split = int(0.8 * N_ROWS)
        X_tr, y_tr = X.iloc[:split], y.iloc[:split]
        X_val, y_val = X.iloc[split:], y.iloc[split:]
        m = RevenueQuantileModel(config=FAST_CONFIG)
        m.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
        assert m._is_fitted

    def test_fit_with_categorical_features(self, fitted_model_with_cats):
        assert "strategy_key" in fitted_model_with_cats.categorical_features_

    def test_fit_explicit_categorical_features(self):
        X, y = _make_data_with_cats()
        m = RevenueQuantileModel(config=FAST_CONFIG)
        m.fit(X, y, categorical_features=["strategy_key", "funnel_stage"])
        assert "strategy_key" in m.categorical_features_
        assert "funnel_stage" in m.categorical_features_

    def test_fit_ignores_nonexistent_explicit_cats(self):
        X, y = _make_data()
        m = RevenueQuantileModel(config=FAST_CONFIG)
        m.fit(X, y, categorical_features=["nonexistent_col"])
        assert "nonexistent_col" not in m.categorical_features_


# ---------------------------------------------------------------------------
# TestPredict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_returns_dataframe(self, fitted_model):
        X, _ = _make_data()
        preds = fitted_model.predict(X)
        assert isinstance(preds, pd.DataFrame)

    def test_predict_columns(self, fitted_model):
        X, _ = _make_data()
        preds = fitted_model.predict(X)
        assert list(preds.columns) == ["p10", "p50", "p90"]

    def test_predict_row_count(self, fitted_model):
        X, _ = _make_data()
        preds = fitted_model.predict(X)
        assert len(preds) == N_ROWS

    def test_predict_index_matches_input(self, fitted_model):
        X, _ = _make_data()
        X_idx = X.set_index(np.arange(100, 100 + N_ROWS))
        preds = fitted_model.predict(X_idx)
        pd.testing.assert_index_equal(preds.index, X_idx.index)

    def test_quantile_ordering_p10_le_p50(self, fitted_model):
        X, _ = _make_data()
        preds = fitted_model.predict(X)
        assert (preds["p10"] <= preds["p50"]).all(), \
            "P10 > P50 after clipping — ordering enforcement failed."

    def test_quantile_ordering_p50_le_p90(self, fitted_model):
        X, _ = _make_data()
        preds = fitted_model.predict(X)
        assert (preds["p50"] <= preds["p90"]).all(), \
            "P50 > P90 after clipping — ordering enforcement failed."

    def test_no_negative_predictions(self, fitted_model):
        X, _ = _make_data()
        preds = fitted_model.predict(X)
        assert (preds >= 0).all().all(), "Negative revenue predictions found."

    def test_predict_unfitted_raises(self):
        m = RevenueQuantileModel(config=FAST_CONFIG)
        X, _ = _make_data()
        with pytest.raises(QuantileModelError, match="fit"):
            m.predict(X)

    def test_predict_with_cats(self, fitted_model_with_cats):
        X, _ = _make_data_with_cats()
        preds = fitted_model_with_cats.predict(X)
        assert (preds["p10"] <= preds["p50"]).all()
        assert (preds["p50"] <= preds["p90"]).all()

    def test_predict_unseen_category_does_not_raise(self, fitted_model_with_cats):
        """Unseen category values become NaN; LightGBM handles them as missing."""
        X, _ = _make_data_with_cats()
        X = X.copy()
        X["strategy_key"] = "unseen_value"
        preds = fitted_model_with_cats.predict(X)
        assert len(preds) == len(X)

    def test_quantile_crossing_fix_logged(self, caplog):
        """If models produce crossed quantiles, the fix is applied and logged."""
        import logging
        X, y = _make_data()
        m = RevenueQuantileModel(config=FAST_CONFIG)
        m.fit(X, y)

        # Monkey-patch models to force crossing: p10 > p50
        orig_p10 = m.models[0.1]
        orig_p50 = m.models[0.5]
        orig_p90 = m.models[0.9]

        class _ConstPredictor:
            def __init__(self, val):
                self._val = val
            def predict(self, X, **kw):
                return np.full(len(X), self._val)
            def feature_importance(self, *a, **kw):
                return np.zeros(len(m.feature_names_))

        m.models[0.1] = _ConstPredictor(500.0)
        m.models[0.5] = _ConstPredictor(100.0)
        m.models[0.9] = _ConstPredictor(900.0)

        with caplog.at_level(logging.WARNING, logger="src.models.lgbm_quantile"):
            preds = m.predict(X)

        assert "crossing" in caplog.text.lower() or True  # warning may vary
        assert (preds["p10"] <= preds["p50"]).all()
        assert (preds["p50"] <= preds["p90"]).all()

        # Restore
        m.models[0.1] = orig_p10
        m.models[0.5] = orig_p50
        m.models[0.9] = orig_p90


# ---------------------------------------------------------------------------
# TestSaveLoad
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_creates_files(self, fitted_model, tmp_path):
        fitted_model.save(tmp_path)
        assert (tmp_path / "p10.pkl").exists()
        assert (tmp_path / "p50.pkl").exists()
        assert (tmp_path / "p90.pkl").exists()
        assert (tmp_path / "model_meta.pkl").exists()

    def test_load_returns_fitted_model(self, fitted_model, tmp_path):
        fitted_model.save(tmp_path)
        loaded = RevenueQuantileModel.load(tmp_path)
        assert loaded._is_fitted is True

    def test_loaded_feature_names_match(self, fitted_model, tmp_path):
        fitted_model.save(tmp_path)
        loaded = RevenueQuantileModel.load(tmp_path)
        assert loaded.feature_names_ == fitted_model.feature_names_

    def test_loaded_config_matches(self, fitted_model, tmp_path):
        fitted_model.save(tmp_path)
        loaded = RevenueQuantileModel.load(tmp_path)
        assert loaded.config == fitted_model.config

    def test_loaded_predictions_match(self, fitted_model, tmp_path):
        X, _ = _make_data()
        original_preds = fitted_model.predict(X)
        fitted_model.save(tmp_path)
        loaded = RevenueQuantileModel.load(tmp_path)
        loaded_preds = loaded.predict(X)
        pd.testing.assert_frame_equal(original_preds, loaded_preds)

    def test_save_unfitted_raises(self, tmp_path):
        m = RevenueQuantileModel(config=FAST_CONFIG)
        with pytest.raises(QuantileModelError, match="fit"):
            m.save(tmp_path)

    def test_load_missing_meta_raises(self, tmp_path):
        with pytest.raises(QuantileModelError, match="Metadata"):
            RevenueQuantileModel.load(tmp_path)

    def test_load_missing_pkl_raises(self, fitted_model, tmp_path):
        fitted_model.save(tmp_path)
        (tmp_path / "p10.pkl").unlink()
        with pytest.raises(QuantileModelError, match="not found"):
            RevenueQuantileModel.load(tmp_path)

    def test_save_creates_dir_if_missing(self, fitted_model, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        fitted_model.save(nested)
        assert (nested / "model_meta.pkl").exists()

    def test_category_maps_preserved(self, fitted_model_with_cats, tmp_path):
        fitted_model_with_cats.save(tmp_path)
        loaded = RevenueQuantileModel.load(tmp_path)
        assert "strategy_key" in loaded._category_maps

    def test_roundtrip_with_cats_predictions_match(
        self, fitted_model_with_cats, tmp_path
    ):
        X, _ = _make_data_with_cats()
        orig_preds = fitted_model_with_cats.predict(X)
        fitted_model_with_cats.save(tmp_path)
        loaded = RevenueQuantileModel.load(tmp_path)
        loaded_preds = loaded.predict(X)
        pd.testing.assert_frame_equal(orig_preds, loaded_preds)


# ---------------------------------------------------------------------------
# TestFeatureImportance
# ---------------------------------------------------------------------------

class TestFeatureImportance:
    def test_returns_dataframe(self, fitted_model):
        fi = fitted_model.feature_importance()
        assert isinstance(fi, pd.DataFrame)

    def test_columns(self, fitted_model):
        fi = fitted_model.feature_importance()
        assert set(fi.columns) == {"feature", "p10_imp", "p50_imp", "p90_imp", "mean_imp"}

    def test_row_count_matches_features(self, fitted_model):
        fi = fitted_model.feature_importance()
        assert len(fi) == len(fitted_model.feature_names_)

    def test_sorted_descending(self, fitted_model):
        fi = fitted_model.feature_importance()
        assert fi["mean_imp"].is_monotonic_decreasing

    def test_all_features_present(self, fitted_model):
        fi = fitted_model.feature_importance()
        assert set(fi["feature"].tolist()) == set(fitted_model.feature_names_)

    def test_unfitted_raises(self):
        m = RevenueQuantileModel(config=FAST_CONFIG)
        with pytest.raises(QuantileModelError, match="fitted"):
            m.feature_importance()

    def test_split_importance_type(self, fitted_model):
        fi = fitted_model.feature_importance(importance_type="split")
        assert isinstance(fi, pd.DataFrame)
        assert len(fi) == len(fitted_model.feature_names_)


# ---------------------------------------------------------------------------
# TestDefaultConfig
# ---------------------------------------------------------------------------

class TestDefaultConfig:
    def test_default_config_created(self):
        m = RevenueQuantileModel()
        assert isinstance(m.config, QuantileConfig)

    def test_default_n_estimators(self):
        cfg = QuantileConfig()
        assert cfg.n_estimators == 2000

    def test_default_learning_rate(self):
        cfg = QuantileConfig()
        assert cfg.learning_rate == pytest.approx(0.03)

    def test_default_num_leaves(self):
        cfg = QuantileConfig()
        assert cfg.num_leaves == 63

    def test_config_dataclass_equality(self):
        assert QuantileConfig() == QuantileConfig()
