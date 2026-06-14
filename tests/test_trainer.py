"""
tests/test_trainer.py
======================
Unit and integration tests for src/models/trainer.py.

Uses a small synthetic feature store (4 campaigns × 200 days) with
n_estimators=10 and max_folds=2 so tests finish quickly in CI.

Covers: metric functions, feature column selection, full training pipeline,
model persistence, evaluation report JSON, TrainingResult fields.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.lgbm_quantile import QuantileConfig
from src.models.trainer import (
    FoldMetrics,
    TrainerError,
    TrainingResult,
    coverage,
    get_feature_columns,
    mae,
    mape,
    pinball_loss,
    rmse,
    smape,
    train,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

FAST_CONFIG = QuantileConfig(
    n_estimators=10,
    learning_rate=0.1,
    num_leaves=7,
    verbose=-1,
    random_state=42,
)

N_CAMPAIGNS = 4
N_DAYS = 200
RNG = np.random.default_rng(42)


def _make_feature_store(
    n_campaigns: int = N_CAMPAIGNS,
    n_days: int = N_DAYS,
) -> pd.DataFrame:
    """Small synthetic feature store that mimics the real schema."""
    rows = []
    start = pd.Timestamp("2024-01-01")
    dates = pd.date_range(start=start, periods=n_days, freq="D")

    for cid in range(n_campaigns):
        for i, date in enumerate(dates):
            rev = float(cid * 50 + i + RNG.normal(0, 5))
            rev = max(rev, 0.0)
            rows.append(
                {
                    "campaign_id":      f"camp_{cid}",
                    "campaign_name":    f"Campaign {cid}",
                    "platform":         "google",
                    "date":             date,
                    "revenue_attributed": rev,
                    "spend":            float(cid * 10 + i * 0.5),
                    "clicks":           float(cid * 100 + i * 5),
                    "impressions":      float(cid * 1000 + i * 50),
                    "attribution_mature": i < (n_days - 14),
                    # Numeric lag/rolling features
                    "revenue_lag_1":    max(rev - 10, 0.0),
                    "revenue_lag_7":    max(rev - 5, 0.0),
                    "spend_lag_1":      float(cid * 10 + (i - 1) * 0.5) if i > 0 else 0.0,
                    "revenue_roll_mean_7":  max(rev - 8, 0.0),
                    "revenue_roll_mean_14": max(rev - 6, 0.0),
                    # Calendar features
                    "day_of_week":      int(date.dayofweek),
                    "day_of_month":     int(date.day),
                    "week_of_year":     int(date.isocalendar()[1]),
                    "month":            int(date.month),
                    "quarter":          int(date.quarter),
                    "year":             int(date.year),
                    "is_weekend":       date.dayofweek >= 5,
                    # Taxonomy
                    "strategy_key":     f"strategy_{cid}",
                    "funnel_stage":     "lower",
                    "audience_strategy": "retargeting",
                    "format":           "search",
                    "ad_product_type":  "pmax",
                    "is_brand":         bool(cid % 2 == 0),
                    "is_non_brand":     bool(cid % 2 == 1),
                    "is_upper_funnel":  False,
                    "cross_engine_pair_flag": False,
                    # Holiday
                    "is_holiday":       False,
                    "is_bfcm":          False,
                    "is_cyber_week":    False,
                    "is_holiday_season": False,
                    "days_to_black_friday":    30,
                    "days_since_black_friday": 60,
                    "days_to_thanksgiving":    20,
                    "days_since_thanksgiving": 70,
                    "holiday_intensity_score": 0.0,
                    # Budget
                    "budget_utilization": RNG.uniform(0.5, 0.95),
                    "budget_headroom":    RNG.uniform(0, 100),
                    # ROAS lags
                    "roas_lag_7":  rev / max(float(cid * 10 + 5), 1.0),
                    "roas_lag_14": rev / max(float(cid * 10 + 5), 1.0),
                    "roas_lag_28": rev / max(float(cid * 10 + 5), 1.0),
                    # Misc excluded
                    "holiday_name":    None,
                }
            )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["channel_format"] = pd.Categorical(
        ["search"] * len(df), categories=["search", "display", "video"]
    )
    return df.reset_index(drop=True)


@pytest.fixture(scope="module")
def feature_store() -> pd.DataFrame:
    return _make_feature_store()


@pytest.fixture(scope="module")
def training_result(feature_store, tmp_path_factory) -> TrainingResult:
    model_dir = tmp_path_factory.mktemp("models")
    report_path = model_dir / "eval_report.json"
    return train(
        fs=feature_store,
        model_dir=model_dir,
        report_path=report_path,
        config=FAST_CONFIG,
        validator_kwargs={"min_train_days": 60, "max_folds": 2, "val_window_days": 30},
    )


# ---------------------------------------------------------------------------
# TestMetricFunctions
# ---------------------------------------------------------------------------

class TestMetricFunctions:
    """Test each public metric function with known inputs."""

    def test_mae_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_mae_known(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 31.0])
        expected = (2.0 + 2.0 + 1.0) / 3.0
        assert mae(y_true, y_pred) == pytest.approx(expected)

    def test_rmse_zero(self):
        y = np.array([5.0, 10.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_rmse_known(self):
        y_true = np.array([0.0, 10.0])
        y_pred = np.array([0.0, 20.0])
        assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(50.0))

    def test_mape_perfect(self):
        y = np.array([100.0, 200.0])
        assert mape(y, y, epsilon=0.0) == pytest.approx(0.0)

    def test_mape_epsilon_avoids_division_by_zero(self):
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([10.0, 110.0])
        result = mape(y_true, y_pred, epsilon=1.0)
        assert np.isfinite(result)
        assert result >= 0.0

    def test_smape_symmetric(self):
        y_true = np.array([100.0])
        y_pred = np.array([200.0])
        result_1 = smape(y_true, y_pred)
        result_2 = smape(y_pred, y_true)
        assert result_1 == pytest.approx(result_2, rel=1e-5)

    def test_smape_perfect(self):
        y = np.array([50.0, 150.0])
        assert smape(y, y, epsilon=0.0) == pytest.approx(0.0)

    def test_pinball_p50_is_half_mae(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([5.0, 25.0, 40.0])
        pb = pinball_loss(y_true, y_pred, alpha=0.5)
        expected = mae(y_true, y_pred) / 2.0
        assert pb == pytest.approx(expected)

    def test_pinball_p10_penalises_overestimate(self):
        """For P10: over-predicting is penalised at (1-0.1)=0.9."""
        y_true = np.array([10.0])
        y_pred = np.array([20.0])  # over-prediction by 10
        result = pinball_loss(y_true, y_pred, alpha=0.1)
        assert result == pytest.approx(0.9 * 10.0)

    def test_pinball_p90_penalises_underestimate(self):
        """For P90: under-predicting is penalised at 0.9."""
        y_true = np.array([20.0])
        y_pred = np.array([10.0])  # under-prediction by 10
        result = pinball_loss(y_true, y_pred, alpha=0.9)
        assert result == pytest.approx(0.9 * 10.0)

    def test_coverage_all_inside(self):
        y = np.array([5.0, 10.0, 15.0])
        assert coverage(y, np.zeros(3), np.full(3, 100.0)) == pytest.approx(1.0)

    def test_coverage_none_inside(self):
        y = np.array([5.0, 10.0])
        assert coverage(y, np.full(2, 50.0), np.full(2, 100.0)) == pytest.approx(0.0)

    def test_coverage_half_inside(self):
        y = np.array([5.0, 150.0])
        result = coverage(y, np.zeros(2), np.full(2, 100.0))
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# TestGetFeatureColumns
# ---------------------------------------------------------------------------

class TestGetFeatureColumns:
    def test_target_excluded(self, feature_store):
        cols = get_feature_columns(feature_store)
        assert "revenue_attributed" not in cols

    def test_date_excluded(self, feature_store):
        cols = get_feature_columns(feature_store)
        assert "date" not in cols

    def test_campaign_id_excluded(self, feature_store):
        cols = get_feature_columns(feature_store)
        assert "campaign_id" not in cols

    def test_platform_excluded(self, feature_store):
        cols = get_feature_columns(feature_store)
        assert "platform" not in cols

    def test_attribution_mature_excluded(self, feature_store):
        cols = get_feature_columns(feature_store)
        assert "attribution_mature" not in cols

    def test_holiday_name_excluded(self, feature_store):
        cols = get_feature_columns(feature_store)
        assert "holiday_name" not in cols

    def test_numeric_features_included(self, feature_store):
        cols = get_feature_columns(feature_store)
        for c in ["revenue_lag_1", "spend_lag_1", "clicks", "day_of_week"]:
            assert c in cols, f"{c} should be in feature columns"

    def test_returns_list(self, feature_store):
        assert isinstance(get_feature_columns(feature_store), list)

    def test_column_order_preserved(self, feature_store):
        cols = get_feature_columns(feature_store)
        expected = [c for c in feature_store.columns if c not in {
            "revenue_attributed", "date", "campaign_id", "campaign_name",
            "platform", "attribution_mature", "holiday_name"
        }]
        assert cols == expected


# ---------------------------------------------------------------------------
# TestTrainPipeline
# ---------------------------------------------------------------------------

class TestTrainPipeline:
    def test_returns_training_result(self, training_result):
        assert isinstance(training_result, TrainingResult)

    def test_fold_count(self, training_result):
        assert training_result.n_folds == 2

    def test_fold_metrics_list_length(self, training_result):
        assert len(training_result.fold_metrics) == 2

    def test_fold_metrics_type(self, training_result):
        for fm in training_result.fold_metrics:
            assert isinstance(fm, FoldMetrics)

    def test_model_is_fitted(self, training_result):
        assert training_result.model._is_fitted

    def test_feature_importance_is_dataframe(self, training_result):
        assert isinstance(training_result.feature_importance, pd.DataFrame)

    def test_n_features_positive(self, training_result):
        assert training_result.n_features > 0

    def test_feature_columns_list(self, training_result):
        assert isinstance(training_result.feature_columns, list)
        assert len(training_result.feature_columns) == training_result.n_features

    def test_n_total_train_rows_positive(self, training_result):
        assert training_result.n_total_train_rows > 0

    def test_aggregated_metrics_keys(self, training_result):
        expected_keys = {
            "mean_mae_p50", "std_mae_p50",
            "mean_rmse_p50", "std_rmse_p50",
            "mean_mape_p50", "std_mape_p50",
            "mean_smape_p50", "std_smape_p50",
            "mean_pinball_p10", "std_pinball_p10",
            "mean_pinball_p50", "std_pinball_p50",
            "mean_pinball_p90", "std_pinball_p90",
            "mean_coverage_80", "std_coverage_80",
        }
        assert expected_keys.issubset(set(training_result.aggregated_metrics.keys()))

    def test_aggregated_metrics_finite(self, training_result):
        for k, v in training_result.aggregated_metrics.items():
            assert np.isfinite(v), f"Metric {k}={v} is not finite"

    def test_coverage_in_valid_range(self, training_result):
        for fm in training_result.fold_metrics:
            assert 0.0 <= fm.coverage_80 <= 1.0

    def test_pinball_losses_positive(self, training_result):
        for fm in training_result.fold_metrics:
            assert fm.pinball_p10 >= 0.0
            assert fm.pinball_p50 >= 0.0
            assert fm.pinball_p90 >= 0.0

    def test_mae_rmse_positive(self, training_result):
        for fm in training_result.fold_metrics:
            assert fm.mae_p50 >= 0.0
            assert fm.rmse_p50 >= 0.0

    def test_fold_metrics_have_dates(self, training_result):
        for fm in training_result.fold_metrics:
            assert fm.train_start != ""
            assert fm.val_start != ""


# ---------------------------------------------------------------------------
# TestModelPersistence
# ---------------------------------------------------------------------------

class TestModelPersistence:
    def test_model_files_written(self, feature_store, tmp_path):
        model_dir = tmp_path / "saved_model"
        train(
            fs=feature_store,
            model_dir=model_dir,
            report_path=tmp_path / "report.json",
            config=FAST_CONFIG,
            validator_kwargs={"min_train_days": 60, "max_folds": 1, "val_window_days": 30},
        )
        assert (model_dir / "p10.pkl").exists()
        assert (model_dir / "p50.pkl").exists()
        assert (model_dir / "p90.pkl").exists()
        assert (model_dir / "model_meta.pkl").exists()

    def test_saved_model_predicts(self, feature_store, tmp_path):
        from src.models.lgbm_quantile import RevenueQuantileModel

        model_dir = tmp_path / "model2"
        result = train(
            fs=feature_store,
            model_dir=model_dir,
            report_path=tmp_path / "r.json",
            config=FAST_CONFIG,
            validator_kwargs={"min_train_days": 60, "max_folds": 1, "val_window_days": 30},
        )
        loaded = RevenueQuantileModel.load(model_dir)
        X = feature_store[result.feature_columns]
        preds = loaded.predict(X)
        assert len(preds) == len(feature_store)
        assert (preds["p10"] <= preds["p50"]).all()
        assert (preds["p50"] <= preds["p90"]).all()


# ---------------------------------------------------------------------------
# TestEvaluationReport
# ---------------------------------------------------------------------------

class TestEvaluationReport:
    def test_report_file_written(self, feature_store, tmp_path):
        report_path = tmp_path / "report.json"
        train(
            fs=feature_store,
            model_dir=tmp_path / "m",
            report_path=report_path,
            config=FAST_CONFIG,
            validator_kwargs={"min_train_days": 60, "max_folds": 1, "val_window_days": 30},
        )
        assert report_path.exists()

    def test_report_is_valid_json(self, feature_store, tmp_path):
        report_path = tmp_path / "report2.json"
        train(
            fs=feature_store,
            model_dir=tmp_path / "m2",
            report_path=report_path,
            config=FAST_CONFIG,
            validator_kwargs={"min_train_days": 60, "max_folds": 1, "val_window_days": 30},
        )
        with open(report_path) as fh:
            data = json.load(fh)
        assert isinstance(data, dict)

    def test_report_keys(self, feature_store, tmp_path):
        report_path = tmp_path / "report3.json"
        train(
            fs=feature_store,
            model_dir=tmp_path / "m3",
            report_path=report_path,
            config=FAST_CONFIG,
            validator_kwargs={"min_train_days": 60, "max_folds": 1, "val_window_days": 30},
        )
        with open(report_path) as fh:
            data = json.load(fh)
        expected = {"n_folds", "n_features", "feature_columns",
                    "aggregated_metrics", "fold_metrics"}
        assert expected.issubset(data.keys())

    def test_report_fold_metrics_serialisable(self, feature_store, tmp_path):
        report_path = tmp_path / "report4.json"
        train(
            fs=feature_store,
            model_dir=tmp_path / "m4",
            report_path=report_path,
            config=FAST_CONFIG,
            validator_kwargs={"min_train_days": 60, "max_folds": 1, "val_window_days": 30},
        )
        with open(report_path) as fh:
            data = json.load(fh)
        assert isinstance(data["fold_metrics"], list)
        assert len(data["fold_metrics"]) >= 1
        fm = data["fold_metrics"][0]
        assert "mae_p50" in fm
        assert "coverage_80" in fm


# ---------------------------------------------------------------------------
# TestTrainErrorHandling
# ---------------------------------------------------------------------------

class TestTrainErrorHandling:
    def test_missing_feature_store_path_raises(self, tmp_path):
        with pytest.raises(TrainerError, match="not found"):
            train(
                feature_store_path=tmp_path / "nonexistent.parquet",
                model_dir=tmp_path / "m",
                report_path=tmp_path / "r.json",
                config=FAST_CONFIG,
            )

    def test_unsupported_format_raises(self, feature_store, tmp_path):
        bad_path = tmp_path / "data.xlsx"
        bad_path.write_text("fake")
        with pytest.raises(TrainerError, match="Unsupported"):
            train(
                feature_store_path=bad_path,
                model_dir=tmp_path / "m",
                report_path=tmp_path / "r.json",
                config=FAST_CONFIG,
            )

    def test_parquet_loading(self, feature_store, tmp_path):
        parquet_path = tmp_path / "fs.parquet"
        feature_store.to_parquet(parquet_path)
        result = train(
            feature_store_path=parquet_path,
            model_dir=tmp_path / "m_pq",
            report_path=tmp_path / "r_pq.json",
            config=FAST_CONFIG,
            validator_kwargs={"min_train_days": 60, "max_folds": 1, "val_window_days": 30},
        )
        assert isinstance(result, TrainingResult)

    def test_csv_loading(self, feature_store, tmp_path):
        csv_path = tmp_path / "fs.csv"
        feature_store.to_csv(csv_path, index=False)
        result = train(
            feature_store_path=csv_path,
            model_dir=tmp_path / "m_csv",
            report_path=tmp_path / "r_csv.json",
            config=FAST_CONFIG,
            validator_kwargs={"min_train_days": 60, "max_folds": 1, "val_window_days": 30},
        )
        assert isinstance(result, TrainingResult)
