"""
src/models/trainer.py
======================
Full training pipeline for the meridian revenue forecasting system.

Workflow
--------
1. Load or build the feature store.
2. Identify feature columns (all columns except identifiers and target).
3. Generate rolling-origin folds with :class:`~src.models.cross_validator.RollingOriginValidator`.
4. For each fold:
   a. Train :class:`~src.models.lgbm_quantile.RevenueQuantileModel` on mature
      training rows.
   b. Evaluate P10/P50/P90 predictions on mature validation rows.
   c. Record :class:`FoldMetrics`.
5. Retrain a final model on all attribution_mature=True rows (no held-out set).
6. Persist the final model and write an evaluation_report.json.
7. Return a :class:`TrainingResult`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from src.features.feature_store import build_feature_store
from src.models.cross_validator import RollingOriginValidator
from src.models.lgbm_quantile import QuantileConfig, RevenueQuantileModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Columns excluded from the feature matrix
# ---------------------------------------------------------------------------

EXCLUDE_FROM_FEATURES: Final[frozenset[str]] = frozenset(
    {
        "revenue_attributed",  # target
        "date",                # temporal identifier
        "campaign_id",         # campaign identifier
        "campaign_name",       # free-text, not encoded
        "platform",            # already represented by campaign-level categoricals
        "attribution_mature",  # data quality flag — not a signal
        "holiday_name",        # free-text, not encoded
    }
)


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------

def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float,
) -> float:
    """Asymmetric pinball (quantile) loss for a single quantile.

    L(y, ŷ; α) = α · max(y − ŷ, 0) + (1 − α) · max(ŷ − y, 0)
    """
    residual = y_true - y_pred
    loss = np.where(residual >= 0, alpha * residual, (alpha - 1.0) * residual)
    return float(np.mean(loss))


def coverage(
    y_true: np.ndarray,
    y_p10: np.ndarray,
    y_p90: np.ndarray,
) -> float:
    """Fraction of actuals falling within the [P10, P90] prediction interval.

    For a well-calibrated 80 % interval, this should approach 0.80.
    """
    inside = (y_true >= y_p10) & (y_true <= y_p90)
    return float(np.mean(inside))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1.0,
) -> float:
    """Mean Absolute Percentage Error with epsilon smoothing.

    The epsilon term prevents division by zero for zero-revenue rows
    (common in Bing campaigns).  Returns a percentage value (0–100).
    """
    return float(
        np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + epsilon)) * 100.0
    )


def smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1.0,
) -> float:
    """Symmetric Mean Absolute Percentage Error with epsilon smoothing.

    Returns a percentage value (0–100).
    """
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + epsilon
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FoldMetrics:
    """Evaluation metrics for a single cross-validation fold."""
    fold_id: int
    train_start: str       # ISO date
    train_end: str
    val_start: str
    val_end: str
    n_train: int
    n_val: int
    n_val_mature: int      # rows where attribution_mature=True in validation set
    mae_p50: float
    rmse_p50: float
    mape_p50: float
    smape_p50: float
    pinball_p10: float
    pinball_p50: float
    pinball_p90: float
    coverage_80: float     # fraction of actuals in [P10, P90]


@dataclass
class TrainingResult:
    """Complete output of the :func:`train` function."""
    fold_metrics: list[FoldMetrics]
    aggregated_metrics: dict[str, float]
    model: RevenueQuantileModel
    feature_importance: pd.DataFrame
    n_folds: int
    n_total_train_rows: int
    n_features: int
    feature_columns: list[str]


# ---------------------------------------------------------------------------
# Feature column selection
# ---------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the ordered list of feature columns for model training.

    Excludes all identifiers, the target, and metadata columns listed in
    :data:`EXCLUDE_FROM_FEATURES`.

    Parameters
    ----------
    df : pd.DataFrame
        The feature store DataFrame.

    Returns
    -------
    list[str]
        Column names to use as model inputs (preserves original order).
    """
    return [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TrainerError(Exception):
    pass


def _compute_fold_metrics(
    fold_id: int,
    fold,
    y_true: np.ndarray,
    preds: pd.DataFrame,
    n_val_mature: int,
    n_train: int,
) -> FoldMetrics:
    p50 = preds["p50"].values
    p10 = preds["p10"].values
    p90 = preds["p90"].values

    return FoldMetrics(
        fold_id=fold_id,
        train_start=str(fold.train_start.date()),
        train_end=str(fold.train_end.date()),
        val_start=str(fold.val_start.date()),
        val_end=str(fold.val_end.date()),
        n_train=n_train,
        n_val=len(y_true),
        n_val_mature=n_val_mature,
        mae_p50=mae(y_true, p50),
        rmse_p50=rmse(y_true, p50),
        mape_p50=mape(y_true, p50),
        smape_p50=smape(y_true, p50),
        pinball_p10=pinball_loss(y_true, p10, 0.1),
        pinball_p50=pinball_loss(y_true, p50, 0.5),
        pinball_p90=pinball_loss(y_true, p90, 0.9),
        coverage_80=coverage(y_true, p10, p90),
    )


def _aggregate_metrics(fold_metrics: list[FoldMetrics]) -> dict[str, float]:
    """Average each numeric metric across all folds."""
    keys = [
        "mae_p50", "rmse_p50", "mape_p50", "smape_p50",
        "pinball_p10", "pinball_p50", "pinball_p90", "coverage_80",
    ]
    result: dict[str, float] = {}
    for k in keys:
        values = [getattr(m, k) for m in fold_metrics]
        result[f"mean_{k}"] = float(np.mean(values))
        result[f"std_{k}"] = float(np.std(values))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train(
    fs: pd.DataFrame | None = None,
    feature_store_path: str | Path | None = None,
    data_dir: str | Path = "dataset",
    model_dir: str | Path = "models",
    report_path: str | Path = "evaluation_report.json",
    config: QuantileConfig | None = None,
    validator_kwargs: dict[str, Any] | None = None,
) -> TrainingResult:
    """Train the meridian revenue quantile forecasting model.

    Parameters
    ----------
    fs : pd.DataFrame | None
        Pre-built feature store DataFrame.  If provided, skips all loading.
    feature_store_path : str | Path | None
        Path to a parquet/CSV feature store file.  Used if *fs* is None.
    data_dir : str | Path
        Directory with raw CSVs.  Used if both *fs* and *feature_store_path*
        are None.
    model_dir : str | Path
        Directory to write final model artefacts.
    report_path : str | Path
        Path to write the JSON evaluation report.
    config : QuantileConfig | None
        LightGBM hyperparameters.  Defaults to :class:`QuantileConfig`.
    validator_kwargs : dict | None
        Extra keyword arguments forwarded to :class:`RollingOriginValidator`.
        Example: ``{"min_train_days": 90, "max_folds": 3}``.

    Returns
    -------
    TrainingResult
        Fold metrics, aggregated metrics, final model, and feature metadata.

    Raises
    ------
    TrainerError
        If the feature store cannot be loaded or validation fails.
    """
    config = config or QuantileConfig()
    validator_kwargs = validator_kwargs or {}

    # --- Step 1: Load feature store ---
    if fs is not None:
        logger.info("Using provided feature store (%d rows)", len(fs))
    elif feature_store_path is not None:
        fsp = Path(feature_store_path)
        if not fsp.exists():
            raise TrainerError(f"Feature store file not found: {fsp}")
        logger.info("Loading feature store from %s", fsp)
        if fsp.suffix == ".parquet":
            fs = pd.read_parquet(fsp)
        elif fsp.suffix == ".csv":
            fs = pd.read_csv(fsp, parse_dates=["date"])
        else:
            raise TrainerError(
                f"Unsupported feature store format: {fsp.suffix}. "
                "Use .parquet or .csv"
            )
    else:
        logger.info("Building feature store from raw data in %s", data_dir)
        fs = build_feature_store(data_dir=data_dir)

    # --- Step 2: Feature columns ---
    feature_cols = get_feature_columns(fs)
    n_features = len(feature_cols)
    logger.info("Feature matrix: %d features", n_features)

    # Attribution mature mask
    if "attribution_mature" in fs.columns:
        mature_mask = fs["attribution_mature"].fillna(False).astype(bool)
    else:
        mature_mask = pd.Series(True, index=fs.index)

    target_col = "revenue_attributed"
    if target_col not in fs.columns:
        raise TrainerError(
            f"Target column '{target_col}' not found in feature store."
        )

    # --- Step 3: Rolling-origin cross-validation ---
    validator = RollingOriginValidator(**validator_kwargs)
    folds = validator.split(fs)
    logger.info("Cross-validation: %d folds", len(folds))

    # --- Step 4: Per-fold training and evaluation ---
    fold_metrics_list: list[FoldMetrics] = []
    total_train_rows = 0

    for fold in folds:
        logger.info("Fold %d | train→%s | val %s→%s",
                    fold.fold_id,
                    fold.train_end.date(),
                    fold.val_start.date(),
                    fold.val_end.date())

        # Training data: index selection from the fold
        train_df = fs.loc[fold.train_indices]
        val_df   = fs.loc[fold.val_indices]

        # Filter validation to mature rows for metric computation
        if "attribution_mature" in val_df.columns:
            val_mature = val_df[val_df["attribution_mature"].fillna(False)]
        else:
            val_mature = val_df

        if len(val_mature) == 0:
            logger.warning("Fold %d: no mature validation rows — skipping.", fold.fold_id)
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val   = val_mature[feature_cols]
        y_val   = val_mature[target_col]

        # Fit fold model (with early stopping)
        fold_model = RevenueQuantileModel(config=config)
        fold_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        preds = fold_model.predict(X_val)
        metrics = _compute_fold_metrics(
            fold_id=fold.fold_id,
            fold=fold,
            y_true=y_val.values,
            preds=preds,
            n_val_mature=len(val_mature),
            n_train=len(train_df),
        )
        fold_metrics_list.append(metrics)
        total_train_rows += len(train_df)

        logger.info(
            "Fold %d done | MAE_P50=%.2f | MAPE_P50=%.1f%% | "
            "Coverage_80=%.2f | Pinball_P50=%.2f",
            fold.fold_id,
            metrics.mae_p50,
            metrics.mape_p50,
            metrics.coverage_80,
            metrics.pinball_p50,
        )

    if not fold_metrics_list:
        raise TrainerError(
            "All folds were skipped (no mature validation rows). "
            "Check the attribution_mature column and date coverage."
        )

    # --- Step 5: Aggregate metrics ---
    aggregated = _aggregate_metrics(fold_metrics_list)
    logger.info(
        "CV summary | mean_MAE_P50=%.2f | mean_MAPE_P50=%.1f%% | "
        "mean_Coverage_80=%.3f",
        aggregated["mean_mae_p50"],
        aggregated["mean_mape_p50"],
        aggregated["mean_coverage_80"],
    )

    # --- Step 6: Retrain final model on all mature data ---
    logger.info("Retraining final model on all attribution_mature=True rows ...")
    all_mature_df = fs[mature_mask]
    X_all = all_mature_df[feature_cols]
    y_all = all_mature_df[target_col]

    final_model = RevenueQuantileModel(config=config)
    final_model.fit(X_all, y_all)  # no validation set; train full n_estimators rounds

    importance_df = final_model.feature_importance()

    # --- Step 7: Persist model ---
    out_dir = Path(model_dir)
    final_model.save(out_dir)
    logger.info("Final model saved to %s", out_dir)

    # --- Step 8: Write evaluation report ---
    report: dict[str, Any] = {
        "n_folds": len(fold_metrics_list),
        "n_features": n_features,
        "feature_columns": feature_cols,
        "aggregated_metrics": aggregated,
        "fold_metrics": [asdict(m) for m in fold_metrics_list],
    }
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Evaluation report written to %s", rp)

    return TrainingResult(
        fold_metrics=fold_metrics_list,
        aggregated_metrics=aggregated,
        model=final_model,
        feature_importance=importance_df,
        n_folds=len(fold_metrics_list),
        n_total_train_rows=total_train_rows,
        n_features=n_features,
        feature_columns=feature_cols,
    )
