"""
src/models/lgbm_quantile.py
============================
LightGBM quantile regression model for campaign revenue forecasting.

Three independent models are trained — one per quantile (P10, P50, P90) — using
LightGBM's ``objective="quantile"`` with the corresponding alpha value.

Quantile ordering is enforced post-prediction: if P10 > P50 or P50 > P90 for
any row (quantile crossing), the predictions are clipped to restore monotonicity.
All predictions are clipped to ≥ 0 (revenue cannot be negative).

Persistence uses pickle for the complete model state, including hyperparameter
config, feature names, and category mappings.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError(
        "lightgbm is required: pip install lightgbm>=4.0"
    ) from exc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUANTILES: Final[tuple[float, ...]] = (0.1, 0.5, 0.9)

_Q_NAME: Final[dict[float, str]] = {0.1: "p10", 0.5: "p50", 0.9: "p90"}

# Columns that need categorical treatment (object dtype → pd.Categorical)
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "format",
    "audience_strategy",
    "funnel_stage",
    "ad_product_type",
    "strategy_key",
    "channel_format",  # already category dtype in the feature store
)

_META_FILENAME: Final[str] = "model_meta.pkl"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class QuantileConfig:
    """LightGBM hyperparameters tuned for ecommerce revenue quantile regression.

    Recommended settings for the production meridian dataset (25 k rows,
    136 campaigns, strong BFCM seasonality, Bing zero-inflation):

    - ``num_leaves=63``: expressive enough for interaction effects without
      overfitting on the smaller-campaign tail.
    - ``min_child_samples=20``: prevents leaf nodes driven by single-campaign
      outliers (e.g. BFCM spikes in brand campaigns).
    - ``lambda_l1=0.1, lambda_l2=0.1``: mild regularisation appropriate for
      many sparse lag features.
    - ``learning_rate=0.03`` with ``n_estimators=2000`` and early stopping
      is the standard slow-LR / many-round recipe for robust quantile fits.
    """
    n_estimators: int        = 2000
    learning_rate: float     = 0.03
    num_leaves: int          = 63
    min_child_samples: int   = 20
    feature_fraction: float  = 0.8
    bagging_fraction: float  = 0.8
    bagging_freq: int        = 5
    lambda_l1: float         = 0.1
    lambda_l2: float         = 0.1
    max_depth: int           = 6
    early_stopping_rounds: int = 100
    verbose: int             = -1
    n_jobs: int              = -1
    random_state: int        = 42


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class QuantileModelError(Exception):
    pass


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RevenueQuantileModel:
    """Three LightGBM quantile models (P10, P50, P90) with shared interface.

    Parameters
    ----------
    config : QuantileConfig | None
        Hyperparameter configuration.  Defaults to :class:`QuantileConfig`
        with the recommended production settings.
    """

    def __init__(self, config: QuantileConfig | None = None) -> None:
        self.config: QuantileConfig = config or QuantileConfig()
        self.models: dict[float, lgb.LGBMRegressor] = {}
        self.feature_names_: list[str] = []
        self.categorical_features_: list[str] = []
        self._category_maps: dict[str, pd.CategoricalDtype] = {}
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_categoricals(self, X: pd.DataFrame) -> list[str]:
        """Return columns that should be treated as categorical."""
        return [
            c for c in X.columns
            if c in set(CATEGORICAL_FEATURES) or X[c].dtype.name == "category"
        ]

    def _prepare_X(
        self,
        X: pd.DataFrame,
        fit_categories: bool = False,
    ) -> pd.DataFrame:
        """Align categorical dtypes with training schema.

        On first call (``fit_categories=True``):
          - Object columns in :attr:`categorical_features_` are cast to
            ``pd.Categorical`` and their dtype is recorded.
          - Already-categorical columns have their dtype recorded.

        On subsequent calls (``fit_categories=False``):
          - Categories from training are applied; unseen values become NaN
            (LightGBM treats NaN as missing and handles it natively).
        """
        X = X.copy()
        for col in self.categorical_features_:
            if col not in X.columns:
                continue
            if fit_categories:
                if X[col].dtype.name != "category":
                    X[col] = X[col].astype("category")
                self._category_maps[col] = X[col].dtype
            else:
                if col in self._category_maps:
                    known = self._category_maps[col].categories
                    # Replace unseen values with NaN before constructing the
                    # Categorical. Pandas 4 will raise if non-null values are
                    # not in the dtype's categories; NaN is handled natively
                    # by LightGBM as missing.
                    col_vals = X[col].where(X[col].isin(known), other=np.nan)
                    X[col] = pd.Categorical(col_vals, categories=known)
                elif X[col].dtype.name != "category":
                    X[col] = X[col].astype("category")
        return X

    def _make_lgbm(self, alpha: float) -> lgb.LGBMRegressor:
        cfg = self.config
        return lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            min_child_samples=cfg.min_child_samples,
            feature_fraction=cfg.feature_fraction,
            bagging_fraction=cfg.bagging_fraction,
            bagging_freq=cfg.bagging_freq,
            lambda_l1=cfg.lambda_l1,
            lambda_l2=cfg.lambda_l2,
            max_depth=cfg.max_depth,
            verbose=cfg.verbose,
            n_jobs=cfg.n_jobs,
            random_state=cfg.random_state,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        categorical_features: list[str] | None = None,
    ) -> "RevenueQuantileModel":
        """Fit P10, P50, and P90 quantile models.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix (must not contain the target column).
        y_train : pd.Series
            Training revenue targets (``revenue_attributed``).
        X_val : pd.DataFrame | None
            Held-out features for early stopping.  If None, early stopping
            is disabled and ``n_estimators`` rounds are always trained.
        y_val : pd.Series | None
            Held-out targets for early stopping.
        categorical_features : list[str] | None
            Explicit list of categorical column names.  If None, columns in
            :data:`CATEGORICAL_FEATURES` and/or already-categorical dtype
            are auto-detected.

        Returns
        -------
        RevenueQuantileModel
            Self (supports chaining).
        """
        if categorical_features is not None:
            self.categorical_features_ = [
                c for c in categorical_features if c in X_train.columns
            ]
        else:
            self.categorical_features_ = self._detect_categoricals(X_train)

        self.feature_names_ = list(X_train.columns)

        X_tr = self._prepare_X(X_train, fit_categories=True)
        use_val = X_val is not None and y_val is not None
        if use_val:
            X_v = self._prepare_X(X_val[self.feature_names_], fit_categories=False)

        for alpha in QUANTILES:
            name = _Q_NAME[alpha]
            logger.info("Training %s (alpha=%.1f) ...", name.upper(), alpha)

            model = self._make_lgbm(alpha)

            fit_kwargs: dict = {"categorical_feature": "auto"}
            if use_val:
                fit_kwargs["eval_set"] = [(X_v, y_val)]
                fit_kwargs["callbacks"] = [
                    lgb.early_stopping(
                        self.config.early_stopping_rounds,
                        verbose=False,
                    ),
                    lgb.log_evaluation(period=-1),
                ]

            model.fit(X_tr, y_train, **fit_kwargs)
            self.models[alpha] = model

            n_iter = getattr(model, "best_iteration_", None) or model.n_estimators
            logger.info(
                "%s fitted: best_iteration=%s  n_features=%d",
                name.upper(),
                n_iter,
                len(self.feature_names_),
            )

        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict P10, P50, P90 for *X*.

        Quantile ordering (P10 ≤ P50 ≤ P90) is guaranteed for every row.
        Crossing intervals are corrected by clipping:
          - P10 is clipped to ≤ P50
          - P90 is clipped to ≥ P50

        All predictions are clipped to ≥ 0 (revenue cannot be negative).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with the same schema as training.

        Returns
        -------
        pd.DataFrame
            Index matches *X*.  Columns: ``p10``, ``p50``, ``p90``.

        Raises
        ------
        QuantileModelError
            If the model has not been fitted yet.
        """
        if not self._is_fitted:
            raise QuantileModelError(
                "Model has not been fitted. Call fit() before predict()."
            )

        X_prep = self._prepare_X(
            X[self.feature_names_], fit_categories=False
        )

        p10 = self.models[0.1].predict(X_prep)
        p50 = self.models[0.5].predict(X_prep)
        p90 = self.models[0.9].predict(X_prep)

        # Detect crossings before fixing
        n_low = int(np.sum(p10 > p50))
        n_high = int(np.sum(p50 > p90))
        if n_low > 0 or n_high > 0:
            logger.warning(
                "Quantile crossing: %d rows P10>P50, %d rows P50>P90. "
                "Auto-correcting via clipping.",
                n_low, n_high,
            )
            p10 = np.minimum(p10, p50)
            p90 = np.maximum(p90, p50)

        # Revenue floor
        p10 = np.maximum(p10, 0.0)
        p50 = np.maximum(p50, 0.0)
        p90 = np.maximum(p90, 0.0)

        return pd.DataFrame(
            {"p10": p10, "p50": p50, "p90": p90},
            index=X.index,
        )

    def feature_importance(
        self,
        importance_type: str = "gain",
    ) -> pd.DataFrame:
        """Return feature importances averaged across all three quantile models.

        Parameters
        ----------
        importance_type : str
            ``"gain"`` (default) or ``"split"``.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``p10_imp``, ``p50_imp``, ``p90_imp``,
            ``mean_imp``.  Sorted by ``mean_imp`` descending.

        Raises
        ------
        QuantileModelError
            If the model has not been fitted.
        """
        if not self._is_fitted:
            raise QuantileModelError("Model not fitted.")

        records: dict[str, list] = {
            "feature":   self.feature_names_,
            "p10_imp":   list(
                self.models[0.1].booster_.feature_importance(importance_type)
            ),
            "p50_imp":   list(
                self.models[0.5].booster_.feature_importance(importance_type)
            ),
            "p90_imp":   list(
                self.models[0.9].booster_.feature_importance(importance_type)
            ),
        }
        df = pd.DataFrame(records)
        df["mean_imp"] = df[["p10_imp", "p50_imp", "p90_imp"]].mean(axis=1)
        return df.sort_values("mean_imp", ascending=False).reset_index(drop=True)

    def save(self, model_dir: str | Path) -> None:
        """Persist all three models and metadata to *model_dir*.

        Creates ``p10.pkl``, ``p50.pkl``, ``p90.pkl``, and ``model_meta.pkl``.

        Parameters
        ----------
        model_dir : str | Path
            Destination directory (created recursively if absent).

        Raises
        ------
        QuantileModelError
            If the model has not been fitted.
        """
        if not self._is_fitted:
            raise QuantileModelError(
                "Cannot save an unfitted model. Call fit() first."
            )

        out_dir = Path(model_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for alpha in QUANTILES:
            name = _Q_NAME[alpha]
            path = out_dir / f"{name}.pkl"
            with open(path, "wb") as fh:
                pickle.dump(self.models[alpha], fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Saved %s → %s", name.upper(), path)

        meta = {
            "config":                self.config,
            "feature_names_":        self.feature_names_,
            "categorical_features_": self.categorical_features_,
            "_category_maps":        self._category_maps,
        }
        meta_path = out_dir / _META_FILENAME
        with open(meta_path, "wb") as fh:
            pickle.dump(meta, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved metadata → %s", meta_path)

    @classmethod
    def load(cls, model_dir: str | Path) -> "RevenueQuantileModel":
        """Load a previously saved :class:`RevenueQuantileModel`.

        Parameters
        ----------
        model_dir : str | Path
            Directory produced by :meth:`save`.

        Returns
        -------
        RevenueQuantileModel
            Fully restored model ready for :meth:`predict`.

        Raises
        ------
        QuantileModelError
            If any required file is missing.
        """
        src = Path(model_dir)

        meta_path = src / _META_FILENAME
        if not meta_path.exists():
            raise QuantileModelError(
                f"Metadata file not found: {meta_path}. "
                "Did you save the model with RevenueQuantileModel.save()?"
            )

        with open(meta_path, "rb") as fh:
            meta = pickle.load(fh)

        instance = cls(config=meta["config"])
        instance.feature_names_        = meta["feature_names_"]
        instance.categorical_features_ = meta["categorical_features_"]
        instance._category_maps        = meta["_category_maps"]

        for alpha in QUANTILES:
            name = _Q_NAME[alpha]
            path = src / f"{name}.pkl"
            if not path.exists():
                raise QuantileModelError(
                    f"Model file not found: {path}"
                )
            with open(path, "rb") as fh:
                instance.models[alpha] = pickle.load(fh)

        instance._is_fitted = True
        logger.info("Loaded RevenueQuantileModel from %s", src)
        return instance
