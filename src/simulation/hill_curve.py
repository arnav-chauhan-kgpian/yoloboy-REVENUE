"""
src/simulation/hill_curve.py
==============================
Hill saturation curve for campaign revenue response modeling.

    f(x) = v_max * (x/K)^n / (1 + (x/K)^n)

Parameters
----------
v_max : float
    Revenue ceiling — maximum achievable revenue at infinite spend.
K : float
    Half-saturation constant — spend level at which revenue = v_max / 2.
n : float
    Hill coefficient — controls steepness of the S-curve.
    n=1  → hyperbolic (concave from origin)
    n>1  → sigmoidal (inflection point before K)

Key properties
--------------
- f(0) = 0
- f(K) = v_max / 2
- f'(x) always positive, f''(x) eventually negative (diminishing returns)
- Marginal ROAS = f'(x) = v_max·n·(x/K)^n / [x · (1+(x/K)^n)^2]
- Saturation score = f(x)/v_max ∈ [0, 1]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_DATA_POINTS: Final[int] = 8
MIN_SPEND_STD_FRACTION: Final[float] = 0.05   # spend std / mean must exceed this
RELIABLE_R2_THRESHOLD: Final[float] = 0.25
MAX_N_ESTIMATOR: Final[int] = 10000           # curve_fit max function evaluations


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HillCurveParams:
    v_max: float   # revenue ceiling
    K: float       # half-saturation spend
    n: float       # Hill coefficient


@dataclass
class HillFitDiagnostics:
    n_points: int
    r_squared: float
    is_reliable: bool
    fit_reason: str


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class HillCurveError(Exception):
    pass


# ---------------------------------------------------------------------------
# Core function (numerically stable)
# ---------------------------------------------------------------------------

def _hill(x: np.ndarray, v_max: float, K: float, n: float) -> np.ndarray:
    """Evaluate Hill function, safe against zero-spend and overflow."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    pos = x > 0
    ratio = (x[pos] / K) ** n
    out[pos] = v_max * ratio / (1.0 + ratio)
    return out


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        return 1.0
    return float(np.clip(1.0 - ss_res / ss_tot, -1.0, 1.0))


# ---------------------------------------------------------------------------
# HillCurve class
# ---------------------------------------------------------------------------

class HillCurve:
    """Hill saturation model with fit diagnostics and analytical derivatives."""

    def __init__(
        self,
        params: HillCurveParams,
        diagnostics: HillFitDiagnostics,
    ) -> None:
        self.params = params
        self.diagnostics = diagnostics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, spend: float | np.ndarray) -> float | np.ndarray:
        """Revenue at spend level(s)."""
        scalar = np.isscalar(spend)
        arr = np.atleast_1d(np.asarray(spend, dtype=float))
        result = _hill(arr, self.params.v_max, self.params.K, self.params.n)
        return float(result[0]) if scalar else result

    def marginal_roas(self, spend: float) -> float:
        """d(revenue)/d(spend) at the given spend level.

        At spend=0: uses the limit (v_max * n / K).
        """
        if spend <= 0:
            return self.params.v_max * self.params.n / self.params.K
        p = self.params
        ratio = (spend / p.K) ** p.n
        return float(p.v_max * p.n * ratio / (spend * (1.0 + ratio) ** 2))

    def saturation_score(self, spend: float) -> float:
        """Current saturation: f(spend) / v_max ∈ [0, 1].

        0.0 = far from saturation (high marginal returns).
        1.0 = fully saturated (no marginal returns).
        """
        if spend <= 0 or self.params.v_max <= 0:
            return 0.0
        return float(self.evaluate(spend) / self.params.v_max)

    def spend_for_saturation(self, target: float = 0.80) -> float:
        """Spend level needed to reach *target* fraction of v_max.

        Derived by inverting the Hill equation:
            x = K · (s / (1 − s))^(1/n)
        where s = target saturation fraction.
        """
        target = float(np.clip(target, 1e-6, 1.0 - 1e-6))
        ratio = target / (1.0 - target)
        return float(self.params.K * (ratio ** (1.0 / self.params.n)))

    @property
    def is_reliable(self) -> bool:
        return self.diagnostics.is_reliable

    @property
    def r_squared(self) -> float:
        return self.diagnostics.r_squared

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        spend: np.ndarray,
        revenue: np.ndarray,
        max_v_max_factor: float = 5.0,
    ) -> "HillCurve":
        """Fit Hill curve via non-linear least squares (scipy curve_fit).

        Falls back to a linear-ROAS model when data is insufficient or
        the fit does not converge.

        Parameters
        ----------
        spend, revenue :
            Parallel arrays of daily spend and revenue values.
        max_v_max_factor :
            Upper bound on v_max expressed as a multiple of max(revenue).

        Returns
        -------
        HillCurve
            Always succeeds; check ``is_reliable`` for fit quality.
        """
        spend = np.asarray(spend, dtype=float).ravel()
        revenue = np.asarray(revenue, dtype=float).ravel()

        # Clean data
        valid = (spend >= 0) & (revenue >= 0) & np.isfinite(spend) & np.isfinite(revenue)
        spend, revenue = spend[valid], revenue[valid]

        # Remove zero-spend rows for curve fitting (kept for diagnostics only)
        mask = spend > 0
        s, r = spend[mask], revenue[mask]
        n_pts = len(s)

        if n_pts < MIN_DATA_POINTS:
            return cls._linear_fallback(spend, revenue, n_pts, f"too few nonzero points ({n_pts})")

        spend_mean = s.mean()
        if s.std() < spend_mean * MIN_SPEND_STD_FRACTION:
            return cls._linear_fallback(spend, revenue, n_pts, "spend variance too low")

        max_rev = max(r.max(), 1.0)
        max_sp  = s.max()
        med_sp  = float(np.median(s))

        p0     = [max_rev * 1.5, med_sp, 1.0]
        bounds = (
            [max_rev * 0.1,     max_sp * 0.001, 0.3],
            [max_rev * max_v_max_factor, max_sp * 200.0, 6.0],
        )

        try:
            popt, _ = curve_fit(
                _hill, s, r,
                p0=p0,
                bounds=bounds,
                maxfev=MAX_N_ESTIMATOR,
                method="trf",
            )
        except (RuntimeError, ValueError) as exc:
            logger.debug("Hill fit failed: %s", exc)
            return cls._linear_fallback(spend, revenue, n_pts, str(exc))

        v_max, K, n = float(popt[0]), float(popt[1]), float(popt[2])
        r2 = _r_squared(r, _hill(s, v_max, K, n))
        reliable = r2 >= RELIABLE_R2_THRESHOLD

        diag = HillFitDiagnostics(
            n_points=n_pts,
            r_squared=r2,
            is_reliable=reliable,
            fit_reason="ok" if reliable else f"low R² ({r2:.3f})",
        )
        return cls(HillCurveParams(v_max=v_max, K=K, n=n), diag)

    @classmethod
    def _linear_fallback(
        cls,
        spend: np.ndarray,
        revenue: np.ndarray,
        n_pts: int,
        reason: str,
    ) -> "HillCurve":
        """Linear ROAS model: f(x) ≈ ROAS * x, achieved by K → ∞."""
        if spend.sum() > 0:
            roas = float(revenue.sum() / spend.sum())
        else:
            roas = 1.0
        roas = max(roas, 0.0)

        large_K = max(float(spend.max()) * 1_000 if len(spend) else 1.0, 1e6)
        params = HillCurveParams(
            v_max=large_K * roas,
            K=large_K,
            n=1.0,
        )
        diag = HillFitDiagnostics(
            n_points=n_pts,
            r_squared=0.0,
            is_reliable=False,
            fit_reason=f"linear fallback: {reason}",
        )
        logger.debug("Linear fallback for Hill curve (reason: %s, ROAS=%.3f)", reason, roas)
        return cls(params, diag)
