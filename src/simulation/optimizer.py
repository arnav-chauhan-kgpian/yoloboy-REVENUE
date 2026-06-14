"""
src/simulation/optimizer.py
============================
Budget optimizer that maximises projected total daily revenue across campaigns
subject to a total spend constraint, using scipy SLSQP.

The objective is concave (sum of Hill functions), so SLSQP finds the global
optimum reliably.  At optimum, all marginal ROAS values are equal across
active campaigns (classic equal-marginal-return condition).

Constraints
-----------
- Budget equality: sum(spend_i) = total_budget
- Per-campaign bounds derived from max_reallocation_pct:
    lower_i = current_i * (1 - max_reallocation_pct)
    upper_i = current_i * (1 + max_reallocation_pct)
  Explicit OptimizationConstraint entries override these defaults.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from scipy.optimize import minimize

from src.simulation.response_curve import CampaignResponseCurve

logger = logging.getLogger(__name__)

_SLSQP_FTOL: Final[float] = 1e-10
_SLSQP_MAX_ITER: Final[int] = 2000


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OptimizationConstraint:
    """Hard per-campaign spend bounds (override max_reallocation_pct)."""
    campaign_id: str
    min_daily_spend: float = 0.0
    max_daily_spend: float = float("inf")


@dataclass
class CampaignAllocation:
    campaign_id: str
    campaign_name: str
    platform: str
    baseline_spend: float
    optimal_spend: float
    spend_delta: float
    spend_delta_pct: float
    baseline_revenue: float
    optimal_revenue: float
    revenue_delta: float
    revenue_delta_pct: float
    baseline_marginal_roas: float
    optimal_marginal_roas: float
    saturation_at_optimal: float


@dataclass
class OptimizationResult:
    optimal_allocations: list[CampaignAllocation]
    total_budget: float
    baseline_total_revenue: float
    optimal_total_revenue: float
    revenue_lift: float
    revenue_lift_pct: float
    baseline_roas: float
    optimal_roas: float
    risk_score: float
    converged: bool
    n_iterations: int
    solver_message: str
    budget_allocated: float      # sanity: should equal total_budget
    budget_error: float          # |budget_allocated - total_budget|


class OptimizerError(Exception):
    pass


# ---------------------------------------------------------------------------
# Objective and gradient (for SLSQP)
# ---------------------------------------------------------------------------

def _neg_revenue(x: np.ndarray, curves: list[CampaignResponseCurve]) -> float:
    return -sum(float(c.hill.evaluate(xi)) for c, xi in zip(curves, x))


def _neg_revenue_grad(
    x: np.ndarray, curves: list[CampaignResponseCurve]
) -> np.ndarray:
    return np.array([-c.hill.marginal_roas(float(xi)) for c, xi in zip(curves, x)])


# ---------------------------------------------------------------------------
# Risk score
# ---------------------------------------------------------------------------

def _risk_score(
    curves: list[CampaignResponseCurve],
    spend: np.ndarray,
    total_budget: float,
) -> float:
    if total_budget <= 0:
        return 0.0
    saturation_risk  = 0.0
    reliability_risk = 0.0
    concentration    = 0.0
    for c, s in zip(curves, spend):
        share = s / total_budget
        saturation_risk  += share * c.hill.saturation_score(float(s))
        reliability_risk += share * (0.0 if c.is_reliable else 1.0)
        concentration    += share ** 2
    return float(np.clip(0.5 * saturation_risk + 0.35 * reliability_risk + 0.15 * concentration, 0, 1))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize_budget(
    curves: dict[str, CampaignResponseCurve],
    total_budget: float,
    constraints: list[OptimizationConstraint] | None = None,
    max_reallocation_pct: float = 0.40,
) -> OptimizationResult:
    """Find spend allocation maximising projected total daily revenue.

    Parameters
    ----------
    curves :
        Campaign response curves.
    total_budget :
        Total daily spend budget (must equal sum of current spend, or desired level).
    constraints :
        Optional per-campaign hard bounds that override ``max_reallocation_pct``.
    max_reallocation_pct :
        Maximum fractional change per campaign from baseline spend.
        Default 0.40 → each campaign spend can change ±40%.

    Returns
    -------
    OptimizationResult
    """
    if not curves:
        raise OptimizerError("No response curves provided.")
    if total_budget <= 0:
        raise OptimizerError(f"total_budget must be positive, got {total_budget}")

    constraint_map: dict[str, OptimizationConstraint] = {
        c.campaign_id: c for c in (constraints or [])
    }

    ordered_curves = list(curves.values())
    campaign_ids   = [c.campaign_id for c in ordered_curves]

    # Build bounds
    bounds_list = []
    x0 = []
    for c in ordered_curves:
        baseline = c.avg_daily_spend
        if c.campaign_id in constraint_map:
            lo = constraint_map[c.campaign_id].min_daily_spend
            hi = constraint_map[c.campaign_id].max_daily_spend
        else:
            lo = baseline * (1.0 - max_reallocation_pct)
            hi = baseline * (1.0 + max_reallocation_pct)
        lo = max(lo, 0.0)
        bounds_list.append((lo, hi))
        x0.append(baseline)

    x0 = np.array(x0, dtype=float)

    # Scale x0 to match total_budget
    x0_sum = x0.sum()
    if x0_sum > 0:
        x0 = x0 * (total_budget / x0_sum)
    else:
        x0 = np.full(len(x0), total_budget / len(x0))

    # Clip to bounds after scaling
    for i, (lo, hi) in enumerate(bounds_list):
        x0[i] = np.clip(x0[i], lo, hi)

    scipy_constraints = [
        {
            "type": "eq",
            "fun": lambda x: x.sum() - total_budget,
            "jac": lambda x: np.ones_like(x),
        }
    ]

    result = minimize(
        _neg_revenue,
        x0,
        args=(ordered_curves,),
        jac=_neg_revenue_grad,
        method="SLSQP",
        bounds=bounds_list,
        constraints=scipy_constraints,
        options={"ftol": _SLSQP_FTOL, "maxiter": _SLSQP_MAX_ITER, "disp": False},
    )

    optimal_spend = result.x
    converged     = bool(result.success)
    if not converged:
        logger.warning("Budget optimizer did not converge: %s", result.message)

    # Build allocation records
    allocations: list[CampaignAllocation] = []
    total_baseline_rev = 0.0
    total_optimal_rev  = 0.0

    for c, baseline_s, optimal_s in zip(ordered_curves, x0, optimal_spend):
        optimal_s  = max(float(optimal_s), 0.0)
        baseline_r = c.avg_daily_revenue
        optimal_r  = float(c.hill.evaluate(optimal_s))
        total_baseline_rev += baseline_r
        total_optimal_rev  += optimal_r

        spend_delta     = optimal_s - c.avg_daily_spend
        spend_delta_pct = (spend_delta / c.avg_daily_spend * 100.0) if c.avg_daily_spend > 0 else 0.0
        rev_delta       = optimal_r - baseline_r
        rev_delta_pct   = (rev_delta / baseline_r * 100.0) if baseline_r > 0 else 0.0

        allocations.append(
            CampaignAllocation(
                campaign_id=c.campaign_id,
                campaign_name=c.campaign_name,
                platform=c.platform,
                baseline_spend=c.avg_daily_spend,
                optimal_spend=optimal_s,
                spend_delta=spend_delta,
                spend_delta_pct=spend_delta_pct,
                baseline_revenue=baseline_r,
                optimal_revenue=optimal_r,
                revenue_delta=rev_delta,
                revenue_delta_pct=rev_delta_pct,
                baseline_marginal_roas=c.hill.marginal_roas(c.avg_daily_spend),
                optimal_marginal_roas=c.hill.marginal_roas(optimal_s),
                saturation_at_optimal=c.hill.saturation_score(optimal_s),
            )
        )

    revenue_lift     = total_optimal_rev - total_baseline_rev
    revenue_lift_pct = (revenue_lift / total_baseline_rev * 100.0) if total_baseline_rev > 0 else 0.0
    budget_allocated = float(optimal_spend.sum())
    baseline_roas    = (total_baseline_rev / sum(c.avg_daily_spend for c in ordered_curves)) \
                        if sum(c.avg_daily_spend for c in ordered_curves) > 0 else 0.0
    optimal_roas     = (total_optimal_rev / budget_allocated) if budget_allocated > 0 else 0.0

    risk = _risk_score(ordered_curves, optimal_spend, total_budget)

    logger.info(
        "Optimizer: converged=%s | lift=%.2f%% | risk=%.3f | budget_error=%.4f",
        converged, revenue_lift_pct, risk, abs(budget_allocated - total_budget),
    )

    return OptimizationResult(
        optimal_allocations=allocations,
        total_budget=total_budget,
        baseline_total_revenue=total_baseline_rev,
        optimal_total_revenue=total_optimal_rev,
        revenue_lift=revenue_lift,
        revenue_lift_pct=revenue_lift_pct,
        baseline_roas=baseline_roas,
        optimal_roas=optimal_roas,
        risk_score=risk,
        converged=converged,
        n_iterations=int(result.nit),
        solver_message=result.message,
        budget_allocated=budget_allocated,
        budget_error=abs(budget_allocated - total_budget),
    )
