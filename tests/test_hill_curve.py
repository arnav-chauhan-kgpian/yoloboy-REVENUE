"""
tests/test_hill_curve.py
=========================
Unit tests for src/simulation/hill_curve.py.

Tests mathematical properties, fitting, saturation, marginal ROAS,
and fallback behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.simulation.hill_curve import HillCurve, HillCurveParams, HillFitDiagnostics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_params(v_max=1000.0, K=500.0, n=1.0) -> HillCurveParams:
    return HillCurveParams(v_max=v_max, K=K, n=n)


def _make_diag(reliable=True, r2=0.8) -> HillFitDiagnostics:
    return HillFitDiagnostics(n_points=50, r_squared=r2, is_reliable=reliable, fit_reason="ok")


def _make_curve(v_max=1000.0, K=500.0, n=1.0) -> HillCurve:
    return HillCurve(_make_params(v_max, K, n), _make_diag())


def _make_synthetic_data(
    v_max=1000.0, K=500.0, n=1.5, noise_std=20.0, n_points=60
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic Hill-curve data with noise."""
    rng = np.random.default_rng(42)
    spend = np.linspace(10, 2000, n_points)
    ratio = (spend / K) ** n
    revenue = v_max * ratio / (1 + ratio) + rng.normal(0, noise_std, n_points)
    revenue = np.maximum(revenue, 0.0)
    return spend, revenue


# ---------------------------------------------------------------------------
# TestEvaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_zero_spend_returns_zero(self):
        c = _make_curve()
        assert c.evaluate(0.0) == pytest.approx(0.0)

    def test_at_K_returns_half_vmax(self):
        v_max, K = 1000.0, 500.0
        c = _make_curve(v_max=v_max, K=K, n=1.0)
        assert c.evaluate(K) == pytest.approx(v_max / 2, rel=1e-5)

    def test_large_spend_approaches_vmax(self):
        c = _make_curve(v_max=1000.0, K=500.0, n=2.0)
        assert c.evaluate(1e6) > 999.0

    def test_monotonically_increasing(self):
        c = _make_curve()
        spends = np.linspace(0, 5000, 100)
        values = c.evaluate(spends)
        diffs = np.diff(values)
        assert (diffs >= 0).all(), "Hill curve must be monotonically non-decreasing"

    def test_scalar_input_returns_scalar(self):
        c = _make_curve()
        result = c.evaluate(300.0)
        assert isinstance(result, float)

    def test_array_input_returns_array(self):
        c = _make_curve()
        arr = np.array([0.0, 100.0, 500.0, 1000.0])
        result = c.evaluate(arr)
        assert result.shape == arr.shape

    def test_negative_spend_returns_zero(self):
        c = _make_curve()
        assert c.evaluate(-100.0) == pytest.approx(0.0)

    def test_hill_coefficient_effect(self):
        """Higher n → more sigmoidal (lower value before K, higher after K)."""
        c1 = _make_curve(n=1.0)
        c2 = _make_curve(n=3.0)
        # Before K: c2 < c1 (sigmoid starts slow)
        assert c2.evaluate(200.0) < c1.evaluate(200.0)


# ---------------------------------------------------------------------------
# TestMarginalROAS
# ---------------------------------------------------------------------------

class TestMarginalROAS:
    def test_positive_marginal_roas(self):
        c = _make_curve()
        assert c.marginal_roas(100.0) > 0
        assert c.marginal_roas(500.0) > 0
        assert c.marginal_roas(1000.0) > 0

    def test_diminishing_returns(self):
        """Marginal ROAS must be decreasing."""
        c = _make_curve()
        spends = [100.0, 300.0, 500.0, 1000.0, 2000.0]
        mroas = [c.marginal_roas(s) for s in spends]
        for i in range(1, len(mroas)):
            assert mroas[i] < mroas[i - 1], (
                f"Marginal ROAS not decreasing: {mroas[i-1]:.4f} → {mroas[i]:.4f}"
            )

    def test_zero_spend_marginal_roas(self):
        """At spend=0: marginal ROAS = v_max * n / K."""
        c = _make_curve(v_max=1000.0, K=500.0, n=2.0)
        expected = 1000.0 * 2.0 / 500.0
        assert c.marginal_roas(0.0) == pytest.approx(expected, rel=1e-5)

    def test_marginal_roas_always_positive(self):
        c = _make_curve()
        for s in [0, 1, 10, 100, 1000, 10000]:
            assert c.marginal_roas(float(s)) >= 0


# ---------------------------------------------------------------------------
# TestSaturationScore
# ---------------------------------------------------------------------------

class TestSaturationScore:
    def test_zero_spend_zero_saturation(self):
        c = _make_curve()
        assert c.saturation_score(0.0) == pytest.approx(0.0)

    def test_at_K_half_saturation(self):
        c = _make_curve(n=1.0)
        assert c.saturation_score(500.0) == pytest.approx(0.5, rel=1e-4)

    def test_saturation_bounded_01(self):
        c = _make_curve()
        for spend in [0, 1, 100, 1e6]:
            s = c.saturation_score(float(spend))
            assert 0.0 <= s <= 1.0

    def test_saturation_monotone(self):
        c = _make_curve()
        sats = [c.saturation_score(s) for s in [0, 100, 500, 1000, 5000]]
        for i in range(1, len(sats)):
            assert sats[i] >= sats[i - 1]


# ---------------------------------------------------------------------------
# TestSpendForSaturation
# ---------------------------------------------------------------------------

class TestSpendForSaturation:
    def test_spend_for_50pct(self):
        c = _make_curve(v_max=1000.0, K=500.0, n=1.0)
        s = c.spend_for_saturation(0.5)
        assert s == pytest.approx(500.0, rel=1e-4)

    def test_spend_for_80pct(self):
        c = _make_curve(v_max=1000.0, K=500.0, n=1.0)
        s = c.spend_for_saturation(0.8)
        # For n=1: x = K * sat/(1-sat) = 500 * 0.8/0.2 = 2000
        assert s == pytest.approx(2000.0, rel=1e-4)

    def test_round_trip(self):
        c = _make_curve()
        for target in [0.2, 0.5, 0.75, 0.9]:
            spend = c.spend_for_saturation(target)
            actual_sat = c.saturation_score(spend)
            assert actual_sat == pytest.approx(target, rel=1e-3)


# ---------------------------------------------------------------------------
# TestFitting
# ---------------------------------------------------------------------------

class TestFitting:
    def test_fit_on_known_data(self):
        spend, revenue = _make_synthetic_data(v_max=1000.0, K=500.0, n=1.5, noise_std=5.0)
        c = HillCurve.fit(spend, revenue)
        # Fitted curve should predict close to v_max for very high spend
        assert c.evaluate(10000.0) > 700.0

    def test_fit_r_squared_positive(self):
        spend, revenue = _make_synthetic_data(noise_std=5.0)
        c = HillCurve.fit(spend, revenue)
        assert c.r_squared >= 0.0

    def test_fit_reliable_on_clean_data(self):
        spend, revenue = _make_synthetic_data(noise_std=2.0)
        c = HillCurve.fit(spend, revenue)
        # Clean data → curve should be reliable
        assert c.is_reliable

    def test_fit_insufficient_data_falls_back(self):
        spend   = np.array([100.0, 200.0, 300.0])
        revenue = np.array([50.0, 80.0, 100.0])
        c = HillCurve.fit(spend, revenue)
        # Not reliable because too few points
        assert not c.is_reliable

    def test_fit_zero_variance_spend_falls_back(self):
        spend   = np.full(30, 500.0)
        revenue = np.random.default_rng(0).uniform(400, 600, 30)
        c = HillCurve.fit(spend, revenue)
        assert not c.is_reliable

    def test_fit_all_zero_spend_does_not_crash(self):
        spend   = np.zeros(20)
        revenue = np.zeros(20)
        c = HillCurve.fit(spend, revenue)
        assert isinstance(c, HillCurve)

    def test_fit_preserves_monotonicity(self):
        spend, revenue = _make_synthetic_data()
        c = HillCurve.fit(spend, revenue)
        test_points = np.linspace(1, 3000, 50)
        values = c.evaluate(test_points)
        diffs = np.diff(values)
        assert (diffs >= -0.01).all(), "Fitted curve must be monotone"

    def test_fit_evaluate_range(self):
        spend, revenue = _make_synthetic_data(v_max=1000.0)
        c = HillCurve.fit(spend, revenue)
        for s in spend:
            val = c.evaluate(float(s))
            assert val >= 0.0

    def test_linear_fallback_has_roas_slope(self):
        """Linear fallback: evaluate(spend) ≈ ROAS * spend."""
        spend   = np.full(5, 500.0)   # no variance → linear fallback
        revenue = np.full(5, 2000.0)
        c = HillCurve.fit(spend, revenue)
        expected_roas = 2000.0 / 500.0
        # For linear model at x≪K: f(x) ≈ (v_max/K)*x = roas*x
        ratio = c.evaluate(100.0) / 100.0
        assert ratio == pytest.approx(expected_roas, rel=0.1)


# ---------------------------------------------------------------------------
# TestIsReliable
# ---------------------------------------------------------------------------

class TestIsReliable:
    def test_is_reliable_property(self):
        c = _make_curve()
        assert c.is_reliable is True

    def test_not_reliable_when_flagged(self):
        diag = _make_diag(reliable=False, r2=0.1)
        c = HillCurve(_make_params(), diag)
        assert c.is_reliable is False

    def test_r_squared_property(self):
        diag = _make_diag(r2=0.73)
        c = HillCurve(_make_params(), diag)
        assert c.r_squared == pytest.approx(0.73)
