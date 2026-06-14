"""
tests/test_response_curve.py
==============================
Unit tests for src/simulation/response_curve.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.simulation.response_curve import (
    CampaignResponseCurve,
    ResponseCurveError,
    build_response_curves,
    compute_current_spend,
    platform_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_CAMPAIGNS = 4
N_DAYS = 90
RNG = np.random.default_rng(7)


def _make_feature_store(
    n_campaigns: int = N_CAMPAIGNS,
    n_days: int = N_DAYS,
) -> pd.DataFrame:
    """Synthetic feature store with spend-revenue relationships."""
    start = pd.Timestamp("2024-01-01")
    dates = pd.date_range(start=start, periods=n_days, freq="D")
    rows = []
    platforms = ["google", "meta", "bing", "google"]

    for cid in range(n_campaigns):
        roas = [3.0, 2.5, 1.5, 4.0][cid % 4]
        base_spend = [100, 200, 50, 150][cid % 4]
        platform = platforms[cid % len(platforms)]

        for i, date in enumerate(dates):
            # Add spend variance so Hill curve can be fit
            spend_noise = RNG.normal(0, base_spend * 0.3)
            spend = max(base_spend + spend_noise + i * 0.5, 5.0)
            # Revenue is a non-linear function of spend (approx Hill)
            v_max, K = roas * base_spend * 2, base_spend
            revenue = v_max * spend / (K + spend) + RNG.normal(0, 10)
            revenue = max(revenue, 0.0)
            is_mature = i < (n_days - 14)
            rows.append(
                {
                    "campaign_id":       f"camp_{cid}",
                    "campaign_name":     f"Campaign {cid}",
                    "platform":          platform,
                    "date":              date,
                    "spend":             spend,
                    "revenue_attributed": revenue,
                    "attribution_mature": is_mature,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fs() -> pd.DataFrame:
    return _make_feature_store()


@pytest.fixture(scope="module")
def curves(fs) -> dict:
    return build_response_curves(fs)


# ---------------------------------------------------------------------------
# TestBuildResponseCurves
# ---------------------------------------------------------------------------

class TestBuildResponseCurves:
    def test_returns_dict(self, curves):
        assert isinstance(curves, dict)

    def test_one_curve_per_campaign(self, fs, curves):
        n_campaigns = fs["campaign_id"].nunique()
        assert len(curves) == n_campaigns

    def test_keys_are_campaign_ids(self, fs, curves):
        expected = set(fs["campaign_id"].unique())
        assert set(curves.keys()) == expected

    def test_values_are_response_curves(self, curves):
        for c in curves.values():
            assert isinstance(c, CampaignResponseCurve)

    def test_missing_columns_raises(self):
        bad_df = pd.DataFrame({"campaign_id": ["a"], "spend": [100.0]})
        with pytest.raises(ResponseCurveError, match="missing"):
            build_response_curves(bad_df)

    def test_avg_daily_spend_positive(self, curves):
        for c in curves.values():
            assert c.avg_daily_spend >= 0.0

    def test_avg_daily_revenue_positive(self, curves):
        for c in curves.values():
            assert c.avg_daily_revenue >= 0.0

    def test_saturation_score_bounded(self, curves):
        for c in curves.values():
            assert 0.0 <= c.saturation_score <= 1.0

    def test_marginal_roas_positive(self, curves):
        for c in curves.values():
            assert c.current_marginal_roas >= 0.0

    def test_r_squared_in_range(self, curves):
        for c in curves.values():
            assert -1.0 <= c.r_squared <= 1.0

    def test_platform_preserved(self, fs, curves):
        for cid, c in curves.items():
            expected_platform = fs[fs["campaign_id"] == cid]["platform"].iloc[0]
            assert c.platform == expected_platform


# ---------------------------------------------------------------------------
# TestProject
# ---------------------------------------------------------------------------

class TestProject:
    def test_project_returns_tuple(self, curves):
        c = next(iter(curves.values()))
        result = c.project(c.avg_daily_spend)
        assert isinstance(result, tuple) and len(result) == 2

    def test_higher_spend_higher_revenue(self, curves):
        """More spend → more projected revenue (monotone Hill curve)."""
        for c in curves.values():
            if not c.is_reliable:
                continue
            rev_low,  _ = c.project(c.avg_daily_spend * 0.5)
            rev_high, _ = c.project(c.avg_daily_spend * 2.0)
            assert rev_high >= rev_low, f"Campaign {c.campaign_id}: non-monotone projection"

    def test_zero_spend_zero_revenue(self, curves):
        for c in curves.values():
            rev, _ = c.project(0.0)
            assert rev == pytest.approx(0.0, abs=1.0)

    def test_lift_pct_sign_consistency(self, curves):
        """Increase spend → positive lift pct."""
        for c in curves.values():
            _, lift = c.project(c.avg_daily_spend * 1.5)
            assert lift >= 0, f"Campaign {c.campaign_id}: positive spend change → negative lift"

    def test_baseline_lift_near_zero(self, curves):
        """At current spend, lift should be ≈ 0%."""
        for c in curves.values():
            _, lift = c.project(c.avg_daily_spend)
            # Should be very close to 0 (Hill(current) vs avg_daily_revenue ≈ Hill(current))
            assert abs(lift) < 50  # allow some drift from averaging


# ---------------------------------------------------------------------------
# TestCurrentSpend
# ---------------------------------------------------------------------------

class TestCurrentSpend:
    def test_current_spend_returns_dict(self, curves):
        cs = compute_current_spend(curves)
        assert isinstance(cs, dict)

    def test_keys_match_curves(self, curves):
        cs = compute_current_spend(curves)
        assert set(cs.keys()) == set(curves.keys())

    def test_all_values_positive(self, curves):
        cs = compute_current_spend(curves)
        for v in cs.values():
            assert v >= 0.0


# ---------------------------------------------------------------------------
# TestPlatformSummary
# ---------------------------------------------------------------------------

class TestPlatformSummary:
    def test_returns_dataframe(self, curves):
        df = platform_summary(curves)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self, curves):
        df = platform_summary(curves)
        assert len(df) == len(curves)

    def test_expected_columns(self, curves):
        df = platform_summary(curves)
        expected = {
            "campaign_id", "campaign_name", "platform",
            "avg_daily_spend", "avg_daily_revenue",
            "saturation_score", "is_reliable", "r_squared",
        }
        assert expected.issubset(df.columns)

    def test_saturation_scores_bounded(self, curves):
        df = platform_summary(curves)
        assert (df["saturation_score"] >= 0).all()
        assert (df["saturation_score"] <= 1).all()
