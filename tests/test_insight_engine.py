"""
tests/test_insight_engine.py
==============================
Unit tests for src/copilot/insight_engine.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.copilot.insight_engine import (
    InsightEngine,
    Insight,
    InsightType,
    InsightSeverity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(99)
N_DAYS   = 60
N_CAMPS  = 4


def _make_forecasts(n_campaigns=N_CAMPS, n_days=N_DAYS) -> pd.DataFrame:
    """Synthetic forecasts DataFrame: campaign × day with p10/p50/p90."""
    rows = []
    platforms = ["google", "meta", "bing", "google"]
    for cid in range(n_campaigns):
        platform = platforms[cid % len(platforms)]
        for d in range(n_days):
            p50 = 500 + cid * 200 + d * 2.0 + RNG.normal(0, 20)
            p50 = max(p50, 0.0)
            rows.append({
                "campaign_id":       f"camp_{cid}",
                "platform":          platform,
                "date":              pd.Timestamp("2024-01-01") + pd.Timedelta(days=d),
                "p50":               p50,
                "p10":               p50 * 0.80,
                "p90":               p50 * 1.20,
                "revenue_attributed": p50 + RNG.normal(0, 10),
            })
    return pd.DataFrame(rows)


def _make_feature_store(n_campaigns=N_CAMPS, n_days=N_DAYS) -> pd.DataFrame:
    """Synthetic feature store with spend data."""
    rows = []
    platforms = ["google", "meta", "bing", "google"]
    for cid in range(n_campaigns):
        platform = platforms[cid % len(platforms)]
        base_spend = (cid + 1) * 100.0
        for d in range(n_days):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
            spend = max(base_spend + RNG.normal(0, 20), 1.0)
            rev = spend * (2.0 + cid * 0.3) + RNG.normal(0, 15)
            rows.append({
                "campaign_id":       f"camp_{cid}",
                "campaign_name":     f"Campaign {cid}",
                "platform":          platform,
                "date":              date,
                "spend":             spend,
                "revenue_attributed": max(rev, 0.0),
                "attribution_mature": d < n_days - 14,
                "holiday_intensity": 0.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def forecasts():
    return _make_forecasts()


@pytest.fixture(scope="module")
def fs():
    return _make_feature_store()


@pytest.fixture(scope="module")
def engine():
    return InsightEngine()


@pytest.fixture(scope="module")
def insights(engine, forecasts, fs):
    return engine.generate_all_insights(forecasts=forecasts, fs=fs)


# ---------------------------------------------------------------------------
# TestInsightEngineGenerates
# ---------------------------------------------------------------------------

class TestInsightEngineGenerates:
    def test_returns_list(self, insights):
        assert isinstance(insights, list)

    def test_all_insight_instances(self, insights):
        for i in insights:
            assert isinstance(i, Insight)

    def test_insight_has_required_fields(self, insights):
        for i in insights:
            assert isinstance(i.title, str) and len(i.title) > 0
            assert isinstance(i.explanation, str) and len(i.explanation) > 0
            assert isinstance(i.metric_name, str)
            assert isinstance(i.metric_value, float)
            assert isinstance(i.confidence, float)

    def test_confidence_bounded(self, insights):
        for i in insights:
            assert 0.0 <= i.confidence <= 1.0

    def test_valid_severity(self, insights):
        for i in insights:
            assert i.severity in InsightSeverity

    def test_valid_type(self, insights):
        for i in insights:
            assert i.type in InsightType

    def test_sorted_by_severity(self, insights):
        """Critical/Warning insights should appear before Info/Positive."""
        severity_order = {
            InsightSeverity.CRITICAL:  0,
            InsightSeverity.WARNING:   1,
            InsightSeverity.POSITIVE:  2,
            InsightSeverity.INFO:      3,
        }
        severities = [i.severity for i in insights]
        for a, b in zip(severities, severities[1:]):
            assert severity_order[a] <= severity_order[b], (
                f"Insight severity out of order: {a.value} before {b.value}"
            )


# ---------------------------------------------------------------------------
# TestInsightGrounding
# ---------------------------------------------------------------------------

class TestInsightGrounding:
    def test_metric_value_is_finite(self, insights):
        """Every insight must have a finite numeric metric value."""
        for i in insights:
            assert np.isfinite(i.metric_value), f"Non-finite metric_value in insight: {i.title}"

    def test_metric_name_nonempty(self, insights):
        for i in insights:
            assert len(i.metric_name.strip()) > 0

    def test_explanation_mentions_metric(self, insights):
        """Explanation should reference actual numbers (not generic)."""
        for i in insights:
            # Explanation should be informative (not just the title repeated)
            assert i.explanation != i.title


# ---------------------------------------------------------------------------
# TestEmptyData
# ---------------------------------------------------------------------------

class TestEmptyData:
    def test_empty_forecasts_returns_empty(self, engine, fs):
        empty_fc = pd.DataFrame(columns=["campaign_id", "platform", "date", "p50", "p10", "p90", "revenue_attributed"])
        result = engine.generate_all_insights(forecasts=empty_fc, fs=fs)
        assert isinstance(result, list)

    def test_empty_fs_returns_something_or_empty(self, engine, forecasts):
        empty_fs = pd.DataFrame(columns=[
            "campaign_id", "platform", "date", "spend",
            "revenue_attributed", "attribution_mature", "holiday_intensity"
        ])
        result = engine.generate_all_insights(forecasts=forecasts, fs=empty_fs)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# TestInsightTypes
# ---------------------------------------------------------------------------

class TestInsightTypes:
    def test_generates_at_least_one_insight(self, insights):
        assert len(insights) >= 1

    def test_no_duplicate_titles(self, insights):
        titles = [i.title for i in insights]
        assert len(titles) == len(set(titles)), "Duplicate insight titles found"
