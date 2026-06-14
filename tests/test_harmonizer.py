"""
tests/test_harmonizer.py
========================
Pytest suite for src/data/harmonizer.py.

Coverage
--------
- Canonical schema: columns, order, dtypes, row count
- Per-platform transformations (Google, Meta, Bing)
- Google micros-to-currency conversion
- Meta revenue column rename and reach preservation
- Bing PascalCase normalisation
- channel_format mapping for Google and Bing (type-level)
- channel_format inference for Meta (name-based, all 16 campaign patterns)
- attribution_mature flag: per-platform cutoff, correct immature row counts
- Validation: negative spend, negative revenue, duplicate composite key
- Unit tests for _infer_meta_channel_format and _validate_canonical

All integration fixtures use scope="module" to load and harmonize once per
test session (< 15 s total data load).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pytest

from src.data.harmonizer import (
    ATTRIBUTION_MATURITY_DAYS,
    CANONICAL_CHANNEL_FORMATS,
    CANONICAL_COLUMNS,
    HarmonizeError,
    _infer_meta_channel_format,
    _maturity_cutoff,
    _normalize_bing,
    _normalize_google,
    _normalize_meta,
    _validate_canonical,
    harmonize,
    harmonize_from_dir,
)
from src.data.loader import RawDataset, load_raw_data

# ---------------------------------------------------------------------------
# Forensic constants (data contracts — fail here means source data changed)
# ---------------------------------------------------------------------------

GOOGLE_ROWS: Final[int] = 19_272
META_ROWS:   Final[int] =  3_417
BING_ROWS:   Final[int] =  2_873
TOTAL_ROWS:  Final[int] = 25_562
CANONICAL_COL_COUNT: Final[int] = 14

# Rows whose date falls in the trailing 14-day immature window per platform
GOOGLE_IMMATURE_ROWS: Final[int] = 333
META_IMMATURE_ROWS:   Final[int] =  66
BING_IMMATURE_ROWS:   Final[int] =  70
TOTAL_IMMATURE_ROWS:  Final[int] = 469

# Google max spend per row in currency ($) after micros conversion
GOOGLE_MAX_SPEND_USD: Final[float] = 46_406.0   # raw max micros / 1e6, rounded up

# Meta reach: 31.8% zeros, max 78125 (forensic contract)
META_REACH_MAX: Final[float] = 78_125.0

# Platform max dates (forensic contract)
GOOGLE_DATE_MAX: Final[date] = date(2026, 6, 4)
META_DATE_MAX:   Final[date] = date(2026, 6, 5)
BING_DATE_MAX:   Final[date] = date(2026, 6, 5)


# ---------------------------------------------------------------------------
# Module-scope fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data_dir() -> Path:
    return Path("dataset")


@pytest.fixture(scope="module")
def dataset(data_dir: Path) -> RawDataset:
    return load_raw_data(data_dir)


@pytest.fixture(scope="module")
def canon(dataset: RawDataset) -> pd.DataFrame:
    return harmonize(dataset)


@pytest.fixture(scope="module")
def google_rows(canon: pd.DataFrame) -> pd.DataFrame:
    return canon[canon["platform"] == "google"].copy()


@pytest.fixture(scope="module")
def meta_rows(canon: pd.DataFrame) -> pd.DataFrame:
    return canon[canon["platform"] == "meta"].copy()


@pytest.fixture(scope="module")
def bing_rows(canon: pd.DataFrame) -> pd.DataFrame:
    return canon[canon["platform"] == "bing"].copy()


# ---------------------------------------------------------------------------
# TestCanonicalSchema
# ---------------------------------------------------------------------------

class TestCanonicalSchema:
    """Top-level shape and schema contracts for the harmonized DataFrame."""

    def test_total_row_count(self, canon: pd.DataFrame) -> None:
        assert len(canon) == TOTAL_ROWS, (
            f"Expected {TOTAL_ROWS} total rows; got {len(canon)}"
        )

    def test_column_count(self, canon: pd.DataFrame) -> None:
        assert canon.shape[1] == CANONICAL_COL_COUNT

    def test_column_names_match_canonical(self, canon: pd.DataFrame) -> None:
        assert set(canon.columns) == set(CANONICAL_COLUMNS)

    def test_column_order_matches_canonical(self, canon: pd.DataFrame) -> None:
        assert list(canon.columns) == list(CANONICAL_COLUMNS), (
            f"Column order mismatch.\n  expected: {list(CANONICAL_COLUMNS)}\n"
            f"  got: {list(canon.columns)}"
        )

    def test_platform_values_are_exactly_three(self, canon: pd.DataFrame) -> None:
        assert set(canon["platform"].unique()) == {"google", "meta", "bing"}

    def test_channel_format_values_are_canonical(self, canon: pd.DataFrame) -> None:
        observed = set(canon["channel_format"].unique())
        unknown = observed - CANONICAL_CHANNEL_FORMATS
        assert not unknown, f"Unknown channel_format values: {unknown}"

    def test_no_duplicate_composite_key(self, canon: pd.DataFrame) -> None:
        dupes = canon.duplicated(subset=["platform", "campaign_id", "date"])
        assert not dupes.any(), f"{dupes.sum()} duplicate (platform, campaign_id, date) rows"

    def test_harmonize_from_dir_matches_harmonize(
        self, dataset: RawDataset, data_dir: Path
    ) -> None:
        direct = harmonize(dataset)
        from_dir = harmonize_from_dir(data_dir)
        assert direct.shape == from_dir.shape
        assert list(direct.columns) == list(from_dir.columns)


# ---------------------------------------------------------------------------
# TestDtypes
# ---------------------------------------------------------------------------

class TestDtypes:
    """Dtype contracts for every canonical column."""

    def test_platform_is_category(self, canon: pd.DataFrame) -> None:
        assert canon["platform"].dtype.name == "category"

    def test_channel_format_is_category(self, canon: pd.DataFrame) -> None:
        assert canon["channel_format"].dtype.name == "category"

    def test_campaign_id_is_int64(self, canon: pd.DataFrame) -> None:
        assert canon["campaign_id"].dtype == np.dtype("int64")

    def test_campaign_name_is_object(self, canon: pd.DataFrame) -> None:
        assert canon["campaign_name"].dtype == np.dtype("object")

    def test_date_is_datetime64(self, canon: pd.DataFrame) -> None:
        assert pd.api.types.is_datetime64_ns_dtype(canon["date"])

    def test_spend_is_float64(self, canon: pd.DataFrame) -> None:
        assert canon["spend"].dtype == np.dtype("float64")

    def test_revenue_attributed_is_float64(self, canon: pd.DataFrame) -> None:
        assert canon["revenue_attributed"].dtype == np.dtype("float64")

    def test_clicks_is_float64(self, canon: pd.DataFrame) -> None:
        assert canon["clicks"].dtype == np.dtype("float64")

    def test_impressions_is_float64(self, canon: pd.DataFrame) -> None:
        assert canon["impressions"].dtype == np.dtype("float64")

    def test_conversions_is_float64(self, canon: pd.DataFrame) -> None:
        assert canon["conversions"].dtype == np.dtype("float64")

    def test_daily_budget_is_float64(self, canon: pd.DataFrame) -> None:
        assert canon["daily_budget"].dtype == np.dtype("float64")

    def test_reach_is_float64(self, canon: pd.DataFrame) -> None:
        assert canon["reach"].dtype == np.dtype("float64")

    def test_video_views_is_float64(self, canon: pd.DataFrame) -> None:
        assert canon["video_views"].dtype == np.dtype("float64")

    def test_attribution_mature_is_bool(self, canon: pd.DataFrame) -> None:
        assert canon["attribution_mature"].dtype == np.dtype("bool")


# ---------------------------------------------------------------------------
# TestGoogleNormalization
# ---------------------------------------------------------------------------

class TestGoogleNormalization:
    """Transformations specific to the Google Ads platform rows."""

    def test_google_row_count(self, google_rows: pd.DataFrame) -> None:
        assert len(google_rows) == GOOGLE_ROWS

    def test_spend_is_currency_not_micros(self, google_rows: pd.DataFrame) -> None:
        # Raw max micros = 46 405 520 000 → USD ≈ $46 405.52
        # If micros were not converted, max would be > 1e9; in currency it's < 1e5.
        assert google_rows["spend"].max() < 1_000_000, (
            "Google spend appears to still be in micros — division by 1e6 not applied."
        )

    def test_spend_max_is_within_expected_range(self, google_rows: pd.DataFrame) -> None:
        assert google_rows["spend"].max() == pytest.approx(GOOGLE_MAX_SPEND_USD, abs=1.0)

    def test_spend_non_negative(self, google_rows: pd.DataFrame) -> None:
        assert (google_rows["spend"] >= 0).all()

    def test_revenue_attributed_non_negative(self, google_rows: pd.DataFrame) -> None:
        assert (google_rows["revenue_attributed"] >= 0).all()

    def test_video_views_are_present(self, google_rows: pd.DataFrame) -> None:
        non_null = google_rows["video_views"].notna()
        assert non_null.any(), "Google video_views should have non-NaN values"

    def test_video_views_non_negative(self, google_rows: pd.DataFrame) -> None:
        non_null = google_rows["video_views"].dropna()
        assert (non_null >= 0).all()

    def test_reach_is_all_null(self, google_rows: pd.DataFrame) -> None:
        assert google_rows["reach"].isna().all(), (
            "Google rows should have NaN reach (reach not in Google dataset)"
        )

    def test_conversions_are_present(self, google_rows: pd.DataFrame) -> None:
        assert google_rows["conversions"].notna().any()

    def test_channel_format_contains_pmax(self, google_rows: pd.DataFrame) -> None:
        # PERFORMANCE_MAX is the dominant Google channel type (13 982 rows)
        assert "PMax" in google_rows["channel_format"].values

    def test_channel_format_contains_search(self, google_rows: pd.DataFrame) -> None:
        assert "Search" in google_rows["channel_format"].values

    def test_channel_format_contains_upperfunnel(self, google_rows: pd.DataFrame) -> None:
        # VIDEO + DEMAND_GEN + DISPLAY all map here
        assert "UpperFunnel" in google_rows["channel_format"].values

    def test_channel_format_all_valid(self, google_rows: pd.DataFrame) -> None:
        observed = set(google_rows["channel_format"].unique())
        assert observed.issubset(CANONICAL_CHANNEL_FORMATS)

    def test_date_column_named_date(self, google_rows: pd.DataFrame) -> None:
        assert "date" in google_rows.columns
        assert "segments_date" not in google_rows.columns

    def test_campaign_id_preserved(self, google_rows: pd.DataFrame) -> None:
        assert google_rows["campaign_id"].nunique() == 92


# ---------------------------------------------------------------------------
# TestMetaNormalization
# ---------------------------------------------------------------------------

class TestMetaNormalization:
    """Transformations specific to the Meta Ads platform rows."""

    def test_meta_row_count(self, meta_rows: pd.DataFrame) -> None:
        assert len(meta_rows) == META_ROWS

    def test_revenue_attributed_came_from_conversion_column(
        self, meta_rows: pd.DataFrame
    ) -> None:
        # The raw 'conversion' column had mean ≈ $485, max ≈ $26 539
        assert meta_rows["revenue_attributed"].mean() > 100, (
            "Meta revenue_attributed mean < $100 — column may be event count, not value"
        )
        assert meta_rows["revenue_attributed"].max() > 10_000

    def test_revenue_attributed_has_fractional_values(
        self, meta_rows: pd.DataFrame
    ) -> None:
        non_zero = meta_rows["revenue_attributed"][meta_rows["revenue_attributed"] > 0]
        assert (non_zero % 1 != 0).any(), (
            "All Meta revenue_attributed values are whole numbers — suspicious"
        )

    def test_conversion_column_not_present(self, meta_rows: pd.DataFrame) -> None:
        assert "conversion" not in meta_rows.columns, (
            "Raw 'conversion' column survived into canonical — should have been renamed"
        )

    def test_conversions_count_is_all_null(self, meta_rows: pd.DataFrame) -> None:
        assert meta_rows["conversions"].isna().all(), (
            "Meta 'conversions' (event count) should be all NaN — not in dataset"
        )

    def test_video_views_is_all_null(self, meta_rows: pd.DataFrame) -> None:
        assert meta_rows["video_views"].isna().all()

    def test_reach_is_preserved(self, meta_rows: pd.DataFrame) -> None:
        assert meta_rows["reach"].notna().any(), "Meta reach should have non-NaN values"

    def test_reach_max_matches_forensic_value(self, meta_rows: pd.DataFrame) -> None:
        assert meta_rows["reach"].max() == pytest.approx(META_REACH_MAX, abs=1.0)

    def test_reach_non_negative(self, meta_rows: pd.DataFrame) -> None:
        assert (meta_rows["reach"] >= 0).all()

    def test_spend_unchanged(self, meta_rows: pd.DataFrame, dataset: RawDataset) -> None:
        # Meta spend is already in currency — harmonizer must not divide it
        assert meta_rows["spend"].sum() == pytest.approx(
            dataset.meta["spend"].sum(), rel=1e-6
        )

    def test_date_column_named_date(self, meta_rows: pd.DataFrame) -> None:
        assert "date" in meta_rows.columns
        assert "date_start" not in meta_rows.columns

    def test_campaign_id_preserved(self, meta_rows: pd.DataFrame) -> None:
        assert meta_rows["campaign_id"].nunique() == 16

    def test_channel_format_adv_plus_campaigns_are_pmax(
        self, meta_rows: pd.DataFrame
    ) -> None:
        adv_plus = meta_rows[meta_rows["campaign_name"].str.contains("Adv_Plus")]
        assert (adv_plus["channel_format"] == "PMax").all()

    def test_channel_format_dpa_campaigns_are_shopping(
        self, meta_rows: pd.DataFrame
    ) -> None:
        dpa = meta_rows[meta_rows["campaign_name"].str.contains("DPA")]
        assert (dpa["channel_format"] == "Shopping").all()

    def test_channel_format_brand_campaigns_are_brand(
        self, meta_rows: pd.DataFrame
    ) -> None:
        brand = meta_rows[meta_rows["campaign_name"].str.contains("Brand")]
        assert (brand["channel_format"] == "Brand").all()

    def test_channel_format_generic_campaign_is_other(
        self, meta_rows: pd.DataFrame
    ) -> None:
        generic = meta_rows[meta_rows["campaign_name"].isin(
            ["Generic_Campaign_01", "Generic_Campaign_02"]
        )]
        assert (generic["channel_format"] == "Other").all()


# ---------------------------------------------------------------------------
# TestBingNormalization
# ---------------------------------------------------------------------------

class TestBingNormalization:
    """Transformations specific to the Bing Ads platform rows."""

    def test_bing_row_count(self, bing_rows: pd.DataFrame) -> None:
        assert len(bing_rows) == BING_ROWS

    def test_reach_is_all_null(self, bing_rows: pd.DataFrame) -> None:
        assert bing_rows["reach"].isna().all()

    def test_video_views_is_all_null(self, bing_rows: pd.DataFrame) -> None:
        assert bing_rows["video_views"].isna().all()

    def test_revenue_attributed_matches_raw(
        self, bing_rows: pd.DataFrame, dataset: RawDataset
    ) -> None:
        assert bing_rows["revenue_attributed"].sum() == pytest.approx(
            dataset.bing["Revenue"].sum(), rel=1e-6
        )

    def test_spend_matches_raw(
        self, bing_rows: pd.DataFrame, dataset: RawDataset
    ) -> None:
        assert bing_rows["spend"].sum() == pytest.approx(
            dataset.bing["Spend"].sum(), rel=1e-6
        )

    def test_campaign_name_column_present(self, bing_rows: pd.DataFrame) -> None:
        assert "campaign_name" in bing_rows.columns
        assert "CampaignName" not in bing_rows.columns

    def test_campaign_id_column_present(self, bing_rows: pd.DataFrame) -> None:
        assert "campaign_id" in bing_rows.columns
        assert "CampaignId" not in bing_rows.columns

    def test_date_column_named_date(self, bing_rows: pd.DataFrame) -> None:
        assert "date" in bing_rows.columns
        assert "TimePeriod" not in bing_rows.columns

    def test_channel_format_contains_pmax(self, bing_rows: pd.DataFrame) -> None:
        assert "PMax" in bing_rows["channel_format"].values

    def test_channel_format_contains_search(self, bing_rows: pd.DataFrame) -> None:
        assert "Search" in bing_rows["channel_format"].values

    def test_channel_format_contains_shopping(self, bing_rows: pd.DataFrame) -> None:
        assert "Shopping" in bing_rows["channel_format"].values

    def test_channel_format_contains_upperfunnel(self, bing_rows: pd.DataFrame) -> None:
        assert "UpperFunnel" in bing_rows["channel_format"].values

    def test_channel_format_all_valid(self, bing_rows: pd.DataFrame) -> None:
        observed = set(bing_rows["channel_format"].unique())
        assert observed.issubset(CANONICAL_CHANNEL_FORMATS)

    def test_campaign_id_preserved(self, bing_rows: pd.DataFrame) -> None:
        assert bing_rows["campaign_id"].nunique() == 28


# ---------------------------------------------------------------------------
# TestAttributionMaturity
# ---------------------------------------------------------------------------

class TestAttributionMaturity:
    """Attribution-maturity flag: correct cutoff, correct counts per platform."""

    def test_attribution_mature_is_bool_dtype(self, canon: pd.DataFrame) -> None:
        assert canon["attribution_mature"].dtype == np.dtype("bool")

    def test_total_immature_row_count(self, canon: pd.DataFrame) -> None:
        n_immature = (~canon["attribution_mature"]).sum()
        assert n_immature == TOTAL_IMMATURE_ROWS, (
            f"Expected {TOTAL_IMMATURE_ROWS} immature rows; got {n_immature}"
        )

    def test_google_immature_row_count(self, google_rows: pd.DataFrame) -> None:
        n = (~google_rows["attribution_mature"]).sum()
        assert n == GOOGLE_IMMATURE_ROWS

    def test_meta_immature_row_count(self, meta_rows: pd.DataFrame) -> None:
        n = (~meta_rows["attribution_mature"]).sum()
        assert n == META_IMMATURE_ROWS

    def test_bing_immature_row_count(self, bing_rows: pd.DataFrame) -> None:
        n = (~bing_rows["attribution_mature"]).sum()
        assert n == BING_IMMATURE_ROWS

    def test_google_immature_dates_are_last_14_days(
        self, google_rows: pd.DataFrame
    ) -> None:
        max_date = google_rows["date"].max()
        cutoff   = max_date - pd.Timedelta(days=ATTRIBUTION_MATURITY_DAYS)
        immature = google_rows[~google_rows["attribution_mature"]]
        assert (immature["date"] > cutoff).all(), (
            "Some Google immature rows have dates inside the mature window"
        )
        mature = google_rows[google_rows["attribution_mature"]]
        assert (mature["date"] <= cutoff).all(), (
            "Some Google mature rows have dates inside the immature window"
        )

    def test_meta_immature_dates_are_last_14_days(
        self, meta_rows: pd.DataFrame
    ) -> None:
        max_date = meta_rows["date"].max()
        cutoff   = max_date - pd.Timedelta(days=ATTRIBUTION_MATURITY_DAYS)
        immature = meta_rows[~meta_rows["attribution_mature"]]
        assert (immature["date"] > cutoff).all()
        mature = meta_rows[meta_rows["attribution_mature"]]
        assert (mature["date"] <= cutoff).all()

    def test_bing_immature_dates_are_last_14_days(
        self, bing_rows: pd.DataFrame
    ) -> None:
        max_date = bing_rows["date"].max()
        cutoff   = max_date - pd.Timedelta(days=ATTRIBUTION_MATURITY_DAYS)
        immature = bing_rows[~bing_rows["attribution_mature"]]
        assert (immature["date"] > cutoff).all()
        mature = bing_rows[bing_rows["attribution_mature"]]
        assert (mature["date"] <= cutoff).all()

    def test_mature_fraction_is_dominant(self, canon: pd.DataFrame) -> None:
        mature_frac = canon["attribution_mature"].mean()
        assert mature_frac > 0.95, (
            f"Expected >95% mature rows; got {mature_frac:.1%} — cutoff logic may be wrong"
        )

    def test_each_platform_cutoff_is_independent(self, canon: pd.DataFrame) -> None:
        for platform in ("google", "meta", "bing"):
            sub = canon[canon["platform"] == platform]
            max_date = sub["date"].max()
            cutoff   = max_date - pd.Timedelta(days=ATTRIBUTION_MATURITY_DAYS)
            immature = sub[~sub["attribution_mature"]]
            mature   = sub[sub["attribution_mature"]]
            assert (immature["date"] > cutoff).all(), f"{platform}: immature dates wrong"
            assert (mature["date"] <= cutoff).all(),  f"{platform}: mature dates wrong"


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------

class TestValidation:
    """HarmonizeError is raised exactly when expected by _validate_canonical."""

    def _minimal_valid_df(self) -> pd.DataFrame:
        """Return a minimal canonical DataFrame that passes all checks."""
        return pd.DataFrame({
            "platform":            ["google"],
            "campaign_id":         [1],
            "campaign_name":       ["Test"],
            "date":                [pd.Timestamp("2025-01-01")],
            "spend":               [10.0],
            "revenue_attributed":  [20.0],
            "clicks":              [100.0],
            "impressions":         [1000.0],
            "conversions":         [5.0],
            "daily_budget":        [50.0],
            "channel_format":      ["Search"],
            "reach":               [np.nan],
            "video_views":         [np.nan],
            "attribution_mature":  [True],
        })

    def test_valid_df_does_not_raise(self) -> None:
        _validate_canonical(self._minimal_valid_df())

    def test_negative_spend_raises(self) -> None:
        df = self._minimal_valid_df()
        df.loc[0, "spend"] = -1.0
        with pytest.raises(HarmonizeError, match="negative spend"):
            _validate_canonical(df)

    def test_negative_revenue_raises(self) -> None:
        df = self._minimal_valid_df()
        df.loc[0, "revenue_attributed"] = -0.01
        with pytest.raises(HarmonizeError, match="negative revenue_attributed"):
            _validate_canonical(df)

    def test_duplicate_composite_key_raises(self) -> None:
        df = pd.concat([self._minimal_valid_df(), self._minimal_valid_df()], ignore_index=True)
        with pytest.raises(HarmonizeError, match="duplicate"):
            _validate_canonical(df)

    def test_harmonize_error_has_details_on_negative_spend(self) -> None:
        df = self._minimal_valid_df()
        df.loc[0, "spend"] = -5.0
        try:
            _validate_canonical(df)
            pytest.fail("HarmonizeError was not raised")
        except HarmonizeError as exc:
            assert exc.details, "HarmonizeError.details should contain sample rows"

    def test_real_data_passes_validation(self, canon: pd.DataFrame) -> None:
        # If harmonize() returned without error, validation already passed.
        # This re-runs it explicitly to confirm the DataFrame remains valid.
        _validate_canonical(canon)

    def test_spend_non_negative_in_full_dataset(self, canon: pd.DataFrame) -> None:
        assert (canon["spend"] >= 0).all()

    def test_revenue_attributed_non_negative_in_full_dataset(
        self, canon: pd.DataFrame
    ) -> None:
        assert (canon["revenue_attributed"] >= 0).all()


# ---------------------------------------------------------------------------
# TestChannelFormatMapping — unit tests via _infer_meta_channel_format
# ---------------------------------------------------------------------------

class TestMetaChannelFormatInference:
    """
    Unit tests for _infer_meta_channel_format using the actual 16 Meta
    campaign names observed in the dataset.
    """

    def _infer(self, *names: str) -> list[str]:
        return _infer_meta_channel_format(pd.Series(list(names))).tolist()

    # --- Adv_Plus (PMax) — highest priority ---

    def test_adv_plus_maps_to_pmax(self) -> None:
        assert self._infer("Prospecting_Adv_Plus_Campaign_01") == ["PMax"]

    def test_adv_plus_campaign_02_maps_to_pmax(self) -> None:
        assert self._infer("Prospecting_Adv_Plus_Campaign_02") == ["PMax"]

    def test_adv_plus_overrides_prospecting(self) -> None:
        # "Prospecting_Adv_Plus_*" contains both — PMax must win
        result = _infer_meta_channel_format(pd.Series(["Prospecting_Adv_Plus_X"]))
        assert result.iloc[0] == "PMax"

    # --- DPA (Shopping) ---

    def test_dpa_maps_to_shopping(self) -> None:
        assert self._infer("Prospecting_DPA_Campaign_01") == ["Shopping"]
        assert self._infer("Prospecting_DPA_Campaign_02") == ["Shopping"]
        assert self._infer("Prospecting_DPA_Campaign_04") == ["Shopping"]

    def test_remarketing_dpa_maps_to_shopping(self) -> None:
        assert self._infer("Remarketing_DPA_Campaign_01") == ["Shopping"]
        assert self._infer("Remarketing_DPA_Campaign_02") == ["Shopping"]
        assert self._infer("Remarketing_DPA_Campaign_03") == ["Shopping"]

    def test_dpa_overrides_remarketing(self) -> None:
        # "Remarketing_DPA_*" contains both — Shopping must win
        result = _infer_meta_channel_format(pd.Series(["Remarketing_DPA_X"]))
        assert result.iloc[0] == "Shopping"

    def test_dpa_overrides_prospecting(self) -> None:
        result = _infer_meta_channel_format(pd.Series(["Prospecting_DPA_X"]))
        assert result.iloc[0] == "Shopping"

    # --- Brand ---

    def test_generic_brand_maps_to_brand(self) -> None:
        assert self._infer("Generic_Brand_Campaign_01") == ["Brand"]

    def test_prospecting_brand_maps_to_brand(self) -> None:
        assert self._infer("Prospecting_Brand_Campaign_01") == ["Brand"]
        assert self._infer("Prospecting_Brand_Campaign_02") == ["Brand"]

    def test_remarketing_brand_maps_to_brand(self) -> None:
        assert self._infer("Remarketing_Brand_Campaign_01") == ["Brand"]
        assert self._infer("Remarketing_Brand_Campaign_02") == ["Brand"]
        assert self._infer("Remarketing_Brand_Campaign_03") == ["Brand"]

    def test_brand_overrides_remarketing(self) -> None:
        result = _infer_meta_channel_format(pd.Series(["Remarketing_Brand_X"]))
        assert result.iloc[0] == "Brand"

    def test_brand_overrides_prospecting(self) -> None:
        result = _infer_meta_channel_format(pd.Series(["Prospecting_Brand_X"]))
        assert result.iloc[0] == "Brand"

    # --- Other ---

    def test_generic_campaign_maps_to_other(self) -> None:
        assert self._infer("Generic_Campaign_01") == ["Other"]
        assert self._infer("Generic_Campaign_02") == ["Other"]

    def test_empty_name_maps_to_other(self) -> None:
        result = _infer_meta_channel_format(pd.Series([""]))
        assert result.iloc[0] == "Other"

    def test_na_name_maps_to_other(self) -> None:
        result = _infer_meta_channel_format(pd.Series([None]))
        assert result.iloc[0] == "Other"

    def test_bulk_all_16_campaigns(self) -> None:
        campaigns = [
            ("Generic_Brand_Campaign_01",         "Brand"),
            ("Generic_Campaign_01",               "Other"),
            ("Generic_Campaign_02",               "Other"),
            ("Prospecting_Adv_Plus_Campaign_01",  "PMax"),
            ("Prospecting_Adv_Plus_Campaign_02",  "PMax"),
            ("Prospecting_Brand_Campaign_01",     "Brand"),
            ("Prospecting_Brand_Campaign_02",     "Brand"),
            ("Prospecting_DPA_Campaign_01",       "Shopping"),
            ("Prospecting_DPA_Campaign_02",       "Shopping"),
            ("Prospecting_DPA_Campaign_04",       "Shopping"),
            ("Remarketing_Brand_Campaign_01",     "Brand"),
            ("Remarketing_Brand_Campaign_02",     "Brand"),
            ("Remarketing_Brand_Campaign_03",     "Brand"),
            ("Remarketing_DPA_Campaign_01",       "Shopping"),
            ("Remarketing_DPA_Campaign_02",       "Shopping"),
            ("Remarketing_DPA_Campaign_03",       "Shopping"),
        ]
        names    = pd.Series([c[0] for c in campaigns])
        expected = [c[1] for c in campaigns]
        result   = _infer_meta_channel_format(names).tolist()
        assert result == expected, (
            "\n".join(
                f"  {n}: expected={e}, got={r}"
                for n, e, r in zip([c[0] for c in campaigns], expected, result)
                if e != r
            )
        )


# ---------------------------------------------------------------------------
# TestChannelFormatGoogleBing — channel type mapping via normalize functions
# ---------------------------------------------------------------------------

class TestChannelFormatGoogleBing:
    """Unit tests for Google/Bing channel type mapping via the normalize functions."""

    def _google_df(self, channel_type: str) -> pd.DataFrame:
        """Minimal synthetic Google raw row."""
        return pd.DataFrame({
            "campaign_id":                        [1],
            "segments_date":                      [pd.Timestamp("2025-01-01")],
            "metrics_clicks":                     [10],
            "metrics_conversions":                [1.0],
            "metrics_cost_micros":                [1_000_000],
            "metrics_impressions":                [100],
            "metrics_video_views":                [0],
            "metrics_conversions_value":          [50.0],
            "campaign_advertising_channel_type":  [channel_type],
            "campaign_budget_amount":             [100.0],
            "campaign_name":                      ["Test"],
        })

    def _bing_df(self, campaign_type: str) -> pd.DataFrame:
        """Minimal synthetic Bing raw row."""
        return pd.DataFrame({
            "CampaignId":    [1],
            "TimePeriod":    [pd.Timestamp("2025-01-01")],
            "Revenue":       [50.0],
            "Spend":         [10.0],
            "Clicks":        [5],
            "Impressions":   [100],
            "Conversions":   [2.0],
            "CampaignType":  [campaign_type],
            "DailyBudget":   [20.0],
            "CampaignName":  ["Test"],
        })

    # Google mappings
    def test_google_search_maps_to_search(self) -> None:
        out = _normalize_google(self._google_df("SEARCH"))
        assert out["channel_format"].iloc[0] == "Search"

    def test_google_pmax_maps_to_pmax(self) -> None:
        out = _normalize_google(self._google_df("PERFORMANCE_MAX"))
        assert out["channel_format"].iloc[0] == "PMax"

    def test_google_shopping_maps_to_shopping(self) -> None:
        out = _normalize_google(self._google_df("SHOPPING"))
        assert out["channel_format"].iloc[0] == "Shopping"

    def test_google_video_maps_to_upperfunnel(self) -> None:
        out = _normalize_google(self._google_df("VIDEO"))
        assert out["channel_format"].iloc[0] == "UpperFunnel"

    def test_google_demand_gen_maps_to_upperfunnel(self) -> None:
        out = _normalize_google(self._google_df("DEMAND_GEN"))
        assert out["channel_format"].iloc[0] == "UpperFunnel"

    def test_google_display_maps_to_upperfunnel(self) -> None:
        out = _normalize_google(self._google_df("DISPLAY"))
        assert out["channel_format"].iloc[0] == "UpperFunnel"

    def test_google_unknown_type_maps_to_other(self) -> None:
        out = _normalize_google(self._google_df("UNKNOWN_FUTURE_TYPE"))
        assert out["channel_format"].iloc[0] == "Other"

    # Bing mappings
    def test_bing_search_maps_to_search(self) -> None:
        out = _normalize_bing(self._bing_df("Search"))
        assert out["channel_format"].iloc[0] == "Search"

    def test_bing_pmax_maps_to_pmax(self) -> None:
        out = _normalize_bing(self._bing_df("PerformanceMax"))
        assert out["channel_format"].iloc[0] == "PMax"

    def test_bing_shopping_maps_to_shopping(self) -> None:
        out = _normalize_bing(self._bing_df("Shopping"))
        assert out["channel_format"].iloc[0] == "Shopping"

    def test_bing_audience_maps_to_upperfunnel(self) -> None:
        out = _normalize_bing(self._bing_df("Audience"))
        assert out["channel_format"].iloc[0] == "UpperFunnel"

    def test_bing_unknown_type_maps_to_other(self) -> None:
        out = _normalize_bing(self._bing_df("Unknown"))
        assert out["channel_format"].iloc[0] == "Other"


# ---------------------------------------------------------------------------
# TestMicrosConversion — unit tests for Google spend calculation
# ---------------------------------------------------------------------------

class TestMicrosConversion:
    """Verify the micros-to-currency conversion is exact."""

    def _google_row(self, micros: int) -> pd.DataFrame:
        return pd.DataFrame({
            "campaign_id":                        [1],
            "segments_date":                      [pd.Timestamp("2025-01-01")],
            "metrics_clicks":                     [0],
            "metrics_conversions":                [0.0],
            "metrics_cost_micros":                [micros],
            "metrics_impressions":                [0],
            "metrics_video_views":                [0],
            "metrics_conversions_value":          [0.0],
            "campaign_advertising_channel_type":  ["SEARCH"],
            "campaign_budget_amount":             [100.0],
            "campaign_name":                      ["Test"],
        })

    def test_zero_micros_gives_zero_spend(self) -> None:
        out = _normalize_google(self._google_row(0))
        assert out["spend"].iloc[0] == 0.0

    def test_one_million_micros_gives_one_dollar(self) -> None:
        out = _normalize_google(self._google_row(1_000_000))
        assert out["spend"].iloc[0] == pytest.approx(1.0)

    def test_typical_micros_converts_correctly(self) -> None:
        out = _normalize_google(self._google_row(46_405_520_000))
        assert out["spend"].iloc[0] == pytest.approx(46_405.52, abs=0.01)

    def test_spend_is_float64_after_conversion(self) -> None:
        out = _normalize_google(self._google_row(1_000_000))
        assert out["spend"].dtype == np.dtype("float64")


# ---------------------------------------------------------------------------
# TestMaturityCutoffHelper
# ---------------------------------------------------------------------------

class TestMaturityCutoffHelper:
    """Unit tests for _maturity_cutoff helper function."""

    def test_cutoff_is_14_days_before_max(self) -> None:
        dates = pd.Series(pd.date_range("2025-01-01", periods=30, freq="D"))
        cutoff = _maturity_cutoff(dates)
        expected = pd.Timestamp("2025-01-30") - pd.Timedelta(days=14)
        assert cutoff == expected

    def test_single_date_cutoff_is_14_days_before(self) -> None:
        dates = pd.Series([pd.Timestamp("2025-06-01")])
        cutoff = _maturity_cutoff(dates)
        assert cutoff == pd.Timestamp("2025-05-18")

    def test_cutoff_excludes_exactly_14_days(self) -> None:
        max_date = pd.Timestamp("2026-06-04")
        dates = pd.date_range("2024-01-01", end=max_date, freq="D")
        cutoff = _maturity_cutoff(pd.Series(dates))
        immature_mask = pd.Series(dates) > cutoff
        # Dates from cutoff+1 to max_date inclusive = 14 days
        assert immature_mask.sum() == 14
