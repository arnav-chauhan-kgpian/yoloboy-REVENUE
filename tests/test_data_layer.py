"""
tests/test_data_layer.py
========================
Comprehensive test suite for the Phase 1 data layer (src/data/loader.py).

Every expected value used as a constant in this file was derived from the
forensic analysis of the actual CSV files — they are DATA CONTRACTS.  If
any constant fails, the underlying source files have changed.

Test organisation
-----------------
TestDataLoaderInit          — construction, path validation, error cases
TestLoadRawDataConvenience  — module-level convenience function
TestRawDatasetContainer     — RawDataset dataclass helpers
TestGoogleSchema            — column names, dtypes, index-artifact removal
TestGoogleDataIntegrity     — row counts, key uniqueness, value constraints
TestMetaSchema              — column names, dtypes, index-artifact removal
TestMetaDataIntegrity       — row counts, key uniqueness, value constraints
TestBingSchema              — column names, dtypes, index-artifact removal
TestBingDataIntegrity       — row counts, key uniqueness, value constraints
TestCrossDataset            — relationships and overlap between platforms

Running
-------
    cd D:\\AIgnition
    pytest tests/test_data_layer.py -v
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure project root is importable when pytest runs from any working dir.
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.loader import (
    DataLoadError,
    DataLoader,
    RawDataset,
    load_raw_data,
)

# ---------------------------------------------------------------------------
# Forensic-confirmed data contracts
# Changing any of these values means the source CSV files have changed.
# ---------------------------------------------------------------------------

# --- Row counts (confirmed during forensic analysis) ----------------------
GOOGLE_ROWS: Final[int] = 19_272
META_ROWS:   Final[int] =  3_417
BING_ROWS:   Final[int] =  2_873
TOTAL_ROWS:  Final[int] = 25_562

# --- Column counts (after Unnamed: 0 artifact is dropped) -----------------
GOOGLE_COLS: Final[int] = 11
META_COLS:   Final[int] = 12
BING_COLS:   Final[int] = 10

# --- Unique campaign counts ------------------------------------------------
GOOGLE_CAMPAIGNS: Final[int] = 92
META_CAMPAIGNS:   Final[int] = 16
BING_CAMPAIGNS:   Final[int] = 28

# --- NULL counts (forensically confirmed) ---------------------------------
GOOGLE_BUDGET_NULLS: Final[int] = 14   # campaign_budget_amount
META_BUDGET_NULLS:   Final[int] =  7   # daily_budget
BING_TOTAL_NULLS:    Final[int] =  0   # Bing has no NULLs

# --- Date ranges ----------------------------------------------------------
GOOGLE_DATE_MIN: Final[date] = date(2024,  1,  1)
GOOGLE_DATE_MAX: Final[date] = date(2026,  6,  4)
META_DATE_MIN:   Final[date] = date(2024,  5, 23)
META_DATE_MAX:   Final[date] = date(2026,  6,  5)
BING_DATE_MIN:   Final[date] = date(2024,  5, 25)
BING_DATE_MAX:   Final[date] = date(2026,  6,  5)

# --- Cross-platform relationships -----------------------------------------
# 27 campaign names appear identically in both Google and Bing (confirmed).
GOOGLE_BING_SHARED_NAME_COUNT: Final[int] = 27

# --- Bing zero-inflation --------------------------------------------------
# Confirmed: 915 rows have Spend = 0; median Revenue = 0.
BING_ZERO_SPEND_ROWS: Final[int] = 915

# --- Bing DailyBudget bounds (confirmed: only 10.0 and 20.0 observed) ----
BING_BUDGET_MIN: Final[float] = 10.0
BING_BUDGET_MAX: Final[float] = 20.0

# --- Google channel type vocabulary (confirmed from forensics) ------------
GOOGLE_CHANNEL_TYPES: Final[frozenset[str]] = frozenset({
    "SEARCH",
    "PERFORMANCE_MAX",
    "VIDEO",
    "DEMAND_GEN",
    "SHOPPING",
    "DISPLAY",
})

# --- Bing campaign type vocabulary ----------------------------------------
BING_CAMPAIGN_TYPES: Final[frozenset[str]] = frozenset({
    "Search",
    "PerformanceMax",
    "Shopping",
    "Audience",
})

# --- Column name contracts ------------------------------------------------
GOOGLE_COLUMNS: Final[frozenset[str]] = frozenset({
    "campaign_id",
    "segments_date",
    "metrics_clicks",
    "metrics_conversions",
    "metrics_cost_micros",
    "metrics_impressions",
    "metrics_video_views",
    "metrics_conversions_value",
    "campaign_advertising_channel_type",
    "campaign_budget_amount",
    "campaign_name",
})

META_COLUMNS: Final[frozenset[str]] = frozenset({
    "campaign_id",
    "date_start",
    "cpc",
    "cpm",
    "ctr",
    "reach",
    "spend",
    "clicks",
    "impressions",
    "conversion",
    "daily_budget",
    "campaign_name",
})

BING_COLUMNS: Final[frozenset[str]] = frozenset({
    "CampaignId",
    "TimePeriod",
    "Revenue",
    "Spend",
    "Clicks",
    "Impressions",
    "Conversions",
    "CampaignType",
    "DailyBudget",
    "CampaignName",
})

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data_dir() -> Path:
    """Resolved path to the dataset directory."""
    path = _PROJECT_ROOT / "dataset"
    if not path.exists():
        pytest.skip(f"Dataset directory not found: {path}. Run from project root.")
    return path


@pytest.fixture(scope="module")
def dataset(data_dir: Path) -> RawDataset:
    """
    Load the full RawDataset once per test module.

    Using module scope avoids re-reading ~25,000 rows for every individual
    test while still isolating each test's assertions.
    """
    return load_raw_data(data_dir)


@pytest.fixture(scope="module")
def google(dataset: RawDataset) -> pd.DataFrame:
    """Google Ads DataFrame extracted from the module-scope dataset."""
    return dataset.google


@pytest.fixture(scope="module")
def meta(dataset: RawDataset) -> pd.DataFrame:
    """Meta Ads DataFrame extracted from the module-scope dataset."""
    return dataset.meta


@pytest.fixture(scope="module")
def bing(dataset: RawDataset) -> pd.DataFrame:
    """Bing / Microsoft Ads DataFrame extracted from the module-scope dataset."""
    return dataset.bing


# ---------------------------------------------------------------------------
# TestDataLoaderInit
# ---------------------------------------------------------------------------

class TestDataLoaderInit:
    """Tests for DataLoader construction and directory/file validation."""

    def test_valid_dir_does_not_raise(self, data_dir: Path) -> None:
        """DataLoader should initialise without error when data_dir exists."""
        loader = DataLoader(data_dir=data_dir)
        assert loader.data_dir == data_dir.resolve()

    def test_data_dir_property_returns_absolute_path(self, data_dir: Path) -> None:
        """The .data_dir property should return the absolute resolved path."""
        loader = DataLoader(data_dir=data_dir)
        assert loader.data_dir.is_absolute(), (
            "data_dir property must return an absolute path."
        )

    def test_accepts_string_path(self, data_dir: Path) -> None:
        """DataLoader should accept a plain string in addition to Path objects."""
        loader = DataLoader(data_dir=str(data_dir))
        assert loader.data_dir.is_dir()

    def test_missing_dir_raises_file_not_found(self, tmp_path: Path) -> None:
        """FileNotFoundError must be raised when data_dir does not exist."""
        nonexistent = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="Data directory not found"):
            DataLoader(data_dir=nonexistent)

    def test_file_path_raises_not_a_directory(self, tmp_path: Path) -> None:
        """NotADirectoryError must be raised when data_dir points to a file."""
        a_file = tmp_path / "not_a_directory.txt"
        a_file.write_text("hello")
        with pytest.raises(NotADirectoryError):
            DataLoader(data_dir=a_file)

    def test_individual_google_loader_matches_dataset(
        self, data_dir: Path, google: pd.DataFrame
    ) -> None:
        """load_google() should produce the same shape as dataset.google."""
        loader = DataLoader(data_dir=data_dir)
        df = loader.load_google()
        assert df.shape == google.shape

    def test_individual_meta_loader_matches_dataset(
        self, data_dir: Path, meta: pd.DataFrame
    ) -> None:
        """load_meta() should produce the same shape as dataset.meta."""
        loader = DataLoader(data_dir=data_dir)
        df = loader.load_meta()
        assert df.shape == meta.shape

    def test_individual_bing_loader_matches_dataset(
        self, data_dir: Path, bing: pd.DataFrame
    ) -> None:
        """load_bing() should produce the same shape as dataset.bing."""
        loader = DataLoader(data_dir=data_dir)
        df = loader.load_bing()
        assert df.shape == bing.shape


# ---------------------------------------------------------------------------
# TestLoadRawDataConvenience
# ---------------------------------------------------------------------------

class TestLoadRawDataConvenience:
    """Tests for the module-level load_raw_data() convenience function."""

    def test_returns_rawdataset_instance(self, data_dir: Path) -> None:
        """load_raw_data() must return a RawDataset."""
        result = load_raw_data(data_dir)
        assert isinstance(result, RawDataset)

    def test_shapes_match_loader_load_all(
        self, data_dir: Path, dataset: RawDataset
    ) -> None:
        """
        load_raw_data() must produce identical shapes to DataLoader.load_all().
        Verifies the convenience function is a faithful wrapper.
        """
        result = load_raw_data(data_dir)
        assert result.google.shape == dataset.google.shape
        assert result.meta.shape   == dataset.meta.shape
        assert result.bing.shape   == dataset.bing.shape

    def test_raises_file_not_found_for_bad_path(self, tmp_path: Path) -> None:
        """load_raw_data() must propagate FileNotFoundError for missing dir."""
        with pytest.raises(FileNotFoundError):
            load_raw_data(tmp_path / "no_such_dir")


# ---------------------------------------------------------------------------
# TestRawDatasetContainer
# ---------------------------------------------------------------------------

class TestRawDatasetContainer:
    """Tests for RawDataset dataclass helpers."""

    def test_total_rows(self, dataset: RawDataset) -> None:
        """total_rows() must equal the sum of all three platform row counts."""
        assert dataset.total_rows() == TOTAL_ROWS, (
            f"Expected {TOTAL_ROWS} total rows; got {dataset.total_rows()}."
        )

    def test_is_not_empty(self, dataset: RawDataset) -> None:
        """is_empty() must return False when all three DataFrames are populated."""
        assert not dataset.is_empty()

    def test_is_empty_for_empty_dataframes(self) -> None:
        """is_empty() must return True when any DataFrame is empty."""
        empty_ds = RawDataset(
            google=pd.DataFrame(),
            meta=pd.DataFrame(),
            bing=pd.DataFrame(),
        )
        assert empty_ds.is_empty()

    def test_summary_returns_string(self, dataset: RawDataset) -> None:
        """summary() must return a non-empty string."""
        result = dataset.summary()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summary_contains_all_platform_labels(self, dataset: RawDataset) -> None:
        """summary() must mention all three platforms by name."""
        result = dataset.summary()
        for platform in ("Google", "Meta", "Bing"):
            assert platform in result, (
                f"Platform label '{platform}' not found in summary()."
            )

    def test_summary_contains_total_row_count(self, dataset: RawDataset) -> None:
        """summary() must include the total row count."""
        result = dataset.summary()
        assert str(TOTAL_ROWS) in result.replace(",", ""), (
            f"Total row count {TOTAL_ROWS} not found in summary()."
        )

    def test_summary_contains_per_platform_rows(self, dataset: RawDataset) -> None:
        """summary() must include each platform's row count."""
        result = dataset.summary().replace(",", "")
        for count in (GOOGLE_ROWS, META_ROWS, BING_ROWS):
            assert str(count) in result, (
                f"Row count {count} not found in summary()."
            )

    def test_summary_for_empty_dataset_does_not_raise(self) -> None:
        """summary() must not raise for a dataset containing empty DataFrames."""
        empty_ds = RawDataset(
            google=pd.DataFrame(),
            meta=pd.DataFrame(),
            bing=pd.DataFrame(),
        )
        result = empty_ds.summary()
        assert "EMPTY" in result


# ---------------------------------------------------------------------------
# TestGoogleSchema
# ---------------------------------------------------------------------------

class TestGoogleSchema:
    """Column-level schema tests for the Google Ads DataFrame."""

    def test_shape(self, google: pd.DataFrame) -> None:
        assert google.shape == (GOOGLE_ROWS, GOOGLE_COLS), (
            f"Expected shape {(GOOGLE_ROWS, GOOGLE_COLS)}; got {google.shape}."
        )

    def test_exact_column_names(self, google: pd.DataFrame) -> None:
        """Exact set of columns must match the forensic contract."""
        assert set(google.columns) == GOOGLE_COLUMNS, (
            f"Column mismatch.\n"
            f"  Missing : {GOOGLE_COLUMNS - set(google.columns)}\n"
            f"  Extra   : {set(google.columns) - GOOGLE_COLUMNS}"
        )

    def test_index_artifact_absent(self, google: pd.DataFrame) -> None:
        """'Unnamed: 0' must have been dropped."""
        assert "Unnamed: 0" not in google.columns

    def test_date_column_is_datetime64(self, google: pd.DataFrame) -> None:
        assert pd.api.types.is_datetime64_any_dtype(google["segments_date"]), (
            f"segments_date dtype should be datetime64; got {google['segments_date'].dtype}."
        )

    def test_cost_micros_is_int64(self, google: pd.DataFrame) -> None:
        """
        metrics_cost_micros must remain int64 — the harmonizer divides by 1e6.
        If the loader accidentally converted to dollars, dtype would be float64.
        """
        assert google["metrics_cost_micros"].dtype == "int64", (
            f"Expected int64 for metrics_cost_micros; "
            f"got {google['metrics_cost_micros'].dtype}. "
            "Has the loader incorrectly applied the micros-to-currency conversion?"
        )

    def test_campaign_id_is_int64(self, google: pd.DataFrame) -> None:
        assert google["campaign_id"].dtype == "int64"

    def test_metrics_clicks_is_int64(self, google: pd.DataFrame) -> None:
        assert google["metrics_clicks"].dtype == "int64"

    def test_metrics_impressions_is_int64(self, google: pd.DataFrame) -> None:
        assert google["metrics_impressions"].dtype == "int64"

    def test_metrics_video_views_is_int64(self, google: pd.DataFrame) -> None:
        assert google["metrics_video_views"].dtype == "int64"

    def test_metrics_conversions_is_float64(self, google: pd.DataFrame) -> None:
        """Conversions are fractional due to attribution weighting."""
        assert google["metrics_conversions"].dtype == "float64"

    def test_conversions_value_is_float64(self, google: pd.DataFrame) -> None:
        assert google["metrics_conversions_value"].dtype == "float64"

    def test_budget_amount_is_float64(self, google: pd.DataFrame) -> None:
        """float64 is required to accommodate the 14 NULL values."""
        assert google["campaign_budget_amount"].dtype == "float64"

    def test_channel_type_is_object(self, google: pd.DataFrame) -> None:
        assert google["campaign_advertising_channel_type"].dtype == "object"

    def test_campaign_name_is_object(self, google: pd.DataFrame) -> None:
        assert google["campaign_name"].dtype == "object"


# ---------------------------------------------------------------------------
# TestGoogleDataIntegrity
# ---------------------------------------------------------------------------

class TestGoogleDataIntegrity:
    """Row-level integrity and value-constraint tests for Google Ads data."""

    def test_campaign_count(self, google: pd.DataFrame) -> None:
        n = google["campaign_id"].nunique()
        assert n == GOOGLE_CAMPAIGNS, (
            f"Expected {GOOGLE_CAMPAIGNS} unique Google campaigns; found {n}."
        )

    def test_date_range_min(self, google: pd.DataFrame) -> None:
        actual_min = google["segments_date"].min().date()
        assert actual_min == GOOGLE_DATE_MIN, (
            f"Google min date: expected {GOOGLE_DATE_MIN}, got {actual_min}."
        )

    def test_date_range_max(self, google: pd.DataFrame) -> None:
        actual_max = google["segments_date"].max().date()
        assert actual_max == GOOGLE_DATE_MAX, (
            f"Google max date: expected {GOOGLE_DATE_MAX}, got {actual_max}."
        )

    def test_no_duplicate_campaign_date_keys(self, google: pd.DataFrame) -> None:
        """Primary key (campaign_id, segments_date) must be unique."""
        dupes = google.duplicated(subset=["campaign_id", "segments_date"]).sum()
        assert dupes == 0, (
            f"Found {dupes} duplicate (campaign_id, segments_date) rows in Google data."
        )

    def test_campaign_id_to_name_is_injective(self, google: pd.DataFrame) -> None:
        """Each campaign_id maps to exactly one campaign_name (no renames)."""
        max_names_per_id = (
            google.groupby("campaign_id")["campaign_name"].nunique().max()
        )
        assert max_names_per_id == 1, (
            f"Some Google campaign_id maps to more than one name "
            f"(max {max_names_per_id}). Campaign may have been renamed."
        )

    def test_campaign_name_to_id_is_injective(self, google: pd.DataFrame) -> None:
        """Each campaign_name maps to exactly one campaign_id (no ID reuse)."""
        max_ids_per_name = (
            google.groupby("campaign_name")["campaign_id"].nunique().max()
        )
        assert max_ids_per_name == 1, (
            f"Some Google campaign_name maps to more than one ID "
            f"(max {max_ids_per_name})."
        )

    def test_budget_null_count(self, google: pd.DataFrame) -> None:
        """Exactly 14 NULLs in campaign_budget_amount (forensically confirmed)."""
        nulls = google["campaign_budget_amount"].isnull().sum()
        assert nulls == GOOGLE_BUDGET_NULLS, (
            f"Expected {GOOGLE_BUDGET_NULLS} NULL budget rows; found {nulls}."
        )

    def test_no_other_column_has_nulls(self, google: pd.DataFrame) -> None:
        """Only campaign_budget_amount should contain NULLs."""
        null_cols = {
            col: google[col].isnull().sum()
            for col in google.columns
            if col != "campaign_budget_amount" and google[col].isnull().any()
        }
        assert not null_cols, (
            f"Unexpected NULLs in Google columns: {null_cols}"
        )

    def test_cost_micros_are_non_negative(self, google: pd.DataFrame) -> None:
        """Ad spend cannot be negative."""
        assert (google["metrics_cost_micros"] >= 0).all(), (
            "Found negative values in metrics_cost_micros."
        )

    def test_cost_micros_scale_is_consistent_with_micros(
        self, google: pd.DataFrame
    ) -> None:
        """
        If cost were already in dollars, the mean would be ~$101.
        In micros, the mean should be ~101,000,000 (> 1,000,000).
        This guards against the harmonizer's division being applied in the loader.
        """
        mean_micros = google["metrics_cost_micros"].mean()
        assert mean_micros > 1_000_000, (
            f"Mean of metrics_cost_micros is {mean_micros:.0f}, which is too small "
            "for micros. The loader may have incorrectly applied the ÷1e6 conversion."
        )

    def test_channel_type_values_are_known(self, google: pd.DataFrame) -> None:
        """All channel type values must belong to the forensically-confirmed set."""
        observed = set(google["campaign_advertising_channel_type"].dropna().unique())
        unexpected = observed - GOOGLE_CHANNEL_TYPES
        assert not unexpected, (
            f"Unknown Google channel types found: {unexpected}. "
            f"Known types: {GOOGLE_CHANNEL_TYPES}"
        )

    def test_budget_positive_where_not_null(self, google: pd.DataFrame) -> None:
        """Non-null budgets must be strictly positive."""
        non_null = google["campaign_budget_amount"].dropna()
        assert (non_null > 0).all(), (
            "Found zero or negative campaign_budget_amount values."
        )

    def test_conversions_value_non_negative(self, google: pd.DataFrame) -> None:
        """Attributed revenue cannot be negative."""
        assert (google["metrics_conversions_value"] >= 0).all()

    def test_clicks_non_negative(self, google: pd.DataFrame) -> None:
        assert (google["metrics_clicks"] >= 0).all()

    def test_impressions_non_negative(self, google: pd.DataFrame) -> None:
        assert (google["metrics_impressions"] >= 0).all()


# ---------------------------------------------------------------------------
# TestMetaSchema
# ---------------------------------------------------------------------------

class TestMetaSchema:
    """Column-level schema tests for the Meta Ads DataFrame."""

    def test_shape(self, meta: pd.DataFrame) -> None:
        assert meta.shape == (META_ROWS, META_COLS), (
            f"Expected shape {(META_ROWS, META_COLS)}; got {meta.shape}."
        )

    def test_exact_column_names(self, meta: pd.DataFrame) -> None:
        assert set(meta.columns) == META_COLUMNS, (
            f"Column mismatch.\n"
            f"  Missing : {META_COLUMNS - set(meta.columns)}\n"
            f"  Extra   : {set(meta.columns) - META_COLUMNS}"
        )

    def test_index_artifact_absent(self, meta: pd.DataFrame) -> None:
        assert "Unnamed: 0" not in meta.columns

    def test_date_column_is_datetime64(self, meta: pd.DataFrame) -> None:
        assert pd.api.types.is_datetime64_any_dtype(meta["date_start"]), (
            f"date_start dtype should be datetime64; got {meta['date_start'].dtype}."
        )

    def test_campaign_id_is_int64(self, meta: pd.DataFrame) -> None:
        """Meta campaign IDs are 16-digit integers (e.g. 120210921616440533)."""
        assert meta["campaign_id"].dtype == "int64"

    def test_clicks_is_float64(self, meta: pd.DataFrame) -> None:
        """Meta reports clicks as float — distinct from Google/Bing int."""
        assert meta["clicks"].dtype == "float64"

    def test_impressions_is_float64(self, meta: pd.DataFrame) -> None:
        """Meta reports impressions as float."""
        assert meta["impressions"].dtype == "float64"

    def test_conversion_is_float64(self, meta: pd.DataFrame) -> None:
        assert meta["conversion"].dtype == "float64"

    def test_spend_is_float64(self, meta: pd.DataFrame) -> None:
        assert meta["spend"].dtype == "float64"

    def test_daily_budget_is_float64(self, meta: pd.DataFrame) -> None:
        assert meta["daily_budget"].dtype == "float64"

    def test_cpc_cpm_ctr_are_float64(self, meta: pd.DataFrame) -> None:
        for col in ("cpc", "cpm", "ctr"):
            assert meta[col].dtype == "float64", (
                f"Expected float64 for {col}; got {meta[col].dtype}."
            )

    def test_reach_is_float64(self, meta: pd.DataFrame) -> None:
        assert meta["reach"].dtype == "float64"


# ---------------------------------------------------------------------------
# TestMetaDataIntegrity
# ---------------------------------------------------------------------------

class TestMetaDataIntegrity:
    """Row-level integrity and value-constraint tests for Meta Ads data."""

    def test_campaign_count(self, meta: pd.DataFrame) -> None:
        n = meta["campaign_id"].nunique()
        assert n == META_CAMPAIGNS, (
            f"Expected {META_CAMPAIGNS} unique Meta campaigns; found {n}."
        )

    def test_date_range_min(self, meta: pd.DataFrame) -> None:
        actual_min = meta["date_start"].min().date()
        assert actual_min == META_DATE_MIN, (
            f"Meta min date: expected {META_DATE_MIN}, got {actual_min}."
        )

    def test_date_range_max(self, meta: pd.DataFrame) -> None:
        actual_max = meta["date_start"].max().date()
        assert actual_max == META_DATE_MAX, (
            f"Meta max date: expected {META_DATE_MAX}, got {actual_max}."
        )

    def test_no_duplicate_campaign_date_keys(self, meta: pd.DataFrame) -> None:
        """Primary key (campaign_id, date_start) must be unique."""
        dupes = meta.duplicated(subset=["campaign_id", "date_start"]).sum()
        assert dupes == 0, (
            f"Found {dupes} duplicate (campaign_id, date_start) rows in Meta data."
        )

    def test_campaign_id_to_name_is_injective(self, meta: pd.DataFrame) -> None:
        """Each campaign_id maps to exactly one campaign_name."""
        max_names = meta.groupby("campaign_id")["campaign_name"].nunique().max()
        assert max_names == 1, (
            f"Some Meta campaign_id maps to more than one name (max {max_names})."
        )

    def test_budget_null_count(self, meta: pd.DataFrame) -> None:
        """Exactly 7 NULLs in daily_budget (forensically confirmed)."""
        nulls = meta["daily_budget"].isnull().sum()
        assert nulls == META_BUDGET_NULLS, (
            f"Expected {META_BUDGET_NULLS} NULL budget rows; found {nulls}."
        )

    def test_conversion_is_revenue_value_not_event_count(
        self, meta: pd.DataFrame
    ) -> None:
        """
        The 'conversion' column is REVENUE VALUE (monetary), not an event count.

        Evidence:
          - Mean ≈ $485 (impossible as a per-day count)
          - Max  ≈ $26,539 (impossible as a per-day count)
          - Values contain decimal places (float, not integer-like)

        This test guards against future misinterpretation of the column.
        """
        non_zero = meta.loc[meta["conversion"] > 0, "conversion"]
        assert non_zero.mean() > 100, (
            f"Mean non-zero conversion is {non_zero.mean():.2f}. "
            "This is too low for revenue values (expected ~$485). "
            "Has the column been incorrectly treated as an event count?"
        )
        assert non_zero.max() > 10_000, (
            f"Max conversion is {non_zero.max():.2f}. "
            "Expected max ~$26,539 for revenue values."
        )
        # Revenue values must have decimal components (they are not integer-like)
        has_decimal = (non_zero % 1 != 0).any()
        assert has_decimal, (
            "All non-zero 'conversion' values are whole numbers — "
            "unusual for monetary revenue; expected fractional cents."
        )

    def test_reach_column_properties(self, meta: pd.DataFrame) -> None:
        """
        'reach' is a real audience-size metric, NOT entirely zero.
        31.8% of rows are zero (2 of 16 campaigns have all-zero reach);
        the remaining 68.2% are meaningful non-negative values.
        """
        assert (meta["reach"] >= 0).all(), "'reach' must be non-negative"
        pct_zero = (meta["reach"] == 0.0).mean()
        assert 0.25 < pct_zero < 0.40, (
            f"Expected ~31.8% zeros in 'reach' (forensic contract); got {pct_zero:.1%}."
        )
        assert meta["reach"].max() > 50_000, (
            f"Expected max reach > 50,000; got {meta['reach'].max()}"
        )

    def test_spend_non_negative(self, meta: pd.DataFrame) -> None:
        assert (meta["spend"] >= 0).all()

    def test_budget_positive_where_not_null(self, meta: pd.DataFrame) -> None:
        non_null = meta["daily_budget"].dropna()
        assert (non_null > 0).all()

    def test_no_nulls_outside_daily_budget(self, meta: pd.DataFrame) -> None:
        """Only daily_budget should contain NULLs."""
        null_cols = {
            col: meta[col].isnull().sum()
            for col in meta.columns
            if col != "daily_budget" and meta[col].isnull().any()
        }
        assert not null_cols, (
            f"Unexpected NULLs in Meta columns: {null_cols}"
        )


# ---------------------------------------------------------------------------
# TestBingSchema
# ---------------------------------------------------------------------------

class TestBingSchema:
    """Column-level schema tests for the Bing / Microsoft Ads DataFrame."""

    def test_shape(self, bing: pd.DataFrame) -> None:
        assert bing.shape == (BING_ROWS, BING_COLS), (
            f"Expected shape {(BING_ROWS, BING_COLS)}; got {bing.shape}."
        )

    def test_exact_column_names(self, bing: pd.DataFrame) -> None:
        """Bing uses PascalCase column names — normalisation is harmonizer's job."""
        assert set(bing.columns) == BING_COLUMNS, (
            f"Column mismatch.\n"
            f"  Missing : {BING_COLUMNS - set(bing.columns)}\n"
            f"  Extra   : {set(bing.columns) - BING_COLUMNS}"
        )

    def test_index_artifact_absent(self, bing: pd.DataFrame) -> None:
        assert "Unnamed: 0" not in bing.columns

    def test_date_column_is_datetime64(self, bing: pd.DataFrame) -> None:
        assert pd.api.types.is_datetime64_any_dtype(bing["TimePeriod"]), (
            f"TimePeriod dtype should be datetime64; got {bing['TimePeriod'].dtype}."
        )

    def test_campaign_id_is_int64(self, bing: pd.DataFrame) -> None:
        assert bing["CampaignId"].dtype == "int64"

    def test_clicks_is_int64(self, bing: pd.DataFrame) -> None:
        """Bing reports Clicks as int, unlike Meta which uses float."""
        assert bing["Clicks"].dtype == "int64"

    def test_impressions_is_int64(self, bing: pd.DataFrame) -> None:
        assert bing["Impressions"].dtype == "int64"

    def test_revenue_is_float64(self, bing: pd.DataFrame) -> None:
        assert bing["Revenue"].dtype == "float64"

    def test_spend_is_float64(self, bing: pd.DataFrame) -> None:
        assert bing["Spend"].dtype == "float64"

    def test_conversions_is_float64(self, bing: pd.DataFrame) -> None:
        assert bing["Conversions"].dtype == "float64"

    def test_daily_budget_is_float64(self, bing: pd.DataFrame) -> None:
        assert bing["DailyBudget"].dtype == "float64"

    def test_campaign_type_is_object(self, bing: pd.DataFrame) -> None:
        assert bing["CampaignType"].dtype == "object"


# ---------------------------------------------------------------------------
# TestBingDataIntegrity
# ---------------------------------------------------------------------------

class TestBingDataIntegrity:
    """Row-level integrity and value-constraint tests for Bing data."""

    def test_campaign_count(self, bing: pd.DataFrame) -> None:
        n = bing["CampaignId"].nunique()
        assert n == BING_CAMPAIGNS, (
            f"Expected {BING_CAMPAIGNS} unique Bing campaigns; found {n}."
        )

    def test_date_range_min(self, bing: pd.DataFrame) -> None:
        actual_min = bing["TimePeriod"].min().date()
        assert actual_min == BING_DATE_MIN, (
            f"Bing min date: expected {BING_DATE_MIN}, got {actual_min}."
        )

    def test_date_range_max(self, bing: pd.DataFrame) -> None:
        actual_max = bing["TimePeriod"].max().date()
        assert actual_max == BING_DATE_MAX, (
            f"Bing max date: expected {BING_DATE_MAX}, got {actual_max}."
        )

    def test_no_duplicate_campaign_date_keys(self, bing: pd.DataFrame) -> None:
        """Primary key (CampaignId, TimePeriod) must be unique."""
        dupes = bing.duplicated(subset=["CampaignId", "TimePeriod"]).sum()
        assert dupes == 0, (
            f"Found {dupes} duplicate (CampaignId, TimePeriod) rows in Bing data."
        )

    def test_campaign_id_to_name_is_injective(self, bing: pd.DataFrame) -> None:
        """Each CampaignId maps to exactly one CampaignName."""
        max_names = bing.groupby("CampaignId")["CampaignName"].nunique().max()
        assert max_names == 1, (
            f"Some Bing CampaignId maps to more than one name (max {max_names})."
        )

    def test_campaign_name_to_id_is_injective(self, bing: pd.DataFrame) -> None:
        """Each CampaignName maps to exactly one CampaignId."""
        max_ids = bing.groupby("CampaignName")["CampaignId"].nunique().max()
        assert max_ids == 1, (
            f"Some Bing CampaignName maps to more than one ID (max {max_ids})."
        )

    def test_zero_nulls_total(self, bing: pd.DataFrame) -> None:
        """Bing has no NULL values in any column (confirmed in forensics)."""
        total_nulls = bing.isnull().sum().sum()
        assert total_nulls == BING_TOTAL_NULLS, (
            f"Expected {BING_TOTAL_NULLS} total NULLs in Bing; found {total_nulls}. "
            f"Affected columns: {bing.columns[bing.isnull().any()].tolist()}"
        )

    def test_daily_budget_min(self, bing: pd.DataFrame) -> None:
        """Minimum Bing DailyBudget is 10.0 (forensically confirmed)."""
        assert bing["DailyBudget"].min() == BING_BUDGET_MIN, (
            f"Expected DailyBudget min = {BING_BUDGET_MIN}; "
            f"got {bing['DailyBudget'].min()}."
        )

    def test_daily_budget_max(self, bing: pd.DataFrame) -> None:
        """Maximum Bing DailyBudget is 20.0 (forensically confirmed)."""
        assert bing["DailyBudget"].max() == BING_BUDGET_MAX, (
            f"Expected DailyBudget max = {BING_BUDGET_MAX}; "
            f"got {bing['DailyBudget'].max()}."
        )

    def test_budget_constant_per_campaign(self, bing: pd.DataFrame) -> None:
        """
        Each Bing campaign must have exactly one distinct DailyBudget value.

        This is a critical data contract: budgets are constant per campaign
        across all three platforms.  The budget simulation layer depends on
        this invariant — time-varying budgets would invalidate the saturation
        response curves.
        """
        distinct_budgets = bing.groupby("CampaignId")["DailyBudget"].nunique()
        non_constant = distinct_budgets[distinct_budgets > 1]
        assert non_constant.empty, (
            f"Bing campaigns with >1 distinct budget value: "
            f"{non_constant.to_dict()}. "
            "Budget constancy invariant violated."
        )

    def test_zero_inflation_revenue_median_is_zero(self, bing: pd.DataFrame) -> None:
        """
        Bing Revenue is heavily zero-inflated: median = 0.
        This confirms the hurdle model design for Bing is necessary.
        """
        assert bing["Revenue"].median() == 0.0, (
            f"Expected Bing Revenue median = 0.0; got {bing['Revenue'].median()}. "
            "If the median is now non-zero, the zero-inflation may have changed "
            "and the hurdle model justification should be re-evaluated."
        )

    def test_zero_spend_row_count(self, bing: pd.DataFrame) -> None:
        """Exactly 915 Bing rows have Spend = 0 (forensically confirmed)."""
        zero_spend = (bing["Spend"] == 0).sum()
        assert zero_spend == BING_ZERO_SPEND_ROWS, (
            f"Expected {BING_ZERO_SPEND_ROWS} zero-spend Bing rows; found {zero_spend}."
        )

    def test_campaign_type_values_are_known(self, bing: pd.DataFrame) -> None:
        """All CampaignType values must belong to the forensically-confirmed set."""
        observed = set(bing["CampaignType"].dropna().unique())
        unexpected = observed - BING_CAMPAIGN_TYPES
        assert not unexpected, (
            f"Unknown Bing CampaignType values: {unexpected}. "
            f"Known types: {BING_CAMPAIGN_TYPES}"
        )

    def test_revenue_non_negative(self, bing: pd.DataFrame) -> None:
        assert (bing["Revenue"] >= 0).all()

    def test_spend_non_negative(self, bing: pd.DataFrame) -> None:
        assert (bing["Spend"] >= 0).all()

    def test_no_revenue_without_spend(self, bing: pd.DataFrame) -> None:
        """
        Rows where Spend = 0 must also have Revenue = 0.
        A non-zero Revenue with zero Spend would indicate an attribution
        anomaly that could silently corrupt the hurdle model training.
        """
        zero_spend_nonzero_revenue = (
            (bing["Spend"] == 0) & (bing["Revenue"] > 0)
        ).sum()
        assert zero_spend_nonzero_revenue == 0, (
            f"Found {zero_spend_nonzero_revenue} Bing rows with "
            "Spend = 0 but Revenue > 0. This is an attribution anomaly."
        )


# ---------------------------------------------------------------------------
# TestCrossDataset
# ---------------------------------------------------------------------------

class TestCrossDataset:
    """Tests for relationships and overlap between the three platform files."""

    def test_total_rows_across_all_platforms(self, dataset: RawDataset) -> None:
        total = len(dataset.google) + len(dataset.meta) + len(dataset.bing)
        assert total == TOTAL_ROWS, (
            f"Expected {TOTAL_ROWS} total rows; got {total}."
        )

    def test_no_campaign_id_overlap_google_and_meta(
        self, google: pd.DataFrame, meta: pd.DataFrame
    ) -> None:
        """
        Google and Meta campaign IDs must be disjoint.
        They are issued by different platforms with different ID namespaces.
        """
        shared = set(google["campaign_id"].unique()) & set(meta["campaign_id"].unique())
        assert not shared, (
            f"Unexpected campaign_id overlap between Google and Meta: {shared}"
        )

    def test_no_campaign_id_overlap_google_and_bing(
        self, google: pd.DataFrame, bing: pd.DataFrame
    ) -> None:
        """Google and Bing campaign IDs must be disjoint."""
        shared = set(google["campaign_id"].unique()) & set(bing["CampaignId"].unique())
        assert not shared, (
            f"Unexpected campaign_id overlap between Google and Bing: {shared}"
        )

    def test_no_campaign_id_overlap_meta_and_bing(
        self, meta: pd.DataFrame, bing: pd.DataFrame
    ) -> None:
        """Meta and Bing campaign IDs must be disjoint."""
        shared = set(meta["campaign_id"].unique()) & set(bing["CampaignId"].unique())
        assert not shared, (
            f"Unexpected campaign_id overlap between Meta and Bing: {shared}"
        )

    def test_google_bing_shared_campaign_name_count(
        self, google: pd.DataFrame, bing: pd.DataFrame
    ) -> None:
        """
        Exactly 27 campaign names appear in both Google and Bing.

        These are the same campaign strategies mirrored across two search
        engines.  The taxonomy parser uses this to construct cross-engine
        pairs.  A count change means campaigns were added, removed, or renamed.
        """
        google_names = set(google["campaign_name"].unique())
        bing_names   = set(bing["CampaignName"].unique())
        shared = google_names & bing_names
        assert len(shared) == GOOGLE_BING_SHARED_NAME_COUNT, (
            f"Expected {GOOGLE_BING_SHARED_NAME_COUNT} shared Google↔Bing "
            f"campaign names; found {len(shared)}.\n"
            f"Shared names: {sorted(shared)}"
        )

    def test_meta_shares_no_names_with_google(
        self, google: pd.DataFrame, meta: pd.DataFrame
    ) -> None:
        """
        Meta uses a different naming taxonomy (Prospecting/Remarketing/DPA)
        and shares no campaign names with Google.
        """
        google_names = set(google["campaign_name"].unique())
        meta_names   = set(meta["campaign_name"].unique())
        shared = google_names & meta_names
        assert not shared, (
            f"Unexpected name overlap between Google and Meta: {shared}"
        )

    def test_meta_shares_no_names_with_bing(
        self, meta: pd.DataFrame, bing: pd.DataFrame
    ) -> None:
        """Meta shares no campaign names with Bing (confirmed in forensics)."""
        meta_names = set(meta["campaign_name"].unique())
        bing_names = set(bing["CampaignName"].unique())
        shared = meta_names & bing_names
        assert not shared, (
            f"Unexpected name overlap between Meta and Bing: {shared}"
        )

    def test_date_ranges_overlap_across_platforms(
        self, google: pd.DataFrame, meta: pd.DataFrame, bing: pd.DataFrame
    ) -> None:
        """
        All three platforms must have overlapping date ranges — otherwise
        cross-platform aggregate features (total system spend/revenue) cannot
        be computed for a common window.
        """
        google_min = google["segments_date"].min()
        google_max = google["segments_date"].max()
        meta_min   = meta["date_start"].min()
        meta_max   = meta["date_start"].max()
        bing_min   = bing["TimePeriod"].min()
        bing_max   = bing["TimePeriod"].max()

        overlap_start = max(google_min, meta_min, bing_min)
        overlap_end   = min(google_max, meta_max, bing_max)

        assert overlap_start < overlap_end, (
            f"Platforms have no overlapping date window.\n"
            f"  Google: {google_min.date()} to {google_max.date()}\n"
            f"  Meta  : {meta_min.date()} to {meta_max.date()}\n"
            f"  Bing  : {bing_min.date()} to {bing_max.date()}"
        )

    def test_google_budget_constant_per_campaign(
        self, google: pd.DataFrame
    ) -> None:
        """
        Each Google campaign must have exactly one distinct budget value.
        Mirrors the Bing budget-constancy test — this invariant holds for
        all three platforms and is the foundation of the budget simulation layer.
        """
        distinct = google.groupby("campaign_id")["campaign_budget_amount"].nunique()
        # Campaigns with all-NULL budgets return nunique=0; treat those as constant.
        violations = distinct[distinct > 1]
        assert violations.empty, (
            f"Google campaigns with >1 distinct budget: {violations.to_dict()}."
        )

    def test_meta_budget_constant_per_campaign(
        self, meta: pd.DataFrame
    ) -> None:
        """Each Meta campaign must have exactly one distinct daily_budget value."""
        distinct = meta.groupby("campaign_id")["daily_budget"].nunique()
        violations = distinct[distinct > 1]
        assert violations.empty, (
            f"Meta campaigns with >1 distinct budget: {violations.to_dict()}."
        )
