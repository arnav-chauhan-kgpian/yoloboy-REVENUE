"""
tests/test_taxonomy_parser.py
=============================
Pytest suite for src/data/taxonomy_parser.py.

Coverage
--------
- _parse_google_bing_name: all format/audience variants, edge cases
- _parse_meta_name: all 16 actual Meta campaign names
- _make_strategy_key: space normalisation, None handling, all platforms
- parse_taxonomy: schema, row count, dtypes
- Google/Bing fields: format, audience_strategy, campaign_number
- Meta fields: funnel_stage, ad_product_type, campaign_number
- strategy_key: correctness for representative campaigns
- cross_engine_pair_flag: 27 shared names, 54 flagged rows, Meta=0
- Boolean flags: is_brand (16), is_non_brand (103), is_upper_funnel (10)
- TaxonomyParseError: duplicate campaign_id and campaign_name
- _validate_metadata directly
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import pytest

from src.data.harmonizer import harmonize
from src.data.loader import load_raw_data
from src.data.taxonomy_parser import (
    CAMPAIGN_METADATA_COLUMNS,
    GOOGLE_BING_PLATFORMS,
    META_PLATFORM,
    TaxonomyParseError,
    _make_strategy_key,
    _parse_google_bing_name,
    _parse_meta_name,
    _validate_metadata,
    parse_taxonomy,
)

# ---------------------------------------------------------------------------
# Forensic constants — data contracts
# ---------------------------------------------------------------------------

TOTAL_CAMPAIGNS: Final[int] = 136
GOOGLE_CAMPAIGNS: Final[int] = 92
META_CAMPAIGNS: Final[int] = 16
BING_CAMPAIGNS: Final[int] = 28
METADATA_COL_COUNT: Final[int] = 14

# cross_engine_pair_flag counts (27 shared names × 2 platforms)
CROSS_ENGINE_SHARED_NAMES: Final[int] = 27
CROSS_ENGINE_FLAGGED_ROWS: Final[int] = 54   # 27 in Google + 27 in Bing

# Boolean flag totals (derived from exhaustive name analysis)
BRAND_CAMPAIGNS: Final[int]       = 16   # 5G TM + 5B TM + 6M Brand
NON_BRAND_CAMPAIGNS: Final[int]   = 103  # 72G NTM + 23B NTM + 8M DPA/Adv+
UPPER_FUNNEL_CAMPAIGNS: Final[int] = 10  # 8G (Video+Display+DemandGen) + 2B (DemandGen)

# Google TM campaigns (all SEARCH type)
GOOGLE_TM_COUNT: Final[int] = 5
BING_TM_COUNT: Final[int]   = 5

# Meta brand/non-brand/neither breakdown
META_BRAND_COUNT: Final[int]      = 6   # Generic_Brand + 2×Prospecting_Brand + 3×Remarketing_Brand
META_NON_BRAND_COUNT: Final[int]  = 8   # 2×Adv_Plus + 3×DPA_Prospecting + 3×DPA_Remarketing
META_NEITHER_COUNT: Final[int]    = 2   # Generic_Campaign_01 / 02


# ---------------------------------------------------------------------------
# Module-scope fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data_dir() -> Path:
    return Path("dataset")


@pytest.fixture(scope="module")
def campaign_meta(data_dir: Path) -> pd.DataFrame:
    canon = harmonize(load_raw_data(data_dir))
    return parse_taxonomy(canon)


@pytest.fixture(scope="module")
def google_meta(campaign_meta: pd.DataFrame) -> pd.DataFrame:
    return campaign_meta[campaign_meta["platform"] == "google"].copy()


@pytest.fixture(scope="module")
def meta_meta(campaign_meta: pd.DataFrame) -> pd.DataFrame:
    return campaign_meta[campaign_meta["platform"] == "meta"].copy()


@pytest.fixture(scope="module")
def bing_meta(campaign_meta: pd.DataFrame) -> pd.DataFrame:
    return campaign_meta[campaign_meta["platform"] == "bing"].copy()


# ---------------------------------------------------------------------------
# TestParseGoogleBingName — unit tests for _parse_google_bing_name
# ---------------------------------------------------------------------------

class TestParseGoogleBingName:
    """Unit tests for the Google/Bing name parser with all observed patterns."""

    # --- NTM campaigns ---

    def test_pmax_ntm_01(self) -> None:
        r = _parse_google_bing_name("Pmax_NTM_Campaign_01")
        assert r["format"] == "Pmax"
        assert r["audience_strategy"] == "NTM"
        assert r["campaign_number"] == 1

    def test_pmax_ntm_49(self) -> None:
        r = _parse_google_bing_name("Pmax_NTM_Campaign_49")
        assert r["format"] == "Pmax"
        assert r["audience_strategy"] == "NTM"
        assert r["campaign_number"] == 49

    def test_search_ntm(self) -> None:
        r = _parse_google_bing_name("Search_NTM_Campaign_17")
        assert r["format"] == "Search"
        assert r["audience_strategy"] == "NTM"
        assert r["campaign_number"] == 17

    def test_shopping_ntm(self) -> None:
        r = _parse_google_bing_name("Shopping_NTM_Campaign_03")
        assert r["format"] == "Shopping"
        assert r["audience_strategy"] == "NTM"
        assert r["campaign_number"] == 3

    def test_demand_gen_ntm_space_in_format(self) -> None:
        r = _parse_google_bing_name("Demand Gen_NTM_Campaign_01")
        assert r["format"] == "Demand Gen"   # space preserved in parsed field
        assert r["audience_strategy"] == "NTM"
        assert r["campaign_number"] == 1

    def test_demand_gen_ntm_02(self) -> None:
        r = _parse_google_bing_name("Demand Gen_NTM_Campaign_02")
        assert r["format"] == "Demand Gen"
        assert r["audience_strategy"] == "NTM"
        assert r["campaign_number"] == 2

    def test_video_ntm(self) -> None:
        r = _parse_google_bing_name("Video_NTM_Campaign_01")
        assert r["format"] == "Video"
        assert r["audience_strategy"] == "NTM"
        assert r["campaign_number"] == 1

    # --- TM campaigns ---

    def test_search_tm_01(self) -> None:
        r = _parse_google_bing_name("Search_TM_Campaign_01")
        assert r["format"] == "Search"
        assert r["audience_strategy"] == "TM"
        assert r["campaign_number"] == 1

    def test_search_tm_05(self) -> None:
        r = _parse_google_bing_name("Search_TM_Campaign_05")
        assert r["format"] == "Search"
        assert r["audience_strategy"] == "TM"
        assert r["campaign_number"] == 5

    def test_search_tm_06_bing_only(self) -> None:
        r = _parse_google_bing_name("Search_TM_Campaign_06")
        assert r["audience_strategy"] == "TM"
        assert r["campaign_number"] == 6

    # --- Campaigns without audience tag ---

    def test_pmax_no_audience(self) -> None:
        r = _parse_google_bing_name("Pmax_Campaign_01")
        assert r["format"] == "Pmax"
        assert r["audience_strategy"] is None
        assert r["campaign_number"] == 1

    def test_display_no_audience(self) -> None:
        r = _parse_google_bing_name("Display_Campaign_01")
        assert r["format"] == "Display"
        assert r["audience_strategy"] is None
        assert r["campaign_number"] == 1

    def test_video_no_audience(self) -> None:
        r = _parse_google_bing_name("Video_Campaign_03")
        assert r["format"] == "Video"
        assert r["audience_strategy"] is None
        assert r["campaign_number"] == 3

    def test_search_no_audience(self) -> None:
        r = _parse_google_bing_name("Search_Campaign_01")
        assert r["format"] == "Search"
        assert r["audience_strategy"] is None
        assert r["campaign_number"] == 1

    # --- Type assertions ---

    def test_campaign_number_is_int(self) -> None:
        r = _parse_google_bing_name("Pmax_NTM_Campaign_07")
        assert isinstance(r["campaign_number"], int)

    def test_all_keys_present(self) -> None:
        r = _parse_google_bing_name("Pmax_NTM_Campaign_01")
        assert set(r.keys()) == {"format", "audience_strategy", "campaign_number"}

    # --- Edge case: name without _Campaign_ token ---

    def test_malformed_name_returns_gracefully(self) -> None:
        r = _parse_google_bing_name("Malformed_Name")
        assert r["format"] == "Malformed_Name"
        assert r["audience_strategy"] is None
        assert r["campaign_number"] is None


# ---------------------------------------------------------------------------
# TestParseMetaName — unit tests for _parse_meta_name (all 16 campaigns)
# ---------------------------------------------------------------------------

class TestParseMetaName:
    """Unit tests for Meta name parser against the complete set of 16 campaigns."""

    def _parse(self, name: str) -> dict:
        return _parse_meta_name(name)

    # --- Generic ---

    def test_generic_campaign_01_no_product(self) -> None:
        r = self._parse("Generic_Campaign_01")
        assert r["funnel_stage"] == "Generic"
        assert r["ad_product_type"] is None
        assert r["campaign_number"] == 1

    def test_generic_campaign_02_no_product(self) -> None:
        r = self._parse("Generic_Campaign_02")
        assert r["funnel_stage"] == "Generic"
        assert r["ad_product_type"] is None
        assert r["campaign_number"] == 2

    def test_generic_brand_campaign(self) -> None:
        r = self._parse("Generic_Brand_Campaign_01")
        assert r["funnel_stage"] == "Generic"
        assert r["ad_product_type"] == "Brand"
        assert r["campaign_number"] == 1

    # --- Prospecting ---

    def test_prospecting_adv_plus_01(self) -> None:
        r = self._parse("Prospecting_Adv_Plus_Campaign_01")
        assert r["funnel_stage"] == "Prospecting"
        assert r["ad_product_type"] == "Adv_Plus"
        assert r["campaign_number"] == 1

    def test_prospecting_adv_plus_02(self) -> None:
        r = self._parse("Prospecting_Adv_Plus_Campaign_02")
        assert r["funnel_stage"] == "Prospecting"
        assert r["ad_product_type"] == "Adv_Plus"
        assert r["campaign_number"] == 2

    def test_prospecting_brand_01(self) -> None:
        r = self._parse("Prospecting_Brand_Campaign_01")
        assert r["funnel_stage"] == "Prospecting"
        assert r["ad_product_type"] == "Brand"
        assert r["campaign_number"] == 1

    def test_prospecting_brand_02(self) -> None:
        r = self._parse("Prospecting_Brand_Campaign_02")
        assert r["funnel_stage"] == "Prospecting"
        assert r["ad_product_type"] == "Brand"
        assert r["campaign_number"] == 2

    def test_prospecting_dpa_01(self) -> None:
        r = self._parse("Prospecting_DPA_Campaign_01")
        assert r["funnel_stage"] == "Prospecting"
        assert r["ad_product_type"] == "DPA"
        assert r["campaign_number"] == 1

    def test_prospecting_dpa_02(self) -> None:
        r = self._parse("Prospecting_DPA_Campaign_02")
        assert r["funnel_stage"] == "Prospecting"
        assert r["ad_product_type"] == "DPA"

    def test_prospecting_dpa_04(self) -> None:
        r = self._parse("Prospecting_DPA_Campaign_04")
        assert r["funnel_stage"] == "Prospecting"
        assert r["ad_product_type"] == "DPA"
        assert r["campaign_number"] == 4

    # --- Remarketing ---

    def test_remarketing_brand_01(self) -> None:
        r = self._parse("Remarketing_Brand_Campaign_01")
        assert r["funnel_stage"] == "Remarketing"
        assert r["ad_product_type"] == "Brand"
        assert r["campaign_number"] == 1

    def test_remarketing_brand_02(self) -> None:
        r = self._parse("Remarketing_Brand_Campaign_02")
        assert r["funnel_stage"] == "Remarketing"
        assert r["ad_product_type"] == "Brand"

    def test_remarketing_brand_03(self) -> None:
        r = self._parse("Remarketing_Brand_Campaign_03")
        assert r["funnel_stage"] == "Remarketing"
        assert r["ad_product_type"] == "Brand"
        assert r["campaign_number"] == 3

    def test_remarketing_dpa_01(self) -> None:
        r = self._parse("Remarketing_DPA_Campaign_01")
        assert r["funnel_stage"] == "Remarketing"
        assert r["ad_product_type"] == "DPA"
        assert r["campaign_number"] == 1

    def test_remarketing_dpa_02(self) -> None:
        r = self._parse("Remarketing_DPA_Campaign_02")
        assert r["funnel_stage"] == "Remarketing"
        assert r["ad_product_type"] == "DPA"

    def test_remarketing_dpa_03(self) -> None:
        r = self._parse("Remarketing_DPA_Campaign_03")
        assert r["funnel_stage"] == "Remarketing"
        assert r["ad_product_type"] == "DPA"
        assert r["campaign_number"] == 3

    # --- Type / structure ---

    def test_campaign_number_is_int(self) -> None:
        r = self._parse("Prospecting_DPA_Campaign_01")
        assert isinstance(r["campaign_number"], int)

    def test_all_keys_present(self) -> None:
        r = self._parse("Generic_Campaign_01")
        assert set(r.keys()) == {"funnel_stage", "ad_product_type", "campaign_number"}

    def test_malformed_name_returns_gracefully(self) -> None:
        r = self._parse("NoUnderscoreCampaignToken")
        assert r["funnel_stage"] is None
        assert r["ad_product_type"] is None
        assert r["campaign_number"] is None

    def test_bulk_all_16_campaigns(self) -> None:
        """Validate all 16 Meta campaign names in one assertion block."""
        cases = [
            ("Generic_Brand_Campaign_01",         "Generic",     "Brand",    1),
            ("Generic_Campaign_01",               "Generic",     None,       1),
            ("Generic_Campaign_02",               "Generic",     None,       2),
            ("Prospecting_Adv_Plus_Campaign_01",  "Prospecting", "Adv_Plus", 1),
            ("Prospecting_Adv_Plus_Campaign_02",  "Prospecting", "Adv_Plus", 2),
            ("Prospecting_Brand_Campaign_01",     "Prospecting", "Brand",    1),
            ("Prospecting_Brand_Campaign_02",     "Prospecting", "Brand",    2),
            ("Prospecting_DPA_Campaign_01",       "Prospecting", "DPA",      1),
            ("Prospecting_DPA_Campaign_02",       "Prospecting", "DPA",      2),
            ("Prospecting_DPA_Campaign_04",       "Prospecting", "DPA",      4),
            ("Remarketing_Brand_Campaign_01",     "Remarketing", "Brand",    1),
            ("Remarketing_Brand_Campaign_02",     "Remarketing", "Brand",    2),
            ("Remarketing_Brand_Campaign_03",     "Remarketing", "Brand",    3),
            ("Remarketing_DPA_Campaign_01",       "Remarketing", "DPA",      1),
            ("Remarketing_DPA_Campaign_02",       "Remarketing", "DPA",      2),
            ("Remarketing_DPA_Campaign_03",       "Remarketing", "DPA",      3),
        ]
        failures: list[str] = []
        for name, exp_funnel, exp_product, exp_num in cases:
            r = _parse_meta_name(name)
            if r["funnel_stage"] != exp_funnel:
                failures.append(f"{name}: funnel={r['funnel_stage']!r} != {exp_funnel!r}")
            if r["ad_product_type"] != exp_product:
                failures.append(f"{name}: product={r['ad_product_type']!r} != {exp_product!r}")
            if r["campaign_number"] != exp_num:
                failures.append(f"{name}: number={r['campaign_number']} != {exp_num}")
        assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# TestMakeStrategyKey — unit tests for _make_strategy_key
# ---------------------------------------------------------------------------

class TestMakeStrategyKey:
    """Unit tests for strategy_key construction."""

    def test_google_pmax_ntm(self) -> None:
        assert _make_strategy_key("google", "Pmax", "NTM") == "google_Pmax_NTM"

    def test_google_search_tm(self) -> None:
        assert _make_strategy_key("google", "Search", "TM") == "google_Search_TM"

    def test_google_shopping_ntm(self) -> None:
        assert _make_strategy_key("google", "Shopping", "NTM") == "google_Shopping_NTM"

    def test_bing_pmax_ntm(self) -> None:
        assert _make_strategy_key("bing", "Pmax", "NTM") == "bing_Pmax_NTM"

    def test_bing_search_tm(self) -> None:
        assert _make_strategy_key("bing", "Search", "TM") == "bing_Search_TM"

    def test_demand_gen_space_replaced_with_underscore(self) -> None:
        assert _make_strategy_key("google", "Demand Gen", "NTM") == "google_Demand_Gen_NTM"
        assert _make_strategy_key("bing", "Demand Gen", "NTM") == "bing_Demand_Gen_NTM"

    def test_meta_prospecting_dpa(self) -> None:
        assert _make_strategy_key("meta", "Prospecting", "DPA") == "meta_Prospecting_DPA"

    def test_meta_remarketing_brand(self) -> None:
        assert _make_strategy_key("meta", "Remarketing", "Brand") == "meta_Remarketing_Brand"

    def test_meta_prospecting_adv_plus(self) -> None:
        assert _make_strategy_key("meta", "Prospecting", "Adv_Plus") == "meta_Prospecting_Adv_Plus"

    def test_none_part2_omitted(self) -> None:
        assert _make_strategy_key("google", "Pmax", None) == "google_Pmax"
        assert _make_strategy_key("meta", "Generic", None) == "meta_Generic"

    def test_both_none_gives_platform_only(self) -> None:
        assert _make_strategy_key("google", None, None) == "google"

    def test_none_part1_skipped(self) -> None:
        # part1=None but part2 present: unusual but should not crash
        result = _make_strategy_key("google", None, "NTM")
        assert result == "google_NTM"

    def test_no_spaces_in_output(self) -> None:
        key = _make_strategy_key("bing", "Demand Gen", "NTM")
        assert " " not in key


# ---------------------------------------------------------------------------
# TestCampaignMetadataSchema
# ---------------------------------------------------------------------------

class TestCampaignMetadataSchema:
    """Top-level shape, columns, and dtype contracts."""

    def test_total_row_count(self, campaign_meta: pd.DataFrame) -> None:
        assert len(campaign_meta) == TOTAL_CAMPAIGNS

    def test_column_count(self, campaign_meta: pd.DataFrame) -> None:
        assert campaign_meta.shape[1] == METADATA_COL_COUNT

    def test_column_names_match_canonical(self, campaign_meta: pd.DataFrame) -> None:
        assert set(campaign_meta.columns) == set(CAMPAIGN_METADATA_COLUMNS)

    def test_column_order_matches_canonical(self, campaign_meta: pd.DataFrame) -> None:
        assert list(campaign_meta.columns) == list(CAMPAIGN_METADATA_COLUMNS)

    def test_platform_values(self, campaign_meta: pd.DataFrame) -> None:
        assert set(campaign_meta["platform"].unique()) == {"google", "meta", "bing"}

    def test_campaign_id_unique_within_platform(self, campaign_meta: pd.DataFrame) -> None:
        dup = campaign_meta.duplicated(subset=["platform", "campaign_id"])
        assert not dup.any()

    def test_campaign_name_unique_within_platform(self, campaign_meta: pd.DataFrame) -> None:
        dup = campaign_meta.duplicated(subset=["platform", "campaign_name"])
        assert not dup.any()

    def test_strategy_key_no_spaces(self, campaign_meta: pd.DataFrame) -> None:
        has_space = campaign_meta["strategy_key"].str.contains(" ")
        assert not has_space.any(), "strategy_key values must not contain spaces"

    def test_strategy_key_starts_with_platform(self, campaign_meta: pd.DataFrame) -> None:
        for _, row in campaign_meta.iterrows():
            assert row["strategy_key"].startswith(row["platform"]), (
                f"strategy_key {row['strategy_key']!r} does not start with platform {row['platform']!r}"
            )

    # Dtypes
    def test_campaign_id_is_int64(self, campaign_meta: pd.DataFrame) -> None:
        import numpy as np
        assert campaign_meta["campaign_id"].dtype == np.dtype("int64")

    def test_campaign_number_is_nullable_int(self, campaign_meta: pd.DataFrame) -> None:
        assert campaign_meta["campaign_number"].dtype == pd.Int64Dtype()

    def test_bool_flags_are_bool(self, campaign_meta: pd.DataFrame) -> None:
        import numpy as np
        for col in ("cross_engine_pair_flag", "is_brand", "is_non_brand", "is_upper_funnel"):
            assert campaign_meta[col].dtype == np.dtype("bool"), (
                f"{col} should be bool, got {campaign_meta[col].dtype}"
            )


# ---------------------------------------------------------------------------
# TestGoogleBingCampaigns
# ---------------------------------------------------------------------------

class TestGoogleBingCampaigns:
    """Campaign count and parsed field tests for Google and Bing."""

    def test_google_campaign_count(self, google_meta: pd.DataFrame) -> None:
        assert len(google_meta) == GOOGLE_CAMPAIGNS

    def test_bing_campaign_count(self, bing_meta: pd.DataFrame) -> None:
        assert len(bing_meta) == BING_CAMPAIGNS

    def test_google_format_is_always_populated(self, google_meta: pd.DataFrame) -> None:
        assert google_meta["format"].notna().all()

    def test_bing_format_is_always_populated(self, bing_meta: pd.DataFrame) -> None:
        assert bing_meta["format"].notna().all()

    def test_google_funnel_stage_is_all_null(self, google_meta: pd.DataFrame) -> None:
        assert google_meta["funnel_stage"].isna().all()

    def test_google_ad_product_type_is_all_null(self, google_meta: pd.DataFrame) -> None:
        assert google_meta["ad_product_type"].isna().all()

    def test_bing_funnel_stage_is_all_null(self, bing_meta: pd.DataFrame) -> None:
        assert bing_meta["funnel_stage"].isna().all()

    def test_google_formats_are_known(self, google_meta: pd.DataFrame) -> None:
        known_formats = {"Pmax", "Search", "Shopping", "Display", "Video", "Demand Gen"}
        assert set(google_meta["format"].dropna().unique()).issubset(known_formats)

    def test_bing_formats_are_known(self, bing_meta: pd.DataFrame) -> None:
        known_formats = {"Pmax", "Search", "Shopping", "Demand Gen"}
        assert set(bing_meta["format"].dropna().unique()).issubset(known_formats)

    def test_audience_strategy_values(self, google_meta: pd.DataFrame) -> None:
        valid = {"TM", "NTM", None}
        actual = set(google_meta["audience_strategy"].unique())
        assert actual.issubset(valid | {float("nan")}), f"Unexpected values: {actual - valid}"

    def test_google_campaign_number_range(self, google_meta: pd.DataFrame) -> None:
        nums = google_meta["campaign_number"].dropna()
        assert (nums >= 1).all()
        assert (nums <= 49).all()  # Pmax_NTM_Campaign_49 is the highest observed

    def test_bing_campaign_number_range(self, bing_meta: pd.DataFrame) -> None:
        nums = bing_meta["campaign_number"].dropna()
        assert (nums >= 1).all()
        assert (nums <= 19).all()  # Pmax_NTM_Campaign_19 is highest in Bing

    def test_google_pmax_ntm_strategy_key_format(self, google_meta: pd.DataFrame) -> None:
        pmax_ntm = google_meta[google_meta["campaign_name"] == "Pmax_NTM_Campaign_01"]
        assert len(pmax_ntm) == 1
        assert pmax_ntm["strategy_key"].iloc[0] == "google_Pmax_NTM"

    def test_google_search_tm_strategy_key_format(self, google_meta: pd.DataFrame) -> None:
        search_tm = google_meta[google_meta["campaign_name"] == "Search_TM_Campaign_01"]
        assert len(search_tm) == 1
        assert search_tm["strategy_key"].iloc[0] == "google_Search_TM"

    def test_bing_demand_gen_strategy_key_no_space(self, bing_meta: pd.DataFrame) -> None:
        dg = bing_meta[bing_meta["campaign_name"] == "Demand Gen_NTM_Campaign_01"]
        assert len(dg) == 1
        assert dg["strategy_key"].iloc[0] == "bing_Demand_Gen_NTM"

    def test_google_untagged_pmax_strategy_key(self, google_meta: pd.DataFrame) -> None:
        pmax = google_meta[google_meta["campaign_name"] == "Pmax_Campaign_01"]
        assert len(pmax) == 1
        assert pmax["strategy_key"].iloc[0] == "google_Pmax"

    def test_google_brand_tm_count(self, google_meta: pd.DataFrame) -> None:
        assert google_meta["is_brand"].sum() == GOOGLE_TM_COUNT

    def test_bing_brand_tm_count(self, bing_meta: pd.DataFrame) -> None:
        assert bing_meta["is_brand"].sum() == BING_TM_COUNT


# ---------------------------------------------------------------------------
# TestMetaCampaigns
# ---------------------------------------------------------------------------

class TestMetaCampaigns:
    """Parsed field tests for Meta."""

    def test_meta_campaign_count(self, meta_meta: pd.DataFrame) -> None:
        assert len(meta_meta) == META_CAMPAIGNS

    def test_meta_format_is_all_null(self, meta_meta: pd.DataFrame) -> None:
        assert meta_meta["format"].isna().all()

    def test_meta_audience_strategy_is_all_null(self, meta_meta: pd.DataFrame) -> None:
        assert meta_meta["audience_strategy"].isna().all()

    def test_meta_funnel_stage_always_populated(self, meta_meta: pd.DataFrame) -> None:
        assert meta_meta["funnel_stage"].notna().all()

    def test_meta_funnel_stage_values(self, meta_meta: pd.DataFrame) -> None:
        assert set(meta_meta["funnel_stage"].unique()).issubset({"Generic", "Prospecting", "Remarketing"})

    def test_meta_ad_product_type_values(self, meta_meta: pd.DataFrame) -> None:
        valid = {"Brand", "DPA", "Adv_Plus", None}
        for v in meta_meta["ad_product_type"]:
            assert v in valid, f"Unexpected ad_product_type: {v!r}"

    def test_meta_brand_campaign_count(self, meta_meta: pd.DataFrame) -> None:
        assert meta_meta["is_brand"].sum() == META_BRAND_COUNT

    def test_meta_non_brand_campaign_count(self, meta_meta: pd.DataFrame) -> None:
        assert meta_meta["is_non_brand"].sum() == META_NON_BRAND_COUNT

    def test_meta_neither_campaign_count(self, meta_meta: pd.DataFrame) -> None:
        neither = (~meta_meta["is_brand"]) & (~meta_meta["is_non_brand"])
        assert neither.sum() == META_NEITHER_COUNT

    def test_meta_brand_and_non_brand_mutually_exclusive(self, meta_meta: pd.DataFrame) -> None:
        both = meta_meta["is_brand"] & meta_meta["is_non_brand"]
        assert not both.any(), "A campaign cannot be both brand and non-brand"

    def test_meta_upper_funnel_is_zero(self, meta_meta: pd.DataFrame) -> None:
        assert meta_meta["is_upper_funnel"].sum() == 0

    def test_meta_cross_engine_flag_is_zero(self, meta_meta: pd.DataFrame) -> None:
        assert meta_meta["cross_engine_pair_flag"].sum() == 0

    def test_specific_strategy_keys(self, meta_meta: pd.DataFrame) -> None:
        cases = {
            "Prospecting_DPA_Campaign_01":      "meta_Prospecting_DPA",
            "Remarketing_Brand_Campaign_01":    "meta_Remarketing_Brand",
            "Prospecting_Adv_Plus_Campaign_01": "meta_Prospecting_Adv_Plus",
            "Generic_Campaign_01":              "meta_Generic",
            "Generic_Brand_Campaign_01":        "meta_Generic_Brand",
        }
        for name, expected_key in cases.items():
            row = meta_meta[meta_meta["campaign_name"] == name]
            assert len(row) == 1, f"Campaign {name!r} not found"
            actual = row["strategy_key"].iloc[0]
            assert actual == expected_key, (
                f"{name}: expected strategy_key={expected_key!r}, got={actual!r}"
            )

    def test_meta_campaign_numbers_parsed(self, meta_meta: pd.DataFrame) -> None:
        # All 16 Meta campaigns have a number — none should be NA
        assert meta_meta["campaign_number"].notna().all()


# ---------------------------------------------------------------------------
# TestCrossEnginePairs
# ---------------------------------------------------------------------------

class TestCrossEnginePairs:
    """cross_engine_pair_flag: 27 shared names → 54 flagged rows."""

    def test_total_flagged_rows(self, campaign_meta: pd.DataFrame) -> None:
        assert campaign_meta["cross_engine_pair_flag"].sum() == CROSS_ENGINE_FLAGGED_ROWS

    def test_google_flagged_rows(self, google_meta: pd.DataFrame) -> None:
        assert google_meta["cross_engine_pair_flag"].sum() == CROSS_ENGINE_SHARED_NAMES

    def test_bing_flagged_rows(self, bing_meta: pd.DataFrame) -> None:
        assert bing_meta["cross_engine_pair_flag"].sum() == CROSS_ENGINE_SHARED_NAMES

    def test_meta_flagged_rows_is_zero(self, meta_meta: pd.DataFrame) -> None:
        assert meta_meta["cross_engine_pair_flag"].sum() == 0

    def test_flagged_google_names_all_in_bing(
        self, google_meta: pd.DataFrame, bing_meta: pd.DataFrame
    ) -> None:
        flagged_google = set(google_meta.loc[google_meta["cross_engine_pair_flag"], "campaign_name"])
        all_bing_names = set(bing_meta["campaign_name"])
        missing = flagged_google - all_bing_names
        assert not missing, f"Flagged Google names not in Bing: {missing}"

    def test_flagged_bing_names_all_in_google(
        self, google_meta: pd.DataFrame, bing_meta: pd.DataFrame
    ) -> None:
        flagged_bing = set(bing_meta.loc[bing_meta["cross_engine_pair_flag"], "campaign_name"])
        all_google_names = set(google_meta["campaign_name"])
        missing = flagged_bing - all_google_names
        assert not missing, f"Flagged Bing names not in Google: {missing}"

    def test_pmax_ntm_campaigns_are_cross_engine(
        self, google_meta: pd.DataFrame, bing_meta: pd.DataFrame
    ) -> None:
        # Pmax_NTM_Campaign_01 through 19 are shared — spot-check 01
        for meta_df, platform in [(google_meta, "google"), (bing_meta, "bing")]:
            row = meta_df[meta_df["campaign_name"] == "Pmax_NTM_Campaign_01"]
            assert row["cross_engine_pair_flag"].iloc[0], (
                f"Pmax_NTM_Campaign_01 not flagged in {platform}"
            )

    def test_google_only_campaigns_not_flagged(self, google_meta: pd.DataFrame) -> None:
        # Search_TM_Campaign_01 exists only in Google, not in Bing → not cross-engine
        row = google_meta[google_meta["campaign_name"] == "Search_TM_Campaign_01"]
        assert len(row) == 1
        assert not row["cross_engine_pair_flag"].iloc[0]

    def test_bing_only_campaign_not_flagged(self, bing_meta: pd.DataFrame) -> None:
        # Search_TM_Campaign_06 exists only in Bing → not cross-engine
        row = bing_meta[bing_meta["campaign_name"] == "Search_TM_Campaign_06"]
        assert len(row) == 1
        assert not row["cross_engine_pair_flag"].iloc[0]


# ---------------------------------------------------------------------------
# TestBooleanFlags
# ---------------------------------------------------------------------------

class TestBooleanFlags:
    """is_brand, is_non_brand, is_upper_funnel global totals and logic."""

    def test_total_brand_campaigns(self, campaign_meta: pd.DataFrame) -> None:
        assert campaign_meta["is_brand"].sum() == BRAND_CAMPAIGNS

    def test_total_non_brand_campaigns(self, campaign_meta: pd.DataFrame) -> None:
        assert campaign_meta["is_non_brand"].sum() == NON_BRAND_CAMPAIGNS

    def test_total_upper_funnel_campaigns(self, campaign_meta: pd.DataFrame) -> None:
        assert campaign_meta["is_upper_funnel"].sum() == UPPER_FUNNEL_CAMPAIGNS

    def test_brand_and_non_brand_mutually_exclusive(self, campaign_meta: pd.DataFrame) -> None:
        both = campaign_meta["is_brand"] & campaign_meta["is_non_brand"]
        assert not both.any(), (
            f"{both.sum()} campaigns flagged as both brand and non-brand"
        )

    def test_total_classified_and_unclassified(self, campaign_meta: pd.DataFrame) -> None:
        classified = campaign_meta["is_brand"] | campaign_meta["is_non_brand"]
        unclassified = ~classified
        # 15 Google untagged + 2 Meta generic = 17 unclassified
        assert unclassified.sum() == 17

    def test_upper_funnel_google_count(self, google_meta: pd.DataFrame) -> None:
        # 3 Video_Campaign + 1 Video_NTM + 2 Display + 2 Demand Gen = 8
        assert google_meta["is_upper_funnel"].sum() == 8

    def test_upper_funnel_bing_count(self, bing_meta: pd.DataFrame) -> None:
        # 2 Demand Gen_NTM_Campaign (Audience CampaignType → UpperFunnel)
        assert bing_meta["is_upper_funnel"].sum() == 2

    def test_upper_funnel_meta_is_zero(self, meta_meta: pd.DataFrame) -> None:
        assert meta_meta["is_upper_funnel"].sum() == 0

    def test_tm_campaigns_are_brand_in_google(self, google_meta: pd.DataFrame) -> None:
        tm_rows = google_meta[google_meta["audience_strategy"] == "TM"]
        assert tm_rows["is_brand"].all()
        assert not tm_rows["is_non_brand"].any()

    def test_ntm_campaigns_are_non_brand_in_google(self, google_meta: pd.DataFrame) -> None:
        ntm_rows = google_meta[google_meta["audience_strategy"] == "NTM"]
        assert ntm_rows["is_non_brand"].all()
        assert not ntm_rows["is_brand"].any()

    def test_untagged_google_are_neither(self, google_meta: pd.DataFrame) -> None:
        untagged = google_meta[google_meta["audience_strategy"].isna()]
        assert not untagged["is_brand"].any()
        assert not untagged["is_non_brand"].any()

    def test_meta_dpa_campaigns_are_non_brand(self, meta_meta: pd.DataFrame) -> None:
        dpa = meta_meta[meta_meta["ad_product_type"] == "DPA"]
        assert dpa["is_non_brand"].all()
        assert not dpa["is_brand"].any()

    def test_meta_adv_plus_campaigns_are_non_brand(self, meta_meta: pd.DataFrame) -> None:
        adv = meta_meta[meta_meta["ad_product_type"] == "Adv_Plus"]
        assert adv["is_non_brand"].all()
        assert not adv["is_brand"].any()

    def test_meta_brand_product_campaigns_are_brand(self, meta_meta: pd.DataFrame) -> None:
        brand = meta_meta[meta_meta["ad_product_type"] == "Brand"]
        assert brand["is_brand"].all()
        assert not brand["is_non_brand"].any()


# ---------------------------------------------------------------------------
# TestValidation — TaxonomyParseError on bad input
# ---------------------------------------------------------------------------

class TestValidation:
    """_validate_metadata raises TaxonomyParseError on integrity violations."""

    def _valid_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "platform":     ["google", "google"],
            "campaign_id":  [1, 2],
            "campaign_name": ["Camp_A", "Camp_B"],
        })

    def test_valid_df_does_not_raise(self) -> None:
        _validate_metadata(self._valid_df())

    def test_duplicate_campaign_id_raises(self) -> None:
        df = pd.DataFrame({
            "platform":      ["google", "google"],
            "campaign_id":   [1, 1],      # duplicate
            "campaign_name": ["Camp_A", "Camp_B"],
        })
        with pytest.raises(TaxonomyParseError, match="campaign_id"):
            _validate_metadata(df)

    def test_duplicate_campaign_name_raises(self) -> None:
        df = pd.DataFrame({
            "platform":      ["google", "google"],
            "campaign_id":   [1, 2],
            "campaign_name": ["Camp_A", "Camp_A"],   # duplicate
        })
        with pytest.raises(TaxonomyParseError, match="campaign_name"):
            _validate_metadata(df)

    def test_duplicate_id_only_within_same_platform(self) -> None:
        # Same campaign_id across DIFFERENT platforms is allowed
        df = pd.DataFrame({
            "platform":      ["google", "bing"],
            "campaign_id":   [1, 1],   # same id, different platform → OK
            "campaign_name": ["Camp_A", "Camp_A"],  # same name, different platform → OK
        })
        _validate_metadata(df)  # must not raise

    def test_taxonomy_parse_error_has_details(self) -> None:
        df = pd.DataFrame({
            "platform":      ["google", "google"],
            "campaign_id":   [1, 1],
            "campaign_name": ["A", "B"],
        })
        try:
            _validate_metadata(df)
            pytest.fail("TaxonomyParseError was not raised")
        except TaxonomyParseError as exc:
            assert exc.details, "TaxonomyParseError.details should be non-empty"

    def test_real_data_passes_validation(self, campaign_meta: pd.DataFrame) -> None:
        _validate_metadata(campaign_meta)
