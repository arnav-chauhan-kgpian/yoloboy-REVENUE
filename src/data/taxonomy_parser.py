"""
src/data/taxonomy_parser.py
============================
Campaign taxonomy parsing layer for the meridian 2026 marketing mix engine.

Purpose
-------
Accept the canonical DataFrame from harmonizer.py and produce a
``campaign_metadata`` DataFrame with one row per unique campaign (136 total
across all three platforms). Extracts structured taxonomy from campaign
names and computes strategy-classification features used by the forecasting
models.

Inputs
------
Canonical ``pd.DataFrame`` from ``src.data.harmonizer.harmonize()``.
Required columns: platform, campaign_id, campaign_name, channel_format.

Naming conventions observed in the actual dataset
--------------------------------------------------
Google / Bing (identical convention):
    {Format}[_{TM|NTM}]_Campaign_{N}
    Examples:
      Pmax_NTM_Campaign_01        → format=Pmax,      audience=NTM, number=1
      Search_TM_Campaign_02       → format=Search,     audience=TM,  number=2
      Shopping_NTM_Campaign_01    → format=Shopping,   audience=NTM, number=1
      Demand Gen_NTM_Campaign_01  → format=Demand Gen, audience=NTM, number=1
      Pmax_Campaign_01            → format=Pmax,       audience=None, number=1
      Display_Campaign_01         → format=Display,    audience=None, number=1
      Video_Campaign_01           → format=Video,      audience=None, number=1

Meta (different convention):
    {FunnelStage}[_{AdProductType}]_Campaign_{N}
    Examples:
      Prospecting_DPA_Campaign_01      → funnel=Prospecting, product=DPA,      number=1
      Remarketing_Brand_Campaign_02    → funnel=Remarketing,  product=Brand,    number=2
      Prospecting_Adv_Plus_Campaign_01 → funnel=Prospecting,  product=Adv_Plus, number=1
      Generic_Brand_Campaign_01        → funnel=Generic,      product=Brand,    number=1
      Generic_Campaign_01              → funnel=Generic,      product=None,     number=1

Outputs
-------
``pd.DataFrame`` — campaign_metadata
  One row per unique (platform, campaign_id).  136 rows expected.
  Columns: see CAMPAIGN_METADATA_COLUMNS.

Cross-engine pairs
------------------
27 campaign names appear in BOTH Google and Bing (same strategy, different
platform IDs). These are flagged with cross_engine_pair_flag=True.
Meta campaigns never cross-pair with Google or Bing.

Strategy keys
-------------
Unified string token combining platform and name-parsed taxonomy:
  google_{Format}_{Audience}  e.g. "google_Pmax_NTM", "google_Search_TM"
  bing_{Format}_{Audience}    e.g. "bing_Pmax_NTM"
  meta_{Funnel}_{AdProduct}   e.g. "meta_Prospecting_DPA", "meta_Generic"
Spaces in format names (e.g. "Demand Gen") are replaced with underscores.

Boolean feature flags
---------------------
  is_brand      : TM campaigns (Google/Bing) or Brand ad-product (Meta)
  is_non_brand  : NTM campaigns (Google/Bing) or DPA/Adv_Plus (Meta)
  is_upper_funnel: channel_format == "UpperFunnel" (Video/Demand Gen/Display)

Note: campaigns without a TM/NTM label (e.g. Pmax_Campaign_01, Display_Campaign_01)
and Meta generic campaigns with no ad-product type get is_brand=False AND
is_non_brand=False — they are unclassified, not "non-brand".

Validation
----------
Raises TaxonomyParseError if:
  - campaign_id is not unique within a platform
  - campaign_name is not unique within a platform
"""

from __future__ import annotations

import logging
from typing import Final, TypedDict

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GOOGLE_BING_PLATFORMS: Final[frozenset[str]] = frozenset({"google", "bing"})
META_PLATFORM: Final[str] = "meta"

# Meta funnel stages — order matters for prefix-matching (longer before shorter).
_META_FUNNEL_STAGES: Final[tuple[str, ...]] = (
    "Prospecting",
    "Remarketing",
    "Generic",
)

CAMPAIGN_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "platform",
    "campaign_id",
    "campaign_name",
    "channel_format",
    "format",
    "audience_strategy",
    "funnel_stage",
    "ad_product_type",
    "campaign_number",
    "strategy_key",
    "cross_engine_pair_flag",
    "is_brand",
    "is_non_brand",
    "is_upper_funnel",
)


# ---------------------------------------------------------------------------
# TypedDict return types for the private parsing helpers
# ---------------------------------------------------------------------------

class _GoogleBingParsed(TypedDict):
    format: str | None
    audience_strategy: str | None
    campaign_number: int | None


class _MetaParsed(TypedDict):
    funnel_stage: str | None
    ad_product_type: str | None
    campaign_number: int | None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TaxonomyParseError(Exception):
    """
    Raised when campaign_metadata validation detects an integrity violation.

    Attributes
    ----------
    details : str
        Additional context (platform, affected campaigns).
    """

    def __init__(self, message: str, details: str = "") -> None:
        super().__init__(message)
        self.details = details

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base}\n  details: {self.details}" if self.details else base


# ---------------------------------------------------------------------------
# Private: name-parsing helpers
# ---------------------------------------------------------------------------

def _parse_google_bing_name(name: str) -> _GoogleBingParsed:
    """
    Parse a Google or Bing campaign name into structured taxonomy fields.

    Naming convention:  {Format}[_{TM|NTM}]_Campaign_{N}

    Parsing rules
    -------------
    1. Split on the first ``_Campaign_`` token.
       Left part = prefix.  Right part = campaign number string.
    2. If prefix ends with ``_NTM`` → audience_strategy="NTM", strip suffix.
       If prefix ends with ``_TM``  → audience_strategy="TM",  strip suffix.
       Otherwise                    → audience_strategy=None.
    3. Remaining prefix is the format (may contain a space, e.g. "Demand Gen").

    Parameters
    ----------
    name : str
        Raw campaign_name from Google or Bing data.

    Returns
    -------
    _GoogleBingParsed
        TypedDict with keys: format, audience_strategy, campaign_number.

    Examples
    --------
    >>> _parse_google_bing_name("Pmax_NTM_Campaign_01")
    {'format': 'Pmax', 'audience_strategy': 'NTM', 'campaign_number': 1}
    >>> _parse_google_bing_name("Search_TM_Campaign_02")
    {'format': 'Search', 'audience_strategy': 'TM', 'campaign_number': 2}
    >>> _parse_google_bing_name("Demand Gen_NTM_Campaign_01")
    {'format': 'Demand Gen', 'audience_strategy': 'NTM', 'campaign_number': 1}
    >>> _parse_google_bing_name("Pmax_Campaign_01")
    {'format': 'Pmax', 'audience_strategy': None, 'campaign_number': 1}
    """
    if "_Campaign_" not in name:
        logger.warning("Name does not match {Format}[_TM|NTM]_Campaign_{N}: %r", name)
        return {"format": name, "audience_strategy": None, "campaign_number": None}

    prefix, number_str = name.split("_Campaign_", 1)

    try:
        campaign_number: int | None = int(number_str)
    except ValueError:
        logger.warning("Cannot parse campaign number from %r (part: %r)", name, number_str)
        campaign_number = None

    if prefix.endswith("_NTM"):
        fmt: str | None = prefix[:-4]  # strip "_NTM"
        audience: str | None = "NTM"
    elif prefix.endswith("_TM"):
        fmt = prefix[:-3]  # strip "_TM"
        audience = "TM"
    else:
        fmt = prefix
        audience = None

    return {
        "format": fmt or None,
        "audience_strategy": audience,
        "campaign_number": campaign_number,
    }


def _parse_meta_name(name: str) -> _MetaParsed:
    """
    Parse a Meta campaign name into structured taxonomy fields.

    Naming convention:  {FunnelStage}[_{AdProductType}]_Campaign_{N}

    Known funnel stages (checked in order): Prospecting, Remarketing, Generic.
    Known ad product types: Brand, DPA, Adv_Plus.

    Parsing rules
    -------------
    1. Split on the first ``_Campaign_`` token.
    2. Match prefix against known funnel stages (exact prefix match).
       - If prefix == stage_name exactly → funnel=stage, ad_product=None.
       - If prefix starts with stage_name + "_" → funnel=stage,
         ad_product = everything after the underscore.

    Parameters
    ----------
    name : str
        Raw campaign_name from Meta data.

    Returns
    -------
    _MetaParsed
        TypedDict with keys: funnel_stage, ad_product_type, campaign_number.

    Examples
    --------
    >>> _parse_meta_name("Prospecting_DPA_Campaign_01")
    {'funnel_stage': 'Prospecting', 'ad_product_type': 'DPA', 'campaign_number': 1}
    >>> _parse_meta_name("Prospecting_Adv_Plus_Campaign_01")
    {'funnel_stage': 'Prospecting', 'ad_product_type': 'Adv_Plus', 'campaign_number': 1}
    >>> _parse_meta_name("Remarketing_Brand_Campaign_02")
    {'funnel_stage': 'Remarketing', 'ad_product_type': 'Brand', 'campaign_number': 2}
    >>> _parse_meta_name("Generic_Campaign_01")
    {'funnel_stage': 'Generic', 'ad_product_type': None, 'campaign_number': 1}
    """
    if "_Campaign_" not in name:
        logger.warning("Meta name does not match {Funnel}[_Product]_Campaign_{N}: %r", name)
        return {"funnel_stage": None, "ad_product_type": None, "campaign_number": None}

    prefix, number_str = name.split("_Campaign_", 1)

    try:
        campaign_number: int | None = int(number_str)
    except ValueError:
        logger.warning("Cannot parse campaign number from %r (part: %r)", name, number_str)
        campaign_number = None

    funnel_stage: str | None = None
    ad_product_type: str | None = None

    for stage in _META_FUNNEL_STAGES:
        if prefix == stage:
            funnel_stage = stage
            ad_product_type = None
            break
        if prefix.startswith(stage + "_"):
            funnel_stage = stage
            remainder = prefix[len(stage) + 1:]
            ad_product_type = remainder or None
            break

    if funnel_stage is None:
        logger.warning("Cannot identify funnel stage in Meta name: %r", name)

    return {
        "funnel_stage": funnel_stage,
        "ad_product_type": ad_product_type,
        "campaign_number": campaign_number,
    }


def _make_strategy_key(
    platform: str,
    part1: str | None,
    part2: str | None,
) -> str:
    """
    Build a strategy_key string from platform and up to two name segments.

    Spaces in segment names are replaced with underscores so the key is a
    valid Python identifier and safe as a categorical feature label.

    Parameters
    ----------
    platform : str
        One of "google", "meta", "bing".
    part1 : str | None
        Primary classification (format for Google/Bing; funnel for Meta).
    part2 : str | None
        Secondary classification (audience for Google/Bing; ad_product for Meta).

    Returns
    -------
    str
        Underscore-joined strategy key.

    Examples
    --------
    >>> _make_strategy_key("google", "Pmax", "NTM")
    'google_Pmax_NTM'
    >>> _make_strategy_key("bing", "Demand Gen", "NTM")
    'bing_Demand_Gen_NTM'
    >>> _make_strategy_key("meta", "Prospecting", "DPA")
    'meta_Prospecting_DPA'
    >>> _make_strategy_key("meta", "Generic", None)
    'meta_Generic'
    >>> _make_strategy_key("google", "Pmax", None)
    'google_Pmax'
    """
    parts: list[str] = [platform]
    # Use isinstance guard: pandas stores None as float NaN in object columns,
    # and bool(float('nan')) is True, so a plain `if part1` check passes NaN
    # through to .replace() and raises AttributeError.
    if isinstance(part1, str) and part1:
        parts.append(part1.replace(" ", "_"))
    if isinstance(part2, str) and part2:
        parts.append(part2.replace(" ", "_"))
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Private: per-platform processing
# ---------------------------------------------------------------------------

def _process_google_bing_campaigns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply name parsing and feature generation to Google/Bing unique campaigns.

    Parameters
    ----------
    df : pd.DataFrame
        Subset of unique campaigns where platform ∈ {"google", "bing"}.
        Required columns: platform, campaign_id, campaign_name, channel_format.

    Returns
    -------
    pd.DataFrame
        Rows in CAMPAIGN_METADATA_COLUMNS order (minus cross_engine_pair_flag,
        which requires the full combined DataFrame — set to False here as a
        placeholder that parse_taxonomy overwrites).
    """
    if df.empty:
        return pd.DataFrame(columns=list(CAMPAIGN_METADATA_COLUMNS))

    parsed_records = df["campaign_name"].apply(_parse_google_bing_name)
    parsed = pd.DataFrame(parsed_records.tolist(), index=df.index)

    result = df[["platform", "campaign_id", "campaign_name", "channel_format"]].copy()
    result = result.reset_index(drop=True)
    parsed = parsed.reset_index(drop=True)

    # Normalise: pandas may store None→NaN or pd.NA in object/StringDtype
    # columns; convert to Python None so set-membership tests work correctly.
    def _to_none(s: pd.Series) -> pd.Series:
        s = s.astype(object)
        return s.where(s.notna(), None)

    result["format"]            = _to_none(parsed["format"])
    result["audience_strategy"] = _to_none(parsed["audience_strategy"])
    result["funnel_stage"]      = None
    result["ad_product_type"]   = None
    result["campaign_number"]   = parsed["campaign_number"]

    result["strategy_key"] = [
        _make_strategy_key(p, f, a)
        for p, f, a in zip(
            result["platform"],
            result["format"],
            result["audience_strategy"],
        )
    ]

    aud = result["audience_strategy"].fillna("")
    result["is_brand"]       = aud == "TM"
    result["is_non_brand"]   = aud == "NTM"
    result["is_upper_funnel"] = result["channel_format"] == "UpperFunnel"

    # Placeholder — parse_taxonomy overwrites this once both platforms are combined
    result["cross_engine_pair_flag"] = False

    return result


def _process_meta_campaigns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply name parsing and feature generation to Meta unique campaigns.

    Parameters
    ----------
    df : pd.DataFrame
        Subset of unique campaigns where platform == "meta".
        Required columns: platform, campaign_id, campaign_name, channel_format.

    Returns
    -------
    pd.DataFrame
        Rows in CAMPAIGN_METADATA_COLUMNS order.
        cross_engine_pair_flag is always False for Meta campaigns.
    """
    if df.empty:
        return pd.DataFrame(columns=list(CAMPAIGN_METADATA_COLUMNS))

    parsed_records = df["campaign_name"].apply(_parse_meta_name)
    parsed = pd.DataFrame(parsed_records.tolist(), index=df.index)

    result = df[["platform", "campaign_id", "campaign_name", "channel_format"]].copy()
    result = result.reset_index(drop=True)
    parsed = parsed.reset_index(drop=True)

    # Normalise: pandas may store None→NaN or pd.NA in object/StringDtype
    # columns; convert to Python None so set-membership tests work correctly.
    def _to_none(s: pd.Series) -> pd.Series:
        s = s.astype(object)
        return s.where(s.notna(), None)

    result["format"]            = None
    result["audience_strategy"] = None
    result["funnel_stage"]      = _to_none(parsed["funnel_stage"])
    result["ad_product_type"]   = _to_none(parsed["ad_product_type"])
    result["campaign_number"]   = parsed["campaign_number"]

    result["strategy_key"] = [
        _make_strategy_key(p, f, a)
        for p, f, a in zip(
            result["platform"],
            result["funnel_stage"],
            result["ad_product_type"],
        )
    ]

    prod = result["ad_product_type"].fillna("")
    result["is_brand"]       = prod == "Brand"
    result["is_non_brand"]   = prod.isin({"DPA", "Adv_Plus"})
    result["is_upper_funnel"] = result["channel_format"] == "UpperFunnel"

    # Meta campaigns never cross-pair with Google or Bing
    result["cross_engine_pair_flag"] = False

    return result


# ---------------------------------------------------------------------------
# Private: validation
# ---------------------------------------------------------------------------

def _validate_metadata(df: pd.DataFrame) -> None:
    """
    Validate that campaign_id and campaign_name are unique within each platform.

    Parameters
    ----------
    df : pd.DataFrame
        The assembled campaign_metadata DataFrame.

    Raises
    ------
    TaxonomyParseError
        If any campaign_id or campaign_name appears more than once within a
        platform.
    """
    for col, label in [("campaign_id", "campaign_id"), ("campaign_name", "campaign_name")]:
        dup_counts = (
            df.groupby("platform", observed=True)[col]
            .apply(lambda s: int(s.duplicated().sum()))
        )
        bad = dup_counts[dup_counts > 0]
        if not bad.empty:
            raise TaxonomyParseError(
                f"Duplicate {label} values found within platform.",
                details=bad.to_string(),
            )

    logger.info("Taxonomy validation passed: %d unique campaigns", len(df))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_taxonomy(canon: pd.DataFrame) -> pd.DataFrame:
    """
    Parse campaign taxonomy from the canonical harmonized DataFrame.

    Parameters
    ----------
    canon : pd.DataFrame
        Output of ``src.data.harmonizer.harmonize()``.
        Must contain: platform, campaign_id, campaign_name, channel_format.

    Returns
    -------
    pd.DataFrame
        campaign_metadata with columns CAMPAIGN_METADATA_COLUMNS.
        One row per unique (platform, campaign_id).  136 rows expected.

        Key columns:

        format           : str | None — ad format for Google/Bing
                           (Pmax, Search, Shopping, Video, Display, Demand Gen)
        audience_strategy: str | None — TM, NTM, or None for Google/Bing
        funnel_stage     : str | None — Prospecting, Remarketing, Generic for Meta
        ad_product_type  : str | None — Brand, DPA, Adv_Plus for Meta
        campaign_number  : Int64     — parsed integer from campaign name suffix
        strategy_key     : str       — "{platform}_{type}[_{subtype}]"
        cross_engine_pair_flag: bool — True for 54 campaigns (27 Google + 27 Bing)
                           that share a name across both platforms
        is_brand         : bool — TM (Google/Bing) or Brand ad-product (Meta)
        is_non_brand     : bool — NTM (Google/Bing) or DPA/Adv_Plus (Meta)
        is_upper_funnel  : bool — channel_format == "UpperFunnel"

    Raises
    ------
    TaxonomyParseError
        If campaign_id or campaign_name is not unique within any platform.

    Examples
    --------
    >>> from src.data.loader import load_raw_data
    >>> from src.data.harmonizer import harmonize
    >>> from src.data.taxonomy_parser import parse_taxonomy
    >>> canon = harmonize(load_raw_data())
    >>> meta = parse_taxonomy(canon)
    >>> meta.shape
    (136, 14)
    >>> meta[meta["platform"] == "google"]["is_brand"].sum()
    5
    """
    logger.info("=== TaxonomyParser: start ===")

    # --- 1. Extract one row per unique campaign ---------------------------------
    unique_campaigns = (
        canon[["platform", "campaign_id", "campaign_name", "channel_format"]]
        .drop_duplicates(subset=["platform", "campaign_id"])
        .copy()
        .reset_index(drop=True)
    )
    logger.info("Unique campaigns extracted: %d", len(unique_campaigns))

    # --- 2. Split by platform group and parse -----------------------------------
    gb_mask   = unique_campaigns["platform"].isin(GOOGLE_BING_PLATFORMS)
    meta_mask = unique_campaigns["platform"] == META_PLATFORM

    gb_parsed   = _process_google_bing_campaigns(unique_campaigns[gb_mask])
    meta_parsed = _process_meta_campaigns(unique_campaigns[meta_mask])

    # --- 3. Combine ------------------------------------------------------------
    combined = pd.concat([gb_parsed, meta_parsed], ignore_index=True)

    # --- 4. Cross-engine pair flag: requires both Google and Bing names --------
    google_names = set(combined.loc[combined["platform"] == "google", "campaign_name"])
    bing_names   = set(combined.loc[combined["platform"] == "bing",   "campaign_name"])
    shared_names = google_names & bing_names

    combined["cross_engine_pair_flag"] = combined["campaign_name"].isin(shared_names)
    logger.info(
        "Cross-engine pairs: %d shared campaign names → %d flagged rows",
        len(shared_names),
        combined["cross_engine_pair_flag"].sum(),
    )

    # --- 5. Enforce dtypes -----------------------------------------------------
    combined["campaign_number"]      = pd.array(combined["campaign_number"], dtype=pd.Int64Dtype())
    combined["cross_engine_pair_flag"] = combined["cross_engine_pair_flag"].astype("bool")
    combined["is_brand"]              = combined["is_brand"].astype("bool")
    combined["is_non_brand"]          = combined["is_non_brand"].astype("bool")
    combined["is_upper_funnel"]       = combined["is_upper_funnel"].astype("bool")

    # --- 6. Validate -----------------------------------------------------------
    _validate_metadata(combined)

    # --- 7. Return in canonical column order -----------------------------------
    result = combined[list(CAMPAIGN_METADATA_COLUMNS)].reset_index(drop=True)

    logger.info(
        "=== TaxonomyParser complete: %d campaigns | %d brand | %d non-brand | "
        "%d upper-funnel | %d cross-engine pairs ===",
        len(result),
        result["is_brand"].sum(),
        result["is_non_brand"].sum(),
        result["is_upper_funnel"].sum(),
        result["cross_engine_pair_flag"].sum(),
    )

    return result
