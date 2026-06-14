"""
src/data/loader.py
==================
Raw data ingestion layer for the meridian 2026 marketing mix intelligence
engine.

Purpose
-------
Read the three advertising-platform CSV files (Google Ads, Meta Ads,
Microsoft / Bing Ads), apply the minimum structural cleanup required before
harmonization, and return strongly-typed DataFrames inside a RawDataset
container.

Assumptions
-----------
- All three files live in a single directory (default: ``dataset/``). If the
  default ``dataset/`` path is present but does not contain CSVs, the loader
  falls back to ``data/`` (repository sample data).
- Column names match those confirmed during forensic analysis of the actual
  files.  Any deviation raises DataLoadError immediately.
- Every file contains an ``Unnamed: 0`` column — a pandas ``to_csv`` index
  artifact.  It carries no information and is dropped on load.
- Meta's ``conversion`` column contains monetary conversion VALUE (revenue),
  not an event count.  Mean ≈ $485, max ≈ $26,539.  This naming ambiguity is
  explicitly documented here and handled (renamed) in harmonizer.py.
- Google's ``metrics_cost_micros`` is in micros (integer).  The ÷ 1e6
  conversion to currency is the harmonizer's responsibility.
- Date strings are ISO 8601 (``YYYY-MM-DD``) in all three files.
- ``campaign_budget_amount`` (Google) has 14 NULLs; ``daily_budget`` (Meta)
  has 7 NULLs — both are represented as float64 and passed through as-is.

Inputs
------
``data_dir`` : str | Path
    Path to the directory that contains the three CSV files.

Outputs
-------
``RawDataset``
    Dataclass with three DataFrames:
      .google — 11 columns, ~19 272 rows
      .meta   — 12 columns, ~3 417  rows
      .bing   — 10 columns, ~2 873  rows
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical file names
# ---------------------------------------------------------------------------
# src/data/loader.py -> src/data -> src -> project root
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR: Final[Path] = _PROJECT_ROOT / "dataset"
_FALLBACK_DATA_DIR: Final[Path] = _PROJECT_ROOT / "data"

GOOGLE_FILENAME: Final[str] = "google_ads_campaign_stats.csv"
META_FILENAME: Final[str] = "meta_ads_campaign_stats.csv"
BING_FILENAME: Final[str] = "bing_campaign_stats.csv"

# Pandas to_csv() index artifact present in every file.
_INDEX_ARTIFACT: Final[str] = "Unnamed: 0"

# ---------------------------------------------------------------------------
# Expected columns — frozensets validated after loading.
# These are the post-load columns (index artifact already removed).
# ---------------------------------------------------------------------------
_GOOGLE_EXPECTED: Final[frozenset[str]] = frozenset({
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

_META_EXPECTED: Final[frozenset[str]] = frozenset({
    "campaign_id",
    "date_start",
    "cpc",
    "cpm",
    "ctr",
    "reach",
    "spend",
    "clicks",
    "impressions",
    "conversion",       # Revenue VALUE — renamed to revenue_attributed in harmonizer
    "daily_budget",
    "campaign_name",
})

_BING_EXPECTED: Final[frozenset[str]] = frozenset({
    "CampaignId",       # Bing uses PascalCase — normalised in harmonizer
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
# dtype maps passed to pd.read_csv.
# Keys must match the raw CSV column names (including PascalCase for Bing).
# The date column is intentionally read as object here and converted below;
# read_csv parse_dates is avoided for explicit error surfacing.
# ---------------------------------------------------------------------------
_GOOGLE_DTYPES: Final[dict[str, str]] = {
    "campaign_id":                            "int64",
    "metrics_clicks":                         "int64",
    "metrics_conversions":                    "float64",
    "metrics_cost_micros":                    "int64",
    "metrics_impressions":                    "int64",
    "metrics_video_views":                    "int64",
    "metrics_conversions_value":              "float64",
    "campaign_advertising_channel_type":      "object",
    "campaign_budget_amount":                 "float64",   # 14 NULLs
    "campaign_name":                          "object",
}

_META_DTYPES: Final[dict[str, str]] = {
    "campaign_id":    "int64",    # Large IDs (e.g. 120210921616440533) fit int64
    "cpc":            "float64",
    "cpm":            "float64",
    "ctr":            "float64",
    "reach":          "float64",  # ~entirely 0.0 — flagged for harmonizer
    "spend":          "float64",
    "clicks":         "float64",  # Meta reports clicks as float
    "impressions":    "float64",  # Meta reports impressions as float
    "conversion":     "float64",  # Revenue VALUE, not count — see module docstring
    "daily_budget":   "float64",  # 7 NULLs
    "campaign_name":  "object",
}

_BING_DTYPES: Final[dict[str, str]] = {
    "CampaignId":    "int64",
    "Revenue":       "float64",
    "Spend":         "float64",
    "Clicks":        "int64",
    "Impressions":   "int64",
    "Conversions":   "float64",
    "CampaignType":  "object",
    "DailyBudget":   "float64",
    "CampaignName":  "object",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class DataLoadError(Exception):
    """
    Raised when a raw data file cannot be loaded or fails schema validation.

    Attributes
    ----------
    platform : str
        The advertising platform label for which loading failed.
    path : Path | None
        The file path that triggered the error, if known.
    """

    def __init__(self, message: str, platform: str = "", path: Path | None = None) -> None:
        super().__init__(message)
        self.platform = platform
        self.path = path

    def __str__(self) -> str:
        base = super().__str__()
        parts = [base]
        if self.platform:
            parts.append(f"  platform : {self.platform}")
        if self.path:
            parts.append(f"  file     : {self.path}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# RawDataset
# ---------------------------------------------------------------------------
@dataclass
class RawDataset:
    """
    Container for the three unharmonized advertising-platform DataFrames.

    This is the direct output of DataLoader.load_all() / load_raw_data().
    Each DataFrame retains its native column names and dtypes; no business
    transformations have been applied.  Pass this to harmonizer.py next.

    Attributes
    ----------
    google : pd.DataFrame
        Raw Google Ads data.  11 columns after index-artifact removal.
        Date column: ``segments_date`` (datetime64[ns]).
    meta : pd.DataFrame
        Raw Meta Ads data.  12 columns after index-artifact removal.
        Date column: ``date_start`` (datetime64[ns]).
        Note: ``conversion`` = revenue value, not event count.
    bing : pd.DataFrame
        Raw Bing / Microsoft Ads data.  10 columns after index-artifact
        removal.  Date column: ``TimePeriod`` (datetime64[ns]).
        Uses PascalCase column names throughout.
    """

    google: pd.DataFrame
    meta: pd.DataFrame
    bing: pd.DataFrame

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def is_empty(self) -> bool:
        """Return True if any of the three DataFrames is empty."""
        return self.google.empty or self.meta.empty or self.bing.empty

    def total_rows(self) -> int:
        """Total row count across all three platforms."""
        return len(self.google) + len(self.meta) + len(self.bing)

    def summary(self) -> str:
        """
        Return a human-readable multi-line summary suitable for log output.

        Example output::

            RawDataset — 25,562 total rows across 3 platforms
              Google : 19,272 rows | 11 cols | 2024-01-01 to 2026-06-04 | 92 campaigns
              Meta   :  3,417 rows | 12 cols | 2024-05-23 to 2026-06-05 | 16 campaigns
              Bing   :  2,873 rows | 10 cols | 2024-05-25 to 2026-06-05 | 28 campaigns
        """
        specs = [
            ("Google", self.google, "segments_date", "campaign_id"),
            ("Meta",   self.meta,   "date_start",    "campaign_id"),
            ("Bing",   self.bing,   "TimePeriod",     "CampaignId"),
        ]
        lines = [f"RawDataset — {self.total_rows():,} total rows across 3 platforms"]
        for label, df, date_col, id_col in specs:
            if df.empty:
                lines.append(f"  {label:<6}: EMPTY")
                continue
            lines.append(
                f"  {label:<6}: {len(df):>6,} rows | {df.shape[1]:>2} cols | "
                f"{df[date_col].min().date()} to {df[date_col].max().date()} | "
                f"{df[id_col].nunique()} campaigns"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
class DataLoader:
    """
    Loads the raw advertising-platform CSV files with minimal structural
    cleanup.

    Responsibilities
    ----------------
    - Locate and read the three CSV files from a configurable directory.
    - Drop the ``Unnamed: 0`` pandas index artifact from each file.
    - Parse the date string column of each file to ``datetime64[ns]``.
    - Enforce numeric dtypes declared in the forensic schema constants.
    - Validate that all expected columns are present and raise promptly if not.

    Explicitly NOT responsible for
    --------------------------------
    - Unit conversions (Google micros → currency): see ``harmonizer.py``.
    - Column renaming to the canonical schema: see ``harmonizer.py``.
    - Flagging the Meta ``conversion`` ambiguity: documented here, resolved
      in ``harmonizer.py``.
    - Taxonomy parsing: see ``taxonomy_parser.py``.
    - Business-rule validation: see ``validator.py``.

    Parameters
    ----------
    data_dir : str | Path
        Path to the directory containing the three platform CSV files.
        Defaults to ``"dataset"`` (relative to the project root / CWD).

    Raises
    ------
    FileNotFoundError
        If ``data_dir`` does not exist or any expected CSV file is absent.
    NotADirectoryError
        If ``data_dir`` resolves to a file, not a directory.
    DataLoadError
        If a CSV file fails to parse or is missing expected columns.

    Examples
    --------
    >>> loader = DataLoader(data_dir="dataset")
    >>> dataset = loader.load_all()
    >>> dataset.google.shape
    (19272, 11)
    """

    def __init__(self, data_dir: str | Path = "dataset") -> None:
        self._data_dir = Path(data_dir)
        self._check_directory()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        """Resolved absolute path to the data directory."""
        return self._data_dir.resolve()

    # ------------------------------------------------------------------
    # Private: directory / file checks
    # ------------------------------------------------------------------
    def _check_directory(self) -> None:
        """Validate that data_dir exists and is a directory."""
        resolved = self._data_dir.resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"Data directory not found: '{resolved}'\n"
                "Point data_dir at the folder containing the three platform CSVs."
            )
        if not resolved.is_dir():
            raise NotADirectoryError(
                f"data_dir must be a directory, but a file was found: '{resolved}'"
            )
        logger.debug("Data directory confirmed: %s", resolved)

    def _resolve_file(self, filename: str, platform: str) -> Path:
        """
        Return the resolved Path for a file inside data_dir.

        Parameters
        ----------
        filename : str  Canonical file name constant.
        platform : str  Human-readable label for error messages.

        Raises
        ------
        FileNotFoundError
            If the file does not exist at the expected location.
        """
        path = (self._data_dir / filename).resolve()
        if not path.exists():
            if self._data_dir.resolve() == _DEFAULT_DATA_DIR and _FALLBACK_DATA_DIR.exists():
                fallback_path = (_FALLBACK_DATA_DIR / filename).resolve()
                # Fall back only when the specific expected CSV is present.
                if fallback_path.exists():
                    logger.warning(
                        "Using fallback file for %s: '%s' (requested from '%s').",
                        platform,
                        fallback_path,
                        path,
                    )
                    return fallback_path
            raise FileNotFoundError(
                f"{platform}: expected file not found.\n"
                f"  expected : '{path}'\n"
                f"  data_dir : '{self.data_dir}'\n"
                "Ensure the file exists and data_dir is set correctly."
            )
        return path

    # ------------------------------------------------------------------
    # Private: core CSV reader
    # ------------------------------------------------------------------
    def _read_csv(
        self,
        path: Path,
        dtype_map: dict[str, str],
        date_col: str,
        platform: str,
    ) -> pd.DataFrame:
        """
        Read a single platform CSV, drop the index artifact, and parse dates.

        Parameters
        ----------
        path       : Path           Absolute path to the CSV file.
        dtype_map  : dict[str, str] Column-to-dtype mapping for read_csv.
        date_col   : str            Name of the date column to parse.
        platform   : str            Human-readable platform name for logging.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame: index artifact removed, date column parsed.

        Raises
        ------
        DataLoadError
            On any I/O error, CSV parsing failure, or missing date column.
        """
        logger.info("[%s] Reading file: %s", platform, path)

        # --- Read CSV -------------------------------------------------------
        try:
            df = pd.read_csv(path, dtype=dtype_map, low_memory=False)
        except Exception as exc:
            raise DataLoadError(
                f"Failed to read CSV file: {exc}",
                platform=platform,
                path=path,
            ) from exc

        original_rows, original_cols = df.shape
        logger.debug("[%s] Raw shape: %d rows × %d cols", platform, original_rows, original_cols)

        # --- Drop pandas index artifact -------------------------------------
        if _INDEX_ARTIFACT in df.columns:
            df = df.drop(columns=[_INDEX_ARTIFACT])
            logger.debug("[%s] Dropped '%s' column.", platform, _INDEX_ARTIFACT)
        else:
            logger.warning(
                "[%s] Expected index-artifact column '%s' was not found — "
                "file may have been re-exported without it.",
                platform,
                _INDEX_ARTIFACT,
            )

        # --- Parse date column ----------------------------------------------
        if date_col not in df.columns:
            raise DataLoadError(
                f"Date column '{date_col}' not found. "
                f"Columns present: {sorted(df.columns)}",
                platform=platform,
                path=path,
            )
        try:
            df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d")
        except Exception as exc:
            # Attempt without strict format as fallback and warn
            logger.warning(
                "[%s] Strict ISO-8601 parse failed for '%s' (%s); "
                "retrying with inferred format.",
                platform,
                date_col,
                exc,
            )
            try:
                df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
            except Exception as exc2:
                raise DataLoadError(
                    f"Cannot parse date column '{date_col}': {exc2}",
                    platform=platform,
                    path=path,
                ) from exc2

        # --- Log summary ----------------------------------------------------
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        if cols_with_nulls:
            logger.info("[%s] Columns with NULLs: %s", platform, cols_with_nulls)

        logger.info(
            "[%s] Loaded: %d rows | %d cols | dates %s to %s",
            platform,
            len(df),
            df.shape[1],
            df[date_col].min().date(),
            df[date_col].max().date(),
        )
        return df

    # ------------------------------------------------------------------
    # Private: column schema validation
    # ------------------------------------------------------------------
    def _check_columns(
        self,
        df: pd.DataFrame,
        expected: frozenset[str],
        platform: str,
        path: Path,
    ) -> None:
        """
        Raise DataLoadError if any expected column is absent.

        Unexpected *extra* columns are logged as warnings and passed through —
        they do not fail validation because future file schema additions should
        not break the loader.

        Parameters
        ----------
        df       : pd.DataFrame       DataFrame to inspect.
        expected : frozenset[str]     Column names that must be present.
        platform : str                Human-readable platform name.
        path     : Path               Source file path (for error context).

        Raises
        ------
        DataLoadError
            If one or more expected columns are missing.
        """
        present = set(df.columns)
        missing = expected - present
        extra = present - expected

        if missing:
            raise DataLoadError(
                f"Schema mismatch — missing columns: {sorted(missing)}. "
                f"Columns present: {sorted(present)}",
                platform=platform,
                path=path,
            )
        if extra:
            logger.warning(
                "[%s] Unexpected columns (carried through to harmonizer): %s",
                platform,
                sorted(extra),
            )
        logger.debug("[%s] Column schema validated (%d columns).", platform, len(present))

    # ------------------------------------------------------------------
    # Public: per-platform loaders
    # ------------------------------------------------------------------
    def load_google(self) -> pd.DataFrame:
        """
        Load the Google Ads campaign stats CSV.

        Post-load schema
        ----------------
        campaign_id                       int64
        segments_date                     datetime64[ns]
        metrics_clicks                    int64
        metrics_conversions               float64
        metrics_cost_micros               int64         ← micros; ÷1e6 in harmonizer
        metrics_impressions               int64
        metrics_video_views               int64
        metrics_conversions_value         float64       ← attributed revenue
        campaign_advertising_channel_type object
        campaign_budget_amount            float64       ← 14 NULLs expected
        campaign_name                     object

        Returns
        -------
        pd.DataFrame
            11 columns, ~19 272 rows.

        Raises
        ------
        FileNotFoundError
            If ``google_ads_campaign_stats.csv`` is not found.
        DataLoadError
            If the file cannot be parsed or expected columns are missing.
        """
        path = self._resolve_file(GOOGLE_FILENAME, "Google")
        df = self._read_csv(path, _GOOGLE_DTYPES, "segments_date", "Google")
        self._check_columns(df, _GOOGLE_EXPECTED, "Google", path)
        return df

    def load_meta(self) -> pd.DataFrame:
        """
        Load the Meta Ads campaign stats CSV.

        Post-load schema
        ----------------
        campaign_id   int64
        date_start    datetime64[ns]
        cpc           float64   ← same-day derived ratio; use only as lag in features
        cpm           float64   ← same-day derived ratio; use only as lag in features
        ctr           float64   ← same-day derived ratio; use only as lag in features
        reach         float64   ← ~entirely 0.0; flagged unusable in harmonizer
        spend         float64
        clicks        float64   ← Meta reports as float
        impressions   float64   ← Meta reports as float
        conversion    float64   ← REVENUE VALUE, not event count (mean≈$485, max≈$26,539)
        daily_budget  float64   ← 7 NULLs expected
        campaign_name object

        Important
        ---------
        The ``conversion`` column contains monetary revenue, not an event count.
        This is the single largest interpretation risk in the raw data.  The
        harmonizer renames it ``revenue_attributed`` and sets
        ``meta_conversion_is_value = True`` as a metadata flag on the DataFrame.

        Returns
        -------
        pd.DataFrame
            12 columns, ~3 417 rows.

        Raises
        ------
        FileNotFoundError
            If ``meta_ads_campaign_stats.csv`` is not found.
        DataLoadError
            If the file cannot be parsed or expected columns are missing.
        """
        path = self._resolve_file(META_FILENAME, "Meta")
        df = self._read_csv(path, _META_DTYPES, "date_start", "Meta")
        self._check_columns(df, _META_EXPECTED, "Meta", path)
        return df

    def load_bing(self) -> pd.DataFrame:
        """
        Load the Bing / Microsoft Ads campaign stats CSV.

        Post-load schema
        ----------------
        CampaignId    int64
        TimePeriod    datetime64[ns]
        Revenue       float64   ← zero-inflated; 32% of rows have spend = 0
        Spend         float64
        Clicks        int64
        Impressions   int64
        Conversions   float64
        CampaignType  object
        DailyBudget   float64   ← range 10–20 only; constant per campaign
        CampaignName  object

        Note
        ----
        Bing uses PascalCase column names.  All normalization to snake_case
        canonical names happens in ``harmonizer.py``.

        Returns
        -------
        pd.DataFrame
            10 columns, ~2 873 rows.

        Raises
        ------
        FileNotFoundError
            If ``bing_campaign_stats.csv`` is not found.
        DataLoadError
            If the file cannot be parsed or expected columns are missing.
        """
        path = self._resolve_file(BING_FILENAME, "Bing")
        df = self._read_csv(path, _BING_DTYPES, "TimePeriod", "Bing")
        self._check_columns(df, _BING_EXPECTED, "Bing", path)
        return df

    # ------------------------------------------------------------------
    # Public: load all
    # ------------------------------------------------------------------
    def load_all(self) -> RawDataset:
        """
        Load all three platform CSV files and return a ``RawDataset``.

        This is the primary public interface of the loader.  Calls
        ``load_google()``, ``load_meta()``, and ``load_bing()`` in sequence
        and packages the resulting DataFrames.

        Returns
        -------
        RawDataset
            Container with ``.google``, ``.meta``, ``.bing`` DataFrames.

        Raises
        ------
        FileNotFoundError
            If any platform file is missing.
        DataLoadError
            If any file fails to parse or fails schema validation.
        """
        logger.info("=== DataLoader: starting full dataset load from '%s' ===", self.data_dir)

        google = self.load_google()
        meta   = self.load_meta()
        bing   = self.load_bing()

        dataset = RawDataset(google=google, meta=meta, bing=bing)
        logger.info("=== DataLoader: load complete ===\n%s", dataset.summary())
        return dataset


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------
def load_raw_data(data_dir: str | Path = "dataset") -> RawDataset:
    """
    Load all three platform CSV files in one call.

    This is a thin convenience wrapper around ``DataLoader(data_dir).load_all()``.
    Use this in scripts and notebooks.  Use the ``DataLoader`` class directly
    when you need per-platform loading or want to inject a custom path at
    construction time.

    Parameters
    ----------
    data_dir : str | Path
        Path to the directory containing the three platform CSV files.
        Defaults to ``"dataset"`` (relative to CWD / project root).

    Returns
    -------
    RawDataset
        Container with ``.google``, ``.meta``, ``.bing`` DataFrames.

    Raises
    ------
    FileNotFoundError
        If ``data_dir`` or any expected file does not exist.
    DataLoadError
        If any file fails to parse or fails schema validation.

    Examples
    --------
    >>> from src.data.loader import load_raw_data
    >>> ds = load_raw_data()          # uses default "dataset/" directory
    >>> ds.google.shape
    (19272, 11)
    >>> ds.meta.dtypes["conversion"]
    dtype('float64')
    >>> ds.bing["TimePeriod"].dtype
    dtype('<M8[ns]')
    >>> print(ds.summary())
    RawDataset — 25,562 total rows across 3 platforms
      Google : 19,272 rows | 11 cols | 2024-01-01 to 2026-06-04 | 92 campaigns
      Meta   :  3,417 rows | 12 cols | 2024-05-23 to 2026-06-05 | 16 campaigns
      Bing   :  2,873 rows | 10 cols | 2024-05-25 to 2026-06-05 | 28 campaigns
    """
    return DataLoader(data_dir=data_dir).load_all()
