"""
src/features/holiday_calendar.py
=================================
Ecommerce holiday calendar generator for the meridian forecasting engine.

Produces a daily DataFrame covering US Federal Holidays and key ecommerce
events (BFCM, Prime Day, Back-to-School, Holiday Season) with proximity
features and a composite intensity score.

All dates are computed algorithmically — no hardcoded yearly lookup tables.
"""

from __future__ import annotations

import bisect
import logging
import math
from datetime import date, timedelta
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weekday indices (ISO: Monday=0)
# ---------------------------------------------------------------------------
_MON: Final[int] = 0
_TUE: Final[int] = 1
_WED: Final[int] = 2
_THU: Final[int] = 3
_FRI: Final[int] = 4
_SAT: Final[int] = 5
_SUN: Final[int] = 6

# ---------------------------------------------------------------------------
# Intensity score tuning constants
# ---------------------------------------------------------------------------
_BFCM_DECAY_DAYS: Final[float] = 7.0
_PRIME_DECAY_DAYS: Final[float] = 3.0
_PRIME_BASE: Final[float] = 0.55
_HOLIDAY_SEASON_BASE: Final[float] = 0.30
_BACK_TO_SCHOOL_BASE: Final[float] = 0.25

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
CALENDAR_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "holiday_name",
    "is_holiday",
    "is_bfcm",
    "is_cyber_week",
    "is_holiday_season",
    "days_to_black_friday",
    "days_since_black_friday",
    "days_to_thanksgiving",
    "days_since_thanksgiving",
    "holiday_intensity_score",
)


# ---------------------------------------------------------------------------
# Private date helpers
# ---------------------------------------------------------------------------

def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """Return the nth occurrence of *weekday* in *year*-*month* (1-indexed).

    Parameters
    ----------
    year, month : int
    weekday : int  — 0=Monday … 6=Sunday
    n : int  — 1=first, 2=second, …

    Raises
    ------
    ValueError
        If the nth occurrence does not exist in the given month.
    """
    first_day = date(year, month, 1)
    days_ahead = (weekday - first_day.weekday()) % 7
    first_occurrence = first_day + timedelta(days=days_ahead)
    target = first_occurrence + timedelta(weeks=n - 1)
    if target.month != month:
        raise ValueError(
            f"No {n}-th weekday={weekday} exists in {year}-{month:02d}"
        )
    return target


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """Return the last occurrence of *weekday* in *year*-*month*."""
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
    days_back = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=days_back)


# ---------------------------------------------------------------------------
# Public holiday date functions
# ---------------------------------------------------------------------------

def thanksgiving(year: int) -> date:
    """4th Thursday of November for *year*."""
    return _nth_weekday_of_month(year, 11, _THU, 4)


def black_friday(year: int) -> date:
    """Day after Thanksgiving (always a Friday)."""
    return thanksgiving(year) + timedelta(days=1)


def cyber_monday(year: int) -> date:
    """Monday after Black Friday (Black Friday + 3 days)."""
    return black_friday(year) + timedelta(days=3)


def prime_day(year: int) -> date:
    """Approximate Prime Day: 3rd Tuesday of July.

    Amazon Prime Day is typically held in mid-July.  The 3rd Tuesday of July
    is used as a stable algorithmic approximation for forecasting purposes.
    """
    return _nth_weekday_of_month(year, 7, _TUE, 3)


def cyber_week_dates(year: int) -> frozenset[date]:
    """Return the 5-day Cyber Week window: Thanksgiving through Cyber Monday."""
    td = thanksgiving(year)
    cm = cyber_monday(year)
    days: list[date] = []
    d = td
    while d <= cm:
        days.append(d)
        d += timedelta(days=1)
    return frozenset(days)


def federal_holidays(year: int) -> dict[date, str]:
    """Return all US Federal Holidays for *year* as {date: name}.

    Computed algorithmically:
    - Fixed dates: New Year's Day, Independence Day, Veterans Day, Christmas
    - Nth-weekday rules: MLK (3rd Mon Jan), Presidents Day (3rd Mon Feb),
      Memorial Day (last Mon May), Labor Day (1st Mon Sep),
      Columbus Day (2nd Mon Oct), Thanksgiving (4th Thu Nov)
    """
    holidays: dict[date, str] = {
        # Fixed-date holidays
        date(year, 1, 1): "New Year's Day",
        date(year, 7, 4): "Independence Day",
        date(year, 11, 11): "Veterans Day",
        date(year, 12, 25): "Christmas Day",
        # Calculated holidays
        _nth_weekday_of_month(year, 1, _MON, 3): "Martin Luther King Jr. Day",
        _nth_weekday_of_month(year, 2, _MON, 3): "Presidents Day",
        _last_weekday_of_month(year, 5, _MON): "Memorial Day",
        _nth_weekday_of_month(year, 9, _MON, 1): "Labor Day",
        _nth_weekday_of_month(year, 10, _MON, 2): "Columbus Day",
        thanksgiving(year): "Thanksgiving Day",
    }
    return holidays


# ---------------------------------------------------------------------------
# Intensity score
# ---------------------------------------------------------------------------

def holiday_intensity_score(
    d: date,
    all_black_fridays: list[date],
    all_prime_days: list[date],
) -> float:
    """Compute a 0.0–1.0 holiday intensity score for date *d*.

    Components (max taken across all):
    - BFCM proximity: exponential decay from nearest Black Friday
      (half-life = 7 days; peaks at 1.0 on Black Friday itself)
    - Prime Day proximity: 0.55 × exp(-dist/3) (peaks at 0.55 on Prime Day)
    - Holiday season (Nov–Dec): flat 0.30 baseline
    - Back to School (Aug 1–Sep 15): flat 0.25 baseline

    Parameters
    ----------
    d : date
    all_black_fridays : list[date]  — sorted, may span multiple years
    all_prime_days : list[date]  — sorted, may span multiple years
    """
    # BFCM proximity
    bfcm_score = 0.0
    if all_black_fridays:
        min_dist = min(abs((d - bf).days) for bf in all_black_fridays)
        bfcm_score = math.exp(-min_dist / _BFCM_DECAY_DAYS)

    # Prime Day proximity
    prime_score = 0.0
    if all_prime_days:
        min_dist_p = min(abs((d - pd).days) for pd in all_prime_days)
        prime_score = _PRIME_BASE * math.exp(-min_dist_p / _PRIME_DECAY_DAYS)

    # Seasonal baselines
    holiday_season_score = _HOLIDAY_SEASON_BASE if d.month in (11, 12) else 0.0
    bts_score = _BACK_TO_SCHOOL_BASE if (
        d.month == 8 or (d.month == 9 and d.day <= 15)
    ) else 0.0

    raw = max(bfcm_score, prime_score, holiday_season_score, bts_score)
    return round(min(1.0, max(0.0, raw)), 4)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_years(start_year: int, end_year: int) -> None:
    if not isinstance(start_year, int) or not isinstance(end_year, int):
        raise TypeError(
            f"start_year and end_year must be int, got "
            f"{type(start_year).__name__} and {type(end_year).__name__}"
        )
    if start_year > end_year:
        raise ValueError(
            f"start_year ({start_year}) must be <= end_year ({end_year})"
        )
    if end_year > 2100:
        raise ValueError(f"end_year ({end_year}) exceeds supported maximum of 2100")


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_holiday_calendar(
    start_year: int = 2024,
    end_year: int = 2027,
) -> pd.DataFrame:
    """Build a daily holiday calendar DataFrame from January 1 of *start_year*
    through December 31 of *end_year*.

    Parameters
    ----------
    start_year : int, default 2024
    end_year : int, default 2027

    Returns
    -------
    pd.DataFrame
        One row per calendar date with columns defined in CALENDAR_COLUMNS.
        ``date`` column is datetime64[ns].  All boolean columns are bool.
        ``holiday_intensity_score`` is float64 in [0.0, 1.0].

    Raises
    ------
    TypeError
        If *start_year* or *end_year* are not int.
    ValueError
        If *start_year* > *end_year* or *end_year* > 2100.
    """
    _validate_years(start_year, end_year)

    logger.info(
        "Building holiday calendar %d-%d", start_year, end_year
    )

    # Extended year range (±1) ensures days_to/days_since are always defined
    # for dates at the edges of the requested window.
    ext_years = range(start_year - 1, end_year + 2)

    # Sorted anchor date lists for O(log n) binary search
    bf_list: list[date] = sorted(black_friday(y) for y in ext_years)
    td_list: list[date] = sorted(thanksgiving(y) for y in ext_years)
    pd_list: list[date] = sorted(prime_day(y) for y in ext_years)

    # Sets for O(1) membership tests
    bf_set: frozenset[date] = frozenset(bf_list)
    cm_set: frozenset[date] = frozenset(cyber_monday(y) for y in ext_years)

    cyber_week_set: set[date] = set()
    for y in ext_years:
        cyber_week_set.update(cyber_week_dates(y))

    # Federal holidays only for the requested range (no need for extended years)
    all_fed: dict[date, str] = {}
    for y in range(start_year, end_year + 1):
        all_fed.update(federal_holidays(y))

    # Black Friday and Prime Day lists clipped to ±1 year for intensity score
    # (only closest events matter for exponential decay; distant ones contribute ~0)
    bf_for_intensity: list[date] = bf_list
    pd_for_intensity: list[date] = pd_list

    # Generate full date range
    start_date = date(start_year, 1, 1)
    end_date = date(end_year, 12, 31)
    n_days = (end_date - start_date).days + 1

    rows: list[dict] = []

    for i in range(n_days):
        d = start_date + timedelta(days=i)

        # ---- holiday_name ----
        name = all_fed.get(d, "")
        if d in bf_set:
            name = "Black Friday"
        elif d in cm_set:
            name = "Cyber Monday"
        elif d in frozenset(pd_list):
            if not name:  # don't overwrite a federal holiday name
                name = "Prime Day (approx)"

        # ---- boolean flags ----
        is_holiday = d in all_fed
        is_bfcm = d in bf_set or d in cm_set
        is_cyber_week = d in cyber_week_set
        is_holiday_season = d.month in (11, 12)

        # ---- days_to_black_friday ----
        idx_future_bf = bisect.bisect_left(bf_list, d)
        days_to_bf = (bf_list[idx_future_bf] - d).days if idx_future_bf < len(bf_list) else 0

        # ---- days_since_black_friday ----
        idx_past_bf = bisect.bisect_right(bf_list, d) - 1
        days_since_bf = (d - bf_list[idx_past_bf]).days if idx_past_bf >= 0 else 0

        # ---- days_to_thanksgiving ----
        idx_future_td = bisect.bisect_left(td_list, d)
        days_to_td = (td_list[idx_future_td] - d).days if idx_future_td < len(td_list) else 0

        # ---- days_since_thanksgiving ----
        idx_past_td = bisect.bisect_right(td_list, d) - 1
        days_since_td = (d - td_list[idx_past_td]).days if idx_past_td >= 0 else 0

        # ---- intensity score ----
        intensity = holiday_intensity_score(d, bf_for_intensity, pd_for_intensity)

        rows.append(
            {
                "date": d,
                "holiday_name": name,
                "is_holiday": is_holiday,
                "is_bfcm": is_bfcm,
                "is_cyber_week": is_cyber_week,
                "is_holiday_season": is_holiday_season,
                "days_to_black_friday": days_to_bf,
                "days_since_black_friday": days_since_bf,
                "days_to_thanksgiving": days_to_td,
                "days_since_thanksgiving": days_since_td,
                "holiday_intensity_score": intensity,
            }
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    # pandas 2.0+ infers pd.StringDtype for all-string columns built from dicts;
    # tests assert dtype == object (the pre-2.0 behaviour).
    df["holiday_name"] = df["holiday_name"].astype(object)

    for col in ("is_holiday", "is_bfcm", "is_cyber_week", "is_holiday_season"):
        df[col] = df[col].astype(bool)

    for col in (
        "days_to_black_friday",
        "days_since_black_friday",
        "days_to_thanksgiving",
        "days_since_thanksgiving",
    ):
        df[col] = df[col].astype(int)

    df["holiday_intensity_score"] = df["holiday_intensity_score"].astype(float)

    logger.info(
        "Holiday calendar built: %d rows, %d holidays, %d BFCM days",
        len(df),
        df["is_holiday"].sum(),
        df["is_bfcm"].sum(),
    )

    return df[list(CALENDAR_COLUMNS)]
