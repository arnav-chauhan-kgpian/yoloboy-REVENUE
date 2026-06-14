"""
tests/test_holiday_calendar.py
================================
Comprehensive pytest suite for src/features/holiday_calendar.py.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Final

import pandas as pd
import pytest

from src.features.holiday_calendar import (
    CALENDAR_COLUMNS,
    black_friday,
    build_holiday_calendar,
    cyber_monday,
    cyber_week_dates,
    federal_holidays,
    holiday_intensity_score,
    prime_day,
    thanksgiving,
)

# ---------------------------------------------------------------------------
# Forensic constants — verified against algorithmic computation
# ---------------------------------------------------------------------------

# Thanksgiving = 4th Thursday of November
THANKSGIVING_2024: Final[date] = date(2024, 11, 28)
THANKSGIVING_2025: Final[date] = date(2025, 11, 27)
THANKSGIVING_2026: Final[date] = date(2026, 11, 26)
THANKSGIVING_2027: Final[date] = date(2027, 11, 25)

BLACK_FRIDAY_2024: Final[date] = date(2024, 11, 29)
BLACK_FRIDAY_2025: Final[date] = date(2025, 11, 28)
BLACK_FRIDAY_2026: Final[date] = date(2026, 11, 27)
BLACK_FRIDAY_2027: Final[date] = date(2027, 11, 26)

CYBER_MONDAY_2024: Final[date] = date(2024, 12, 2)
CYBER_MONDAY_2025: Final[date] = date(2025, 12, 1)
CYBER_MONDAY_2026: Final[date] = date(2026, 11, 30)
CYBER_MONDAY_2027: Final[date] = date(2027, 11, 29)

TOTAL_DAYS_2024_2027: Final[int] = 366 + 365 + 365 + 365  # 2024 is a leap year
FEDERAL_HOLIDAYS_PER_YEAR: Final[int] = 10
TOTAL_FEDERAL_HOLIDAY_DAYS: Final[int] = FEDERAL_HOLIDAYS_PER_YEAR * 4  # 4 years
TOTAL_BFCM_DAYS: Final[int] = 8  # 2 per year × 4 years

# Module-scoped fixture — build once for all tests
@pytest.fixture(scope="module")
def cal() -> pd.DataFrame:
    return build_holiday_calendar(2024, 2027)


# ===========================================================================
# TestThanksgivingCalculation
# ===========================================================================

class TestThanksgivingCalculation:
    """Thanksgiving must be the 4th Thursday of November for each year."""

    @pytest.mark.parametrize("year,expected", [
        (2024, THANKSGIVING_2024),
        (2025, THANKSGIVING_2025),
        (2026, THANKSGIVING_2026),
        (2027, THANKSGIVING_2027),
    ])
    def test_thanksgiving_date(self, year: int, expected: date) -> None:
        assert thanksgiving(year) == expected

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_thanksgiving_is_thursday(self, year: int) -> None:
        assert thanksgiving(year).weekday() == 3  # Thursday=3

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_thanksgiving_is_in_november(self, year: int) -> None:
        assert thanksgiving(year).month == 11

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_thanksgiving_is_in_last_week_of_november(self, year: int) -> None:
        # 4th Thursday is always between Nov 22 and Nov 28 (inclusive)
        td = thanksgiving(year)
        assert 22 <= td.day <= 28

    def test_thanksgiving_future_year(self) -> None:
        # Verify the function works for future years beyond our standard range
        td_2030 = thanksgiving(2030)
        assert td_2030.weekday() == 3
        assert td_2030.month == 11
        assert 22 <= td_2030.day <= 28


# ===========================================================================
# TestBlackFridayCalculation
# ===========================================================================

class TestBlackFridayCalculation:
    """Black Friday must be the day after Thanksgiving, always a Friday."""

    @pytest.mark.parametrize("year,expected", [
        (2024, BLACK_FRIDAY_2024),
        (2025, BLACK_FRIDAY_2025),
        (2026, BLACK_FRIDAY_2026),
        (2027, BLACK_FRIDAY_2027),
    ])
    def test_black_friday_date(self, year: int, expected: date) -> None:
        assert black_friday(year) == expected

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_black_friday_is_friday(self, year: int) -> None:
        assert black_friday(year).weekday() == 4  # Friday=4

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_black_friday_is_day_after_thanksgiving(self, year: int) -> None:
        assert black_friday(year) == thanksgiving(year) + timedelta(days=1)

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_black_friday_month(self, year: int) -> None:
        # BF is always in November
        assert black_friday(year).month == 11

    def test_black_friday_future_year(self) -> None:
        bf_2030 = black_friday(2030)
        assert bf_2030.weekday() == 4
        assert bf_2030 == thanksgiving(2030) + timedelta(days=1)


# ===========================================================================
# TestCyberMondayCalculation
# ===========================================================================

class TestCyberMondayCalculation:
    """Cyber Monday must be the Monday after Black Friday (BF + 3 days)."""

    @pytest.mark.parametrize("year,expected", [
        (2024, CYBER_MONDAY_2024),
        (2025, CYBER_MONDAY_2025),
        (2026, CYBER_MONDAY_2026),
        (2027, CYBER_MONDAY_2027),
    ])
    def test_cyber_monday_date(self, year: int, expected: date) -> None:
        assert cyber_monday(year) == expected

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_cyber_monday_is_monday(self, year: int) -> None:
        assert cyber_monday(year).weekday() == 0  # Monday=0

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_cyber_monday_is_bf_plus_3(self, year: int) -> None:
        assert cyber_monday(year) == black_friday(year) + timedelta(days=3)

    def test_cyber_monday_2024_is_in_december(self) -> None:
        assert cyber_monday(2024).month == 12

    def test_cyber_monday_2026_is_in_november(self) -> None:
        # 2026 BF is Nov 27; Nov 27 + 3 = Nov 30
        assert cyber_monday(2026).month == 11

    def test_cyber_monday_future_year(self) -> None:
        cm_2030 = cyber_monday(2030)
        assert cm_2030.weekday() == 0
        assert cm_2030 == black_friday(2030) + timedelta(days=3)


# ===========================================================================
# TestCyberWeekWindow
# ===========================================================================

class TestCyberWeekWindow:
    """Cyber Week = Thanksgiving through Cyber Monday, inclusive (5 days)."""

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_cyber_week_length(self, year: int) -> None:
        cw = cyber_week_dates(year)
        assert len(cw) == 5

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_cyber_week_starts_on_thanksgiving(self, year: int) -> None:
        cw = cyber_week_dates(year)
        assert thanksgiving(year) in cw

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_cyber_week_includes_black_friday(self, year: int) -> None:
        cw = cyber_week_dates(year)
        assert black_friday(year) in cw

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_cyber_week_ends_on_cyber_monday(self, year: int) -> None:
        cw = cyber_week_dates(year)
        assert cyber_monday(year) in cw

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_cyber_week_is_consecutive_days(self, year: int) -> None:
        cw = sorted(cyber_week_dates(year))
        for a, b in zip(cw, cw[1:]):
            assert (b - a).days == 1

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_day_before_thanksgiving_not_in_cyber_week(self, year: int) -> None:
        cw = cyber_week_dates(year)
        assert (thanksgiving(year) - timedelta(days=1)) not in cw

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_day_after_cyber_monday_not_in_cyber_week(self, year: int) -> None:
        cw = cyber_week_dates(year)
        assert (cyber_monday(year) + timedelta(days=1)) not in cw


# ===========================================================================
# TestFederalHolidays
# ===========================================================================

class TestFederalHolidays:
    """Verify US Federal Holidays are computed correctly."""

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_federal_holiday_count(self, year: int) -> None:
        assert len(federal_holidays(year)) == FEDERAL_HOLIDAYS_PER_YEAR

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_new_years_day(self, year: int) -> None:
        fh = federal_holidays(year)
        assert date(year, 1, 1) in fh
        assert "New Year" in fh[date(year, 1, 1)]

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_independence_day(self, year: int) -> None:
        fh = federal_holidays(year)
        assert date(year, 7, 4) in fh

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_veterans_day(self, year: int) -> None:
        fh = federal_holidays(year)
        assert date(year, 11, 11) in fh

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_christmas(self, year: int) -> None:
        fh = federal_holidays(year)
        assert date(year, 12, 25) in fh

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_thanksgiving_in_federal_holidays(self, year: int) -> None:
        fh = federal_holidays(year)
        assert thanksgiving(year) in fh

    def test_mlk_2024(self) -> None:
        fh = federal_holidays(2024)
        mlk = date(2024, 1, 15)
        assert mlk in fh
        assert mlk.weekday() == 0  # Monday

    def test_memorial_day_2024(self) -> None:
        fh = federal_holidays(2024)
        mem = date(2024, 5, 27)
        assert mem in fh
        assert mem.weekday() == 0  # Monday

    def test_labor_day_2024(self) -> None:
        fh = federal_holidays(2024)
        labor = date(2024, 9, 2)
        assert labor in fh
        assert labor.weekday() == 0  # Monday

    def test_presidents_day_2025(self) -> None:
        fh = federal_holidays(2025)
        pres = date(2025, 2, 17)
        assert pres in fh
        assert pres.weekday() == 0  # Monday


# ===========================================================================
# TestPrimeDay
# ===========================================================================

class TestPrimeDay:
    """Prime Day approximation: 3rd Tuesday of July."""

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_prime_day_is_tuesday(self, year: int) -> None:
        assert prime_day(year).weekday() == 1  # Tuesday

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_prime_day_is_in_july(self, year: int) -> None:
        assert prime_day(year).month == 7

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
    def test_prime_day_is_in_mid_july(self, year: int) -> None:
        # 3rd Tuesday of July: day 15–21
        assert 15 <= prime_day(year).day <= 21

    def test_prime_day_2024_specific(self) -> None:
        assert prime_day(2024) == date(2024, 7, 16)

    def test_prime_day_2025_specific(self) -> None:
        assert prime_day(2025) == date(2025, 7, 15)


# ===========================================================================
# TestHolidayIntensityScore
# ===========================================================================

class TestHolidayIntensityScore:
    """holiday_intensity_score must peak at BF=1.0, decay correctly."""

    @pytest.fixture
    def bf_list(self) -> list[date]:
        return [black_friday(y) for y in range(2023, 2029)]

    @pytest.fixture
    def pd_list(self) -> list[date]:
        return [prime_day(y) for y in range(2023, 2029)]

    def test_black_friday_score_is_one(self, bf_list, pd_list) -> None:
        score = holiday_intensity_score(BLACK_FRIDAY_2024, bf_list, pd_list)
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_cyber_monday_score_high(self, bf_list, pd_list) -> None:
        # CM is 3 days after BF: exp(-3/7) ≈ 0.651
        score = holiday_intensity_score(CYBER_MONDAY_2024, bf_list, pd_list)
        assert score > 0.6

    def test_thanksgiving_score_high(self, bf_list, pd_list) -> None:
        # Thanksgiving is 1 day before BF: exp(-1/7) ≈ 0.867
        score = holiday_intensity_score(THANKSGIVING_2024, bf_list, pd_list)
        assert score > 0.8

    def test_cyber_week_score_elevated(self, bf_list, pd_list) -> None:
        # Saturday of cyber week (BF+1)
        sat = BLACK_FRIDAY_2024 + timedelta(days=1)
        score = holiday_intensity_score(sat, bf_list, pd_list)
        assert score > 0.8

    def test_prime_day_score_significant(self, bf_list, pd_list) -> None:
        # On Prime Day, intensity should be 0.55
        pd_2024 = prime_day(2024)
        score = holiday_intensity_score(pd_2024, bf_list, pd_list)
        assert score == pytest.approx(0.55, abs=0.01)

    def test_mid_summer_score_low(self, bf_list, pd_list) -> None:
        # June 15 is far from any event
        d = date(2024, 6, 15)
        score = holiday_intensity_score(d, bf_list, pd_list)
        assert score < 0.1

    def test_back_to_school_score(self, bf_list, pd_list) -> None:
        # August 1 → back_to_school_base = 0.25
        aug1 = date(2024, 8, 1)
        score = holiday_intensity_score(aug1, bf_list, pd_list)
        assert score >= 0.25

    def test_holiday_season_score(self, bf_list, pd_list) -> None:
        # November 1 (before BFCM proximity kicks in meaningfully)
        nov1 = date(2024, 11, 1)
        score = holiday_intensity_score(nov1, bf_list, pd_list)
        # holiday_season_base = 0.30; BF is 28 days away → exp(-28/7)≈0.018
        assert score == pytest.approx(0.30, abs=0.01)

    def test_score_bounded_zero_to_one(self, bf_list, pd_list) -> None:
        for year in [2024, 2025, 2026, 2027]:
            for month in range(1, 13):
                d = date(year, month, 1)
                s = holiday_intensity_score(d, bf_list, pd_list)
                assert 0.0 <= s <= 1.0, f"Score {s} out of bounds for {d}"

    def test_score_is_float(self, bf_list, pd_list) -> None:
        score = holiday_intensity_score(BLACK_FRIDAY_2024, bf_list, pd_list)
        assert isinstance(score, float)

    def test_empty_lists_return_zero(self) -> None:
        score = holiday_intensity_score(date(2024, 6, 1), [], [])
        assert score == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# TestCalendarSchema
# ===========================================================================

class TestCalendarSchema:
    """DataFrame structure, dtypes, and row count."""

    def test_row_count(self, cal: pd.DataFrame) -> None:
        assert len(cal) == TOTAL_DAYS_2024_2027

    def test_all_columns_present(self, cal: pd.DataFrame) -> None:
        assert list(cal.columns) == list(CALENDAR_COLUMNS)

    def test_date_dtype(self, cal: pd.DataFrame) -> None:
        assert pd.api.types.is_datetime64_any_dtype(cal["date"])

    def test_boolean_dtypes(self, cal: pd.DataFrame) -> None:
        for col in ("is_holiday", "is_bfcm", "is_cyber_week", "is_holiday_season"):
            assert cal[col].dtype == bool, f"{col} must be bool"

    def test_integer_dtypes(self, cal: pd.DataFrame) -> None:
        for col in (
            "days_to_black_friday",
            "days_since_black_friday",
            "days_to_thanksgiving",
            "days_since_thanksgiving",
        ):
            assert pd.api.types.is_integer_dtype(cal[col]), f"{col} must be int"

    def test_intensity_score_dtype(self, cal: pd.DataFrame) -> None:
        assert pd.api.types.is_float_dtype(cal["holiday_intensity_score"])

    def test_no_duplicate_dates(self, cal: pd.DataFrame) -> None:
        assert cal["date"].is_unique

    def test_dates_sorted(self, cal: pd.DataFrame) -> None:
        assert cal["date"].is_monotonic_increasing

    def test_date_range_starts_jan1_2024(self, cal: pd.DataFrame) -> None:
        assert cal["date"].iloc[0] == pd.Timestamp("2024-01-01")

    def test_date_range_ends_dec31_2027(self, cal: pd.DataFrame) -> None:
        assert cal["date"].iloc[-1] == pd.Timestamp("2027-12-31")

    def test_no_null_values(self, cal: pd.DataFrame) -> None:
        assert cal.isnull().sum().sum() == 0

    def test_holiday_name_column_is_string(self, cal: pd.DataFrame) -> None:
        assert cal["holiday_name"].dtype == object


# ===========================================================================
# TestBFCMFlags
# ===========================================================================

class TestBFCMFlags:
    """is_bfcm must be True on BF and CM, False elsewhere."""

    def test_total_bfcm_days(self, cal: pd.DataFrame) -> None:
        assert cal["is_bfcm"].sum() == TOTAL_BFCM_DAYS

    @pytest.mark.parametrize("bf_date", [
        BLACK_FRIDAY_2024, BLACK_FRIDAY_2025, BLACK_FRIDAY_2026, BLACK_FRIDAY_2027
    ])
    def test_is_bfcm_on_black_friday(self, cal: pd.DataFrame, bf_date: date) -> None:
        row = cal[cal["date"] == pd.Timestamp(bf_date)]
        assert row["is_bfcm"].iloc[0] == True

    @pytest.mark.parametrize("cm_date", [
        CYBER_MONDAY_2024, CYBER_MONDAY_2025, CYBER_MONDAY_2026, CYBER_MONDAY_2027
    ])
    def test_is_bfcm_on_cyber_monday(self, cal: pd.DataFrame, cm_date: date) -> None:
        row = cal[cal["date"] == pd.Timestamp(cm_date)]
        assert row["is_bfcm"].iloc[0] == True

    @pytest.mark.parametrize("td_date", [
        THANKSGIVING_2024, THANKSGIVING_2025, THANKSGIVING_2026, THANKSGIVING_2027
    ])
    def test_is_bfcm_false_on_thanksgiving(self, cal: pd.DataFrame, td_date: date) -> None:
        row = cal[cal["date"] == pd.Timestamp(td_date)]
        assert row["is_bfcm"].iloc[0] == False

    def test_is_bfcm_false_on_christmas(self, cal: pd.DataFrame) -> None:
        row = cal[cal["date"] == pd.Timestamp("2024-12-25")]
        assert row["is_bfcm"].iloc[0] == False

    def test_black_friday_name_in_holiday_name_column(self, cal: pd.DataFrame) -> None:
        row = cal[cal["date"] == pd.Timestamp(BLACK_FRIDAY_2024)]
        assert row["holiday_name"].iloc[0] == "Black Friday"

    def test_cyber_monday_name_in_holiday_name_column(self, cal: pd.DataFrame) -> None:
        row = cal[cal["date"] == pd.Timestamp(CYBER_MONDAY_2024)]
        assert row["holiday_name"].iloc[0] == "Cyber Monday"


# ===========================================================================
# TestCyberWeekFlags
# ===========================================================================

class TestCyberWeekFlags:
    """is_cyber_week covers exactly Thanksgiving through Cyber Monday."""

    def test_total_cyber_week_days(self, cal: pd.DataFrame) -> None:
        # 5 days per year × 4 years = 20 days
        assert cal["is_cyber_week"].sum() == 5 * 4

    @pytest.mark.parametrize("td_date", [
        THANKSGIVING_2024, THANKSGIVING_2025, THANKSGIVING_2026, THANKSGIVING_2027
    ])
    def test_thanksgiving_in_cyber_week(self, cal: pd.DataFrame, td_date: date) -> None:
        row = cal[cal["date"] == pd.Timestamp(td_date)]
        assert row["is_cyber_week"].iloc[0] == True

    @pytest.mark.parametrize("bf_date", [
        BLACK_FRIDAY_2024, BLACK_FRIDAY_2025, BLACK_FRIDAY_2026, BLACK_FRIDAY_2027
    ])
    def test_black_friday_in_cyber_week(self, cal: pd.DataFrame, bf_date: date) -> None:
        row = cal[cal["date"] == pd.Timestamp(bf_date)]
        assert row["is_cyber_week"].iloc[0] == True

    @pytest.mark.parametrize("cm_date", [
        CYBER_MONDAY_2024, CYBER_MONDAY_2025, CYBER_MONDAY_2026, CYBER_MONDAY_2027
    ])
    def test_cyber_monday_in_cyber_week(self, cal: pd.DataFrame, cm_date: date) -> None:
        row = cal[cal["date"] == pd.Timestamp(cm_date)]
        assert row["is_cyber_week"].iloc[0] == True

    def test_day_before_thanksgiving_not_cyber_week(self, cal: pd.DataFrame) -> None:
        d = THANKSGIVING_2024 - timedelta(days=1)
        row = cal[cal["date"] == pd.Timestamp(d)]
        assert row["is_cyber_week"].iloc[0] == False

    def test_day_after_cyber_monday_not_cyber_week(self, cal: pd.DataFrame) -> None:
        d = CYBER_MONDAY_2024 + timedelta(days=1)
        row = cal[cal["date"] == pd.Timestamp(d)]
        assert row["is_cyber_week"].iloc[0] == False


# ===========================================================================
# TestHolidaySeasonFlag
# ===========================================================================

class TestHolidaySeasonFlag:
    """is_holiday_season must cover November 1 through December 31."""

    def test_november_1_is_holiday_season(self, cal: pd.DataFrame) -> None:
        row = cal[cal["date"] == pd.Timestamp("2024-11-01")]
        assert row["is_holiday_season"].iloc[0] == True

    def test_december_31_is_holiday_season(self, cal: pd.DataFrame) -> None:
        row = cal[cal["date"] == pd.Timestamp("2024-12-31")]
        assert row["is_holiday_season"].iloc[0] == True

    def test_october_31_not_holiday_season(self, cal: pd.DataFrame) -> None:
        row = cal[cal["date"] == pd.Timestamp("2024-10-31")]
        assert row["is_holiday_season"].iloc[0] == False

    def test_january_1_not_holiday_season(self, cal: pd.DataFrame) -> None:
        row = cal[cal["date"] == pd.Timestamp("2025-01-01")]
        assert row["is_holiday_season"].iloc[0] == False

    def test_holiday_season_day_count_per_year(self, cal: pd.DataFrame) -> None:
        # Nov=30, Dec=31 = 61 days per year × 4 years = 244
        assert cal["is_holiday_season"].sum() == 61 * 4


# ===========================================================================
# TestDaysToFromBlackFriday
# ===========================================================================

class TestDaysToFromBlackFriday:
    """Verify days_to_black_friday and days_since_black_friday."""

    def test_days_to_zero_on_black_friday(self, cal: pd.DataFrame) -> None:
        row = cal[cal["date"] == pd.Timestamp(BLACK_FRIDAY_2024)]
        assert row["days_to_black_friday"].iloc[0] == 0

    def test_days_since_zero_on_black_friday(self, cal: pd.DataFrame) -> None:
        row = cal[cal["date"] == pd.Timestamp(BLACK_FRIDAY_2024)]
        assert row["days_since_black_friday"].iloc[0] == 0

    def test_days_to_one_day_before_black_friday(self, cal: pd.DataFrame) -> None:
        d = BLACK_FRIDAY_2024 - timedelta(days=1)
        row = cal[cal["date"] == pd.Timestamp(d)]
        assert row["days_to_black_friday"].iloc[0] == 1

    def test_days_since_one_day_after_black_friday(self, cal: pd.DataFrame) -> None:
        d = BLACK_FRIDAY_2024 + timedelta(days=1)
        row = cal[cal["date"] == pd.Timestamp(d)]
        assert row["days_since_black_friday"].iloc[0] == 1

    def test_days_to_nonnegative(self, cal: pd.DataFrame) -> None:
        assert (cal["days_to_black_friday"] >= 0).all()

    def test_days_since_nonnegative(self, cal: pd.DataFrame) -> None:
        assert (cal["days_since_black_friday"] >= 0).all()

    def test_cyber_monday_days_since(self, cal: pd.DataFrame) -> None:
        # CM 2024 is 3 days after BF 2024
        row = cal[cal["date"] == pd.Timestamp(CYBER_MONDAY_2024)]
        assert row["days_since_black_friday"].iloc[0] == 3

    def test_thanksgiving_days_to(self, cal: pd.DataFrame) -> None:
        # Thanksgiving is 1 day before BF
        row = cal[cal["date"] == pd.Timestamp(THANKSGIVING_2024)]
        assert row["days_to_black_friday"].iloc[0] == 1


# ===========================================================================
# TestDaysToFromThanksgiving
# ===========================================================================

class TestDaysToFromThanksgiving:
    """Verify days_to_thanksgiving and days_since_thanksgiving."""

    def test_days_to_zero_on_thanksgiving(self, cal: pd.DataFrame) -> None:
        row = cal[cal["date"] == pd.Timestamp(THANKSGIVING_2024)]
        assert row["days_to_thanksgiving"].iloc[0] == 0

    def test_days_since_zero_on_thanksgiving(self, cal: pd.DataFrame) -> None:
        row = cal[cal["date"] == pd.Timestamp(THANKSGIVING_2024)]
        assert row["days_since_thanksgiving"].iloc[0] == 0

    def test_days_to_one_day_before_thanksgiving(self, cal: pd.DataFrame) -> None:
        d = THANKSGIVING_2024 - timedelta(days=1)
        row = cal[cal["date"] == pd.Timestamp(d)]
        assert row["days_to_thanksgiving"].iloc[0] == 1

    def test_days_since_one_day_after_thanksgiving(self, cal: pd.DataFrame) -> None:
        d = THANKSGIVING_2024 + timedelta(days=1)
        row = cal[cal["date"] == pd.Timestamp(d)]
        assert row["days_since_thanksgiving"].iloc[0] == 1

    def test_days_to_nonnegative(self, cal: pd.DataFrame) -> None:
        assert (cal["days_to_thanksgiving"] >= 0).all()

    def test_days_since_nonnegative(self, cal: pd.DataFrame) -> None:
        assert (cal["days_since_thanksgiving"] >= 0).all()

    def test_black_friday_days_since_thanksgiving(self, cal: pd.DataFrame) -> None:
        # BF is 1 day after Thanksgiving
        row = cal[cal["date"] == pd.Timestamp(BLACK_FRIDAY_2024)]
        assert row["days_since_thanksgiving"].iloc[0] == 1


# ===========================================================================
# TestIsHolidayFlag
# ===========================================================================

class TestIsHolidayFlag:
    """is_holiday must track US Federal Holidays only."""

    def test_thanksgiving_is_holiday(self, cal: pd.DataFrame) -> None:
        row = cal[cal["date"] == pd.Timestamp(THANKSGIVING_2024)]
        assert row["is_holiday"].iloc[0] == True

    def test_black_friday_is_not_holiday(self, cal: pd.DataFrame) -> None:
        # BF is an ecommerce event, not a federal holiday
        row = cal[cal["date"] == pd.Timestamp(BLACK_FRIDAY_2024)]
        assert row["is_holiday"].iloc[0] == False

    def test_cyber_monday_is_not_holiday(self, cal: pd.DataFrame) -> None:
        row = cal[cal["date"] == pd.Timestamp(CYBER_MONDAY_2024)]
        assert row["is_holiday"].iloc[0] == False

    def test_christmas_is_holiday(self, cal: pd.DataFrame) -> None:
        row = cal[cal["date"] == pd.Timestamp("2024-12-25")]
        assert row["is_holiday"].iloc[0] == True

    def test_new_years_day_is_holiday(self, cal: pd.DataFrame) -> None:
        row = cal[cal["date"] == pd.Timestamp("2024-01-01")]
        assert row["is_holiday"].iloc[0] == True

    def test_total_holiday_days(self, cal: pd.DataFrame) -> None:
        # 10 federal holidays × 4 years = 40
        assert cal["is_holiday"].sum() == TOTAL_FEDERAL_HOLIDAY_DAYS

    def test_thanksgiving_holiday_name(self, cal: pd.DataFrame) -> None:
        row = cal[cal["date"] == pd.Timestamp(THANKSGIVING_2024)]
        assert "Thanksgiving" in row["holiday_name"].iloc[0]


# ===========================================================================
# TestCalendarIntensityInDataFrame
# ===========================================================================

class TestCalendarIntensityInDataFrame:
    """Verify intensity score values in the built DataFrame."""

    def test_black_friday_intensity_in_df(self, cal: pd.DataFrame) -> None:
        row = cal[cal["date"] == pd.Timestamp(BLACK_FRIDAY_2024)]
        assert row["holiday_intensity_score"].iloc[0] == pytest.approx(1.0, abs=1e-3)

    def test_cyber_monday_intensity_above_0_6(self, cal: pd.DataFrame) -> None:
        row = cal[cal["date"] == pd.Timestamp(CYBER_MONDAY_2024)]
        assert row["holiday_intensity_score"].iloc[0] > 0.6

    def test_thanksgiving_intensity_above_0_8(self, cal: pd.DataFrame) -> None:
        row = cal[cal["date"] == pd.Timestamp(THANKSGIVING_2024)]
        assert row["holiday_intensity_score"].iloc[0] > 0.8

    def test_june_intensity_near_zero(self, cal: pd.DataFrame) -> None:
        row = cal[cal["date"] == pd.Timestamp("2024-06-15")]
        assert row["holiday_intensity_score"].iloc[0] < 0.1

    def test_august_intensity_above_0_2(self, cal: pd.DataFrame) -> None:
        # Back to School season
        row = cal[cal["date"] == pd.Timestamp("2024-08-01")]
        assert row["holiday_intensity_score"].iloc[0] >= 0.25

    def test_november_1_intensity(self, cal: pd.DataFrame) -> None:
        # holiday_season_base = 0.30
        row = cal[cal["date"] == pd.Timestamp("2024-11-01")]
        assert row["holiday_intensity_score"].iloc[0] == pytest.approx(0.30, abs=0.01)

    def test_all_intensities_in_range(self, cal: pd.DataFrame) -> None:
        assert (cal["holiday_intensity_score"] >= 0.0).all()
        assert (cal["holiday_intensity_score"] <= 1.0).all()

    def test_bfcm_days_have_highest_avg_intensity(self, cal: pd.DataFrame) -> None:
        bfcm_mean = cal[cal["is_bfcm"]]["holiday_intensity_score"].mean()
        non_bfcm_mean = cal[~cal["is_bfcm"]]["holiday_intensity_score"].mean()
        assert bfcm_mean > non_bfcm_mean


# ===========================================================================
# TestFutureYears
# ===========================================================================

class TestFutureYears:
    """Calendar must work correctly for future years (2026–2027)."""

    def test_thanksgiving_2026_in_calendar(self, cal: pd.DataFrame) -> None:
        row = cal[cal["date"] == pd.Timestamp(THANKSGIVING_2026)]
        assert len(row) == 1
        assert row["is_holiday"].iloc[0] == True

    def test_black_friday_2027_in_calendar(self, cal: pd.DataFrame) -> None:
        row = cal[cal["date"] == pd.Timestamp(BLACK_FRIDAY_2027)]
        assert len(row) == 1
        assert row["is_bfcm"].iloc[0] == True

    def test_cyber_monday_2026_in_november(self, cal: pd.DataFrame) -> None:
        row = cal[cal["date"] == pd.Timestamp(CYBER_MONDAY_2026)]
        assert row["date"].iloc[0].month == 11

    def test_cyber_week_2027_correct_length(self, cal: pd.DataFrame) -> None:
        td = pd.Timestamp(THANKSGIVING_2027)
        cm = pd.Timestamp(CYBER_MONDAY_2027)
        mask = (cal["date"] >= td) & (cal["date"] <= cm)
        assert mask.sum() == 5

    def test_calendar_beyond_default_range(self) -> None:
        # build_holiday_calendar must accept years beyond 2027
        cal_2030 = build_holiday_calendar(2030, 2030)
        assert len(cal_2030) == 365  # 2030 is not a leap year
        assert cal_2030["date"].iloc[0] == pd.Timestamp("2030-01-01")


# ===========================================================================
# TestValidation
# ===========================================================================

class TestValidation:
    """Input validation raises appropriate errors."""

    def test_raises_on_start_after_end(self) -> None:
        with pytest.raises(ValueError, match="start_year"):
            build_holiday_calendar(2027, 2024)

    def test_raises_on_year_too_far(self) -> None:
        with pytest.raises(ValueError, match="2100"):
            build_holiday_calendar(2024, 2200)

    def test_raises_on_non_int_start(self) -> None:
        with pytest.raises(TypeError):
            build_holiday_calendar("2024", 2027)  # type: ignore[arg-type]

    def test_raises_on_non_int_end(self) -> None:
        with pytest.raises(TypeError):
            build_holiday_calendar(2024, 2027.5)  # type: ignore[arg-type]

    def test_single_year_valid(self) -> None:
        cal_single = build_holiday_calendar(2025, 2025)
        assert len(cal_single) == 365

    def test_leap_year_2024(self) -> None:
        cal_2024 = build_holiday_calendar(2024, 2024)
        assert len(cal_2024) == 366
