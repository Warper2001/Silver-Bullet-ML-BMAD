"""Unit tests for kill_zone_filter with DST boundary dates (Story 1.3, AC #13).

Tests spring-forward 2026-03-08 and fall-back 2026-11-01 boundaries.
Verifies that 09:30–11:00 ET is correct on both sides of each DST transition.

AR21: kill zone is 09:30 (inclusive) – 11:00 (exclusive) America/New_York.
NFR19: must use tz-aware comparison; no fixed UTC offset.
"""

from __future__ import annotations

import zoneinfo

import pandas as pd
import pytest

from src.research.strategy_core import StrategyConfig, kill_zone_filter

NY_TZ = zoneinfo.ZoneInfo("America/New_York")
_CONFIG = StrategyConfig()  # kill_zone_start_et=09:30, kill_zone_end_et=11:00


def _ts(year: int, month: int, day: int, hour: int, minute: int, second: int = 0) -> pd.Timestamp:
    """Create a tz-aware pd.Timestamp in America/New_York."""
    return pd.Timestamp(
        year=year, month=month, day=day,
        hour=hour, minute=minute, second=second,
        tz=NY_TZ,
    )


# ---------------------------------------------------------------------------
# Parameterised boundary cases
# ---------------------------------------------------------------------------

# (description, year, month, day, hour, minute, second, expected_result)
_BOUNDARY_CASES = [
    # --- Spring-forward boundary: 2026-03-08 ---
    # Day before spring-forward (EST, UTC-5)
    ("pre-spring: 09:29:59 → False", 2026, 3, 7, 9, 29, 59, False),
    ("pre-spring: 09:30:00 → True",  2026, 3, 7, 9, 30,  0, True),
    ("pre-spring: 10:59:59 → True",  2026, 3, 7, 10, 59, 59, True),
    ("pre-spring: 11:00:00 → False", 2026, 3, 7, 11,  0,  0, False),
    # Day of spring-forward (EDT, UTC-4 by 09:30 AM)
    ("on-spring: 09:29:59 → False",  2026, 3, 8, 9, 29, 59, False),
    ("on-spring: 09:30:00 → True",   2026, 3, 8, 9, 30,  0, True),
    ("on-spring: 10:59:59 → True",   2026, 3, 8, 10, 59, 59, True),
    ("on-spring: 11:00:00 → False",  2026, 3, 8, 11,  0,  0, False),
    # Day after spring-forward (EDT, UTC-4)
    ("post-spring: 09:29:59 → False", 2026, 3, 9, 9, 29, 59, False),
    ("post-spring: 09:30:00 → True",  2026, 3, 9, 9, 30,  0, True),
    ("post-spring: 10:59:59 → True",  2026, 3, 9, 10, 59, 59, True),
    ("post-spring: 11:00:00 → False", 2026, 3, 9, 11,  0,  0, False),

    # --- Fall-back boundary: 2026-11-01 ---
    # Day before fall-back (EDT, UTC-4)
    ("pre-fall: 09:29:59 → False", 2026, 10, 31, 9, 29, 59, False),
    ("pre-fall: 09:30:00 → True",  2026, 10, 31, 9, 30,  0, True),
    ("pre-fall: 10:59:59 → True",  2026, 10, 31, 10, 59, 59, True),
    ("pre-fall: 11:00:00 → False", 2026, 10, 31, 11,  0,  0, False),
    # Day of fall-back (EST, UTC-5 by 09:30 AM — clocks fell back at 2 AM)
    ("on-fall: 09:29:59 → False",  2026, 11,  1, 9, 29, 59, False),
    ("on-fall: 09:30:00 → True",   2026, 11,  1, 9, 30,  0, True),
    ("on-fall: 10:59:59 → True",   2026, 11,  1, 10, 59, 59, True),
    ("on-fall: 11:00:00 → False",  2026, 11,  1, 11,  0,  0, False),
    # Day after fall-back (EST, UTC-5)
    ("post-fall: 09:29:59 → False", 2026, 11, 2, 9, 29, 59, False),
    ("post-fall: 09:30:00 → True",  2026, 11, 2, 9, 30,  0, True),
    ("post-fall: 10:59:59 → True",  2026, 11, 2, 10, 59, 59, True),
    ("post-fall: 11:00:00 → False", 2026, 11, 2, 11,  0,  0, False),
]


@pytest.mark.parametrize(
    "desc,year,month,day,hour,minute,second,expected",
    _BOUNDARY_CASES,
    ids=[c[0] for c in _BOUNDARY_CASES],
)
def test_kill_zone_dst_boundaries(desc, year, month, day, hour, minute, second, expected):
    ts = _ts(year, month, day, hour, minute, second)
    result = kill_zone_filter(ts, _CONFIG)
    assert result is expected, f"{desc}: got {result}, expected {expected} (ts={ts})"


# ---------------------------------------------------------------------------
# Custom kill zone config
# ---------------------------------------------------------------------------


class TestKillZoneCustomConfig:
    def test_custom_start_end(self):
        from datetime import time
        config = StrategyConfig(
            kill_zone_start_et=time(10, 0),
            kill_zone_end_et=time(12, 0),
        )
        assert kill_zone_filter(_ts(2026, 6, 1, 9, 59, 59), config) is False
        assert kill_zone_filter(_ts(2026, 6, 1, 10, 0, 0), config) is True
        assert kill_zone_filter(_ts(2026, 6, 1, 11, 59, 59), config) is True
        assert kill_zone_filter(_ts(2026, 6, 1, 12, 0, 0), config) is False

    def test_utc_input_converted_correctly(self):
        """A UTC-aware timestamp should resolve to NY time before comparison."""
        import datetime
        utc_ts = pd.Timestamp("2026-06-01 14:30:00", tz="UTC")  # 10:30 AM EDT
        result = kill_zone_filter(utc_ts, _CONFIG)
        assert result is True  # 10:30 AM EDT is within 09:30–11:00

    def test_midnight_not_in_kill_zone(self):
        assert kill_zone_filter(_ts(2026, 6, 1, 0, 0, 0), _CONFIG) is False

    def test_midday_not_in_kill_zone(self):
        assert kill_zone_filter(_ts(2026, 6, 1, 15, 0, 0), _CONFIG) is False
