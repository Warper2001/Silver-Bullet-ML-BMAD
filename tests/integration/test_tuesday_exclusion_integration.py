"""Integration test: tuesday_exclusion flag in BacktestEngine (Story 2.4).

Uses synthetic price data only (no CSV file dependency).

Day 1 (2025-01-02 Thu): 28 ranging bars — establish H1 history
Day 2 (2025-01-06 Mon): swing high at H1 11:00 ET (high=18025);
    bearish sweep at H1 14:00 ET (high=18028, close=18015)
Day 3 (2025-01-07 Tue): bearish FVG at 09:30 ET:
    c1.low=18025 > c3.high=18014 → gap=11 pts
    entry=18019.5, SL=18074.5, TP=17953.5
    fill at 09:45 (high=18020 ≥ 18019.5)
    TP at 10:30 (low=17950 ≤ 17953.5)

tuesday_exclusion=True  → bar 09:30 Tue (and all Tue bars) skipped → 0 trades
tuesday_exclusion=False → entry processed → 1 trade with Tuesday timestamp
"""

import os
import tempfile
import zoneinfo
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.research.backtest_engine import BacktestEngine
from src.research.strategy_core import StrategyConfig

NY_TZ = zoneinfo.ZoneInfo("America/New_York")


def _et(year, month, day, hour, minute):
    return datetime(year, month, day, hour, minute, tzinfo=NY_TZ)


def _bar(dt_et, o, h, lo, c):
    dt_utc = dt_et.astimezone(timezone.utc)
    return {
        "timestamp": dt_utc.strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "open": o,
        "high": h,
        "low": lo,
        "close": c,
        "volume": 100,
    }


def _ranging(dt_et, mid=18000.0):
    return _bar(dt_et, mid + 2, mid + 4, mid - 4, mid - 2)


def _make_tuesday_synthetic_csv():
    rows = []

    # Day 1 (Thu 2025-01-02): ranging bars to establish H1 history
    t = _et(2025, 1, 2, 9, 0)
    while t.hour < 16:
        rows.append(_ranging(t, 18000.0))
        t += timedelta(minutes=15)

    # Day 2 (Mon 2025-01-06): swing high + bearish sweep
    for mm in (0, 15, 30, 45):
        rows.append(_bar(_et(2025, 1, 6, 9, mm), 18005, 18010, 17995, 18000))
    for mm in (0, 15, 30, 45):
        rows.append(_bar(_et(2025, 1, 6, 10, mm), 18010, 18015, 18000, 18005))
    rows.append(_bar(_et(2025, 1, 6, 11, 0), 18010, 18025, 18000, 18005))  # swing high
    rows.append(_bar(_et(2025, 1, 6, 11, 15), 18005, 18012, 17998, 18000))
    rows.append(_bar(_et(2025, 1, 6, 11, 30), 18000, 18010, 17995, 17998))
    rows.append(_bar(_et(2025, 1, 6, 11, 45), 17998, 18010, 17990, 17995))
    for mm in (0, 15, 30, 45):
        rows.append(_bar(_et(2025, 1, 6, 12, mm), 18000, 18012, 17992, 18000))
    for mm in (0, 15, 30, 45):
        rows.append(_bar(_et(2025, 1, 6, 13, mm), 18000, 18008, 17992, 18000))
    rows.append(_bar(_et(2025, 1, 6, 14, 0), 18015, 18028, 18010, 18012))  # sweep
    rows.append(_bar(_et(2025, 1, 6, 14, 15), 18012, 18020, 18008, 18010))
    rows.append(_bar(_et(2025, 1, 6, 14, 30), 18010, 18018, 18006, 18008))
    rows.append(_bar(_et(2025, 1, 6, 14, 45), 18008, 18015, 18005, 18015))
    for mm in (0, 15, 30, 45):
        rows.append(_bar(_et(2025, 1, 6, 15, mm), 18012, 18018, 18005, 18010))

    # Day 3 (Tue 2025-01-07): FVG + fill + TP
    rows.append(_bar(_et(2025, 1, 7, 9, 0), 18030, 18035, 18025, 18026))   # c1
    rows.append(_bar(_et(2025, 1, 7, 9, 15), 18024, 18025, 18014, 18015))  # c2 bearish
    rows.append(_bar(_et(2025, 1, 7, 9, 30), 18012, 18014, 18006, 18008))  # c3 gap=11
    rows.append(_bar(_et(2025, 1, 7, 9, 45), 18015, 18020, 18012, 18014))  # fill
    rows.append(_bar(_et(2025, 1, 7, 10, 0), 18013, 18014, 17985, 17986))
    rows.append(_bar(_et(2025, 1, 7, 10, 15), 17986, 17987, 17960, 17962))
    rows.append(_bar(_et(2025, 1, 7, 10, 30), 17962, 17963, 17950, 17952))  # TP
    t = _et(2025, 1, 7, 10, 45)
    while t.hour < 16:
        rows.append(_ranging(t, 17952.0))
        t += timedelta(minutes=15)

    df = pd.DataFrame(rows)
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


@pytest.fixture(scope="module")
def tuesday_csv():
    path = _make_tuesday_synthetic_csv()
    yield path
    os.unlink(path)


def _run(csv_path, config):
    return BacktestEngine(csv_path, config=config).run()


def test_tuesday_exclusion_blocks_tuesday_entries(tuesday_csv):
    """tuesday_exclusion=True must block the Tuesday FVG entry; False must allow it."""
    trades_on = _run(tuesday_csv, StrategyConfig(bearish_only=True, tuesday_exclusion=True))
    trades_off = _run(tuesday_csv, StrategyConfig(bearish_only=True, tuesday_exclusion=False))
    assert len(trades_off) >= 1, (
        "Synthetic data must produce at least one trade with tuesday_exclusion=False"
    )
    tuesday_entries = [
        t for t in trades_off
        if t.timestamp_entry.astimezone(NY_TZ).weekday() == 1
    ]
    assert len(tuesday_entries) >= 1, (
        "trades with tuesday_exclusion=False must include a Tuesday entry"
    )
    assert len(trades_on) == 0, (
        f"tuesday_exclusion=True must block the Tuesday entry; got {len(trades_on)} trades"
    )


def test_tuesday_exclusion_false_allows_more_trades(tuesday_csv):
    """tuesday_exclusion=False produces >= trade count of tuesday_exclusion=True."""
    trades_on = _run(tuesday_csv, StrategyConfig(bearish_only=True, tuesday_exclusion=True))
    trades_off = _run(tuesday_csv, StrategyConfig(bearish_only=True, tuesday_exclusion=False))
    assert len(trades_off) >= 1, "Baseline (filter OFF) must produce at least one trade"
    assert len(trades_off) >= len(trades_on), (
        f"Expected trades_off ({len(trades_off)}) >= trades_on ({len(trades_on)})"
    )


def test_tuesday_exclusion_default_matches_true(tuesday_csv):
    """StrategyConfig() default (tuesday_exclusion=True) produces same result as explicit True."""
    trades_default = _run(tuesday_csv, StrategyConfig(bearish_only=True))
    trades_explicit = _run(tuesday_csv, StrategyConfig(bearish_only=True, tuesday_exclusion=True))
    assert len(trades_default) == len(trades_explicit), (
        "Default tuesday_exclusion must produce same result as explicit True"
    )
