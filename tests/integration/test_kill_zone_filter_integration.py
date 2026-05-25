"""Integration test: BacktestEngine blocks outside-kill-zone entries when
enable_kill_zone_filter=True.

Uses synthetic price data only (no CSV file dependency) per AC #6.

Price structure crafted to produce exactly one bearish FVG+sweep trade during
the 09:30 ET kill zone on Day 3 (2025-01-06, Monday):

    Day 1 (2025-01-02 Thu): 28 ranging bars — establish H1 history
        (< 20 H1 bars → vol_ok defaults True).
    Day 2 (2025-01-03 Fri): 5-bar swing high at H1 11:00 ET (high=18025); bearish
        sweep at H1 14:00 ET (high=18028, close=18015); within last 6 H1 bars of
        h1_slice when Day 3 is processed.
    Day 3 (2025-01-06 Mon): bearish FVG at 09:30 ET:
        c1.low=18025 > c3.high=18014 → gap=11 pts
        entry=18019.5, SL=18074.5, TP=17953.5
        fill at 09:45 (high=18020 ≥ 18019.5)
        TP at 10:30 (low=17950 ≤ 17953.5)

ATR gate check (debug-verified): gap=11 ≥ 0.5 × ATR≈16 = 8 ✓
Dollar gate: 11 × $5 = $55 ≤ $60 ✓
H1 ratio gate: gap=11 ≥ 0.25 × h1_atr_fallback(10) = 2.5 ✓
"""

import os
import tempfile
import zoneinfo
from datetime import datetime, time as dtime, timedelta, timezone

import pandas as pd
import pytest

from src.research.backtest_engine import BacktestEngine
from src.research.strategy_core import StrategyConfig

NY_TZ = zoneinfo.ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

def _et(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=NY_TZ)


def _bar(dt_et: datetime, o: float, h: float, lo: float, c: float) -> dict:
    dt_utc = dt_et.astimezone(timezone.utc)
    return {
        "timestamp": dt_utc.strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "open": o, "high": h, "low": lo, "close": c, "volume": 100,
    }


def _ranging(dt_et: datetime, mid: float = 18000.0) -> dict:
    return _bar(dt_et, mid + 2, mid + 4, mid - 4, mid - 2)


def _make_synthetic_bars_csv() -> str:
    """Write synthetic 15m bars to a temp CSV and return its path.

    All bars are 15m resolution passed as-is to BacktestEngine (which treats
    them as 1m bars, resamples to H1 by grouping on the hour boundary).
    """
    rows: list[dict] = []

    # ── Day 1: 2025-01-02 (Thursday) — 28 ranging bars ─────────────────────
    t = _et(2025, 1, 2, 9, 0)
    while t.hour < 16:
        rows.append(_ranging(t, 18000.0))
        t += timedelta(minutes=15)

    # ── Day 2: 2025-01-03 (Friday) — swing high + bearish sweep ────────────
    # H1 09:00 (high=18010, below swing high)
    for mm in (0, 15, 30, 45):
        rows.append(_bar(_et(2025, 1, 3, 9, mm), 18005, 18010, 17995, 18000))

    # H1 10:00 (high=18015)
    for mm in (0, 15, 30, 45):
        rows.append(_bar(_et(2025, 1, 3, 10, mm), 18010, 18015, 18000, 18005))

    # H1 11:00 — SWING HIGH (H1 high=18025; surrounding H1 bars have lower highs)
    rows.append(_bar(_et(2025, 1, 3, 11,  0), 18010, 18025, 18000, 18005))
    rows.append(_bar(_et(2025, 1, 3, 11, 15), 18005, 18012, 17998, 18000))
    rows.append(_bar(_et(2025, 1, 3, 11, 30), 18000, 18010, 17995, 17998))
    rows.append(_bar(_et(2025, 1, 3, 11, 45), 17998, 18010, 17990, 17995))

    # H1 12:00 (high=18012)
    for mm in (0, 15, 30, 45):
        rows.append(_bar(_et(2025, 1, 3, 12, mm), 18000, 18012, 17992, 18000))

    # H1 13:00 (high=18008)
    for mm in (0, 15, 30, 45):
        rows.append(_bar(_et(2025, 1, 3, 13, mm), 18000, 18008, 17992, 18000))

    # H1 14:00 — BEARISH SWEEP (H1 high=18028 > swing 18025; H1 close=18015 < 18025)
    rows.append(_bar(_et(2025, 1, 3, 14,  0), 18015, 18028, 18010, 18012))
    rows.append(_bar(_et(2025, 1, 3, 14, 15), 18012, 18020, 18008, 18010))
    rows.append(_bar(_et(2025, 1, 3, 14, 30), 18010, 18018, 18006, 18008))
    rows.append(_bar(_et(2025, 1, 3, 14, 45), 18008, 18015, 18005, 18015))

    # H1 15:00 (high=18018)
    for mm in (0, 15, 30, 45):
        rows.append(_bar(_et(2025, 1, 3, 15, mm), 18012, 18018, 18005, 18010))

    # ── Day 3: 2025-01-06 (Monday) — FVG at 09:30, fill 09:45, TP 10:30 ───
    # c1 (09:00): low=18025 anchors the FVG gap
    rows.append(_bar(_et(2025, 1, 6, 9,  0), 18030, 18035, 18025, 18026))
    # c2 (09:15): bearish middle (close < open)
    rows.append(_bar(_et(2025, 1, 6, 9, 15), 18024, 18025, 18014, 18015))
    # c3 (09:30): high=18014 → gap=18025-18014=11; kill_zone=True; trade armed
    # gap=11 ≥ 0.5×ATR≈16=8 ✓; 11×$5=$55 ≤ $60 ✓
    rows.append(_bar(_et(2025, 1, 6, 9, 30), 18012, 18014, 18006, 18008))
    # fill bar (09:45): high=18020 ≥ entry_price=18019.5
    rows.append(_bar(_et(2025, 1, 6, 9, 45), 18015, 18020, 18012, 18014))
    # post-fill bars declining toward TP=17953.5
    rows.append(_bar(_et(2025, 1, 6, 10,  0), 18013, 18014, 17985, 17986))
    rows.append(_bar(_et(2025, 1, 6, 10, 15), 17986, 17987, 17960, 17962))
    # TP hit: low=17950 ≤ tp=17953.5
    rows.append(_bar(_et(2025, 1, 6, 10, 30), 17962, 17963, 17950, 17952))
    # remaining bars of Day 3
    t = _et(2025, 1, 6, 10, 45)
    while t.hour < 16:
        rows.append(_ranging(t, 17952.0))
        t += timedelta(minutes=15)

    df = pd.DataFrame(rows)
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


@pytest.fixture(scope="module")
def synthetic_csv():
    path = _make_synthetic_bars_csv()
    yield path
    os.unlink(path)


def _run(csv_path: str, config: StrategyConfig) -> list:
    return BacktestEngine(csv_path, config=config).run()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_kill_zone_filter_all_trades_in_window(synthetic_csv):
    """All trades with enable_kill_zone_filter=True are inside [09:30, 11:00) ET.

    Non-vacuous: asserts at least one trade is produced by the synthetic data.
    """
    config = StrategyConfig(bearish_only=True, enable_kill_zone_filter=True)
    trades = _run(synthetic_csv, config)

    assert len(trades) >= 1, (
        "Synthetic data must produce at least one trade with kill zone enabled"
    )
    for t in trades:
        entry_ny = t.timestamp_entry.astimezone(NY_TZ).time()
        assert dtime(9, 30) <= entry_ny < dtime(11, 0), (
            f"Trade outside kill zone: entry at {entry_ny} ({t.timestamp_entry})"
        )
    assert all(t.kill_zone_active for t in trades), (
        "All KZ-filtered trades must have kill_zone_active=True"
    )


def test_kill_zone_filter_reduces_trade_count(synthetic_csv):
    """Disabling the kill zone filter produces >= trades as enabling it.

    Non-vacuous: asserts filter_off produces at least one trade.
    """
    trades_on = _run(synthetic_csv, StrategyConfig(bearish_only=True, enable_kill_zone_filter=True))
    trades_off = _run(synthetic_csv, StrategyConfig(bearish_only=True, enable_kill_zone_filter=False))

    assert len(trades_off) >= 1, "Baseline (filter OFF) must produce at least one trade"
    assert len(trades_off) >= len(trades_on), (
        f"Expected filter_off ({len(trades_off)}) >= filter_on ({len(trades_on)})"
    )


def test_kill_zone_filter_disabled_unchanged_behavior(synthetic_csv):
    """enable_kill_zone_filter=False is identical to StrategyConfig default (False)."""
    trades_explicit = _run(synthetic_csv, StrategyConfig(bearish_only=True, enable_kill_zone_filter=False))
    trades_default = _run(synthetic_csv, StrategyConfig(bearish_only=True))

    assert len(trades_explicit) == len(trades_default), (
        "enable_kill_zone_filter=False must produce same result as default (False)"
    )
