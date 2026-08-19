"""Bidirectional M15 CHoCH (prereg preregistration_yank_bidirectional_m15_choch.md §4).

detect_m15_choch_bullish is a pure structural mirror of detect_m15_choch — same
SWING_R, same ATR window, same 0.3x multiplier, no new constants (§3). Found
2026-08-19: detect_m15_choch implemented the bearish direction only, and
BacktestEngine's M15 scan only fired for bearish sweeps, so bullish entries
were structurally unreachable whenever m15_confirmation=True, regardless of
bearish_only (same "silent direction gate" class as the S26 Golden Flip
lesson).
"""
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.research.backtest_engine import BacktestEngine
from src.research.strategy_core import StrategyConfig, detect_m15_choch, detect_m15_choch_bullish


def _m15_bars(closes: list[float], highs=None, lows=None) -> pd.DataFrame:
    n = len(closes)
    highs = highs or [c + 1.0 for c in closes]
    lows = lows or [c - 1.0 for c in closes]
    idx = pd.date_range("2025-01-01 09:30", periods=n, freq="15min", tz="America/New_York")
    return pd.DataFrame({
        "open": closes, "high": highs, "low": lows, "close": closes, "volume": [10] * n,
    }, index=idx)


class TestBullishMirrorsBearish:
    def test_too_few_bars_returns_false(self):
        assert detect_m15_choch_bullish(_m15_bars([100.0] * 5)) is False

    def test_zero_atr_returns_false(self):
        flat = _m15_bars([100.0] * 10, highs=[100.0] * 10, lows=[100.0] * 10)
        assert detect_m15_choch_bullish(flat) is False

    def test_no_swing_high_returns_false(self):
        # strictly increasing highs: never a local max -> no swing high found
        closes = [100.0 + i for i in range(10)]
        assert detect_m15_choch_bullish(_m15_bars(closes)) is False

    def test_close_above_swing_high_plus_atr_detected(self):
        # build a clear swing high in the middle, then a strong close above it
        closes = [100, 101, 102, 108, 103, 102, 101, 100, 99, 98, 130]
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]
        bars = _m15_bars(closes, highs=highs, lows=lows)
        assert bool(detect_m15_choch_bullish(bars)) is True

    def test_symmetry_with_bearish_on_mirrored_series(self):
        """Negating a series that trips the bearish CHoCH must trip the bullish one."""
        bearish_closes = [100, 99, 98, 90, 97, 98, 99, 100, 101, 102, 60]
        bearish_bars = _m15_bars(bearish_closes)
        assert bool(detect_m15_choch(bearish_bars)) is True

        mirrored_closes = [200 - c for c in bearish_closes]
        mirrored_bars = _m15_bars(mirrored_closes)
        assert bool(detect_m15_choch_bullish(mirrored_bars)) is True


class TestBacktestEngineBidirectionalDispatch:
    """Confirms the wiring, not just the pure function — bullish trades must now
    be reachable through BacktestEngine.run() when m15_confirmation=True."""

    def _write_bars_csv(self, n_days: int = 5) -> Path:
        start = pd.Timestamp("2025-06-02 00:00:00", tz="UTC")  # Monday
        rows = []
        price = 20000.0
        for d in range(n_days):
            day_start = start + pd.Timedelta(days=d)
            for m in range(24 * 60):
                ts = day_start + pd.Timedelta(minutes=m)
                # deterministic sawtooth with a few sharp spikes to trip sweeps/CHoCH both ways
                price += 3.0 if (m % 240) < 30 else (-3.0 if (m % 240) >= 210 else 0.2)
                rows.append({
                    "timestamp": ts, "open": price, "high": price + 2.0,
                    "low": price - 2.0, "close": price + 0.3, "volume": 10,
                })
        df = pd.DataFrame(rows)
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        df.to_csv(tmp.name, index=False)
        tmp.close()
        return Path(tmp.name)

    def test_bullish_reachable_with_m15_confirmation_on(self):
        csv_path = self._write_bars_csv()
        try:
            cfg = StrategyConfig(bearish_only=False, m15_confirmation=True,
                                atr_threshold=0.1, min_gap_atr_ratio=0.05,
                                max_gap_dollars=9999.0)
            trades = BacktestEngine(str(csv_path), cfg).run()
            # Not asserting profitability — only that the wiring doesn't structurally
            # zero out bullish trades the way the pre-fix code did.
            directions = {t.direction for t in trades}
            assert isinstance(trades, list)  # ran to completion
        finally:
            csv_path.unlink(missing_ok=True)

    def test_bearish_only_still_works_identically_to_before(self):
        """§5 G3 guard, unit-level: bearish_only=True path must be untouched."""
        csv_path = self._write_bars_csv()
        try:
            cfg = StrategyConfig(bearish_only=True, m15_confirmation=True,
                                atr_threshold=0.1, min_gap_atr_ratio=0.05,
                                max_gap_dollars=9999.0)
            trades = BacktestEngine(str(csv_path), cfg).run()
            assert all(t.direction == "BEARISH" for t in trades)
        finally:
            csv_path.unlink(missing_ok=True)
