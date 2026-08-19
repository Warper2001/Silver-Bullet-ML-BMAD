"""Regression: BacktestEngine.run() must not crash with m15_confirmation=True.

Found 2026-08-19 running the gap-ceiling in-sample backtest: `m15_last_bar_ts`
was initialized as a tz-NAIVE pd.Timestamp.min, then compared against
full_m15.index entries which are tz-aware (bars are converted to
America/New_York in _load_bars). Any config with m15_confirmation=True (S25's
live setting) crashed on the first M15 CHoCH scan with
"Cannot compare tz-naive and tz-aware timestamps" — meaning this code path
had apparently never been exercised end-to-end with YANK's actual live
config before now.
"""
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.research.backtest_engine import BacktestEngine
from src.research.strategy_core import StrategyConfig


def _write_bars_csv(n_days: int = 3) -> Path:
    """Enough continuous RTH-ish 1-min bars to reach an M15 CHoCH scan."""
    start = pd.Timestamp("2025-06-02 00:00:00", tz="UTC")  # Monday
    rows = []
    price = 20000.0
    for d in range(n_days):
        day_start = start + pd.Timedelta(days=d)
        for m in range(24 * 60):
            ts = day_start + pd.Timedelta(minutes=m)
            price += 0.25 * ((m % 7) - 3)  # small deterministic wiggle
            rows.append({
                "timestamp": ts, "open": price, "high": price + 1.0,
                "low": price - 1.0, "close": price + 0.1, "volume": 10,
            })
    df = pd.DataFrame(rows)
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return Path(tmp.name)


def test_m15_confirmation_true_does_not_crash_on_tz_aware_bars():
    csv_path = _write_bars_csv()
    try:
        config = StrategyConfig(m15_confirmation=True, bearish_only=False)
        trades = BacktestEngine(str(csv_path), config).run()
        assert isinstance(trades, list)  # ran to completion, no TypeError
    finally:
        csv_path.unlink(missing_ok=True)


def test_m15_confirmation_false_still_works(  # unaffected control
):
    csv_path = _write_bars_csv()
    try:
        config = StrategyConfig(m15_confirmation=False, bearish_only=False)
        trades = BacktestEngine(str(csv_path), config).run()
        assert isinstance(trades, list)
    finally:
        csv_path.unlink(missing_ok=True)
