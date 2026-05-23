"""Integration test: tuesday_exclusion flag in BacktestEngine (Story 2.4).

Verifies that BacktestEngine respects config.tuesday_exclusion:
- When True (default): no trades entered on Tuesdays
- When False: Tuesday entries are not blocked

Uses the same 2025 training CSV as the research scripts, limiting to
4 weeks (small enough to run quickly, enough to span Tuesdays).
"""

import zoneinfo

import pandas as pd
import pytest

from src.research.backtest_engine import BacktestEngine
from src.research.strategy_core import StrategyConfig

CSV_PATH = "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
NY_TZ = zoneinfo.ZoneInfo("America/New_York")


@pytest.fixture(scope="module")
def bars_15m_4weeks():
    """First 4 weeks of 2025 1-min bars resampled to 15m, written to temp CSV."""
    import os
    import tempfile

    bars = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    if bars["timestamp"].dt.tz is None:
        bars["timestamp"] = bars["timestamp"].dt.tz_localize("UTC")
    else:
        bars["timestamp"] = bars["timestamp"].dt.tz_convert("UTC")
    bars["timestamp"] = bars["timestamp"].dt.tz_convert("America/New_York")
    bars = bars.set_index("timestamp").sort_index()
    bars = bars.drop(columns=["notional"], errors="ignore")
    bars = bars.head(4 * 5 * 390)  # 4 weeks × 5 days × 390 min/day (approx)

    m15 = (
        bars.resample("15min")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open", "high", "low", "close"])
    )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df_out = m15.reset_index()
            df_out["timestamp"] = df_out["timestamp"].dt.tz_convert("UTC")
            df_out.to_csv(f, index=False)
            tmp_path = f.name
        yield tmp_path
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_tuesday_exclusion_true_no_tuesday_entries(bars_15m_4weeks):
    """tuesday_exclusion=True (default): no trades should have Tuesday entry timestamps."""
    config = StrategyConfig(bearish_only=True, tuesday_exclusion=True)
    trades = BacktestEngine(bars_15m_4weeks, config=config).run()
    tuesday_trades = [
        t for t in trades
        if t.timestamp_entry is not None
        and t.timestamp_entry.astimezone(NY_TZ).weekday() == 1
    ]
    assert tuesday_trades == [], (
        f"Expected no Tuesday trades with tuesday_exclusion=True, "
        f"got {len(tuesday_trades)}: {[t.timestamp_entry for t in tuesday_trades]}"
    )


def test_tuesday_exclusion_false_allows_more_trades(bars_15m_4weeks):
    """tuesday_exclusion=False produces >= trade count of tuesday_exclusion=True."""
    config_on = StrategyConfig(bearish_only=True, tuesday_exclusion=True)
    config_off = StrategyConfig(bearish_only=True, tuesday_exclusion=False)
    trades_on = BacktestEngine(bars_15m_4weeks, config=config_on).run()
    trades_off = BacktestEngine(bars_15m_4weeks, config=config_off).run()
    assert len(trades_off) >= len(trades_on), (
        f"tuesday_exclusion=False should produce >= trades vs True: "
        f"{len(trades_off)} vs {len(trades_on)}"
    )


def test_tuesday_exclusion_default_matches_true(bars_15m_4weeks):
    """StrategyConfig() default (tuesday_exclusion=True) produces same result as explicit True."""
    config_default = StrategyConfig(bearish_only=True)
    config_explicit = StrategyConfig(bearish_only=True, tuesday_exclusion=True)
    trades_default = BacktestEngine(bars_15m_4weeks, config=config_default).run()
    trades_explicit = BacktestEngine(bars_15m_4weeks, config=config_explicit).run()
    assert len(trades_default) == len(trades_explicit)
