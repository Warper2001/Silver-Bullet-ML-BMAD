"""Integration test: BacktestEngine blocks outside-kill-zone entries when
enable_kill_zone_filter=True.

Verifies that:
1. All trades produced with enable_kill_zone_filter=True have kill_zone_active=True.
2. All entry timestamps fall in [09:30, 11:00) ET (DST-aware).
3. Disabling the filter produces more trades (or equal if data is sparse).
"""

import os
import tempfile
import zoneinfo
from datetime import time as dtime

import pandas as pd
import pytest

from src.research.backtest_engine import BacktestEngine
from src.research.strategy_core import StrategyConfig

NY_TZ = zoneinfo.ZoneInfo("America/New_York")
CSV_PATH = "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"


def _resample_to_15m(bars: pd.DataFrame) -> pd.DataFrame:
    if bars["timestamp"].dt.tz is None:
        bars["timestamp"] = bars["timestamp"].dt.tz_localize("UTC")
    bars["timestamp"] = bars["timestamp"].dt.tz_convert("America/New_York")
    bars = bars.set_index("timestamp").sort_index()
    bars = bars.drop(columns=["notional"], errors="ignore")
    return (
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


def _run_engine(bars_15m: pd.DataFrame, config: StrategyConfig) -> list:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df_out = bars_15m.reset_index()
            df_out["timestamp"] = df_out["timestamp"].dt.tz_convert("UTC")
            df_out.to_csv(f, index=False)
            tmp_path = f.name
        return BacktestEngine(tmp_path, config=config).run()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@pytest.fixture(scope="module")
def bars_15m_small():
    if not os.path.exists(CSV_PATH):
        pytest.skip("Training CSV not available")
    raw = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    raw = raw.head(4 * 390)  # ~4 weeks of 1-min bars — enough for a few trades
    return _resample_to_15m(raw)


def test_kill_zone_filter_all_trades_in_window(bars_15m_small):
    """All trades with enable_kill_zone_filter=True must be inside [09:30, 11:00) ET."""
    config = StrategyConfig(bearish_only=True, enable_kill_zone_filter=True)
    trades = _run_engine(bars_15m_small, config)

    for t in trades:
        entry_ny = t.timestamp_entry.astimezone(NY_TZ).time()
        assert dtime(9, 30) <= entry_ny < dtime(11, 0), (
            f"Trade outside kill zone: entry at {entry_ny} ({t.timestamp_entry})"
        )
    assert all(t.kill_zone_active for t in trades), (
        "All KZ-filtered trades must have kill_zone_active=True"
    )


def test_kill_zone_filter_reduces_trade_count(bars_15m_small):
    """Disabling kill zone filter produces >= trades as enabling it."""
    config_on = StrategyConfig(bearish_only=True, enable_kill_zone_filter=True)
    config_off = StrategyConfig(bearish_only=True, enable_kill_zone_filter=False)
    trades_on = _run_engine(bars_15m_small, config_on)
    trades_off = _run_engine(bars_15m_small, config_off)
    assert len(trades_off) >= len(trades_on), (
        f"Expected filter_off ({len(trades_off)}) >= filter_on ({len(trades_on)})"
    )


def test_kill_zone_filter_disabled_unchanged_behavior(bars_15m_small):
    """With enable_kill_zone_filter=False, behavior is identical to default StrategyConfig."""
    config_explicit_off = StrategyConfig(bearish_only=True, enable_kill_zone_filter=False)
    config_default = StrategyConfig(bearish_only=True)
    trades_explicit = _run_engine(bars_15m_small, config_explicit_off)
    trades_default = _run_engine(bars_15m_small, config_default)
    assert len(trades_explicit) == len(trades_default), (
        "enable_kill_zone_filter=False must produce same result as default (False)"
    )
