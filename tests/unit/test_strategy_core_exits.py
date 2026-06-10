"""Unit tests for make_entry_decision and check_exit (Story 1.3, AC #11).

Covers:
- make_entry_decision: passing filters, each failing filter, misaligned direction.
- check_exit: TP / SL / TIME_STOP / no-exit, for both BEARISH and BULLISH trades.

Benchmark harness (AC #14) is in the ``if __name__ == "__main__"`` block below.
"""

from __future__ import annotations

import statistics
import time as _time_mod

import pandas as pd
import pytest
import zoneinfo

from src.research.strategy_core import (
    Direction,
    EntryDecision,
    ExitDecision,
    ExitReason,
    FVGSignal,
    StrategyConfig,
    SweepSignal,
    check_exit,
    make_entry_decision,
)

NY_TZ = zoneinfo.ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIG = StrategyConfig()  # defaults: sl_mult=5, tp_mult=6, max_hold_bars=60


def _bearish_sweep() -> SweepSignal:
    return SweepSignal(direction=Direction.BEARISH, bars_ago=1, sweep_price=200.0)


def _bullish_sweep() -> SweepSignal:
    return SweepSignal(direction=Direction.BULLISH, bars_ago=1, sweep_price=100.0)


def _bearish_fvg(entry: float = 100.0, gap: float = 4.0) -> FVGSignal:
    return FVGSignal(
        direction=Direction.BEARISH,
        gap_size=gap,
        entry_price=entry,
        high=entry + gap / 2,
        low=entry - gap / 2,
    )


def _bullish_fvg(entry: float = 200.0, gap: float = 4.0) -> FVGSignal:
    return FVGSignal(
        direction=Direction.BULLISH,
        gap_size=gap,
        entry_price=entry,
        high=entry + gap / 2,
        low=entry - gap / 2,
    )


def _bar(high: float, low: float, close: float | None = None) -> pd.Series:
    if close is None:
        close = (high + low) / 2
    return pd.Series({"high": high, "low": low, "close": close})


# ---------------------------------------------------------------------------
# make_entry_decision
# ---------------------------------------------------------------------------


class TestMakeEntryDecision:
    def test_bearish_aligned_all_pass(self):
        decision = make_entry_decision(
            _bearish_sweep(), _bearish_fvg(entry=100.0, gap=4.0), _CONFIG,
            kill_zone=True, vol_regime=True,
        )
        assert decision is not None
        assert decision.direction == Direction.BEARISH
        assert decision.entry_price == pytest.approx(100.0)
        assert decision.sl_price == pytest.approx(100.0 + 5.0 * 4.0)  # 120.0
        assert decision.tp_price == pytest.approx(100.0 - 6.0 * 4.0)  # 76.0
        assert decision.contracts == _CONFIG.contracts_per_trade

    def test_bullish_aligned_all_pass(self):
        decision = make_entry_decision(
            _bullish_sweep(), _bullish_fvg(entry=200.0, gap=4.0), _CONFIG,
            kill_zone=True, vol_regime=True,
        )
        assert decision is not None
        assert decision.direction == Direction.BULLISH
        assert decision.entry_price == pytest.approx(200.0)
        assert decision.sl_price == pytest.approx(200.0 - 5.0 * 4.0)  # 180.0
        assert decision.tp_price == pytest.approx(200.0 + 6.0 * 4.0)  # 224.0

    def test_misaligned_direction_returns_none(self):
        """Bearish sweep + bullish FVG → None."""
        decision = make_entry_decision(
            _bearish_sweep(), _bullish_fvg(), _CONFIG,
            kill_zone=True,
        )
        assert decision is None

    def test_misaligned_direction_opposite_returns_none(self):
        """Bullish sweep + bearish FVG → None."""
        decision = make_entry_decision(
            _bullish_sweep(), _bearish_fvg(), _CONFIG,
            kill_zone=True,
        )
        assert decision is None

    def test_single_filter_false_returns_none(self):
        decision = make_entry_decision(
            _bearish_sweep(), _bearish_fvg(), _CONFIG,
            kill_zone=False,
        )
        assert decision is None

    def test_one_of_multiple_filters_false_returns_none(self):
        decision = make_entry_decision(
            _bearish_sweep(), _bearish_fvg(), _CONFIG,
            kill_zone=True,
            vol_regime=False,
            m15_confirm=True,
        )
        assert decision is None

    def test_all_filters_false_returns_none(self):
        decision = make_entry_decision(
            _bearish_sweep(), _bearish_fvg(), _CONFIG,
            kill_zone=False, vol_regime=False,
        )
        assert decision is None

    def test_no_filter_kwargs_passes(self):
        """No extra filter kwargs → no filter check, should return decision."""
        decision = make_entry_decision(
            _bearish_sweep(), _bearish_fvg(), _CONFIG,
        )
        assert decision is not None

    def test_contracts_from_config(self):
        config = StrategyConfig(contracts_per_trade=3)
        decision = make_entry_decision(
            _bearish_sweep(), _bearish_fvg(), config,
        )
        assert decision.contracts == 3

    def test_sl_tp_multipliers_from_config(self):
        config = StrategyConfig(sl_multiplier=2.0, tp_multiplier=3.0)
        decision = make_entry_decision(
            _bearish_sweep(), _bearish_fvg(entry=100.0, gap=10.0), config,
        )
        assert decision.sl_price == pytest.approx(100.0 + 2.0 * 10.0)
        assert decision.tp_price == pytest.approx(100.0 - 3.0 * 10.0)


# ---------------------------------------------------------------------------
# check_exit
# ---------------------------------------------------------------------------


class TestCheckExitBearish:
    """Short trade: SL is above entry, TP is below entry."""

    def _trade(self, entry: float = 100.0, gap: float = 4.0) -> EntryDecision:
        return EntryDecision(
            direction=Direction.BEARISH,
            entry_price=entry,
            sl_price=entry + _CONFIG.sl_multiplier * gap,  # 120.0
            tp_price=entry - _CONFIG.tp_multiplier * gap,  # 76.0
            contracts=5,
        )

    def test_sl_hit_bar_high_above_sl(self):
        trade = self._trade()
        result = check_exit(_bar(high=121.0, low=90.0), trade, bars_held=5, config=_CONFIG)
        assert result is not None
        assert result.reason == ExitReason.SL
        assert result.exit_price == pytest.approx(trade.sl_price)

    def test_tp_hit_bar_low_below_tp(self):
        trade = self._trade()
        result = check_exit(_bar(high=105.0, low=75.0), trade, bars_held=5, config=_CONFIG)
        assert result is not None
        assert result.reason == ExitReason.TP
        assert result.exit_price == pytest.approx(trade.tp_price)

    def test_sl_takes_precedence_over_tp_when_both_hit(self):
        """Bar that simultaneously breaches both SL and TP → SL wins (first check)."""
        trade = self._trade()
        result = check_exit(_bar(high=125.0, low=70.0), trade, bars_held=5, config=_CONFIG)
        assert result.reason == ExitReason.SL

    def test_time_stop_when_neither_level_hit(self):
        trade = self._trade()
        result = check_exit(
            _bar(high=105.0, low=85.0, close=95.0),
            trade,
            bars_held=_CONFIG.max_hold_bars,
            config=_CONFIG,
        )
        assert result is not None
        assert result.reason == ExitReason.TIME_STOP
        assert result.exit_price == pytest.approx(95.0)

    def test_no_exit_inside_levels_under_time_stop(self):
        trade = self._trade()
        result = check_exit(
            _bar(high=105.0, low=85.0),
            trade,
            bars_held=_CONFIG.max_hold_bars - 1,
            config=_CONFIG,
        )
        assert result is None

    def test_sl_at_exact_boundary(self):
        trade = self._trade()
        # bar.high == sl_price exactly → SL
        result = check_exit(_bar(high=trade.sl_price, low=90.0), trade, bars_held=1, config=_CONFIG)
        assert result.reason == ExitReason.SL

    def test_tp_at_exact_boundary(self):
        trade = self._trade()
        # bar.low == tp_price exactly → TP
        result = check_exit(_bar(high=105.0, low=trade.tp_price), trade, bars_held=1, config=_CONFIG)
        assert result.reason == ExitReason.TP

    def test_time_stop_at_exact_max_hold_bars(self):
        trade = self._trade()
        result = check_exit(
            _bar(high=105.0, low=95.0, close=100.0),
            trade,
            bars_held=_CONFIG.max_hold_bars,
            config=_CONFIG,
        )
        assert result.reason == ExitReason.TIME_STOP

    def test_sl_not_hit_one_tick_inside(self):
        trade = self._trade()
        result = check_exit(
            _bar(high=trade.sl_price - 0.01, low=90.0),
            trade,
            bars_held=1,
            config=_CONFIG,
        )
        assert result is None or result.reason != ExitReason.SL


class TestS27BreakevenAndTrailingStops:
    """S27 Hold-Period Exit Management: breakeven stop and trailing stop tests."""

    # ── Bearish (SHORT) helpers ──────────────────────────────────────────────

    def _bearish_trade(self, entry: float = 100.0, gap: float = 4.0) -> EntryDecision:
        return EntryDecision(
            direction=Direction.BEARISH,
            entry_price=entry,
            sl_price=entry + _CONFIG.sl_multiplier * gap,  # 120.0
            tp_price=entry - _CONFIG.tp_multiplier * gap,  # 76.0
            contracts=5,
        )

    def _config_be(self, trigger_r: float = 2.0) -> StrategyConfig:
        return StrategyConfig(enable_breakeven_stop=True, breakeven_trigger_r=trigger_r)

    def _config_trail(self, trail_mult: float = 1.5) -> StrategyConfig:
        return StrategyConfig(enable_trailing_stop=True, trailing_stop_mult=trail_mult)

    def _config_both(self) -> StrategyConfig:
        return StrategyConfig(enable_breakeven_stop=True, breakeven_trigger_r=2.0,
                              enable_trailing_stop=True, trailing_stop_mult=1.5)

    # ── Test: BREAKEVEN_SL fires when MFE ≥ trigger and price returns to entry ──

    def test_breakeven_sl_fires_bearish(self):
        """Bearish: after MFE ≥ 2×gap, bar that reaches entry price exits as BREAKEVEN_SL."""
        trade = self._bearish_trade(entry=100.0, gap=4.0)
        cfg = self._config_be(trigger_r=2.0)
        # gap=4 → trigger = 2×4 = 8 pts; MFE=8 means price fell to 92
        mfe_pts = 8.0
        # Price bounces back to entry (bar.high = 100.0 = entry)
        result = check_exit(_bar(high=100.0, low=90.0), trade, bars_held=10, config=cfg, mfe_pts=mfe_pts)
        assert result is not None
        assert result.reason == ExitReason.BREAKEVEN_SL
        assert result.exit_price == pytest.approx(100.0)  # entry price = breakeven level

    def test_breakeven_sl_pnl_approx_negative_commission(self):
        """BREAKEVEN_SL exit price equals entry → P&L = −commission (no directional loss)."""
        trade = self._bearish_trade(entry=100.0, gap=4.0)
        cfg = self._config_be(trigger_r=2.0)
        mfe_pts = 8.0
        result = check_exit(_bar(high=100.0, low=90.0), trade, bars_held=10, config=cfg, mfe_pts=mfe_pts)
        assert result is not None
        # P&L at breakeven = (entry_price − exit_price) × point_value × contracts − commission
        # = (100 − 100) × 2 × 5 − 4 = −4 (only commission)
        pnl = (trade.entry_price - result.exit_price) * 2.0 * 5 - _CONFIG.commission_per_roundtrip
        assert pnl == pytest.approx(-_CONFIG.commission_per_roundtrip)

    def test_breakeven_sl_not_fired_when_mfe_below_trigger(self):
        """Breakeven trigger not yet reached → no BREAKEVEN_SL even if price returns to entry."""
        trade = self._bearish_trade(entry=100.0, gap=4.0)
        cfg = self._config_be(trigger_r=2.0)
        # MFE only 1×gap = 4 pts, well below 2×gap trigger
        mfe_pts = 4.0
        result = check_exit(_bar(high=100.0, low=90.0), trade, bars_held=10, config=cfg, mfe_pts=mfe_pts)
        # Should not exit (price at entry but trigger not armed; entry < sl_price so no hard SL hit)
        assert result is None or result.reason != ExitReason.BREAKEVEN_SL

    def test_breakeven_sl_fires_bullish(self):
        """Bullish: after MFE ≥ 2×gap, bar that falls back to entry exits as BREAKEVEN_SL."""
        trade = EntryDecision(
            direction=Direction.BULLISH,
            entry_price=200.0,
            sl_price=200.0 - _CONFIG.sl_multiplier * 4.0,  # 180.0
            tp_price=200.0 + _CONFIG.tp_multiplier * 4.0,  # 224.0
            contracts=5,
        )
        cfg = self._config_be(trigger_r=2.0)
        mfe_pts = 8.0  # 2×gap → trigger armed
        # Price falls back to entry (bar.low = 200.0)
        result = check_exit(_bar(high=205.0, low=200.0), trade, bars_held=10, config=cfg, mfe_pts=mfe_pts)
        assert result is not None
        assert result.reason == ExitReason.BREAKEVEN_SL
        assert result.exit_price == pytest.approx(200.0)

    # ── Test: TRAILING_SL fires when price retraces by trailing_mult × gap from MFE peak ──

    def test_trailing_sl_fires_bearish(self):
        """Bearish: after MFE ≥ 1.5×gap, price retraces by 1.5×gap from MFE peak → TRAILING_SL."""
        trade = self._bearish_trade(entry=100.0, gap=4.0)
        cfg = self._config_trail(trail_mult=1.5)
        # gap=4, trail_mult=1.5 → trail distance = 6 pts
        # MFE=10 pts → MFE peak price = 100 - 10 = 90; trail SL = 90 + 6 = 96
        mfe_pts = 10.0
        # bar.high = 96 → hits trail SL
        result = check_exit(_bar(high=96.0, low=88.0), trade, bars_held=10, config=cfg, mfe_pts=mfe_pts)
        assert result is not None
        assert result.reason == ExitReason.TRAILING_SL
        assert result.exit_price == pytest.approx(96.0)  # trail SL level = 90 + 6

    def test_trailing_sl_not_fired_when_mfe_below_activation(self):
        """Trailing stop not activated when MFE < trailing_stop_mult × gap_size."""
        trade = self._bearish_trade(entry=100.0, gap=4.0)
        cfg = self._config_trail(trail_mult=1.5)
        # MFE = only 4 pts (1×gap), activation requires 1.5×gap = 6 pts
        mfe_pts = 4.0
        result = check_exit(_bar(high=99.0, low=95.0), trade, bars_held=10, config=cfg, mfe_pts=mfe_pts)
        assert result is None  # no exit — still inside barriers

    def test_trailing_sl_fires_bullish(self):
        """Bullish: after MFE ≥ 1.5×gap, price falls by 1.5×gap from MFE peak → TRAILING_SL."""
        trade = EntryDecision(
            direction=Direction.BULLISH,
            entry_price=200.0,
            sl_price=200.0 - _CONFIG.sl_multiplier * 4.0,  # 180.0
            tp_price=200.0 + _CONFIG.tp_multiplier * 4.0,  # 224.0
            contracts=5,
        )
        cfg = self._config_trail(trail_mult=1.5)
        # gap=4, trail=1.5 → trail dist=6; MFE=10 → peak=210; trail SL=210−6=204
        mfe_pts = 10.0
        result = check_exit(_bar(high=208.0, low=204.0), trade, bars_held=10, config=cfg, mfe_pts=mfe_pts)
        assert result is not None
        assert result.reason == ExitReason.TRAILING_SL
        assert result.exit_price == pytest.approx(204.0)

    # ── Test: flags off → behaviour identical to pre-S27 ──

    def test_flags_off_no_change_bearish(self):
        """With all S27 flags False, check_exit output is identical to the pre-extension baseline."""
        trade = self._bearish_trade()
        mfe_pts = 10.0  # would arm both if flags were on
        # Test each pre-existing exit type
        sl_result = check_exit(_bar(high=121.0, low=90.0), trade, bars_held=5, config=_CONFIG, mfe_pts=mfe_pts)
        assert sl_result is not None and sl_result.reason == ExitReason.SL

        tp_result = check_exit(_bar(high=105.0, low=75.0), trade, bars_held=5, config=_CONFIG, mfe_pts=mfe_pts)
        assert tp_result is not None and tp_result.reason == ExitReason.TP

        time_result = check_exit(_bar(high=105.0, low=85.0, close=95.0), trade,
                                 bars_held=_CONFIG.max_hold_bars, config=_CONFIG, mfe_pts=mfe_pts)
        assert time_result is not None and time_result.reason == ExitReason.TIME_STOP

        no_exit = check_exit(_bar(high=105.0, low=85.0), trade,
                             bars_held=_CONFIG.max_hold_bars - 1, config=_CONFIG, mfe_pts=mfe_pts)
        assert no_exit is None

    def test_flags_off_no_change_default_mfe(self):
        """With default mfe_pts=0.0 and flags off, original exit behaviour is preserved."""
        trade = self._bearish_trade()
        # No mfe_pts argument at all — should behave like old API
        result_sl = check_exit(_bar(high=121.0, low=90.0), trade, bars_held=5, config=_CONFIG)
        assert result_sl is not None and result_sl.reason == ExitReason.SL

        result_time = check_exit(_bar(high=105.0, low=85.0, close=95.0), trade,
                                 bars_held=_CONFIG.max_hold_bars, config=_CONFIG)
        assert result_time is not None and result_time.reason == ExitReason.TIME_STOP

    # ── Test: both flags on — breakeven then trailing take effect ──

    def test_both_flags_trailing_tighter_than_breakeven(self):
        """When both flags on, trailing (lower price for SHORT) is tighter than breakeven → TRAILING_SL fires."""
        trade = self._bearish_trade(entry=100.0, gap=4.0)
        cfg = self._config_both()  # be_trigger=2.0×gap=8, trail_mult=1.5×gap=6
        # MFE = 8 pts → MFE peak price = 92; trail SL = 92 + 6 = 98; breakeven SL = 100 (entry)
        # For SHORT: tighter = lower price → trail (98) < breakeven (100) → TRAILING_SL is active
        mfe_pts = 8.0
        # bar.high = 98 → hits trail SL (98), which is below breakeven (100)
        result = check_exit(_bar(high=98.0, low=88.0), trade, bars_held=10, config=cfg, mfe_pts=mfe_pts)
        assert result is not None
        assert result.reason == ExitReason.TRAILING_SL
        assert result.exit_price == pytest.approx(98.0)

    def test_both_flags_breakeven_only_when_trail_not_armed(self):
        """When both flags on but only breakeven armed (MFE < trail threshold), BREAKEVEN_SL fires."""
        # Use a high breakeven trigger vs a very low trail mult to create a gap
        cfg = StrategyConfig(enable_breakeven_stop=True, breakeven_trigger_r=1.0,
                             enable_trailing_stop=True, trailing_stop_mult=3.0)
        trade = self._bearish_trade(entry=100.0, gap=4.0)
        # gap=4: breakeven at 1×gap=4 pts, trail at 3×gap=12 pts
        # MFE=6 pts: breakeven armed (6≥4), trail NOT armed (6<12)
        # Breakeven SL = 100 (entry); trail not active
        mfe_pts = 6.0
        result = check_exit(_bar(high=100.0, low=90.0), trade, bars_held=10, config=cfg, mfe_pts=mfe_pts)
        assert result is not None
        assert result.reason == ExitReason.BREAKEVEN_SL
        assert result.exit_price == pytest.approx(100.0)


class TestCheckExitBullish:
    """Long trade: SL is below entry, TP is above entry."""

    def _trade(self, entry: float = 200.0, gap: float = 4.0) -> EntryDecision:
        return EntryDecision(
            direction=Direction.BULLISH,
            entry_price=entry,
            sl_price=entry - _CONFIG.sl_multiplier * gap,  # 180.0
            tp_price=entry + _CONFIG.tp_multiplier * gap,  # 224.0
            contracts=5,
        )

    def test_sl_hit_bar_low_below_sl(self):
        trade = self._trade()
        result = check_exit(_bar(high=205.0, low=179.0), trade, bars_held=5, config=_CONFIG)
        assert result is not None
        assert result.reason == ExitReason.SL
        assert result.exit_price == pytest.approx(trade.sl_price)

    def test_tp_hit_bar_high_above_tp(self):
        trade = self._trade()
        result = check_exit(_bar(high=225.0, low=195.0), trade, bars_held=5, config=_CONFIG)
        assert result is not None
        assert result.reason == ExitReason.TP
        assert result.exit_price == pytest.approx(trade.tp_price)

    def test_sl_takes_precedence_when_both_hit(self):
        trade = self._trade()
        result = check_exit(_bar(high=230.0, low=175.0), trade, bars_held=5, config=_CONFIG)
        assert result.reason == ExitReason.SL

    def test_time_stop(self):
        trade = self._trade()
        result = check_exit(
            _bar(high=205.0, low=195.0, close=200.0),
            trade,
            bars_held=_CONFIG.max_hold_bars,
            config=_CONFIG,
        )
        assert result.reason == ExitReason.TIME_STOP
        assert result.exit_price == pytest.approx(200.0)

    def test_no_exit_inside_levels(self):
        trade = self._trade()
        result = check_exit(
            _bar(high=205.0, low=195.0),
            trade,
            bars_held=_CONFIG.max_hold_bars - 1,
            config=_CONFIG,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Benchmark harness (AC #14) — lives outside pure functions, prints to stdout
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import zoneinfo as _zi
    import numpy as _np
    from src.research.strategy_core import (
        calc_atr,
        calc_profit_factor,
        calc_sharpe,
        calc_max_drawdown,
        detect_fvg,
        detect_liquidity_sweep,
        resample_to_h1,
        volatility_regime_filter,
    )

    N = 1_000
    ny_tz = _zi.ZoneInfo("America/New_York")

    # Synthetic inputs for decision functions
    sweep = _bearish_sweep()
    fvg = _bearish_fvg()
    config = _CONFIG
    bar = _bar(high=105.0, low=95.0, close=100.0)
    trade = EntryDecision(
        direction=Direction.BEARISH,
        entry_price=100.0,
        sl_price=120.0,
        tp_price=76.0,
        contracts=5,
    )
    pnls = [10.0, -5.0, 15.0, -8.0, 20.0] * 20
    daily_returns = [0.001 * i for i in range(-10, 10)]
    equity = list(_np.cumsum([p / 100 for p in pnls]))

    # Synthetic M1 bar DataFrame (50 bars, canonical AR9 schema)
    _m1_idx = pd.date_range("2025-06-02 09:00:00", periods=50, freq="1min", tz=ny_tz)
    _m1_close = 20000.0 + _np.cumsum(_np.random.default_rng(42).normal(0, 2, 50))
    _m1_bars = pd.DataFrame(
        {
            "open": _m1_close - 1,
            "high": _m1_close + 2,
            "low": _m1_close - 2,
            "close": _m1_close,
            "volume": _np.ones(50, dtype=_np.int64) * 100,
        },
        index=_m1_idx,
    )
    _m1_bars.index.name = "timestamp"

    # Synthetic H1 bar DataFrame (150 bars — exceeds vol_regime_lookback=120)
    _h1_idx = pd.date_range("2025-01-01 09:00:00", periods=150, freq="1h", tz=ny_tz)
    _h1_close = 20000.0 + _np.cumsum(_np.random.default_rng(7).normal(0, 10, 150))
    _h1_bars = pd.DataFrame(
        {
            "open": _h1_close - 5,
            "high": _h1_close + 10,
            "low": _h1_close - 10,
            "close": _h1_close,
            "volume": _np.ones(150, dtype=_np.int64) * 500,
        },
        index=_h1_idx,
    )
    _h1_bars.index.name = "timestamp"

    def _bench(fn, *args, **kwargs):
        latencies = []
        for _ in range(N):
            t0 = _time_mod.perf_counter_ns()
            fn(*args, **kwargs)
            latencies.append(_time_mod.perf_counter_ns() - t0)
        latencies.sort()
        p50 = latencies[N // 2] / 1_000
        p95 = latencies[int(N * 0.95)] / 1_000
        mx = latencies[-1] / 1_000
        print(f"  {fn.__name__:30s}  p50={p50:8.1f}µs  p95={p95:8.1f}µs  max={mx:8.1f}µs")

    print(f"\n--- strategy_core latency benchmark ({N} iterations) ---")
    _bench(make_entry_decision, sweep, fvg, config, kill_zone=True)
    _bench(check_exit, bar, trade, 5, config)
    _bench(calc_profit_factor, pnls)
    _bench(calc_sharpe, daily_returns)
    _bench(calc_max_drawdown, equity)
    _bench(calc_atr, _m1_bars)
    _bench(detect_fvg, _m1_bars, config, 100.0)
    _bench(resample_to_h1, _m1_bars)
    _bench(volatility_regime_filter, _h1_bars, config)
    _bench(detect_liquidity_sweep, _h1_bars, config)
    print("--- done ---\n")
