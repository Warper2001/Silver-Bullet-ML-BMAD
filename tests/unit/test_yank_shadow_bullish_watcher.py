"""Bullish shadow watcher (2026-08-19) — read-only, never trades.

Built after preregistration_yank_bidirectional_m15_choch.md's Amendment 2: the
§5 backtest pass didn't survive a temporal split (H1 net loser, H2 strongly
positive), so before any live-wiring decision, this watches real forward
bullish signals — using the same validated detect_m15_choch_bullish gate as
BacktestEngine — and logs what a bidirectional bot would have done, without
ever placing an order.

Contract under test:
  1. the shadow path never calls _ts_client / StatePersistence / touches
     active_trade — structurally incapable of real execution
  2. a synthetic bullish scenario arms, fills, and closes a shadow trade,
     logged to a file distinct from the real trade log
  3. backfill bars never arm or advance a shadow trade (same discipline as
     the real path, same restart-replay defect class this project has hit
     repeatedly)
  4. the shadow log path is never the real trade log path
"""
import os
from datetime import datetime, timezone

os.environ.setdefault("PROJECTX_ACCOUNT_ID", "1")

import pytest

from src.data.models import DollarBar
from src.research import yank_streaming_working as y
from src.research.strategy_core import Direction, EntryDecision, StrategyConfig, SweepSignal


class ExplodingClient:
    """Any call is a test failure — the shadow path must never reach this."""
    def __getattr__(self, name):
        def _boom(*a, **kw):
            raise AssertionError(f"shadow path called _ts_client.{name} — must never execute")
        return _boom


def _bar(ts, o, h, l, c, v=10) -> DollarBar:
    return DollarBar(timestamp=ts, open=o, high=h, low=l, close=c, volume=v,
                     notional_value=o * v * 5.0)


def make_bot(tmp_path, is_backfill=False):
    bot = object.__new__(y.Tier2StreamingTrader)
    bot._ts_client = ExplodingClient()
    bot._strategy_config = StrategyConfig(bearish_only=True)  # real path stays bearish-only
    bot._point_value = 2.0
    bot._contracts = 2
    bot._h1_atr = 50.0
    bot._last_vol_regime_pct = 0.5
    bot._is_backfill = is_backfill
    bot.dollar_bars = []
    bot.h1_bullish_sweep_active = True
    bot._cached_sweep = SweepSignal(direction=Direction.BULLISH, bars_ago=0, sweep_price=100.0)
    bot._shadow_bullish_m15_choch_active = True
    bot._shadow_trade = None
    bot.active_trade = None  # real path — must stay untouched
    bot._active_entry_decision = None
    bot._shadow_logger = y.TradeLogger(log_path=tmp_path / "yank_shadow_bullish_trades.csv")
    return bot


class TestShadowNeverExecutes:
    def test_detect_entry_never_touches_ts_client(self, tmp_path):
        bot = make_bot(tmp_path)
        bot.dollar_bars = [_bar(datetime(2025, 6, 2, 14, m, tzinfo=timezone.utc),
                                100.0, 101.0, 99.0, 100.0) for m in range(20)]
        bar = bot.dollar_bars[-1]
        bot._detect_shadow_bullish_entry(bar, is_backfill=False)  # must not raise / must not call client
        assert bot.active_trade is None  # real state untouched

    def test_advance_never_touches_ts_client(self, tmp_path):
        bot = make_bot(tmp_path)
        bot._shadow_trade = {
            "entry_time": datetime(2025, 6, 2, 14, 0, tzinfo=timezone.utc),
            "entry_price": 100.0, "tp_price": 108.0, "sl_price": 96.0,
            "gap_size": 4.0, "h1_sweep_bars_ago": 0, "pending": True, "bars_held": 0,
            "entry_decision": EntryDecision(direction=Direction.BULLISH, entry_price=100.0,
                                            sl_price=96.0, tp_price=108.0, contracts=2),
        }
        bar = _bar(datetime(2025, 6, 2, 14, 1, tzinfo=timezone.utc), 99.0, 100.5, 98.5, 100.0)
        bot._advance_shadow_trade(bar)  # must not raise / must not call client
        assert bot.active_trade is None


class TestShadowLifecycle:
    def test_arms_fills_and_closes_on_tp(self, tmp_path):
        bot = make_bot(tmp_path)
        t0 = datetime(2025, 6, 2, 14, 0, tzinfo=timezone.utc)
        # 3-bar bullish FVG: c3.low > c1.high, c2 bullish body, gap satisfies config
        bot.dollar_bars = (
            [_bar(t0.replace(minute=m), 100.0, 100.5, 99.5, 100.0) for m in range(17)]
            + [_bar(t0.replace(minute=17), 100.0, 100.5, 99.5, 100.2)]   # c1
            + [_bar(t0.replace(minute=18), 100.2, 108.0, 100.2, 107.0)]  # c2 (bullish body)
            + [_bar(t0.replace(minute=19), 110.0, 111.0, 109.5, 110.5)]  # c3 (low > c1.high)
        )
        cfg = StrategyConfig(bearish_only=True, atr_threshold=0.0, min_gap_atr_ratio=0.0,
                             max_gap_dollars=9999.0, sl_multiplier=2.0, tp_multiplier=8.0,
                             entry_pct=0.5, max_pending_bars=240, max_hold_bars=60,
                             commission_per_roundtrip=4.0)
        bot._strategy_config = cfg
        bot._h1_atr = 50.0

        bar = bot.dollar_bars[-1]
        bot._detect_shadow_bullish_entry(bar, is_backfill=False)
        assert bot._shadow_trade is not None, "a qualifying bullish FVG must arm a shadow trade"
        assert bot._shadow_trade["pending"] is True

        entry_price = bot._shadow_trade["entry_price"]
        tp_price = bot._shadow_trade["tp_price"]

        fill_bar = _bar(t0.replace(minute=20), entry_price + 0.5, entry_price + 0.5,
                        entry_price - 0.1, entry_price)
        bot._advance_shadow_trade(fill_bar)
        assert bot._shadow_trade["pending"] is False

        tp_bar = _bar(t0.replace(minute=21), tp_price, tp_price + 1.0, tp_price - 0.5, tp_price)
        bot._advance_shadow_trade(tp_bar)
        assert bot._shadow_trade is None, "closed trade must clear shadow state"

        rows = list(__import__("csv").DictReader(open(bot._shadow_logger._log_path)))
        assert len(rows) == 1
        assert rows[0]["direction"] == "LONG"
        assert rows[0]["exit_reason"] == "TP"

    def test_backfill_bars_never_arm_a_shadow_trade(self, tmp_path):
        bot = make_bot(tmp_path, is_backfill=True)
        t0 = datetime(2025, 6, 2, 14, 0, tzinfo=timezone.utc)
        bot.dollar_bars = (
            [_bar(t0.replace(minute=m), 100.0, 100.5, 99.5, 100.0) for m in range(17)]
            + [_bar(t0.replace(minute=17), 100.0, 100.5, 99.5, 100.2)]
            + [_bar(t0.replace(minute=18), 100.2, 108.0, 100.2, 107.0)]
            + [_bar(t0.replace(minute=19), 110.0, 111.0, 109.5, 110.5)]
        )
        bot._strategy_config = StrategyConfig(bearish_only=True, atr_threshold=0.0,
                                              min_gap_atr_ratio=0.0, max_gap_dollars=9999.0)
        bar = bot.dollar_bars[-1]
        bot._detect_shadow_bullish_entry(bar, is_backfill=True)
        assert bot._shadow_trade is None, "backfill must never arm a shadow trade"

    def test_backfill_never_advances_an_existing_shadow_trade(self, tmp_path):
        bot = make_bot(tmp_path, is_backfill=True)
        bot._shadow_trade = {
            "entry_time": datetime(2025, 6, 2, 14, 0, tzinfo=timezone.utc),
            "entry_price": 100.0, "tp_price": 108.0, "sl_price": 96.0,
            "gap_size": 4.0, "h1_sweep_bars_ago": 0, "pending": False, "bars_held": 0,
            "entry_decision": EntryDecision(direction=Direction.BULLISH, entry_price=100.0,
                                            sl_price=96.0, tp_price=108.0, contracts=2),
        }
        bar = _bar(datetime(2025, 6, 2, 14, 1, tzinfo=timezone.utc), 108.0, 108.5, 107.5, 108.0)
        bot._advance_shadow_trade(bar)
        assert bot._shadow_trade is not None, "backfill must not advance/close a shadow trade"


class TestShadowLogIsolation:
    def test_shadow_log_path_differs_from_real_trade_log(self, tmp_path):
        shadow_logger = y.TradeLogger(log_path=tmp_path / "yank_shadow_bullish_trades.csv")
        assert shadow_logger._log_path != y.TradeLogger._LOG_PATH
        assert "shadow" in str(shadow_logger._log_path)
