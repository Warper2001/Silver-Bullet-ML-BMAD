"""Unit tests for YANK crash-recovery hardening
(seal: preregistration_yank_recovery_hardening.md):
fill-aware combine reconcile (D1), backfill risk-state guard (D2),
backfill advance guard (D3)."""
import asyncio
import os
from datetime import datetime, timezone

os.environ.setdefault("PROJECTX_ACCOUNT_ID", "1")

from src.research import yank_streaming_working as y  # noqa: E402


class FakeClient:
    def __init__(self, open_map):
        self.open_map = open_map  # order_id -> True / False / None (unknown)
        self.placed_exits = []
        self.canceled = []

    async def is_order_open(self, oid):
        return self.open_map.get(str(oid), False)

    async def place_exit_orders(self, decision, account_id):
        self.placed_exits.append((decision, account_id))
        return "TP-NEW", "SL-NEW"

    async def cancel_order(self, oid):
        self.canceled.append(oid)


class FakeRisk:
    def restore_from_state(self, state, today):
        self.restored = dict(state)

    def to_state_dict(self):
        return {"daily_pnl": -212.0, "daily_halted": False,
                "last_trading_date": "2026-07-13"}


BASE_STATE = {
    "direction": "SHORT", "entry_price": 29690.5, "tp_price": 29478.5,
    "sl_price": 29743.5, "entry_time": "2026-07-13T13:35:00+00:00",
    "sim_entry_order_id": "E1", "sim_tp_order_id": "TP1", "sim_sl_order_id": "SL1",
    "gap_size": 26.5, "h1_sweep_bars_ago": 0, "m15_confirmed": True,
    "kill_zone_active": True, "vol_regime_pct": 0.04, "ml_proba": 0.576,
    "daily_pnl": -212.0, "daily_halted": False, "last_trading_date": "2026-07-13",
}


def make_bot(open_map):
    bot = object.__new__(y.Tier2StreamingTrader)
    bot._on_combine = True
    bot._ts_client = FakeClient(open_map)
    bot._exec_account = "ACCT"
    bot._contracts = 2
    bot._risk_manager = FakeRisk()
    bot.active_trade = None
    bot._active_entry_decision = None
    return bot


def recover(bot, state):
    """Run _recover_from_state with StatePersistence patched; returns (cleared, saved)."""
    cleared, saved = [], []
    orig = (y.StatePersistence.load_state, y.StatePersistence.clear_state,
            y.StatePersistence.save_state)
    y.StatePersistence.load_state = classmethod(lambda cls: dict(state))
    y.StatePersistence.clear_state = classmethod(lambda cls: cleared.append(True))
    y.StatePersistence.save_state = classmethod(lambda cls, st: saved.append(st))
    try:
        asyncio.new_event_loop().run_until_complete(bot._recover_from_state())
    finally:
        (y.StatePersistence.load_state, y.StatePersistence.clear_state,
         y.StatePersistence.save_state) = orig
    return cleared, saved


def test_phantom_closed_during_downtime_not_resumed():
    """07-13 case: entry filled, both brackets gone → clear state, do NOT resume."""
    bot = make_bot({"E1": False, "TP1": False, "SL1": False})
    cleared, _ = recover(bot, BASE_STATE)
    assert bot.active_trade is None
    assert bot._active_entry_decision is None
    assert cleared == [True]
    assert bot._ts_client.placed_exits == []


def test_live_bracket_resumes_with_decision():
    """SL still working → genuinely live trade: resume + reconstruct decision."""
    bot = make_bot({"E1": False, "TP1": False, "SL1": True})
    cleared, _ = recover(bot, BASE_STATE)
    assert cleared == []
    t = bot.active_trade
    assert t is not None and t.sim_sl_order_id == "SL1" and not t.pending_entry
    d = bot._active_entry_decision
    assert d is not None
    assert d.direction == y.Direction.BEARISH
    assert d.tp_price == 29478.5 and d.sl_price == 29743.5 and d.contracts == 2
    assert bot._ts_client.placed_exits == []  # brackets exist — no re-arm


def test_unknown_bracket_status_resumes_protectively():
    bot = make_bot({"E1": False, "TP1": False, "SL1": None})
    cleared, _ = recover(bot, BASE_STATE)
    assert cleared == [] and bot.active_trade is not None


def test_null_exit_ids_rearms_protection():
    """Crash inside the fill-detect window: resume AND place TP/SL from state prices."""
    state = dict(BASE_STATE, sim_tp_order_id=None, sim_sl_order_id=None)
    bot = make_bot({"E1": False})
    cleared, saved = recover(bot, state)
    assert cleared == []
    assert len(bot._ts_client.placed_exits) == 1
    assert bot.active_trade.sim_tp_order_id == "TP-NEW"
    assert bot.active_trade.sim_sl_order_id == "SL-NEW"
    assert saved and saved[-1]["sim_sl_order_id"] == "SL-NEW"  # IDs persisted


def test_pending_entry_cancelled_and_cleared():
    """Existing behavior preserved: entry order still open → cancel + clear."""
    bot = make_bot({"E1": True})
    cleared, _ = recover(bot, BASE_STATE)
    assert cleared == [True]
    assert "E1" in bot._ts_client.canceled
    assert bot.active_trade is None


def test_backfill_does_not_reset_daily_pnl():
    rm = y.RiskManager()
    rm._daily_pnl = -212.0
    rm._daily_halted = False
    rm._last_trading_date = datetime(2026, 7, 13, tzinfo=timezone.utc).date()
    # Backfill bar from the PREVIOUS day would normally trigger a rollover reset
    # (and a later bar back on 07-13 would reset again) — must be a no-op now.
    old_bar = datetime(2026, 7, 12, 10, 0, tzinfo=timezone.utc)
    assert rm.check_and_update(old_bar, -300.0, is_backfill=True) is False
    assert rm._daily_pnl == -212.0
    assert rm._last_trading_date == datetime(2026, 7, 13, tzinfo=timezone.utc).date()
    # Live bar on the same restored day: no reset either (dates match).
    live_bar = datetime(2026, 7, 13, 19, 0, tzinfo=timezone.utc)
    rm.check_and_update(live_bar, -300.0)
    assert rm._daily_pnl == -212.0


def test_backfill_does_not_advance_resumed_trade():
    bot = make_bot({})
    bot._is_backfill = True
    bot.active_trade = y.ActiveTrade(
        bar_index=0, entry_time=datetime(2026, 7, 13, 13, 35, tzinfo=timezone.utc),
        direction="SHORT", entry_price=29690.5, tp_price=29478.5, sl_price=29743.5,
        pending_entry=False)
    bar = type("B", (), {"timestamp": datetime(2026, 7, 12, 20, 15, tzinfo=timezone.utc),
                         "high": 29750.0, "low": 29600.0, "close": 29700.0})()
    out = asyncio.new_event_loop().run_until_complete(bot._advance_active_trade(bar))
    assert out is False
    assert bot.active_trade.bars_held == 0  # untouched — historical bar
