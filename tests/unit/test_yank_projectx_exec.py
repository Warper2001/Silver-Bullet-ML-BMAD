"""YANK ProjectX execution port — the deferred-bracket lifecycle.

On TradeStation SIM, the entry+TP+SL bracket is placed all at once. On ProjectX the
entry limit is placed alone and TP/SL are DEFERRED until the entry fills. This test
drives _advance_active_trade through a fill on the combine path and asserts the bot
places its exits then (and only then), tagging them as its own order IDs.
"""
import datetime as dt

import pytest
import pytz

from src.research.yank_streaming_working import Tier2StreamingTrader, ActiveTrade
from src.research.strategy_core import EntryDecision, Direction
from src.data.models import DollarBar

UTC = pytz.UTC


class _Exec:
    """Mock ProjectX exec: records place_exit_orders / close / cancel calls."""
    def __init__(self):
        self.placed = None
        self.closed = None
        self.cancelled = []

    async def place_exit_orders(self, decision, account_id):
        self.placed = (decision, account_id)
        return ("TP9", "SL9")

    async def close_position_at_market(self, direction, account_id, contracts=None):
        self.closed = (direction, account_id, contracts)
        return "CL1"

    async def cancel_order(self, oid):
        self.cancelled.append(oid)
        return True


def _bar(o, h, l, c):
    # 2026-07-01 14:00 UTC = 10:00 ET = 09:00 CT -> NOT in the 15:08-17:00 CT flatten window
    ts = UTC.localize(dt.datetime(2026, 7, 1, 14, 0))
    return DollarBar(timestamp=ts, open=o, high=h, low=l, close=c, volume=100, notional_value=1.0)


@pytest.mark.asyncio
async def test_projectx_places_exits_on_fill():
    t = Tier2StreamingTrader(symbol="MNQM26")
    t._on_combine = True
    t._exec_account = "23884932"
    t._contracts = 2
    t._ts_client = _Exec()

    dec = EntryDecision(direction=Direction.BEARISH, entry_price=30000.0,
                        sl_price=30100.0, tp_price=29900.0, contracts=2)
    t._active_entry_decision = dec
    t.active_trade = ActiveTrade(
        bar_index=0, entry_time=dt.datetime(2026, 7, 1, 13, 30),
        direction="SHORT", entry_price=30000.0, tp_price=29900.0, sl_price=30100.0,
        sim_entry_order_id="E1", pending_entry=True,
    )
    # bar fills the SHORT (high >= 30000) but does NOT hit TP (low>29900) or SL (high<30100)
    await t._advance_active_trade(_bar(30000, 30005, 29950, 30000))

    assert t.active_trade is not None and not t.active_trade.pending_entry   # filled
    assert t._ts_client.placed is not None                                  # exits placed on fill
    assert t._ts_client.placed[1] == "23884932"                             # routed to combine acct
    assert t.active_trade.sim_tp_order_id == "TP9"
    assert t.active_trade.sim_sl_order_id == "SL9"
    assert t._ts_client.closed is None                                      # no exit yet


@pytest.mark.asyncio
async def test_sim_path_does_not_defer_exits():
    """On the SIM path, TP/SL already exist (set at submit) — no deferred placement."""
    t = Tier2StreamingTrader(symbol="MNQM26")
    t._on_combine = False
    t._exec_account = "SIM2797251F"
    t._ts_client = _Exec()   # would record if place_exit_orders were (wrongly) called

    dec = EntryDecision(direction=Direction.BEARISH, entry_price=30000.0,
                        sl_price=30100.0, tp_price=29900.0, contracts=5)
    t._active_entry_decision = dec
    t.active_trade = ActiveTrade(
        bar_index=0, entry_time=dt.datetime(2026, 7, 1, 13, 30),
        direction="SHORT", entry_price=30000.0, tp_price=29900.0, sl_price=30100.0,
        sim_entry_order_id="E1", sim_tp_order_id="TP_SIM", sim_sl_order_id="SL_SIM",
        pending_entry=True,
    )
    await t._advance_active_trade(_bar(30000, 30005, 29950, 30000))
    assert t._ts_client.placed is None                  # SIM: no deferred placement
    assert t.active_trade.sim_tp_order_id == "TP_SIM"   # unchanged
