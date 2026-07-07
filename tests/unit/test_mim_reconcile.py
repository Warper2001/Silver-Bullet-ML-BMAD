"""Unit tests for MIM-NB ops hardening (seal: preregistration_mim_nb_ops_hardening.md):
stop-fill reconcile decision routing + early-close session gate."""
import asyncio
import os

import pytest

os.environ.setdefault("PROJECTX_ACCOUNT_ID", "1")

from src.research import mim_nb_live as m  # noqa: E402


class FakePx:
    def __init__(self, open_flag):
        self.open_flag = open_flag
        self.canceled = []

    async def is_order_open(self, oid):
        return self.open_flag

    async def cancel_orders(self, ids):
        self.canceled += list(ids)
        return []


def make_bot(position=1, entry_px=30000.0, cat_stop_id=111, open_flag=False):
    bot = m.MimNbLive()
    bot.position = position
    bot.entry_px = entry_px
    bot.entry_t = "10:30"
    bot.cat_stop_id = cat_stop_id
    bot.px = FakePx(open_flag)
    bot.booked = []
    bot._record_trade = lambda px, t, reason: bot.booked.append((px, reason)) or setattr(bot, "position", 0)
    return bot


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_stop_still_open_noop():
    bot = make_bot(open_flag=True)
    run(bot._reconcile_stop_fill())
    assert bot.booked == [] and bot.cat_stop_id == 111


def test_stop_status_unknown_noop():
    bot = make_bot(open_flag=None)
    run(bot._reconcile_stop_fill())
    assert bot.booked == [] and bot.cat_stop_id == 111


def test_genuine_fill_books_cat_stop_at_stop_level():
    bot = make_bot(position=1, entry_px=30000.0)

    async def fill(oid):
        return {"orderId": oid, "price": 29748.25, "side": 1, "size": 1,
                "fees": 0.36, "profitAndLoss": -503.5}
    bot._find_fill_for = fill
    run(bot._reconcile_stop_fill())
    assert bot.booked == [(30000.0 - m.CAT_STOP_PTS, "CAT_STOP")]
    assert bot.cat_stop_id is None
    assert 111 in bot.px.canceled  # mirror hygiene


def test_external_close_books_broker_truth_and_safe_mode():
    bot = make_bot(position=1, entry_px=30012.25)

    async def nofill(oid):
        return None

    async def closing():
        return {"orderId": 999, "price": 29940.5, "side": 1, "size": 1,
                "profitAndLoss": -165.0, "creationTimestamp": "2026-07-06T17:30:58Z"}
    bot._find_fill_for = nofill
    bot._find_closing_fill = closing
    bot.day_deactivated = False
    run(bot._reconcile_stop_fill())
    assert bot.booked == [(29940.5, "EXTERNAL_CLOSE")]
    assert bot.day_deactivated is True
    assert bot.cat_stop_id is None


def test_vanished_stop_replaces_protection():
    bot = make_bot(position=-1, entry_px=29300.0)

    async def none_(*a):
        return None
    bot._find_fill_for = none_
    bot._find_closing_fill = none_
    placed = []

    async def order(otype, side, price=None):
        placed.append((otype, side, price))
        return 222
    bot._order = order
    run(bot._reconcile_stop_fill())
    assert bot.booked == []
    assert bot.cat_stop_id == 222
    assert placed == [(m._TYPE_STOP if hasattr(m, "_TYPE_STOP") else placed[0][0],
                       placed[0][1], 29300.0 + m.CAT_STOP_PTS)]


def test_flat_or_no_stop_noop():
    bot = make_bot(position=0)
    run(bot._reconcile_stop_fill())
    bot2 = make_bot(cat_stop_id=None)
    run(bot2._reconcile_stop_fill())
    assert bot.booked == [] and bot2.booked == []


def test_early_close_gate_stands_down():
    bot = make_bot(position=0, cat_stop_id=None)
    assert "2026-11-26" in m.EARLY_CLOSE_DATES
    bar = {"TimeStamp": "2026-11-26T15:30:00Z",  # 10:30 ET on Thanksgiving
           "Open": 30000.0, "High": 30010.0, "Low": 29990.0, "Close": 30005.0,
           "TotalVolume": 100}
    run(bot.on_bar(bar))
    assert bot.day is None          # no session initialized
    assert bot.open_d is None       # no open anchor
    assert bot.today_moves == {}    # no sigma accumulation
    assert bot._early_close_logged is not None
