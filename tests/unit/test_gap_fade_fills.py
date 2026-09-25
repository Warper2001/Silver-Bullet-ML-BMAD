"""Unit tests for gap-fade realized-fill logging (Friday-review blocker #1)."""
import asyncio

import pytest

from src.research import gap_fade_live as g


class FakeTs:
    def __init__(self, statuses):
        self.statuses = statuses

    async def fetch_orders_status(self, order_ids):
        return {str(k): v for k, v in self.statuses.items() if k in [str(i) for i in order_ids if i]}


class CaptureLog:
    def __init__(self):
        self.rows = []

    def append(self, row):
        self.rows.append(row)


def make_trader(statuses):
    t = object.__new__(g.GapFadeTrader)
    t._ts = FakeTs(statuses)
    t._fills_log = CaptureLog()
    return t


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _patch_sleep(monkeypatch):
    async def nosleep(_):
        return None
    monkeypatch.setattr(g.asyncio, "sleep", nosleep)


def test_realized_pnl_long_tp(monkeypatch):
    _patch_sleep(monkeypatch)
    c = g.CONTRACTS
    t = make_trader({"1": {"status": "FLL", "exec_price": 29205.5, "exec_qty": c},
                     "2": {"status": "FLL", "exec_price": 29388.0, "exec_qty": c}})
    run(t._log_realized_fills({"date_et": "2026-07-08", "dir": "L", "outcome": "fill",
                               "direction": 1, "modeled_pnl": 366.0 * c,
                               "entry_id": "1", "exit_role": "tp", "exit_id": "2"}))
    row = t._fills_log.rows[0]
    assert row["realized_pnl_usd"] == pytest.approx(365.0 * c)   # (29388.0-29205.5)*2 per ct
    assert row["delta_usd"] == pytest.approx(-1.0 * c)
    assert row["exit_role"] == "tp"


def test_realized_pnl_short_direction(monkeypatch):
    _patch_sleep(monkeypatch)
    c = g.CONTRACTS
    t = make_trader({"1": {"status": "FLL", "exec_price": 30000.0, "exec_qty": c},
                     "2": {"status": "FLL", "exec_price": 29900.0, "exec_qty": c}})
    run(t._log_realized_fills({"date_et": "2026-07-09", "dir": "S", "outcome": "stop",
                               "direction": -1, "modeled_pnl": 190.0 * c,
                               "entry_id": "1", "exit_role": "sl", "exit_id": "2"}))
    assert t._fills_log.rows[0]["realized_pnl_usd"] == pytest.approx(200.0 * c)


def test_incomplete_broker_data_records_blanks(monkeypatch):
    _patch_sleep(monkeypatch)
    t = make_trader({"1": {"status": "FLL", "exec_price": 29205.5, "exec_qty": 1}})
    run(t._log_realized_fills({"date_et": "2026-07-08", "dir": "L", "outcome": "time",
                               "direction": 1, "modeled_pnl": -50.0,
                               "entry_id": "1", "exit_role": "close", "exit_id": None}))
    row = t._fills_log.rows[0]
    assert row["realized_pnl_usd"] == "" and row["delta_usd"] == ""
    assert row["entry_exec"] == 29205.5 and row["exit_exec"] == ""


def test_fetch_failure_never_raises(monkeypatch):
    _patch_sleep(monkeypatch)
    t = object.__new__(g.GapFadeTrader)

    class Boom:
        async def fetch_orders_status(self, ids):
            raise RuntimeError("api down")
    t._ts = Boom()
    t._fills_log = CaptureLog()
    run(t._log_realized_fills({"date_et": "x", "dir": "L", "outcome": "fill",
                               "direction": 1, "modeled_pnl": 0.0,
                               "entry_id": "1", "exit_role": "tp", "exit_id": "2"}))
    assert t._fills_log.rows == []   # swallowed, logged, no crash


def test_stop_wakes_idle_wait_promptly():
    """stop() must end run()'s idle wait at once, not after the 300s outside-RTH sleep
    (systemd kills after 90s; every journaled stop before this fix ended in SIGKILL)."""
    t = object.__new__(g.GapFadeTrader)
    t.running = False
    t.http = None
    t._trade_open = False
    ticks = []

    async def fake_tick():
        ticks.append(1)

    t._process_tick = fake_tick

    async def scenario():
        task = asyncio.ensure_future(t.run())
        await asyncio.sleep(0.05)                 # first tick done, now in the idle wait
        await t.stop()
        return await asyncio.wait_for(task, timeout=2.0)

    assert run(scenario()) is True               # orderly stop, well inside 2s
    assert ticks == [1]
