"""
Regression test for the frozen-firstdate bug in S26CombineTrader._resolve_front_month.

Root cause (measured live against TradeStation 2026-08-21): once the poll loop's
`since` (== self.last_ts) catches up to the most recent available bar, TradeStation
returns 404 "No data available" for that exact firstdate instead of 200+empty. The
poll loop's non-200 branch never advances last_ts, so every subsequent poll re-asks
the identical doomed query forever. The confirmed-live roll-check branch of
_resolve_front_month is the only code path that periodically re-examines this
state; before the fix, it declined to roll (correctly -- the symbol IS live) but
took no action to unstick the frozen poll.

Fix: that branch now resets self.bars/self.last_ts (same reset the actual-roll
branch already used two lines below), unsticking the next poll's `since`.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research.btc_s26_combine import S26CombineTrader


def _make_trader(symbol="MBTQ26", last_ts="frozen-sentinel", bars=None):
    """Construct a trader without running __init__ (skips TradeDatabase + model load)."""
    t = object.__new__(S26CombineTrader)
    t.symbol = symbol
    t.bars = bars if bars is not None else [{"timestamp": "old-bar"}] * 5
    t.last_ts = last_ts
    t.consecutive_empty_polls = 30
    t._stale_alert_fired = True
    t.auth = AsyncMock()
    t.auth.authenticate.return_value = "fake-token"
    return t


@pytest.mark.asyncio
async def test_still_live_resets_frozen_state_instead_of_noop():
    """Confirmed-live-but-frozen: must clear last_ts AND bars, not just last_ts."""
    t = _make_trader()
    with patch.object(S26CombineTrader, "_probe_symbol", new=AsyncMock(return_value=100)):
        rolled = await t._resolve_front_month("stale data")

    assert rolled is False  # correct: this symbol is genuinely live, no roll needed
    assert t.last_ts is None
    assert t.bars == []
    assert t.consecutive_empty_polls == 0
    assert t._stale_alert_fired is False


@pytest.mark.asyncio
async def test_genuine_roll_still_resets_the_same_way():
    """Regression guard: the actual-roll branch's existing reset is unchanged."""
    t = _make_trader(symbol="MBTQ26")

    async def fake_probe(self, sym, headers):
        return 100 if sym == "MBTU26" else 0

    with patch.object(S26CombineTrader, "_probe_symbol", new=fake_probe):
        with patch("src.research.btc_s26_combine.S26CombineTrader._candidate_symbols", return_value=["MBTQ26", "MBTU26"]):
            rolled = await t._resolve_front_month("startup")

    assert rolled is True
    assert t.symbol == "MBTU26"
    assert t.last_ts is None
    assert t.bars == []


@pytest.mark.asyncio
async def test_dead_contract_still_fails_loudly_no_silent_reset():
    """A genuinely dead contract (0 bars for every candidate) must not be
    mistaken for the frozen-but-live case -- no reset, no roll, keeps the
    symbol so the CRITICAL log fires every poll rather than going quiet."""
    t = _make_trader()
    with patch.object(S26CombineTrader, "_probe_symbol", new=AsyncMock(return_value=0)):
        with patch("src.research.btc_s26_combine.S26CombineTrader._candidate_symbols", return_value=["MBTQ26", "MBTU26", "MBTV26"]):
            rolled = await t._resolve_front_month("stale data")

    assert rolled is False
    assert t.symbol == "MBTQ26"
    assert t.last_ts == "frozen-sentinel"  # untouched -- still stuck, and still loud about it
