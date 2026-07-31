"""Commingling-safe startup recovery for MIM-NB (shares account+contract with YANK).

The critical safety property: _reconcile_startup must NEVER flatten net position or
cancel-all orders (that would clobber YANK). It recovers from MIM's own state file
and verifies its own cat-stop by ID. These tests inject the persisted state + a fake
ProjectX client and assert only own-ID actions occur.
"""
import pytest


@pytest.fixture
def bot(monkeypatch):
    monkeypatch.setenv("PROJECTX_ACCOUNT_ID", "23884932")
    monkeypatch.setenv("MIM_NB_AUTOROLL", "0")
    from src.research.mim_nb_live import MimNbLive
    return MimNbLive()


class _Px:
    """Fake ProjectXClient. is_order_open returns the configured flag; records any
    cancels. NOTE: it has NO cancel_all_pending_orders / close_position_at_market —
    if the code under test tried to call them, the test would AttributeError (proving
    those net-position paths are gone)."""
    def __init__(self, open_flag):
        self._open = open_flag
        self.cancelled = []

    async def is_order_open(self, oid):
        return self._open

    async def cancel_orders(self, ids):
        self.cancelled.extend(ids)
        return [str(i) for i in ids if i]


async def _run(bot, state, open_flag):
    bot.px = _Px(open_flag)
    bot._load_persisted_position = lambda: state
    booked = []
    def _rec(px, t, reason):
        booked.append((px, reason))
        bot.position = 0
    bot._record_trade = _rec
    await bot._reconcile_startup()
    return booked


@pytest.mark.asyncio
async def test_believed_flat_no_action(bot):
    booked = await _run(bot, {"position": 0}, open_flag=True)
    assert bot.position == 0
    assert bot.px.cancelled == [] and booked == []


@pytest.mark.asyncio
async def test_resume_holding_when_cat_stop_live(bot):
    state = {"position": -1, "entry_px": 30000.0, "entry_t": "13:30", "cat_stop_id": 111, "day": "2026-06-17"}
    booked = await _run(bot, state, open_flag=True)
    assert bot.position == -1 and bot.entry_px == 30000.0 and bot.cat_stop_id == 111
    assert booked == [] and bot.px.cancelled == []          # nothing flattened/cancelled


@pytest.mark.asyncio
async def test_offline_cat_stop_fill_is_booked(bot):
    # Derive the stop distance from the sealed constant rather than hardcoding it. This
    # assertion read `30000.0 - 500.0` and silently went stale when the catstop-250 seal
    # halved CAT_STOP_PTS, failing on a correct system.
    from src.research.mim_nb_live import CAT_STOP_PTS
    state = {"position": 1, "entry_px": 30000.0, "entry_t": "13:30", "cat_stop_id": 222, "day": "2026-06-17"}
    booked = await _run(bot, state, open_flag=False)         # cat-stop gone -> filled offline
    assert booked and booked[0][1] == "CAT_STOP_OFFLINE"
    assert booked[0][0] == 30000.0 - CAT_STOP_PTS            # LONG stop = entry - cat-stop
    assert bot.cat_stop_id is None
    assert bot.px.cancelled == [222]                          # only OUR cat-stop id


@pytest.mark.asyncio
async def test_offline_cat_stop_fill_is_booked_short(bot):
    """Mirror of the LONG case — the SHORT half of the same stop_px expression was
    never covered, so a sign error there would have gone unnoticed."""
    from src.research.mim_nb_live import CAT_STOP_PTS
    state = {"position": -1, "entry_px": 30000.0, "entry_t": "13:30", "cat_stop_id": 444, "day": "2026-06-17"}
    booked = await _run(bot, state, open_flag=False)
    assert booked and booked[0][1] == "CAT_STOP_OFFLINE"
    assert booked[0][0] == 30000.0 + CAT_STOP_PTS            # SHORT stop = entry + cat-stop
    assert bot.cat_stop_id is None
    assert bot.px.cancelled == [444]


@pytest.mark.asyncio
async def test_unknown_status_resumes_holding(bot):
    state = {"position": -1, "entry_px": 30000.0, "cat_stop_id": 333}
    booked = await _run(bot, state, open_flag=None)          # API error
    assert bot.position == -1 and bot.cat_stop_id == 333
    assert booked == [] and bot.px.cancelled == []
