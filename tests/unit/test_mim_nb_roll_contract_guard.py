"""Roll contract contamination — spec spec-mim-nb-roll-contract-contamination-fix.

On 2026-09-15 MIM-NB auto-rolled MNQU26 → MNQZ26 *inside* `on_bar`, then anchored the
session on the pre-roll bar it was already holding. `open_d` became 29127.00 (U26) while
every later mark was Z26, about 293 points higher, so the bands sat ~293 pts below the
mark and `c > ub` was arithmetic rather than signal: a forced ENTER_LONG held to EOD,
−$355 on the live combine. Two further effects — `_prev_close_for_symbol` handed
back the retired contract's close (logged as `spread +0.00 pt`), and all 390 moves were
computed as a Z26 close over a U26 open, ~1% each, folded into 14 sessions of sigma.

The repair has two halves, and these tests pin both:

  1. bars are stamped with the contract they were FETCHED under (never the handle-time
     symbol — the roll happens between fetch and handle, which is the whole defect), and
     a bar whose stamp is not the active contract may not touch open_d, today_moves,
     VWAP or sigma;
  2. a hash-chained session→contract record, so the prior-close lookup and the sigma
     seed can tell which contract a recorded session belongs to.

Conventions follow tests/unit/test_mim_nb_sigma_provenance.py: hand-written CSV
fixtures, `object.__new__(MimNbLive)` with only the attributes under test, module
globals patched with monkeypatch. No network, no credentials.
"""
import csv
import logging
from datetime import datetime

import pytest
import pytz

from src.research import mim_nb_live as M
from src.research.mim_nb_live import (MimNbLive, BAR_SYMBOL_KEY, FULL_SESSION_BARS,
                                      LOOKBACK_DAYS, RTH_FIRST, RTH_LAST)

ET = pytz.timezone("America/New_York")
OLD, NEW = "MNQU26", "MNQZ26"
DAY = "2026-09-15"

# The incident's own numbers: the U26 open the bot latched onto, and the Z26 level the
# marks actually printed at (~293 pts higher).
U26_OPEN = 29127.00
Z26_OPEN = 29420.25


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolate_records(tmp_path, monkeypatch):
    """Point both chained records at tmp_path so no test touches data/mim_nb/."""
    monkeypatch.setattr(M, "SESSIONS_CSV", tmp_path / "sessions.csv")
    monkeypatch.setattr(M, "sessions_log",
                        M.ChainedCsv(tmp_path / "sessions.csv",
                                     ["day_et", "symbol", "first_ts_utc", "mixed",
                                      "detail"]))
    monkeypatch.setattr(M, "bars_log",
                        M.ChainedCsv(tmp_path / "bars_raw_written.csv",
                                     ["ts_utc", "open", "high", "low", "close",
                                      "volume", "received_at"]))
    monkeypatch.setattr(M, "BARS_RAW_CSV", tmp_path / "bars_raw.csv")
    monkeypatch.setattr(M, "WARMUP_CSV", tmp_path / "no_warmup.csv")


def _bar(hm, o, c, symbol=NEW, day=DAY, volume=10):
    """A TradeStation-shaped bar as _ts_get_bars returns it. ET → UTC is +4h in EDT."""
    h, m = hm.split(":")
    b = {"TimeStamp": f"{day}T{int(h) + 4:02d}:{m}:00Z",
         "Open": o, "High": max(o, c) + 1, "Low": min(o, c) - 1,
         "Close": c, "TotalVolume": volume}
    if symbol is not None:
        b[BAR_SYMBOL_KEY] = symbol
    return b


def _bot(symbol=NEW, day=None, sigma_hist=None, prev_close=29000.0):
    """A MimNbLive carrying only what on_bar touches on the paths under test."""
    o = object.__new__(MimNbLive)
    o.symbol = symbol
    o.day = day
    o.open_d = None
    o.cum_pv = 0.0
    o.cum_v = 0.0
    o.today_moves = {}
    o.today_saw_close = False
    o.position = 0
    o.day_pnl = 0.0
    o.day_deactivated = False
    o.entry_px = 0.0
    o.entry_t = None
    o.cat_stop_id = None
    o.prev_close = prev_close
    o.sigma_hist = sigma_hist if sigma_hist is not None else {}
    o.sigma_days = []
    o._early_close_logged = None
    # buffer bookkeeping: logged at every entry mark, gates nothing since the floor
    # gating removal — present so an entry attempt reaches _enter and is caught there
    o._realized_pnl = 0.0
    o._mll_eod_hwm = M.COMBINE_START_BALANCE
    o._buffer_source = "own-ledger"
    o._session_contract = None
    o._session_contract_day = None
    o._session_contract_mixed = False
    o._roll_drop_key = None
    o._roll_drop_n = 0
    o._save_state = lambda: None

    async def _no_roll():
        """The roll itself is exercised in TestPrevCloseAtRoll; here it has already
        happened (self.symbol is the NEW contract) — which is precisely the state in
        which the stale bar arrives."""
        return None
    o._maybe_roll = _no_roll

    async def _no_entry(*a, **k):
        raise AssertionError("an entry was attempted on a session that must stand down")
    o._enter = _no_entry
    return o


def _write_bars_csv(path, days, symbols=None, marks=None):
    """bars_raw-shaped CSV: one whole-ish session per day, opens drifting per day."""
    marks = marks or [RTH_FIRST, "10:00", "12:00", "15:30", RTH_LAST]
    rows = ["ts_utc,open,high,low,close,volume,received_at,chain"]
    for i, d in enumerate(days):
        base = 29000.0 + i * 10 + (300.0 if symbols and symbols[i] == NEW else 0.0)
        for j, hm in enumerate(marks):
            h, m = hm.split(":")
            ts = f"{d}T{int(h) + 4:02d}:{m}:00Z"
            c = base + j
            rows.append(f"{ts},{base},{c + 5},{base - 5},{c},100,{ts},abc")
    path.write_text("\n".join(rows) + "\n")


def _write_sessions_csv(path, rows):
    """rows = [(day, symbol, mixed), ...] — written as the bot would append them."""
    lines = ["day_et,symbol,first_ts_utc,mixed,detail,chain"]
    for d, sym, mixed in rows:
        lines.append(f"{d},{sym},{d}T13:31:00Z,{mixed},,abc")
    path.write_text("\n".join(lines) + "\n")


async def _drive(bot, bars):
    for b in bars:
        await bot.on_bar(b)


# ----------------------------------------------------------------------
# Matrix row 1 — normal session
# ----------------------------------------------------------------------
class TestNormalSession:
    """Every bar fetched under the current symbol: unchanged from today."""

    @pytest.mark.asyncio
    async def test_open_anchor_and_moves_are_unchanged(self):
        bot = _bot()
        await _drive(bot, [_bar(RTH_FIRST, Z26_OPEN, Z26_OPEN + 5),
                           _bar("09:32", Z26_OPEN + 5, Z26_OPEN + 9)])
        assert bot.open_d == Z26_OPEN
        assert bot.today_moves[RTH_FIRST] == abs((Z26_OPEN + 5) / Z26_OPEN - 1.0)
        assert bot.today_moves["09:32"] == abs((Z26_OPEN + 9) / Z26_OPEN - 1.0)
        assert bot.cum_v == 20

    @pytest.mark.asyncio
    async def test_stamping_changes_nothing_versus_an_unstamped_bar(self):
        """Acceptance: a single-contract session behaves identically to today.

        Driven twice with the same bars, once carrying the fetch stamp and once with no
        stamp at all (the pre-fix shape), the resulting state must be identical.
        """
        bars = [_bar(RTH_FIRST, Z26_OPEN, Z26_OPEN + 5),
                _bar("09:32", Z26_OPEN + 5, Z26_OPEN + 9),
                _bar("09:33", Z26_OPEN + 9, Z26_OPEN - 2)]
        stamped, legacy = _bot(), _bot()
        await _drive(stamped, bars)
        await _drive(legacy, [{k: v for k, v in b.items() if k != BAR_SYMBOL_KEY}
                              for b in bars])
        assert stamped.open_d == legacy.open_d
        assert stamped.today_moves == legacy.today_moves
        assert (stamped.cum_pv, stamped.cum_v) == (legacy.cum_pv, legacy.cum_v)

    @pytest.mark.asyncio
    async def test_session_is_recorded_under_the_active_contract(self, tmp_path):
        bot = _bot()
        await _drive(bot, [_bar(RTH_FIRST, Z26_OPEN, Z26_OPEN + 5),
                           _bar("09:32", Z26_OPEN + 5, Z26_OPEN + 9)])
        rec = MimNbLive._read_session_symbols(tmp_path / "sessions.csv")
        assert rec == {DAY: {"symbol": NEW, "mixed": False}}

    @pytest.mark.asyncio
    async def test_whole_session_still_folds_into_sigma(self, monkeypatch):
        """The guard must not disturb the 16:00 fold on a clean session."""
        monkeypatch.setattr(M, "FULL_SESSION_BARS", 2)
        bot = _bot()
        await _drive(bot, [_bar(RTH_FIRST, Z26_OPEN, Z26_OPEN + 5),
                           _bar(RTH_LAST, Z26_OPEN + 5, Z26_OPEN + 20)])
        assert bot.sigma_days == [DAY]
        assert bot.prev_close == Z26_OPEN + 20


# ----------------------------------------------------------------------
# Matrix row 2 — roll at the session boundary (the 2026-09-15 shape)
# ----------------------------------------------------------------------
class TestRollAtSessionBoundary:
    """A bar fetched under U26 arrives after _maybe_roll has switched to Z26."""

    @pytest.mark.asyncio
    async def test_stale_open_bar_cannot_anchor_the_session(self):
        bot = _bot(symbol=NEW)
        await _drive(bot, [_bar(RTH_FIRST, U26_OPEN, U26_OPEN + 5, symbol=OLD)])
        assert bot.open_d is None, "a retired contract's open must never anchor the day"
        assert bot.today_moves == {}
        assert bot.cum_v == 0.0

    @pytest.mark.asyncio
    async def test_the_whole_session_stands_down_and_takes_no_entry(self):
        """After the dropped 09:31 bar, the real Z26 bars cannot re-open the session:
        open_d is None and the poll loop consumes each timestamp once, so the only
        outcome is a stand-down. `_enter` raises if anything tries."""
        sigma = {hm: [0.001] * LOOKBACK_DAYS
                 for hm in [RTH_FIRST, "10:00", "10:30", RTH_LAST]}
        bot = _bot(symbol=NEW, sigma_hist=sigma)
        await _drive(bot, [_bar(RTH_FIRST, U26_OPEN, U26_OPEN + 5, symbol=OLD),
                           _bar("10:00", Z26_OPEN, Z26_OPEN + 40),
                           _bar("10:30", Z26_OPEN + 40, Z26_OPEN + 90),
                           _bar(RTH_LAST, Z26_OPEN + 90, Z26_OPEN + 120)])
        assert bot.position == 0
        assert bot.open_d is None
        assert bot.today_moves == {}

    @pytest.mark.asyncio
    async def test_rolled_session_is_not_folded_into_sigma(self, monkeypatch):
        monkeypatch.setattr(M, "FULL_SESSION_BARS", 2)
        bot = _bot(symbol=NEW)
        await _drive(bot, [_bar(RTH_FIRST, U26_OPEN, U26_OPEN + 5, symbol=OLD),
                           _bar(RTH_LAST, Z26_OPEN, Z26_OPEN + 20)])
        assert bot.sigma_days == []
        assert bot.sigma_hist == {}

    @pytest.mark.asyncio
    async def test_stale_bar_is_still_recorded_in_bars_raw(self, tmp_path):
        """Dropping is a state decision, not a censorship one: the bar record stays a
        complete account of what the feed delivered."""
        bot = _bot(symbol=NEW)
        await _drive(bot, [_bar(RTH_FIRST, U26_OPEN, U26_OPEN + 5, symbol=OLD)])
        rows = list(csv.DictReader((tmp_path / "bars_raw_written.csv").open()))
        assert len(rows) == 1 and float(rows[0]["open"]) == U26_OPEN

    @pytest.mark.asyncio
    async def test_drop_is_logged_once_naming_both_symbols_and_the_ts(self, caplog):
        bot = _bot(symbol=NEW)
        with caplog.at_level(logging.WARNING):
            await _drive(bot, [_bar(RTH_FIRST, U26_OPEN, U26_OPEN + 5, symbol=OLD),
                               _bar("09:32", U26_OPEN + 5, U26_OPEN + 8, symbol=OLD)])
        guard = [r for r in caplog.records if "ROLL GUARD" in r.getMessage()]
        assert len(guard) == 1, "log once per roll, not once per dropped bar"
        msg = guard[0].getMessage()
        assert OLD in msg and NEW in msg
        assert f"{DAY}T13:31:00Z" in msg
        assert bot._roll_drop_n == 2, "every dropped bar is still counted"

    @pytest.mark.asyncio
    async def test_rolled_session_is_marked_mixed_in_the_record(self, tmp_path):
        bot = _bot(symbol=NEW)
        await _drive(bot, [_bar(RTH_FIRST, U26_OPEN, U26_OPEN + 5, symbol=OLD),
                           _bar("10:00", Z26_OPEN, Z26_OPEN + 40)])
        rec = MimNbLive._read_session_symbols(tmp_path / "sessions.csv")
        assert rec[DAY]["mixed"] is True

    @pytest.mark.asyncio
    async def test_unstamped_bars_are_never_dropped(self):
        """Legacy/replay bars carry no stamp. Absence must not be read as staleness, or
        every such bar would stand the bot down."""
        bot = _bot(symbol=NEW)
        b = _bar(RTH_FIRST, Z26_OPEN, Z26_OPEN + 5, symbol=None)
        await bot.on_bar(b)
        assert bot.open_d == Z26_OPEN


# ----------------------------------------------------------------------
# Matrix rows 3, 4, 6 — prior close at a roll
# ----------------------------------------------------------------------
class TestPrevCloseContractFilter:

    @pytest.mark.asyncio
    async def test_record_of_another_contract_is_not_returned(self, tmp_path,
                                                              monkeypatch):
        """Matrix row 3: only old-contract sessions on file → use the fetch."""
        _write_bars_csv(tmp_path / "bars_raw.csv", ["2026-09-12"], symbols=[OLD])
        _write_sessions_csv(tmp_path / "sessions.csv", [("2026-09-12", OLD, 0)])
        bot = object.__new__(MimNbLive)
        bot.symbol = NEW
        fetched = []

        async def _fetch(barsback=1500):
            fetched.append(barsback)
            return [{"TimeStamp": "2026-09-12T20:00:00Z", "Open": 1, "High": 1,
                     "Low": 1, "Close": 29999.75, "TotalVolume": 1},
                    {"TimeStamp": "2026-09-12T13:31:00Z", "Open": 1, "High": 1,
                     "Low": 1, "Close": 29900.0, "TotalVolume": 1}]
        bot._ts_get_bars = _fetch

        px = await bot._prev_close_for_symbol(NEW)
        assert fetched, "another contract's record must not answer; fetch instead"
        assert px == 29999.75

    @pytest.mark.asyncio
    async def test_record_of_the_requested_contract_is_used(self, tmp_path):
        """Matrix row 4: a session recorded under the requested symbol is trusted."""
        _write_bars_csv(tmp_path / "bars_raw.csv", ["2026-09-12"], symbols=[NEW])
        _write_sessions_csv(tmp_path / "sessions.csv", [("2026-09-12", NEW, 0)])
        bot = object.__new__(MimNbLive)
        bot.symbol = NEW

        async def _boom(barsback=1500):
            raise AssertionError("the recorded session should have been used")
        bot._ts_get_bars = _boom

        px = await bot._prev_close_for_symbol(NEW)
        assert px == 29300.0 + 4      # base 29000 + 300 (NEW) + last mark index

    @pytest.mark.asyncio
    async def test_legacy_session_without_a_record_is_not_trusted(self, tmp_path):
        """Matrix row 6: a session predating SESSIONS_CSV is unknown, so it is refused
        here and the broker fetch answers instead."""
        _write_bars_csv(tmp_path / "bars_raw.csv", ["2026-09-12"], symbols=[NEW])
        bot = object.__new__(MimNbLive)
        bot.symbol = NEW
        fetched = []

        async def _fetch(barsback=1500):
            fetched.append(barsback)
            return []
        bot._ts_get_bars = _fetch

        assert await bot._prev_close_for_symbol(NEW) is None
        assert fetched, "an unrecorded session must not be trusted as this contract"

    @pytest.mark.asyncio
    async def test_mixed_session_is_not_trusted(self, tmp_path):
        _write_bars_csv(tmp_path / "bars_raw.csv", ["2026-09-12"], symbols=[NEW])
        _write_sessions_csv(tmp_path / "sessions.csv", [("2026-09-12", NEW, 1)])
        bot = object.__new__(MimNbLive)
        bot.symbol = NEW
        fetched = []

        async def _fetch(barsback=1500):
            fetched.append(barsback)
            return []
        bot._ts_get_bars = _fetch

        assert await bot._prev_close_for_symbol(NEW) is None
        assert fetched

    @pytest.mark.asyncio
    async def test_fetch_failure_still_reaches_the_re_derivation_failed_path(
            self, tmp_path):
        """Matrix row 3, error column: a failing fetch returns None, which _maybe_roll
        reports as 'ROLL prev_close re-derivation FAILED'."""
        _write_bars_csv(tmp_path / "bars_raw.csv", ["2026-09-12"], symbols=[OLD])
        _write_sessions_csv(tmp_path / "sessions.csv", [("2026-09-12", OLD, 0)])
        bot = object.__new__(MimNbLive)
        bot.symbol = NEW

        async def _fetch(barsback=1500):
            raise RuntimeError("broker down")
        bot._ts_get_bars = _fetch

        assert await bot._prev_close_for_symbol(NEW) is None


class TestZeroSpreadIsLoud:
    """A same-price re-derivation across contracts is the signature of this bug."""

    @pytest.mark.asyncio
    async def test_identical_prev_close_across_contracts_is_reported_critical(
            self, caplog, monkeypatch):
        monkeypatch.setattr(M, "AUTOROLL", True)
        bot = object.__new__(MimNbLive)
        bot.symbol = OLD
        bot.position = 0
        bot.prev_close = 29127.00
        bot.open_d = 29127.00
        bot.today_moves = {"10:00": 0.001}
        bot.today_saw_close = False
        bot.http = bot.px_auth = None

        async def _front(http, px_auth, root="MNQ"):
            return NEW
        monkeypatch.setattr(M, "resolve_front_month", _front)
        monkeypatch.setattr(M, "orders_log",
                            M.ChainedCsv(M.SESSIONS_CSV.parent / "orders_t.csv",
                                         ["ts_utc", "event", "order_id", "otype",
                                          "side", "size", "price", "outcome",
                                          "detail"]))
        bot._apply_symbol = lambda s: setattr(bot, "symbol", s)

        async def _same(sym):
            return 29127.00        # the contaminated answer
        bot._prev_close_for_symbol = _same

        with caplog.at_level(logging.WARNING):
            await bot._maybe_roll()

        assert any(r.levelno == logging.CRITICAL and "SUSPECT" in r.getMessage()
                   for r in caplog.records), "a +0.00 cross-contract spread is loud"
        assert any("ROLL prev_close re-derived" in r.getMessage()
                   for r in caplog.records), "the existing log line must be kept"


# ----------------------------------------------------------------------
# Matrix row 5 — sigma seeding after a roll
# ----------------------------------------------------------------------
class TestSigmaSeeding:

    def _seed(self, tmp_path, monkeypatch, days, symbols, session_rows):
        monkeypatch.setattr(M, "FULL_SESSION_BARS", 5)
        _write_bars_csv(tmp_path / "bars_raw.csv", days, symbols=symbols)
        _write_sessions_csv(tmp_path / "sessions.csv", session_rows)
        bot = object.__new__(MimNbLive)
        bot.sigma_hist, bot.sigma_days, bot.prev_close = {}, [], None
        bot._seed_sigma_from_bars()
        return bot

    def test_mixed_session_is_excluded_and_the_rest_still_seed(self, tmp_path,
                                                               monkeypatch):
        days = ["2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11"]
        bot = self._seed(tmp_path, monkeypatch, days, [OLD, OLD, NEW, NEW],
                         [("2026-09-08", OLD, 0), ("2026-09-09", OLD, 0),
                          ("2026-09-10", NEW, 1), ("2026-09-11", NEW, 0)])
        assert "2026-09-10" not in bot.sigma_days
        assert bot.sigma_days == ["2026-09-08", "2026-09-09", "2026-09-11"]

    def test_sessions_of_another_contract_still_seed(self, tmp_path, monkeypatch):
        """A move is a dimensionless ratio, so the filter must not be a contract match —
        that would starve depth and block entries for LOOKBACK_DAYS after every roll."""
        days = ["2026-09-08", "2026-09-09", "2026-09-11"]
        bot = self._seed(tmp_path, monkeypatch, days, [OLD, OLD, NEW],
                         [("2026-09-08", OLD, 0), ("2026-09-09", OLD, 0),
                          ("2026-09-11", NEW, 0)])
        assert bot.sigma_days == days

    def test_legacy_sessions_with_no_record_still_seed(self, tmp_path, monkeypatch):
        """Matrix row 6: absence of provenance must not retroactively void history."""
        days = ["2026-09-08", "2026-09-09", "2026-09-11"]
        monkeypatch.setattr(M, "FULL_SESSION_BARS", 5)
        _write_bars_csv(tmp_path / "bars_raw.csv", days, symbols=[OLD, OLD, NEW])
        bot = object.__new__(MimNbLive)
        bot.sigma_hist, bot.sigma_days, bot.prev_close = {}, [], None
        bot._seed_sigma_from_bars()
        assert bot.sigma_days == days


# ----------------------------------------------------------------------
# Catch-up: a restart must not undo the stand-down
# ----------------------------------------------------------------------
class TestCatchUpAfterARoll:

    def _fixed_now(self, hm="14:00", day=DAY):
        y, mo, dd = (int(x) for x in day.split("-"))
        h, m = (int(x) for x in hm.split(":"))
        pinned = ET.localize(datetime(y, mo, dd, h, m))

        class _FakeDT(datetime):
            @classmethod
            def now(cls, tz=None):
                return pinned.astimezone(tz) if tz else pinned.replace(tzinfo=None)
        return _FakeDT

    def _catchup_bot(self):
        o = object.__new__(MimNbLive)
        o.symbol = NEW
        o.day = None
        o.open_d = None
        o.today_saw_close = False
        o.cum_pv = o.cum_v = 0.0
        o.today_moves = {}
        o.day_pnl = 0.0
        o.day_deactivated = False
        o.last_bar_ts = None

        async def _boom(*a, **k):
            raise AssertionError("_catch_up_today made a network call")
        o._ts_get_bars = _boom
        return o

    @pytest.mark.asyncio
    async def test_restart_into_a_rolled_session_stands_down(self, tmp_path,
                                                             monkeypatch, caplog):
        """Without this the bar record — which has no contract of its own — would hand
        the stale U26 open straight back on the next crash restart."""
        _write_bars_csv(tmp_path / "bars_raw.csv", [DAY], symbols=[NEW])
        _write_sessions_csv(tmp_path / "sessions.csv", [(DAY, NEW, 1)])
        monkeypatch.setattr(M, "datetime", self._fixed_now())

        bot = self._catchup_bot()
        with caplog.at_level(logging.WARNING):
            await bot._catch_up_today()

        assert bot.open_d is None and bot.today_moves == {}
        assert "CATCHUP_ROLLED" in caplog.text

    @pytest.mark.asyncio
    async def test_restart_into_a_clean_session_is_unchanged(self, tmp_path,
                                                             monkeypatch):
        _write_bars_csv(tmp_path / "bars_raw.csv", [DAY], symbols=[NEW])
        _write_sessions_csv(tmp_path / "sessions.csv", [(DAY, NEW, 0)])
        monkeypatch.setattr(M, "datetime", self._fixed_now())

        bot = self._catchup_bot()
        await bot._catch_up_today()
        assert bot.open_d == 29300.0

    @pytest.mark.asyncio
    async def test_restart_with_no_provenance_on_file_is_unchanged(self, tmp_path,
                                                                   monkeypatch):
        _write_bars_csv(tmp_path / "bars_raw.csv", [DAY], symbols=[NEW])
        monkeypatch.setattr(M, "datetime", self._fixed_now())

        bot = self._catchup_bot()
        await bot._catch_up_today()
        assert bot.open_d == 29300.0


# ----------------------------------------------------------------------
# The stamp itself
# ----------------------------------------------------------------------
class TestFetchTimeStamping:

    @pytest.mark.asyncio
    async def test_tradestation_bars_are_stamped_with_the_symbol(self, monkeypatch):
        bot = object.__new__(MimNbLive)
        bot.symbol = OLD
        bot.contract_id = "CON.F.US.MNQ.U26"
        bot._data_source = "tradestation"

        class _Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"Bars": [{"TimeStamp": "2026-09-15T13:31:00Z", "Open": 1,
                                  "High": 1, "Low": 1, "Close": 1, "TotalVolume": 1}]}

        class _Http:
            async def get(self, url, headers=None):
                assert OLD in url
                return _Resp()
        bot.http = _Http()

        async def _tok():
            return "t"
        bot._ts_token = _tok

        bars = await bot._ts_get_bars(barsback=1)
        assert bars[0][BAR_SYMBOL_KEY] == OLD

    @pytest.mark.asyncio
    async def test_projectx_bars_are_stamped_too(self, monkeypatch):
        bot = object.__new__(MimNbLive)
        bot.symbol = NEW
        bot.contract_id = "CON.F.US.MNQ.Z26"
        bot._data_source = "projectx"
        bot._data_px_live = False
        bot.http = bot.px_auth = None

        async def _fetch(http, px_auth, contract_id, *, now_utc, live, barsback=None):
            return [{"TimeStamp": "2026-09-15T13:31:00Z", "Open": 1, "High": 1,
                     "Low": 1, "Close": 1, "TotalVolume": 1}]
        monkeypatch.setattr(M, "fetch_px_ts_shaped", _fetch)

        bars = await bot._ts_get_bars(barsback=1)
        assert bars[0][BAR_SYMBOL_KEY] == NEW

    @pytest.mark.asyncio
    async def test_the_stamp_is_the_fetch_time_symbol_not_the_handle_time_one(self):
        """The defect in one assertion: the roll happens between fetch and handle, so a
        stamp taken when the bar is handled would read NEW and let the bar through."""
        bot = object.__new__(MimNbLive)
        bot.symbol = OLD
        bot.contract_id = "CON.F.US.MNQ.U26"
        bot._data_source = "tradestation"

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"Bars": [{"TimeStamp": "2026-09-15T13:31:00Z", "Open": 1,
                                  "High": 1, "Low": 1, "Close": 1, "TotalVolume": 1}]}

        class _Http:
            async def get(self, url, headers=None):
                return _Resp()
        bot.http = _Http()

        async def _tok():
            return "t"
        bot._ts_token = _tok

        bars = await bot._ts_get_bars(barsback=1)
        bot.symbol = NEW                       # the roll, after the fetch
        assert bars[0][BAR_SYMBOL_KEY] == OLD


# ----------------------------------------------------------------------
# Append-only / byte-compatibility guarantees
# ----------------------------------------------------------------------
class TestRecordCompatibility:

    def test_bars_raw_columns_are_unchanged(self):
        """The hash-chained bar record must keep its exact header: every tools/ reader
        goes by column name, and ChainedCsv would emit rows wider than the header that
        the live file was created with."""
        assert M.bars_log.fields == ["ts_utc", "open", "high", "low", "close",
                                     "volume", "received_at", "chain"]

    def test_sessions_file_is_its_own_chain_and_verifies(self, tmp_path):
        """tools/verify_chain.py derives fields from the file's own header, so the new
        file verifies with no change there. Walk the same algorithm here."""
        import hashlib
        log = M.sessions_log
        log.append({"day_et": "2026-09-15", "symbol": NEW,
                    "first_ts_utc": "2026-09-15T13:31:00Z", "mixed": 1, "detail": "x"})
        log.append({"day_et": "2026-09-16", "symbol": NEW,
                    "first_ts_utc": "2026-09-16T13:31:00Z", "mixed": 0, "detail": ""})
        with (tmp_path / "sessions.csv").open(newline="") as fh:
            reader = csv.DictReader(fh)
            fields = [f for f in reader.fieldnames if f != "chain"]
            head = "GENESIS"
            for row in reader:
                payload = "|".join(str(row.get(k, "")) for k in fields)
                head = hashlib.sha256(
                    (head + "|" + payload).encode()).hexdigest()[:16]
                assert row["chain"] == head

    def test_reader_treats_mixed_as_sticky_across_appended_rows(self, tmp_path):
        """A restart re-appends a clean row for a day already known to be mixed; the
        collapse must never launder it back to clean."""
        _write_sessions_csv(tmp_path / "sessions.csv",
                            [(DAY, NEW, 1), (DAY, NEW, 0)])
        rec = MimNbLive._read_session_symbols(tmp_path / "sessions.csv")
        assert rec[DAY]["mixed"] is True

    def test_rows_disagreeing_on_symbol_are_mixed(self, tmp_path):
        _write_sessions_csv(tmp_path / "sessions.csv",
                            [(DAY, OLD, 0), (DAY, NEW, 0)])
        rec = MimNbLive._read_session_symbols(tmp_path / "sessions.csv")
        assert rec[DAY] == {"symbol": OLD, "mixed": True}

    def test_missing_file_is_empty_not_an_error(self, tmp_path):
        assert MimNbLive._read_session_symbols(tmp_path / "absent.csv") == {}


def test_full_session_bars_constant_untouched():
    """Guard against a strategy-parameter drift sneaking in with this fix."""
    assert FULL_SESSION_BARS == 390
    assert LOOKBACK_DAYS == 14
