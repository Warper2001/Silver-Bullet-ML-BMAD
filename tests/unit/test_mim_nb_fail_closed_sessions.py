"""Fail-closed session acceptance — prereg sigma-provenance Amendment 3 (D-G1..D-G4).

The old rule accepted any day whose first RTH bar was 09:31 and which had a 16:00 bar.
2026-07-13 (169 bars — a four-hour hole from a mid-session halt) sailed through it and
skewed every affected minute label for the next 14 sessions. After the 2026-08-06 file
damage the record also contained 2026-07-31 with 417 rows for 389 distinct minutes, which
a naive length check reads as *longer* than complete.

A session now counts only if it is whole: FULL_SESSION_BARS distinct RTH minute labels.
Refusing to contribute beats contributing something wrong.
"""
import pytest

from src.research import mim_nb_live as M
from src.research.mim_nb_live import MimNbLive, FULL_SESSION_BARS, RTH_FIRST, RTH_LAST


def _session(n_minutes=FULL_SESSION_BARS, dupes=0, drop_open=False, drop_close=False):
    """A synthetic RTH session as (hm, open, close) rows, 09:31 + n-1 following minutes."""
    rows, h, m = [], 9, 31
    for i in range(n_minutes):
        rows.append((f"{h:02d}:{m:02d}", 100.0, 100.0 + i))
        m += 1
        if m > 59:
            m, h = 0, h + 1
    if dupes:
        rows.extend(rows[:dupes])              # re-append overlap, as the git revert made
    if drop_open:
        rows = [r for r in rows if r[0] != RTH_FIRST]
    if drop_close:
        rows = [r for r in rows if r[0] != RTH_LAST]
    return rows


def _accept(rows):
    """The seed's acceptance predicate, as implemented."""
    labels = {hm for hm, _o, _c in rows}
    return len(labels) == FULL_SESSION_BARS


class TestSeedAcceptance:
    def test_complete_session_is_accepted(self):
        """D-G4 — the rule must not reject good days."""
        assert _accept(_session()) is True

    @pytest.mark.parametrize("n", [1, 169, 388, 389])
    def test_incomplete_session_is_rejected(self, n):
        """D-G1 — 169 is the real 2026-07-13 halt session; 388 is 2026-07-29."""
        assert _accept(_session(n)) is False

    def test_duplicates_collapse_and_do_not_fake_completeness(self):
        """D-G2 — 2026-07-31: 417 rows, 389 distinct minutes. Must still reject."""
        rows = _session(389, dupes=28)
        assert len(rows) > FULL_SESSION_BARS, "raw length exceeds a full session"
        assert _accept(rows) is False, "duplicates must not disguise a short session"

    def test_duplicates_on_a_complete_session_still_accepted(self):
        rows = _session(FULL_SESSION_BARS, dupes=25)
        assert len(rows) > FULL_SESSION_BARS
        assert _accept(rows) is True

    def test_hole_in_the_middle_is_caught(self):
        """The failure the old rule could not see: correct endpoints, missing middle."""
        rows = [r for r in _session() if not ("11:30" <= r[0] <= "14:30")]
        assert rows[0][0] == RTH_FIRST and any(r[0] == RTH_LAST for r in rows), \
            "endpoints intact — the old rule would have accepted this"
        assert _accept(rows) is False


class TestFoldFailsClosed:
    """D-G3 — the same rule on the live path."""

    def _bot(self, n_labels, day="2026-08-06"):
        o = object.__new__(MimNbLive)
        o.sigma_hist, o.sigma_days = {}, []
        o.day, o.open_d = day, 100.0
        o.today_moves = {f"m{i}": 0.001 for i in range(n_labels)}
        o.today_saw_close = True
        return o

    def test_partial_live_session_is_not_folded(self):
        bot = self._bot(200)
        bot._fold_today_into_sigma()
        assert bot.sigma_days == [], "a partial session must contribute nothing"
        assert bot.sigma_hist == {}

    def test_complete_live_session_is_folded(self):
        bot = self._bot(FULL_SESSION_BARS)
        bot._fold_today_into_sigma()
        assert bot.sigma_days == ["2026-08-06"]
        assert len(bot.sigma_hist) == FULL_SESSION_BARS

    def test_fold_still_idempotent(self):
        bot = self._bot(FULL_SESSION_BARS)
        bot._fold_today_into_sigma()
        bot._fold_today_into_sigma()
        assert bot.sigma_days == ["2026-08-06"]
        assert all(len(v) == 1 for v in bot.sigma_hist.values())

    def test_rejection_is_logged_with_the_count(self, caplog):
        """D-G3 — a silent refusal is how the original defect hid for a month."""
        import logging
        bot = self._bot(200)
        with caplog.at_level(logging.WARNING):
            bot._fold_today_into_sigma()
        assert "FOLD REJECT" in caplog.text
        assert "200" in caplog.text and str(FULL_SESSION_BARS) in caplog.text


class TestConstant:
    def test_full_session_is_390_minutes(self):
        """09:31..16:00 inclusive. If this changes, the RTH window changed."""
        assert FULL_SESSION_BARS == 390
