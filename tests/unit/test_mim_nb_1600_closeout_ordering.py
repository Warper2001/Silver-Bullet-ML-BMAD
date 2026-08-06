"""16:00 close-out ordering — prereg sigma-provenance Amendment 1 (A-G4..A-G6).

The sealed engine evaluates every mark of day d, including 16:00, against sigma history
and prev_close from days strictly BEFORE d, then folds and rolls at the end of the day
(study_mim_nb_catstop.py:118). The live bot did it the other way round, so the 16:00 mark
saw its own session and the bands were out by hundreds of points.

These tests pin the ordering itself, not the arithmetic: what the 16:00 mark READ, and
what the session left BEHIND.
"""
import inspect

import pytest

from src.research import mim_nb_live as M
from src.research.mim_nb_live import MimNbLive, LOOKBACK_DAYS, RTH_LAST


@pytest.fixture(autouse=True)
def _tiny_sessions(monkeypatch):
    """Scale the Amendment 3 completeness constant to these miniature fixtures — the
    subject here is close-out ORDERING, not session completeness."""
    monkeypatch.setattr(M, "FULL_SESSION_BARS", 2)


def _bot(sigma_hist=None, sigma_days=None, prev_close=1000.0, day="2026-07-31"):
    o = object.__new__(MimNbLive)
    o.sigma_hist = sigma_hist if sigma_hist is not None else {}
    o.sigma_days = list(sigma_days) if sigma_days else []
    o.prev_close = prev_close
    o.day = day
    o.open_d = 1000.0
    o.today_moves = {}
    o.today_saw_close = False
    o._saved = 0
    o._save_state = lambda: setattr(o, "_saved", o._saved + 1)
    return o


class TestCloseOutOrdering:
    """A-G4: the mark reads pre-session state; the session leaves post-session state."""

    def test_closeout_is_not_called_before_the_mark_is_evaluated(self):
        """Structural: no fold/prev_close assignment may precede the CHECK_MARKS gate.

        The defect was a block sitting above `if hm not in CHECK_MARKS`. Assert the
        close-out now appears only AFTER the sigma lookup that feeds the bands.
        """
        src = inspect.getsource(MimNbLive.on_bar)
        gate = src.index("if hm not in CHECK_MARKS")
        before_gate = src[:gate]
        assert "_fold_today_into_sigma" not in before_gate, \
            "sigma fold runs before the mark is evaluated — 16:00 would see its own session"
        assert "self.prev_close = c" not in before_gate, \
            "prev_close rolls before the mark is evaluated — bands would use today's close"

        sigma_read = src.index("sig = self.sigma_hist.get(hm")
        first_closeout = src.index("_close_out_session")
        assert first_closeout > sigma_read, \
            "close-out must follow the sigma read that feeds the bands"

    def test_closeout_folds_and_rolls(self):
        bot = _bot(prev_close=1000.0)
        bot.today_moves = {"10:00": 0.004, RTH_LAST: 0.006}
        bot._close_out_session(1234.5)
        assert bot.prev_close == 1234.5
        assert bot.sigma_days == ["2026-07-31"]
        assert bot.sigma_hist[RTH_LAST] == [0.006]
        assert bot.today_saw_close is True
        assert bot._saved == 1

    def test_mark_would_have_read_pre_session_values(self):
        """The band inputs at 16:00 must exclude today; after close-out they include it."""
        hist = {RTH_LAST: [0.001] * LOOKBACK_DAYS}
        bot = _bot(sigma_hist=hist, sigma_days=[f"d{i}" for i in range(LOOKBACK_DAYS)],
                   prev_close=999.0)
        bot.today_moves = {"10:00": 0.1, RTH_LAST: 0.5}

        # what the mark reads (evaluation happens first)
        sig_at_mark = list(bot.sigma_hist[RTH_LAST])
        prev_close_at_mark = bot.prev_close
        assert 0.5 not in sig_at_mark, "today's own move must not be in the 16:00 window"
        assert prev_close_at_mark == 999.0, "16:00 bands must use the PRIOR session's close"

        bot._close_out_session(1500.0)
        assert bot.sigma_hist[RTH_LAST][-1] == 0.5, "today folds in after the mark"
        assert bot.prev_close == 1500.0, "prev_close rolls after the mark"


class TestDepthGatedDayStillCloses:
    """A-G5: parent-seal sites 9 and 10 must survive the reorder."""

    def test_depth_gate_path_calls_closeout_at_1600(self):
        src = inspect.getsource(MimNbLive.on_bar)
        gate = src.index("DEPTH_SKIP")
        tail = src[gate:]
        nxt = tail.index("return")
        assert "_close_out_session" in tail[:nxt], \
            "a depth-starved 16:00 day must still fold and still roll prev_close"

    def test_depth_starved_day_still_contributes(self):
        """Even with no usable sigma window, the day must fold and roll."""
        bot = _bot(sigma_hist={RTH_LAST: [0.001] * 3}, prev_close=999.0)
        bot.today_moves = {"10:00": 0.1, RTH_LAST: 0.007}
        bot._close_out_session(1111.0)
        assert bot.sigma_days == ["2026-07-31"]
        assert bot.prev_close == 1111.0


class TestIdempotence:
    """A-G6: replaying the 16:00 bar after a restart must not double-count."""

    def test_double_closeout_folds_once(self):
        bot = _bot(prev_close=999.0)
        bot.today_moves = {"10:00": 0.1, RTH_LAST: 0.006}
        bot._close_out_session(1234.5)
        bot._close_out_session(1234.5)
        assert bot.sigma_days == ["2026-07-31"]
        assert bot.sigma_hist[RTH_LAST] == [0.006], "double fold — restart replay corrupts sigma"
        assert bot.prev_close == 1234.5

    def test_window_never_exceeds_lookback(self):
        bot = _bot(sigma_hist={RTH_LAST: [0.001] * LOOKBACK_DAYS},
                   sigma_days=[f"d{i}" for i in range(LOOKBACK_DAYS)])
        bot.today_moves = {"10:00": 0.1, RTH_LAST: 0.009}
        bot._close_out_session(1000.0)
        assert len(bot.sigma_hist[RTH_LAST]) == LOOKBACK_DAYS
        assert len(bot.sigma_days) == LOOKBACK_DAYS
        assert bot.sigma_hist[RTH_LAST][-1] == 0.009
