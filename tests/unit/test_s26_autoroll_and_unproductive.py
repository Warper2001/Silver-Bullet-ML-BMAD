"""s26-combine auto-roll + the healthcheck's "active but not working" check.

Background: trader-s26-combine polled the expired contract MBTN26 from 2026-07-30,
taking ~3,400 HTTP 404s a day with zero trades, while systemd reported the service active
and its log stayed fresh — writing 404s to a log IS freshness. Nothing flagged it for a
week. This is the second time the hand-set symbol went stale (MBTM26 before it).
"""
import inspect
import re
from datetime import datetime, timedelta, timezone

import pytest


# --------------------------------------------------------------------------- auto-roll
from src.research.btc_s26_combine import S26CombineTrader, MONTH_CODES, ROLL_LOOKAHEAD_MONTHS


class TestCandidateSymbols:
    def test_front_month_first(self):
        # August 2026 -> Q is the August code
        c = S26CombineTrader._candidate_symbols(datetime(2026, 8, 6, tzinfo=timezone.utc))
        assert c[0] == "MBTQ26"
        assert len(c) == ROLL_LOOKAHEAD_MONTHS

    def test_rolls_over_the_year(self):
        c = S26CombineTrader._candidate_symbols(datetime(2026, 12, 15, tzinfo=timezone.utc))
        assert c[:3] == ["MBTZ26", "MBTF27", "MBTG27"], "December must roll into next year"

    @pytest.mark.parametrize("month,code", list(enumerate(MONTH_CODES, start=1)))
    def test_every_month_maps_to_its_cme_code(self, month, code):
        c = S26CombineTrader._candidate_symbols(datetime(2026, month, 15, tzinfo=timezone.utc))
        assert c[0] == f"MBT{code}26"


class TestProbeSemantics:
    def test_probe_uses_firstdate_not_barsback(self):
        """The bug this nearly shipped with.

        `barsback=1` returns the last bar a contract EVER printed, so an expired contract
        answers 200-with-a-bar and looks alive — measured 2026-08-06: MBTN26 (expired)
        returned 200/1 bar to barsback=1 while the bot's own firstdate poll 404'd. A probe
        that cannot tell dead from live makes the whole auto-roll a no-op.
        """
        src = inspect.getsource(S26CombineTrader._probe_symbol)
        # Check the CODE, not the docstring — which mentions barsback precisely to
        # explain why it must not be used.
        body = src.split('"""')[2] if src.count('"""') >= 2 else src
        assert "firstdate=" in body, "probe must ask for RECENT bars"
        assert "barsback" not in body, "barsback cannot distinguish an expired contract"

    def test_probe_returns_a_count_not_a_bool(self):
        """A count is what lets the caller reject a barely-quoted far month."""
        src = inspect.getsource(S26CombineTrader._probe_symbol)
        assert "len(" in src


class TestRollGuards:
    @pytest.mark.asyncio
    async def test_pinned_symbol_disables_roll(self, monkeypatch):
        import src.research.btc_s26_combine as M
        monkeypatch.setattr(M, "SYMBOL_OVERRIDE", "MBTZ26")
        bot = object.__new__(S26CombineTrader)
        bot.symbol = "MBTN26"
        assert await bot._resolve_front_month("test") is False
        assert bot.symbol == "MBTN26", "an explicit pin must be honoured"

    @pytest.mark.asyncio
    async def test_autoroll_off_disables_roll(self, monkeypatch):
        import src.research.btc_s26_combine as M
        monkeypatch.setattr(M, "SYMBOL_OVERRIDE", None)
        monkeypatch.setattr(M, "AUTOROLL", False)
        bot = object.__new__(S26CombineTrader)
        bot.symbol = "MBTN26"
        assert await bot._resolve_front_month("test") is False

    @pytest.mark.asyncio
    async def test_roll_discards_the_old_contract_bars(self, monkeypatch):
        """Splicing two contracts' bars would corrupt every indicator downstream."""
        import src.research.btc_s26_combine as M
        monkeypatch.setattr(M, "SYMBOL_OVERRIDE", None)
        monkeypatch.setattr(M, "AUTOROLL", True)
        monkeypatch.setattr(M, "MIN_PROBE_BARS", 10)

        bot = object.__new__(S26CombineTrader)
        bot.symbol, bot.bars, bot.last_ts = "MBTN26", [{"close": 1}, {"close": 2}], "old"
        bot.consecutive_empty_polls, bot._stale_alert_fired = 99, True

        class _Auth:
            async def authenticate(self):
                return "tok"
        bot.auth = _Auth()

        async def fake_probe(sym, headers):
            return 0 if sym == "MBTN26" else 500
        bot._probe_symbol = fake_probe
        monkeypatch.setattr(S26CombineTrader, "_candidate_symbols",
                            staticmethod(lambda now=None: ["MBTN26", "MBTQ26"]))

        assert await bot._resolve_front_month("test") is True
        assert bot.symbol == "MBTQ26"
        assert bot.bars == [] and bot.last_ts is None
        assert bot.consecutive_empty_polls == 0 and bot._stale_alert_fired is False


# ------------------------------------------------------- healthcheck unproductive check
def _load_healthcheck(tmp_base):
    """Exec the healthcheck with BASE pointed at a temp tree.

    Resolved relative to THIS file, not an absolute path, so the test exercises the
    checkout it lives in — an absolute path silently tests the wrong copy from a worktree.
    """
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent.parent
    src = (repo / "tools" / "combine_ops_healthcheck.py").read_text()
    src = src.replace('BASE = Path(__file__).resolve().parent.parent',
                      f'BASE = Path(r"{tmp_base}")')
    ns = {"__name__": "hc_undertest"}
    exec(compile(src, "hc", "exec"), ns)
    return ns


class TestUnproductiveDetection:
    def _write(self, tmp_path, lines):
        logs = tmp_path / "logs"
        logs.mkdir(exist_ok=True)
        (logs / "bot.log").write_text("\n".join(lines) + "\n")
        return tmp_path

    def test_counts_recent_distress_markers(self, tmp_path):
        now = datetime.now()
        lines = [f"{(now - timedelta(minutes=i)):%Y-%m-%d %H:%M:%S},000 | WARNING | "
                 f"HTTP 404 from TS API for MBTN26" for i in range(20)]
        ns = _load_healthcheck(self._write(tmp_path, lines))
        assert ns["recent_unproductive_hits"]("logs/bot.log") == 20

    def test_ignores_markers_outside_the_window(self, tmp_path):
        old = datetime.now() - timedelta(hours=6)
        lines = [f"{(old - timedelta(minutes=i)):%Y-%m-%d %H:%M:%S},000 | WARNING | "
                 f"HTTP 404 from TS API" for i in range(50)]
        ns = _load_healthcheck(self._write(tmp_path, lines))
        assert ns["recent_unproductive_hits"]("logs/bot.log") == 0, \
            "a fault that stopped hours ago must not alarm forever"

    def test_healthy_log_scores_zero(self, tmp_path):
        now = datetime.now()
        lines = [f"{now:%Y-%m-%d %H:%M:%S},000 | INFO | polling MBTQ26 ok"] * 40
        ns = _load_healthcheck(self._write(tmp_path, lines))
        assert ns["recent_unproductive_hits"]("logs/bot.log") == 0

    def test_missing_log_returns_none(self, tmp_path):
        ns = _load_healthcheck(tmp_path)
        assert ns["recent_unproductive_hits"]("logs/nope.log") is None, \
            "freshness already reports a missing log; don't double-report one fault"

    def test_stale_data_and_autoroll_failure_are_markers(self, tmp_path):
        now = datetime.now()
        lines = [f"{now:%Y-%m-%d %H:%M:%S},000 | CRITICAL | STALE_DATA: no bars",
                 f"{now:%Y-%m-%d %H:%M:%S},000 | CRITICAL | AUTOROLL FAILED: none returned"]
        ns = _load_healthcheck(self._write(tmp_path, lines))
        assert ns["recent_unproductive_hits"]("logs/bot.log") == 2

    def test_threshold_tolerates_incidental_noise(self, tmp_path):
        """A couple of 4xx is ordinary API noise — alarming on that gets the check ignored."""
        ns = _load_healthcheck(tmp_path)
        assert ns["UNPRODUCTIVE_MIN_HITS"] > 1
