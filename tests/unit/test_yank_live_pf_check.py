"""Tests for tools/yank_live_pf_check.py. No live data; everything is synthetic."""
import importlib.util
import json
import sqlite3
from datetime import timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ylpf", ROOT / "tools/yank_live_pf_check.py")
ylpf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ylpf)

UTC = timezone.utc
AFTER = ylpf.ML_OFF_UTC + timedelta(days=1)
BEFORE = ylpf.ML_OFF_UTC - timedelta(days=30)


def make_db(path: Path, rows) -> None:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE trades (trader_id TEXT, timestamp TEXT, pnl REAL, symbol TEXT, "
                "exit_reason TEXT, write_mode TEXT)")
    con.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


def test_pf_and_formatting():
    assert ylpf.pf_of([100.0, -50.0]) == pytest.approx(2.0)
    assert ylpf.pf_of([-100.0, -50.0]) == pytest.approx(0.0)
    assert ylpf.pf_of([100.0]) is None            # no losing trade
    assert ylpf.pf_of([]) is None
    assert ylpf.fmt_pf(None, 0) == "n/a (no trades)"
    assert ylpf.fmt_pf(None, 3) == "n/a (no losing trade)"
    assert ylpf.fmt_pf(1.23456) == "1.235"


def test_live_trades_filters_backfill_and_other_traders(tmp_path, monkeypatch):
    db = tmp_path / "trades.db"
    make_db(db, [
        ("trader-yank", "2026-09-16T10:00:00+00:00", 100.0, "MNQZ26", "TP", "realtime"),
        ("trader-yank", "2026-09-16T11:00:00+00:00", -40.0, "MNQZ26", "SL", "backfilled"),  # excluded
        ("trader-mim-nb", "2026-09-16T12:00:00+00:00", 500.0, "MNQZ26", "TP", "realtime"),  # excluded
        ("trader-yank", "2026-09-16T09:00:00", -60.0, "MNQZ26", "SL", "realtime"),          # naive ts
    ])
    monkeypatch.setattr(ylpf, "TRADES_DB", db)
    t = ylpf.live_trades()
    assert [x[1] for x in t] == [-60.0, 100.0]            # sorted oldest first, naive treated as UTC
    assert all(x[0].tzinfo is not None for x in t)


def test_missing_ledger_raises_not_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(ylpf, "TRADES_DB", tmp_path / "nope.db")
    with pytest.raises(ylpf.LedgerUnavailable):
        ylpf.live_trades()


def test_main_reports_unknown_when_ledger_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ylpf, "TRADES_DB", tmp_path / "nope.db")
    monkeypatch.setattr(ylpf, "THRESHOLD_JSON", tmp_path / "t.json")
    monkeypatch.setattr(ylpf, "CONFIG_YAML", tmp_path / "c.yaml")
    assert ylpf.main() == 0
    out = capsys.readouterr().out
    assert "LEDGER UNAVAILABLE" in out and "FORWARD_STOP: UNKNOWN" in out
    assert "N=0" not in out                                # never reports zero trades on missing data


@pytest.mark.parametrize("pnls,expect", [
    ([], "WAITING (0/20 trades)"),
    ([10.0] * 19, "WAITING (19/20 trades)"),
    ([100.0, -50.0] * 10, "OK"),                           # PF 2.0 at N=20
    ([10.0, -100.0] * 10, "REVIEW TRIGGERED"),             # PF 0.1 at N=20
    ([100.0] * 20, "OK"),                                  # undefined PF, no losses
])
def test_forward_stop_transitions(pnls, expect):
    trades = [(AFTER + timedelta(minutes=i), p, "MNQZ26", "TP") for i, p in enumerate(pnls)]
    stats = ylpf.window_stats(trades)
    _txt, verdict = ylpf.section_forward_stop(stats, stats)
    assert verdict.startswith(expect)


def test_window_split_on_cutover():
    trades = [(BEFORE, 50.0, "MNQU26", "TP"), (AFTER, -20.0, "MNQZ26", "SL")]
    since = [t for t in trades if t[0] >= ylpf.ML_OFF_UTC]
    assert ylpf.window_stats(trades)["n"] == 2
    assert ylpf.window_stats(since)["n"] == 1 and ylpf.window_stats(since)["net"] == -20.0


@pytest.mark.parametrize("gate,yaml_val,expect", [
    (0.0, "ml_threshold: 0.0", "clean"),
    (0.5, "ml_threshold: 0.0", "DRIFT"),                   # ML filter switched back on
    (0.0, "ml_threshold: 0.50", "DRIFT"),
    (None, "ml_threshold: 0.0", "UNKNOWN"),                # gate file missing
])
def test_config_status(tmp_path, monkeypatch, gate, yaml_val, expect):
    tj = tmp_path / "t.json"
    if gate is not None:
        tj.write_text(json.dumps({"threshold": gate}))
    cy = tmp_path / "c.yaml"
    cy.write_text(f"sl_multiplier: 2.0\n{yaml_val}  # comment\n")
    monkeypatch.setattr(ylpf, "THRESHOLD_JSON", tj)
    monkeypatch.setattr(ylpf, "CONFIG_YAML", cy)
    _txt, status = ylpf.section_config()
    assert status == expect
