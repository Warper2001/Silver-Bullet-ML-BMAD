"""Synthetic tests for tools/yank_frontmonth_revalidation.py. No market data, no holdout."""
import asyncio
import csv
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("yfr", ROOT / "tools/yank_frontmonth_revalidation.py")
yfr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(yfr)

UTC = timezone.utc


def raw_json(path: Path, recs: list[tuple[str, float, str]]) -> None:
    path.write_text(json.dumps([{"High": str(c + 1), "Low": str(c - 1), "Open": str(c), "Close": str(c),
                                 "TimeStamp": ts, "TotalVolume": "10", "Contract": k} for ts, c, k in recs], indent=2))


def csv26(path: Path, rows: list[tuple[str, float]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume", "notional"])
        for ts, c in rows:
            w.writerow([ts, c, c + 1, c - 1, c, 5, c * 5 * 20])


END = datetime(2026, 5, 19, 23, 59, 59, tzinfo=UTC)


def test_build_corrected_segments_delta_and_drop(tmp_path):
    raw_json(tmp_path / "raw.json", [
        ("2026-01-05T15:00:00Z", 100.0, "MNQH26"),                     # S1
        ("2026-03-03T14:00:00Z", 200.0, "MNQH26"),                     # S2 session 03-03, pure
        ("2026-03-10T14:00:00Z", 300.0, "MNQH26"),                     # S2 session 03-10, interleaved
        ("2026-03-10T14:01:00Z", 520.0, "MNQM26"),
        ("2026-03-11T14:00:00Z", 400.0, "MNQH26"),                     # S2 session 03-11, pure (last kept)
        ("2026-03-11T14:01:00Z", 401.0, "MNQH26"),
        ("2026-03-11T14:02:00Z", 402.0, "MNQH26"),
        ("2026-03-12T14:00:00Z", 999.0, "MNQM26"),                     # after roll: raw ignored
    ])
    csv26(tmp_path / "c26.csv", [
        ("2026-03-11T14:00:00+00:00", 620.0),                          # common minutes: diffs 220, 222, 221
        ("2026-03-11T14:01:00+00:00", 623.0),
        ("2026-03-11T14:02:00+00:00", 623.0),
        ("2026-03-12T14:00:00+00:00", 700.0),                          # S3
        ("2026-05-20T14:00:00+00:00", 800.0),                          # after END: excluded
    ])
    m = yfr.build_corrected_2026(tmp_path / "raw.json", tmp_path / "c26.csv", END, tmp_path / "out.csv")
    assert m["S2_sessions_dropped"] == ["2026-03-10"]
    assert m["S2_sessions_kept"] == ["2026-03-03", "2026-03-11"]
    assert m["delta_pts"] == 221.0 and m["delta_common_minutes"] == 3
    rows = list(csv.DictReader(open(tmp_path / "out.csv")))
    assert [r["timestamp"] for r in rows] == ["2026-01-05T15:00:00+00:00", "2026-03-03T14:00:00+00:00",
                                              "2026-03-11T14:00:00+00:00", "2026-03-11T14:01:00+00:00",
                                              "2026-03-11T14:02:00+00:00", "2026-03-12T14:00:00+00:00"]
    s3 = rows[-1]
    assert float(s3["close"]) == 700.0 - 221.0 and float(s3["open"]) == 700.0 - 221.0
    assert float(s3["notional"]) == pytest.approx((700.0 - 221.0) * 5 * 20)
    assert float(rows[0]["notional"]) == pytest.approx(100.0 * 10 * 20)
    assert m["out_sha256"] == yfr.sha256(tmp_path / "out.csv")


def test_s1_foreign_label_refuses(tmp_path):
    raw_json(tmp_path / "raw.json", [("2026-02-10T15:00:00Z", 100.0, "MNQM26")])
    csv26(tmp_path / "c26.csv", [])
    with pytest.raises(SystemExit, match="S1"):
        yfr.build_corrected_2026(tmp_path / "raw.json", tmp_path / "c26.csv", END, tmp_path / "out.csv")


def test_no_kept_s2_leaves_s3_unshifted(tmp_path):
    raw_json(tmp_path / "raw.json", [("2026-03-10T14:00:00Z", 300.0, "MNQH26"),
                                     ("2026-03-10T14:01:00Z", 520.0, "MNQM26")])
    csv26(tmp_path / "c26.csv", [("2026-03-12T14:00:00+00:00", 700.0)])
    m = yfr.build_corrected_2026(tmp_path / "raw.json", tmp_path / "c26.csv", END, tmp_path / "out.csv")
    assert m["delta_unshifted"] and m["delta_pts"] == 0.0 and m["S2_sessions_kept"] == []
    assert float(list(csv.DictReader(open(tmp_path / "out.csv")))[0]["close"]) == 700.0


def _git(repo, *a):
    subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)


def test_sealing_commit_check(tmp_path):
    repo = tmp_path / "r"
    (repo / "tools").mkdir(parents=True)
    (repo / "_bmad-output").mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / yfr.HARNESS_REL).write_bytes(b"harness v1")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "no doc")
    no_doc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    (repo / yfr.SEALED_DOC).write_text("sealed")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "seal")
    sealed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    yfr.check_sealing_commit(repo, sealed, b"harness v1")
    with pytest.raises(SystemExit, match="differs"):
        yfr.check_sealing_commit(repo, sealed, b"harness v2")
    with pytest.raises(SystemExit, match="does not contain"):
        yfr.check_sealing_commit(repo, no_doc, b"harness v1")
    with pytest.raises(SystemExit, match="not a commit"):
        yfr.check_sealing_commit(repo, "HEAD; rm -rf /", b"harness v1")


@pytest.mark.parametrize("ml,noml,want", [
    ({"n": 28, "pf": 1.25}, {"n": 35, "pf": 1.10}, "KEEP"),
    ({"n": 28, "pf": 1.15}, {"n": 35, "pf": 1.10}, "REVERT"),          # below 1.20
    ({"n": 28, "pf": 1.30}, {"n": 35, "pf": 1.40}, "REVERT"),          # not above no-ML
    ({"n": 24, "pf": 3.00}, {"n": 35, "pf": 1.00}, "INCONCLUSIVE"),    # N below 25
    ({"n": 28, "pf": None}, {"n": 35, "pf": 1.00}, "REVERT"),          # no losing trade -> undefined PF
])
def test_decide(ml, noml, want):
    assert yfr.decide(ml, noml).startswith(want)


def _trades(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["entry_time", "exit_time", "direction", "entry_price", "exit_price",
                                          "exit_type", "bars_held", "pnl"])
        w.writeheader()
        for t, p in rows:
            w.writerow({"entry_time": t, "exit_time": t, "direction": "SHORT", "entry_price": 1, "exit_price": 1,
                        "exit_type": "tp", "bars_held": 1, "pnl": p})


def test_score_g0_gate_and_segments(tmp_path, monkeypatch):
    june = {"ml": tmp_path / "jml.csv", "noml": tmp_path / "jnoml.csv"}
    rows = [("2025-06-02T14:00:00+00:00", 100.0), ("2026-01-05T15:00:00+00:00", -50.0),
            ("2026-03-03T14:00:00+00:00", 80.0), ("2026-04-01T14:00:00+00:00", 120.0)]
    for p in june.values():
        _trades(p, rows)
    monkeypatch.setattr(yfr, "JUNE_CSV", june)
    out = tmp_path / "out"
    out.mkdir()
    _trades(out / "trades_original_ml.csv", rows)
    _trades(out / "trades_original_noml.csv", rows[:-1])                  # G0 mismatch on no-ML
    r = yfr.score(SimpleNamespace(out_dir=str(out)))
    assert r["G0"]["ml"]["reproduces_june_row_for_row"] and not r["G0"]["noml"]["reproduces_june_row_for_row"]
    assert r["verdict"].startswith("NOT INTERPRETED")
    _trades(out / "trades_original_noml.csv", rows)
    _trades(out / "trades_corrected_ml.csv", rows)
    _trades(out / "trades_corrected_noml.csv", rows)
    r = yfr.score(SimpleNamespace(out_dir=str(out)))
    assert r["G0_passed"]
    seg = r["corrected_ml"]
    assert (seg["S1"]["n"], seg["S2"]["n"], seg["S3"]["n"], seg["2026"]["n"]) == (1, 1, 1, 3)
    assert seg["2026"]["pnl"] == 150.0 and seg["2026"]["pf"] == pytest.approx(200 / 50)
    assert r["verdict"].startswith("INCONCLUSIVE")                        # 3 < 25


def test_run_logs_access_before_reading_bars(tmp_path, monkeypatch):
    """Fake engine: the ACCESS_LOG row must be appended before any bar file is loaded."""
    eng = tmp_path / "eng"
    (eng / "src/research").mkdir(parents=True)
    for d in ("src", "src/research"):
        (eng / d / "__init__.py").write_text("")
    calls = tmp_path / "calls.txt"
    (eng / "backtest_tier2_1year_validation.py").write_text(f"""
from datetime import datetime, timezone
from pathlib import Path
START_DATE = datetime(2025, 5, 19, tzinfo=timezone.utc)
END_DATE = datetime(2026, 5, 19, 23, 59, 59, tzinfo=timezone.utc)
ACCESS_LOG_PATH = Path('x')
INSTRUMENTS = {{'mnq': {{'symbol': 'MNQM26'}}}}
def _rec(s):
    open({str(calls)!r}, 'a').write(s + '\\n')
def verify_preregistration(sha): _rec('verify')
def append_access_log(sha, argv): _rec('log')
def load_bars(p, start=None, end=None):
    _rec('load')
    from types import SimpleNamespace
    return [SimpleNamespace(timestamp=start or end)]
async def run_backtest(bars, ml_threshold=None, symbol=None): _rec('replay'); return []
def build_report(t, s, e): return ('', [], [])
""")
    (eng / "src/research/config_loader.py").write_text(
        "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass Cfg:\n    max_daily_loss: float = -750.0\n"
        "def load_strategy_config(p): return Cfg()\n")
    (eng / "src/research/tier2_streaming_working.py").write_text(
        "from src.research.config_loader import Cfg\ndef _build_strategy_config(): return Cfg()\n")
    monkeypatch.setattr(yfr, "check_sealing_commit", lambda *a: None)
    monkeypatch.setattr(yfr, "check_pinned", lambda root: {})
    monkeypatch.setattr(yfr, "seal_yaml", lambda root, out: (out / "cfg.yaml", (out / "cfg.yaml").write_text("x"))[0])
    monkeypatch.setattr(yfr, "guard", lambda roots: {})
    for mod in ("backtest_tier2_1year_validation", "src", "src.research", "src.research.tier2_streaming_working",
                "src.research.config_loader"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(engine_root=str(eng), out_dir=str(tmp_path / "o"), mode="original", ml_threshold=0.5,
                           preregistration="abc1234", access_log=str(tmp_path / "log.md"),
                           data_root=str(tmp_path), raw_json=str(tmp_path / "raw.json"))
    asyncio.run(yfr.run(args))
    assert calls.read_text().split() == ["verify", "log", "load", "load", "replay"]
