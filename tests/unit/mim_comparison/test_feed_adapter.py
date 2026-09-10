import csv
import fcntl
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone

import pytest

from research.mim_comparison.feed_adapter import adapter as a

STAMP = "2026-09-10T14:00:00+00:00"
RECEIPT = "2026-09-10T14:00:01+00:00"


def context(contract="U26", source="tradestation", time="13:59:40,000"):
    return f"2026-09-10 {time} | INFO | DATA: {source} (signal) | px_contract=CON.F.US.MNQ.{contract} live=False\n"


def request(contract="U26", time="14:00:00,900"):
    return f'2026-09-10 {time} | INFO | HTTP Request: GET https://api.tradestation.com/v3/marketdata/barcharts/MNQ{contract}?interval=1&unit=Minute&barsback=60 "HTTP/1.1 200 OK"\n'


def row(head="GENESIS", event=STAMP, receipt=RECEIPT, price="100", chain=None):
    values = [event, price, "102", "99", "101", "10", receipt]
    head = (
        chain
        or hashlib.sha256((head + "|" + "|".join(values)).encode()).hexdigest()[:16]
    )
    return a.csv_line(values + [head]), head


@pytest.fixture
def source(tmp_path):
    bars, log, state = tmp_path / "bars.csv", tmp_path / "log", tmp_path / "state"
    bars.write_bytes(a.csv_line(a.FIELDS))
    log.write_text(context() + request())
    state.mkdir()
    (state / "freeze.json").write_text("{}")
    return {"bars": str(bars), "log": str(log), "state": str(state)}


def append(config, data):
    with open(config["bars"], "ab") as f:
        f.write(data)


def decisions(config):
    with sqlite3.connect(Path(config["state"]) / "journal.sqlite") as db:
        return [
            json.loads(r[0])
            for r in db.execute("SELECT decision FROM observations ORDER BY id")
        ]


def feed(config):
    with open(Path(config["state"]) / "feed.csv") as f:
        return list(csv.DictReader(f))


def test_causal_receipt_historical_and_no_rewrite(source):
    append(source, row()[0])
    result = a.collect(source)
    actual = feed(source)[0]
    assert actual["contract"] == "MNQU26"
    assert actual["timestamp"] == STAMP and actual["received_at"] == RECEIPT
    assert [actual[k] for k in a.FIELDS[1:6]] == ["100", "102", "99", "101", "10"]
    assert result["timely_at_adapter"] == 0
    assert json.loads(actual["request_evidence"])[0]["byte_offset"] > 0
    assert a.collect(source)["mapped"] == 1
    assert len(feed(source)) == 1


@pytest.mark.parametrize(
    "logs,reason",
    [
        (context() + request("U26") + request("M26"), "ambiguous_contract"),
        (context() + request(time="14:00:01,001"), "absent_causal_request"),
        (context() + request(time="13:59:45,999"), "absent_causal_request"),
        (request(), "unknown_or_contradictory_signal_context"),
        (
            context(source="projectx") + request(),
            "unknown_or_contradictory_signal_context",
        ),
        (context("M26") + request(), "unknown_or_contradictory_signal_context"),
        (
            context()
            + "2026-09-10 13:59:50,000 | INFO | MIM-NB LIVE — MNQU26\n"
            + request(),
            "unknown_or_contradictory_signal_context",
        ),
        (
            context()
            + request()
            + "2026-09-10 14:00:00,950 | INFO | MIM-NB LIVE — MNQU26\n",
            "receipt_context_changed",
        ),
    ],
)
def test_rejections(source, logs, reason):
    Path(source["log"]).write_text(logs)
    append(source, row()[0])
    a.collect(source)
    assert decisions(source)[0]["reason"] == reason
    assert not feed(source)


def test_roll_keeps_causal_identity(source):
    Path(source["log"]).write_text(
        context()
        + request()
        + "2026-09-10 14:00:02,000 | WARNING | AUTOROLL: front month MNQU26 → MNQZ26\n"
        + request("Z26", "14:01:00,900")
    )
    first, head = row()
    second, _ = row(head, "2026-09-10T14:01:00+00:00", "2026-09-10T14:01:01+00:00")
    append(source, first + second)
    a.collect(source)
    assert [r["contract"] for r in feed(source)] == ["MNQU26", "MNQZ26"]


@pytest.mark.parametrize(
    "price,receipt",
    [
        ("nan", RECEIPT),
        ("-1", RECEIPT),
        ("100", "2099-01-01T00:00:00+00:00"),
        ("100", "2026-09-10T14:00:01"),
    ],
)
def test_invalid_values_and_receipts(source, price, receipt):
    append(source, row(price=price, receipt=receipt)[0])
    a.collect(source)
    assert decisions(source)[0]["status"] == "excluded"


def test_rejection_permanent_even_new_evidence(source):
    Path(source["log"]).write_text(context())
    first, head = row()
    append(source, first)
    a.collect(source)
    with open(source["log"], "a") as f:
        f.write(request())
    append(source, row(head)[0])
    a.collect(source)
    assert not feed(source)
    assert "permanent" in decisions(source)[1]["reason"]


def test_chain_break_new_unanchored_segment(source):
    bad, head = row(chain="broken")
    good, _ = row(head, "2026-09-10T13:59:00+00:00")
    append(source, bad + good)
    a.collect(source)
    assert decisions(source)[0]["reason"] == "chain_break"
    assert feed(source)[0]["chain_status"] == "unanchored_after_break"


def test_partial_resume_and_changed_pending_prefix(source):
    raw, _ = row()
    append(source, raw[:-1])
    a.collect(source)
    assert not decisions(source)
    append(source, raw[-1:])
    a.collect(source)
    assert len(feed(source)) == 1
    a.collect(source)
    assert len(feed(source)) == 1


@pytest.mark.parametrize("mutation", ["prefix", "truncate", "replace"])
def test_source_integrity(source, mutation):
    append(source, row()[0])
    a.collect(source)
    path = Path(source["bars"])
    if mutation == "prefix":
        path.write_bytes(path.read_bytes().replace(b"100", b"101"))
    elif mutation == "truncate":
        path.write_bytes(b"")
    else:
        path.rename(path.with_suffix(".old"))
        path.write_bytes(path.with_suffix(".old").read_bytes())
    with pytest.raises(ValueError, match="prefix|truncated|replaced"):
        a.collect(source)


def test_torn_feed_recovery_and_tamper(source, monkeypatch):
    append(source, row()[0])
    original = a.sync_feed
    calls = []

    def crash(db, state):
        calls.append(True)
        if len(calls) == 2:
            pending = db.execute("SELECT output FROM observations").fetchone()[0]
            with open(state / "feed.csv", "ab") as f:
                f.write(pending[:20])
            raise RuntimeError("crash")
        original(db, state)

    monkeypatch.setattr(a, "sync_feed", crash)
    with pytest.raises(RuntimeError):
        a.collect(source)
    monkeypatch.setattr(a, "sync_feed", original)
    a.collect(source)
    assert len(feed(source)) == 1
    output = Path(source["state"]) / "feed.csv"
    output.write_bytes(output.read_bytes().replace(b"MNQU26", b"MNQZ26"))
    with pytest.raises(ValueError, match="prefix"):
        a.collect(source)


def test_actual_sandbox_and_freeze_and_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "RUNS", tmp_path / "runs")
    bars, log, state = tmp_path / "bars.csv", tmp_path / "log", a.RUNS / "adapter"
    bars.write_bytes(a.csv_line(a.FIELDS) + row()[0])
    log.write_text(context() + request())
    result = a.launch(bars, log, state, "UTC")
    assert result["mapped"] == 1
    assert Path(result["invocation"]).is_dir()
    probe = """from research.mim_comparison.feed_adapter.sandbox import install_socket_filter
import os,socket
install_socket_filter()
for path in ['/inputs/bars.csv','/inputs/log']:
 try: open(path,'a').write('bad'); raise AssertionError('write allowed')
 except OSError: pass
assert not os.path.exists('/root/.ssh')
assert not os.path.exists('/root/.codex')
try: socket.socket(); raise AssertionError('socket allowed')
except OSError: pass
"""
    completed = subprocess.run(
        a.sandbox_command(bars, log, state, [sys.executable, "-c", probe]),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    with open(state / "lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already running"):
            a.launch(bars, log, state, "UTC")
    freeze = json.loads((state / "freeze.json").read_text())
    freeze["join_seconds"] = 20
    (state / "freeze.json").write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match="drift"):
        a.launch(bars, log, state, "UTC")


def test_mutation_during_read_stops_state(source, monkeypatch):
    append(source, row()[0])
    original = a.attribute

    def mutate(db, raw, now):
        result = original(db, raw, now)
        path = Path(source["bars"])
        path.write_bytes(path.read_bytes().replace(b"100", b"101"))
        return result

    monkeypatch.setattr(a, "attribute", mutate)
    with pytest.raises(ValueError, match="during consumption"):
        a.collect(source)
    assert not feed(source)
    with pytest.raises(ValueError, match="permanently stopped"):
        a.collect(source)


def test_malformed_identifiable_first_row_is_tombstoned(source):
    append(source, (STAMP + ",bad\n").encode())
    a.collect(source)
    append(source, row()[0])
    a.collect(source)
    assert not feed(source)
    assert "permanent" in decisions(source)[1]["reason"]


def test_unknown_event_corruption_stops_mapping(source):
    append(source, b"garbage\n")
    a.collect(source)
    append(source, row()[0])
    a.collect(source)
    assert not feed(source)
    assert decisions(source)[1]["reason"] == "prior_unidentified_corruption"


def test_crash_preserves_counts(source, monkeypatch):
    append(source, row()[0])
    original = a.sync_feed
    calls = []

    def crash(db, state):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("crash")
        original(db, state)

    monkeypatch.setattr(a, "sync_feed", crash)
    with pytest.raises(RuntimeError):
        a.collect(source)
    monkeypatch.setattr(a, "sync_feed", original)
    report = a.collect(source)
    assert report["mapped"] == 1
    assert (
        len(Path(report["invocation"], "evidence.jsonl").read_text().splitlines()) == 1
    )


def test_bar_boundary_precedes_log_snapshot(source, monkeypatch):
    first, head = row()
    append(source, first)
    original = a.checked_source

    def append_during_log(db, kind, path):
        if kind == "log":
            extra, _ = row(head, "2026-09-10T13:59:00+00:00")
            append(source, extra)
        return original(db, kind, path)

    monkeypatch.setattr(a, "checked_source", append_during_log)
    a.collect(source)
    assert len(decisions(source)) == 1


def test_initial_current_snapshot_is_excluded(source, monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 10, 14, 0, 2, tzinfo=timezone.utc)

    monkeypatch.setattr(a, "datetime", Clock)
    append(source, row()[0])
    a.collect(source)
    assert not feed(source)
    assert decisions(source)[0]["reason"] == "initial_snapshot_replay_not_prospective"


def test_duplicate_roll_preserves_evidence(source):
    roll = "2026-09-10 13:59:50,000 | WARNING | AUTOROLL: front month MNQU26 → MNQZ26\n"
    Path(source["log"]).write_text(context() + roll + roll + request("Z26"))
    append(source, row()[0])
    a.collect(source)
    evidence = json.loads(feed(source)[0]["request_evidence"])
    assert len(evidence[0]["context"]) == 3
    assert feed(source)[0]["contract"] == "MNQZ26"


def test_contradictory_roll_still_excluded(source):
    Path(source["log"]).write_text(
        context()
        + "2026-09-10 13:59:50,000 | WARNING | AUTOROLL: front month MNQM26 → MNQZ26\n"
        + request("Z26")
    )
    append(source, row()[0])
    a.collect(source)
    assert not feed(source)


def test_oversized_evidence_permanently_excluded(source):
    Path(source["log"]).write_text(context() + request() * 300)
    raw, head = row()
    append(source, raw)
    a.collect(source)
    assert decisions(source)[0]["reason"] == "oversized_request_evidence"
    assert not feed(source)  # Standard unchanged csv reader remains usable.
    append(source, row(head)[0])
    a.collect(source)
    assert "permanent" in decisions(source)[1]["reason"]


@pytest.mark.parametrize("seconds,timely", [(1, True), (60, True), (60.001, False)])
def test_fresh_append_timeliness_boundary(source, monkeypatch, seconds, timely):
    from datetime import timedelta

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 10, 14, 0, tzinfo=timezone.utc) + timedelta(
                seconds=seconds
            )

    monkeypatch.setattr(a, "datetime", Clock)
    a.collect(source)
    append(source, row()[0])
    result = a.collect(source)
    assert decisions(source)[0]["timely_at_adapter"] is timely
    assert result["timely_at_adapter"] == int(timely)
    assert feed(source)[0]["timestamp"] == STAMP
    assert feed(source)[0]["received_at"] == RECEIPT
    a.collect(source)
    assert len(feed(source)) == 1


def test_snapshot_fsync_failure_keeps_cursor(source, monkeypatch):
    import os

    append(source, row()[0])
    original = a.os.fsync
    failed = []

    def fail_once(fd):
        target = os.readlink(f"/proc/self/fd/{fd}")
        if "/poll-" in target and not failed:
            failed.append(target)
            raise OSError("snapshot fsync interruption")
        original(fd)

    monkeypatch.setattr(a.os, "fsync", fail_once)
    with pytest.raises(OSError, match="fsync interruption"):
        a.collect(source)
    with a.connect(Path(source["state"])) as db:
        assert a.get(db, "snapshot_id", 0) == 0
    failed_snapshot = next(Path(source["state"]).glob("poll-*"))
    assert "fsync interruption" in (failed_snapshot / "report.md").read_text()
    result = a.collect(source)
    snapshot = Path(result["invocation"])
    assert len((snapshot / "evidence.jsonl").read_text().splitlines()) == 1
    assert (snapshot / "report.md").exists()
    assert (
        "report.md" in json.loads((snapshot / "manifest.json").read_text())["artifacts"]
    )
    assert len(feed(source)) == 1


@pytest.mark.parametrize("mutation", ["prefix", "truncate", "replace"])
def test_log_integrity(source, mutation):
    a.collect(source)
    path = Path(source["log"])
    if mutation == "prefix":
        path.write_text(path.read_text().replace("barsback=60", "barsback=61"))
    elif mutation == "truncate":
        path.write_bytes(b"")
    else:
        old = path.with_suffix(".old")
        path.rename(old)
        path.write_bytes(old.read_bytes())
    with pytest.raises(ValueError, match="prefix|truncated|replaced"):
        a.collect(source)


def test_partial_log_completion_cannot_reinstate(source):
    Path(source["log"]).write_text(context() + request().rstrip("\n"))
    raw, head = row()
    append(source, raw)
    a.collect(source)
    assert decisions(source)[0]["reason"] == "absent_causal_request"
    with open(source["log"], "a") as log:
        log.write("\n")
    append(source, row(head)[0])
    a.collect(source)
    a.collect(source)
    assert not feed(source)
    with a.connect(Path(source["state"])) as db:
        assert (
            db.execute("SELECT COUNT(*) FROM evidence WHERE kind='request'").fetchone()[
                0
            ]
            == 1
        )


def test_log_index_interruption_rolls_back(source, monkeypatch):
    original = a.parse_log
    seen = []

    def interrupt(raw):
        seen.append(raw)
        if len(seen) == 2:
            raise RuntimeError("index interrupted")
        return original(raw)

    monkeypatch.setattr(a, "parse_log", interrupt)
    with pytest.raises(RuntimeError, match="index interrupted"):
        a.collect(source)
    monkeypatch.setattr(a, "parse_log", original)
    append(source, row()[0])
    a.collect(source)
    a.collect(source)
    assert len(feed(source)) == 1
    with a.connect(Path(source["state"])) as db:
        assert db.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 2


def test_worker_entry_installs_socket_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "RUNS", tmp_path / "runs")
    bars, log, state = tmp_path / "bars", tmp_path / "log", a.RUNS / "state"
    bars.write_bytes(a.csv_line(a.FIELDS))
    log.write_text(context())
    result = a.launch(bars, log, state, "UTC")
    assert Path(result["invocation"]).is_dir()
    script = """import sys,socket
from research.mim_comparison.feed_adapter import __main__ as worker
def probe(config):
 try: socket.socket()
 except PermissionError: return {'denied': True}
 raise AssertionError('actual worker collector could create socket')
worker.collect = probe
sys.argv = ['adapter', '--worker']
worker.main()
"""
    completed = subprocess.run(
        a.sandbox_command(bars, log, state, [sys.executable, "-c", script]),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["denied"] is True


def test_feed_enters_unchanged_shadow_and_excludes_missing_session(
    source, tmp_path, monkeypatch
):
    import pandas as pd
    from research.mim_comparison import shadow
    from research.mim_comparison.artifacts import BASE, digest
    from test_comparison import session

    append(source, row()[0])
    a.collect(source)
    state, output = tmp_path / "shadow-state", tmp_path / "shadow-output"
    state.mkdir()
    output.mkdir()
    warm = pd.concat(
        [
            session(str(day.date())).bars
            for day in pd.bdate_range("2026-08-03", "2026-08-28")
        ]
    )
    warm_path = tmp_path / "warmup.csv"
    warm.to_csv(warm_path, index=False)
    (state / "freeze.json").write_text(
        json.dumps(
            {
                "protocol": {"freeze": "2026-09-09T22:00:00+00:00"},
                "warmup_hash": digest(warm_path),
                "labels": "end",
                "source": {p.name: digest(p) for p in BASE.glob("*.py")},
            }
        )
    )
    monkeypatch.setattr(shadow, "_now", lambda: pd.Timestamp("2026-09-10T20:02:00Z"))
    monkeypatch.setattr(shadow, "install_socket_filter", lambda: None)
    shadow.collect(
        {
            "state": str(state),
            "warmup": str(warm_path),
            "data": str(Path(source["state"]) / "feed.csv"),
            "labels": "end",
            "result": str(output / "status.json"),
        }
    )
    with sqlite3.connect(state / "journal.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM invalid_rows").fetchone()[0] == 0
        assert db.execute("SELECT available,reason FROM observations").fetchone() == (
            0,
            "late_backfilled_or_future_receipt",
        )
        eligible, reason = db.execute(
            "SELECT eligible,reason FROM sessions WHERE day='2026-09-10'"
        ).fetchone()
        assert eligible == 0 and "missing" in reason
