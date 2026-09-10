import ctypes
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_comparison import session
from research.mim_comparison import shadow
from research.mim_comparison.artifacts import BASE, digest
from research.mim_comparison.data import audit_select
from research.mim_comparison.engine import ARMS, simulate
from research.mim_comparison.references import author_vectorized, deployed_event
from research.mim_comparison.statistics import decision


@pytest.fixture
def collector(tmp_path, monkeypatch):
    warm = pd.concat(
        [
            session(str(d.date())).bars
            for d in pd.bdate_range("2025-01-01", "2025-01-29")
        ]
    )
    warm.to_csv(tmp_path / "warmup.csv", index=False)
    state = tmp_path / "state"
    state.mkdir()
    freeze = {
        "protocol": {"freeze": "2025-01-29T22:00:00+00:00"},
        "warmup_hash": digest(tmp_path / "warmup.csv"),
        "labels": "end",
        "source": {p.name: digest(p) for p in BASE.glob("*.py")},
    }
    (state / "freeze.json").write_text(json.dumps(freeze))
    clock = [pd.Timestamp("2025-01-30T14:31:01Z")]
    original = shadow._normalize

    def normalize(raw, labels):
        try:
            receipt = pd.Timestamp(raw.get("received_at"))
            if receipt.tzinfo is not None:
                clock[0] = receipt
        except (ValueError, TypeError):
            pass
        return original(raw, labels)

    monkeypatch.setattr(shadow, "_normalize", normalize)
    monkeypatch.setattr(shadow, "_now", lambda: clock[0])
    monkeypatch.setattr(shadow, "install_socket_filter", lambda: None)
    counter = [0]

    def run(frame):
        counter[0] += 1
        frame.to_csv(tmp_path / "feed.csv", index=False)
        output = tmp_path / str(counter[0])
        output.mkdir()
        config = {
            "state": str(state),
            "warmup": str(tmp_path / "warmup.csv"),
            "data": str(tmp_path / "feed.csv"),
            "labels": "end",
            "result": str(output / "status.json"),
        }
        shadow.collect(config)
        return (
            json.loads((output / "status.json").read_text()),
            sqlite3.connect(state / "journal.sqlite"),
            output,
        )

    today = session("2025-01-30").bars
    today["received_at"] = (today.timestamp + pd.Timedelta(seconds=1)).map(
        lambda t: t.isoformat()
    )
    return run, today, clock


@pytest.mark.parametrize(
    "field,value",
    [
        ("open", "NaN"),
        ("open", "inf"),
        ("open", "bad-number"),
        ("received_at", "bad-receipt"),
        ("timestamp", "bad-timestamp"),
        ("contract", None),
    ],
)
def test_invalid_first_row_cannot_be_repaired(collector, field, value):
    run, today, clock = collector
    first = today.iloc[:1].copy().astype(object)
    first.loc[0, field] = value
    status, db, _ = run(first)
    assert status["invalid_first_rows"] == 1
    db.close()
    status, db, _ = run(today.iloc[:1])
    assert status["eligible_sessions"] == 0
    assert db.execute("SELECT COUNT(*) FROM invalid_rows").fetchone()[0] == 1
    assert db.execute("SELECT COUNT(*) FROM session_flags").fetchone()[0] >= 1
    if field not in ("timestamp", "contract"):
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
    db.close()


def test_out_of_order_timely_rows_never_become_eligible(collector):
    run, today, clock = collector
    order = [1, 0] + list(range(2, 390))
    today.loc[:1, "received_at"] = today.timestamp.iloc[1].isoformat()
    status, db, _ = run(today.iloc[order])
    assert (
        db.execute("SELECT COUNT(*) FROM observations WHERE available=1").fetchone()[0]
        == 390
    )
    assert status["eligible_sessions"] == 0
    assert db.execute(
        "SELECT 1 FROM session_flags WHERE reason='missing_prefix_at_first_observation'"
    ).fetchone()
    db.close()


def test_eod_grace_then_timely_completion_and_silent_day(collector):
    run, today, clock = collector
    status, db, _ = run(today.iloc[:-1])
    db.close()
    clock[0] = pd.Timestamp("2025-01-30T21:00:30Z")
    status, db, _ = run(today.iloc[:0])
    assert not db.execute("SELECT * FROM sessions").fetchall()
    db.close()
    today.loc[389, "received_at"] = "2025-01-30T21:00:45Z"
    status, db, _ = run(today.iloc[-1:])
    assert status["eligible_sessions"] == 1
    db.close()
    clock[0] = pd.Timestamp("2025-01-31T21:01:01Z")
    status, db, _ = run(today.iloc[:0])
    assert db.execute(
        "SELECT eligible,reason FROM sessions WHERE day='2025-01-31'"
    ).fetchone() == (0, "missing_late_or_exchange_closed_unverified")
    db.close()


def test_collection_stops_at_120_and_deadline(collector):
    run, today, clock = collector
    _, db, _ = run(today.iloc[:0])
    for i in range(120):
        db.execute(
            "INSERT INTO sessions VALUES (?,?,1,?)", (f"fixture-{i}", "MNQH25", "")
        )
    db.commit()
    db.close()
    status, db, _ = run(today.iloc[:1])
    assert status["eligible_sessions"] == 120
    assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
    db.close()


def test_deadline_no_new_observations(collector):
    run, today, clock = collector
    clock[0] = pd.Timestamp("2025-11-01T00:00:00Z")
    status, db, _ = run(today.iloc[:1])
    assert status["collection_status"] == "incomplete/inconclusive"
    assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
    db.close()


def test_future_receipt_rejected(collector, monkeypatch):
    run, today, clock = collector
    monkeypatch.setattr(shadow, "_now", lambda: pd.Timestamp("2025-01-30T14:31:00Z"))
    status, db, _ = run(today.iloc[:1])
    assert db.execute("SELECT available,reason FROM observations").fetchone() == (
        0,
        "late_backfilled_or_future_receipt",
    )
    db.close()


def test_old_rows_do_not_reconstruct_context(collector, monkeypatch):
    run, today, clock = collector
    monkeypatch.setattr(shadow, "_now", lambda: pd.Timestamp("2025-01-31T14:31:00Z"))
    monkeypatch.setattr(
        shadow, "context_for", lambda *a, **k: pytest.fail("old rows need no context")
    )
    status, db, _ = run(today)
    assert (
        db.execute("SELECT COUNT(*) FROM observations WHERE available=0").fetchone()[0]
        == 390
    )
    db.close()


def test_nanoseconds_rejected():
    first = session()
    second = session("2025-01-07")
    second.bars.loc[0, "timestamp"] += pd.Timedelta(nanoseconds=1)
    selected, errors = audit_select(pd.concat([first.bars, second.bars]))
    assert not selected and any("nonminute_timestamp" in e["exclusion"] for e in errors)
    raw = second.bars.iloc[0].to_dict()
    raw["timestamp"] = raw["timestamp"].isoformat()
    raw["received_at"] = "2025-01-07T14:31:01Z"
    with pytest.raises(ValueError, match="Nonminute"):
        shadow._normalize(raw, "end")


def test_market_fill_timestamp_and_stop_uncertainty():
    s = session()
    s.bars.loc[29, ["close", "high"]] = [10010, 10010]
    s.bars.loc[32, "low"] = 9700
    _, ledger, _ = simulate(s, np.zeros(390), ARMS[0], 2)
    assert ledger[0]["event_timestamp"] == s.bars.timestamp.iloc[31].isoformat()
    assert (
        ledger[0]["modeled_fill_timestamp"]
        == (s.bars.timestamp.iloc[31] - pd.Timedelta(minutes=1)).isoformat()
    )
    stop = next(row for row in ledger if row["reason"] == "CAT_STOP")
    assert (
        stop["modeled_fill_timestamp"] is None
        and stop["fill_time_basis"] == "intrabar_unknown_within_event_minute"
    )


def test_published_neutral_vwap_reversal_sequence_matches_reference():
    s = session(price=100.0)
    # Long, sampled equality/neutral, band breakout rejected by high HLC3 VWAP, short, reversal long.
    samples = [102.0, 100.0, 101.0, 98.0, 104.0]
    for i, c in zip([29, 59, 89, 119, 149], samples):
        s.bars.loc[i, ["close", "high", "low"]] = [c, max(100, c), min(100, c)]
    s.bars.loc[60:89, "high"] = 110.0
    ref = author_vectorized(s.bars, 100.0, np.zeros(390))
    _, _, decisions = simulate(s, np.zeros(390), ARMS[3], 1)
    assert [r["target"] for r in decisions[:5]] == [1, 0, 0, -1, 1]
    for i, dec in zip(range(29, 360, 30), decisions):
        assert dec["target"] == ref["signal"][i]
        assert ref["exposure"][i + 1] == dec["target"]
    assert ref["exposure"][59] == 1 and ref["exposure"][60] == 0


def test_deployed_mixed_sequence_and_double_catstop_guard():
    s = session()
    for i, c in zip([29, 59, 89, 119], [10010.0, 10000.0, 9990.0, 10020.0]):
        s.bars.loc[i, ["close", "high", "low"]] = [c, max(c, 10001), min(c, 9999)]
    _, _, marks = simulate(s, np.zeros(390), ARMS[0])
    p = 0
    realized = 0.0
    anchor = None
    for mark in marks:
        old = p
        p, realized, actions = deployed_event(
            mark["signal_price"], 10000, 10000, 0, p, realized, anchor=anchor
        )
        if p and p != old:
            anchor = mark["signal_price"]
        assert mark["target"] == p
    t = session()
    t.bars.loc[[29, 59, 89], ["close", "high"]] = 10010.0
    t.bars.loc[[32, 62], "low"] = 9760.0
    _, guarded, decisions = simulate(t, np.zeros(390), ARMS[0])
    stops = [r for r in guarded if r["reason"] == "CAT_STOP"]
    assert len(stops) == 2 and stops[-1]["reference_realized_gross"] == -1000
    assert decisions[2]["target"] == 0
    _, _, unguarded = simulate(t, np.zeros(390), ARMS[2])
    assert unguarded[2]["target"] == 1


@pytest.mark.parametrize("high_a,high_b", [(0.0, -1.0), (40.0, 30.0)])
def test_highest_cost_veto(high_a, high_b):
    records = []
    for i in range(120):
        for cost, a, b in [(2.24, 10.0, 30.0), (6.24, high_a, high_b)]:
            records.extend(
                [
                    {"day": str(i), "delay": 2, "cost": cost, "arm": "A", "net": a},
                    {"day": str(i), "delay": 2, "cost": cost, "arm": "B", "net": b},
                ]
            )
    result = decision(pd.DataFrame(records), True)
    assert (
        result["decision"] == "inconclusive"
        and result["highest_cost_positive"] is False
    )


def test_actual_sandbox_filesystem_and_environment(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    data = tmp_path / "data.csv"
    data.write_text("private research input")
    credential = tmp_path / "credential-canary"
    credential.write_text("must not be readable")
    probe = """
import json,os,socket
from pathlib import Path
from research.mim_comparison.shadow import install_socket_filter
install_socket_filter()
result={}
result['credential_absent']=not Path(CREDENTIAL).exists()
result['environment_absent']='MIM_TEST_SECRET' not in os.environ
result['adapter_absent']=not Path('/code/research/mim_comparison/evidence/mim_nb_live.py').exists()
for key,path in [('source','/code/research/mim_comparison/engine.py'),('input','/inputs/data.csv'),('production','/root/Silver-Bullet-ML-BMAD/data/mim_nb/canary')]:
 try:
  with open(path,'a') as f:f.write('forbidden')
  result[key+'_denied']=False
 except OSError:result[key+'_denied']=True
try:socket.socket();result['socket_denied']=False
except OSError:result['socket_denied']=True
Path('/output/allowed').write_text('allowed')
result['output_allowed']=True
Path('/output/probe.json').write_text(json.dumps(result))
""".replace("CREDENTIAL", repr(str(credential)))
    cmd = shadow.sandbox_command(
        output, data, data, state, [sys.executable, "-c", probe]
    )
    completed = subprocess.run(
        cmd,
        env=dict(os.environ, MIM_TEST_SECRET="credential-canary"),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert all(json.loads((output / "probe.json").read_text()).values())
    assert data.read_text() == "private research input"


def test_seccomp_rejects_compatibility_abi():
    code = """
import ctypes,mmap
from research.mim_comparison.shadow import install_socket_filter
region=mmap.mmap(-1,4096,prot=mmap.PROT_READ|mmap.PROT_WRITE|mmap.PROT_EXEC)
region.write(bytes.fromhex('b814000000cd80c3'))
call=ctypes.CFUNCTYPE(ctypes.c_int)(ctypes.addressof(ctypes.c_char.from_buffer(region)))
install_socket_filter()
assert call()==-1
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_author_sizing_first_daily_return_stays_missing():
    from research.mim_comparison.engine import sizing_quantity

    returns = [float("nan")] + np.linspace(-0.05, 0.05, 20).tolist()
    _, vol15, lev15 = sizing_quantity(100000, 10000, returns, 15)
    _, vol16, lev16 = sizing_quantity(100000, 10000, returns, 16)
    assert np.isnan(vol15) and lev15 == 4
    assert np.isfinite(vol16) and lev16 < 4


def test_static_warmup_audited_once(tmp_path, monkeypatch):
    warm = pd.concat(
        [
            session(str(d.date())).bars
            for d in pd.bdate_range("2025-01-01", "2025-01-29")
        ]
    )
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE observations(day TEXT,event TEXT,contract TEXT,payload TEXT,available INTEGER)"
    )
    original = shadow.audit_select
    calls = []

    def audit(frame):
        calls.append(len(frame))
        return original(frame)

    monkeypatch.setattr(shadow, "audit_select", audit)
    cache = {}
    shadow.context_for("2025-01-30", warm, db, cache)
    shadow.context_for("2025-01-31", warm, db, cache)
    assert calls == [len(warm)]


def test_historical_input_and_source_drift_fail_closed(tmp_path, monkeypatch):
    from research.mim_comparison import artifacts, historical

    monkeypatch.setattr(artifacts, "RUNS", tmp_path / "runs")
    data = tmp_path / "data.csv"
    pd.concat([session().bars, session("2025-01-07").bars]).to_csv(data, index=False)
    path, _ = artifacts.make_run("audit", [data], {"data": str(data), "labels": "end"})
    original = historical.audit_select

    def mutate_after_read(frame):
        result = original(frame)
        with data.open("a") as out:
            out.write("\n")
        return result

    monkeypatch.setattr(historical, "audit_select", mutate_after_read)
    with pytest.raises(ValueError, match="input changed"):
        historical.run(data, "end", path, True)
    other, _ = artifacts.make_run("audit", [data], {"data": str(data), "labels": "end"})
    digest_original = artifacts.digest
    monkeypatch.setattr(
        artifacts,
        "digest",
        lambda p: "modified" if Path(p) == BASE / "engine.py" else digest_original(p),
    )
    with pytest.raises(ValueError, match="Implementation changed"):
        artifacts.verify_run(other, data)


def test_historical_reports_every_scenario_and_paired_daily(tmp_path, monkeypatch):
    from research.mim_comparison import artifacts, historical

    monkeypatch.setattr(artifacts, "RUNS", tmp_path / "runs")
    data = tmp_path / "data.csv"
    pd.concat(
        [session(str(d.date())).bars for d in pd.bdate_range("2025-01-01", periods=18)]
    ).to_csv(data, index=False)
    path, _ = artifacts.make_run(
        "historical", [data], {"data": str(data), "labels": "end"}
    )
    historical.run(data, "end", path)
    report = json.loads((path / "report.json").read_text())
    assert len(report["scenario_contrasts"]) == 6
    paired = pd.read_csv(path / "paired-daily.csv")
    assert set(paired.delay) == {1, 2} and set(paired.cost) == {2.24, 3.24, 6.24}
    assert (paired.B_minus_A == paired.B - paired.A).all()


def test_shadow_stream_manifest_and_original_history_binding(tmp_path, monkeypatch):
    from research.mim_comparison import artifacts

    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(artifacts, "RUNS", root)
    monkeypatch.setattr(shadow, "RUNS", root)
    warm = tmp_path / "warm.csv"
    session().bars.to_csv(warm, index=False)
    data = tmp_path / "feed.csv"
    frame = session().bars.iloc[:1].copy()
    frame["received_at"] = frame.timestamp.map(
        lambda t: (t + pd.Timedelta(seconds=1)).isoformat()
    )
    frame.to_csv(data, index=False)

    def history():
        path, protocol = artifacts.make_run(
            "historical", [warm], {"data": str(warm), "labels": "end"}
        )
        artifacts.write_json(
            path / "report.json",
            {
                "mde": {
                    "sessions": 120,
                    "power": 0.8,
                    "two_sided_alpha": 0.05,
                    "minimum_detectable_increment_usd": 1.0,
                }
            },
        )
        artifacts.seal(path)
        return path, protocol

    initial, protocol = history()
    config = {
        "data": str(data),
        "warmup": str(warm),
        "labels": "end",
        "historical_run": str(initial),
        "state": None,
    }
    run, _ = artifacts.make_run("shadow", [data, warm], config)
    manifest = json.loads((run / "manifest.json").read_text())
    assert str(data.resolve()) not in manifest["inputs"]
    assert "journal" in manifest["stream_input"]["integrity_basis"]
    shadow.launch(
        run,
        data,
        warm,
        "end",
        dict(protocol, freeze="2026-10-01T00:00:00+00:00"),
        initial,
    )
    freeze = json.loads((run / "collector/freeze.json").read_text())
    assert freeze["protocol"]["freeze"] == protocol["freeze"]
    replacement, _ = history()
    next_run, _ = artifacts.make_run("shadow", [data, warm], config)
    with pytest.raises(ValueError, match="Original historical manifest binding"):
        shadow.launch(
            next_run, data, warm, "end", protocol, replacement, run / "collector"
        )
    # Invalid power/config cannot freeze a new experiment.
    (replacement / "report.json").chmod(0o644)
    (replacement / "report.json").write_text(json.dumps({"mde": {}}))
    with pytest.raises(ValueError, match="integrity"):
        shadow.launch(next_run, data, warm, "end", protocol, replacement)


def test_weekend_sessions_cannot_enter_sigma_or_block_monday():
    friday = session("2025-01-03")
    saturday = session("2025-01-04")
    monday = session("2025-01-06")
    selected, errors = audit_select(
        pd.concat([friday.bars, saturday.bars, monday.bars])
    )
    assert [s.day for s in selected] == ["2025-01-06"]
    assert any(
        e["day"] == "2025-01-04" and e["exclusion"] == "weekend_rth_closed"
        for e in errors
    )
    assert shadow._valid_complete(saturday.bars) is False
    raw = saturday.bars.iloc[0].to_dict()
    raw["timestamp"] = raw["timestamp"].isoformat()
    raw["received_at"] = "2025-01-04T14:31:01Z"
    with pytest.raises(ValueError, match="Weekend"):
        shadow._normalize(raw, "end")
