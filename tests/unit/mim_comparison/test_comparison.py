import json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from research.mim_comparison.data import Session, expiry, audit_select, load
from research.mim_comparison.engine import ARMS, simulate, sizing_quantity
from research.mim_comparison.references import (
    author_vectorized,
    author_sigma,
    deployed_event,
    log_reconciliation,
)
from research.mim_comparison.statistics import decision, stationary_means


def session(day="2025-01-06", contract="MNQH25", price=10000.0):
    times = pd.date_range(
        day + " 09:31", periods=390, freq="min", tz="America/New_York"
    )
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "day": day,
            "contract": contract,
            "minute": range(571, 961),
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price,
            "volume": 100.0,
        }
    )
    return Session(day, contract, frame, price)


def test_expiry_dst_and_labels(tmp_path):
    assert str(expiry("MNQH25")) == "2025-03-21"
    with pytest.raises(ValueError):
        expiry("MNQ_CONT")
    a = session("2025-03-07")
    b = session("2025-03-10")
    assert a.bars.timestamp.iloc[0].utcoffset() != b.bars.timestamp.iloc[0].utcoffset()
    f = a.bars.copy()
    f["timestamp"] -= pd.Timedelta(minutes=1)
    f.to_csv(tmp_path / "data.csv", index=False)
    assert load(tmp_path / "data.csv", "start").minute.tolist() == list(range(571, 961))
    f.drop(columns="contract").to_csv(tmp_path / "missing.csv", index=False)
    with pytest.raises(ValueError):
        load(tmp_path / "missing.csv", "start")


def test_selection_no_current_availability_lookahead_and_roll():
    a = session()
    b = session(contract="MNQM25")
    b.bars.volume = 200
    c = session("2025-01-07")
    d = session("2025-01-07", contract="MNQM25")
    selected, _ = audit_select(pd.concat([a.bars, b.bars, c.bars, d.bars]))
    assert selected[0].contract == "MNQM25"
    selected, excluded = audit_select(pd.concat([a.bars, b.bars, c.bars]))
    assert selected == []
    assert any(
        e["exclusion"] == "selected_contract_incomplete_no_fallback" for e in excluded
    )
    b.bars.volume = 100
    selected, _ = audit_select(pd.concat([a.bars, b.bars, c.bars, d.bars]))
    assert selected[0].contract == "MNQH25"


def test_bad_sessions_excluded():
    a = session()
    b = session("2025-01-07")
    b.bars = pd.concat([b.bars.iloc[:-1], b.bars.iloc[:1]])
    selected, errors = audit_select(pd.concat([a.bars, b.bars]))
    assert not selected
    assert any("duplicate" in e["exclusion"] for e in errors)


def test_author_neutral_zero_volume_and_sigma_no_lookahead():
    s = session()
    z = np.zeros(390)
    ref = author_vectorized(s.bars, s.previous_close, z)
    assert not ref["signal"].any()
    s.bars.volume = 0
    assert np.isnan(author_vectorized(s.bars, s.previous_close, z)["vwap"]).all()
    moves = np.arange(20 * 390).reshape(20, 390) / 10000
    first = author_sigma(moves)
    moves[15:] = 900
    np.testing.assert_array_equal(first[:16], author_sigma(moves)[:16])
    assert np.isnan(first[13]).all() and np.isfinite(first[14]).all()


def test_execution_delay_gap_stop_anchor_and_determinism():
    s = session()
    s.bars.loc[29, "close"] = 10010
    s.bars.loc[29, "high"] = 10010
    s.bars.loc[30, ["open", "high", "low", "close"]] = [10020, 10021, 10019, 10020]
    s.bars.loc[31, ["open", "high", "low", "close"]] = [9700, 9701, 9699, 9700]
    row, ledger, _ = simulate(s, np.zeros(390), ARMS[0], 2)
    assert ledger[0]["fill"] == 9700 and ledger[0]["signal_price"] == 10010
    assert ledger[1]["reason"] == "CAT_STOP" and ledger[1]["fill"] == 9700
    assert ledger[1]["reference_realized_gross"] == -500
    assert row["gross"] == 0 and row["net"] == -2.24
    assert (row, ledger) == simulate(s, np.zeros(390), ARMS[0], 2)[:2]
    _, optimistic, _ = simulate(s, np.zeros(390), ARMS[0], 1)
    assert optimistic[0]["fill"] == 10020 and optimistic[1]["fill"] == 9700


def test_reference_guard_equality_rejection_and_unknown():
    assert deployed_event(100, 100, 100, 0, 0, 0)[0] == 0
    p, g, a = deployed_event(
        100, 100, 100, 0, 1, -500, anchor=350, broker="stop_filled"
    )
    assert p == 0 and g == -1000
    assert deployed_event(110, 100, 100, 0, 0, 0, broker="entry_rejected")[0] == 0
    assert deployed_event(110, 100, 100, 0, 0, 0, broker="stop_rejected")[0] == 0
    assert deployed_event(100, 100, 100, 0, 1, 0, anchor=100, broker="unknown")[0] == 1
    assert deployed_event(105, 100, 100, 0, 1, 0, anchor=100, eod=True)[1] == 10


def test_deployed_reference_engine_target_agreement():
    s = session()
    s.bars.loc[29, "close"] = 10010
    s.bars.loc[29, "high"] = 10010
    _, _, d = simulate(s, np.zeros(390), ARMS[0])
    target, _, _ = deployed_event(10010, 10000, 10000, 0, 0, 0)
    assert d[0]["target"] == target == 1


def test_sizing_lag_bankers_round():
    returns = np.linspace(-0.03, 0.03, 30).tolist()
    a = sizing_quantity(100000, 20000, returns, 16)
    returns[15:] = [99] * 15
    assert a == sizing_quantity(100000, 20000, returns, 16)
    assert sizing_quantity(100000, 20000, [], 0, False)[0] == 2
    assert sizing_quantity(100000, 20000, [], 0, True)[0] == 10


def test_log_precision_and_bootstrap():
    r = log_reconciliation(
        pd.DataFrame(
            [
                {
                    "ts_et": "x",
                    "close": "100.00",
                    "ub": "100.00",
                    "lb": "90",
                    "sigma": ".001",
                    "open_d": "100",
                }
            ]
        )
    )
    assert r[0]["classification"] == "rounded_threshold_ambiguity"
    np.testing.assert_array_equal(
        stationary_means([1, 2, 3], 5, 100), stationary_means([1, 2, 3], 5, 100)
    )
    assert decision(pd.DataFrame())["decision"] == "incomplete/inconclusive"


def test_final_frozen_rule():
    rows = []
    for i in range(120):
        for cost in (2.24, 6.24):
            for arm, value in [("A", 10.0), ("B", 30.0)]:
                rows.append(
                    {"day": str(i), "delay": 2, "cost": cost, "arm": arm, "net": value}
                )
    assert (
        decision(pd.DataFrame(rows), True)["decision"] == "supports_further_validation"
    )


def test_seccomp_denies_socket_in_child():
    code = "from research.mim_comparison.shadow import install_socket_filter; import socket; install_socket_filter(); socket.socket()"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode != 0 and "PermissionError" in result.stderr


def test_actual_ast_reference_fixtures():
    import asyncio
    from research.mim_comparison.evidence.deployed_fixture_audit import run

    for case in asyncio.run(run()):
        broker = (
            "exit_rejected"
            if case.get("reject_exits_only")
            else (
                "stop_filled"
                if case["stop_filled"]
                else "entry_rejected" if case["rejected"] else "normal"
            )
        )
        p, g, _ = deployed_event(
            case["close"],
            100.0,
            100.0,
            0.01,
            case["initial_position"],
            case["initial_realized"],
            anchor=case["initial_anchor"],
            eod=case["eod"],
            broker=broker,
        )
        assert p == case["position"]
        assert g == pytest.approx(case["realized"])


def test_streaming_durable_replay_and_corrections(tmp_path, monkeypatch):
    from research.mim_comparison import shadow

    days = pd.bdate_range("2025-01-01", "2025-01-29")
    warm = pd.concat([session(str(d.date())).bars for d in days])
    warm.to_csv(tmp_path / "warmup.csv", index=False)
    today = session("2025-01-30").bars
    today.loc[29, "close"] = 10010.0
    today.loc[29, "high"] = 10010.0
    today["received_at"] = (today.timestamp + pd.Timedelta(seconds=1)).map(
        lambda x: x.isoformat()
    )
    today.to_csv(tmp_path / "today.csv", index=False)
    state = tmp_path / "state"
    state.mkdir()
    first = tmp_path / "one"
    first.mkdir()
    from research.mim_comparison.artifacts import BASE, digest

    freeze = {
        "protocol": {"freeze": "2025-01-29T22:00:00+00:00"},
        "warmup_hash": digest(tmp_path / "warmup.csv"),
        "labels": "end",
        "source": {p.name: digest(p) for p in BASE.glob("*.py")},
    }
    (state / "freeze.json").write_text(json.dumps(freeze))
    clock = [pd.Timestamp("2025-01-30T21:00:01Z")]
    original = shadow._normalize

    def normalize(raw, labels):
        result = original(raw, labels)
        clock[0] = pd.Timestamp(result["timestamp"]) + pd.Timedelta(seconds=1)
        return result

    monkeypatch.setattr(shadow, "_normalize", normalize)
    monkeypatch.setattr(shadow, "_now", lambda: clock[0])
    monkeypatch.setattr(shadow, "install_socket_filter", lambda: None)
    cfg = {
        "state": str(state),
        "warmup": str(tmp_path / "warmup.csv"),
        "data": str(tmp_path / "today.csv"),
        "labels": "end",
        "result": str(first / "status.json"),
    }
    shadow.collect(cfg)
    status = json.loads((first / "status.json").read_text())
    assert status["eligible_sessions"] == 1
    import sqlite3

    db = sqlite3.connect(state / "journal.sqlite")
    before = db.execute("SELECT * FROM decisions ORDER BY event,arm").fetchall()
    assert len(before) == 24
    today.loc[29, "close"] = 10020.0
    today.loc[29, "high"] = 10020.0
    today.to_csv(tmp_path / "today.csv", index=False)
    second = tmp_path / "two"
    second.mkdir()
    cfg["result"] = str(second / "status.json")
    shadow.collect(cfg)
    assert json.loads((second / "status.json").read_text())["corrections_ignored"] == 1
    assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 390
    assert db.execute("SELECT * FROM decisions ORDER BY event,arm").fetchall() == before
    db.close()


def test_120_session_authenticated_evaluate_and_tamper(tmp_path):
    import sqlite3
    from research.mim_comparison import shadow
    from research.mim_comparison.artifacts import BASE, digest, seal, write_json

    warm = pd.concat(
        [
            session(str(d.date()), contract="MNQZ25").bars
            for d in pd.bdate_range("2025-01-01", "2025-01-29")
        ]
    )
    warmup = tmp_path / "warm.csv"
    warm.to_csv(warmup, index=False)
    source = tmp_path / "source-run"
    source.mkdir()
    out = tmp_path / "result"
    out.mkdir()
    db = sqlite3.connect(source / "journal-snapshot.sqlite")
    db.execute(
        "CREATE TABLE observations(event TEXT,contract TEXT,day TEXT,payload TEXT,collected TEXT,available INTEGER,reason TEXT)"
    )
    db.execute("CREATE TABLE decisions(event TEXT,arm TEXT,payload TEXT)")
    db.execute(
        "CREATE TABLE sessions(day TEXT,contract TEXT,eligible INTEGER,reason TEXT)"
    )
    for d in pd.bdate_range("2025-01-30", periods=120):
        day = str(d.date())
        s = session(day, contract="MNQZ25")
        s.bars["received_at"] = (s.bars.timestamp + pd.Timedelta(seconds=1)).map(
            lambda t: t.isoformat()
        )
        db.execute("INSERT INTO sessions VALUES (?,?,1,?)", (day, s.contract, ""))
        for row in s.bars.to_dict("records"):
            row["timestamp"] = row["timestamp"].isoformat()
            db.execute(
                "INSERT INTO observations VALUES (?,?,?,?,?,1,?)",
                (
                    row["timestamp"],
                    s.contract,
                    day,
                    json.dumps(row),
                    row["received_at"],
                    "",
                ),
            )
        for arm in ARMS[:2]:
            _, _, decisions = simulate(s, np.zeros(390), arm, 2, 2.24)
            for dec in decisions:
                db.execute(
                    "INSERT INTO decisions VALUES (?,?,?)",
                    (dec["event_timestamp"], arm.name, json.dumps(dec)),
                )
    db.commit()
    db.close()
    freeze = {
        "protocol": {"freeze": "2025-01-29T22:00:00+00:00"},
        "warmup_hash": digest(warmup),
        "labels": "end",
        "source": {p.name: digest(p) for p in BASE.glob("*.py")},
    }
    write_json(source / "freeze-snapshot.json", freeze)
    write_json(
        source / "manifest.json",
        {"command": "shadow", "config": {"warmup": str(warmup)}},
    )
    write_json(
        source / "status.json",
        {
            "eligible_sessions": 120,
            "freeze_hash": digest(source / "freeze-snapshot.json"),
            "journal_hash": digest(source / "journal-snapshot.sqlite"),
        },
    )
    seal(source)
    shadow.evaluate(source, out)
    result = json.loads((out / "decision.json").read_text())
    assert result["decision"] == "failure" and result["eligible_sessions"] == 120
    (source / "status.json").chmod(0o644)
    (source / "status.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        shadow.evaluate(source, tmp_path / "tampered")
