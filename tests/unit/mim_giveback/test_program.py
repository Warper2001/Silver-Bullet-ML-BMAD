from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.mim_comparison.data import Session
from research.mim_robustness.engine import Arm
from research.mim_robustness.features import features
from research.mim_robustness.study import trade_ledger, reconcile_accounting
from research.mim_giveback import artifacts
from research.mim_giveback.__main__ import _resolve_invocation, main
from research.mim_giveback.config import CONFIG, DEFAULT_DATA, DEFAULT_SOURCE_RUN
from research.mim_giveback.data import load_source, threshold_family
from research.mim_giveback.engine import simulate
from research.mim_giveback.prospective import (
    _read_future,
    collect,
    evaluate,
    protocol_committed,
)
from research.mim_giveback.statistics import power_from_null
from research.mim_giveback.study import _donor_trigger_sets, _shift_mapping, run_power
from research.mim_giveback import workflow


def session(prices=None, lows=None, day="2025-01-06"):
    close = np.full(390, 10000.0) if prices is None else np.asarray(prices, dtype=float)
    timestamp = pd.date_range(
        day + " 09:31", periods=390, freq="min", tz="America/New_York"
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "day": day,
            "contract": "MNQH25",
            "minute": np.arange(571, 961),
            "open": close.copy(),
            "high": close + 1,
            "low": close - 1 if lows is None else np.asarray(lows, dtype=float),
            "close": close,
            "volume": 100.0,
        }
    )
    return Session(day, "MNQH25", frame, 10000.0)


def trending_session(stop_before_exit=False, reversal=False):
    close = np.full(390, 10000.0)
    close[29:59] = np.linspace(10020, 10060, 30)
    close[59:] = 10030
    if reversal:
        close[59:] = 9970
    lows = close - 1
    if stop_before_exit:
        close[60] = 9760
        lows[60] = 9750
    result = session(close, lows)
    # Entry fill uses row 31 open; hold it at the signal price.
    result.bars.loc[31, "open"] = 10020
    result.bars["high"] = np.maximum(
        result.bars.high, result.bars[["open", "close"]].max(axis=1) + 1
    )
    result.bars["low"] = np.minimum(
        result.bars.low, result.bars[["open", "close"]].min(axis=1) - 1
    )
    return result


def run_fixture(value, **kwargs):
    sigma = np.full(390, 0.001)
    return simulate(
        value,
        sigma,
        Arm("A"),
        feature_values=features(value, sigma),
        **kwargs,
    )


def test_identity_pairing_is_refused_before_statistic():
    with pytest.raises(AssertionError, match="FIREWALL VIOLATION"):
        _shift_mapping(["a", "b"], {}, 0)
    with pytest.raises(AssertionError, match="FIREWALL VIOLATION"):
        _shift_mapping(["a", "b"], {}, 2)


def test_run_power_transfers_nonidentity_donor_triggers(monkeypatch):
    days = [f"2025-01-{day:02d}" for day in range(1, 8)]
    baseline = pd.DataFrame({"day": days, "net": np.zeros(len(days))})
    trades = pd.DataFrame({"net": [2.0, -1.0], "exit_reason": ["EOD", "STOP"]})
    monkeypatch.setattr(
        "research.mim_giveback.study.reconcile_baseline",
        lambda *_: (baseline, pd.DataFrame(), pd.DataFrame(), trades),
    )
    observed = []

    def fake_simulate(_prepared, external=None, **_kwargs):
        observed.append(external)
        daily = pd.DataFrame(
            {
                "day": days,
                "net": [float(sum(external[day])) for day in days],
            }
        )
        return daily, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr("research.mim_giveback.study.simulate_sessions", fake_simulate)
    monkeypatch.setattr(
        "research.mim_giveback.study.power_from_null",
        lambda null, effect: (pd.DataFrame(), {"verdict": "UNDERPOWERED"}),
    )
    marks = pd.DataFrame(
        {
            "day": days,
            "mfe": 1.0,
            "giveback": 1.0,
            "mark_timestamp": [
                f"{day}T10:{i:02d}:00-05:00" for i, day in enumerate(days)
            ],
        }
    )
    result = run_power(
        [object()] * len(days),
        {"daily": baseline, "marks": marks},
        pd.DataFrame({"threshold": [0.5]}),
    )
    donor = _donor_trigger_sets(marks, 0.5)
    assert observed[0] == _shift_mapping(days, donor, CONFIG["null_shifts"][0])
    assert len(result["null"]) == len(days) * len(CONFIG["null_shifts"])
    assert result["null"].delta.ne(0).all()


def test_giveback_exit_uses_delayed_open_and_costs_once():
    result, events, decisions = run_fixture(trending_session(), giveback_threshold=0.25)
    exits = [row for row in events if row["reason"] == "GIVEBACK_EXIT"]
    assert exits and exits[0]["event_timestamp"].endswith("10:32:00-05:00")
    assert exits[0]["modeled_fill_timestamp"].endswith("10:31:00-05:00")
    ledger = pd.DataFrame([dict(row, delay=2) for row in events])
    decision_frame = pd.DataFrame([dict(row, delay=2) for row in decisions])
    trades = trade_ledger(ledger, decision_frame)
    daily = pd.DataFrame([result])
    reconcile_accounting(daily, ledger, trades)
    assert trades.loc[trades.exit_reason.eq("GIVEBACK_EXIT"), "costs"].eq(2.24).all()


def test_stop_before_pending_overlay_exit_wins_and_gap_is_preserved():
    _, events, _ = run_fixture(
        trending_session(stop_before_exit=True), giveback_threshold=0.25
    )
    reasons = [row["reason"] for row in events]
    assert "CAT_STOP" in reasons
    assert reasons[:2] == ["ENTRY", "CAT_STOP"]
    assert not any(
        row["reason"] == "GIVEBACK_EXIT"
        and row["event_timestamp"].endswith("10:32:00-05:00")
        for row in events
    )
    stop = next(row for row in events if row["reason"] == "CAT_STOP")
    assert stop["fill"] == 9760
    assert stop["modeled_fill_timestamp"] is None


def test_reversal_decision_takes_priority_over_overlay():
    _, events, decisions = run_fixture(
        trending_session(reversal=True), giveback_threshold=0.01
    )
    mark = next(
        row for row in decisions if row["event_timestamp"].endswith("10:30:00-05:00")
    )
    assert mark["reason"] == "REVERSAL"
    assert not mark["giveback_trigger"]
    assert any(row["reason"] == "REVERSAL" for row in events)


def test_external_trigger_is_ignored_while_flat():
    flat = session()
    result, events, _ = run_fixture(flat, external_trigger_minutes={600, 630, 660})
    assert not events
    assert result["net"] == 0


def test_threshold_family_is_mechanical_and_unique():
    marks = pd.DataFrame({"mfe": np.ones(100), "giveback": np.arange(100, dtype=float)})
    family = threshold_family(marks)
    assert family["quantile"].tolist() == CONFIG["threshold_quantiles"]
    assert family.threshold.is_monotonic_increasing


def test_power_verdict_never_claims_aligned_returns(monkeypatch):
    monkeypatch.setitem(CONFIG, "bootstrap_draws", 20)
    monkeypatch.setitem(CONFIG, "endpoint_sessions", 20)
    rows = []
    for threshold in range(9):
        for shift in CONFIG["null_shifts"]:
            for index in range(60):
                rows.append(
                    {
                        "threshold": float(threshold),
                        "shift": shift,
                        "session_index": index,
                        "delta": float((index % 3) - 1),
                    }
                )
    table, verdict = power_from_null(pd.DataFrame(rows), minimum_effect=0.01)
    repeated, repeated_verdict = power_from_null(
        pd.DataFrame(rows), minimum_effect=0.01
    )
    pd.testing.assert_frame_equal(table, repeated)
    assert verdict == repeated_verdict
    assert verdict["identity_pairings_evaluated"] == 0
    assert verdict["aligned_candidate_returns_evaluated"] is False


def test_nonpowered_parent_denies_inventory(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "power_verdict.json").write_text('{"verdict":"UNDERPOWERED"}')
    monkeypatch.setattr(
        artifacts, "verify_run", lambda _: {"command": "power", "config": CONFIG}
    )
    with pytest.raises(ValueError, match="terminal"):
        workflow.inventory(tmp_path / "output", parent)


def test_powered_inventory_preserves_all_nine_candidates(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    output = tmp_path / "output"
    parent.mkdir()
    output.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    (source / "completion.json").write_text("source")
    data = tmp_path / "bars.csv"
    data.write_text("bars")
    (parent / "completion.json").write_text("power")
    (parent / "power_verdict.json").write_text('{"verdict":"POWERED"}')
    pd.DataFrame(
        {"quantile": CONFIG["threshold_quantiles"], "threshold": range(9)}
    ).to_csv(parent / "candidate_family.csv", index=False)
    monkeypatch.setattr(
        artifacts,
        "verify_run",
        lambda _: {
            "command": "power",
            "config": CONFIG,
            "bindings": {"source_run": str(source), "data": str(data)},
        },
    )
    workflow.inventory(output, parent)
    family = pd.read_csv(output / "threshold_inventory.csv")
    assert len(family) == 9
    assert not family.candidate_return_calculated.any()


def test_sweep_rejects_source_or_data_not_bound_by_power(tmp_path, monkeypatch):
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    (inventory / "inventory_status.json").write_text(
        json.dumps({"source_completion_hash": "expected", "data_hash": "expected"})
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "completion.json").write_text("different")
    data = tmp_path / "data.csv"
    data.write_text("different")
    monkeypatch.setattr(
        artifacts, "verify_run", lambda _: {"command": "inventory", "config": CONFIG}
    )
    with pytest.raises(ValueError, match="differs from powered"):
        workflow.sweep(tmp_path / "out", inventory, source, data)


def test_source_run_and_baseline_are_manifest_bound():
    source = load_source(DEFAULT_SOURCE_RUN, DEFAULT_DATA)
    assert len(source["marks"]) == 6673
    assert source["summary"]["net_profit"] == pytest.approx(21889.76)


def test_artifact_refuses_unsafe_path_and_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "RUNS", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    with pytest.raises(ValueError, match="direct"):
        artifacts.create("power", [], tmp_path / "outside")
    output = tmp_path / "runs" / "fixed"
    path = artifacts.create("power", [], output)
    with pytest.raises(FileExistsError):
        artifacts.create("power", [], output)
    artifacts.json_write(path / "result.json", {"ok": True})
    artifacts.seal(path)
    assert artifacts.verify_run(path)["command"] == "power"
    assert all(not (item.stat().st_mode & 0o222) for item in [path, *path.rglob("*")])
    (path / "result.json").chmod(0o644)
    with pytest.raises(ValueError, match="writable"):
        artifacts.verify_run(path)


def test_artifact_rejects_input_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "RUNS", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    source = tmp_path / "input.csv"
    source.write_text("first")
    path = artifacts.create("power", [source], tmp_path / "runs" / "run")
    artifacts.json_write(path / "result.json", {"ok": True})
    artifacts.seal(path)
    source.write_text("changed")
    with pytest.raises(ValueError, match="input drift"):
        artifacts.verify_run(path)


def test_artifact_rejects_input_drift_before_successful_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "RUNS", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    source = tmp_path / "input.csv"
    source.write_text("first")
    path = artifacts.create("power", [source], tmp_path / "runs" / "run")
    source.write_text("changed")
    with pytest.raises(ValueError, match="before seal"):
        artifacts.seal(path)


def test_repository_input_bindings_are_worktree_portable(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    source = root / "data" / "source.csv"
    source.parent.mkdir()
    source.write_text("bound")
    monkeypatch.setattr(artifacts, "ROOT", root)
    assert artifacts.portable_path(source) == "repo:data/source.csv"
    assert artifacts.resolve_path("repo:data/source.csv") == source


def test_evaluate_protocol_is_inferred_from_collection_manifest(tmp_path):
    collection = tmp_path / "collection"
    collection.mkdir()
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    history = tmp_path / "history.csv"
    history.write_text("history")
    (protocol / "protocol.json").write_text(json.dumps({"history_data": str(history)}))
    (protocol / "completion.json").write_text("protocol")
    (collection / "completion.json").write_text("collection")
    (collection / "manifest.json").write_text(
        json.dumps({"bindings": {"protocol": str(protocol)}})
    )
    args = type(
        "Args",
        (),
        {"command": "evaluate", "run": collection, "protocol": None},
    )()
    inputs, bindings = _resolve_invocation(args)
    assert inputs == [
        collection / "completion.json",
        protocol / "completion.json",
        history,
    ]
    assert bindings["protocol"] == protocol


def test_corrupt_invocation_resolution_seals_failure_evidence(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    collection = tmp_path / "collection"
    collection.mkdir()
    (collection / "manifest.json").write_text("not-json")
    output = runs / "failed"
    monkeypatch.setattr(artifacts, "RUNS", runs)
    assert main(["evaluate", "--run", str(collection), "--output", str(output)]) == 2
    assert (output / "failure.json").exists()
    assert (output / "completion.json").exists()
    assert not (output / "final_verdict.json").exists()


def test_failed_invocation_removes_partial_favorable_outputs(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(artifacts, "RUNS", runs)
    path = artifacts.create("evaluate", [], runs / "failed")
    (path / "final_verdict.json").write_text('{"verdict":"SUPPORT"}')
    artifacts.fail(path, ValueError("broken evaluation"))
    assert not (path / "final_verdict.json").exists()
    assert (path / "failure.json").exists()
    assert artifacts.verify_run(path)["command"] == "evaluate"


def frozen_rules(endpoint=500):
    return {
        key: (endpoint if key == "endpoint_sessions" else CONFIG[key])
        for key in (
            "delay",
            "round_trip_cost",
            "quantity",
            "target_pf",
            "profit_retention",
            "endpoint_sessions",
            "bootstrap_blocks",
            "bootstrap_draws",
            "seed",
            "alpha",
        )
    }


def protocol_payload(history: Path, endpoint=500):
    return {
        "frozen_at": "2026-09-13T00:00:00+00:00",
        "deadline": "2029-03-13T00:00:00+00:00",
        "selected_threshold": 0.5,
        "history_data": str(history),
        "history_data_hash": "hh",
        "rules": frozen_rules(endpoint),
    }


def future_csv(path: Path, duplicate=False, start="2026-09-14 09:31"):
    timestamps = pd.date_range(start, periods=2, freq="min", tz="America/New_York")
    frame = pd.DataFrame(
        {
            "contract": "MNQZ26",
            "timestamp": timestamps.astype(str),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
            "received_at": (
                timestamps.tz_convert("UTC") + pd.Timedelta(seconds=30)
            ).astype(str),
        }
    )
    if duplicate:
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame.to_csv(path, index=False)


def test_future_input_requires_unique_causal_receipts(tmp_path):
    path = tmp_path / "future.csv"
    future_csv(path)
    as_of = datetime(2026, 9, 15, tzinfo=timezone.utc)
    assert _read_future(path, as_of=as_of).timely.all()
    future_csv(path, duplicate=True)
    with pytest.raises(ValueError, match="Duplicate"):
        _read_future(path, as_of=as_of)
    future_csv(path)
    invalid = pd.read_csv(path)
    invalid.loc[0, "high"] = 98.0
    invalid.to_csv(path, index=False)
    with pytest.raises(ValueError, match="OHLC ordering"):
        _read_future(path, as_of=as_of)


def test_collection_records_coverage_without_efficacy(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    history = tmp_path / "history.csv"
    history.write_text("history")
    (protocol / "protocol.json").write_text(json.dumps(protocol_payload(history)))
    (protocol / "completion.json").write_text("protocol")
    data = tmp_path / "future.csv"
    future_csv(data)
    monkeypatch.setattr(
        "research.mim_giveback.prospective.verify_run",
        lambda _: {"command": "freeze"},
    )
    monkeypatch.setattr(
        "research.mim_giveback.prospective.protocol_committed",
        lambda _: ("abc123", pd.Timestamp("2026-09-13T01:00:00Z")),
    )
    monkeypatch.setattr(
        "research.mim_giveback.prospective.digest",
        lambda path: ("hh" if Path(path) == history else "protocol-hash"),
    )
    monkeypatch.setattr(
        "research.mim_giveback.prospective.eligibility",
        lambda *_: pd.DataFrame(
            [
                {
                    "day": "2026-09-14",
                    "contract": "MNQZ26",
                    "eligible": False,
                    "exclusion": "incomplete",
                }
            ]
        ),
    )
    observations, eligible, status = collect(
        protocol, data, as_of=datetime(2026, 9, 15, tzinfo=timezone.utc)
    )
    assert len(observations) == 2 and len(eligible) == 1
    assert status["protocol_git_commit"] == "abc123"
    assert status["efficacy_calculated"] is False
    assert status["interim_pf_calculated"] is False
    future_csv(data, start="2026-09-12 09:31")
    with pytest.raises(ValueError, match="pre-freeze"):
        collect(
            protocol,
            data,
            as_of=datetime(2026, 9, 15, tzinfo=timezone.utc),
        )


def test_collection_chain_is_strict_append_and_protocol_bound(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol"
    prior = tmp_path / "prior"
    protocol.mkdir()
    prior.mkdir()
    history = tmp_path / "history.csv"
    history.write_text("history")
    (protocol / "protocol.json").write_text(json.dumps(protocol_payload(history)))
    (protocol / "completion.json").write_text("protocol")
    (prior / "completion.json").write_text("prior")
    prior_data = tmp_path / "prior.csv"
    future_csv(prior_data)
    prior_frame = pd.read_csv(prior_data)
    prior_frame["timely"] = True
    prior_frame.to_csv(prior / "observations.csv", index=False)
    (prior / "collection_status.json").write_text(
        json.dumps(
            {
                "protocol_completion_hash": "protocol-hash",
                "eligible_sessions": 0,
            }
        )
    )
    monkeypatch.setattr(
        "research.mim_giveback.prospective.verify_run",
        lambda path: {"command": "freeze" if path == protocol else "collect"},
    )
    monkeypatch.setattr(
        "research.mim_giveback.prospective.protocol_committed",
        lambda _: ("abc123", pd.Timestamp("2026-09-13T01:00:00Z")),
    )
    monkeypatch.setattr(
        "research.mim_giveback.prospective.digest",
        lambda path: (
            "hh"
            if Path(path) == history
            else (
                "prior-hash"
                if Path(path) == prior / "completion.json"
                else "protocol-hash"
            )
        ),
    )
    monkeypatch.setattr(
        "research.mim_giveback.prospective.eligibility",
        lambda *_: pd.DataFrame(
            [
                {
                    "day": "2026-09-15",
                    "contract": None,
                    "eligible": False,
                    "exclusion": "x",
                }
            ]
        ),
    )
    incoming = tmp_path / "incoming.csv"
    future_csv(incoming, start="2026-09-15 09:31")
    chain, _, status = collect(
        protocol,
        incoming,
        prior,
        as_of=datetime(2026, 9, 16, tzinfo=timezone.utc),
    )
    assert len(chain) == 4
    assert status["prior_collection_hash"] == "prior-hash"

    future_csv(incoming)
    with pytest.raises(ValueError, match="strict append"):
        collect(
            protocol,
            incoming,
            prior,
            as_of=datetime(2026, 9, 16, tzinfo=timezone.utc),
        )

    prior_status = json.loads((prior / "collection_status.json").read_text())
    prior_status["protocol_completion_hash"] = "different"
    (prior / "collection_status.json").write_text(json.dumps(prior_status))
    future_csv(incoming, start="2026-09-15 09:31")
    with pytest.raises(ValueError, match="protocol mismatch"):
        collect(
            protocol,
            incoming,
            prior,
            as_of=datetime(2026, 9, 16, tzinfo=timezone.utc),
        )


def test_future_receipts_fail_closed_for_late_preboundary_and_equivalent_duplicates(
    tmp_path,
):
    path = tmp_path / "future.csv"
    future_csv(path)
    frame = pd.read_csv(path)
    frame.loc[0, "received_at"] = "2026-09-14T13:33:00+00:00"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="execution deadline"):
        _read_future(path, as_of=datetime(2026, 9, 15, tzinfo=timezone.utc))

    future_csv(path)
    frame = pd.read_csv(path)
    duplicate = frame.iloc[[0]].copy()
    duplicate["timestamp"] = "2026-09-14T13:31:00+00:00"
    frame = pd.concat([frame, duplicate], ignore_index=True)
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        _read_future(path, as_of=datetime(2026, 9, 15, tzinfo=timezone.utc))


def test_protocol_commit_requires_current_completion_bytes(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    (protocol / "completion.json").write_bytes(b"current")
    monkeypatch.setattr("research.mim_giveback.prospective.ROOT", tmp_path)

    def fake_run(command, **_kwargs):
        if "log" in command:
            return type("Result", (), {"stdout": "abc123\n", "returncode": 0})()
        if "show" in command and "-s" in command:
            return type(
                "Result",
                (),
                {"stdout": "2026-09-13T01:00:00+00:00\n", "returncode": 0},
            )()
        if "show" in command:
            return type("Result", (), {"stdout": b"committed", "returncode": 0})()
        return type("Result", (), {"stdout": "", "returncode": 0})()

    monkeypatch.setattr("research.mim_giveback.prospective.subprocess.run", fake_run)
    with pytest.raises(ValueError, match="differs from committed"):
        protocol_committed(protocol)


def test_early_evaluation_emits_no_efficacy(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol"
    collection = tmp_path / "collection"
    protocol.mkdir()
    collection.mkdir()
    history = tmp_path / "history.csv"
    history.write_text("history")
    (protocol / "protocol.json").write_text(json.dumps(protocol_payload(history)))
    (protocol / "completion.json").write_text("p")
    (collection / "completion.json").write_text("c")
    (collection / "collection_status.json").write_text(
        json.dumps({"protocol_completion_hash": "ph", "history_data_hash": "hh"})
    )
    pd.DataFrame(
        columns=[
            "contract",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "received_at",
            "timely",
        ]
    ).to_csv(collection / "observations.csv", index=False)
    pd.DataFrame(
        [
            {
                "day": "2026-09-14",
                "contract": None,
                "eligible": False,
                "exclusion": "missing",
            }
        ]
    ).to_csv(collection / "eligibility.csv", index=False)
    monkeypatch.setattr(
        "research.mim_giveback.prospective.verify_run",
        lambda path: {"command": "freeze" if path == protocol else "collect"},
    )
    monkeypatch.setattr(
        "research.mim_giveback.prospective.digest",
        lambda path: "hh" if Path(path) == history else "ph",
    )
    result, payload = evaluate(
        protocol, collection, as_of=datetime(2026, 9, 15, tzinfo=timezone.utc)
    )
    assert result == {
        "verdict": "COLLECTING",
        "eligible_sessions": 0,
        "efficacy_calculated": False,
    }
    assert payload is None


def test_endpoint_after_deadline_is_inconclusive_without_efficacy(
    tmp_path, monkeypatch
):
    protocol = tmp_path / "protocol"
    collection = tmp_path / "collection"
    protocol.mkdir()
    collection.mkdir()
    history = tmp_path / "history.csv"
    history.write_text("history")
    frozen = protocol_payload(history)
    frozen["deadline"] = "2026-09-20T00:00:00+00:00"
    (protocol / "protocol.json").write_text(json.dumps(frozen))
    (protocol / "completion.json").write_text("p")
    (collection / "completion.json").write_text("c")
    (collection / "collection_status.json").write_text(
        json.dumps({"protocol_completion_hash": "ph", "history_data_hash": "hh"})
    )
    days = pd.date_range("2026-09-14", periods=500, freq="D")
    pd.DataFrame(
        {
            "timestamp": (days + pd.Timedelta(hours=20)).astype(str),
            "contract": "MNQZ26",
        }
    ).to_csv(collection / "observations.csv", index=False)
    pd.DataFrame(
        {
            "day": days.strftime("%Y-%m-%d"),
            "contract": "MNQZ26",
            "eligible": True,
            "exclusion": None,
        }
    ).to_csv(collection / "eligibility.csv", index=False)
    monkeypatch.setattr(
        "research.mim_giveback.prospective.verify_run",
        lambda path: {"command": "freeze" if path == protocol else "collect"},
    )
    monkeypatch.setattr(
        "research.mim_giveback.prospective.digest",
        lambda path: "hh" if Path(path) == history else "ph",
    )
    result, payload = evaluate(
        protocol, collection, as_of=datetime(2029, 3, 14, tzinfo=timezone.utc)
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["efficacy_calculated"] is False
    assert payload is None


@pytest.mark.parametrize(
    ("failed_gate", "expected"),
    [
        (None, "SUPPORT"),
        ("pf", "FAIL"),
        ("net_retention", "FAIL"),
        ("paired_lower95", "FAIL"),
        ("sensitivity_consistent", "FAIL"),
    ],
)
def test_endpoint_evaluation_applies_all_success_gates(
    tmp_path, monkeypatch, failed_gate, expected
):
    protocol = tmp_path / "protocol"
    collection = tmp_path / "collection"
    protocol.mkdir()
    collection.mkdir()
    history = tmp_path / "history.csv"
    history.write_text("history")
    (protocol / "protocol.json").write_text(json.dumps(protocol_payload(history)))
    (protocol / "completion.json").write_text("p")
    (collection / "completion.json").write_text("c")
    (collection / "collection_status.json").write_text(
        json.dumps({"protocol_completion_hash": "ph", "history_data_hash": "hh"})
    )
    days = pd.date_range("2026-09-14", periods=500, freq="D")
    pd.DataFrame(
        {
            "timestamp": (days + pd.Timedelta(hours=20)).astype(str),
            "contract": "MNQZ26",
        }
    ).to_csv(collection / "observations.csv", index=False)
    pd.DataFrame(
        {
            "day": days.strftime("%Y-%m-%d"),
            "contract": "MNQZ26",
            "eligible": True,
            "exclusion": None,
        }
    ).to_csv(collection / "eligibility.csv", index=False)
    monkeypatch.setattr(
        "research.mim_giveback.prospective.verify_run",
        lambda path: {"command": "freeze" if path == protocol else "collect"},
    )
    monkeypatch.setattr(
        "research.mim_giveback.prospective.digest",
        lambda path: "hh" if Path(path) == history else "ph",
    )
    monkeypatch.setattr(
        "research.mim_giveback.prospective.prepare_future", lambda *_: ["prepared"]
    )
    baseline_daily = pd.DataFrame({"day": range(500), "net": 1.0})
    candidate_daily = pd.DataFrame({"day": range(500), "net": 2.0})
    baseline_trades = pd.DataFrame({"net": [1.0], "exit_reason": ["EOD"]})
    candidate_trades = pd.DataFrame({"net": [2.0], "exit_reason": ["EOD"]})
    calls = iter(
        [
            (baseline_daily, pd.DataFrame(), pd.DataFrame(), baseline_trades),
            (
                candidate_daily,
                pd.DataFrame(),
                pd.DataFrame(),
                candidate_trades,
            ),
        ]
    )
    monkeypatch.setattr(
        "research.mim_giveback.prospective.simulate_sessions",
        lambda *a, **k: next(calls),
    )
    monkeypatch.setattr(
        "research.mim_giveback.prospective.performance",
        lambda trades, _daily: (
            {"net": 100.0, "pf": 2.0, "pf_infinite": False}
            if trades is baseline_trades
            else {
                "net": 89.0 if failed_gate == "net_retention" else 200.0,
                "pf": 1.2 if failed_gate == "pf" else 2.0,
                "pf_infinite": False,
            }
        ),
    )

    def fake_ci(_values, block, _seed, **_kwargs):
        lower = 0.5
        if failed_gate == "paired_lower95" and block == CONFIG["bootstrap_blocks"][0]:
            lower = -0.5
        if (
            failed_gate == "sensitivity_consistent"
            and block == CONFIG["bootstrap_blocks"][1]
        ):
            lower = -0.5
        return {"mean": 1.0, "lower95": lower, "upper95": 1.5}

    monkeypatch.setattr("research.mim_giveback.prospective.one_sided_ci", fake_ci)
    result, payload = evaluate(protocol, collection)
    assert result["verdict"] == expected
    assert result["efficacy_calculated"] is True
    if failed_gate:
        assert result["gates"][failed_gate] is False
    else:
        assert all(result["gates"].values())
    assert payload is not None


def test_semantic_verifier_rejects_forged_final_gate(tmp_path, monkeypatch):
    target = tmp_path / "evaluation"
    protocol = tmp_path / "protocol"
    target.mkdir()
    protocol.mkdir()
    (protocol / "protocol.json").write_text(
        json.dumps({"rules": frozen_rules(endpoint=2)})
    )
    paired = pd.DataFrame(
        {
            "day": ["a", "b"],
            "baseline_net": [1.0, 1.0],
            "candidate_net": [2.0, 2.0],
            "candidate_minus_baseline": [1.0, 1.0],
        }
    )
    paired.to_csv(target / "paired_daily.csv", index=False)
    pd.DataFrame({"day": ["a", "b"], "net": [1.0, 1.0]}).to_csv(
        target / "baseline_daily.csv", index=False
    )
    pd.DataFrame({"day": ["a", "b"], "net": [2.0, 2.0]}).to_csv(
        target / "candidate_daily.csv", index=False
    )
    pd.DataFrame({"net": [4.0]}).to_csv(target / "candidate_trades.csv", index=False)
    verdict = {
        "verdict": "SUPPORT",
        "eligible_sessions": 2,
        "efficacy_calculated": True,
        "baseline": {"net": 2.0},
        "candidate": {"net": 4.0, "pf": 2.0, "pf_infinite": False},
        "paired_ci": {
            "5": {"lower95": 0.5},
            "10": {"lower95": -0.5},
            "20": {"lower95": 0.5},
        },
        "gates": {
            "pf": True,
            "net_retention": True,
            "paired_lower95": True,
            "sensitivity_consistent": True,
        },
    }
    (target / "final_verdict.json").write_text(json.dumps(verdict))
    monkeypatch.setattr(
        artifacts,
        "verify_run",
        lambda _: {"command": "evaluate", "bindings": {"protocol": str(protocol)}},
    )
    with pytest.raises(ValueError, match="gate verification"):
        workflow.verify_semantics(target)
