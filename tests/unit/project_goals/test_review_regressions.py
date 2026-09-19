"""Synthetic evidence and isolated process fixtures for review fixes."""

import json
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from research.project_goals import accounts, audit, power, scheduler
from research.project_goals.archive import restore_chunks
from research.project_goals.common import digest, write_json, write_csv
from research.project_goals.portfolio import marked_curves, portfolio_report
from research.project_goals.reconciliation import pair_fills, realized_economics
from research.project_goals.session_report import session_report
from tests.unit.project_goals.test_evidence import fill, evidence


def attribution_fixture(tmp_path, raw, strategy="MIM", line=None):
    source = tmp_path / "source"
    source.write_text(line or "order_id\n10\n")
    mapping = tmp_path / "mapping.json"
    record = dict(
        strategy=strategy,
        source=str(source),
        line=2 if strategy == "MIM" else 1,
        sha256=digest(source),
        source_prefix_bytes=source.stat().st_size,
    )
    write_json(mapping, dict(orders={"10": record}))
    fills = tmp_path / "fills.json"
    write_json(fills, raw)
    return fills, mapping, record


@pytest.mark.parametrize(
    "bad", [dict(size=-1), dict(price=0), dict(side=3), dict(accountId=None)]
)
def test_invalid_batch_cannot_emit_downstream_results(tmp_path, bad):
    fills, mapping, _ = attribution_fixture(tmp_path, [fill(**bad)])
    result, rows, canonical = audit.attributed_snapshot(fills, mapping)
    assert result["status"] == "INVALID_EVIDENCE" and not canonical
    assert not pair_fills(rows)[0]
    assert realized_economics(rows)["status"] == "INVALID_EVIDENCE"


def test_conflicting_duplicate_poisoned_batch_is_retained_but_not_paired(tmp_path):
    fills, mapping, _ = attribution_fixture(tmp_path, [fill(), fill(price=200)])
    result, rows, canonical = audit.attributed_snapshot(fills, mapping)
    assert not canonical and result["economic_sensitivity"] is None
    assert rows[0]["evidence_valid"] is False
    assert not pair_fills(rows)[0]


@pytest.mark.parametrize(
    "strategy,line",
    [("MIM", "order_id\n2100\n"), ("YANK", "INFO ProjectX market close #2100\n")],
)
def test_attribution_requires_exact_order_id(tmp_path, strategy, line):
    fills, mapping, _ = attribution_fixture(tmp_path, [fill()], strategy, line)
    with pytest.raises(ValueError, match="exact"):
        audit.attributed_snapshot(fills, mapping)


def test_inconsistent_source_declarations_rejected(tmp_path):
    fills, mapping, record = attribution_fixture(tmp_path, [fill()])
    write_json(
        mapping, dict(orders={"10": record, "11": dict(record, source_prefix_bytes=1)})
    )
    with pytest.raises(ValueError, match="inconsistent"):
        audit.attributed_snapshot(fills, mapping)


def test_account_epoch_args_propagate_and_time_sort_uses_actual_instants(tmp_path):
    raw = [
        fill(creationTimestamp="2026-08-13T09:00:00-04:00"),
        fill(
            id=2,
            side=1,
            profitAndLoss=10,
            price=105,
            creationTimestamp="2026-08-13T13:30:00Z",
        ),
    ]
    fills, mapping, _ = attribution_fixture(tmp_path, raw)
    result, rows, canonical = audit.attributed_snapshot(
        fills, mapping, account=2, epoch_start="2026-08-01T00:00:00Z"
    )
    assert [r["fill_id"] for r in rows] == [1, 2]
    assert all(r["epoch"] == "2026-08-01T00:00:00Z" for r in canonical)
    assert pair_fills(rows, {"MNQU26": 2})[0][0]["reconstructed_gross"] == 10
    assert realized_economics(list(reversed(rows))) == realized_economics(rows)


def test_unknown_epoch_is_not_cross_session_fifo(tmp_path):
    _, rows = audit.reconcile(
        [
            fill(),
            fill(
                id=2,
                side=1,
                price=105,
                profitAndLoss=10,
                creationTimestamp="2026-08-14T14:00:00Z",
            ),
        ]
    )
    assert not pair_fills(rows, {"MNQU26": 2})[0]
    assert realized_economics(rows)["status"] == "PER_SESSION_CONDITIONAL_ONLY"
    fills, marks, coverage = evidence()
    fills[0]["epoch"] = "configured_epoch_requires_statement"
    assert marked_curves(fills, marks, coverage)[0]["status"] == "INSUFFICIENT_DATA"


def export_fixture(
    tmp_path,
    days=("2026-08-13",),
    contract="CON.F.US.MNQ.U26",
    epoch="confirmed-input-epoch",
):
    fills, marks, _ = evidence()
    all_fills = []
    bars = []
    for day_index, day in enumerate(days):
        for f in fills:
            all_fills.append(
                dict(
                    f,
                    id=f["id"] + day_index * 2,
                    contract=contract,
                    epoch=epoch,
                    timestamp=f["timestamp"].replace("2026-08-13", day),
                )
            )
        for m in marks:
            stamp = datetime.fromisoformat(
                m["timestamp"].replace("Z", "+00:00")
            ) - timedelta(minutes=1)
            bars.append(
                dict(t=stamp.isoformat().replace("2026-08-13", day), c=m["close"])
            )
    path = tmp_path / (contract.replace(".", "_") + ".json")
    write_json(
        path,
        dict(
            request=dict(
                contractId=contract, unit=2, unitNumber=1, includePartialBar=False
            ),
            response=dict(success=True, bars=bars),
        ),
    )
    return all_fills, path


def test_unknown_epoch_reports_separate_sessions_not_stitched_balance(tmp_path):
    fills, path = export_fixture(tmp_path, ("2026-08-13", "2026-08-14"), epoch=None)
    report, rows = session_report(fills, [path], operating_cost_monthly=29)
    assert report["status"] == "PER_SESSION_CONDITIONAL_UNKNOWN_EPOCH"
    assert len(report["per_session"]) == 2 and "scenarios" not in report
    assert report["operating_expense_allocation"].startswith("UNKNOWN")
    assert all(r["accounting_scope"].startswith("PER_SESSION") for r in rows)


def test_multiple_actual_epochs_refuse_even_if_each_day_flat(tmp_path):
    fills, path = export_fixture(tmp_path, ("2026-08-13", "2026-08-14"))
    fills[-1]["epoch"] = "another-epoch"
    assert session_report(fills, [path])[0]["status"] == "INSUFFICIENT_DATA"


def test_multi_day_and_contract_exports_and_operating_cost(tmp_path):
    u, p = export_fixture(tmp_path, ("2026-08-13", "2026-08-14"))
    z, q = export_fixture(
        tmp_path, ("2026-08-13", "2026-08-14"), contract="CON.F.US.MNQ.Z26"
    )
    for row in z:
        row["id"] += 100
        row["strategy"] = "YANK"
    report, _ = session_report(u + z, [p, q], operating_cost_monthly=29)
    assert report["status"] == "CONDITIONAL_DESCRIPTIVE"
    actual = report["scenarios"]["actual"]
    assert actual["metrics"]["sessions"] == 2
    assert (
        actual["monthly_before_operating_cost"]["2026-08"]
        - actual["monthly_after_operating_cost"]["2026-08"]
        == 29
    )
    duplicate = json.loads(p.read_text())
    duplicate["response"]["bars"][0]["c"] += 1
    bad = tmp_path / "conflict.json"
    write_json(bad, duplicate)
    with pytest.raises(ValueError, match="conflicting duplicate mark"):
        session_report(u + z, [p, q, bad])


def test_unsupported_contract_and_empty_results_fail_closed(tmp_path):
    fills, path = export_fixture(tmp_path, contract="CON.F.US.ES.U26")
    with pytest.raises(ValueError, match="unsupported contract"):
        session_report(fills, [path])
    assert session_report([], [])[0]["status"] == "INSUFFICIENT_DATA"
    raw = json.loads(path.read_text())
    raw["request"]["contractId"] = "CON.F.US.MNQ.U26"
    raw["response"]["bars"] = []
    write_json(path, raw)
    assert session_report(fills, [path])[0]["status"] == "INSUFFICIENT_DATA"


def test_zero_candidate_exposure_is_not_equal_exposure():
    rows = [
        dict(
            timestamp="2026-08-13T14:00:00Z",
            strategy=s,
            cumulative_net=1.0,
            gross_notional=100.0 if s == "GAP" else 0.0,
            baseline_units=1.0,
        )
        for s in ("MIM", "YANK", "GAP")
    ]
    result = portfolio_report(rows)
    assert (
        result["scenarios"]["MIM1_YANK2_same_gross_exposure"]["exposure_scale"] is None
    )


def sessions(pnls, traded=True, payouts=None):
    return [
        dict(
            session=(datetime(2026, 8, 1) + timedelta(days=i)).date().isoformat(),
            net_pnl=v,
            intraday_min_pnl=min(0, v),
            traded=traded,
            requested_payout=(payouts or {}).get(i, 0),
        )
        for i, v in enumerate(pnls)
    ]


def test_nontrading_rows_do_not_qualify_combine_or_standard():
    assert (
        accounts.model_path(sessions([1500, 1500], False))["ending_phase"] == "combine"
    )
    result = accounts.model_path(
        sessions([300] * 5, False, {4: 500}), starting_phase="xfa_standard"
    )
    assert result["operator_withdrawals"] == 0 and result["rows"][-1]["request_denied"]


def test_breach_and_insolvency_terminal_balance_unknown():
    for phase, pnl, capital in [
        ("combine", -2000, 50000),
        ("self_funded", -10000, 10000),
    ]:
        result = accounts.model_path(sessions([pnl]), starting_phase=phase)
        assert result["end_balance"] is None
        assert result["last_verified_balance"] == capital
        assert result["rows"][-1]["terminal_balance"] is None


def test_standard_payout_cash_floor_counter_reset_and_denial():
    result = accounts.model_path(
        sessions([300] * 7, payouts={4: 500, 5: 500}), starting_phase="xfa_standard"
    )
    paid = result["rows"][4]
    assert (
        paid["balance"] == 1000
        and paid["operator_withdrawal"] == 450
        and paid["floor"] == 0
    )
    assert paid["qualifying_winning_days"] == 0 and paid["traded_days_since_reset"] == 0
    assert result["rows"][5]["request_denied"]
    assert result["operator_withdrawals"] == 450 and result["end_balance"] == 1600


def test_consistency_payout_and_combine_phase_reset():
    result = accounts.model_path(
        sessions([1000] * 3, payouts={2: 1500}), starting_phase="xfa_consistency"
    )
    assert result["end_balance"] == 1500 and result["operator_withdrawals"] == 1350
    assert (
        result["rows"][-1]["floor"] == 0
        and result["rows"][-1]["traded_days_since_reset"] == 0
    )
    combined = accounts.model_path(sessions([1500, 1500]))
    assert combined["ending_phase"] == "xfa_standard" and combined["end_balance"] == 0
    assert (
        combined["rows"][-1]["floor"] == -2000
        and combined["rows"][-1]["traded_days_since_reset"] == 0
    )


def test_consistency_denies_large_best_day_and_balance_limit():
    for pnls, payout in [([2000, 100, 100], 500), ([1000] * 3, 1600)]:
        result = accounts.model_path(
            sessions(pnls, payouts={2: payout}), starting_phase="xfa_consistency"
        )
        assert (
            result["operator_withdrawals"] == 0 and result["rows"][-1]["request_denied"]
        )


def paired_csv(path, matrix, start="2026-01-01"):
    rows = []
    day = datetime.fromisoformat(start)
    for i, values in enumerate(matrix):
        row = dict(zip(power.REQUIRED[1:], values))
        row.update(
            session=str((day + timedelta(days=i)).date()),
            corrected_contract=True,
            complete_costs=True,
        )
        rows.append(row)
    write_csv(path, rows)


def synthetic_power_context(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    n = 120
    matrix = np.column_stack(
        (
            rng.normal(0, 10, n),
            rng.normal(0, 2, n),
            rng.normal(20, 2, n),
            np.full(n, 100.0),
            np.full(n, 200.0),
            np.full(n, 100.0),
            np.ones((n, 3)),
        )
    )
    calibration = tmp_path / "calibration.csv"
    paired_csv(calibration, matrix, "2025-01-01")
    artifact = power.sweep(
        calibration, effects=(20.0,), horizons=(n,), blocks=(1, 3), draws=256, seed=7
    )
    assert artifact["portfolio_sweep"] and all(
        r["adequate"] for r in artifact["portfolio_sweep"]
    )
    power_path = tmp_path / "power.json"
    write_json(power_path, artifact)
    reg = dict(
        status="registered",
        blocks=[1, 3],
        cost_stress=[0.0, 1.0],
        draws=256,
        seed=7,
        required_sessions=n,
        alpha=0.05,
        allocations={"MIM": 1, "YANK": 2, "GAP": 1},
        source_hashes=power.source_hashes(),
        power_sha256=digest(power_path),
        code_sha256=digest(power.__file__),
        power_status="ADEQUATE_BOTH_ENDPOINTS",
        effect_usd=20.0,
        calibration_commit="a" * 40,
        registered_at="2025-12-01T00:00:00Z",
        freshness_after="2025-12-01T00:00:00Z",
    )
    monkeypatch.setattr(
        power,
        "committed_registration",
        lambda *a: (reg, "b" * 40, datetime(2025, 12, 1, tzinfo=timezone.utc)),
    )
    monkeypatch.setattr(
        power,
        "git_read",
        lambda repo, args: (
            power_path.read_text().strip()
            if args[0] == "show" and ":" in args[-1]
            else ("2025-11-01T00:00:00Z" if args[0] == "show" else "")
        ),
    )
    data = tmp_path / "synthetic-fresh.csv"
    paired_csv(data, matrix)
    return matrix, reg, data, power_path, calibration


def test_valid_synthetic_evaluation_pass_is_deterministic(tmp_path, monkeypatch):
    _, _, data, power_path, calibration = synthetic_power_context(tmp_path, monkeypatch)
    result = power.evaluate(data, tmp_path / "reg", power_path, calibration, tmp_path)
    assert result["status"] == "PASS" and result["confirmatory"]
    assert result == power.evaluate(
        data, tmp_path / "reg", power_path, calibration, tmp_path
    )


def test_synthetic_positive_expectancy_can_fail_risk_endpoint(tmp_path, monkeypatch):
    matrix, _, data, power_path, calibration = synthetic_power_context(
        tmp_path, monkeypatch
    )
    rng = np.random.default_rng(73)
    matrix[:, 0] = rng.normal(20, 1, len(matrix))
    matrix[:, 1] = rng.normal(1, 0.2, len(matrix))
    matrix[:, 2] = rng.normal(20, 20, len(matrix))
    paired_csv(data, matrix)
    result = power.evaluate(data, tmp_path / "reg", power_path, calibration, tmp_path)
    assert result["status"] == "FAIL"
    assert all(r["standalone_lower"] > 0 for r in result["results"])
    assert any(r["delta_sharpe_lower"] < 0 for r in result["results"])


def test_synthetic_baseline_pass_can_fail_registered_cost_stress(tmp_path, monkeypatch):
    matrix, _, data, power_path, calibration = synthetic_power_context(
        tmp_path, monkeypatch
    )
    matrix[:, 8] = 100
    paired_csv(data, matrix)
    result = power.evaluate(data, tmp_path / "reg", power_path, calibration, tmp_path)
    assert result["status"] == "FAIL"
    assert all(r["pass_gate"] for r in result["results"] if r["cost"] == 0)
    assert all(not r["pass_gate"] for r in result["results"] if r["cost"] == 1)


@pytest.mark.parametrize(
    "change",
    [{"alpha": 0.1}, {"blocks": [1]}, {"cost_stress": [0.0, 2.0]}, {"seed": 8}],
)
def test_registration_must_select_exact_calibrated_design_before_data_read(
    tmp_path, monkeypatch, change
):
    _, reg, data, power_path, calibration = synthetic_power_context(
        tmp_path, monkeypatch
    )
    reg.update(change)
    data.unlink()
    result = power.evaluate(data, tmp_path / "reg", power_path, calibration, tmp_path)
    assert "calibrated simultaneous design" in result["reason"]


@pytest.mark.parametrize("seed", [-1, None, True, 1.5])
def test_invalid_seed_refused(seed):
    design = power.calibration_design([1], [0, 1], 0.05, 128, seed)
    with pytest.raises(ValueError, match="seed"):
        power.validate_design(design)


def test_draw_count_must_resolve_simultaneous_tail():
    with pytest.raises(ValueError, match="resolve"):
        power.validate_design(power.calibration_design([1, 3], [0, 1], 0.05, 200, 7))


def fixture_poll(tmp_path, monkeypatch, wait_for_gate=False):
    import shlex

    root = tmp_path / "root"
    base = root / "research/mim_comparison"
    base.mkdir(parents=True)
    runs = base / "runs"
    old = runs / "20200101-shadow-old"
    old.mkdir(parents=True)
    (old / "preserve").write_bytes(b"old governing artifacts")
    script = tmp_path / "child.py"
    script.write_text(
        """import hashlib,json,time
from pathlib import Path
root=Path(__file__).parent/'root'
runs=root/'research/mim_comparison/runs'
"""
        + (
            """while not (root/'release').exists(): time.sleep(.01)
"""
            if wait_for_gate
            else ""
        )
        + """for name,seal,key in [('20260910-contract-feed/poll-fixture','manifest.json','artifacts'),('20260919-shadow-fixture','completion.json','sha256')]:
    directory=runs/name
    directory.mkdir(parents=True)
    payload=b'completed immutable bytes'*100
    (directory/'payload').write_bytes(payload)
    (directory/seal).write_text(json.dumps({key:{'payload':hashlib.sha256(payload).hexdigest()}}))
    if key=='artifacts': print(json.dumps({'invocation':str(directory)}),flush=True)
    else:
        print(json.dumps({'window':'ignored derived window'}),flush=True)
        print(directory,flush=True)
"""
    )
    (base / "poll.sh").write_text(
        "exec " + shlex.quote(sys.executable) + " " + shlex.quote(str(script)) + "\n"
    )
    monkeypatch.setattr(
        scheduler, "preflight", lambda root: dict(status="VERIFIED_FIXTURE")
    )
    monkeypatch.setattr(scheduler, "health", lambda root: dict(eligible_sessions=0))
    return root, tmp_path / "state", old


def test_successful_actual_child_formats_archive_and_preserve_old(
    tmp_path, monkeypatch
):
    root, state, old = fixture_poll(tmp_path, monkeypatch)
    result = scheduler.poll_once(root, state)
    assert result["status"] == "SUCCESS" and len(result["archives"]) == 2
    for i, entry in enumerate(result["archives"]):
        restored = tmp_path / ("restored" + str(i))
        restore_chunks(entry["manifest"], restored)
        assert (restored / "payload").read_bytes() == b"completed immutable bytes" * 100
    assert (old / "preserve").read_bytes() == b"old governing artifacts"
    assert not (state / "child.json").exists()


def test_stale_child_observable_never_killed_and_blocks_replacement(
    tmp_path, monkeypatch
):
    root, state, _ = fixture_poll(tmp_path, monkeypatch, wait_for_gate=True)
    result = {}
    thread = threading.Thread(
        target=lambda: result.update(scheduler.poll_once(root, state, stale_after=0.05))
    )
    thread.start()
    try:
        deadline = time.monotonic() + 3
        heartbeat = {}
        while time.monotonic() < deadline:
            if (state / "heartbeat.json").exists():
                heartbeat = json.loads((state / "heartbeat.json").read_text())
                if heartbeat["status"] == "OPERATOR_RECOVERY_REQUIRED_CHILD_ALIVE":
                    break
            time.sleep(0.01)
        assert heartbeat["child_alive"] and heartbeat["operator_recovery_required"]
        assert scheduler.child_health(state)["child_alive"]
        assert scheduler.poll_once(root, state)["status"] == "OVERLAP_REFUSED"
    finally:
        (root / "release").touch()  # Child exits itself; no signal or termination.
        thread.join(3)
    assert not thread.is_alive()
    assert result["status"] == "SUCCESS" and result["stale_child_observed"]
    assert scheduler.child_health(state)["status"] == "NO_PENDING_CHILD"
    from types import SimpleNamespace

    assert (
        scheduler.poll_once(
            root, state, runner=lambda *a, **k: SimpleNamespace(returncode=1)
        )["status"]
        == "POLL_FAILED"
    )


def test_orphan_receipt_refuses_replacement_without_signaling(tmp_path, monkeypatch):
    import os

    root, state, _ = fixture_poll(tmp_path, monkeypatch)
    state.mkdir()
    write_json(
        state / "child.json",
        dict(
            identity=scheduler.process_identity(os.getpid()),
            started_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
    result = scheduler.poll_once(root, state)
    assert result["status"] == "OPERATOR_RECOVERY_REQUIRED" and result["child_alive"]
    assert (state / "child.json").exists()


def test_cli_preserves_insufficiency_and_propagates_expenses(tmp_path, monkeypatch):
    from research.project_goals.__main__ import main

    fills, path = export_fixture(tmp_path)
    fill_csv = tmp_path / "fills.csv"
    write_csv(fill_csv, fills)
    # Restrict directory to the single mark export.
    marks = tmp_path / "mark-exports"
    marks.mkdir()
    path.rename(marks / path.name)
    out = tmp_path / "report"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "project-goals",
            "report",
            "--fills",
            str(fill_csv),
            "--broker-marks",
            str(marks),
            "--operating-cost-monthly",
            "29",
            "--output",
            str(out),
        ],
    )
    main()
    result = json.loads((out / "evidence.json").read_text())
    for report in [result, *result["cost_sweeps"].values()]:
        scenario = report["scenarios"]["actual"]
        assert scenario["monthly_before_operating_cost"]["2026-08"] - scenario[
            "monthly_after_operating_cost"
        ]["2026-08"] == pytest.approx(29)
    fills[0]["actual_cost"] = None
    write_csv(fill_csv, fills)
    out = tmp_path / "insufficient"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "project-goals",
            "report",
            "--fills",
            str(fill_csv),
            "--broker-marks",
            str(marks),
            "--output",
            str(out),
        ],
    )
    main()
    result = json.loads((out / "evidence.json").read_text())
    assert result["status"] == "INSUFFICIENT_DATA" and not result["cost_sweeps"]
