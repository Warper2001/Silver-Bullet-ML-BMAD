"""Independent fixtures for every frozen edge-case matrix row."""

import json
from pathlib import Path

import pandas as pd
import pytest

from research.pf_improvement import artifacts as art
from research.pf_improvement import __main__ as cli
from research.pf_improvement.carry import build_carry
from research.pf_improvement.execution import build_execution, classify_gate
from research.pf_improvement.marks import build_marks, held_mark_features


def test_exact_ids_validate_side_size_and_complete_mim_net():
    result = build_execution()
    mim = result["round_trips"].query("strategy == 'MIM'")
    exact = mim.query("join_method == 'exact_order_id'")
    unattributed = result["round_trips"].query(
        "strategy == 'UNATTRIBUTED_SHARED_ACCOUNT'"
    )
    assert len(mim) == 25 and len(exact) == 5 and len(unattributed) == 2
    assert exact.strategy_side_size_validated.all()
    assert exact.complete_costs.all() and exact.broker_net.notna().all()
    assert (exact.broker_net == exact.broker_gross - exact.costs).all()
    assert exact.signed_difference.sum() == pytest.approx(-6.0)
    assert unattributed.modeled_gross.isna().all()


def test_missing_causality_preserves_candidates_and_is_insufficient():
    result = build_execution()
    assert result["gate"]["verdict"] == "INSUFFICIENT_CAUSAL_EVIDENCE"
    assert (result["round_trips"].join_method == "unmatched_broker_candidate").any()
    gap = result["round_trips"].query("strategy == 'GAP'")
    assert gap.broker_net.isna().all() and not gap.causal_decomposition_available.any()
    assert gap.signed_difference.sum() == pytest.approx(-113.0)
    coverage = result["coverage"].set_index("source")
    assert coverage.loc["mim_orders", "integrity_status"] == "BROKEN_UNREGISTERED"
    assert coverage.loc["mim_decisions", "integrity_status"] == "BROKEN_UNREGISTERED"
    assert coverage.loc["gap_decisions", "integrity_status"] == "KNOWN_SCAR"
    assert coverage.loc["gap_trades", "duplicate_business_keys"] == "2026-06-25"


def test_three_state_execution_gate():
    def gate(coverage, defect, causal, unchanged=True):
        return classify_gate(
            complete_current_coverage=coverage,
            recurring_current_economic_defect=defect,
            exact_causality=causal,
            unchanged_intended_behavior=unchanged,
        )

    assert gate(True, True, True) == "CURRENT_REPAIR_CANDIDATE"
    assert gate(True, False, True) == "NO_REPAIRABLE_MECHANISM"
    assert gate(False, False, False) == "INSUFFICIENT_CAUSAL_EVIDENCE"
    assert gate(True, True, True, False) == "INSUFFICIENT_CAUSAL_EVIDENCE"


def test_carry_matrix_separates_venues_and_keeps_multiple_park_reasons():
    result = build_carry()
    matrix = result["matrix"]
    assert (
        len(matrix) == 36
        and matrix.root.nunique() == 12
        and matrix.venue.nunique() == 3
    )
    assert set(result["verdict"]["overall"]) >= {
        "PARK_ACCOUNT",
        "PARK_DATA",
        "PARK_POWER",
    }
    assert matrix.park_reasons.str.count(r"PARK_").eq(3).all()
    assert not matrix.returns_calculated.any() and not matrix.questions_sent.any()
    statuses = {"usable", "requires verification/acquisition", "unavailable"}
    for column in (
        "product_permission_status",
        "overnight_status",
        "integer_sizing_status",
        "margin_status",
        "cost_quote_status",
        "calendar_status",
    ):
        assert set(matrix[column]) <= statuses
    assert set(zip(matrix.root, matrix.venue)) == {
        (root, venue)
        for root in (
            "CL",
            "NG",
            "RB",
            "HO",
            "HG",
            "ZC",
            "ZW",
            "ZS",
            "ZM",
            "ZL",
            "LE",
            "HE",
        )
        for venue in ("Topstep", "TradeStation_SIM", "future_self_funded")
    }
    assert (
        matrix.query("venue == 'Topstep'")
        .path_classification.eq("UNAVAILABLE_OVERNIGHT")
        .all()
    )
    assert result["verdict"]["questions_prepared_not_sent"]
    assert (
        "## Data source" in result["questions"]
        and "## Account paths" in result["questions"]
    )


def test_output_refuses_overwrite_escape_and_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(art, "RUNS", tmp_path / "runs")
    run = art.create("audit", tmp_path / "runs" / "one")
    with pytest.raises(FileExistsError):
        art.create("audit", run)
    with pytest.raises(ValueError, match="direct runs child"):
        art.create("audit", tmp_path / "escape")
    (tmp_path / "runs" / "link").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="Symlink"):
        art.create("audit", tmp_path / "runs" / "link")


def test_holdout_and_unapproved_input_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="holdout"):
        art.readable(tmp_path / "data/sealed_holdout/x.csv")
    with pytest.raises(ValueError, match="approved inventory"):
        art.input_path("data/not-approved.csv")
    with pytest.raises(ValueError, match="manifest-bound"):
        art.validate_requested_paths(
            tmp_path,
            art.DEFAULT_GAP_DATA,
            art.DEFAULT_DIAGNOSTIC_RUN,
            art.DEFAULT_CARRY_DATA,
        )


def test_inventory_detects_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(art, "RUNS", tmp_path / "runs")
    run = art.create("audit")
    (run / "evidence.txt").write_text("ok")
    art.seal(run)
    (run / "evidence.txt").chmod(0o644)
    (run / "evidence.txt").write_text("changed")
    with pytest.raises(ValueError, match="inventory changed"):
        art.verify_inventory(run)


def test_seal_protects_child_but_parent_accepts_new_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(art, "RUNS", tmp_path / "runs")
    first = art.create("audit")
    (first / "payload").write_text("ok")
    art.seal(first)
    assert not first.stat().st_mode & 0o222
    assert (tmp_path / "runs").stat().st_mode & 0o200
    second = art.create("audit")
    assert second.parent == first.parent


def test_seal_failure_does_not_publish_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(art, "RUNS", tmp_path / "runs")
    run = art.create("audit")
    payload = run / "payload"
    payload.write_text("ok")
    original = Path.chmod

    def fail_payload(self, mode):
        if self == payload:
            raise OSError("chmod failed")
        return original(self, mode)

    monkeypatch.setattr(Path, "chmod", fail_payload)
    with pytest.raises(OSError, match="chmod failed"):
        art.seal(run)
    assert not (run / "completion.json").exists()


def test_failed_invocation_preserves_and_seals_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(art, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(
        art,
        "validate_inputs",
        lambda: (_ for _ in ()).throw(ValueError("corrupt input")),
    )
    monkeypatch.setattr("sys.argv", ["pf_improvement", "audit"])
    with pytest.raises(ValueError, match="corrupt input"):
        cli.main()
    run = next((tmp_path / "runs").iterdir())
    failure = json.loads((run / "failure.json").read_text())
    assert failure["error"] == "corrupt input"
    assert (run / "completion.json").is_file()
    assert all(
        not (item.stat().st_mode & 0o222) for item in run.rglob("*") if item.is_file()
    )


def test_privacy_scan_rejects_raw_account_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(art, "RUNS", tmp_path / "runs")
    run = art.create("audit")
    raw = json.loads(art.input_path("data/mim_nb/projectx_fills.json").read_text())
    (run / "leak.txt").write_text(str(raw[0]["accountId"]))
    with pytest.raises(ValueError, match="Privacy drift"):
        art._privacy_check(run)


def test_execution_exports_only_pseudonymous_broker_ids():
    result = build_execution()
    raw = json.loads(art.input_path("data/mim_nb/projectx_fills.json").read_text())
    rendered = result["events"].to_csv(index=False) + result["round_trips"].to_csv(
        index=False
    )
    for row in raw:
        assert str(row["id"]) not in rendered and str(row["orderId"]) not in rendered


def test_exact_held_slice_excludes_pre_entry_post_mark_and_stop_minute():
    times = pd.date_range(
        "2026-01-05 09:59", periods=9, freq="min", tz="America/New_York"
    )
    session = pd.DataFrame(
        {"timestamp": times, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}
    )
    session.loc[0, ["high", "low"]] = [10000.0, 1.0]  # pre-entry
    session.loc[2, ["high", "low", "close"]] = [105.0, 98.0, 102.0]
    session.loc[6, ["high", "low", "close"]] = [103.0, 99.0, 103.0]  # mark
    session.loc[7, ["high", "low"]] = [9999.0, 1.0]  # stop minute
    session.loc[8, ["high", "low"]] = [8888.0, 2.0]  # post-mark
    observed = held_mark_features(session, times[1], times[6], 1, 100.0, 103.0)
    assert observed == {
        "current_gross": 6.0,
        "current_net": 4.88,
        "mfe": 10.0,
        "mae": 4.0,
        "giveback": 4.0,
        "completed_bar_coverage": 5,
    }
    with pytest.raises(ValueError, match="unique continuous"):
        held_mark_features(session.drop(index=3), times[1], times[6], 1, 100.0, 103.0)
    with pytest.raises(ValueError, match="unique continuous"):
        held_mark_features(
            pd.concat([session, session.iloc[[3]]]), times[1], times[6], 1, 100.0, 103.0
        )


def test_candidate_cli_stops_before_later_builders(tmp_path, monkeypatch):
    monkeypatch.setattr(art, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(art, "validate_inputs", lambda: [])
    monkeypatch.setattr(art, "verify", lambda path, sealed=False: {"verified": True})
    gate = {
        "verdict": "CURRENT_REPAIR_CANDIDATE",
        "stops_later_stages": True,
        "complete_current_coverage": True,
        "exact_causality": True,
        "unchanged_intended_behavior": True,
        "recurring_current_economic_defect": True,
        "recoverable_dollars": 1.0,
        "reasons": ["fixture"],
    }
    execution = {
        "events": pd.DataFrame([{"x": 1}]),
        "round_trips": pd.DataFrame([{"x": 1}]),
        "coverage": pd.DataFrame([{"x": 1}]),
        "gate": gate,
    }
    monkeypatch.setattr(
        "research.pf_improvement.execution.build_execution", lambda: execution
    )
    monkeypatch.setattr(
        "research.pf_improvement.marks.build_marks", lambda: pytest.fail("marks called")
    )
    monkeypatch.setattr(
        "research.pf_improvement.carry.build_carry", lambda: pytest.fail("carry called")
    )
    output = tmp_path / "runs" / "candidate"
    monkeypatch.setattr("sys.argv", ["pf_improvement", "run", "--output", str(output)])
    cli.main()
    assert (output / "repair_specification.md").is_file()
    assert not (output / "mim_decision_marks.csv").exists()


def test_stopped_inventory_and_report_claim_corruption_are_rejected():
    with pytest.raises(ValueError, match="forbidden later-stage"):
        art._verify_stopped_inventory(art.STOP_REQUIRED | {"mim_decision_marks.csv"})
    art._verify_report_claims("gate pf", "gate", ["gate", "pf"], ["gate"])
    with pytest.raises(ValueError, match="Report claims"):
        art._verify_report_claims("gate", "gate", ["gate", "pf"], ["gate"])


def test_audit_semantics_reject_corruption(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_text("source")
    monitor = tmp_path / "monitor.csv"
    monitor.write_text("run_at,strategy\n2026-01-01T00:00:00+00:00,MIM\n")
    monkeypatch.setattr(art, "PINNED_INPUTS", {"source": "abc"})
    monkeypatch.setattr(
        art,
        "input_path",
        lambda relative: (
            monitor if relative == "logs/portfolio_decay_shadow.csv" else source
        ),
    )
    run = tmp_path / "run"
    run.mkdir()
    manifest = {"inputs": {"source": "abc"}}
    pd.DataFrame([{"path": "source", "sha256": "abc", "bytes": len("source")}]).to_csv(
        run / "input_inventory.csv", index=False
    )
    (run / "audit.json").write_text(
        json.dumps(
            {
                "valid": True,
                "input_files": 1,
                "raw_account_evidence_copied": False,
                "sealed_holdout_accessed": False,
                "strategy_simulator_invoked": False,
                "alternative_strategy_returns_calculated": False,
            }
        )
    )
    (run / "decay_monitor_coverage.json").write_text(
        json.dumps(
            {
                "rows": 1,
                "observation_timestamps": ["2026-01-01T00:00:00+00:00"],
                "strategy_rows": {"MIM": 1},
                "efficacy_interpreted": False,
                "monitor_invoked": False,
            }
        )
    )
    art._verify_audit(run, manifest)
    payload = json.loads((run / "audit.json").read_text())
    payload["valid"] = False
    (run / "audit.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Audit semantics"):
        art._verify_audit(run, manifest)


def test_completed_run_mark_boundaries_stop_censoring_and_labels():
    result = build_marks()
    marks, trades = result["marks"], result["trades"]
    assert (
        len(marks) == 6673
        and marks.mark_timestamp.str[11:16].between("10:00", "15:30").all()
    )
    assert marks.reversal_mark_exiting_leg.sum() == 7
    joined = marks.merge(
        trades[["trade_id", "exit_event_timestamp", "exit_reason"]], on="trade_id"
    )
    assert (
        pd.to_datetime(joined.mark_timestamp, utc=True)
        < pd.to_datetime(joined.exit_event_timestamp, utc=True)
    ).all()
    assert joined.query("exit_reason == 'CAT_STOP'").completed_bar_coverage.ge(1).all()
    assert trades.exit_reason.value_counts().to_dict() == {
        "EOD_CLOSE_PROXY": 723,
        "CAT_STOP": 71,
        "REVERSAL": 7,
    }
    assert marks.candidate_return.isna().all()
    assert result["verdict"]["verdict"] == "DISTINCT_HYPOTHESIS_REMAINS"
    assert set(result["distributions"].group_dimension) == {
        "all",
        "final_outcome",
        "direction",
        "final_exit_reason",
        "top_5pct_trade",
    }
    corrupted = result["distributions"].copy()
    corrupted.loc[0, "mean"] += 1
    with pytest.raises(ValueError, match="distribution moments"):
        art._verify_distributions(corrupted, marks)
