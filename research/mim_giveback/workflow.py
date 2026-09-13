from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import artifacts
from .config import CONFIG
from .data import load_source, prepare_sessions, threshold_family
from .prospective import (
    collect as collect_future,
    eligibility as future_eligibility,
    evaluate as evaluate_future,
)
from .report import write_report
from .statistics import performance, power_from_null
from .study import run_power, run_sweep


def _require_frozen_config(manifest: dict[str, object]) -> None:
    if manifest.get("config") != CONFIG:
        raise ValueError("Current experiment configuration differs from parent run")


def power(path: Path, source_run: Path, data: Path) -> None:
    source = load_source(source_run, data)
    prepared = prepare_sessions(data)
    family = threshold_family(source["marks"])
    result = run_power(prepared, source, family)
    result["baseline_daily"].to_csv(path / "baseline_daily.csv", index=False)
    result["baseline_trades"].to_csv(path / "baseline_trades.csv", index=False)
    result["family"].to_csv(path / "candidate_family.csv", index=False)
    result["null"].to_csv(path / "mismatched_null_daily.csv", index=False)
    result["power"].to_csv(path / "power_by_threshold.csv", index=False)
    artifacts.json_write(path / "baseline_reconciliation.json", result["baseline"])
    artifacts.json_write(path / "power_verdict.json", result["verdict"])
    write_report(
        path,
        "MIM Giveback Power Gate",
        [
            ("Verdict", result["verdict"]),
            ("Baseline reconciliation", result["baseline"]),
            (
                "Interpretation",
                "A non-POWERED verdict is terminal. No aligned candidate return was evaluated.",
            ),
        ],
    )


def inventory(path: Path, power_run: Path) -> None:
    manifest = artifacts.verify_run(power_run)
    _require_frozen_config(manifest)
    if manifest["command"] != "power":
        raise ValueError("Inventory requires a power run")
    verdict = json.loads((power_run / "power_verdict.json").read_text())
    if verdict["verdict"] != "POWERED":
        raise ValueError("Power verdict is terminal; threshold inventory denied")
    family = pd.read_csv(power_run / "candidate_family.csv")
    family["candidate_return_calculated"] = False
    family.to_csv(path / "threshold_inventory.csv", index=False)
    artifacts.json_write(
        path / "inventory_status.json",
        {
            "verdict": "INVENTORY_COMPLETE",
            "thresholds": len(family),
            "selected_threshold": None,
            "power_completion_hash": artifacts.digest(power_run / "completion.json"),
            "source_run": manifest["bindings"]["source_run"],
            "source_completion_hash": artifacts.digest(
                artifacts.resolve_path(manifest["bindings"]["source_run"])
                / "completion.json"
            ),
            "data": manifest["bindings"]["data"],
            "data_hash": artifacts.digest(
                artifacts.resolve_path(manifest["bindings"]["data"])
            ),
        },
    )
    write_report(
        path,
        "MIM Giveback Threshold Inventory",
        [("Status", {"thresholds": len(family), "returns_calculated": False})],
    )


def sweep(path: Path, inventory_run: Path, source_run: Path, data: Path) -> None:
    parent = artifacts.verify_run(inventory_run)
    _require_frozen_config(parent)
    if parent["command"] != "inventory":
        raise ValueError("Sweep requires a threshold inventory")
    status = json.loads((inventory_run / "inventory_status.json").read_text())
    if (
        artifacts.digest(source_run / "completion.json")
        != status["source_completion_hash"]
        or artifacts.digest(data) != status["data_hash"]
    ):
        raise ValueError("Sweep source or market data differs from powered inventory")
    source = load_source(source_run, data)
    family = pd.read_csv(inventory_run / "threshold_inventory.csv")[
        ["quantile", "threshold"]
    ]
    prepared = prepare_sessions(data)
    result = run_sweep(prepared, source["daily"], family)
    result["table"].to_csv(path / "sweep_results.csv", index=False)
    artifacts.json_write(path / "baseline_metrics.json", result["baseline"])
    artifacts.json_write(
        path / "development_verdict.json",
        {
            "verdict": result["verdict"],
            "selected_threshold": result["selected_threshold"],
            "historical_only": True,
            "promotion_authorized": False,
        },
    )
    if result["selected_payload"] is not None:
        for frame, name in zip(
            result["selected_payload"],
            [
                "selected_daily.csv",
                "selected_ledger.csv",
                "selected_decisions.csv",
                "selected_trades.csv",
            ],
        ):
            frame.to_csv(path / name, index=False)
    write_report(
        path,
        "MIM Giveback Development Sweep",
        [
            (
                "Verdict",
                {
                    "verdict": result["verdict"],
                    "selected_threshold": result["selected_threshold"],
                },
            ),
            (
                "Boundary",
                "Exposed-history development cannot validate or promote the rule.",
            ),
        ],
    )


def freeze(path: Path, sweep_run: Path) -> None:
    manifest = artifacts.verify_run(sweep_run)
    _require_frozen_config(manifest)
    if manifest["command"] != "sweep":
        raise ValueError("Freeze requires a development sweep")
    verdict = json.loads((sweep_run / "development_verdict.json").read_text())
    if (
        verdict["verdict"] != "DEVELOPMENT_PASS"
        or verdict["selected_threshold"] is None
    ):
        raise ValueError("Development did not produce a qualifying threshold")
    frozen = pd.Timestamp(
        json.loads((path / "manifest.json").read_text())["created_at"]
    )
    data_path = artifacts.resolve_path(manifest["bindings"]["data"])
    rules = {
        key: CONFIG[key]
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
    protocol = {
        "name": "MIM-GIVEBACK-1",
        "frozen_at": frozen.isoformat(),
        "deadline": (
            frozen + pd.DateOffset(months=CONFIG["horizon_months"])
        ).isoformat(),
        "endpoint_sessions": CONFIG["endpoint_sessions"],
        "selected_threshold": float(verdict["selected_threshold"]),
        "target_pf": CONFIG["target_pf"],
        "profit_retention": CONFIG["profit_retention"],
        "primary_ci": "one-sided 95% stationary bootstrap, mean block 5",
        "sensitivity_blocks": CONFIG["bootstrap_blocks"],
        "one_parameter_only": True,
        "no_interim_efficacy": True,
        "deployment_authorized": False,
        "sweep_completion_hash": artifacts.digest(sweep_run / "completion.json"),
        "history_data": artifacts.portable_path(data_path),
        "history_data_hash": artifacts.digest(data_path),
        "rules": rules,
    }
    artifacts.json_write(path / "protocol.json", protocol)
    (path / "preregistration.md").write_text(
        "# MIM-GIVEBACK-1 prospective preregistration\n\n"
        + json.dumps(protocol, indent=2, sort_keys=True)
        + "\n\nThis artifact must be committed before collection. "
        + "Support authorizes further validation only.\n",
        encoding="utf-8",
    )
    write_report(
        path,
        "MIM Giveback Prospective Freeze",
        [
            ("Protocol", protocol),
            (
                "Next gate",
                "Commit this sealed run before collecting future observations.",
            ),
        ],
    )


def collect(path: Path, protocol_run: Path, data: Path, prior_run: Path | None) -> None:
    observations, eligible, status = collect_future(protocol_run, data, prior_run)
    observations.to_csv(path / "observations.csv", index=False)
    eligible.to_csv(path / "eligibility.csv", index=False)
    artifacts.json_write(path / "collection_status.json", status)
    write_report(
        path,
        "MIM Giveback Prospective Collection",
        [
            ("Coverage", status),
            (
                "Efficacy",
                "No PF, P&L, or candidate-versus-baseline result was calculated.",
            ),
        ],
    )


def evaluate(path: Path, collection_run: Path, protocol_run: Path) -> None:
    result, payload = evaluate_future(protocol_run, collection_run)
    artifacts.json_write(path / "final_verdict.json", result)
    if payload is not None:
        for frame, name in zip(
            payload,
            [
                "baseline_daily.csv",
                "candidate_daily.csv",
                "candidate_ledger.csv",
                "candidate_decisions.csv",
                "candidate_trades.csv",
            ],
        ):
            frame.to_csv(path / name, index=False)
        paired = payload[0][["day", "net"]].rename(columns={"net": "baseline_net"})
        paired["candidate_net"] = payload[1].net.to_numpy(float)
        paired["candidate_minus_baseline"] = paired.candidate_net - paired.baseline_net
        paired.to_csv(path / "paired_daily.csv", index=False)
    write_report(path, "MIM Giveback Prospective Evaluation", [("Verdict", result)])


def verify_semantics(target: Path) -> dict[str, object]:
    manifest = artifacts.verify_run(target)
    command = manifest["command"]
    if command in {"power", "inventory", "sweep", "freeze"}:
        _require_frozen_config(manifest)
    if (target / "failure.json").exists():
        failure = json.loads((target / "failure.json").read_text())
        expected = False
        if command == "inventory" and "terminal" in failure.get("error", ""):
            parent = artifacts.resolve_path(manifest["bindings"]["power_run"])
            parent_manifest = artifacts.verify_run(parent)
            verdict = json.loads((parent / "power_verdict.json").read_text())
            expected = (
                parent_manifest["command"] == "power"
                and verdict["verdict"] != "POWERED"
            )
        if not expected:
            raise ValueError("Failed-run cause is not independently verified")
        return {
            "verified": True,
            "command": command,
            "failed_run_evidence": True,
            "expected_failure": True,
        }
    if command == "power":
        baseline = pd.read_csv(target / "baseline_daily.csv")
        trades = pd.read_csv(target / "baseline_trades.csv")
        summary = performance(trades, baseline)
        verdict = json.loads((target / "power_verdict.json").read_text())
        null = pd.read_csv(target / "mismatched_null_daily.csv")
        recorded_power = pd.read_csv(target / "power_by_threshold.csv")
        recalculated_power, recalculated_verdict = power_from_null(
            null, verdict["minimum_effect_usd_per_session"]
        )
        source_run = artifacts.resolve_path(manifest["bindings"]["source_run"])
        data_path = artifacts.resolve_path(manifest["bindings"]["data"])
        source = load_source(source_run, data_path)
        regenerated = run_power(
            prepare_sessions(data_path), source, threshold_family(source["marks"])
        )
        null_matches = list(null.columns) == list(
            regenerated["null"].columns
        ) and np.allclose(
            null.select_dtypes(include="number"),
            regenerated["null"].select_dtypes(include="number"),
            atol=1e-12,
            rtol=0,
        )
        if (
            len(baseline) != CONFIG["baseline_sessions"]
            or len(trades) != CONFIG["baseline_trades"]
            or not np.isclose(summary["net"], CONFIG["baseline_net"], atol=1e-8, rtol=0)
            or null["shift"].mod(CONFIG["baseline_sessions"]).eq(0).any()
            or verdict["identity_pairings_evaluated"] != 0
            or verdict["aligned_candidate_returns_evaluated"]
            or verdict != recalculated_verdict
            or verdict != regenerated["verdict"]
            or not null_matches
            or not np.allclose(
                recorded_power.select_dtypes(include="number"),
                recalculated_power.select_dtypes(include="number"),
                atol=1e-12,
                rtol=0,
            )
        ):
            raise ValueError("Independent power verification failed")
    elif command == "inventory":
        family = pd.read_csv(target / "threshold_inventory.csv")
        if len(family) != 9 or family.candidate_return_calculated.any():
            raise ValueError("Independent inventory verification failed")
    elif command == "sweep":
        table = pd.read_csv(target / "sweep_results.csv")
        verdict = json.loads((target / "development_verdict.json").read_text())
        selected = (
            table.loc[table.qualifies, "threshold"].max()
            if table.qualifies.any()
            else None
        )
        if (selected is None) != (verdict["selected_threshold"] is None) or (
            selected is not None
            and not np.isclose(selected, verdict["selected_threshold"])
        ):
            raise ValueError("Independent sweep selection verification failed")
    elif command == "freeze":
        protocol = json.loads((target / "protocol.json").read_text())
        if (
            protocol["endpoint_sessions"] != 500
            or protocol["target_pf"] != 1.4
            or not protocol["one_parameter_only"]
        ):
            raise ValueError("Independent protocol verification failed")
    elif command == "collect":
        from pandas.testing import assert_frame_equal

        status = json.loads((target / "collection_status.json").read_text())
        protocol = artifacts.resolve_path(manifest["bindings"]["protocol"])
        frozen = json.loads((protocol / "protocol.json").read_text())
        history = artifacts.resolve_path(frozen["history_data"])
        expected = future_eligibility(pd.read_csv(target / "observations.csv"), history)
        actual = pd.read_csv(target / "eligibility.csv")
        if status["efficacy_calculated"]:
            raise ValueError("Independent collection verification failed")
        try:
            assert_frame_equal(
                expected.fillna(""), actual.fillna(""), check_dtype=False
            )
        except AssertionError as exc:
            raise ValueError("Independent collection verification failed") from exc
    elif command == "evaluate":
        verdict = json.loads((target / "final_verdict.json").read_text())
        if (
            not verdict["efficacy_calculated"]
            and (target / "paired_daily.csv").exists()
        ):
            raise ValueError("Incomplete evaluation leaked efficacy")
        if verdict["efficacy_calculated"]:
            required = {
                "baseline_daily.csv",
                "candidate_daily.csv",
                "candidate_trades.csv",
                "paired_daily.csv",
            }
            if not all((target / name).exists() for name in required):
                raise ValueError("Complete evaluation is missing efficacy artifacts")
            paired = pd.read_csv(target / "paired_daily.csv")
            baseline_daily = pd.read_csv(target / "baseline_daily.csv")
            candidate_daily = pd.read_csv(target / "candidate_daily.csv")
            pd.read_csv(target / "candidate_trades.csv")
            if (
                len(paired) != verdict["eligible_sessions"]
                or not np.allclose(
                    paired.candidate_minus_baseline,
                    paired.candidate_net - paired.baseline_net,
                    atol=1e-12,
                    rtol=0,
                )
                or not np.allclose(
                    baseline_daily.net, paired.baseline_net, atol=1e-12, rtol=0
                )
                or not np.allclose(
                    candidate_daily.net, paired.candidate_net, atol=1e-12, rtol=0
                )
            ):
                raise ValueError("Evaluation paired outcomes do not reconcile")
            blocks = sorted(verdict["paired_ci"], key=int)
            protocol = artifacts.resolve_path(manifest["bindings"]["protocol"])
            rules = json.loads((protocol / "protocol.json").read_text())["rules"]
            expected_gates = {
                "pf": verdict["candidate"]["pf_infinite"]
                or (verdict["candidate"]["pf"] or 0) >= rules["target_pf"],
                "net_retention": verdict["candidate"]["net"]
                >= rules["profit_retention"] * verdict["baseline"]["net"],
                "paired_lower95": verdict["paired_ci"][blocks[0]]["lower95"] > 0,
                "sensitivity_consistent": all(
                    verdict["paired_ci"][block]["lower95"] > 0 for block in blocks[1:]
                ),
            }
            expected_verdict = "SUPPORT" if all(expected_gates.values()) else "FAIL"
            if (
                verdict["gates"] != expected_gates
                or verdict["verdict"] != expected_verdict
            ):
                raise ValueError("Independent evaluation gate verification failed")
    elif command != "verify":
        raise ValueError("Unknown run command")
    return {"verified": True, "command": command, "failed_run_evidence": False}
