"""CLI for immutable portfolio PF shortlist evidence: audit, run, verify."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import artifacts as art


def _decay_coverage() -> dict[str, object]:
    frame = pd.read_csv(art.input_path("logs/portfolio_decay_shadow.csv"))
    return {
        "rows": len(frame),
        "observation_timestamps": sorted(
            frame.run_at.dropna().astype(str).unique().tolist()
        ),
        "strategy_rows": {
            str(k): int(v)
            for k, v in frame.strategy.value_counts().sort_index().items()
        },
        "efficacy_interpreted": False,
        "monitor_invoked": False,
    }


def _write_audit(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path / "input_inventory.csv", index=False)
    art.json_write(
        path / "audit.json",
        {
            "valid": True,
            "input_files": len(rows),
            "raw_account_evidence_copied": False,
            "sealed_holdout_accessed": False,
            "strategy_simulator_invoked": False,
            "alternative_strategy_returns_calculated": False,
        },
    )
    art.json_write(path / "decay_monitor_coverage.json", _decay_coverage())


def _write_execution(path: Path, execution: dict[str, object]) -> None:
    execution["events"].to_csv(path / "execution_events.csv", index=False)
    execution["round_trips"].to_csv(path / "execution_round_trips.csv", index=False)
    execution["coverage"].to_csv(path / "execution_coverage.csv", index=False)
    art.json_write(path / "execution_gate.json", execution["gate"])


def _stop_report(path: Path, execution: dict[str, object]) -> None:
    specification = """# Current execution repair specification

This run proved a recurring current adverse economic defect with exact causal evidence. Later MIM and carry feasibility stages were not run. The repair must preserve intended strategy behavior, identify the exact execution component, include rollback and parity checks, and receive separate implementation authorization before any live-code change.
"""
    (path / "repair_specification.md").write_text(specification, encoding="utf-8")
    reasons = "".join(f"<li>{r}</li>" for r in execution["gate"]["reasons"])
    (path / "report.md").write_text(
        "# Portfolio PF shortlist\n\n**CURRENT_REPAIR_CANDIDATE.** Later stages stopped. See `repair_specification.md`.\n",
        encoding="utf-8",
    )
    (path / "report.html").write_text(
        f"<!doctype html><html><body><h1>CURRENT_REPAIR_CANDIDATE</h1><ul>{reasons}</ul></body></html>",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "run"):
        child = sub.add_parser(command)
        child.add_argument("--mim-data", type=Path, default=art.DEFAULT_MIM_DATA)
        child.add_argument("--gap-data", type=Path, default=art.DEFAULT_GAP_DATA)
        child.add_argument(
            "--diagnostic-run", type=Path, default=art.DEFAULT_DIAGNOSTIC_RUN
        )
        child.add_argument("--carry-data", type=Path, default=art.DEFAULT_CARRY_DATA)
        child.add_argument("--output", type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        print(art.verify(args.run))
        return

    path = art.create(args.command, args.output)
    print(path, flush=True)
    try:
        art.validate_requested_paths(
            args.mim_data, args.gap_data, args.diagnostic_run, args.carry_data
        )
        input_rows = art.validate_inputs()
        art.freeze(path, args.command, input_rows)
        _write_audit(path, input_rows)
        if args.command == "run":
            from .execution import build_execution

            execution = build_execution()
            _write_execution(path, execution)
            if execution["gate"]["verdict"] == "CURRENT_REPAIR_CANDIDATE":
                _stop_report(path, execution)
            else:
                from .carry import build_carry
                from .marks import build_marks
                from .report import write_reports

                marks = build_marks()
                marks["trades"].to_csv(path / "mim_trades.csv", index=False)
                marks["marks"].to_csv(path / "mim_decision_marks.csv", index=False)
                marks["lineage"].to_csv(path / "mim_lineage_audit.csv", index=False)
                marks["distributions"].to_csv(
                    path / "mim_path_distributions.csv", index=False
                )
                art.json_write(path / "baseline_summary.json", marks["summary"])
                art.json_write(path / "mim_path_verdict.json", marks["verdict"])
                (path / "mim_hypothesis_specification.md").write_text(
                    marks["hypothesis"], encoding="utf-8"
                )
                carry = build_carry()
                carry["matrix"].to_csv(path / "carry_matrix.csv", index=False)
                carry["evidence"].to_csv(path / "carry_evidence.csv", index=False)
                (path / "carry_questions.md").write_text(
                    carry["questions"], encoding="utf-8"
                )
                art.json_write(path / "carry_verdict.json", carry["verdict"])
                write_reports(path, execution, marks, carry)
        result = art.verify(path, sealed=False)
        manifest = __import__("json").loads((path / "manifest.json").read_text())
        if any(
            art.digest(art.ROOT / relative) != expected
            for relative, expected in manifest["source"].items()
        ):
            raise ValueError("Implementation changed during invocation")
        art.seal(path)
        print(result, flush=True)
    except Exception as exc:
        completion = path / "completion.json"
        if completion.exists():
            # A late sealing error must not leave an apparently successful run.
            path.chmod(0o755)
            completion.chmod(0o644)
            completion.unlink()
        if not (path / "failure.json").exists():
            art.json_write(
                path / "failure.json",
                {
                    "type": type(exc).__name__,
                    "error": str(exc),
                    "observed_at": art.now(),
                },
            )
        art.seal(path)
        raise


if __name__ == "__main__":
    main()
