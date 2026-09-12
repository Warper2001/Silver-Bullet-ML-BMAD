"""Research-only commands; no broker imports, orders or production writes."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
from . import artifacts
from .study import execute, reconcile_accounting


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "run"):
        p = sub.add_parser(command)
        p.add_argument("--data", type=Path, default=artifacts.DATA)
        p.add_argument("--baseline-run", type=Path, default=artifacts.BASELINE)
    p = sub.add_parser("evaluate")
    p.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    path = None
    try:
        inputs = (
            [args.data, args.baseline_run / "completion.json"]
            if args.command != "evaluate"
            else [args.run / "completion.json"]
        )
        bindings = (
            dict(
                data=str(args.data.resolve()),
                baseline_run=str(args.baseline_run.resolve()),
            )
            if args.command != "evaluate"
            else dict(run=str(args.run.resolve()))
        )
        path = artifacts.create(args.command, inputs, bindings)
        if args.command == "evaluate":
            original = artifacts.verify(args.run, complete=True)
            if original["command"] != "run" or (args.run / "failure.json").exists():
                raise ValueError("Evaluation requires a successful full experiment run")
            # Re-verify the historical data and baseline as well as experiment artifacts.
            artifacts.verify_baseline(
                Path(original["bindings"]["data"]),
                Path(original["bindings"]["baseline_run"]),
            )
            for name in (
                "daily.csv",
                "ledger.csv",
                "decisions.csv",
                "trades.csv",
                "features.csv",
                "paired-daily.csv",
                "exclusions.csv",
                "audit.json",
                "baseline-reconciliation.json",
            ):
                with open(args.run / name, "rb") as src, open(path / name, "xb") as dst:
                    shutil.copyfileobj(src, dst)
            daily = pd.read_csv(path / "daily.csv", float_precision="round_trip")
            ledger = pd.read_csv(path / "ledger.csv", float_precision="round_trip")
            trades = pd.read_csv(path / "trades.csv", float_precision="round_trip")
            reconcile_accounting(daily, ledger, trades)
            from .report import generate

            generate(path, daily, trades)
            artifacts.verify(args.run, complete=True)
        else:
            execute(path, args.data, args.baseline_run, args.command == "audit")
        artifacts.verify(path)
        artifacts.seal(path)
        print(path, flush=True)
        return 0
    except Exception as exc:
        if path is not None:
            if not (path / "failure.json").exists():
                artifacts.write_json(
                    path / "failure.json",
                    dict(
                        error=str(exc),
                        type=type(exc).__name__,
                        deployment_authorized=False,
                    ),
                )
            if not (path / "report.md").exists():
                with open(path / "report.md", "x") as out:
                    out.write("# Failed research invocation\n\n" + str(exc) + "\n")
            if not (path / "completion.json").exists():
                artifacts.seal(path)
        print(str(path) + ": " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
