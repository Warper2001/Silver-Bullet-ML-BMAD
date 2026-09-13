"""Immutable CLI for the gated MIM profit-giveback experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import artifacts, workflow
from .config import DEFAULT_DATA, DEFAULT_SOURCE_RUN


def _resolve_invocation(args) -> tuple[list[Path], dict[str, object]]:
    inputs: list[Path]
    bindings: dict[str, object]
    if args.command == "power":
        inputs = [args.source_run / "completion.json", args.data]
        bindings = {
            "source_run": args.source_run,
            "data": args.data,
        }
    elif args.command == "inventory":
        inputs = [args.power_run / "completion.json"]
        bindings = {"power_run": args.power_run}
    elif args.command == "sweep":
        inputs = [
            args.inventory_run / "completion.json",
            args.source_run / "completion.json",
            args.data,
        ]
        bindings = {
            "inventory_run": args.inventory_run,
            "source_run": args.source_run,
            "data": args.data,
        }
    elif args.command == "freeze":
        sweep_manifest = json.loads((args.sweep_run / "manifest.json").read_text())
        history = artifacts.resolve_path(sweep_manifest["bindings"]["data"])
        inputs = [args.sweep_run / "completion.json", history]
        bindings = {"sweep_run": args.sweep_run, "data": history}
    elif args.command == "collect":
        protocol = json.loads((args.protocol / "protocol.json").read_text())
        history = artifacts.resolve_path(protocol["history_data"])
        inputs = [args.protocol / "completion.json", args.data, history] + (
            [args.prior_run / "completion.json"] if args.prior_run else []
        )
        bindings = {
            "protocol": args.protocol,
            "data": args.data,
            "history_data": history,
            "prior_run": args.prior_run,
        }
    elif args.command == "evaluate":
        if args.protocol is None:
            collection_manifest = json.loads((args.run / "manifest.json").read_text())
            bound = collection_manifest.get("bindings", {}).get("protocol")
            if not bound:
                raise ValueError("Collection manifest does not bind a protocol")
            args.protocol = artifacts.resolve_path(bound)
        protocol = json.loads((args.protocol / "protocol.json").read_text())
        history = artifacts.resolve_path(protocol["history_data"])
        inputs = [
            args.run / "completion.json",
            args.protocol / "completion.json",
            history,
        ]
        bindings = {
            "run": args.run,
            "protocol": args.protocol,
            "history_data": history,
        }
    else:
        inputs = [args.run / "completion.json"]
        bindings = {"run": args.run}
    return inputs, bindings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("power")
    p.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output", type=Path)
    p = sub.add_parser("inventory")
    p.add_argument("--power-run", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p = sub.add_parser("sweep")
    p.add_argument("--inventory-run", type=Path, required=True)
    p.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output", type=Path)
    p = sub.add_parser("freeze")
    p.add_argument("--sweep-run", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p = sub.add_parser("collect")
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--prior-run", type=Path)
    p.add_argument("--output", type=Path)
    p = sub.add_parser("evaluate")
    p.add_argument("--run", type=Path, required=True, help="completed collection run")
    p.add_argument("--protocol", type=Path)
    p.add_argument("--output", type=Path)
    p = sub.add_parser("verify")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    path = None
    try:
        inputs, bindings = _resolve_invocation(args)
        path = artifacts.create(args.command, inputs, args.output, bindings)
        if args.command == "power":
            workflow.power(path, args.source_run, args.data)
        elif args.command == "inventory":
            workflow.inventory(path, args.power_run)
        elif args.command == "sweep":
            workflow.sweep(path, args.inventory_run, args.source_run, args.data)
        elif args.command == "freeze":
            workflow.freeze(path, args.sweep_run)
        elif args.command == "collect":
            workflow.collect(path, args.protocol, args.data, args.prior_run)
        elif args.command == "evaluate":
            workflow.evaluate(path, args.run, args.protocol)
        else:
            result = workflow.verify_semantics(args.run)
            artifacts.json_write(path / "verification.json", result)
            from .report import write_report

            write_report(
                path, "MIM Giveback Independent Verification", [("Result", result)]
            )
        artifacts.seal(path)
        print(path, flush=True)
        return 0
    except Exception as exc:
        if path is None:
            try:
                path = artifacts.create(
                    args.command,
                    [],
                    getattr(args, "output", None),
                    {"invocation_resolution_failed": True},
                )
            except Exception:
                path = None
        if path is not None:
            artifacts.fail(path, exc)
        print(f"{path}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
