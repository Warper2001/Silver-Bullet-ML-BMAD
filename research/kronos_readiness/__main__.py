"""Commands: documentary assess, bounded current probe, offline synthetic
timing."""

import argparse
import json
from pathlib import Path
from . import FLAGS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    assess = sub.add_parser("assess")
    assess.add_argument("--documentary-root", type=Path, required=True)
    assess.add_argument("--source-pack", type=Path, required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--plan", type=Path, required=True)
    probe.add_argument("--token-path", type=Path, required=True)
    timing = sub.add_parser("timing")
    timing.add_argument("--cache", type=Path, required=True)
    timing.add_argument("--decisions", type=int, default=3)
    design = sub.add_parser("design")
    design.add_argument("--source-pack", type=Path, required=True)
    design.add_argument("--planning-start", required=True)
    for command in (assess, probe, timing, design):
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "assess":
            from .evidence import assess as assess_run

            report = assess_run(
                args.documentary_root, args.source_pack, args.output_dir
            )
        elif args.command == "design":
            from .design import run as design_run

            report = design_run(
                args.source_pack, args.output_dir, args.planning_start
            )
        elif args.command == "probe":
            from .probe import run as probe_run

            report = probe_run(args.plan, args.token_path, args.output_dir)
        else:
            from .timing import run as timing_run

            report = timing_run(args.output_dir, args.cache, args.decisions)
    except Exception:
        # Do not print exception messages that could contain credentials or
        # payloads.
        print(
            json.dumps(
                {
                    **FLAGS,
                    "status": "REFUSED",
                    "reason": (
                        "invalid input, destination or execution failure"
                    ),
                }
            )
        )
        return 2
    print(json.dumps({**FLAGS, "status": report["status"]}))
    return (
        0
        if report["status"] in {"OBSERVED_CURRENT_ONLY", "TIMING_COMPLETED"}
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
