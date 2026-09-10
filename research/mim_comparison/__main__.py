import argparse, json, sys
from pathlib import Path
from .artifacts import make_run, seal, write_json, ensure_outputs


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Isolated MNQ research; no promotion or broker access"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "historical", "shadow"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True)
        p.add_argument("--labels", choices=["start", "end"], required=True)
        if name == "shadow":
            p.add_argument("--warmup", required=True)
            p.add_argument("--historical-run", required=True)
            p.add_argument("--state")
    p = sub.add_parser("evaluate")
    p.add_argument("--shadow-run", required=True)
    args = parser.parse_args(argv)
    config = vars(args)
    inputs = (
        [args.data]
        if hasattr(args, "data")
        else [Path(args.shadow_run) / "status.json"]
    )
    if hasattr(args, "warmup"):
        inputs.append(args.warmup)
    path, protocol = make_run(args.command, inputs, config)
    try:
        if args.command in ("audit", "historical"):
            from .historical import run

            run(args.data, args.labels, path, args.command == "audit")
        elif args.command == "shadow":
            from .shadow import launch

            launch(
                path,
                args.data,
                args.warmup,
                args.labels,
                protocol,
                args.historical_run,
                args.state,
            )
        else:
            from .shadow import evaluate

            evaluate(args.shadow_run, path)
        ensure_outputs(path)
        seal(path)
        print(path)
        return 0
    except Exception as exc:
        write_json(
            path / "failure.json",
            {"error": str(exc), "type": type(exc).__name__, "safe_failure": True},
        )
        ensure_outputs(path)
        seal(path)
        print(str(path) + ": " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
