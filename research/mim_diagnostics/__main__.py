"""CLI: audit, run, verify. All runtime writes are fresh runs children."""

import argparse
from pathlib import Path
import shutil
from . import artifacts as art


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "run"):
        p = sub.add_parser(name)
        p.add_argument("--source-run", type=Path, default=art.SOURCE)
        p.add_argument("--data", type=Path, default=art.DATA)
        p.add_argument("--output", type=Path)
    p = sub.add_parser("verify")
    p.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        print(art.verify(args.run))
        return
    path = art.create(args.command, args.output)
    print(path, flush=True)
    try:
        source, data = art.validate_inputs(args.source_run, args.data)
        from .feasibility import build_inventory, render_inventory

        feasibility, evidence = build_inventory()
        art.freeze(path, args.command, source, data, evidence)
        art.json_write(
            path / "audit.json",
            dict(
                source_completion_sha256=art.digest(source / "completion.json"),
                data_sha256=art.digest(data),
                valid=True,
            ),
        )
        art.json_write(path / "feasibility.json", feasibility)
        (path / "feasibility.md").write_text(render_inventory(feasibility))
        if args.command == "run":
            from .analysis import analyze
            from .report import write_reports

            result = analyze(source, data)
            for name, table in result.items():
                if name == "summary":
                    art.json_write(path / "summary.json", table)
                else:
                    table.to_csv(path / (name + ".csv"), index=False)
            shutil.copyfile(source / "exclusions.csv", path / "exclusions.csv")
            write_reports(path, result, feasibility)
        verified = art.verify(path, sealed=False)
        manifest = __import__("json").loads((path / "manifest.json").read_text())
        if any(
            art.digest(art.ROOT / rel) != h for rel, h in manifest["source"].items()
        ):
            raise ValueError("Implementation changed during invocation")
        art.seal(path)
        print(verified, flush=True)
    except Exception as exc:
        if not (path / "completion.json").exists():
            art.json_write(
                path / "failure.json",
                dict(error=str(exc), type=type(exc).__name__, observed_at=art.now()),
            )
            art.seal(path)
        raise


if __name__ == "__main__":
    main()
