import argparse
import json
from pathlib import Path
from .adapter import PACKAGE, sha, collect, launch
from .sandbox import install_socket_filter


def main():
    parser = argparse.ArgumentParser(
        description="Finite isolated log-inferred contract adapter"
    )
    parser.add_argument("--bars")
    parser.add_argument("--log")
    parser.add_argument("--log-timezone", choices=["UTC"])
    parser.add_argument("--state")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        # Worker is usable only in the fixed namespace layout created by the launcher.
        if str(PACKAGE) != "/code/research/mim_comparison/feed_adapter":
            parser.error("worker requires isolated namespace")
        frozen = json.loads(Path("/state/freeze.json").read_text())
        if frozen["sources"] != {
            p.name: sha(p.read_bytes()) for p in sorted(PACKAGE.glob("*.py"))
        }:
            raise ValueError("sandbox adapter source drift")
        install_socket_filter()
        result = collect(
            {"bars": "/inputs/bars.csv", "log": "/inputs/log", "state": "/state"}
        )
    else:
        if not all((args.bars, args.log, args.state, args.log_timezone)):
            parser.error("--bars, --log, --log-timezone UTC and --state are required")
        result = launch(args.bars, args.log, args.state, args.log_timezone)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
