"""Build the fixed purchased MNQM5 pilot and conditionally replay frozen arms."""

import argparse
import importlib.util
from pathlib import Path
import signal
import sys
import types


def load_runner():
    name = "_yank_native_minute"
    path = Path(__file__).resolve().parents[1] / "research/yank_native_minute"
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(name + ".runner", path / "runner.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", required=True, help="Fresh output directory; never overwritten"
    )
    args = parser.parse_args(argv)

    def interrupted(signum, frame):
        raise KeyboardInterrupt("signal " + str(signum))

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        report = load_runner().run(args.output_dir)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"FAIL_NATIVE_MINUTE: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
    print(
        f"{report['status']}; replay={report['replay_outcome']}; research_status=HOLD_VALIDATION"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
