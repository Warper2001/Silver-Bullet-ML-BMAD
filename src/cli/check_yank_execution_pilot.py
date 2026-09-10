"""Local-only command for the pinned frozen-order execution pilot."""
import argparse
import importlib.util
from pathlib import Path
import types
import sys


def load_audit():
    name = "_yank_execution_pilot"
    path = Path(__file__).resolve().parents[1] / "research/yank_execution_pilot"
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(name + ".audit", path / "audit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    try:
        load_audit().run(args.output_dir)
    except Exception as exc:
        print(f'FAIL_AUDIT_CHECKS: {exc}', file=sys.stderr)
        return 1
    print('PASS_AUDIT_CHECKS; HOLD_VALIDATION')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
