"""Output-only command for fixed, offline YANK provenance evidence."""

import argparse
import importlib.util
import sys
from pathlib import Path


def _load_validator():
    # The research package initializer eagerly imports unrelated strategy code.
    # Load only this fixed standard-library module, preserving its real __file__.
    path = (
        Path(__file__).resolve().parents[1]
        / "research/yank_bar_provenance/validator.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_yank_bar_provenance_validator", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load provenance validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run = _load_validator().run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Fresh directory for canonical diagnostic outputs",
    )
    args = parser.parse_args(argv)
    try:
        report = run(args.output_dir)
    except Exception as exc:
        print(f"FAIL_PROVENANCE_CHECKS: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        f"{report['status']}; data_suitability=BLOCKED; research_status=HOLD_VALIDATION"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
