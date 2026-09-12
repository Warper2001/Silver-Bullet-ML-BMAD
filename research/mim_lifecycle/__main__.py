"""Run the pinned, descriptive lifecycle diagnostic: python -m research.mim_lifecycle."""

import argparse
import sys

import pandas as pd

from research.mim_comparison.data import load
from research.mim_lifecycle import artifacts as a
from research.mim_lifecycle.analysis import analyze, select_inputs, summarize
from research.mim_lifecycle.report import write_report


def run():
    path = a.create()
    try:
        a.bind(path)
        trades, daily = select_inputs(
            pd.read_csv(a.SOURCE_RUN / "trades.csv", float_precision="round_trip"),
            pd.read_csv(a.SOURCE_RUN / "daily.csv", float_precision="round_trip"),
        )
        bars = load(a.DATA, "end")
        paths, lifecycle, landmarks = analyze(trades, daily, bars)
        summary = summarize(lifecycle, landmarks, daily)
        # Recheck all frozen evidence before publishing a success report.
        a.verify_inputs()
        a.verify(path)
        for name, frame in (
            ("paths.csv", paths),
            ("lifecycle.csv", lifecycle),
            ("landmarks.csv", landmarks),
        ):
            frame.to_csv(path / name, index=False, mode="x")
        a.write_json(path / "summary.json", summary)
        write_report(path, summary, lifecycle)
        a.verify(path)
        a.seal(path)
    except Exception as exc:
        # No report can be mistaken for a successful analysis when failure.json exists.
        if not (path / "completion.json").exists():
            a.fail(path, exc)
        print(f"FAILED {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    print(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
