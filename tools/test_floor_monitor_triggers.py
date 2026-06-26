#!/usr/bin/env python3
"""Synthetic test of the combine floor-monitor HALT decision logic.

Per the 2026-06-17 party-mode ops review (Winston): "a monitor that has silently
stopped, or whose logic has a gap, is worse than no monitor — it manufactures false
confidence. Confirm it actually fires." This proves the halt PREDICATE
(`combine_floor_monitor.evaluate_triggers`) returns the right verdict across the
boundary cases — WITHOUT ever invoking `do_halt`, which would flatten the real
combine account and stop both live bots. We test the pure function only.

Derived thresholds under test (from the sealed joint deployment prereg):
  - DISTANCE_TO_FLOOR: equity <= floor + $500  (binding; checked first)
  - COMBINED_PF: combined PF < 0.70 after >= 30 combined trades

Run:   .venv/bin/python tools/test_floor_monitor_triggers.py
       (also works under pytest)
"""
from __future__ import annotations

import importlib
import sys

m = importlib.import_module("src.research.combine_floor_monitor")
evaluate_triggers = m.evaluate_triggers
HD = m.HALT_DISTANCE      # 500.0
PFT = m.PF_THRESHOLD      # 0.70
PFN = m.PF_MIN_TRADES     # 30

FLOOR = 48_000.0
SAFE_EQUITY = FLOOR + 5_000.0   # plenty of room


def _cases():
    """(name, equity, floor, pf, n_trades, expect_halt_substr_or_None)."""
    return [
        # --- distance-to-floor (binding trigger, checked first) ---
        ("safe: lots of room, healthy PF",      SAFE_EQUITY, FLOOR, 1.50, 50, None),
        ("safe: no trades yet",                 SAFE_EQUITY, FLOOR, None,  0, None),
        ("floor breach: equity below floor",    FLOOR - 100, FLOOR, 1.50, 50, "DISTANCE_TO_FLOOR"),
        ("floor breach: exactly at +$500 (<=)", FLOOR + HD,  FLOOR, 1.50, 50, "DISTANCE_TO_FLOOR"),
        ("floor ok: $1 above the +$500 line",   FLOOR + HD + 1, FLOOR, 1.50, 50, None),

        # --- combined PF trigger (only after >= PF_MIN_TRADES) ---
        ("pf breach: low PF after 30 trades",   SAFE_EQUITY, FLOOR, PFT - 0.01, PFN,     "COMBINED_PF"),
        ("pf ok: low PF but < 30 trades",       SAFE_EQUITY, FLOOR, PFT - 0.01, PFN - 1,  None),
        ("pf ok: exactly at 0.70 (not < thr)",  SAFE_EQUITY, FLOOR, PFT,        PFN,      None),
        ("pf ok: PF None (no losses yet)",      SAFE_EQUITY, FLOOR, None,       PFN,      None),

        # --- priority: distance dominates PF when both would fire ---
        ("priority: floor wins over PF",        FLOOR + 100, FLOOR, 0.10, 99, "DISTANCE_TO_FLOOR"),
    ]


def run() -> int:
    passed = failed = 0
    for name, eq, fl, pf, n, expect in _cases():
        got = evaluate_triggers(eq, fl, pf, n)
        if expect is None:
            ok = got is None
        else:
            ok = got is not None and expect in got
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        shown = f"halt({got!r})" if got else "no-halt"
        print(f"  [{status}] {name:42s} -> {shown}")

    print(f"\nthresholds: HALT_DISTANCE=${HD:.0f}  PF_THRESHOLD={PFT}  PF_MIN_TRADES={PFN}")
    print(f"result: {passed} passed, {failed} failed")
    return 1 if failed else 0


# pytest entrypoint
def test_floor_monitor_triggers():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
