"""Integration test: the Part B synthetic battery over the real 2026-06-22 window.

``@pytest.mark.integration`` -- folds the uncommitted GLBX MDP3 capture
``data/tick/_test/glbx-mdp3-20260622.mbo.dbn.zst``.  Skips when the capture is
absent unless ``TICKSIM_REQUIRE_FIXTURE`` is set (same convention as
``tests/integration/test_ticksim_parity_run_part_a_integration.py``).

This is the real end-to-end Part B smoke test: :func:`generate_synthetic_orders`
draws ``n = 1000`` orders over a front-month-filtered slice of the capture, then
:func:`run_part_b` simulates them and every one of the six invariants must hold
(``verdict == "PASS"``, ``violations == ()``).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.ticksim.events import BookEvent, DbnMboSource
from src.ticksim.parity.part_b import run_part_b
from src.ticksim.parity.synthetic import generate_synthetic_orders

pytestmark = pytest.mark.integration

WINDOW = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "tick"
    / "_test"
    / "glbx-mdp3-20260622.mbo.dbn.zst"
)

_NS_PER_MINUTE = 60 * 1_000_000_000
_SCAN = 400_000  # records to scan for the front-month id + a start anchor


def _window_or_skip() -> Path:
    if WINDOW.is_file():
        return WINDOW
    if os.environ.get("TICKSIM_REQUIRE_FIXTURE"):
        pytest.fail(f"TICKSIM_REQUIRE_FIXTURE set but window missing: {WINDOW}")
    pytest.skip(f"2026-06-22 MBO window not present: {WINDOW}")


class _FrontMonthSource:
    """Re-iterable single-instrument view of ``DbnMboSource`` over a ts range."""

    class_rank = 0

    def __init__(self, path: Path, instrument_id: int, lo_ns: int, hi_ns: int) -> None:
        self._inner = DbnMboSource(path)
        self._iid = instrument_id
        self._lo = lo_ns
        self._hi = hi_ns

    def __iter__(self) -> Iterator[BookEvent]:
        for ev in self._inner:
            if ev.instrument_id != self._iid:
                continue
            if ev.ts_event < self._lo:
                continue
            if ev.ts_event > self._hi:
                break
            yield ev


def _anchor(path: Path) -> tuple[int, int]:
    """``(front_month_iid, first_front_month_ts_ns)`` from the first slice."""
    counts: dict[int, int] = {}
    for i, ev in enumerate(DbnMboSource(path)):
        counts[ev.instrument_id] = counts.get(ev.instrument_id, 0) + 1
        if i >= _SCAN:
            break
    iid = max(counts, key=lambda k: counts[k])
    first_ts: int | None = None
    for ev in DbnMboSource(path):
        if ev.instrument_id == iid:
            first_ts = ev.ts_event
            break
    assert first_ts is not None, "capture yielded no front-month records"
    return iid, first_ts


def test_part_b_battery_over_the_2026_06_22_window() -> None:
    path = _window_or_skip()
    iid, first_ts = _anchor(path)

    # a 20-minute submit window, opened 5 min into the front-month stream. Both
    # sources start at `first_ts` so the generator's BookReplay warms the book
    # over the same 5-min lead-in the sim uses -- it just clips candidate draws
    # to [lo_ns, hi_ns) via the generator's own bounds.
    lo_ns = first_ts + 5 * _NS_PER_MINUTE
    hi_ns = lo_ns + 20 * _NS_PER_MINUTE
    gen_source = _FrontMonthSource(path, iid, first_ts, hi_ns)
    sim_source = _FrontMonthSource(path, iid, first_ts, hi_ns + 15 * _NS_PER_MINUTE)

    orders = generate_synthetic_orders(gen_source, lo_ns, hi_ns, n=1000)
    assert len(orders) == 1000

    result = run_part_b(orders, sim_source)

    assert result.verdict == "PASS", result.reason
    assert result.violations == ()
    assert result.n_orders == 1000
