"""Integration test: ``preflight_integrity`` over the real 2026-06-22 MBO window.

``@pytest.mark.integration`` -- folds the uncommitted GLBX MDP3 capture
``data/tick/_test/glbx-mdp3-20260622.mbo.dbn.zst`` front-month subset. Skips
when the capture is absent unless ``TICKSIM_REQUIRE_FIXTURE`` is set (same
convention as ``tests/integration/test_ticksim_parity_run_part_a_integration.py``
and ``tests/unit/test_ticksim_events.py``).

Amendment 9 §A9.3 measured this window: 0 ts violations in 22.5 M records, all
crossed instants transient (2,792 / 19.67 M = 0.014 %), A/C/M/T/F all present.
The preflight should agree: ``verdict == "OK"`` (or a documented known flag),
``missing_actions == ()``, ``persistent_cross_count == 0``, and a transient
cross rate in the same ballpark as 0.014 %.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.ticksim.events import BookEvent, DbnMboSource
from src.ticksim.parity.integrity import preflight_integrity

pytestmark = pytest.mark.integration

WINDOW = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "tick"
    / "_test"
    / "glbx-mdp3-20260622.mbo.dbn.zst"
)

FRONT_MONTH_IID = 42004800  # prereg §A9.4: MNQ front month, 96 % of records


def _window_or_skip() -> Path:
    if WINDOW.is_file():
        return WINDOW
    if os.environ.get("TICKSIM_REQUIRE_FIXTURE"):
        pytest.fail(f"TICKSIM_REQUIRE_FIXTURE set but window missing: {WINDOW}")
    pytest.skip(f"2026-06-22 MBO window not present: {WINDOW}")


class _FrontMonthSource:
    """Re-iterable single-instrument view of ``DbnMboSource`` (front month only)."""

    class_rank = 0

    def __init__(self, path: Path, instrument_id: int) -> None:
        self._inner = DbnMboSource(path)
        self._iid = instrument_id

    def __iter__(self) -> Iterator[BookEvent]:
        for ev in self._inner:
            if ev.instrument_id == self._iid:
                yield ev


def test_preflight_integrity_over_the_2026_06_22_window() -> None:
    path = _window_or_skip()
    report = preflight_integrity(_FrontMonthSource(path, FRONT_MONTH_IID))

    assert report.instrument_id == FRONT_MONTH_IID
    assert report.foreign_instrument_events == 0  # source is front-month filtered
    assert report.n_events > 15_000_000  # ~21.6 M front-month of 22.5 M total
    assert report.ts_regressions == 0  # §A9.3: 0 violations in 22.5 M records
    assert report.missing_actions == ()  # A/C/M/T/F all present (§A9.4)
    assert report.persistent_cross_count == 0  # §A9.3: all crosses transient
    assert report.n_trades > 0
    assert report.session_low_dbn is not None
    assert report.session_high_dbn is not None
    # §A9.3: 0.014 % transient-cross rate -- a loose sanity band
    assert 0 < report.transient_cross_count < report.n_events // 100
    assert report.verdict in ("OK", "FLAGGED")
