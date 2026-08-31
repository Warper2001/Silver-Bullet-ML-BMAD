"""Integration test: ``run_part_a`` over the real 2026-06-22 MBO window.

``@pytest.mark.integration`` -- folds the uncommitted GLBX MDP3 capture
``data/tick/_test/glbx-mdp3-20260622.mbo.dbn.zst`` (the one window on hand; it
straddles the yank live-combine round trips 2026-06-22). Skips when the capture
is absent unless ``TICKSIM_REQUIRE_FIXTURE`` is set (same convention as
``tests/unit/test_ticksim_events.py``).

The 3 trades mirror ``trades.db`` rows for yank ids 2900-2902 -- reconstructed
via ``reconstruct_trades_db_row`` on hand-built row mappings. Their entry / exit
timestamps and prices are pinned *into* the window discovered from the capture's
own first slice, so the test does not hard-code a calendar time or a price level
that a capture regeneration could invalidate.

This is a smoke assertion, not a parity verdict: N = 3 is far below
``PART_A_MIN_N``, so the verdict is a guaranteed ``FAIL`` on the N floor. What is
checked is that ``run_part_a`` completes end-to-end against a real book and
returns a ``PartAResult`` with >= 3 finite ``FillError``s.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.ticksim.book import Book, apply_event
from src.ticksim.events import BookEvent, DbnMboSource
from src.ticksim.parity.part_a import PartAResult, reconstruct_trades_db_row
from src.ticksim.parity.part_a_runner import run_part_a

pytestmark = pytest.mark.integration

WINDOW = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "tick"
    / "_test"
    / "glbx-mdp3-20260622.mbo.dbn.zst"
)

_NS_PER_MINUTE = 60 * 1_000_000_000
_DBN_PER_INDEX_POINT = 1_000_000_000
_SCAN = 400_000  # records to scan for the front-month id + an anchor tick/price


def _window_or_skip() -> Path:
    if WINDOW.is_file():
        return WINDOW
    if os.environ.get("TICKSIM_REQUIRE_FIXTURE"):
        pytest.fail(f"TICKSIM_REQUIRE_FIXTURE set but window missing: {WINDOW}")
    pytest.skip(f"2026-06-22 MBO window not present: {WINDOW}")


class _WindowSource:
    """Re-iterable single-instrument view of ``DbnMboSource`` over a ts range.

    Stands in for the eventual window-loader: it filters the parent-symbol MBO
    stream to one ``instrument_id`` (front month -- ``MNQ.FUT`` is ~96% front
    month, ~4% spread) and clips to ``[lo_ns, hi_ns]`` so ``simulate`` folds only
    the trade's neighbourhood, not the whole 2.5 h capture.
    """

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


def _anchor(path: Path) -> tuple[int, int, int]:
    """``(front_month_iid, anchor_ts_ns, anchor_mid_dbn)`` from the first slice."""
    counts: dict[int, int] = {}
    first_ts: int | None = None
    for i, ev in enumerate(DbnMboSource(path)):
        counts[ev.instrument_id] = counts.get(ev.instrument_id, 0) + 1
        if first_ts is None:
            first_ts = ev.ts_event
        if i >= _SCAN:
            break
    assert first_ts is not None, "capture yielded no records"
    iid = max(counts, key=lambda k: counts[k])

    # walk the front-month book to ~2 min past the start for a real mid price
    book = Book()
    anchor_ts = first_ts + 2 * _NS_PER_MINUTE
    for ev in DbnMboSource(path):
        if ev.instrument_id != iid:
            continue
        if ev.ts_event > anchor_ts:
            break
        apply_event(book, ev)
    bid = book.best_bid_dbn(iid)
    ask = book.best_ask_dbn(iid)
    assert bid is not None and ask is not None, "no front-month BBO at the anchor"
    return iid, anchor_ts, (bid + ask) // 2


def _row(
    trade_id: int, entry_ts_ns: int, hold_min: int, direction: str, mid_dbn: int
) -> dict[str, object]:
    entry_px = mid_dbn / _DBN_PER_INDEX_POINT
    # a small favourable move so the reconstructed prices look like real fills
    move = 3.0 if direction == "L" else -3.0
    return {
        "id": trade_id,
        "trader_id": "yank",
        "timestamp": _iso(entry_ts_ns),
        "direction": direction,
        "entry_price": round(entry_px, 2),
        "exit_price": round(entry_px + move, 2),
        "exit_timestamp": _iso(entry_ts_ns + hold_min * _NS_PER_MINUTE),
        "metadata": {"contracts": 2},
    }


def _iso(ts_ns: int) -> str:
    import datetime as dt

    return (
        dt.datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def test_run_part_a_over_the_2026_06_22_window() -> None:
    path = _window_or_skip()
    iid, anchor_ts, mid_dbn = _anchor(path)

    rows = [
        _row(2900, anchor_ts + 1 * _NS_PER_MINUTE, 5, "L", mid_dbn),
        _row(2901, anchor_ts + 9 * _NS_PER_MINUTE, 4, "S", mid_dbn),
        _row(2902, anchor_ts + 15 * _NS_PER_MINUTE, 6, "L", mid_dbn),
    ]
    trades = [reconstruct_trades_db_row(r) for r in rows]

    def source_for(trade: object) -> _WindowSource:
        del trade  # every trade sits in the same discovered neighbourhood
        return _WindowSource(
            path, iid, anchor_ts - 30 * _NS_PER_MINUTE, anchor_ts + 60 * _NS_PER_MINUTE
        )

    result = run_part_a(trades, source_for)

    assert isinstance(result, PartAResult)
    assert result.stats.n >= 3
    assert len(result.errors) >= 3
    assert math.isfinite(result.stats.mae_ticks)
    assert math.isfinite(result.stats.p90_ticks)
    assert math.isfinite(result.stats.signed_bias_ticks)
    assert result.verdict == "FAIL"  # N = 3 < PART_A_MIN_N -- N-floor FAIL
    assert result.unresolved_misses == 0


def test_front_month_filter_is_the_callers_job() -> None:
    """An unfiltered (multi-instrument) source makes ``sim`` raise -- proving
    ``run_part_a`` does not silently swallow a mixed stream (spec I/O matrix)."""
    from src.ticksim.sim import IntentLogError

    path = _window_or_skip()
    _iid, anchor_ts, mid_dbn = _anchor(path)
    trade = reconstruct_trades_db_row(
        _row(2900, anchor_ts + 1 * _NS_PER_MINUTE, 5, "L", mid_dbn)
    )

    with pytest.raises(IntentLogError):
        run_part_a([trade], lambda _t: DbnMboSource(path))
