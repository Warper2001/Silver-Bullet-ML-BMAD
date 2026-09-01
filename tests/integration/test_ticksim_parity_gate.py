"""Integration test: ``run_parity_gate`` over the real 2026-06-22 MBO window.

``@pytest.mark.integration`` -- folds the uncommitted GLBX MDP3 capture
``data/tick/_test/glbx-mdp3-20260622.mbo.dbn.zst`` (the one window on hand). Skips
when the capture is absent unless ``TICKSIM_REQUIRE_FIXTURE`` is set (same
convention as ``test_ticksim_parity_run_part_a_integration.py``).

This is a smoke assertion, not a parity verdict: 3 yank trades is far below
``PART_A_MIN_N`` so Part A is a guaranteed N-floor FAIL. What is checked is that
:func:`run_parity_gate` completes end to end against a real book + a real
1000-order Part B battery + a real integrity preflight and returns a populated
:class:`~src.ticksim.parity.gate_cli.GateRun`.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.ticksim.book import Book, apply_event
from src.ticksim.events import BookEvent, DbnMboSource
from src.ticksim.parity.gate_cli import GateRun, WindowSpec, run_parity_gate
from src.ticksim.parity.part_a import reconstruct_trades_db_row

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
_SCAN = 400_000


def _window_or_skip() -> Path:
    if WINDOW.is_file():
        return WINDOW
    if os.environ.get("TICKSIM_REQUIRE_FIXTURE"):
        pytest.fail(f"TICKSIM_REQUIRE_FIXTURE set but window missing: {WINDOW}")
    pytest.skip(f"2026-06-22 MBO window not present: {WINDOW}")


class _WindowSource:
    """Re-iterable single-instrument, ts-clipped view of ``DbnMboSource``."""

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
            if ev.ts_event >= self._hi:
                break
            yield ev


def _anchor(path: Path) -> tuple[int, int, int]:
    counts: dict[int, int] = {}
    first_ts: int | None = None
    for i, ev in enumerate(DbnMboSource(path)):
        counts[ev.instrument_id] = counts.get(ev.instrument_id, 0) + 1
        if first_ts is None:
            first_ts = ev.ts_event
        if i >= _SCAN:
            break
    assert first_ts is not None
    iid = max(counts, key=lambda k: counts[k])

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
    assert bid is not None and ask is not None
    return iid, anchor_ts, (bid + ask) // 2


def _iso(ts_ns: int) -> str:
    return (
        dt.datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _row(trade_id: int, entry_ts_ns: int, hold_min: int, direction: str, mid: int):
    entry_px = mid / _DBN_PER_INDEX_POINT
    move = 3.0 if direction == "L" else -3.0
    return {
        "id": trade_id,
        "trader_id": "trader-yank",
        "timestamp": _iso(entry_ts_ns),
        "direction": direction,
        "entry_price": round(entry_px, 2),
        "exit_price": round(entry_px + move, 2),
        "exit_timestamp": _iso(entry_ts_ns + hold_min * _NS_PER_MINUTE),
        "metadata": {"contracts": 2},
    }


def test_run_parity_gate_over_the_2026_06_22_window() -> None:
    path = _window_or_skip()
    iid, anchor_ts, mid = _anchor(path)

    lo_ns = anchor_ts - 30 * _NS_PER_MINUTE
    hi_ns = anchor_ts + 60 * _NS_PER_MINUTE
    windows = {"w0": WindowSpec(lo_ns=lo_ns, hi_ns=hi_ns)}

    trades = [
        reconstruct_trades_db_row(
            _row(2900, anchor_ts + 1 * _NS_PER_MINUTE, 5, "L", mid)
        ),
        reconstruct_trades_db_row(
            _row(2901, anchor_ts + 9 * _NS_PER_MINUTE, 4, "S", mid)
        ),
        reconstruct_trades_db_row(
            _row(2902, anchor_ts + 15 * _NS_PER_MINUTE, 6, "L", mid)
        ),
    ]

    def source_for(_key: str) -> _WindowSource:
        return _WindowSource(path, iid, lo_ns, hi_ns)

    run = run_parity_gate(
        trades,
        windows,
        "w0",
        source_for,
        synthetic_seed=0,
        synthetic_n=1000,
        amendment_number=1,
        cycle_number=1,
        sha="0" * 40,
        date="2026-09-01",
    )

    assert isinstance(run, GateRun)
    assert run.part_a.stats.n >= 3
    assert run.part_b.n_orders == 1000
    assert len(run.integrity_reports) == 1
    assert run.verdict.verdict in ("PASS", "FAIL")
    assert "# Amendment 1 -- Parity gate result (cycle 1)" in run.stub
