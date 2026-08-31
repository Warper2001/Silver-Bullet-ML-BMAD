"""Unit tests for ``src/ticksim/parity/_bookwalk.py`` (spine AD-17).

Every case runs against a hand-built in-memory :class:`BookEventSource` (a list
of :class:`BookEvent`s). The three ``part_a_runner._touch_at`` tests in
``test_ticksim_parity_run_part_a.py`` also exercise ``BookReplay`` through the
wrapper -- these pin the object directly.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from src.ticksim.book import BookSide
from src.ticksim.config import MNQ_TICK_DBN
from src.ticksim.events import BookEvent, MboAction, MboSide
from src.ticksim.parity._bookwalk import BookReplay, BookWalkError

IID = 42004800
TICK = MNQ_TICK_DBN
P = 20_000_000_000_000
B = 1_700_000_000 * 1_000_000_000


class ListSource:
    """Re-iterable in-memory :class:`BookEventSource` (spine AD-18)."""

    class_rank = 0

    def __init__(self, events: list[BookEvent]) -> None:
        self._events = list(events)

    def __iter__(self) -> Iterator[BookEvent]:
        return iter(self._events)


def be(
    action: MboAction,
    side: MboSide,
    order_id: int,
    price_dbn: int,
    size: int,
    ts: int,
    seq: int,
    *,
    instrument_id: int = IID,
) -> BookEvent:
    return BookEvent(
        action=action,
        side=side,
        order_id=order_id,
        price_dbn=price_dbn,
        size=size,
        ts_event=ts,
        sequence=seq,
        instrument_id=instrument_id,
    )


# --------------------------------------------------------------------------- #
# advance_to folds <= ts and stops
# --------------------------------------------------------------------------- #


def test_advance_folds_up_to_and_including_cutoff_and_stops() -> None:
    events = [
        be(MboAction.ADD, MboSide.BID, 1, P, 10, ts=B, seq=1),
        be(MboAction.ADD, MboSide.ASK, 2, P + 4 * TICK, 10, ts=B, seq=2),
        # exactly at the cutoff -> folded (inclusive lower bound)
        be(MboAction.ADD, MboSide.BID, 3, P + TICK, 10, ts=B + 1_000, seq=3),
        # past the cutoff -> held, not folded
        be(MboAction.MODIFY, MboSide.ASK, 2, P + 2 * TICK, 10, ts=B + 5_000, seq=4),
    ]
    replay = BookReplay(ListSource(events))
    replay.advance_to(B + 1_000)

    assert replay.instrument_id == IID
    assert replay.book.snapshot_bbo(IID) == (P + TICK, P + 4 * TICK)


def test_advance_is_incremental_across_calls_and_folds_the_held_event() -> None:
    events = [
        be(MboAction.ADD, MboSide.BID, 1, P, 10, ts=B, seq=1),
        be(MboAction.ADD, MboSide.BID, 2, P + TICK, 10, ts=B + 2_000, seq=2),
        be(MboAction.ADD, MboSide.BID, 3, P + 2 * TICK, 10, ts=B + 4_000, seq=3),
    ]
    replay = BookReplay(ListSource(events))

    replay.advance_to(B)
    assert replay.book.best_bid_dbn(IID) == P

    replay.advance_to(B + 1_000)  # folds nothing new (held event is at B + 2_000)
    assert replay.book.best_bid_dbn(IID) == P

    replay.advance_to(B + 3_000)  # folds the held B + 2_000 event, holds B + 4_000
    assert replay.book.best_bid_dbn(IID) == P + TICK

    replay.advance_to(B + 10_000)
    assert replay.book.best_bid_dbn(IID) == P + 2 * TICK


# --------------------------------------------------------------------------- #
# empty-before-first-ts state
# --------------------------------------------------------------------------- #


def test_empty_before_first_event_ts() -> None:
    events = [be(MboAction.ADD, MboSide.BID, 1, P, 10, ts=B + 10_000, seq=1)]
    replay = BookReplay(ListSource(events))
    replay.advance_to(B)

    assert replay.instrument_id is None
    assert replay.book.instruments == {}
    assert replay.book.snapshot_bbo(IID) == (None, None)


def test_advance_past_end_of_source_is_a_noop() -> None:
    events = [be(MboAction.ADD, MboSide.BID, 1, P, 10, ts=B, seq=1)]
    replay = BookReplay(ListSource(events))
    replay.advance_to(B + 1)
    replay.advance_to(B + 10_000_000)  # source exhausted -- must not raise
    assert replay.book.best_bid_dbn(IID) == P


# --------------------------------------------------------------------------- #
# fail-closed guards
# --------------------------------------------------------------------------- #


def test_non_decreasing_cutoff_is_enforced() -> None:
    events = [be(MboAction.ADD, MboSide.BID, 1, P, 10, ts=B, seq=1)]
    replay = BookReplay(ListSource(events))
    replay.advance_to(B + 5_000)
    with pytest.raises(BookWalkError, match="regressed"):
        replay.advance_to(B + 4_999)


def test_source_ts_event_regression_raises() -> None:
    events = [
        be(MboAction.ADD, MboSide.BID, 1, P, 10, ts=B + 1_000, seq=1),
        be(MboAction.ADD, MboSide.BID, 2, P + TICK, 10, ts=B, seq=2),  # goes back
    ]
    replay = BookReplay(ListSource(events))
    with pytest.raises(BookWalkError, match="ts_event regressed"):
        replay.advance_to(B + 10_000)


def test_replay_is_unusable_after_a_failure() -> None:
    events = [be(MboAction.ADD, MboSide.BID, 1, P, 10, ts=B, seq=1)]
    replay = BookReplay(ListSource(events))
    replay.advance_to(B + 5_000)
    with pytest.raises(BookWalkError, match="regressed"):
        replay.advance_to(B)  # cutoff regression -> replay is now broken
    with pytest.raises(BookWalkError, match="unusable after a prior failure"):
        replay.advance_to(B + 10_000)


def test_multi_instrument_stream_raises() -> None:
    events = [
        be(MboAction.ADD, MboSide.BID, 1, P, 10, ts=B, seq=1),
        be(
            MboAction.ADD,
            MboSide.ASK,
            2,
            P + TICK,
            10,
            ts=B + 1_000,
            seq=2,
            instrument_id=IID + 1,
        ),
    ]
    replay = BookReplay(ListSource(events))
    with pytest.raises(BookWalkError, match="multi-instrument"):
        replay.advance_to(B + 10_000)


class BoomSource:
    """A source whose iterator raises a non-``StopIteration`` error mid-stream."""

    class_rank = 0

    def __iter__(self) -> Iterator[BookEvent]:
        def _gen() -> Iterator[BookEvent]:
            yield be(MboAction.ADD, MboSide.BID, 1, P, 10, ts=B, seq=1)
            raise RuntimeError("feed decode exploded")

        return _gen()


def test_non_stopiteration_iterator_error_is_wrapped() -> None:
    replay = BookReplay(BoomSource())
    with pytest.raises(BookWalkError, match="feed decode exploded"):
        replay.advance_to(B + 10_000)


def test_second_instrument_only_after_cutoff_is_not_folded() -> None:
    # the id that only appears past the cutoff must not trip the guard yet.
    events = [
        be(MboAction.ADD, MboSide.BID, 1, P, 10, ts=B, seq=1),
        be(
            MboAction.ADD,
            MboSide.ASK,
            2,
            P + TICK,
            10,
            ts=B + 9_000,
            seq=2,
            instrument_id=IID + 1,
        ),
    ]
    replay = BookReplay(ListSource(events))
    replay.advance_to(B + 1_000)  # only the IID event is folded
    assert replay.instrument_id == IID
    assert replay.book.size_at_price(IID, BookSide.BID, P) == 10
