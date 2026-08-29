"""Unit + integration tests for ``src.ticksim.book`` (spine AD-3 / AD-9 / AD-22).

Two layers:
  * every row of the spec's I/O & Edge-Case Matrix plus the input-hardening
    rules, driven by small hand-built :class:`_Rec` stubs that satisfy
    ``book.MboRecord`` structurally;
  * one integration test that folds a prefix of the GLBX MDP3 fixture for the
    MNQ front month and checks the reconstructed book is sane.

``F`` note: the spec assumed ``F`` (fill) reduces the hit resting order.
Verification against the fixture and ``databento_dbn``'s own stubs shows ``F``
does *not* touch the book on GLBX -- the reduction arrives as a following
``C`` / ``M`` (every GLBX ``C`` is in fact a full cancel). The "fill reduces
resting" matrix rows are therefore split into (a) ``F`` is informational and
(b) the ``C`` that follows performs the reduction. See ``book.py``'s docstring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from databento_dbn import Action, Side

from src.ticksim.book import (
    UNDEF_ORDER_SIZE,
    Book,
    BookInconsistency,
    BookSide,
    RestingOrder,
    apply_event,
)
from src.ticksim import book as book_module
from src.ticksim.config import MAX_TRANSIENT_CROSS_NS

# The one source of truth for the fixture's front-month instrument id (the book
# itself is instrument-agnostic, so this magic number lives with the tests).
FRONT_MONTH_INSTRUMENT_ID = 42004800
IID = 1  # default instrument id for the hand-built cases


@dataclass(frozen=True)
class _Rec:
    """Minimal structural stand-in for a normalized MBO record (book.MboRecord).

    ``book.MboRecord`` no longer references any ``databento`` type: ``action`` /
    ``side`` are plain ``str`` single-char codes and the price field is
    ``price_dbn`` (the whole vendor boundary lives in ``events.py``). The
    builders below accept the vendor enums for readability and stringify.
    """

    action: str
    side: str
    order_id: int
    price_dbn: int
    size: int
    ts_event: int
    sequence: int
    instrument_id: int = IID


# --- record builders ------------------------------------------------------


def _add(
    order_id: int,
    side: Side,
    price: int,
    size: int,
    ts_event: int,
    sequence: int,
    instrument_id: int = IID,
) -> _Rec:
    return _Rec(
        str(Action.ADD),
        str(side),
        order_id,
        price,
        size,
        ts_event,
        sequence,
        instrument_id,
    )


def _cancel(
    order_id: int,
    side: Side,
    price: int,
    size: int,
    ts_event: int,
    sequence: int,
    instrument_id: int = IID,
) -> _Rec:
    return _Rec(
        str(Action.CANCEL),
        str(side),
        order_id,
        price,
        size,
        ts_event,
        sequence,
        instrument_id,
    )


def _modify(
    order_id: int,
    side: Side,
    price: int,
    size: int,
    ts_event: int,
    sequence: int,
    instrument_id: int = IID,
) -> _Rec:
    return _Rec(
        str(Action.MODIFY),
        str(side),
        order_id,
        price,
        size,
        ts_event,
        sequence,
        instrument_id,
    )


def _fill(
    order_id: int,
    side: Side,
    price: int,
    size: int,
    ts_event: int,
    sequence: int,
    instrument_id: int = IID,
) -> _Rec:
    return _Rec(
        str(Action.FILL),
        str(side),
        order_id,
        price,
        size,
        ts_event,
        sequence,
        instrument_id,
    )


def _trade(
    price: int,
    size: int,
    ts_event: int,
    sequence: int,
    side: Side = Side.ASK,
    order_id: int = 0,
    instrument_id: int = IID,
) -> _Rec:
    return _Rec(
        str(Action.TRADE),
        str(side),
        order_id,
        price,
        size,
        ts_event,
        sequence,
        instrument_id,
    )


def _clear(ts_event: int, sequence: int, instrument_id: int = IID) -> _Rec:
    return _Rec(
        str(Action.CLEAR), str(Side.NONE), 0, 0, 0, ts_event, sequence, instrument_id
    )


def _none(ts_event: int, sequence: int, instrument_id: int = IID) -> _Rec:
    return _Rec(
        str(Action.NONE), str(Side.NONE), 0, 0, 0, ts_event, sequence, instrument_id
    )


def _fold(book: Book, *records: _Rec) -> None:
    for rec in records:
        apply_event(book, rec)


# --- vendor-enum pins (item 16) -----------------------------------------


@pytest.mark.parametrize(
    "member, code",
    [
        (Action.ADD, "A"),
        (Action.CANCEL, "C"),
        (Action.MODIFY, "M"),
        (Action.TRADE, "T"),
        (Action.FILL, "F"),
        (Action.CLEAR, "R"),
        (Action.NONE, "N"),
    ],
)
def test_action_str_code_is_single_char(member: Action, code: str) -> None:
    # apply_event relies on str(Action.X) == the MBO char; a databento_dbn
    # upgrade that changes __str__ must fail here, not silently mis-fold.
    assert str(member) == code


@pytest.mark.parametrize(
    "member, code",
    [(Side.BID, "B"), (Side.ASK, "A"), (Side.NONE, "N")],
)
def test_side_str_code_is_single_char(member: Side, code: str) -> None:
    assert str(member) == code


# --- I/O & Edge-Case Matrix --------------------------------------------


class TestAddAndTouch:
    def test_add_then_best_bid(self) -> None:
        book = Book()
        _fold(
            book,
            _add(1, Side.BID, 100, 1, ts_event=10, sequence=1),
            _add(2, Side.BID, 101, 1, ts_event=11, sequence=2),
        )
        assert book.best_bid_dbn(IID) == 101

    def test_cancel_top(self) -> None:
        book = Book()
        _fold(
            book,
            _add(1, Side.BID, 100, 1, ts_event=10, sequence=1),
            _add(2, Side.BID, 101, 1, ts_event=11, sequence=2),
            _cancel(2, Side.BID, 101, 1, ts_event=12, sequence=3),
        )
        assert book.best_bid_dbn(IID) == 100
        assert book.order_by_id(IID, 2) is None
        book.check_invariants()

    def test_empty_side_queries(self) -> None:
        book = Book()
        apply_event(book, _add(1, Side.BID, 100, 1, ts_event=10, sequence=1))
        assert book.best_ask_dbn(IID) is None
        assert book.snapshot_bbo(IID) == (100, None)
        assert book.size_at_price(IID, BookSide.ASK, 100) == 0

    def test_queries_on_unknown_instrument(self) -> None:
        book = Book()
        assert book.best_bid_dbn(999) is None
        assert book.best_ask_dbn(999) is None
        assert book.snapshot_bbo(999) == (None, None)
        assert book.size_at_price(999, BookSide.BID, 100) == 0
        assert book.order_by_id(999, 1) is None
        assert book.queue_ahead_size(999, BookSide.BID, 100, 10) == 0

    def test_query_on_known_instrument_other_side_or_price(self) -> None:
        book = Book()
        _fold(book, _add(1, Side.BID, 100, 4, ts_event=1, sequence=1))
        # instrument exists, but the ask side / a different bid price are empty
        assert book.size_at_price(IID, BookSide.ASK, 100) == 0
        assert book.size_at_price(IID, BookSide.BID, 99) == 0
        assert book.queue_ahead_size(IID, BookSide.ASK, 100, 10) == 0
        assert book.queue_ahead_size(IID, BookSide.BID, 99, 10) == 0


class TestModify:
    def test_modify_size_in_place_keeps_priority(self) -> None:
        book = Book()
        _fold(
            book,
            _add(1, Side.BID, 100, 5, ts_event=10, sequence=1),
            _modify(1, Side.BID, 100, 3, ts_event=20, sequence=9),
        )
        assert book.size_at_price(IID, BookSide.BID, 100) == 3
        order = book.order_by_id(IID, 1)
        assert order is not None
        assert order.add_ts_ns == 10  # unchanged
        assert order.sequence == 1  # unchanged
        assert order.size == 3
        book.check_invariants()

    def test_modify_price_takes_new_key(self) -> None:
        book = Book()
        _fold(
            book,
            _add(1, Side.BID, 100, 4, ts_event=10, sequence=1),
            _modify(1, Side.BID, 99, 4, ts_event=20, sequence=9),
        )
        assert book.size_at_price(IID, BookSide.BID, 100) == 0
        order = book.order_by_id(IID, 1)
        assert order is not None
        assert order.price_dbn == 99
        assert order.add_ts_ns == 20  # M record's ts_event
        assert order.sequence == 9  # M record's sequence

    def test_modify_size_to_zero_removes_order(self) -> None:
        book = Book()
        _fold(
            book,
            _add(1, Side.BID, 100, 4, ts_event=10, sequence=1),
            _modify(1, Side.BID, 100, 0, ts_event=20, sequence=9),
        )
        assert book.order_by_id(IID, 1) is None
        assert book.best_bid_dbn(IID) is None

    def test_unseen_modify_is_noop_and_counted(self) -> None:
        book = Book()
        apply_event(book, _modify(999, Side.BID, 100, 1, ts_event=10, sequence=1))
        assert book.unseen_cm_count == 1
        assert book.best_bid_dbn(IID) is None

    def test_unseen_price_change_modify_is_noop_and_counted(self) -> None:
        book = Book()
        _fold(book, _add(1, Side.BID, 100, 2, ts_event=1, sequence=1))
        # a price-change M for an order id we've never seen
        apply_event(book, _modify(777, Side.BID, 98, 3, ts_event=2, sequence=2))
        assert book.unseen_cm_count == 1
        assert book.size_at_price(IID, BookSide.BID, 98) == 0
        assert book.best_bid_dbn(IID) == 100

    def test_modify_undef_size_keeps_size(self) -> None:
        book = Book()
        _fold(
            book,
            _add(1, Side.BID, 100, 5, ts_event=1, sequence=1),
            _modify(1, Side.BID, 100, UNDEF_ORDER_SIZE, ts_event=2, sequence=2),
        )
        assert book.size_at_price(IID, BookSide.BID, 100) == 5

    def test_modify_side_flip_raises(self) -> None:
        book = Book()
        _fold(book, _add(1, Side.BID, 100, 5, ts_event=1, sequence=1))
        with pytest.raises(BookInconsistency):
            apply_event(book, _modify(1, Side.ASK, 100, 5, ts_event=2, sequence=2))


class TestFillIsInformational:
    """``F`` does not mutate the book on GLBX (see book.py docstring)."""

    def test_fill_does_not_reduce_resting(self) -> None:
        book = Book()
        _fold(
            book,
            _add(1, Side.ASK, 100, 4, ts_event=10, sequence=1),
            _fill(1, Side.ASK, 100, 3, ts_event=11, sequence=2),
        )
        # F is informational -- size unchanged.
        assert book.size_at_price(IID, BookSide.ASK, 100) == 4
        order = book.order_by_id(IID, 1)
        assert order is not None and order.size == 4

    def test_fill_then_modify_reduces_resting(self) -> None:
        # The real GLBX pattern (order 6878505599343): a partial fill is
        # reflected by an M carrying the new absolute size. "Fill reduces
        # resting -> size 1" matrix row, performed by the M that actually
        # mutates.
        book = Book()
        _fold(
            book,
            _add(1, Side.ASK, 100, 4, ts_event=10, sequence=1),
            _fill(1, Side.ASK, 100, 3, ts_event=11, sequence=2),
            _modify(1, Side.ASK, 100, 1, ts_event=12, sequence=3),
        )
        assert book.size_at_price(IID, BookSide.ASK, 100) == 1
        order = book.order_by_id(IID, 1)
        assert order is not None and order.size == 1
        assert order.add_ts_ns == 10  # same-price M keeps priority
        book.check_invariants()

    def test_fill_to_zero_via_full_cancel(self) -> None:
        # "Fill to zero" matrix row. GLBX sends the trailing C at the *full*
        # resting size when there is no intervening M (order 6878505593200:
        # A sz3 -> F sz1 x3 -> C sz3).
        book = Book()
        _fold(
            book,
            _add(1, Side.ASK, 100, 3, ts_event=10, sequence=1),
            _fill(1, Side.ASK, 100, 1, ts_event=11, sequence=2),
            _fill(1, Side.ASK, 100, 1, ts_event=11, sequence=3),
            _fill(1, Side.ASK, 100, 1, ts_event=11, sequence=4),
            _cancel(1, Side.ASK, 100, 3, ts_event=12, sequence=5),
        )
        assert book.best_ask_dbn(IID) is None
        assert book.order_by_id(IID, 1) is None

    def test_partial_reduction_delta_model(self) -> None:
        # Defensive: GLBX never sends a partial C, but a future events.py source
        # might. Pin the delta interpretation: new_size = existing - record.size.
        book = Book()
        _fold(
            book,
            _add(1, Side.ASK, 100, 4, ts_event=10, sequence=1),
            _cancel(1, Side.ASK, 100, 3, ts_event=12, sequence=3),
        )
        assert book.size_at_price(IID, BookSide.ASK, 100) == 1
        book.check_invariants()

    def test_unseen_fill_is_a_silent_noop(self) -> None:
        book = Book()
        apply_event(book, _fill(999, Side.ASK, 100, 1, ts_event=10, sequence=1))
        # F never looks the order up, so an "unseen F" is not counted.
        assert book.unseen_cm_count == 0

    def test_trade_that_would_cross_is_still_a_noop(self) -> None:
        book = Book()
        _fold(
            book,
            _add(1, Side.BID, 100, 3, ts_event=1, sequence=1),
            _add(2, Side.ASK, 102, 3, ts_event=2, sequence=2),
        )
        before = book.snapshot_bbo(IID)
        # a trade printed *inside* the spread / through a side -- pure no-op
        apply_event(book, _trade(101, 5, ts_event=3, sequence=3, side=Side.BID))
        assert book.snapshot_bbo(IID) == before
        assert book.max_transient_cross_ns == 0


class TestCancel:
    def test_partial_cancel_reduces_level(self) -> None:
        book = Book()
        _fold(
            book,
            _add(1, Side.BID, 100, 5, ts_event=10, sequence=1),
            _cancel(1, Side.BID, 100, 2, ts_event=11, sequence=2),
        )
        assert book.size_at_price(IID, BookSide.BID, 100) == 3
        book.check_invariants()

    def test_cancel_undef_size_is_full_cancel(self) -> None:
        book = Book()
        _fold(
            book,
            _add(1, Side.BID, 100, 7, ts_event=1, sequence=1),
            _cancel(1, Side.BID, 100, UNDEF_ORDER_SIZE, ts_event=2, sequence=2),
        )
        assert book.order_by_id(IID, 1) is None
        assert book.best_bid_dbn(IID) is None

    def test_unseen_cancel_is_noop_and_counted(self) -> None:
        book = Book()
        apply_event(book, _cancel(999, Side.BID, 100, 1, ts_event=10, sequence=1))
        assert book.unseen_cm_count == 1
        assert book.best_bid_dbn(IID) is None

    def test_over_cancel_removes_order_and_counts(self) -> None:
        book = Book()
        _fold(
            book,
            _add(1, Side.BID, 100, 2, ts_event=10, sequence=1),
            _cancel(1, Side.BID, 100, 5, ts_event=11, sequence=2),
        )
        assert book.order_by_id(IID, 1) is None
        assert book.size_at_price(IID, BookSide.BID, 100) == 0
        assert book.best_bid_dbn(IID) is None
        assert book.overcancel_count == 1
        book.check_invariants()


class TestAddHardening:
    def test_duplicate_add_raises(self) -> None:
        book = Book()
        _fold(book, _add(1, Side.BID, 100, 2, ts_event=1, sequence=1))
        with pytest.raises(BookInconsistency):
            apply_event(book, _add(1, Side.BID, 100, 3, ts_event=2, sequence=2))

    def test_add_undef_size_raises(self) -> None:
        book = Book()
        with pytest.raises(BookInconsistency):
            apply_event(
                book, _add(1, Side.BID, 100, UNDEF_ORDER_SIZE, ts_event=1, sequence=1)
            )

    def test_add_zero_size_raises(self) -> None:
        book = Book()
        with pytest.raises(BookInconsistency):
            apply_event(book, _add(1, Side.BID, 100, 0, ts_event=1, sequence=1))

    def test_add_without_market_side_raises(self) -> None:
        book = Book()
        with pytest.raises(BookInconsistency):
            apply_event(book, _add(1, Side.NONE, 100, 1, ts_event=1, sequence=1))


class TestMonotonicClock:
    def test_backwards_ts_event_raises(self) -> None:
        book = Book()
        _fold(book, _add(1, Side.BID, 100, 1, ts_event=100, sequence=1))
        with pytest.raises(BookInconsistency):
            apply_event(book, _add(2, Side.BID, 100, 1, ts_event=99, sequence=2))

    def test_equal_ts_event_is_allowed(self) -> None:
        book = Book()
        _fold(
            book,
            _add(1, Side.BID, 100, 1, ts_event=100, sequence=1),
            _add(2, Side.BID, 101, 1, ts_event=100, sequence=2),
        )
        assert book.best_bid_dbn(IID) == 101
        assert book.last_ts_ns == 100


class TestTradeClearNone:
    def test_trade_record_does_not_change_depth(self) -> None:
        book = Book()
        _fold(
            book,
            _add(1, Side.BID, 100, 3, ts_event=10, sequence=1),
            _add(2, Side.ASK, 101, 2, ts_event=11, sequence=2),
        )
        before = book.snapshot_bbo(IID)
        apply_event(book, _trade(101, 1, ts_event=12, sequence=3))
        assert book.snapshot_bbo(IID) == before
        assert book.size_at_price(IID, BookSide.ASK, 101) == 2

    def test_clear_wipes_instrument(self) -> None:
        book = Book()
        _fold(
            book,
            _add(1, Side.BID, 100, 3, ts_event=10, sequence=1),
            _add(2, Side.ASK, 101, 2, ts_event=11, sequence=2),
            _add(9, Side.BID, 50, 1, ts_event=12, sequence=3, instrument_id=2),
            _clear(ts_event=13, sequence=4),
        )
        assert book.snapshot_bbo(IID) == (None, None)
        assert book.order_by_id(IID, 1) is None
        assert book.size_at_price(IID, BookSide.ASK, 101) == 0
        # a different instrument is untouched
        assert book.best_bid_dbn(2) == 50

    def test_clear_then_rebuild_same_instrument(self) -> None:
        book = Book()
        _fold(
            book,
            _add(1, Side.BID, 100, 3, ts_event=10, sequence=1),
            _add(2, Side.ASK, 101, 2, ts_event=11, sequence=2),
            _clear(ts_event=12, sequence=3),
            _add(3, Side.BID, 105, 4, ts_event=13, sequence=4),
            _add(4, Side.ASK, 106, 5, ts_event=14, sequence=5),
        )
        assert book.snapshot_bbo(IID) == (105, 106)
        assert book.size_at_price(IID, BookSide.BID, 105) == 4
        assert book.order_by_id(IID, 1) is None  # pre-clear order stays gone
        book.check_invariants()

    def test_clear_unknown_instrument_is_noop(self) -> None:
        book = Book()
        apply_event(book, _clear(ts_event=1, sequence=1, instrument_id=777))

    def test_none_action_is_noop(self) -> None:
        book = Book()
        apply_event(book, _add(1, Side.BID, 100, 1, ts_event=10, sequence=1))
        apply_event(book, _none(ts_event=11, sequence=2))
        assert book.best_bid_dbn(IID) == 100
        assert book.unseen_cm_count == 0


class TestQueueAhead:
    def test_arrival_cutoff_excludes_later_orders(self) -> None:
        book = Book()
        _fold(
            book,
            _add(1, Side.BID, 100, 2, ts_event=10, sequence=1),
            _add(2, Side.BID, 100, 3, ts_event=20, sequence=2),
        )
        # arrival at ts==15: only the ts==10 order is ahead
        assert book.queue_ahead_size(IID, BookSide.BID, 100, 15) == 2

    def test_resting_order_at_exactly_our_arrival_ts_is_ahead(self) -> None:
        # AD-22: "our order is always last at its price" -> a venue order stamped
        # at exactly our arrival ns counts as ahead of us (cutoff is '<=').
        book = Book()
        _fold(
            book,
            _add(1, Side.BID, 100, 2, ts_event=10, sequence=1),
            _add(2, Side.BID, 100, 3, ts_event=20, sequence=2),
        )
        assert book.queue_ahead_size(IID, BookSide.BID, 100, 10) == 2
        assert book.queue_ahead_size(IID, BookSide.BID, 100, 20) == 5
        assert book.queue_ahead_size(IID, BookSide.BID, 100, 9) == 0

    def test_empty_level(self) -> None:
        book = Book()
        assert book.queue_ahead_size(IID, BookSide.BID, 100, 10) == 0


class TestCrossedMarket:
    """Cross-timer logic, verified against a small monkeypatched tolerance so it
    is independent of the seal constant (item 19)."""

    TOL = 1_000

    @pytest.fixture(autouse=True)
    def _small_tolerance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(book_module, "MAX_TRANSIENT_CROSS_NS", self.TOL)

    def test_transient_cross_resolves_without_error(self) -> None:
        book = Book()
        _fold(
            book,
            _add(1, Side.ASK, 100, 1, ts_event=1_000, sequence=1),
            _add(2, Side.BID, 100, 1, ts_event=1_000, sequence=2),  # crossed
        )
        apply_event(book, _none(ts_event=1_000 + self.TOL - 1, sequence=3))
        apply_event(
            book,
            _cancel(2, Side.BID, 100, 1, ts_event=1_000 + self.TOL, sequence=4),
        )
        assert book.max_transient_cross_ns == self.TOL - 1
        assert book.best_bid_dbn(IID) is None

    def test_persistent_cross_raises(self) -> None:
        book = Book()
        _fold(
            book,
            _add(1, Side.ASK, 100, 1, ts_event=1_000, sequence=1),
            _add(2, Side.BID, 100, 1, ts_event=1_000, sequence=2),
        )
        with pytest.raises(BookInconsistency):
            apply_event(book, _none(ts_event=1_000 + self.TOL, sequence=3))

    def test_cross_timer_resets_on_uncross(self) -> None:
        book = Book()
        _fold(
            book,
            _add(1, Side.ASK, 100, 1, ts_event=0, sequence=1),
            _add(2, Side.BID, 100, 1, ts_event=0, sequence=2),  # cross starts at 0
            _cancel(2, Side.BID, 100, 1, ts_event=10, sequence=3),  # uncross
            _add(3, Side.BID, 100, 1, ts_event=20, sequence=4),  # new cross at 20
        )
        # far past MAX from the *first* cross, but only just after the second
        apply_event(book, _none(ts_event=20 + self.TOL - 1, sequence=5))
        assert book.best_bid_dbn(IID) == 100  # no raise

    def test_equal_prices_count_as_crossed(self) -> None:
        book = Book()
        _fold(
            book,
            _add(1, Side.ASK, 100, 1, ts_event=0, sequence=1),
            _add(2, Side.BID, 100, 1, ts_event=0, sequence=2),
        )
        with pytest.raises(BookInconsistency):
            apply_event(book, _none(ts_event=self.TOL, sequence=3))


def test_persistent_cross_uses_the_seal_constant() -> None:
    # one check that the real config value is actually wired in (no monkeypatch)
    book = Book()
    _fold(
        book,
        _add(1, Side.ASK, 100, 1, ts_event=0, sequence=1),
        _add(2, Side.BID, 100, 1, ts_event=0, sequence=2),
    )
    apply_event(book, _none(ts_event=MAX_TRANSIENT_CROSS_NS - 1, sequence=3))
    assert book.max_transient_cross_ns == MAX_TRANSIENT_CROSS_NS - 1
    with pytest.raises(BookInconsistency):
        apply_event(book, _none(ts_event=MAX_TRANSIENT_CROSS_NS, sequence=4))


class TestUnhandledAction:
    def test_unknown_action_raises(self) -> None:
        book = Book()
        bad = _Rec("Z", str(Side.BID), 1, 100, 1, 1, 1)
        with pytest.raises(BookInconsistency):
            apply_event(book, bad)


class TestCheckInvariants:
    def test_healthy_book_passes(self) -> None:
        book = Book()
        _fold(
            book,
            _add(1, Side.BID, 100, 3, ts_event=1, sequence=1),
            _add(2, Side.BID, 100, 2, ts_event=2, sequence=2),
            _add(3, Side.ASK, 101, 4, ts_event=3, sequence=3),
        )
        book.check_invariants()

    def test_total_size_mismatch_raises(self) -> None:
        book = Book()
        _fold(book, _add(1, Side.BID, 100, 3, ts_event=1, sequence=1))
        level = book.instruments[IID].side_book(BookSide.BID)[100]
        level.total_size = 999  # corrupt it
        with pytest.raises(BookInconsistency):
            book.check_invariants()

    def test_untracked_cross_raises(self) -> None:
        book = Book()
        sub = book._sub_or_create(IID)
        sub.add(1, RestingOrder(IID, BookSide.BID, 100, 1, 1, 1))
        sub.add(2, RestingOrder(IID, BookSide.ASK, 100, 1, 1, 2))
        # crossed, but cross_start_ns is None (bypassed apply_event)
        with pytest.raises(BookInconsistency):
            book.check_invariants()


# --- integration: fold a fixture prefix --------------------------------

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "tick"
    / "_test"
    / "glbx-mdp3-20260622.mbo.dbn.zst"
)
_TICK_DBN = 250_000_000
_WARMUP = 400_000  # populate the overnight resting book (slow-decaying tail)
_WINDOW = 200_000  # measure steady-state behaviour over the next slice


def _fixture_or_skip() -> Path:
    if _FIXTURE.is_file():
        return _FIXTURE
    if os.environ.get("TICKSIM_REQUIRE_FIXTURE"):
        pytest.fail(f"TICKSIM_REQUIRE_FIXTURE set but fixture missing: {_FIXTURE}")
    pytest.skip(f"GLBX MDP3 fixture not present: {_FIXTURE}")


@pytest.mark.integration
def test_fold_fixture_front_month_prefix() -> None:
    """Fold ~600k front-month records and assert the book stays sane.

    The spec's "first ~200k records, unseen_cm_count / total < 0.01" is split
    into a ``_WARMUP`` slice (whose unseen C/M is dominated by orders resting
    since before the file starts -- the GLBX book is ~24 h, so the fixture opens
    onto a full overnight book we never saw built; measured ~3.8 % at 200k,
    decaying slowly) and a following ``_WINDOW`` over which the steady-state
    unseen rate must clear the 1 % bar (Amendment 9 §A9.2's ~0.3 %-of-C/M
    figure is a steady-state claim, not a warm-up-inclusive one).

    Also checks (spec Tasks / Acceptance): ``ts_event`` non-decreasing across
    the fold (enforced inside ``apply_event`` now), no
    :class:`BookInconsistency`, final ``best_bid < best_ask``, a few-tick
    spread, and ``check_invariants`` at the end.

    The capture is fed through ``events.DbnMboSource`` -- the production path --
    so this also exercises the vendor-record normalization end to end.
    """
    from src.ticksim.events import DbnMboSource

    fixture = _fixture_or_skip()

    book = Book()
    folded = 0
    unseen_before_window = 0
    window_total = 0

    for ev in DbnMboSource(fixture):
        if ev.instrument_id != FRONT_MONTH_INSTRUMENT_ID:
            continue
        apply_event(book, ev)  # raises on non-monotonic ts / persistent cross

        folded += 1
        if folded == _WARMUP:
            unseen_before_window = book.unseen_cm_count
        elif folded > _WARMUP:
            window_total += 1
        if folded >= _WARMUP + _WINDOW:
            break

    assert folded == _WARMUP + _WINDOW, "fixture prefix shorter than expected"

    window_unseen = book.unseen_cm_count - unseen_before_window
    assert window_unseen / window_total < 0.01, (
        f"steady-state unseen C/M rate {window_unseen}/{window_total} "
        f"= {window_unseen / window_total:.4f} exceeds 1 %"
    )

    book.check_invariants()

    best_bid, best_ask = book.snapshot_bbo(FRONT_MONTH_INSTRUMENT_ID)
    assert best_bid is not None and best_ask is not None
    assert best_bid < best_ask, "reconstructed book is crossed at end of fold"
    spread_ticks = (best_ask - best_bid) / _TICK_DBN
    assert spread_ticks <= 4, f"implausible {spread_ticks}-tick spread"

    # no crossed market anywhere in the fold (this fixture slice is clean)
    assert book.max_transient_cross_ns == 0
    assert book.overcancel_count == 0

    # depth is derived, not stored twice: the touch sizes are positive
    assert book.size_at_price(FRONT_MONTH_INSTRUMENT_ID, BookSide.BID, best_bid) > 0
    assert book.size_at_price(FRONT_MONTH_INSTRUMENT_ID, BookSide.ASK, best_ask) > 0
