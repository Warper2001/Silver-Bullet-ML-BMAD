"""Unit tests for ``src.ticksim.fills`` (spine AD-5 / AD-19 / AD-21 / AD-22).

One test per row of the spec's I/O & Edge-Case Matrix plus the four Acceptance
Criteria. Book state is built with tiny hand-rolled :class:`_Rec` stubs folded
through ``book.apply_event`` (the same pattern ``test_ticksim_book.py`` uses);
tracker orders are built through the real ``OrderTracker`` submit / activate /
setter API.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.ticksim import config
from src.ticksim.book import Book, BookSide, RestingOrder, apply_event
from src.ticksim.config import QueueModel as QueueModelKind
from src.ticksim.fills import (
    BackOfQueueModel,
    QueueModel,
    TimePriorityModel,
    decide,
    queue_model_for,
)
from src.ticksim.orders import (
    FillEvent,
    IntentAction,
    Leg,
    OrderIntent,
    OrderKind,
    OrderTracker,
    Side,
)

IID = 1


# --- record stub + builders --------------------------------------------


@dataclass(frozen=True)
class _Rec:
    """Structural stand-in for a normalized MBO record (``book.MboRecord``)."""

    action: str
    side: str
    order_id: int
    price_dbn: int
    size: int
    ts_event: int
    sequence: int
    instrument_id: int = IID


def _add(order_id: int, side: str, price: int, size: int, ts: int, seq: int) -> _Rec:
    return _Rec("A", side, order_id, price, size, ts, seq)


def _trade(price: int, size: int, ts: int, seq: int, side: str = "A") -> _Rec:
    return _Rec("T", side, 0, price, size, ts, seq)


def _cancel(order_id: int, side: str, price: int, size: int, ts: int, seq: int) -> _Rec:
    return _Rec("C", side, order_id, price, size, ts, seq)


def _modify(order_id: int, side: str, price: int, size: int, ts: int, seq: int) -> _Rec:
    return _Rec("M", side, order_id, price, size, ts, seq)


# --- tracker builders --------------------------------------------------


def _working_order(
    tracker: OrderTracker,
    order_id: str,
    kind: OrderKind,
    side: Side,
    size: int,
    *,
    limit_px_dbn: int | None,
    submit_ts_ns: int = 0,
    latency_ns: int = 0,
    leg: Leg = Leg.ENTRY,
) -> None:
    """Submit + activate one order so it is ``WORKING`` at ``submit_ts_ns +
    latency_ns``."""
    intent = OrderIntent(
        action=IntentAction.SUBMIT,
        order_id=order_id,
        trade_id=f"trade-{order_id}",
        leg=leg,
        kind=kind,
        side=side,
        size=size,
        limit_px_dbn=limit_px_dbn,
        submit_ts_ns=submit_ts_ns,
    )
    arrival = submit_ts_ns + latency_ns
    tracker.submit(intent, latency_ns=latency_ns, now_ns=submit_ts_ns)
    tracker.activate_arrivals(now_ns=arrival)


def _passive(
    tracker: OrderTracker,
    order_id: str,
    side: Side,
    size: int,
    price: int,
    *,
    queue_ahead: int,
    cum_trade_vol: int,
    submit_ts_ns: int = 0,
    latency_ns: int = 0,
) -> None:
    _working_order(
        tracker,
        order_id,
        OrderKind.PASSIVE_LIMIT,
        side,
        size,
        limit_px_dbn=price,
        submit_ts_ns=submit_ts_ns,
        latency_ns=latency_ns,
    )
    tracker.set_queue_position(order_id, rank=0, ahead_size=queue_ahead)
    if cum_trade_vol:
        tracker.add_trade_volume(order_id, cum_trade_vol)


# --- passive-limit fill formula (matrix rows 1-3, AC-2) ----------------


class TestPassiveFill:
    def test_queue_not_cleared_no_fill(self) -> None:
        tracker = OrderTracker()
        _passive(tracker, "o1", Side.BUY, 5, 100, queue_ahead=10, cum_trade_vol=8)
        assert decide(Book(), tracker, clock_ns=99, config=config.PRIMARY) == []

    def test_partial_fill(self) -> None:
        tracker = OrderTracker()
        _passive(tracker, "o1", Side.BUY, 5, 100, queue_ahead=10, cum_trade_vol=12)
        fills = decide(Book(), tracker, clock_ns=99, config=config.PRIMARY)
        assert fills == [FillEvent(order_id="o1", px_dbn=100, size=2, ts_ns=99)]

    def test_already_part_filled_caps_at_remaining(self) -> None:
        tracker = OrderTracker()
        _passive(tracker, "o1", Side.BUY, 5, 100, queue_ahead=0, cum_trade_vol=0)
        tracker.apply_fill(
            FillEvent(order_id="o1", px_dbn=100, size=3, ts_ns=1), now_ns=1
        )
        tracker.add_trade_volume("o1", 20)
        fills = decide(Book(), tracker, clock_ns=5, config=config.PRIMARY)
        assert fills == [FillEvent(order_id="o1", px_dbn=100, size=2, ts_ns=5)]

    def test_sell_side_fills_at_its_limit(self) -> None:
        tracker = OrderTracker()
        _passive(tracker, "s1", Side.SELL, 4, 200, queue_ahead=1, cum_trade_vol=3)
        fills = decide(Book(), tracker, clock_ns=7, config=config.PRIMARY)
        assert fills == [FillEvent(order_id="s1", px_dbn=200, size=2, ts_ns=7)]

    def test_ac2_min_k_m(self) -> None:
        # cum_trade_vol exceeds queue_ahead by K=8, M=3 contracts left -> one
        # FillEvent of min(K, M) = 3 at the limit, ts == clock_ns.
        tracker = OrderTracker()
        _passive(tracker, "o1", Side.BUY, 3, 500, queue_ahead=2, cum_trade_vol=10)
        fills = decide(Book(), tracker, clock_ns=1234, config=config.PRIMARY)
        assert fills == [FillEvent(order_id="o1", px_dbn=500, size=3, ts_ns=1234)]


# --- observe_book_event: trades feed cum_trade_vol (matrix rows 4-5) ---


class TestObserveTrade:
    def _order(self, side: Side = Side.BUY, limit: int = 100) -> OrderTracker:
        tracker = OrderTracker()
        _passive(
            tracker,
            "o1",
            side,
            10,
            limit,
            queue_ahead=5,
            cum_trade_vol=0,
            submit_ts_ns=10,
        )
        return tracker

    def test_trade_at_arrival_ts_not_counted(self) -> None:
        tracker = self._order()
        BackOfQueueModel().observe_book_event(
            tracker, _trade(100, 3, ts=10, seq=1), None
        )
        assert tracker.snapshot("o1").cum_trade_vol_since_arrival == 0

    def test_trade_through_price_counted(self) -> None:
        tracker = self._order(side=Side.BUY, limit=100)
        BackOfQueueModel().observe_book_event(
            tracker, _trade(99, 4, ts=11, seq=1), None
        )
        assert tracker.snapshot("o1").cum_trade_vol_since_arrival == 4

    def test_trade_above_buy_limit_not_counted(self) -> None:
        tracker = self._order(side=Side.BUY, limit=100)
        BackOfQueueModel().observe_book_event(
            tracker, _trade(101, 4, ts=11, seq=1), None
        )
        assert tracker.snapshot("o1").cum_trade_vol_since_arrival == 0

    def test_trade_at_sell_limit_counted(self) -> None:
        tracker = self._order(side=Side.SELL, limit=100)
        BackOfQueueModel().observe_book_event(
            tracker, _trade(100, 2, ts=11, seq=1), None
        )
        assert tracker.snapshot("o1").cum_trade_vol_since_arrival == 2

    def test_trade_never_touches_queue_ahead(self) -> None:
        tracker = self._order(side=Side.BUY, limit=100)
        BackOfQueueModel().observe_book_event(
            tracker, _trade(99, 99, ts=11, seq=1), None
        )
        assert tracker.snapshot("o1").queue_ahead == 5  # unchanged


# --- observe_book_event: cancels/mods move queue_ahead (matrix 6-8) ----


class TestObserveCancelModify:
    def _order(self, submit_ts_ns: int = 10) -> OrderTracker:
        tracker = OrderTracker()
        _passive(
            tracker,
            "o1",
            Side.BUY,
            5,
            100,
            queue_ahead=10,
            cum_trade_vol=0,
            submit_ts_ns=submit_ts_ns,
        )
        return tracker

    def test_cancel_ahead_decrements(self) -> None:
        tracker = self._order(submit_ts_ns=10)
        resting = RestingOrder(IID, BookSide.BID, 100, 6, add_ts_ns=5, sequence=1)
        BackOfQueueModel().observe_book_event(
            tracker, _cancel(99, "B", 100, 6, ts=12, seq=9), resting
        )
        assert tracker.snapshot("o1").queue_ahead == 4

    def test_cancel_not_ahead_time_priority_no_decrement(self) -> None:
        tracker = self._order(submit_ts_ns=10)
        # resting stamped exactly at our arrival ts -> not ahead under time-priority
        resting = RestingOrder(IID, BookSide.BID, 100, 6, add_ts_ns=10, sequence=1)
        TimePriorityModel().observe_book_event(
            tracker, _cancel(99, "B", 100, 6, ts=12, seq=9), resting
        )
        assert tracker.snapshot("o1").queue_ahead == 10

    def test_cancel_at_our_arrival_back_of_queue_does_decrement(self) -> None:
        tracker = self._order(submit_ts_ns=10)
        resting = RestingOrder(IID, BookSide.BID, 100, 6, add_ts_ns=10, sequence=1)
        BackOfQueueModel().observe_book_event(
            tracker, _cancel(99, "B", 100, 6, ts=12, seq=9), resting
        )
        assert tracker.snapshot("o1").queue_ahead == 4

    def test_size_down_modify_decrements_by_delta(self) -> None:
        tracker = self._order(submit_ts_ns=10)
        resting = RestingOrder(IID, BookSide.BID, 100, 6, add_ts_ns=5, sequence=1)
        BackOfQueueModel().observe_book_event(
            tracker, _modify(99, "B", 100, 2, ts=12, seq=9), resting
        )
        assert tracker.snapshot("o1").queue_ahead == 6  # 10 - (6 - 2)

    def test_size_up_modify_is_noop(self) -> None:
        tracker = self._order(submit_ts_ns=10)
        resting = RestingOrder(IID, BookSide.BID, 100, 6, add_ts_ns=5, sequence=1)
        BackOfQueueModel().observe_book_event(
            tracker, _modify(99, "B", 100, 9, ts=12, seq=9), resting
        )
        assert tracker.snapshot("o1").queue_ahead == 10

    def test_cancel_at_other_price_is_noop(self) -> None:
        tracker = self._order(submit_ts_ns=10)
        resting = RestingOrder(IID, BookSide.BID, 99, 6, add_ts_ns=5, sequence=1)
        BackOfQueueModel().observe_book_event(
            tracker, _cancel(99, "B", 99, 6, ts=12, seq=9), resting
        )
        assert tracker.snapshot("o1").queue_ahead == 10

    def test_cancel_on_opposite_side_is_noop(self) -> None:
        tracker = self._order(submit_ts_ns=10)
        resting = RestingOrder(IID, BookSide.ASK, 100, 6, add_ts_ns=5, sequence=1)
        BackOfQueueModel().observe_book_event(
            tracker, _cancel(99, "A", 100, 6, ts=12, seq=9), resting
        )
        assert tracker.snapshot("o1").queue_ahead == 10


# --- marketable / marketable-limit walk (matrix rows 9-12) -------------


class TestWalkBook:
    def _marketable(
        self, tracker: OrderTracker, side: Side, size: int, *, limit: int | None = None
    ) -> None:
        kind = OrderKind.MARKETABLE if limit is None else OrderKind.MARKETABLE_LIMIT
        _working_order(tracker, "m1", kind, side, size, limit_px_dbn=limit)

    def test_marketable_buy_deep_book_one_fill(self) -> None:
        book = Book()
        apply_event(book, _add(1, "A", 30000, 5, ts=1, seq=1))
        tracker = OrderTracker()
        self._marketable(tracker, Side.BUY, 3)
        fills = decide(book, tracker, clock_ns=50, config=config.PRIMARY)
        assert fills == [FillEvent(order_id="m1", px_dbn=30000, size=3, ts_ns=50)]

    def test_marketable_buy_walks_levels(self) -> None:
        book = Book()
        apply_event(book, _add(1, "A", 30000, 2, ts=1, seq=1))
        apply_event(book, _add(2, "A", 30025, 10, ts=2, seq=2))
        tracker = OrderTracker()
        self._marketable(tracker, Side.BUY, 5)
        fills = decide(book, tracker, clock_ns=50, config=config.PRIMARY)
        assert fills == [
            FillEvent(order_id="m1", px_dbn=30000, size=2, ts_ns=50),
            FillEvent(order_id="m1", px_dbn=30025, size=3, ts_ns=50),
        ]

    def test_marketable_sell_walks_bids_best_first(self) -> None:
        book = Book()
        apply_event(book, _add(1, "B", 29975, 10, ts=1, seq=1))
        apply_event(book, _add(2, "B", 30000, 2, ts=2, seq=2))
        tracker = OrderTracker()
        self._marketable(tracker, Side.SELL, 5)
        fills = decide(book, tracker, clock_ns=50, config=config.PRIMARY)
        assert fills == [
            FillEvent(order_id="m1", px_dbn=30000, size=2, ts_ns=50),
            FillEvent(order_id="m1", px_dbn=29975, size=3, ts_ns=50),
        ]

    def test_marketable_limit_buy_stops_at_limit(self) -> None:
        book = Book()
        apply_event(book, _add(1, "A", 30000, 2, ts=1, seq=1))
        apply_event(book, _add(2, "A", 30025, 10, ts=2, seq=2))
        tracker = OrderTracker()
        self._marketable(tracker, Side.BUY, 5, limit=30000)
        fills = decide(book, tracker, clock_ns=50, config=config.PRIMARY)
        assert fills == [FillEvent(order_id="m1", px_dbn=30000, size=2, ts_ns=50)]

    def test_marketable_limit_remainder_does_not_rewalk(self) -> None:
        # the book is walked at most once (filled_qty == 0); a partially-filled
        # marketable order is IOC-like -- its remainder stays working but inert.
        book = Book()
        apply_event(book, _add(1, "A", 30000, 2, ts=1, seq=1))  # only 2 <= limit
        apply_event(book, _add(2, "A", 30025, 10, ts=2, seq=2))  # beyond limit
        tracker = OrderTracker()
        self._marketable(tracker, Side.BUY, 5, limit=30010)
        first = decide(book, tracker, clock_ns=50, config=config.PRIMARY)
        assert first == [FillEvent(order_id="m1", px_dbn=30000, size=2, ts_ns=50)]
        tracker.apply_fill(first[0], now_ns=50)
        assert tracker.terminal_state("m1") is None  # remainder 3 still working
        # fresh liquidity appears under the limit -- but the order already walked
        apply_event(book, _add(3, "A", 30005, 4, ts=60, seq=3))
        assert decide(book, tracker, clock_ns=70, config=config.PRIMARY) == []

    def test_marketable_empty_book_side_no_fill_stays_working(self) -> None:
        book = Book()
        apply_event(book, _add(1, "B", 30000, 5, ts=1, seq=1))  # only bids
        tracker = OrderTracker()
        self._marketable(tracker, Side.BUY, 3)
        assert decide(book, tracker, clock_ns=50, config=config.PRIMARY) == []
        assert tracker.terminal_state("m1") is None

    def test_marketable_no_instrument_yet(self) -> None:
        tracker = OrderTracker()
        self._marketable(tracker, Side.BUY, 3)
        assert decide(Book(), tracker, clock_ns=50, config=config.PRIMARY) == []


# --- decide-level edge rows (matrix rows 13-14) -----------------------


class TestDecide:
    def test_no_working_orders_returns_empty(self) -> None:
        tracker = OrderTracker()
        # one in-flight order (submitted, not yet arrived)
        intent = OrderIntent(
            action=IntentAction.SUBMIT,
            order_id="o1",
            trade_id="t1",
            leg=Leg.ENTRY,
            kind=OrderKind.PASSIVE_LIMIT,
            side=Side.BUY,
            size=5,
            limit_px_dbn=100,
            submit_ts_ns=0,
        )
        tracker.submit(intent, latency_ns=1_000, now_ns=0)
        assert decide(Book(), tracker, clock_ns=10, config=config.PRIMARY) == []

    def test_does_not_mutate_book_or_tracker(self) -> None:
        book = Book()
        apply_event(book, _add(1, "A", 30000, 5, ts=1, seq=1))
        tracker = OrderTracker()
        _working_order(
            tracker, "m1", OrderKind.MARKETABLE, Side.BUY, 3, limit_px_dbn=None
        )
        before_depth = book.size_at_price(IID, BookSide.ASK, 30000)
        before_working = tracker.working_order_ids()
        decide(book, tracker, clock_ns=50, config=config.PRIMARY)
        assert book.size_at_price(IID, BookSide.ASK, 30000) == before_depth
        assert tracker.working_order_ids() == before_working
        assert tracker.snapshot("m1").filled_qty == 0

    def test_unknown_queue_model_raises_value_error(self) -> None:
        bogus = SimpleNamespace(queue_model=object())
        with pytest.raises(ValueError):
            queue_model_for(bogus)  # type: ignore[arg-type]

    def test_multi_instrument_book_raises(self) -> None:
        book = Book()
        apply_event(book, _add(1, "A", 30000, 5, ts=1, seq=1))
        apply_event(book, _Rec("A", "A", 2, 40000, 5, 2, 2, instrument_id=2))
        tracker = OrderTracker()
        _working_order(
            tracker, "m1", OrderKind.MARKETABLE, Side.BUY, 3, limit_px_dbn=None
        )
        with pytest.raises(ValueError):
            decide(book, tracker, clock_ns=50, config=config.PRIMARY)


# --- queue models (AC-1) ---------------------------------------------


class TestQueueModels:
    def test_queue_model_for_maps_enum(self) -> None:
        assert isinstance(queue_model_for(config.PRIMARY), BackOfQueueModel)
        assert isinstance(queue_model_for(config.OPTIMISTIC), TimePriorityModel)
        assert config.PRIMARY.queue_model is QueueModelKind.BACK_OF_QUEUE
        assert config.OPTIMISTIC.queue_model is QueueModelKind.TIME_PRIORITY

    def test_fresh_instance_each_call(self) -> None:
        assert queue_model_for(config.PRIMARY) is not queue_model_for(config.PRIMARY)

    def test_ac1_tie_at_arrival_ts(self) -> None:
        book = Book()
        apply_event(book, _add(1, "B", 100, 3, ts=10, seq=1))  # add_ts == arrival
        back = queue_model_for(config.PRIMARY)
        tp = queue_model_for(config.OPTIMISTIC)
        assert back.queue_ahead_size(book, IID, BookSide.BID, 100, 10) == 3
        assert tp.queue_ahead_size(book, IID, BookSide.BID, 100, 10) == 0

    def test_queue_ahead_size_earlier_order_counts_for_both(self) -> None:
        book = Book()
        apply_event(book, _add(1, "B", 100, 3, ts=5, seq=1))
        back = queue_model_for(config.PRIMARY)
        tp = queue_model_for(config.OPTIMISTIC)
        assert back.queue_ahead_size(book, IID, BookSide.BID, 100, 10) == 3
        assert tp.queue_ahead_size(book, IID, BookSide.BID, 100, 10) == 3

    def test_is_queue_model_subclass(self) -> None:
        assert issubclass(BackOfQueueModel, QueueModel)
        assert issubclass(TimePriorityModel, QueueModel)


# --- book.resting_levels (spec Execution task) ------------------------


class TestRestingLevels:
    def test_asks_best_first_ascending(self) -> None:
        book = Book()
        apply_event(book, _add(1, "A", 30025, 4, ts=1, seq=1))
        apply_event(book, _add(2, "A", 30000, 2, ts=2, seq=2))
        assert book.resting_levels(IID, BookSide.ASK) == [(30000, 2), (30025, 4)]

    def test_bids_best_first_descending(self) -> None:
        book = Book()
        apply_event(book, _add(1, "B", 29975, 4, ts=1, seq=1))
        apply_event(book, _add(2, "B", 30000, 2, ts=2, seq=2))
        assert book.resting_levels(IID, BookSide.BID) == [(30000, 2), (29975, 4)]

    def test_unknown_instrument_or_empty_side(self) -> None:
        book = Book()
        apply_event(book, _add(1, "B", 100, 1, ts=1, seq=1))
        assert book.resting_levels(999, BookSide.BID) == []
        assert book.resting_levels(IID, BookSide.ASK) == []

    def test_strict_kw_on_queue_ahead_size(self) -> None:
        book = Book()
        apply_event(book, _add(1, "B", 100, 3, ts=10, seq=1))
        assert book.queue_ahead_size(IID, BookSide.BID, 100, 10) == 3
        assert book.queue_ahead_size(IID, BookSide.BID, 100, 10, strict=True) == 0


# --- review-pass hardening (blind / edge-case / verification-gap) ------


class TestReviewHardening:
    def _sett_bbo(
        self, tracker: OrderTracker, oid: str, bid: int | None, ask: int | None
    ):
        tracker.set_arrival_bbo(oid, bid_dbn=bid, ask_dbn=ask)

    # --- the critical bug: passive fill must not re-emit across ticks ---

    def test_passive_fill_not_re_emitted_when_counters_static(self) -> None:
        tracker = OrderTracker()
        _passive(tracker, "o1", Side.BUY, 5, 100, queue_ahead=10, cum_trade_vol=12)
        t1 = decide(Book(), tracker, clock_ns=1, config=config.PRIMARY)
        assert t1 == [FillEvent(order_id="o1", px_dbn=100, size=2, ts_ns=1)]
        for fe in t1:
            tracker.apply_fill(fe, now_ns=1)
        # no new trade volume -> tick 2 emits nothing (AD-19: this-tick delta)
        assert decide(Book(), tracker, clock_ns=2, config=config.PRIMARY) == []

    def test_passive_fill_emits_only_the_new_increment(self) -> None:
        tracker = OrderTracker()
        _passive(tracker, "o1", Side.BUY, 5, 100, queue_ahead=10, cum_trade_vol=12)
        for fe in decide(Book(), tracker, clock_ns=1, config=config.PRIMARY):
            tracker.apply_fill(fe, now_ns=1)  # filled 2
        tracker.add_trade_volume("o1", 1)  # cum_trade_vol 12 -> 13
        t2 = decide(Book(), tracker, clock_ns=2, config=config.PRIMARY)
        assert t2 == [FillEvent(order_id="o1", px_dbn=100, size=1, ts_ns=2)]

    # --- price-changing M of an order ahead (blind) --------------------

    def test_price_changing_modify_ahead_removes_full_size(self) -> None:
        tracker = OrderTracker()
        _passive(
            tracker,
            "o1",
            Side.BUY,
            5,
            100,
            queue_ahead=10,
            cum_trade_vol=0,
            submit_ts_ns=10,
        )
        resting = RestingOrder(IID, BookSide.BID, 100, 6, add_ts_ns=5, sequence=1)
        # M moves order 99 from our price (100) to 99 -> it left our queue
        BackOfQueueModel().observe_book_event(
            tracker, _modify(99, "B", 99, 6, ts=12, seq=9), resting
        )
        assert tracker.snapshot("o1").queue_ahead == 4  # 10 - 6 (full)

    # --- UNDEF size on a trade record (edge-case) ---------------------

    def test_trade_with_undef_size_is_ignored(self) -> None:
        from src.ticksim.book import UNDEF_ORDER_SIZE

        tracker = OrderTracker()
        _passive(
            tracker,
            "o1",
            Side.BUY,
            5,
            100,
            queue_ahead=5,
            cum_trade_vol=0,
            submit_ts_ns=10,
        )
        BackOfQueueModel().observe_book_event(
            tracker, _trade(100, UNDEF_ORDER_SIZE, ts=11, seq=1), None
        )
        assert tracker.snapshot("o1").cum_trade_vol_since_arrival == 0

    # --- decide raises unconditionally (blind) ----------------------

    def test_decide_raises_on_bad_queue_model_even_passive_only(self) -> None:
        tracker = OrderTracker()
        _passive(tracker, "o1", Side.BUY, 5, 100, queue_ahead=0, cum_trade_vol=9)
        bogus = SimpleNamespace(queue_model=object())
        with pytest.raises(ValueError):
            decide(Book(), tracker, clock_ns=1, config=bogus)  # type: ignore[arg-type]

    def test_decide_raises_multi_instrument_on_passive_only_tick(self) -> None:
        book = Book()
        apply_event(book, _add(1, "B", 100, 5, ts=1, seq=1))
        apply_event(book, _Rec("A", "B", 2, 40000, 5, 2, 2, instrument_id=2))
        tracker = OrderTracker()
        _passive(tracker, "o1", Side.BUY, 5, 100, queue_ahead=0, cum_trade_vol=9)
        with pytest.raises(ValueError):
            decide(book, tracker, clock_ns=5, config=config.PRIMARY)

    # --- marketable re-walk never beats the arrival touch (blind) ------

    def test_marketable_rewalk_capped_at_arrival_touch(self) -> None:
        book = Book()  # empty at arrival -> nothing to fill
        tracker = OrderTracker()
        _working_order(
            tracker, "m1", OrderKind.MARKETABLE, Side.BUY, 3, limit_px_dbn=None
        )
        tracker.set_arrival_bbo("m1", bid_dbn=29975, ask_dbn=30025)
        assert decide(book, tracker, clock_ns=1, config=config.PRIMARY) == []
        # later tick: an ask appears *below* the arrival touch (30000 < 30025)
        apply_event(book, _add(1, "A", 30000, 3, ts=2, seq=1))
        t2 = decide(book, tracker, clock_ns=2, config=config.PRIMARY)
        assert t2 == [FillEvent(order_id="m1", px_dbn=30025, size=3, ts_ns=2)]

    # --- marketable-limit SELL stop-at-limit (coverage gap) -----------

    def test_marketable_limit_sell_stops_at_limit(self) -> None:
        book = Book()
        apply_event(book, _add(1, "B", 30000, 2, ts=1, seq=1))
        apply_event(book, _add(2, "B", 29975, 10, ts=2, seq=2))
        tracker = OrderTracker()
        _working_order(
            tracker,
            "m1",
            OrderKind.MARKETABLE_LIMIT,
            Side.SELL,
            5,
            limit_px_dbn=30000,
        )
        fills = decide(book, tracker, clock_ns=9, config=config.PRIMARY)
        assert fills == [FillEvent(order_id="m1", px_dbn=30000, size=2, ts_ns=9)]

    # --- observe leaves a non-passive order untouched (coverage gap) ---

    def test_observe_ignores_marketable_orders(self) -> None:
        tracker = OrderTracker()
        _working_order(
            tracker, "m1", OrderKind.MARKETABLE, Side.BUY, 5, limit_px_dbn=None
        )
        BackOfQueueModel().observe_book_event(
            tracker, _trade(100, 9, ts=50, seq=1), None
        )
        snap = tracker.snapshot("m1")
        assert snap.cum_trade_vol_since_arrival == 0 and snap.queue_ahead == 0

    # --- passive + marketable in one decide call (coverage gap) -------

    def test_decide_mixed_passive_and_marketable(self) -> None:
        book = Book()
        apply_event(book, _add(1, "A", 30000, 4, ts=1, seq=1))
        tracker = OrderTracker()
        _passive(tracker, "p1", Side.SELL, 3, 200, queue_ahead=1, cum_trade_vol=3)
        _working_order(
            tracker, "m1", OrderKind.MARKETABLE, Side.BUY, 2, limit_px_dbn=None
        )
        fills = decide(book, tracker, clock_ns=7, config=config.PRIMARY)
        assert FillEvent(order_id="p1", px_dbn=200, size=2, ts_ns=7) in fills
        assert FillEvent(order_id="m1", px_dbn=30000, size=2, ts_ns=7) in fills
        assert len(fills) == 2


class TestArrivalClampNeverBreachesLimit:
    """Regression: AD-16 invariant 1's arrival-touch clamp must never push a
    marketable-**limit** fill *through* its own limit (invariant 2).

    Found by `tests/integration/test_ticksim_parity_synthetic.py` on the real
    2026-06-22 capture -- 87 of 1000 synthetic orders breached invariant 2. The
    level test at the top of `_walk_book` bounds `price_dbn`, but the clamp
    `max(price_dbn, arrival_touch)` could raise `fill_px` past the limit
    whenever the market ran away during the 250 ms latency hop. Such an order
    was never marketable at arrival and must take no liquidity at all --
    granting it a fill is an optimistic bias in exactly the adverse direction
    this simulator exists to remove.
    """

    def test_buy_limit_below_arrival_ask_does_not_fill(self) -> None:
        book = Book()
        # a level at 30000 is at/below the limit, so the level test admits it...
        apply_event(book, _add(1, "A", 30000, 5, ts=1, seq=1))
        tracker = OrderTracker()
        _working_order(
            tracker,
            "m1",
            OrderKind.MARKETABLE_LIMIT,
            Side.BUY,
            3,
            limit_px_dbn=30000,
        )
        # ...but the market ran away during the hop: arrival ask 30050 > limit.
        tracker.set_arrival_bbo("m1", bid_dbn=30025, ask_dbn=30050)
        assert decide(book, tracker, clock_ns=50, config=config.PRIMARY) == []

    def test_sell_limit_above_arrival_bid_does_not_fill(self) -> None:
        book = Book()
        apply_event(book, _add(1, "B", 30000, 5, ts=1, seq=1))
        tracker = OrderTracker()
        _working_order(
            tracker,
            "m1",
            OrderKind.MARKETABLE_LIMIT,
            Side.SELL,
            3,
            limit_px_dbn=30000,
        )
        # market ran away downward: arrival bid 29950 < limit.
        tracker.set_arrival_bbo("m1", bid_dbn=29950, ask_dbn=29975)
        assert decide(book, tracker, clock_ns=50, config=config.PRIMARY) == []

    def test_buy_limit_at_arrival_ask_still_fills_at_the_limit(self) -> None:
        """The fix must not suppress a legitimately marketable order."""
        book = Book()
        apply_event(book, _add(1, "A", 30000, 5, ts=1, seq=1))
        tracker = OrderTracker()
        _working_order(
            tracker,
            "m1",
            OrderKind.MARKETABLE_LIMIT,
            Side.BUY,
            3,
            limit_px_dbn=30000,
        )
        tracker.set_arrival_bbo("m1", bid_dbn=29975, ask_dbn=30000)
        assert decide(book, tracker, clock_ns=50, config=config.PRIMARY) == [
            FillEvent(order_id="m1", px_dbn=30000, size=3, ts_ns=50)
        ]

    def test_plain_marketable_is_unaffected_by_the_guard(self) -> None:
        """A true market order has no limit -- it still pays the arrival touch."""
        book = Book()
        apply_event(book, _add(1, "A", 30000, 3, ts=1, seq=1))
        tracker = OrderTracker()
        _working_order(
            tracker, "m1", OrderKind.MARKETABLE, Side.BUY, 3, limit_px_dbn=None
        )
        tracker.set_arrival_bbo("m1", bid_dbn=30025, ask_dbn=30050)
        assert decide(book, tracker, clock_ns=50, config=config.PRIMARY) == [
            FillEvent(order_id="m1", px_dbn=30050, size=3, ts_ns=50)
        ]

    def test_partial_walk_stops_at_the_level_that_would_breach(self) -> None:
        """A deeper level whose clamped price breaches the limit truncates the
        walk rather than filling through."""
        book = Book()
        apply_event(book, _add(1, "A", 30000, 2, ts=1, seq=1))
        apply_event(book, _add(2, "A", 30025, 10, ts=2, seq=2))
        tracker = OrderTracker()
        _working_order(
            tracker,
            "m1",
            OrderKind.MARKETABLE_LIMIT,
            Side.BUY,
            5,
            limit_px_dbn=30025,
        )
        tracker.set_arrival_bbo("m1", bid_dbn=29975, ask_dbn=30000)
        fills = decide(book, tracker, clock_ns=50, config=config.PRIMARY)
        assert fills == [
            FillEvent(order_id="m1", px_dbn=30000, size=2, ts_ns=50),
            FillEvent(order_id="m1", px_dbn=30025, size=3, ts_ns=50),
        ]
        assert all(f.px_dbn <= 30025 for f in fills)
