"""The fill-decision engine + the two seal-frozen queue models (spine AD-5).

``fills.decide(book, tracker, clock_ns, config) -> list[FillEvent]`` is the one
pure authority on *whether an order fills this tick* (spine AD-19). It holds no
state between calls: every per-order counter it needs
(``cum_trade_vol_since_arrival`` and the live ``queue_ahead``) lives on the
order inside :class:`~src.ticksim.orders.OrderTracker`, maintained by the
:meth:`QueueModel.observe_book_event` seam that ``sim.py`` drives immediately
after each ``book.apply_event`` (spine AD-20 / AD-21).

Two queue models (pre-registration §2.1; spine AD-15 / AD-22):
  * :class:`BackOfQueueModel` -- ``config.PRIMARY``; every resting order at our
    price (``add_ts_ns <= arrival``) counts as ahead of us.
  * :class:`TimePriorityModel` -- ``config.OPTIMISTIC``; only strictly-earlier
    resting orders (``add_ts_ns < arrival``) count; ties break so our order is
    last.

Fill formula (Alex decision 2026-08-29 + spine AD-22): the live ``queue_ahead``
is reduced **only** by cancels / modifies of orders ahead of us; trade volume
accumulates separately in ``cum_trade_vol_since_arrival``. AD-22's formula is
*cumulative* -- total fill entitlement to date::

    cumulative = clamp(cum_trade_vol_since_arrival - queue_ahead, 0, size)

and ``decide`` returns the **this-tick delta** ``cumulative - filled_qty``
(AD-19: a ``FillEvent`` is never cumulative). A through-print (an aggressor that
crossed our price) lands as ordinary ``cum_trade_vol_since_arrival`` and
naturally fills us.

Dependencies (spine AD-7): ``book``, ``orders``, ``config`` + stdlib only. No
mutation of ``book`` or ``OrderTracker`` inside :func:`decide` (spine AD-5) --
only :meth:`QueueModel.observe_book_event` touches the tracker, via its two
guarded counter mutators.
"""

from __future__ import annotations

import abc

from .book import Book, BookSide, MboRecord, RestingOrder, UNDEF_ORDER_SIZE
from .config import QueueModel as QueueModelKind, SimConfig
from .orders import FillEvent, OrderKind, OrderSnapshot, OrderTracker, Side

__all__ = [
    "QueueModel",
    "BackOfQueueModel",
    "TimePriorityModel",
    "queue_model_for",
    "decide",
]

# The MBO action codes this module reacts to (``book.py`` normalizes every
# vendor record to these single-char ``str`` codes).
_TRADE, _CANCEL, _MODIFY = "T", "C", "M"


class QueueModel(abc.ABC):
    """One interface, two implementations (spine AD-22).

    Concrete subclasses supply :meth:`queue_ahead_size` (the arrival-tick
    formula ``sim.py`` calls **once** per order) and :meth:`counts_resting_order`
    (does a given resting order sit ahead of ours?). The book-event seam
    (:meth:`observe_book_event`) is identical for both models -- it differs only
    through :meth:`counts_resting_order` -- so it is implemented here once.
    """

    @abc.abstractmethod
    def queue_ahead_size(
        self,
        book: Book,
        instrument_id: int,
        side: BookSide,
        price_dbn: int,
        our_arrival_ts_ns: int,
    ) -> int:
        """Total resting size ahead of an order arriving at
        ``our_arrival_ts_ns`` at ``price_dbn`` on ``side`` (spine AD-22).

        Called exactly once by ``sim.py``, at the order's arrival tick; the
        result is written to ``queue_rank_at_submit`` /
        ``queue_ahead_size_at_submit`` and seeds the live ``queue_ahead``.
        """

    @abc.abstractmethod
    def counts_resting_order(
        self, add_ts_ns: int, sequence: int, snap: OrderSnapshot
    ) -> bool:
        """``True`` iff a venue order stamped ``add_ts_ns`` / ``sequence`` sits
        **ahead of** our working order ``snap`` in its price queue.

        ``sequence`` is part of the interface for symmetry with AD-21's
        tie-break wording, but neither model consults it: our order carries no
        vendor sequence at submit, so the tie-break is purely on ``add_ts_ns``.
        """

    # --- the book-event -> order-state seam (spine AD-21) ----------------

    def observe_book_event(
        self,
        tracker: OrderTracker,
        record: MboRecord,
        resting_before: RestingOrder | None,
    ) -> None:
        """Fold one just-applied MBO ``record`` into the working orders'
        counters (spine AD-21). ``sim.py`` is the sole caller, immediately after
        each ``book.apply_event``.

        ``resting_before`` is the venue order at ``record.order_id`` **before**
        the event (``sim`` looks it up pre-``apply_event``); ``None`` for a
        trade. Enumerated rules only -- nothing else moves a counter:

          * ``T`` at/through a working passive order's limit, strictly after its
            arrival (spine AD-20) -> ``add_trade_volume``.
          * ``C`` (full cancel), a size-down same-price ``M``, or a
            price-changing ``M`` of a resting order **ahead of** a working
            passive order at its price -> ``decrement_queue_ahead`` (a
            price-changing ``M`` removes the whole resting size, since the order
            has left our level).
          * everything else (``A``, size-up same-price ``M``, ``R``, ``N``,
            ``F``) -> no-op. ``R`` (book clear) is an exchange halt/reset event,
            excluded by the AD-13 session mask, not modelled here (prereg §2.2).
        """
        action = str(record.action)
        if action == _TRADE:
            self._observe_trade(tracker, record)
        elif action in (_CANCEL, _MODIFY):
            self._observe_cancel_or_modify(tracker, record, resting_before)
        # A / size-up M / R / N / F -> no-op (spine AD-21)

    def _observe_trade(self, tracker: OrderTracker, record: MboRecord) -> None:
        trade_px = int(record.price_dbn)
        trade_size = int(record.size)
        trade_ts = int(record.ts_event)
        if trade_size <= 0 or trade_size == UNDEF_ORDER_SIZE:
            return  # malformed / sentinel size -- do not credit ~4.3e9 volume
        for order_id in tracker.working_order_ids():
            snap = tracker.snapshot(order_id)
            if snap.kind is not OrderKind.PASSIVE_LIMIT:
                continue
            limit = snap.limit_px_dbn
            if limit is None:
                continue
            if trade_ts <= snap.arrival_ts_ns:
                continue  # strict '>' -- a same-tick trade sorts before arrival
            # Volume accrual is side-agnostic: prereg §2.1 says "trade volume at
            # our price after our arrival" with no aggressor-side qualifier, and
            # a print at/through our limit is queue-clearing volume regardless
            # of which side lifted it. `record.side` (aggressor) is deliberately
            # not consulted.
            at_or_through = (
                trade_px <= limit if snap.side is Side.BUY else trade_px >= limit
            )
            if at_or_through:
                tracker.add_trade_volume(order_id, trade_size)

    def _observe_cancel_or_modify(
        self,
        tracker: OrderTracker,
        record: MboRecord,
        resting_before: RestingOrder | None,
    ) -> None:
        if resting_before is None:
            return
        if str(record.action) == _CANCEL:
            # every GLBX `C` is a full cancel (book.py); the whole resting size
            # leaves the level.
            removed_size = resting_before.size
        elif int(record.price_dbn) != resting_before.price_dbn:
            # a price-changing `M` -- book.apply_event implements it as
            # remove-from-old-level + add-to-new-level, so the order has left
            # `resting_before.price_dbn` (our level) entirely; its new resting
            # spot is behind us (fresh add_ts_ns, priority lost).
            removed_size = resting_before.size
        else:  # same-price modify: only a size *decrease* removes queue ahead
            raw = int(record.size)
            if raw == UNDEF_ORDER_SIZE or raw >= resting_before.size:
                return
            removed_size = resting_before.size - raw
        if removed_size <= 0:
            return
        for order_id in tracker.working_order_ids():
            snap = tracker.snapshot(order_id)
            if snap.kind is not OrderKind.PASSIVE_LIMIT:
                continue
            if not _resting_order_shares_our_price(snap, resting_before):
                continue
            if not self.counts_resting_order(
                resting_before.add_ts_ns, resting_before.sequence, snap
            ):
                continue
            tracker.decrement_queue_ahead(order_id, removed_size)


class BackOfQueueModel(QueueModel):
    """Pre-registration §2.1 primary (``config.PRIMARY``): our order joins the
    back of the queue -- every resting order at our price is ahead of us."""

    def queue_ahead_size(
        self,
        book: Book,
        instrument_id: int,
        side: BookSide,
        price_dbn: int,
        our_arrival_ts_ns: int,
    ) -> int:
        return book.queue_ahead_size(
            instrument_id, side, price_dbn, our_arrival_ts_ns, strict=False
        )

    def counts_resting_order(
        self, add_ts_ns: int, sequence: int, snap: OrderSnapshot
    ) -> bool:
        return add_ts_ns <= snap.arrival_ts_ns


class TimePriorityModel(QueueModel):
    """Pre-registration §2.1 secondary (``config.OPTIMISTIC``): strict venue
    time priority -- only orders stamped *before* our arrival are ahead of us;
    ties break so our order is last at its price."""

    def queue_ahead_size(
        self,
        book: Book,
        instrument_id: int,
        side: BookSide,
        price_dbn: int,
        our_arrival_ts_ns: int,
    ) -> int:
        return book.queue_ahead_size(
            instrument_id, side, price_dbn, our_arrival_ts_ns, strict=True
        )

    def counts_resting_order(
        self, add_ts_ns: int, sequence: int, snap: OrderSnapshot
    ) -> bool:
        return add_ts_ns < snap.arrival_ts_ns


_MODEL_FOR_KIND: dict[QueueModelKind, type[QueueModel]] = {
    QueueModelKind.BACK_OF_QUEUE: BackOfQueueModel,
    QueueModelKind.TIME_PRIORITY: TimePriorityModel,
}


def queue_model_for(config: SimConfig) -> QueueModel:
    """A fresh :class:`QueueModel` instance for ``config.queue_model`` (spine
    AD-5 / AD-22). ``decide`` and ``sim.py`` both go through here.

    Raises:
        ValueError: ``config.queue_model`` maps to no model.
    """
    model_cls = _MODEL_FOR_KIND.get(config.queue_model)
    if model_cls is None:
        raise ValueError(
            f"no queue model for config.queue_model={config.queue_model!r}"
        )
    return model_cls()


def _resting_order_shares_our_price(snap: OrderSnapshot, resting: RestingOrder) -> bool:
    """``True`` iff ``resting`` sits at the same side + price as working passive
    order ``snap`` (single-instrument sim -- the tracker carries no instrument
    id, so side + price is the whole match; spine Deferred: multi-instrument)."""
    if resting.price_dbn != snap.limit_px_dbn:
        return False
    if snap.side is Side.BUY:
        return resting.side is BookSide.BID
    return resting.side is BookSide.ASK


def _sole_instrument_id(book: Book) -> int | None:
    """The single ``instrument_id`` in ``book`` (H1 is single-instrument MNQ
    front-month; spine Deferred). ``None`` if the book is still empty.

    Raises:
        ValueError: the book holds more than one instrument -- ``decide`` cannot
            tell which one a tracker order belongs to (the tracker / intent log
            carry no instrument id).
    """
    instruments = book.instruments
    if not instruments:
        return None
    if len(instruments) > 1:
        raise ValueError(
            "fills.decide requires a single-instrument book; got "
            f"{len(instruments)} instruments {sorted(instruments)}"
        )
    return next(iter(instruments))


def _passive_fill(snap: OrderSnapshot, clock_ns: int) -> list[FillEvent]:
    """This-tick incremental fill for a working ``PASSIVE_LIMIT`` order (spine
    AD-19 / AD-22). ``[]`` when nothing new fills this tick.

    AD-22's formula is *cumulative* -- ``clamp(cum_trade_vol_since_arrival −
    queue_ahead, 0, size)`` is total fill entitlement to date. ``decide`` must
    return the **this-tick delta** (AD-19: never cumulative), so we subtract
    what the tracker has already recorded as filled.
    """
    limit = snap.limit_px_dbn
    if limit is None:  # impossible per OrderIntent schema; guard for mypy + safety
        raise ValueError(f"passive-limit order {snap.order_id!r} has no limit_px_dbn")
    cumulative_entitled = max(0, snap.cum_trade_vol_since_arrival - snap.queue_ahead)
    cumulative_entitled = min(cumulative_entitled, snap.size)
    fill_qty = cumulative_entitled - snap.filled_qty
    if fill_qty <= 0:
        return []
    return [
        FillEvent(order_id=snap.order_id, px_dbn=limit, size=fill_qty, ts_ns=clock_ns)
    ]


def _walk_book(book: Book, snap: OrderSnapshot, clock_ns: int) -> list[FillEvent]:
    """Consume the opposite side of the book for a marketable / marketable-limit
    order, best price first, one :class:`FillEvent` per level at that level's
    price (spine AD-5).

    ``marketable_limit`` stops once a level's price is beyond ``limit_px_dbn``
    (``> limit`` for a BUY, ``< limit`` for a SELL).

    **The book is walked at most once per order** -- only while
    ``filled_qty == 0`` (review 2026-08-29). The book is never depleted (no
    own-order market impact), so a second walk would re-consume the same
    displayed size; a marketable order that partially fills therefore behaves
    IOC-like: its remainder stays working but inert until the AD-13 session
    mask expires it. (A marketable order that found an empty book at arrival
    still gets its one walk on the first tick liquidity appears.) 1--5-lot MNQ
    at the touch fills in full at arrival, so this bites only pathologically
    thin books.

    A fill never prices **better than the touch snapshotted at the arrival tick**
    (spine AD-16 invariant 1): a BUY pays at least ``arrival_best_ask_dbn``, a
    SELL receives at most ``arrival_best_bid_dbn``.
    """
    if snap.filled_qty > 0:
        return []  # already had its one walk -- do not re-consume the book
    instrument_id = _sole_instrument_id(book)
    if instrument_id is None:
        return []
    remaining = snap.size - snap.filled_qty
    if remaining <= 0:
        return []

    opposite = BookSide.ASK if snap.side is Side.BUY else BookSide.BID
    arrival_touch = (
        snap.arrival_best_ask_dbn
        if snap.side is Side.BUY
        else snap.arrival_best_bid_dbn
    )

    limit_px: int | None = None
    if snap.kind is OrderKind.MARKETABLE_LIMIT:
        if snap.limit_px_dbn is None:  # impossible per schema; guard for mypy
            raise ValueError(
                f"marketable-limit order {snap.order_id!r} has no limit_px_dbn"
            )
        limit_px = snap.limit_px_dbn

    out: list[FillEvent] = []
    for price_dbn, level_size in book.resting_levels(instrument_id, opposite):
        if limit_px is not None:
            if snap.side is Side.BUY and price_dbn > limit_px:
                break
            if snap.side is Side.SELL and price_dbn < limit_px:
                break
        take = min(remaining, level_size)
        if take <= 0:
            continue
        fill_px = price_dbn
        if arrival_touch is not None:
            fill_px = (
                max(price_dbn, arrival_touch)
                if snap.side is Side.BUY
                else min(price_dbn, arrival_touch)
            )
        out.append(
            FillEvent(
                order_id=snap.order_id,
                px_dbn=fill_px,
                size=take,
                ts_ns=clock_ns,
            )
        )
        remaining -= take
        if remaining <= 0:
            break
    return out


def decide(
    book: Book, tracker: OrderTracker, clock_ns: int, config: SimConfig
) -> list[FillEvent]:
    """This-tick incremental fills for every working order (spine AD-19).

    Pure: reads ``tracker.snapshot(oid)`` for each ``tracker.working_order_ids()``
    and the book's queries; mutates neither. Returns ``[]`` when nothing fills.
    ``FillEvent``s are strictly this-tick deltas, never cumulative.

    Raises:
        ValueError: ``config.queue_model`` maps to no model (via
            :func:`queue_model_for`), or the book holds more than one instrument.
    """
    # Validate the configured queue model up front (spine AD-22) -- an unknown
    # model is a hard error even on a tick that only has marketable orders.
    queue_model_for(config)
    # And the single-instrument precondition, unconditionally -- a passive-only
    # tick must fail the same way a marketable one does (the tracker carries no
    # instrument id, so a >1-instrument book cross-attributes fills).
    _sole_instrument_id(book)

    fills: list[FillEvent] = []
    for order_id in tracker.working_order_ids():
        snap = tracker.snapshot(order_id)
        if snap.kind is OrderKind.PASSIVE_LIMIT:
            fills.extend(_passive_fill(snap, clock_ns))
        else:  # MARKETABLE / MARKETABLE_LIMIT
            fills.extend(_walk_book(book, snap, clock_ns))
    return fills
