"""Venue-faithful L3 order book + the single ``apply_event`` MBO folder.

This module is the sole authority on MBO -> book-state transitions (spine AD-9).
It is a **passive data structure**: no event loop, no file reading, no stream
merging -- ``sim.py`` drives the fold later (spine AD-20). Only *real venue*
orders live here; our own orders never enter the book (spine AD-3).

Dependencies (spine AD-7): ``src.ticksim.config`` + stdlib + ``sortedcontainers``
(the O(log n) best-bid/ask, spine AD-4). It imports **nothing** from
``databento`` / ``databento_dbn``: the vendor boundary now lives entirely in
``events.py``, which normalizes each vendor record to an ``events.BookEvent``
before ``sim.py`` folds it here (spine AD-18). :class:`MboRecord` is a
structural Protocol keyed on plain ``str`` action / side codes and ``int``
fields -- whatever ``events.py`` yields. It does **not** import ``orders.py`` --
the book holds raw tuples, not ``OrderIntent`` / ``Fill``.

MBO action semantics -- the single-char codes ``events.py`` normalizes to
(verified against the GLBX MDP3 capture
``data/tick/_test/glbx-mdp3-20260622.mbo.dbn.zst`` and the committed slice
``tests/fixtures/mnq_mbo_tiny.dbn.zst``):

  * ``A`` add      -- a new resting order enters the book.
  * ``C`` cancel   -- a resting order is removed. **On GLBX every ``C`` is a
                      full cancel** (``record.size == resting size`` in 811,602
                      of 811,602 ``C`` records over the first 2 M front-month
                      events; zero partials, zero over-cancels). The record's
                      ``size`` is the quantity being cancelled (a *delta*):
                      ``new_size = existing.size - record.size``. The partial
                      branch is therefore defensive -- reached only by a future
                      non-GLBX ``events.py`` source.
  * ``M`` modify   -- a resting order's price and/or size changed; the record
                      carries the **new absolute** price/size.
  * ``T`` trade    -- aggressor trade summary. **Does not affect the book.**
  * ``F`` fill     -- a resting order was (partly) filled. **Does not affect the
                      book.** The matching size reduction arrives as a separate
                      ``C`` or ``M``.
  * ``R`` clear    -- wipe every order + level for that ``instrument_id``.
  * ``N`` none     -- flag-only; a documented no-op.

**Ask-First resolution (spec Boundaries / Design Notes -- accepted, frozen
matrix amended).** The spec assumed ``F`` reduces the hit resting order by
``size`` (remove at 0). Verification against the capture *and* the vendor's own
record-type docs (``FILL``: "An existing order was filled. Does not affect the
book.") shows this is wrong on GLBX: e.g. order ``6878505599343``
folds as ``A sz3 -> F sz1 -> M sz2 -> F sz2 -> C sz2`` -- the ``M`` and ``C``
carry the post-fill sizes and do the book mutation; the ``F`` records are
informational. Treating ``F`` as a reducer double-counts every fill against the
following ``C`` / ``M`` and inflates ``unseen_cm_count`` ~2.5x. This module
folds ``F`` as a no-op (identical to ``T``).

Tolerances (spine AD-9):
  * a ``C`` / ``M`` for an unseen ``order_id`` is a no-op and bumps
    ``Book.unseen_cm_count`` (Amendment 9 §A9.2: a pre-window order, ~0.3 % of
    ``C`` / ``M`` at steady state).
  * an over-cancel (``C`` whose size exceeds the resting size) removes the order
    and bumps ``Book.overcancel_count`` (never seen on GLBX; defensive).
  * a transient crossed market (``best_bid_dbn >= best_ask_dbn``) is tolerated
    for ``< config.MAX_TRANSIENT_CROSS_NS``; a cross persisting ``>=`` that long
    raises :class:`BookInconsistency`.
  * a crossed market **wider than** ``config.STALE_CROSS_MAX_TICKS`` is a
    cold-start ghost, not a market cross: the +/-90-min parity windows carry no
    UTC-midnight snapshot, so a pre-window resting order is never ``A``-dded and
    its stale level can sit on one side of the book for the whole window. Such a
    cross never arms the persistence timer and never raises; it bumps
    ``Book.stale_cross_count`` instead (one per stale-cross *episode*).

Input hardening (this module is AD-9's sole authority, so a malformed record
must fail deterministically, never a bare ``KeyError`` or silent corruption):
  * ``A`` with ``size == UNDEF_ORDER_SIZE`` or ``size <= 0`` -> raise.
  * duplicate ``A`` for an already-resting ``order_id`` -> raise.
  * ``M`` whose side differs from the resting order's side -> raise.
  * ``C`` / ``M`` with ``size == UNDEF_ORDER_SIZE`` -> treat as the resting
    order's full size (``C`` -> full cancel; ``M`` -> size unchanged).
  * a record whose ``ts_event`` is earlier than the last one folded -> raise
    (AD-20 guarantees a monotonic total order).
  * a price level unexpectedly absent when it must exist -> raise.

Integer-only (spine AD-10): ``price_dbn`` / ``ts`` / ``size`` / ``sequence`` are
all ``int``. The clock is ``ts_event`` (spine AD-1); ``ts_recv`` is never read.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import NoReturn, Protocol

from sortedcontainers import SortedDict  # type: ignore[import-untyped]

from .config import MAX_TRANSIENT_CROSS_NS, MNQ_TICK_DBN, STALE_CROSS_MAX_TICKS

__all__ = [
    "Book",
    "BookInconsistency",
    "BookSide",
    "MboRecord",
    "RestingOrder",
    "UNDEF_ORDER_SIZE",
    "apply_event",
]

logger = logging.getLogger(__name__)

# The single-char MBO action / side codes ``events.py`` normalizes every record
# to (``events.MboAction`` / ``events.MboSide`` are ``StrEnum``s over exactly
# these values). :func:`apply_event` compares against them directly, so it never
# needs a vendor import -- ``str(record.action)`` on the Protocol is enough.
_ADD, _CANCEL, _MODIFY, _TRADE, _FILL, _CLEAR, _NONE = "A", "C", "M", "T", "F", "R", "N"
_SIDE_BID, _SIDE_ASK = "B", "A"
_MUTATING = frozenset({_ADD, _CANCEL, _MODIFY, _CLEAR})

# DBN's "undefined size" sentinel (UINT32_MAX). Pinned as a literal so ``book.py``
# stays free of any ``databento`` import (spine AD-18). ``events.py`` rejects an
# undefined *price* on an ``A`` / ``M`` at the normalization boundary; an
# undefined *size* is meaningful on ``C`` / ``M`` (full cancel / size unchanged)
# and is handled below, so ``events.py`` passes it straight through.
UNDEF_ORDER_SIZE = 4_294_967_295


class BookInconsistency(Exception):
    """A book-state invariant the sim's correctness depends on was violated.

    Raised by :func:`apply_event` for a crossed market that persists ``>=``
    ``config.MAX_TRANSIENT_CROSS_NS``, a non-monotonic ``ts_event``, and a
    structurally impossible record (bad size, duplicate add, side flip on
    modify, an ``A`` with no market side, an unhandled action, a missing level).
    """


def _fail(message: str) -> NoReturn:
    """Log then raise -- the one place a :class:`BookInconsistency` is thrown."""
    logger.warning("BookInconsistency: %s", message)
    raise BookInconsistency(message)


class BookSide(Enum):
    """Local bid/ask enum so stored tuples carry no vendor type (spine AD-3)."""

    BID = "bid"
    ASK = "ask"


class MboRecord(Protocol):
    """Structural type :func:`apply_event` consumes.

    The sole production caller is ``sim.py``, which folds ``events.BookEvent``
    records -- whose ``action`` / ``side`` are ``events.MboAction`` /
    ``events.MboSide`` ``str`` enums and whose price field is ``price_dbn`` --
    so a :class:`BookEvent` satisfies this Protocol structurally under
    ``mypy --strict`` (spine AD-18). ``action`` / ``side`` are declared ``str``
    (not a vendor enum) and the price field is named ``price_dbn`` to match
    ``RestingOrder`` / ``BookEvent``; the whole ``databento`` boundary lives in
    ``events.py`` (Ask-First resolution -- accepted). Only these attributes are
    read -- ``ts_recv`` (spine AD-1) and ``flags`` are deliberately absent.
    """

    @property
    def action(self) -> str: ...

    @property
    def side(self) -> str: ...

    @property
    def order_id(self) -> int: ...

    @property
    def price_dbn(self) -> int: ...

    @property
    def size(self) -> int: ...

    @property
    def ts_event(self) -> int: ...

    @property
    def sequence(self) -> int: ...

    @property
    def instrument_id(self) -> int: ...


@dataclass(frozen=True)
class RestingOrder:
    """One real venue order resting in the book (spine AD-3 tuple).

    Frozen: a size/price change produces a new instance. The vendor
    ``order_id`` is *not* a field -- it is the key of the dict that holds the
    order (spine AD-3 tuple shape).

    ``add_ts_ns`` is queue priority for :meth:`Book.queue_ahead_size`.
    ``sequence`` is stored for AD-21's use in ``fills.py`` (deciding which
    ``C`` / ``M`` events land *ahead of* our order) -- it is **not** used by
    ``queue_ahead_size`` (our order has no vendor sequence at submit).
    """

    instrument_id: int
    side: BookSide
    price_dbn: int
    size: int
    add_ts_ns: int
    sequence: int


@dataclass
class _PriceLevel:
    """Aggregate + membership for one price on one side of one instrument."""

    total_size: int = 0
    orders: dict[int, RestingOrder] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.orders


@dataclass
class _InstrumentBook:
    """The L3 book for one ``instrument_id``.

    ``bids`` / ``asks`` are ``SortedDict`` keyed by ``price_dbn`` so the touch
    is O(log n) (spine AD-4). L2 depth is derived from these, never stored twice
    (spine AD-3).
    """

    orders: dict[int, RestingOrder] = field(default_factory=dict)
    bids: SortedDict = field(default_factory=SortedDict)
    asks: SortedDict = field(default_factory=SortedDict)
    cross_start_ns: int | None = None
    # True while the book is crossed *wider* than config.STALE_CROSS_MAX_TICKS
    # (a cold-start ghost -- never timed, never fatal). Used only to count one
    # ``Book.stale_cross_count`` per episode rather than per event.
    stale_cross_open: bool = False

    def side_book(self, side: BookSide) -> SortedDict:
        return self.bids if side is BookSide.BID else self.asks

    def best_bid_dbn(self) -> int | None:
        if not self.bids:
            return None
        return int(self.bids.peekitem(-1)[0])

    def best_ask_dbn(self) -> int | None:
        if not self.asks:
            return None
        return int(self.asks.peekitem(0)[0])

    def _level(self, side: BookSide, price_dbn: int) -> _PriceLevel:
        level: _PriceLevel | None = self.side_book(side).get(price_dbn)
        if level is None:
            _fail(
                f"price level {price_dbn} ({side.value}) is absent but an order "
                f"claims to rest there -- book corruption"
            )
        return level

    def add(self, order_id: int, order: RestingOrder) -> None:
        if order_id in self.orders:
            _fail(f"duplicate ADD for already-resting order_id {order_id}")
        self.orders[order_id] = order
        level = self.side_book(order.side).setdefault(order.price_dbn, _PriceLevel())
        level.orders[order_id] = order
        level.total_size += order.size

    def remove(self, order_id: int, order: RestingOrder) -> None:
        del self.orders[order_id]
        side_book = self.side_book(order.side)
        level = self._level(order.side, order.price_dbn)
        level.orders.pop(order_id, None)
        level.total_size -= order.size
        if level.is_empty():
            del side_book[order.price_dbn]

    def resize(self, order_id: int, order: RestingOrder, new_size: int) -> None:
        """Change ``order``'s size in place, keeping its queue position."""
        level = self._level(order.side, order.price_dbn)
        updated = RestingOrder(
            instrument_id=order.instrument_id,
            side=order.side,
            price_dbn=order.price_dbn,
            size=new_size,
            add_ts_ns=order.add_ts_ns,
            sequence=order.sequence,
        )
        level.total_size += new_size - order.size
        level.orders[order_id] = updated
        self.orders[order_id] = updated

    def clear(self) -> None:
        self.orders.clear()
        self.bids.clear()
        self.asks.clear()
        self.cross_start_ns = None
        self.stale_cross_open = False


@dataclass
class Book:
    """Top-level L3 book: one ``_InstrumentBook`` per ``instrument_id``.

    ``unseen_cm_count`` / ``overcancel_count`` / ``stale_cross_count`` /
    ``max_transient_cross_ns`` / ``last_ts_ns`` are plain attributes ``sim.py``
    folds into the run manifest later (spine AD-9, Design Notes).

    ``stale_cross_count`` counts **episodes** (not events) of a crossed market
    wider than ``config.STALE_CROSS_MAX_TICKS`` -- the cold-start ghosts
    described in the module docstring. It is a tolerance counter: a non-zero
    value never fails a run, it only tells a reader the window's book was
    reconstructed without a snapshot, so the tolerance can be judged.
    """

    instruments: dict[int, _InstrumentBook] = field(default_factory=dict)
    unseen_cm_count: int = 0
    overcancel_count: int = 0
    stale_cross_count: int = 0
    max_transient_cross_ns: int = 0
    last_ts_ns: int = -1  # ts_event of the last record folded (AD-20 monotonic)

    def _sub(self, instrument_id: int) -> _InstrumentBook | None:
        return self.instruments.get(instrument_id)

    def _sub_or_create(self, instrument_id: int) -> _InstrumentBook:
        sub = self.instruments.get(instrument_id)
        if sub is None:
            sub = _InstrumentBook()
            self.instruments[instrument_id] = sub
        return sub

    # --- queries (spine AD-4 / AD-22) ------------------------------------

    def best_bid_dbn(self, instrument_id: int) -> int | None:
        """Highest resting bid price for ``instrument_id``; ``None`` if no bid."""
        sub = self._sub(instrument_id)
        return None if sub is None else sub.best_bid_dbn()

    def best_ask_dbn(self, instrument_id: int) -> int | None:
        """Lowest resting ask price for ``instrument_id``; ``None`` if no ask."""
        sub = self._sub(instrument_id)
        return None if sub is None else sub.best_ask_dbn()

    def snapshot_bbo(self, instrument_id: int) -> tuple[int | None, int | None]:
        """``(best_bid_dbn, best_ask_dbn)`` for ``instrument_id`` in one call."""
        sub = self._sub(instrument_id)
        if sub is None:
            return (None, None)
        return (sub.best_bid_dbn(), sub.best_ask_dbn())

    def size_at_price(self, instrument_id: int, side: BookSide, price_dbn: int) -> int:
        """Total resting size at ``price_dbn`` on ``side``.

        ``0`` if the level is empty or the instrument is unknown.
        """
        sub = self._sub(instrument_id)
        if sub is None:
            return 0
        level = sub.side_book(side).get(price_dbn)
        return 0 if level is None else int(level.total_size)

    def order_by_id(self, instrument_id: int, order_id: int) -> RestingOrder | None:
        """The resting order with ``order_id`` on ``instrument_id``, or ``None``."""
        sub = self._sub(instrument_id)
        if sub is None:
            return None
        return sub.orders.get(order_id)

    def queue_ahead_size(
        self,
        instrument_id: int,
        side: BookSide,
        price_dbn: int,
        our_arrival_ts_ns: int,
        *,
        strict: bool = False,
    ) -> int:
        """Total resting size at ``price_dbn`` that is *ahead of* an order
        arriving at ``our_arrival_ts_ns`` (spine AD-22).

        With ``strict=False`` (default, the back-of-queue reading) the cutoff is
        ``add_ts_ns <= our_arrival_ts_ns``: AD-22 says "our order is always last
        at its price" at submit, so a venue order stamped at exactly our arrival
        ns is ahead of us. With ``strict=True`` (the ``TimePriorityModel``
        reading, ``fills.py``) the cutoff is ``add_ts_ns < our_arrival_ts_ns`` --
        a venue order stamped at exactly our arrival ns is *not* ahead of us
        (ties break so our order is last). ``sequence`` is not consulted -- our
        order has no vendor sequence (that tie-break belongs to ``fills.py`` via
        AD-21, not to this query).

        O(orders-at-that-price) -- that set is small (Design Notes); only the
        touch is required to be O(log n).
        """
        sub = self._sub(instrument_id)
        if sub is None:
            return 0
        level = sub.side_book(side).get(price_dbn)
        if level is None:
            return 0
        if strict:
            return sum(
                o.size for o in level.orders.values() if o.add_ts_ns < our_arrival_ts_ns
            )
        return sum(
            o.size for o in level.orders.values() if o.add_ts_ns <= our_arrival_ts_ns
        )

    def resting_levels(
        self, instrument_id: int, side: BookSide
    ) -> list[tuple[int, int]]:
        """``(price_dbn, total_size)`` for every resting level on ``side``,
        **best price first** (highest bid / lowest ask).

        Used by ``fills._walk_book`` to consume the opposite side for a
        marketable / marketable-limit order (spine AD-5). Read-only -- the walk
        never mutates the book (no own-order market impact). ``[]`` if the
        instrument is unknown or that side is empty.
        """
        sub = self._sub(instrument_id)
        if sub is None:
            return []
        side_book = sub.side_book(side)
        levels = [
            (int(price), int(level.total_size)) for price, level in side_book.items()
        ]
        if side is BookSide.BID:
            levels.reverse()  # SortedDict is ascending; best bid is the highest
        return levels

    def check_invariants(self) -> None:
        """Assert structural integrity across every instrument (spine AD-16 kin).

        Raises :class:`BookInconsistency` if any price level's ``total_size``
        disagrees with the sum of its orders, or if a book is crossed without an
        active transient-cross timer. A cross wider than
        ``config.STALE_CROSS_MAX_TICKS`` is a cold-start ghost (already counted
        in ``stale_cross_count``): it legitimately has no timer, so it is
        tolerated here too -- otherwise the end-of-run check would abort exactly
        the runs :func:`_check_cross` was taught to survive. Cheap enough for
        unit tests and the parity preflight; not called on the hot path.
        """
        for instrument_id, sub in self.instruments.items():
            for side in (BookSide.BID, BookSide.ASK):
                for price_dbn, level in sub.side_book(side).items():
                    member_sum = sum(o.size for o in level.orders.values())
                    if level.total_size != member_sum:
                        _fail(
                            f"instrument {instrument_id} level {price_dbn} "
                            f"({side.value}): total_size {level.total_size} != "
                            f"sum(orders) {member_sum}"
                        )
                    if level.is_empty() or level.total_size <= 0:
                        _fail(
                            f"instrument {instrument_id} level {price_dbn} "
                            f"({side.value}): empty/non-positive level not pruned"
                        )
            bid = sub.best_bid_dbn()
            ask = sub.best_ask_dbn()
            if bid is not None and ask is not None and bid >= ask:
                if sub.cross_start_ns is None and not _is_stale_cross(bid, ask):
                    _fail(
                        f"instrument {instrument_id} crossed (bid {bid} >= ask "
                        f"{ask}) with no active transient-cross timer"
                    )


# --- the sole book mutator (spine AD-9) ---------------------------------


def _book_side(side_char: str) -> BookSide:
    if side_char == _SIDE_BID:
        return BookSide.BID
    if side_char == _SIDE_ASK:
        return BookSide.ASK
    _fail(f"order event carries no market side: {side_char!r}")


def apply_event(book: Book, record: MboRecord) -> None:
    """Fold one MBO ``record`` into ``book`` -- the *only* code that mutates it.

    Handles ``A / C / M / T / F / R / N`` per the module docstring and the
    spec's I/O matrix. A malformed record raises :class:`BookInconsistency`.

    Args:
        book: the book to mutate in place.
        record: a normalized MBO record satisfying :class:`MboRecord`
            (``events.BookEvent`` in production).
    """
    ts = int(record.ts_event)
    if ts < book.last_ts_ns:
        _fail(
            f"ts_event {ts} precedes the last folded ts {book.last_ts_ns} "
            f"-- AD-20 guarantees a monotonic total event order"
        )
    book.last_ts_ns = ts

    action = str(record.action)
    if action == _ADD:
        _apply_add(book, record)
    elif action == _CANCEL:
        _apply_cancel(book, record)
    elif action == _MODIFY:
        _apply_modify(book, record)
    elif action == _CLEAR:
        _apply_clear(book, record)
    elif action in (_TRADE, _FILL, _NONE):
        # T / F / N do not affect the book (see module docstring). A resting
        # order hit by a trade is reduced by its own subsequent C / M.
        pass
    else:
        _fail(f"unhandled MBO action {action!r}")

    _check_cross(book, record, mutated=action in _MUTATING)


def _apply_add(book: Book, record: MboRecord) -> None:
    size = int(record.size)
    if size == UNDEF_ORDER_SIZE or size <= 0:
        _fail(f"ADD with malformed size {size} (order_id {int(record.order_id)})")
    side = _book_side(str(record.side))  # validate before creating the sub-book
    instrument_id = int(record.instrument_id)
    sub = book._sub_or_create(instrument_id)
    order = RestingOrder(
        instrument_id=instrument_id,
        side=side,
        price_dbn=int(record.price_dbn),
        size=size,
        add_ts_ns=int(record.ts_event),
        sequence=int(record.sequence),
    )
    sub.add(int(record.order_id), order)


def _apply_cancel(book: Book, record: MboRecord) -> None:
    sub = book._sub(int(record.instrument_id))
    order_id = int(record.order_id)
    existing = None if sub is None else sub.orders.get(order_id)
    if sub is None or existing is None:
        book.unseen_cm_count += 1
        logger.debug(
            "unseen CANCEL for order_id %d (count=%d)", order_id, book.unseen_cm_count
        )
        return

    raw = int(record.size)
    cancel_size = existing.size if raw == UNDEF_ORDER_SIZE else raw
    if cancel_size > existing.size:
        book.overcancel_count += 1
        logger.warning(
            "over-cancel: order_id %d resting %d, cancel %d (count=%d)",
            order_id,
            existing.size,
            cancel_size,
            book.overcancel_count,
        )
        sub.remove(order_id, existing)
        return

    new_size = existing.size - cancel_size
    if new_size <= 0:
        sub.remove(order_id, existing)
    else:
        sub.resize(order_id, existing, new_size)


def _apply_modify(book: Book, record: MboRecord) -> None:
    sub = book._sub(int(record.instrument_id))
    order_id = int(record.order_id)
    existing = None if sub is None else sub.orders.get(order_id)
    if sub is None or existing is None:
        book.unseen_cm_count += 1
        logger.debug(
            "unseen MODIFY for order_id %d (count=%d)", order_id, book.unseen_cm_count
        )
        return

    if _book_side(str(record.side)) is not existing.side:
        _fail(
            f"MODIFY for order_id {order_id} flips side "
            f"{existing.side.value} -> {str(record.side)!r}"
        )

    new_price = int(record.price_dbn)
    raw = int(record.size)
    new_size = existing.size if raw == UNDEF_ORDER_SIZE else raw

    if new_price == existing.price_dbn:
        # Same price -> queue priority preserved (spine AD-8 spirit): keep the
        # original add_ts_ns / sequence, just change the size.
        if new_size <= 0:
            sub.remove(order_id, existing)
        else:
            sub.resize(order_id, existing, new_size)
        return

    # Price change -> loses queue priority: the order is re-keyed to the M
    # record's ts_event / sequence at the new price.
    sub.remove(order_id, existing)
    if new_size <= 0:
        return
    sub.add(
        order_id,
        RestingOrder(
            instrument_id=existing.instrument_id,
            side=existing.side,
            price_dbn=new_price,
            size=new_size,
            add_ts_ns=int(record.ts_event),
            sequence=int(record.sequence),
        ),
    )


def _apply_clear(book: Book, record: MboRecord) -> None:
    sub = book._sub(int(record.instrument_id))
    if sub is not None:
        sub.clear()


def _is_stale_cross(bid_dbn: int, ask_dbn: int) -> bool:
    """``True`` iff a crossed book is a cold-start ghost, not a market cross.

    The single width test: ``bid - ask`` beyond ``STALE_CROSS_MAX_TICKS`` ticks.
    Read through the module globals so a test can monkeypatch either constant
    (as ``TestCrossedMarket`` already does for ``MAX_TRANSIENT_CROSS_NS``).
    """
    return bid_dbn - ask_dbn > STALE_CROSS_MAX_TICKS * MNQ_TICK_DBN


def _check_cross(book: Book, record: MboRecord, *, mutated: bool) -> None:
    """Advance / reset the transient-cross timer for ``record``'s instrument.

    Called after every :func:`apply_event`. When the action was a book no-op
    (``T`` / ``F`` / ``N``) and no cross is currently open, there is nothing to
    do -- skip the BBO recompute.

    A cross wider than ``config.STALE_CROSS_MAX_TICKS`` (:func:`_is_stale_cross`)
    is a pre-window ghost, not a venue cross: it bumps ``stale_cross_count``
    once per episode, never arms the timer and never :func:`_fail`\\ s. If a
    *timed* cross was open when the book widened past the bound, that timer is
    dropped -- the narrow cross it was measuring is no longer the state of the
    book -- so a later narrow cross arms a fresh timer of its own. Crosses
    inside the bound are unchanged: the seal's 50 ms
    ``MAX_TRANSIENT_CROSS_NS`` still applies and still raises.
    """
    sub = book._sub(int(record.instrument_id))
    if sub is None:
        return
    # A T / F / N cannot move the BBO, so with no timer running there is nothing
    # to recompute -- including while a stale cross is open (its episode can only
    # end on a mutating event).
    if not mutated and sub.cross_start_ns is None:
        return

    bid = sub.best_bid_dbn()
    ask = sub.best_ask_dbn()
    now = int(record.ts_event)

    if bid is not None and ask is not None and bid >= ask:
        if _is_stale_cross(bid, ask):
            sub.cross_start_ns = None  # a ghost cross is never timed
            if not sub.stale_cross_open:
                sub.stale_cross_open = True
                book.stale_cross_count += 1
                logger.debug(
                    "stale (cold-start) cross on instrument %d (bid %d >= ask "
                    "%d, %d ticks) at %d -- tolerated, not timed (count=%d)",
                    int(record.instrument_id),
                    bid,
                    ask,
                    (bid - ask) // MNQ_TICK_DBN,
                    now,
                    book.stale_cross_count,
                )
            return
        sub.stale_cross_open = False
        if sub.cross_start_ns is None:
            sub.cross_start_ns = now
            logger.debug(
                "transient cross opened on instrument %d (bid %d >= ask %d) at %d",
                int(record.instrument_id),
                bid,
                ask,
                now,
            )
        duration = now - sub.cross_start_ns
        if duration > book.max_transient_cross_ns:
            book.max_transient_cross_ns = duration
        if duration >= MAX_TRANSIENT_CROSS_NS:
            _fail(
                f"crossed market on instrument {int(record.instrument_id)} "
                f"(bid {bid} >= ask {ask}) persisted {duration} ns "
                f">= MAX_TRANSIENT_CROSS_NS ({MAX_TRANSIENT_CROSS_NS})"
            )
    else:
        sub.stale_cross_open = False
        if sub.cross_start_ns is not None:
            logger.debug(
                "transient cross resolved on instrument %d after %d ns",
                int(record.instrument_id),
                now - sub.cross_start_ns,
            )
            sub.cross_start_ns = None
