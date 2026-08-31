"""Deterministic Part B synthetic-order generator -- ``generate_synthetic_orders``.

Pre-registration §A8.2 Part B needs "**>= 1000 synthetic orders** (mix of
marketable and passive limit, both sides, sizes 1-5) at random timestamps across
the Tranche 1 data".  ``parity.part_b.run_part_b`` (already built) consumes a
pre-made :class:`~src.ticksim.orders.OrderIntent` list; nothing produced one, and
a hand-written 1000-line JSONL is not reproducible.  This module is that
producer.

:func:`generate_synthetic_orders` draws ``ceil(n * _OVERGEN_FACTOR)`` candidate
orders with random ``(ts, side, kind, size)`` in ``[lo_ns, hi_ns)``, prices the
two limit kinds off the book's BBO at each candidate's timestamp via **one**
:class:`~src.ticksim.parity._bookwalk.BookReplay` forward pass, then emits ``n``
of the priceable ones **evenly spaced across the window**.  The result is
submit-ts-sorted and feeds straight into ``run_part_b``.

Determinism (spine AD-11): the only entropy is ``random.Random(seed)`` -- no
wall-clock, no ``os.urandom``, no ``set`` iteration-order dependence.  The same
``(source events, lo_ns, hi_ns, n, seed)`` returns a **byte-identical** list.
Every ``rng`` call happens up front in a fixed per-candidate order -- ``ts``,
``side``, ``kind``, ``size``, ``offset`` -- *always*, even for a ``MARKETABLE``
candidate where ``offset`` is then discarded, so a ``kind`` draw can never
desync the stream.  Candidates are over-generated so limit orders that cannot be
priced (thin book, warm-up gap, transient cross) can be dropped and still leave
``n``; the book pass keeps every priceable candidate (a ``MARKETABLE`` is always
keepable) and :class:`SyntheticError` is raised if fewer than ``n`` price.  The
``n`` emitted orders are picked at evenly-spaced indices across the priceable
(ts-sorted) list, so they span the whole ``[lo_ns, hi_ns)`` window rather than
its earliest fraction.

Scope (spine AD-2 / AD-4): this module produces intents and nothing else.  It
never imports ``sim`` / ``part_b`` / ``part_a`` / ``report`` / ``databento``,
never calls ``simulate``, never emits a ``replace`` / ``cancel`` / OCO / multi-
leg intent, and does no front-month filtering or ``.dbn.zst`` path resolution --
``source`` is a single-instrument, re-iterable L3 stream and front-month
filtering is the caller's job (the one ``BookReplay`` pass fails closed on a
multi-instrument stream with :class:`~src.ticksim.parity._bookwalk.BookWalkError`).

Dependencies (spine AD-7, parity edge): ``orders``, ``config``, ``_bookwalk``,
``events``.  Relative imports only (``mypy --strict`` duplicate-module-errors on
the absolute form).
"""

from __future__ import annotations

import logging
import math
import random
from typing import NamedTuple

from ..config import MNQ_TICK_DBN, PART_B_MIN_ORDERS
from ..events import BookEventSource
from ..orders import IntentAction, Leg, OrderIntent, OrderKind, Side
from ._bookwalk import BookReplay

__all__ = ["SyntheticError", "generate_synthetic_orders"]

logger = logging.getLogger(__name__)


# --- tuning knobs (NOT seal-bound -- the seal fixes only "mix ... both sides,
#     sizes 1-5" and n >= 1000; these are free to tune) -----------------------

_KINDS: tuple[OrderKind, ...] = (
    OrderKind.MARKETABLE,
    OrderKind.MARKETABLE_LIMIT,
    OrderKind.PASSIVE_LIMIT,
)
"""The three order kinds drawn from, in the fixed order the weights below key on."""

_KIND_WEIGHTS: tuple[int, int, int] = (1, 1, 1)
"""Roughly equal weight for all three kinds.  The seal asks for "marketable
**and** passive limit"; marketable-limit is the natural third.  Equal weight
gives each meaningful representation in a 1000-order battery."""

_MAX_OFFSET_TICKS: int = 5
"""Largest tick offset a limit order is placed from its reference touch.  A
``MARKETABLE_LIMIT`` crosses up to ``offset`` ticks through the touch; a
``PASSIVE_LIMIT`` rests up to ``offset`` ticks away from it.  BBO +/- an integer
number of ``MNQ_TICK_DBN`` stays on the tick grid -- no snapping."""

_OVERGEN_FACTOR: float = 2.0
"""Candidates drawn = ``ceil(n * _OVERGEN_FACTOR)``.  The surplus absorbs limit
candidates that cannot be priced at their timestamp (thin book, warm-up gap,
transient cross).  2.0 clears a fully one-sided book -- which drops one third of
all candidates (the two kind/side combinations that need the absent touch).
Bump it if a real Tranche-1 window's warm-up gaps prove worse; the
``logger.debug`` drop breakdown tells you the rate."""

assert _OVERGEN_FACTOR > 1.0, "_OVERGEN_FACTOR must exceed 1.0 to over-generate"

_ID_PREFIX: str = "synthetic-"
"""``order_id`` / ``trade_id`` prefix.  The emit index (zero-padded to ``n``'s
width) follows, so ids are stable and sort in emit order."""


class SyntheticError(Exception):
    """The synthetic-order request cannot be satisfied.

    Raised for bad bounds (``lo_ns < 0``, ``lo_ns >= hi_ns``, ``n < 1``,
    ``hi_ns - lo_ns < n``), a window too sparse to price ``n`` orders after
    every over-generated candidate has been tried, or an ``OrderKind`` the
    pricer does not handle.  A multi-instrument / mis-ordered ``source`` surfaces
    as :class:`~src.ticksim.parity._bookwalk.BookWalkError` from the book pass,
    not this exception.
    """


class _Candidate(NamedTuple):
    """One drawn-up-front candidate order, before the book pass prices it."""

    ts_ns: int
    side: Side
    kind: OrderKind
    size: int
    offset_ticks: int


def generate_synthetic_orders(
    source: BookEventSource,
    lo_ns: int,
    hi_ns: int,
    *,
    n: int = PART_B_MIN_ORDERS,
    seed: int = 0,
) -> list[OrderIntent]:
    """Draw ``ceil(n * _OVERGEN_FACTOR)`` candidates and emit ``n`` for Part B.

    Every emitted order is a standalone ``SUBMIT`` with a random
    ``(ts, side, kind, size)``; the two limit kinds carry a tick-grid
    ``limit_px_dbn`` derived from the BBO at their ``submit_ts_ns``.  The ``n``
    emitted orders are picked at evenly-spaced indices across the priceable
    (ts-sorted) candidate list, so they span the whole ``[lo_ns, hi_ns)``
    window.

    Args:
        source: a single-instrument, re-iterable L3
            :class:`~src.ticksim.events.BookEventSource` (front-month filtering
            is the caller's job -- the one ``BookReplay`` pass fails closed on a
            multi-instrument stream).
        lo_ns: inclusive lower bound of the submit-timestamp window.
        hi_ns: exclusive upper bound of the submit-timestamp window.
        n: number of orders to return.  Defaults to
            :data:`~src.ticksim.config.PART_B_MIN_ORDERS` (1000).
        seed: the sole entropy source (spine AD-11).

    Returns:
        Exactly ``n`` :class:`~src.ticksim.orders.OrderIntent`s, sorted by
        ``submit_ts_ns``, every one ``action=SUBMIT`` / ``leg=ENTRY`` /
        ``oco_group_id=None`` / ``replaces_order_id=None``, ``trade_id ==
        order_id``.  ``MARKETABLE`` orders carry ``limit_px_dbn=None``; the two
        limit kinds carry a positive tick-grid price.

    Raises:
        SyntheticError: ``lo_ns < 0``, ``lo_ns >= hi_ns``, ``n < 1``,
            ``hi_ns - lo_ns < n``, fewer than ``n`` candidates priced, or an
            unhandled ``OrderKind``.
        BookWalkError: ``source`` is multi-instrument or mis-ordered (from
            :class:`~src.ticksim.parity._bookwalk.BookReplay`).
        BookInconsistency: a structural book check failed while folding the
            source (pass-through from ``book.apply_event`` via ``BookReplay``).

    Note:
        The BBO is read at ``submit_ts_ns``, not ``arrival_ts_ns =
        submit_ts_ns + latency_ns``.  By fill time the market has moved, so the
        ``MARKETABLE_LIMIT`` vs ``PASSIVE_LIMIT`` label is nominal --
        ``run_part_b``'s invariant checks compare against the actual arrival
        state regardless (see Design Notes).
    """
    if lo_ns < 0 or lo_ns >= hi_ns:
        raise SyntheticError(
            f"bad window bounds: need 0 <= lo_ns < hi_ns, got "
            f"lo_ns={lo_ns}, hi_ns={hi_ns}"
        )
    if n < 1:
        raise SyntheticError(f"n must be >= 1, got {n}")
    if hi_ns - lo_ns < n:
        raise SyntheticError(
            f"window is {hi_ns - lo_ns} ns wide -- narrower than n={n}; it "
            f"cannot spread {n} orders across distinct-enough timestamps"
        )

    rng = random.Random(seed)
    n_candidates = math.ceil(n * _OVERGEN_FACTOR)
    candidates: list[_Candidate] = []
    for _ in range(n_candidates):
        ts_ns = rng.randrange(lo_ns, hi_ns)
        side = rng.choice([Side.BUY, Side.SELL])
        kind = rng.choices(_KINDS, weights=_KIND_WEIGHTS)[0]
        size = rng.randint(1, 5)
        # Drawn for every candidate (even MARKETABLE, where it is discarded) so
        # the rng stream stays in lock-step regardless of the kind draw (AD-11).
        offset_ticks = rng.randint(0, _MAX_OFFSET_TICKS)
        candidates.append(_Candidate(ts_ns, side, kind, size, offset_ticks))

    # Stable sort by ts -- ties keep draw order.  One forward BookReplay pass,
    # advance_to in ts order; price every candidate, collect the priceable ones.
    ordered = sorted(candidates, key=lambda c: c.ts_ns)
    replay = BookReplay(source)
    priceable: list[tuple[_Candidate, int | None]] = []
    dropped = 0
    for cand in ordered:
        replay.advance_to(cand.ts_ns)
        limit_px_dbn = _price_limit(cand, replay)
        if cand.kind is not OrderKind.MARKETABLE and limit_px_dbn is None:
            dropped += 1  # un-priceable limit (thin book / warm-up / cross)
            continue
        priceable.append((cand, limit_px_dbn))

    n_marketable = sum(1 for c, _ in priceable if c.kind is OrderKind.MARKETABLE)
    n_limit = len(priceable) - n_marketable
    logger.debug(
        "synthetic: %d marketable, %d limit kept, %d dropped of %d candidates",
        n_marketable,
        n_limit,
        dropped,
        len(ordered),
    )

    if len(priceable) < n:
        raise SyntheticError(
            f"only {len(priceable)} of {n} synthetic orders priceable "
            f"({n_marketable} marketable, {n_limit} limit kept, {dropped} "
            f"dropped of {len(ordered)} candidates) -- widen the window or "
            f"raise _OVERGEN_FACTOR"
        )

    picks = _evenly_spaced_indices(len(priceable), n)
    width = len(str(n))
    out: list[OrderIntent] = []
    for emit_index, src_index in enumerate(picks):
        cand, limit_px_dbn = priceable[src_index]
        order_id = f"{_ID_PREFIX}{emit_index:0{width}d}"
        out.append(
            OrderIntent(
                action=IntentAction.SUBMIT,
                order_id=order_id,
                trade_id=order_id,
                leg=Leg.ENTRY,
                kind=cand.kind,
                side=cand.side,
                size=cand.size,
                limit_px_dbn=limit_px_dbn,
                submit_ts_ns=cand.ts_ns,
                oco_group_id=None,
                replaces_order_id=None,
            )
        )

    # Picks were taken in increasing index order over a ts-sorted list, so `out`
    # is already ts-sorted; a stable re-sort is the cheap guarantee.
    return sorted(out, key=lambda o: o.submit_ts_ns)


def _evenly_spaced_indices(length: int, n: int) -> list[int]:
    """``n`` indices spread across ``range(length)`` (``length >= n >= 1``).

    ``round(k * (length - 1) / (n - 1))`` for ``k in range(n)``; ``n == 1`` ->
    ``[0]``.  Distinct whenever ``length > n`` (the step exceeds 1.0) and
    exactly ``0..n-1`` when ``length == n``.
    """
    if n == 1:
        return [0]
    last = length - 1
    return [round(k * last / (n - 1)) for k in range(n)]


def _price_limit(cand: _Candidate, replay: BookReplay) -> int | None:
    """The ``limit_px_dbn`` for ``cand`` given the book folded to its ``ts_ns``.

    ``None`` for a ``MARKETABLE`` candidate, and ``None`` (the caller drops the
    candidate) for a limit candidate that cannot be priced: the reference touch
    is absent at that tick, the book is crossed (``bid >= ask``, both non-``None``
    -- a transient CME cross is not a valid reference), or the computed price is
    ``<= 0`` (would trip ``OrderIntent``'s ``gt=0`` validator).

    * ``MARKETABLE_LIMIT`` -- crosses / marketable up to ``offset`` ticks
      through the touch: a ``BUY`` is ``ask + offset`` ticks, a ``SELL`` is
      ``bid - offset`` ticks.
    * ``PASSIVE_LIMIT`` -- rests ``offset`` ticks on its own side of the touch:
      a ``BUY`` is ``bid - offset`` ticks, a ``SELL`` is ``ask + offset`` ticks.

    An ``OrderKind`` this function does not handle -> :class:`SyntheticError`
    (an explicit ``else`` -- never a silent passive-formula fallthrough).
    """
    if cand.kind is OrderKind.MARKETABLE:
        return None
    instrument_id = replay.instrument_id
    if instrument_id is None:
        return None  # no event folded yet -- an empty book cannot be priced
    best_bid_dbn, best_ask_dbn = replay.book.snapshot_bbo(instrument_id)
    if (
        best_bid_dbn is not None
        and best_ask_dbn is not None
        and best_bid_dbn >= best_ask_dbn
    ):
        return None  # transient CME crossed / locked book -- not a valid price
    offset_dbn = cand.offset_ticks * MNQ_TICK_DBN

    if cand.kind is OrderKind.MARKETABLE_LIMIT:
        if cand.side is Side.BUY:
            limit = None if best_ask_dbn is None else best_ask_dbn + offset_dbn
        else:
            limit = None if best_bid_dbn is None else best_bid_dbn - offset_dbn
    elif cand.kind is OrderKind.PASSIVE_LIMIT:
        if cand.side is Side.BUY:
            limit = None if best_bid_dbn is None else best_bid_dbn - offset_dbn
        else:
            limit = None if best_ask_dbn is None else best_ask_dbn + offset_dbn
    else:
        raise SyntheticError(f"unhandled kind {cand.kind}")

    if limit is not None and limit <= 0:
        return None
    return limit
