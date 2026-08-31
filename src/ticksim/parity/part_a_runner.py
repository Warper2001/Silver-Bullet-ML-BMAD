"""Part A runner -- wires ``part_a``'s pure core to ``sim.simulate`` over the
real +/-90-min MBO windows (prereg §A8.2 Part A; spine AD-17).

``part_a.py`` reconstructs the live bots' orders and grades the
:class:`~src.ticksim.orders.OrderOutcome`\\ s it is *handed*; it cannot run a
simulation itself and a ``leg_unfilled`` miss comes back with
``signed_error_ticks=None``. This module closes both gaps:

* :func:`run_part_a` -- for every
  :class:`~src.ticksim.parity.part_a.ReconstructedTrade`, get its window
  :class:`~src.ticksim.events.BookEventSource` from the injected ``source_for``
  callable, :func:`~src.ticksim.sim.simulate` the trade's intents over that
  window, check the outcomes are all this trade's,
  :func:`~src.ticksim.parity.part_a.compare_fills`, resolve every
  ``leg_unfilled`` miss to a defined magnitude (spine AD-17), then
  :func:`~src.ticksim.parity.part_a.aggregate` the full error set into the
  :class:`~src.ticksim.parity.part_a.PartAResult` verdict.
* :func:`_touch_at` -- a bounded, read-only book replay returning the window
  book's BBO at a timestamp, used to price an unfilled leg's miss.

Scope (spec Never): this module does not load ``.dbn.zst`` paths, resolve which
window belongs to which trade, or filter to the front-month instrument -- that is
the caller's / a window-loader's / ``cli.py``'s job; ``run_part_a`` takes
``source_for``. It never feeds a real fill price into ``sim`` (AD-17), never
re-derives ``compare_fills`` / ``aggregate`` / the verdict rule / the sign
convention, and never evaluates Part B or the §6 verdict.

The verdict is always the ``PRIMARY`` (decision-bearing) run -- §A8.2 Part A
grades fill-price fidelity of the decision-bearing model, not the deliberately
generous ``OPTIMISTIC`` one (Design Notes). A caller may still pass
``config=OPTIMISTIC`` for a diagnostic stat line.

Dependencies (spine AD-7, widened 2026-08-30): ``sim``, ``events``, ``book``,
``orders``, ``config`` + the sibling ``part_a``. Relative imports only
(``mypy --strict`` duplicate-module-errors on the absolute form).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

from ..book import Book, apply_event
from ..config import MNQ_TICK_DBN, PRIMARY, SimConfig
from ..events import BookEventSource
from ..orders import Side
from ..sim import simulate
from .part_a import (
    FillError,
    PartAError,
    PartAResult,
    ReconstructedTrade,
    aggregate,
    compare_fills,
)

__all__ = ["PART_A_WINDOW_PAD_NS", "run_part_a"]

_NS_PER_MINUTE = 60 * 1_000_000_000

PART_A_WINDOW_PAD_NS: int = 5 * _NS_PER_MINUTE
"""Default padding added to each side of a trade's ``[min stamp, max stamp]``
span before it becomes the ``sim`` ``valid_intervals`` window (spine AD-13). 5
minutes: a boundary order -- an ``intent.submit_ts_ns`` or a ``RealFill.ts_ns``
exactly at the span edge -- must not be expired by the AD-13 half-open mask
before it can fill. Override per call with ``run_part_a(pad_ns=...)``."""


def run_part_a(
    trades: Sequence[ReconstructedTrade],
    source_for: Callable[[ReconstructedTrade], BookEventSource],
    *,
    config: SimConfig = PRIMARY,
    pad_ns: int = PART_A_WINDOW_PAD_NS,
) -> PartAResult:
    """Run Part A over ``trades`` and return the :class:`PartAResult` verdict.

    For each trade: ``source_for(trade)`` yields a single-instrument, re-iterable
    L3 :class:`~src.ticksim.events.BookEventSource` for that trade's +/-90-min
    window (front-month filtering is the caller's job -- a multi-instrument
    stream makes ``sim`` raise :class:`~src.ticksim.sim.IntentLogError`). One
    :func:`~src.ticksim.sim.simulate` call per trade over ``valid_intervals =
    [(max(0, lo - pad_ns), hi + pad_ns)]`` where ``lo`` / ``hi`` span every
    ``intent.submit_ts_ns`` and every ``RealFill.ts_ns`` of the trade. Every
    returned outcome must belong to ``trade.trade_id`` (asserted -- a foreign id
    is a ``sim`` bug); the outcomes are graded by
    :func:`~src.ticksim.parity.part_a.compare_fills`, and every ``leg_unfilled``
    miss is resolved from the window book's touch at the real fill ts plus one
    tick of adverse slip (spine AD-17). ``aggregate`` is called **once** over the
    concatenated per-trade error lists; its :class:`PartAResult` is returned
    verbatim. With ``trades == []`` no ``source_for`` call is made and the result
    is a FAIL with ``n == 0`` (the Part A N floor).

    Args:
        trades: the reconstructed trades to grade. Two entries sharing a
            ``trade_id`` -> :class:`PartAError` (their errors would double-count).
        source_for: returns the window :class:`~src.ticksim.events.BookEventSource`
            for a trade. Must be re-iterable per the AD-18 protocol — a trade
            with an unfilled leg is walked a second time by ``_touch_at``; a
            one-shot source surfaces as a ``RuntimeError`` at that point.
        config: the ``SimConfig`` to simulate under; defaults to ``PRIMARY``
            (the only decision-bearing model -- Design Notes).
        pad_ns: window padding each side of the trade's stamp span; defaults to
            :data:`PART_A_WINDOW_PAD_NS`.

    Raises:
        PartAError: an unfilled leg's miss could not be priced (the window book
            has no touch on the side the order would have crossed at its fill
            ts); a duplicate ``trade_id`` in ``trades``; a mis-ordered or
            multi-instrument window stream seen by ``_touch_at``; or a ``sim``
            outcome whose ``trade_id`` is not the trade's.
        IntentLogError: ``source_for`` returned a multi-instrument stream, or the
            intent log is not causally replayable (propagated from ``sim``).
        InvariantViolation: a simulator invariant broke on a window (from ``sim``).
        BookInconsistency: a structural book check failed, in ``sim`` or in the
            ``_touch_at`` replay (propagated from ``book``).
        ValueError: ``config.latency_ns < 0`` or a malformed interval (from ``sim``).
        OrderStateError: an illegal tracker transition (a bug; from ``orders``).
    """
    seen_ids: set[str] = set()
    all_errors: list[FillError] = []
    for trade in trades:
        if trade.trade_id in seen_ids:
            raise PartAError(
                f"duplicate trade_id {trade.trade_id!r} in trades -- its errors "
                f"would be counted twice toward the Part A N floor"
            )
        seen_ids.add(trade.trade_id)

        source = source_for(trade)
        lo, hi = _window_span(trade)
        valid_intervals = [(max(0, lo - pad_ns), hi + pad_ns)]
        outcomes, _ = simulate(source, trade.intents, config, valid_intervals)
        foreign = {o.trade_id for o in outcomes} - {trade.trade_id}
        if foreign:
            raise PartAError(
                f"sim returned outcomes for trade_id(s) {sorted(foreign)!r} while "
                f"simulating only {trade.trade_id!r} -- a sim/reconstruction bug"
            )
        graded = compare_fills(list(outcomes), trade)
        all_errors.extend(_resolve_leg_unfilled(err, trade, source) for err in graded)
    return aggregate(all_errors)


def _window_span(trade: ReconstructedTrade) -> tuple[int, int]:
    """``(lo, hi)`` spanning every intent ``submit_ts_ns`` and every
    ``RealFill.ts_ns`` of ``trade`` (spec Always: per-trade window interval).
    The ``RealFill.ts_ns`` term is load-bearing for mim-nb-reconstructed trades,
    whose fill ts is later than the submit ts."""
    stamps = [intent.submit_ts_ns for intent in trade.intents]
    stamps.extend(real.ts_ns for real in trade.real_fills)
    return min(stamps), max(stamps)


def _resolve_leg_unfilled(
    err: FillError, trade: ReconstructedTrade, source: BookEventSource
) -> FillError:
    """Resolve a ``leg_unfilled`` miss to a defined magnitude (spine AD-17).

    A miss (``err.signed_error_ticks is None``) is priced as the touch the order
    would have crossed at ``err.real_ts_ns`` **plus one ``MNQ_TICK_DBN`` of
    adverse slip**: a ``BUY`` leg pays ``best_ask_dbn + MNQ_TICK_DBN``, a
    ``SELL`` leg receives ``best_bid_dbn - MNQ_TICK_DBN``. ``signed_error_ticks``
    uses the same sign convention as
    :func:`~src.ticksim.parity.part_a.compare_fills` (positive = sim worse for
    the trader); ``miss_reason`` stays ``"leg_unfilled"`` so a resolved-from-touch
    grade is still identifiable, and ``aggregate``'s unresolved count (which keys
    on ``signed_error_ticks is None``) now sees it as resolved. A non-miss
    ``FillError`` is returned unchanged.

    Raises:
        PartAError: an unresolved ``FillError`` whose ``miss_reason`` is not
            ``"leg_unfilled"`` (a different miss class must not be silently
            repriced); or the window book has no touch on the crossed side at
            ``err.real_ts_ns`` -- an un-priceable miss (incomplete window book, a
            data fault worth surfacing).
    """
    if err.signed_error_ticks is not None:
        return err
    if err.miss_reason != "leg_unfilled":
        raise PartAError(
            f"cannot resolve {err.order_id!r} (trade {err.trade_id!r}): "
            f"unresolved FillError with miss_reason={err.miss_reason!r}, not "
            f"'leg_unfilled'"
        )
    side = _side_of(err, trade)
    best_bid_dbn, best_ask_dbn = _touch_at(source, err.real_ts_ns)
    if side is Side.BUY:
        if best_ask_dbn is None:
            raise PartAError(
                f"un-priceable unfilled {err.leg.value} leg for {err.order_id!r} "
                f"(trade {err.trade_id!r}): window book has no ask at fill ts "
                f"{err.real_ts_ns}"
            )
        sim_vwap_dbn = best_ask_dbn + MNQ_TICK_DBN
    else:
        if best_bid_dbn is None:
            raise PartAError(
                f"un-priceable unfilled {err.leg.value} leg for {err.order_id!r} "
                f"(trade {err.trade_id!r}): window book has no bid at fill ts "
                f"{err.real_ts_ns}"
            )
        sim_vwap_dbn = best_bid_dbn - MNQ_TICK_DBN
    diff = sim_vwap_dbn - err.real_dbn
    signed = (diff if side is Side.BUY else -diff) / MNQ_TICK_DBN
    return dataclasses.replace(
        err, sim_vwap_dbn=sim_vwap_dbn, signed_error_ticks=signed
    )


def _side_of(err: FillError, trade: ReconstructedTrade) -> Side:
    """The order side for ``err`` -- taken from the matching ``RealFill``
    (``FillError`` carries no side; the reconstruction is the authority)."""
    for real in trade.real_fills:
        if (real.order_id, real.leg) == (err.order_id, err.leg):
            return real.side
    raise PartAError(  # pragma: no cover - compare_fills guarantees the match
        f"no RealFill for miss {(err.order_id, err.leg.value)!r} in trade "
        f"{err.trade_id!r}"
    )


def _touch_at(source: BookEventSource, ts_ns: int) -> tuple[int | None, int | None]:
    """The window book's ``(best_bid_dbn, best_ask_dbn)`` at ``ts_ns`` (spine
    AD-17).

    A bounded, read-only replay: a fresh :class:`~src.ticksim.book.Book`, a fresh
    pass of ``source``, folding every event with ``ev.ts_event <= ts_ns`` and
    stopping at the first event past it. No fills, no tracker, no mutation of
    anything outside the local ``Book``. The ``instrument_id`` is captured from
    the first event walked; a later event with a different ``instrument_id`` ->
    :class:`PartAError` (``sim``'s own multi-instrument check is lazy and could
    miss an id that only appears after ``ts_ns``). Event ``ts_event`` must be
    non-decreasing -- a regression -> :class:`PartAError` (a mis-ordered source
    is a window-loader bug; fail loud rather than silently truncate the book).
    Returns ``(None, None)`` if no event at or before ``ts_ns`` was seen (an
    empty book cannot be priced -- the caller surfaces that as a
    :class:`PartAError`).
    """
    book = Book()
    instrument_id: int | None = None
    prev_ts: int | None = None
    for ev in source:
        if prev_ts is not None and ev.ts_event < prev_ts:
            raise PartAError(
                f"window source ts_event regressed ({ev.ts_event} < {prev_ts}) "
                f"-- a mis-ordered stream cannot be replayed for a touch"
            )
        prev_ts = ev.ts_event
        if ev.ts_event > ts_ns:
            break
        if instrument_id is None:
            instrument_id = ev.instrument_id
        elif ev.instrument_id != instrument_id:
            raise PartAError(
                f"window source is multi-instrument ({ev.instrument_id} != "
                f"{instrument_id}) -- front-month filtering is the caller's job"
            )
        apply_event(book, ev)
    if instrument_id is None:
        return (None, None)
    return book.snapshot_bbo(instrument_id)
