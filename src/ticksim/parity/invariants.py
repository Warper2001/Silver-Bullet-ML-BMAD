"""The six Part-B parity invariants as pure assertion functions (spine AD-16).

Pre-registration §A8.2 Part B runs >=1000 synthetic orders through the
simulator and requires **all six invariants** to hold 100%. Spine AD-16 pins
them to a single definition site consumed by both ``parity/part_b.py`` and
``tests/unit/`` -- so the large-N gate run and the unit tests check the *same*
thing, not "subtly different things".

Every ``check_*`` function is **pure**: its inputs are a frozen
:class:`~src.ticksim.orders.OrderIntent` / :class:`~src.ticksim.orders.OrderOutcome`
(plus scalar params); it performs no I/O and no mutation, touches no
``book`` / ``events`` / ``fills`` / ``report`` and never runs a ``SimRun``; and
it either returns ``None`` (the invariant holds) or raises
:class:`~src.ticksim.sim.InvariantViolation` with a message naming the invariant
number, the ``order_id`` and the offending values. Prices in messages stay in
DBN 1e-9 fixed-point units.

Scope of the partial invariants (spine AD-16 / spec Design Notes):

* **Invariant 4.** The full "non-negative and non-increasing until terminal"
  *time series* is a live :class:`~src.ticksim.orders.OrderTracker` guarantee
  (``decrement_queue_ahead`` floors at 0 and never increments;
  ``set_queue_position`` is once-only) already asserted by
  ``tests/unit/test_ticksim_orders.py``. :func:`check_queue_position` checks the
  serialized endpoint only -- it does not rebuild the tracker.
* **Invariant 6.** The AD-20 "only events with ts <= clock" property is a
  sim-loop construction guarantee (monotonic clock + stable merge) covered by
  ``tests/unit/test_ticksim_sim.py``. :func:`check_fill_causality` checks the
  trace it leaves on the ``OrderOutcome``.
* **Invariant 5's liquidity half** ("no fill occurs when the book has no
  liquidity at or through the order's price") has no post-hoc ``OrderOutcome``
  form -- it needs the book state at each fill tick. It is a ``fills.py``
  construction guarantee verified by ``tests/unit/test_ticksim_fills.py``;
  ``parity/part_b.py`` MAY add its own book cross-check as it has the book.
  This module does not own it. :func:`check_partials_within_size` covers only
  invariant 5's cumulative-size half (5a).
"""

from __future__ import annotations

from ..orders import OrderIntent, OrderKind, OrderOutcome, Side, TerminalState
from ..sim import InvariantViolation

__all__ = [
    "check_no_price_improvement",
    "check_within_limit",
    "check_fill_latency",
    "check_queue_position",
    "check_partials_within_size",
    "check_fill_causality",
    "check_time_to_fill",
    "check_adverse_selection",
    "check_order",
]

# Kind guards (spine AD-16). A passive limit filling at its limit inside the
# spread is invariant 2, never a breach of invariant 1.
_MARKETABLE_KINDS = frozenset({OrderKind.MARKETABLE, OrderKind.MARKETABLE_LIMIT})
_LIMIT_KINDS = frozenset({OrderKind.PASSIVE_LIMIT, OrderKind.MARKETABLE_LIMIT})


def check_no_price_improvement(intent: OrderIntent, outcome: OrderOutcome) -> None:
    """Invariant 1: a marketable fill never prices better than the touch
    snapshotted at the arrival tick (spine AD-16(1), AD-20).

    Kind-guarded to ``MARKETABLE`` / ``MARKETABLE_LIMIT`` outcomes only. For a
    ``BUY`` every fill must be ``>= arrival_best_ask_dbn``; for a ``SELL`` every
    fill must be ``<= arrival_best_bid_dbn``. Skipped for the side whose arrival
    quote is ``None`` (nothing to compare against).
    """
    del intent  # invariant 1 reads only the serialized outcome
    if outcome.kind not in _MARKETABLE_KINDS or not outcome.fills:
        return
    same_side_touch = (
        outcome.arrival_best_ask_dbn
        if outcome.side is Side.BUY
        else outcome.arrival_best_bid_dbn
    )
    if same_side_touch is None:
        # a marketable order that filled with no arrival quote on the side it
        # crossed cannot be bounded -- Part B must flag it, not pass it (§A8.2).
        raise InvariantViolation(
            f"invariant 1 (no price improvement) unverifiable for order "
            f"{outcome.order_id!r}: marketable {outcome.side.value} filled but "
            f"the arrival {'ask' if outcome.side is Side.BUY else 'bid'} is None"
        )
    for fill in outcome.fills:
        if outcome.side is Side.BUY and fill.px_dbn < same_side_touch:
            raise InvariantViolation(
                f"invariant 1 (no price improvement) breached for order "
                f"{outcome.order_id!r}: BUY fill px_dbn={fill.px_dbn} is below "
                f"the arrival best ask {same_side_touch}"
            )
        if outcome.side is Side.SELL and fill.px_dbn > same_side_touch:
            raise InvariantViolation(
                f"invariant 1 (no price improvement) breached for order "
                f"{outcome.order_id!r}: SELL fill px_dbn={fill.px_dbn} is above "
                f"the arrival best bid {same_side_touch}"
            )


def check_within_limit(intent: OrderIntent, outcome: OrderOutcome) -> None:
    """Invariant 2: a limit order never fills at a price through its limit
    (spine AD-16(2)).

    Kind-guarded to ``PASSIVE_LIMIT`` / ``MARKETABLE_LIMIT`` outcomes with a
    non-``None`` ``intent.limit_px_dbn``. For a ``BUY`` every fill must be
    ``<= limit``; for a ``SELL`` every fill must be ``>= limit``.
    """
    if outcome.kind not in _LIMIT_KINDS:
        return
    limit = intent.limit_px_dbn
    if limit is None:  # schema-forbidden (OrderIntent validator) -- defensive
        raise InvariantViolation(
            f"invariant 2 (within limit) breached for order "
            f"{outcome.order_id!r}: a {outcome.kind.value} order carries no "
            f"limit_px_dbn"
        )
    # A resting passive limit fills *exactly* at its limit (fills.decide emits
    # px_dbn == limit); it cannot receive price improvement either, which is
    # invariant 1's wording ("a buy never fills better than the best offer")
    # applied to the resting price. A marketable_limit walks the book and fills
    # at successively worse prices up to (not through) the limit.
    for fill in outcome.fills:
        if outcome.kind is OrderKind.PASSIVE_LIMIT:
            if fill.px_dbn != limit:
                raise InvariantViolation(
                    f"invariant 2 (within limit) breached for order "
                    f"{outcome.order_id!r}: passive_limit fill px_dbn="
                    f"{fill.px_dbn} != the resting limit {limit}"
                )
            continue
        if outcome.side is Side.BUY and fill.px_dbn > limit:
            raise InvariantViolation(
                f"invariant 2 (within limit) breached for order "
                f"{outcome.order_id!r}: BUY fill px_dbn={fill.px_dbn} is above "
                f"the limit {limit}"
            )
        if outcome.side is Side.SELL and fill.px_dbn < limit:
            raise InvariantViolation(
                f"invariant 2 (within limit) breached for order "
                f"{outcome.order_id!r}: SELL fill px_dbn={fill.px_dbn} is below "
                f"the limit {limit}"
            )


def check_fill_latency(outcome: OrderOutcome) -> None:
    """Invariant 3: no fill precedes the order's exchange arrival
    (``fill_ts_ns >= arrival_ts_ns``; spine AD-16(3)).

    ``arrival_ts_ns`` is the tick the tracker actually treated as the order's
    arrival: ``submit_ts_ns + latency_ns`` for an un-replaced order, and the
    *original* arrival for a priority-preserving (size-down) ``replace`` (whose
    ``outcome.submit_ts_ns`` is the replace's ts, so recomputing from it would
    false-fail -- review 2026-08-30). ``latency_ns`` is therefore not a
    parameter of this check.
    """
    for fill in outcome.fills:
        if fill.ts_ns < outcome.arrival_ts_ns:
            raise InvariantViolation(
                f"invariant 3 (fill latency) breached for order "
                f"{outcome.order_id!r}: fill ts_ns={fill.ts_ns} precedes the "
                f"order's arrival_ts_ns={outcome.arrival_ts_ns}"
            )


def check_queue_position(outcome: OrderOutcome) -> None:
    """Invariant 4 (structural subset): the serialized queue-position endpoint
    is consistent with the order kind (spine AD-16(4), AD-22).

    A ``passive_limit`` outcome carries **both** ``queue_rank_at_submit`` and
    ``queue_ahead_size_at_submit``, each ``>= 0``; a ``marketable`` /
    ``marketable_limit`` outcome carries **both as ``None``**. The full
    non-increasing time series is an ``OrderTracker`` guarantee tested in
    ``test_ticksim_orders.py`` -- see the module docstring.
    """
    rank = outcome.queue_rank_at_submit
    ahead = outcome.queue_ahead_size_at_submit
    if outcome.kind is OrderKind.PASSIVE_LIMIT:
        # queue position is set at the arrival tick (once the order is WORKING).
        # A passive limit that is REJECTED, or EXPIRED while still IN_FLIGHT
        # (arrival past its valid_interval end), never reached WORKING and
        # legitimately serializes both fields as None -- only require them when
        # the order demonstrably worked, i.e. it has fills or reached FILLED
        # (review 2026-08-30).
        worked = bool(outcome.fills) or outcome.terminal_state is TerminalState.FILLED
        if worked and (rank is None or ahead is None):
            raise InvariantViolation(
                f"invariant 4 (queue position) breached for order "
                f"{outcome.order_id!r}: a passive_limit that filled must carry "
                f"both queue_rank_at_submit and queue_ahead_size_at_submit, got "
                f"rank={rank}, ahead_size={ahead}"
            )
        if (rank is not None and rank < 0) or (ahead is not None and ahead < 0):
            raise InvariantViolation(  # pragma: no cover - schema ge=0 forbids it
                f"invariant 4 (queue position) breached for order "
                f"{outcome.order_id!r}: queue fields must be non-negative, got "
                f"rank={rank}, ahead_size={ahead}"
            )
    elif rank is not None or ahead is not None:
        raise InvariantViolation(
            f"invariant 4 (queue position) breached for order "
            f"{outcome.order_id!r}: a {outcome.kind.value} outcome must carry no "
            f"queue position, got rank={rank}, ahead_size={ahead}"
        )


def check_partials_within_size(intent: OrderIntent, outcome: OrderOutcome) -> None:
    """Invariant 5a: cumulative partial fills never exceed the order size
    (spine AD-16(5)).

    ``sum(f.size) <= intent.size`` always; and ``terminal_state == FILLED``
    implies the sum is *exactly* ``intent.size``. A non-``FILLED`` terminal
    (``CANCELLED`` / ``EXPIRED`` / ``REJECTED``) may sum to ``< intent.size``
    (a legitimate partial-then-terminal) but never ``>``. Invariant 5's
    liquidity half is not checked here -- see the module docstring.
    """
    total = sum(fill.size for fill in outcome.fills)
    if total > intent.size:
        raise InvariantViolation(
            f"invariant 5 (partials within size) breached for order "
            f"{outcome.order_id!r}: cumulative fill size {total} exceeds order "
            f"size {intent.size}"
        )
    if outcome.terminal_state is TerminalState.REJECTED and outcome.fills:
        raise InvariantViolation(
            f"invariant 5 (partials within size) breached for order "
            f"{outcome.order_id!r}: a REJECTED order carries {total} filled "
            f"contracts (a reject happens before the order ever works)"
        )
    if outcome.terminal_state is TerminalState.FILLED and total != intent.size:
        raise InvariantViolation(
            f"invariant 5 (partials within size) breached for order "
            f"{outcome.order_id!r}: terminal_state is FILLED but cumulative "
            f"fill size {total} != order size {intent.size}"
        )
    if total == intent.size and outcome.terminal_state is not TerminalState.FILLED:
        raise InvariantViolation(
            f"invariant 5 (partials within size) breached for order "
            f"{outcome.order_id!r}: cumulative fill size {total} == order size "
            f"but terminal_state is {outcome.terminal_state.value}, not FILLED"
        )


def check_fill_causality(outcome: OrderOutcome) -> None:
    """Invariant 6 (post-hoc trace): the fill timestamps are causally ordered
    (spine AD-16(6), AD-20).

    ``outcome.fills`` ts_ns are **non-decreasing**, and every
    ``fill.ts_ns >= outcome.arrival_ts_ns``. ``fills.decide`` stamps
    ``FillEvent.ts_ns = clock_ns``, so a fill stamped out of order or before
    arrival is the observable signature of a lookahead / clock bug. The AD-20
    merge-ordering property itself is a ``sim`` construction guarantee tested in
    ``test_ticksim_sim.py``.
    """
    prev_ts: int | None = None
    for fill in outcome.fills:
        if fill.ts_ns < outcome.arrival_ts_ns:
            raise InvariantViolation(
                f"invariant 6 (fill causality) breached for order "
                f"{outcome.order_id!r}: fill ts_ns={fill.ts_ns} precedes "
                f"arrival_ts_ns={outcome.arrival_ts_ns}"
            )
        if prev_ts is not None and fill.ts_ns < prev_ts:
            raise InvariantViolation(
                f"invariant 6 (fill causality) breached for order "
                f"{outcome.order_id!r}: fill ts_ns={fill.ts_ns} is out of order "
                f"(preceding fill ts_ns={prev_ts})"
            )
        prev_ts = fill.ts_ns


def check_time_to_fill(outcome: OrderOutcome) -> None:
    """``OrderOutcome`` consistency (orders.py names ``parity/invariants.py`` a
    co-owner of the ``terminal_state == FILLED`` <=> fills family): a
    ``time_to_fill_ns`` is present **iff** ``terminal_state == FILLED``, is
    ``>= 0``, and equals the last fill's ``ts_ns - arrival_ts_ns`` (``sim`` sets
    it from the completing-fill tick).
    """
    ttf = outcome.time_to_fill_ns
    is_filled = outcome.terminal_state is TerminalState.FILLED
    if is_filled != (ttf is not None):
        raise InvariantViolation(
            f"time_to_fill consistency breached for order {outcome.order_id!r}: "
            f"terminal_state={outcome.terminal_state.value}, time_to_fill_ns={ttf}"
        )
    if ttf is None:
        return
    if ttf < 0:
        raise InvariantViolation(
            f"time_to_fill consistency breached for order {outcome.order_id!r}: "
            f"time_to_fill_ns={ttf} is negative"
        )
    if outcome.fills:
        expected = outcome.fills[-1].ts_ns - outcome.arrival_ts_ns
        if ttf != expected:
            raise InvariantViolation(
                f"time_to_fill consistency breached for order "
                f"{outcome.order_id!r}: time_to_fill_ns={ttf} != last fill "
                f"ts_ns {outcome.fills[-1].ts_ns} - arrival_ts_ns "
                f"{outcome.arrival_ts_ns} (= {expected})"
            )


def check_adverse_selection(outcome: OrderOutcome) -> None:
    """``adverse_selection`` structural cases (spine AD-28): it is `False` for
    every ``MARKETABLE`` / ``MARKETABLE_LIMIT`` outcome (a marketable fill
    crosses the spread by design) and for every order that never filled."""
    if not outcome.adverse_selection:
        return
    if outcome.kind in _MARKETABLE_KINDS:
        raise InvariantViolation(
            f"adverse_selection structural breach for order {outcome.order_id!r}: "
            f"a {outcome.kind.value} outcome must never be adverse-flagged (AD-28)"
        )
    if not outcome.fills:
        raise InvariantViolation(
            f"adverse_selection structural breach for order {outcome.order_id!r}: "
            f"an order with no fills must never be adverse-flagged (AD-28)"
        )


def check_order(intent: OrderIntent, outcome: OrderOutcome) -> None:
    """Run the per-order invariant set for one order (spine AD-16): invariants
    1, 2, 3, 4, 5a, 6 plus the ``time_to_fill`` / ``adverse_selection``
    ``OrderOutcome`` consistency checks.

    First validates the ``intent`` <-> ``outcome`` join (``order_id`` / ``kind``
    / ``side`` / ``trade_id`` must match) -- a bad join in ``part_b.py``'s
    intent-log-to-outcome-log pairing would otherwise let every invariant check
    a mismatched pair and Part B pass 100%.

    Each ``check_*`` applies its own ``kind`` guard, so invariants that do not
    apply are no-ops. Evaluation order is fixed; the first breach raises. Does
    **not** cover invariant 5's liquidity half -- ``parity/part_b.py`` owns that
    with the book (see the module docstring).
    """
    for field in ("order_id", "kind", "side", "trade_id"):
        if getattr(intent, field) != getattr(outcome, field):
            raise InvariantViolation(
                f"intent/outcome join mismatch on {field}: intent "
                f"{getattr(intent, field)!r} != outcome {getattr(outcome, field)!r}"
            )
    check_no_price_improvement(intent, outcome)
    check_within_limit(intent, outcome)
    check_fill_latency(outcome)
    check_queue_position(outcome)
    check_partials_within_size(intent, outcome)
    check_fill_causality(outcome)
    check_time_to_fill(outcome)
    check_adverse_selection(outcome)
