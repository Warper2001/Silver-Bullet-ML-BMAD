"""Part B battery runner -- ``run_part_b`` (prereg §A8.2 Part B; spine AD-16).

Pre-registration §A8.2 Part B runs >=1000 synthetic orders through the
simulator and requires **all six invariants to hold 100%** -- any violation is a
Part B FAIL regardless of Part A. :func:`run_part_b` is that battery: one
:func:`~src.ticksim.sim.simulate` over the whole synthetic
:class:`~src.ticksim.orders.OrderIntent` list, then for every
``(OrderIntent, OrderOutcome)`` pair one
:func:`~src.ticksim.parity.invariants.check_order` call, collecting every
:class:`~src.ticksim.sim.InvariantViolation` into a structured
:class:`PartBResult`.

**No book replay** (loopback 1, 2026-08-31). Invariant 5's book-liquidity half
is a ``fills.py`` construction guarantee -- treated here exactly as
``invariants.py`` already treats invariant 4's queue time-series and invariant
6's merge ordering. :data:`PART_B_COVERAGE_NOTE` states verbatim which of the
six are post-hoc-verified at >=1000-order scale here and which are
construction-guaranteed with their ``tests/unit/`` home, so a gate reader knows
exactly what the battery certifies. The synthetic-order *generator* is slice 2 --
``run_part_b`` takes ``intents``.

Dependencies (spine AD-7, parity edge): ``sim``, ``orders``, ``config``,
``invariants`` + ``events`` (for the :class:`~src.ticksim.events.BookEventSource`
type annotation only). It imports neither ``book`` nor ``_bookwalk`` -- there is
no book replay. Relative imports only (``mypy --strict`` duplicate-module-errors
on the absolute form).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ..config import PART_B_MIN_ORDERS, PRIMARY, SimConfig
from ..events import BookEventSource
from ..orders import IntentAction, OrderIntent, OrderOutcome
from ..sim import InvariantViolation, simulate
from .invariants import check_order

__all__ = [
    "PART_B_WINDOW_PAD_NS",
    "PART_B_COVERAGE_NOTE",
    "Violation",
    "PartBResult",
    "PartBError",
    "run_part_b",
]

_NS_PER_MINUTE = 60 * 1_000_000_000

PART_B_WINDOW_PAD_NS: int = 5 * _NS_PER_MINUTE
"""Default padding added to each side of the ``[min submit_ts, max submit_ts]``
span before it becomes the ``sim`` ``valid_intervals`` window (spine AD-13,
mirrors :data:`~src.ticksim.parity.part_a_runner.PART_A_WINDOW_PAD_NS`). 5
minutes: a synthetic order whose ``submit_ts_ns`` sits exactly on the span edge
must not be expired by the AD-13 half-open mask before it can fill. Override per
call with ``run_part_b(pad_ns=...)``."""

PART_B_COVERAGE_NOTE: str = (
    "Part B post-hoc verification coverage (spine AD-16, prereg §A8.2). This "
    "battery is a SCALED POST-HOC CHECK of invariants.check_order against real "
    "sim outputs at >=1000-order scale -- not a re-derivation of the "
    "simulator's internals. The label on each recorded Violation is the token "
    "shown below.\n"
    "  - label '1' (no price improvement vs the arrival touch): POST-HOC here, "
    "invariants.check_no_price_improvement.\n"
    "  - label '2' (never fills through the limit): POST-HOC here, "
    "invariants.check_within_limit.\n"
    "  - label '3' (fill ts >= arrival): POST-HOC here, "
    "invariants.check_fill_latency. The check compares against "
    "outcome.arrival_ts_ns -- the ORIGINAL arrival for a priority-preserving "
    "(size-down) replace, not a recomputed submit_ts_ns + latency_ns -- a "
    "documented invariants.py refinement of the literal prereg 'submit + "
    "latency' wording.\n"
    "  - label '4' (queue position): ENDPOINT verified POST-HOC here "
    "(invariants.check_queue_position); the full non-negative / non-increasing "
    "queue-ahead TIME SERIES is an OrderTracker construction guarantee "
    "(decrement_queue_ahead floors at 0 and never increments; "
    "set_queue_position is once-only), verified in "
    "tests/unit/test_ticksim_orders.py.\n"
    "  - label '5' covers invariant 5's cumulative-partials-<=-size half "
    "POST-HOC here (invariants.check_partials_within_size). Invariant 5's "
    "LIQUIDITY half (no fill when the book has no size at/through the price; a "
    "passive fill only once cumulative trade volume exceeds queue-ahead) is a "
    "fills.py construction guarantee (_walk_book emits only for levels with "
    "size > 0), verified in tests/unit/test_ticksim_fills.py -- NOT re-derived "
    "here and NOT separately labelled (loopback 1, 2026-08-31: an inclusive "
    "book re-walk folds the fill's own same-ts consumption and would "
    "spurious-FAIL a correct sim).\n"
    "  - label '6' (fill-ts causal trace): POST-HOC trace here, "
    "invariants.check_fill_causality; the AD-20 merge ORDERING itself "
    "(monotonic clock + (ts_event, class_rank, sequence, source_index) stable "
    "merge) is a sim construction guarantee, verified in "
    "tests/unit/test_ticksim_sim.py.\n"
    "  - labels 'time_to_fill' / 'adverse_selection' (OrderOutcome "
    "consistency): POST-HOC here, invariants.check_time_to_fill / "
    "invariants.check_adverse_selection.\n"
    "  - label 'sim': simulate itself raised an InvariantViolation mid-run "
    "(§A8.2 'any violation = FAIL')."
)
"""Verbatim statement of what the >=1000-order battery certifies (spec Always /
Design Notes). Carried on every :class:`PartBResult` as ``coverage_note``."""

_INVARIANT_LABEL_RE = re.compile(r"^invariant (\d+) ")


class PartBError(Exception):
    """The Part B battery cannot produce a meaningful verdict.

    A structural fault -- a non-``SUBMIT`` intent, an empty or duplicate
    ``order_id`` in the synthetic list, a negative ``pad_ns``, a ``sim`` run
    that returned a missing / duplicate / foreign / count-mismatched outcome
    set, an ``intent`` <-> ``outcome`` join mismatch surfaced by
    ``check_order``, or ``check_order`` faulting on a malformed outcome.
    Distinct from a recorded :class:`Violation`: a ``PartBError`` means the "do
    all six hold?" question itself is unanswerable, not that an invariant
    failed.
    """


@dataclass(frozen=True)
class Violation:
    """One invariant breach recorded during the sweep (spec Verdict).

    ``order_id`` is the offending order (``""`` for a ``simulate``-raised
    breach). ``invariant`` is ``"1"``..``"6"`` / ``"time_to_fill"`` /
    ``"adverse_selection"`` parsed from the message prefix, ``"sim"`` for a
    ``simulate``-raised breach, ``"unknown"`` if the prefix is unrecognised.
    ``message`` is the verbatim ``InvariantViolation`` text.
    """

    order_id: str
    invariant: str
    message: str


@dataclass(frozen=True)
class PartBResult:
    """The Part B verdict (spec Verdict).

    ``verdict == "PASS"`` iff ``not violations and n_orders >=
    PART_B_MIN_ORDERS and n_orders > 0``. ``n_fill_events`` is
    ``Σ len(outcome.fills)`` -- fill *events*, not contracts. ``violations`` is
    sorted by ``(order_id, invariant)`` for a deterministic FAIL report.
    ``coverage_note`` is :data:`PART_B_COVERAGE_NOTE`.
    """

    n_orders: int
    n_fill_events: int
    violations: tuple[Violation, ...]
    verdict: str
    reason: str
    coverage_note: str


def run_part_b(
    intents: Sequence[OrderIntent],
    source: BookEventSource,
    *,
    config: SimConfig = PRIMARY,
    pad_ns: int = PART_B_WINDOW_PAD_NS,
) -> PartBResult:
    """Run the Part B invariant battery over ``intents`` and return the verdict.

    One :func:`~src.ticksim.sim.simulate` call over ``intents`` and ``source``
    (a single-instrument, re-iterable L3 stream -- front-month filtering is the
    caller's job; a multi-instrument stream makes ``sim`` raise
    ``IntentLogError``), with ``valid_intervals = [(max(0, lo - pad_ns), hi +
    pad_ns)]`` spanning every ``intent.submit_ts_ns``. ``source`` is consumed
    exactly once (no book replay). Then every ``(OrderIntent, OrderOutcome)``
    pair -- joined on ``order_id`` -- is run through
    :func:`~src.ticksim.parity.invariants.check_order`; each
    :class:`~src.ticksim.sim.InvariantViolation` becomes a :class:`Violation`
    and the sweep continues, so the result reports *all* violating orders.

    Args:
        intents: the synthetic order list. Every ``intent.action`` must be
            :attr:`~src.ticksim.orders.IntentAction.SUBMIT` (a ``replace`` /
            ``cancel`` reuses an ``order_id``; the battery is standalone submits
            only), every ``order_id`` must be non-empty (``""`` is reserved for
            the sim-raised :class:`Violation` sentinel), and no ``order_id`` may
            repeat.
        source: the single-instrument L3 :class:`~src.ticksim.events.BookEventSource`.
        config: the ``SimConfig`` to simulate under; defaults to ``PRIMARY``.
        pad_ns: window padding each side of the submit-ts span; must be ``>= 0``;
            defaults to :data:`PART_B_WINDOW_PAD_NS`.

    Returns:
        The :class:`PartBResult`. ``verdict == "PASS"`` iff no invariant was
        violated and ``PART_B_MIN_ORDERS <= n_orders`` with ``n_orders > 0``.

    Raises:
        PartBError: a non-``SUBMIT`` intent; an empty or duplicate ``order_id``
            in ``intents``; ``pad_ns < 0``; ``simulate`` returned a missing /
            duplicate / foreign / count-mismatched outcome set; ``check_order``
            reported an ``intent`` <-> ``outcome`` join mismatch; or
            ``check_order`` faulted on a malformed outcome.

    Pass-through from ``sim`` (not caught here -- structural faults, not
    invariant breaches): ``IntentLogError`` (multi-instrument ``source`` / a
    non-replayable intent log), ``BookInconsistency`` (a structural book check),
    ``ValueError`` (``config.latency_ns < 0`` / a malformed interval), and
    ``OrderStateError`` (an illegal tracker transition -- a bug). A
    ``simulate``-raised ``InvariantViolation`` IS caught and recorded as a FAIL.
    """
    if pad_ns < 0:
        raise PartBError(f"pad_ns must be >= 0, got {pad_ns}")

    intent_list = list(intents)
    _validate_intents(intent_list)

    stamps = [intent.submit_ts_ns for intent in intent_list]
    if stamps:
        lo, hi = min(stamps), max(stamps)
        valid_intervals = [(max(0, lo - pad_ns), hi + pad_ns)]
    else:
        # empty battery: still consume `source` once so a caller cannot mistake
        # a no-op for a run; the verdict is a FAIL on the order-count floor.
        valid_intervals = [(0, 1)]

    try:
        outcomes, _manifest = simulate(source, intent_list, config, valid_intervals)
    except InvariantViolation as exc:
        # a sim-raised invariant breach IS a Part B failure (§A8.2 "any
        # violation = FAIL"), not a crash. IntentLogError / BookInconsistency /
        # ValueError / OrderStateError still propagate (structural faults).
        return _verdict(
            len(intent_list),
            0,
            [Violation(order_id="", invariant="sim", message=str(exc))],
        )

    pairs = _pair_outcomes(intent_list, list(outcomes))

    violations: list[Violation] = []
    for intent, outcome in pairs:
        try:
            check_order(intent, outcome)
        except InvariantViolation as exc:
            message = str(exc)
            if "join mismatch" in message:
                raise PartBError(
                    f"intent/outcome join mismatch for order "
                    f"{intent.order_id!r} ({message}) -- the 'all six hold' "
                    f"verdict is meaningless on a mispaired battery"
                ) from exc
            violations.append(
                Violation(
                    order_id=outcome.order_id,
                    invariant=_invariant_label(message),
                    message=message,
                )
            )
        except Exception as exc:  # a malformed outcome -> AttributeError, etc.
            raise PartBError(
                f"check_order faulted on {outcome.order_id!r}: {exc}"
            ) from exc

    n_fill_events = sum(len(outcome.fills) for _intent, outcome in pairs)
    return _verdict(len(pairs), n_fill_events, violations)


def _validate_intents(intents: Sequence[OrderIntent]) -> None:
    """Every intent is a fresh ``SUBMIT`` with a non-empty, unique ``order_id``
    (spec Always). A non-``SUBMIT``, an empty id, or a duplicate id ->
    :class:`PartBError`."""
    seen: set[str] = set()
    for intent in intents:
        if intent.action is not IntentAction.SUBMIT:
            raise PartBError(
                f"Part B intent {intent.order_id!r} has action "
                f"{intent.action.value}, not SUBMIT -- a replace / cancel reuses "
                f"an order_id and the synthetic battery is standalone submits only"
            )
        if intent.order_id == "":
            raise PartBError(
                "Part B intent has an empty order_id -- '' is reserved for the "
                "sim-raised Violation sentinel and would be indistinguishable in "
                "the sorted FAIL report"
            )
        if intent.order_id in seen:
            raise PartBError(
                f"duplicate order_id {intent.order_id!r} in the Part B intent "
                f"list -- every synthetic order must be distinct"
            )
        seen.add(intent.order_id)


def _pair_outcomes(
    intents: Sequence[OrderIntent], outcomes: Sequence[OrderOutcome]
) -> list[tuple[OrderIntent, OrderOutcome]]:
    """Join each :class:`~src.ticksim.orders.OrderOutcome` to its
    :class:`~src.ticksim.orders.OrderIntent` on ``order_id`` (spec Always).

    ``simulate`` returns exactly one outcome per submitted intent; a missing,
    duplicate, or foreign ``order_id``, or ``len(outcomes) != len(intents)``,
    is a ``sim`` bug that leaves the battery unable to certify a complete run
    -> :class:`PartBError`. Callers downstream may assume ``len(result) ==
    len(intents)``.
    """
    if len(outcomes) != len(intents):
        raise PartBError(
            f"simulate returned {len(outcomes)} outcomes for {len(intents)} "
            f"intents -- Part B cannot certify an incomplete run"
        )
    by_id: dict[str, OrderOutcome] = {}
    for outcome in outcomes:
        if outcome.order_id in by_id:
            raise PartBError(
                f"simulate returned two outcomes for order_id "
                f"{outcome.order_id!r} -- a sim bug"
            )
        by_id[outcome.order_id] = outcome

    intent_ids = {intent.order_id for intent in intents}
    foreign = sorted(set(by_id) - intent_ids)
    if foreign:
        raise PartBError(
            f"simulate returned outcomes for unknown order_id(s) {foreign!r} "
            f"-- a sim bug"
        )

    # count equality + no duplicate + no foreign => every intent has exactly
    # one match in `by_id`; no missing-match branch is reachable.
    return [(intent, by_id[intent.order_id]) for intent in intents]


def _invariant_label(message: str) -> str:
    """Parse the ``check_*`` message prefix into a single-token invariant label
    (spec Always). ``"unknown"`` for an unrecognised prefix. The
    ``"intent/outcome join mismatch"`` prefix never reaches here -- ``run_part_b``
    re-raises it as :class:`PartBError` first."""
    match = _INVARIANT_LABEL_RE.match(message)
    if match is not None:
        return match.group(1)
    if message.startswith("time_to_fill consistency"):
        return "time_to_fill"
    if message.startswith("adverse_selection structural"):
        return "adverse_selection"
    return "unknown"


def _verdict(
    n_orders: int,
    n_fill_events: int,
    violations: Sequence[Violation],
) -> PartBResult:
    """Build the :class:`PartBResult` (spec Verdict). ``PASS`` iff no violation
    and ``n_orders >= PART_B_MIN_ORDERS`` with ``n_orders > 0``; otherwise
    ``FAIL`` with ``reason`` naming every cause and, on a violation FAIL, a
    sorted per-label count breakdown."""
    ordered = tuple(sorted(violations, key=lambda v: (v.order_id, v.invariant)))
    causes: list[str] = []
    if n_orders == 0 or n_orders < PART_B_MIN_ORDERS:
        # n_orders == 0 always FAILs -- a zero-order battery verified nothing,
        # regardless of a misconfigured PART_B_MIN_ORDERS.
        causes.append(
            f"order count {n_orders} < PART_B_MIN_ORDERS ({PART_B_MIN_ORDERS})"
        )
    if ordered:
        counts = Counter(v.invariant for v in ordered)
        breakdown = ", ".join(f"{label}={counts[label]}" for label in sorted(counts))
        causes.append(f"{len(ordered)} invariant violation(s) [{breakdown}]")

    if causes:
        verdict, reason = "FAIL", "; ".join(causes)
    else:
        verdict = "PASS"
        reason = (
            f"all invariants held across {n_orders} orders "
            f"(>= PART_B_MIN_ORDERS {PART_B_MIN_ORDERS})"
        )

    return PartBResult(
        n_orders=n_orders,
        n_fill_events=n_fill_events,
        violations=ordered,
        verdict=verdict,
        reason=reason,
        coverage_note=PART_B_COVERAGE_NOTE,
    )
