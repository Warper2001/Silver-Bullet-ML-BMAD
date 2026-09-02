"""``SimRun`` -- the discrete-event orchestration loop (spine AD-20 / AD-21 / AD-22).

This module is the one place the leaves are wired together. ``simulate(...)`` is a
pure function (spine AD-2 / AD-5): it consumes only a :class:`BookEventSource` and
an iterable of :class:`~src.ticksim.orders.OrderIntent`, and never imports or calls
strategy code. Same ``(source events, intent log, config, valid_intervals)`` =>
byte-identical :class:`~src.ticksim.orders.OrderOutcome` log (spine AD-11); the
:class:`Manifest` is exempt (it carries tool versions).

Determinism (spine AD-1 / AD-11): one monotonic ``int64`` ``ts_event`` ns clock
drives everything; there is **no** wall-clock read anywhere and the sole entropy
source is ``random.Random(config.seed)`` (constructed and recorded, unused in
this slice). Every iteration over a ``dict`` / ``set`` is explicitly sorted.

The tick loop (spine AD-20). ``SimRun`` owns the merge itself -- it never calls
``events.merge_streams`` (that merges :class:`BookEvent` streams only). The "wake"
timestamps are ``{book-event ts} u {intent submit_ts_ns} u {valid_interval
bounds} u {pending deferred intent-effect ts} u {order arrival ts}`` -- a
latency-delayed arrival **is** a wake so the AD-22 queue position / arrival BBO
snapshot at exactly ``arrival_ts``, not at the next later book event. Per distinct
wake ``T`` the six ordered steps are:

  1. fold every book delta at ``T`` (``apply_event`` + ``observe_book_event``);
     always, even outside the session mask (spine AD-21: sim is the sole driver);
  2. intent records with ``submit_ts_ns == T`` -- ``SUBMIT`` -> ``tracker.submit``,
     ``CANCEL`` / ``REPLACE`` -> a deferred effect one latency hop later (AD-8);
  3. deferred ``CANCEL`` / ``REPLACE`` effects due by ``T``;
  4. arrivals -- ``activate_arrivals`` + the AD-22 queue position (once) + the
     arrival-tick BBO snapshot;
  5. fills -- ``fills.decide`` once, then ``tracker.apply_fill`` for each;
  6. adverse-selection (AD-28) -- ``_step_adverse``: latch ``hit`` on every open
     ``_AdverseCheck`` whose 1 s window contains this tick, **only when a book
     delta was folded** (the BBO moves only on a book event; a bare arrival /
     bound wake must not latch, AD-11); seal the ones whose window has closed
     (write ``adverse_selection`` on the tracker order). Runs every tick, in
     *and* out of the mask (the book is continuous), and once more at run end.
     Not a second replay (AD-14). This is the "deferred fill application" of
     AD-20's paradigm; it is not one of AD-20's three merged-stream classes.

Session mask (spine AD-13): ``valid_intervals`` are canonicalized on construction
(non-empty, sorted, and **merged** -- overlapping or contiguous windows become one,
so there is no internal seam). They are half-open ``[start, end)``. Book events
outside the mask are still folded (the book stays continuous); steps 2, 4 and 5 are
skipped there. **Step 3** (deferred ``CANCEL`` / ``REPLACE`` effects) drains
regardless of the mask so a latency hop landing past an interval end cannot wedge
the loop -- its target is by then already ``EXPIRED`` and the effect is dropped. At
each interval ``end`` every live order is force-expired (``tracker.expire_all``);
``book.check_invariants()`` runs at every boundary and at run end.
``adverse_selection`` (AD-28) is computed here by a bounded deferred-check queue
(step 6): a passive fill @P latches ``adverse_selection = True`` if, at any tick
strictly after the fill and within ``ADVERSE_SELECTION_WINDOW_NS`` (1 s), the
same-side quote has moved away (BUY: ``best_bid < P``; SELL: ``best_ask > P``; a
``None`` quote never triggers). Each check seals at its deadline or at run end;
marketable fills are never marked. Not a second replay (AD-14).

Dependencies (spine AD-7): ``.config`` / ``.book`` / ``.orders`` / ``.events`` /
``.fills`` + stdlib. Relative imports (``mypy --strict`` duplicate-module errors
on the absolute form).
"""

from __future__ import annotations

import logging
import platform
import random
from dataclasses import dataclass
from heapq import heappop, heappush
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_metadata_version
from typing import Any, Iterable, Iterator, Sequence

from .book import Book, BookSide, RestingOrder, apply_event
from .config import ADVERSE_SELECTION_WINDOW_NS, SimConfig
from .events import BookEvent, BookEventSource, MboAction
from .fills import decide, queue_model_for
from .orders import (
    IntentAction,
    LiveState,
    OrderIntent,
    OrderKind,
    OrderOutcome,
    OrderTracker,
    Side,
    TerminalState,
)

__all__ = [
    "IntentLogError",
    "InvariantViolation",
    "Manifest",
    "SimRun",
    "simulate",
]

logger = logging.getLogger(__name__)

# Book-event actions for which a pre-``apply_event`` resting-order lookup is
# meaningful (spine AD-21 ``resting_before``); ``A`` / ``T`` / ``F`` / ``R`` / ``N``
# pass ``None``.
_CM_ACTIONS: tuple[MboAction, ...] = (MboAction.CANCEL, MboAction.MODIFY)

# Progress cadence for the info log (spine Consistency Conventions: progress at a
# record interval, never per-event). The parity fixture is ~22.5M records.
_PROGRESS_EVERY = 1_000_000


@dataclass
class _AdverseCheck:
    """One in-flight AD-28 adverse-selection check for a passive fill.

    Private, sim-internal, never serialized. ``hit`` is the only mutable field --
    latched ``True`` the first tick (strictly after the fill) that the same-side
    quote is away from ``price_dbn`` within the 1 s window, and never un-latched
    (the predicate is "any point in the window", Alex-pinned 2026-08-29). The
    fill tick is ``deadline_ns - ADVERSE_SELECTION_WINDOW_NS``.
    """

    order_id: str
    price_dbn: int
    side: Side
    deadline_ns: int
    hit: bool = False


class IntentLogError(Exception):
    """The intent log is not causally replayable, or the book stream is
    multi-instrument (spine AD-2 / AD-13a).

    Raised up front (``submit_ts_ns`` regression; ``CANCEL`` / ``REPLACE`` of an
    unseen ``order_id``; duplicate ``SUBMIT`` ``order_id``; a ``submit_ts_ns``
    outside every ``valid_interval``) and during the loop if the book-event
    stream yields more than one ``instrument_id`` (the intent log / tracker
    carry none -- spine Deferred: multi-instrument).
    """


class InvariantViolation(Exception):
    """A simulator invariant the parity verdict depends on was violated
    (spine AD-13c / AD-16).

    Defined here; ``parity/invariants.py`` (a later slice) imports it from this
    module. Raised for a fill decided outside the session mask (structurally
    impossible given step 5 only runs in-mask -- a guard, not an expected path)
    and for a deferred effect that lands on a still-live order outside the mask.
    """


def _pkg_version(name: str) -> str:
    """Installed version of ``name`` for the manifest; ``"unknown"`` if absent."""
    try:
        return _pkg_metadata_version(name)
    except PackageNotFoundError:  # pragma: no cover - both are hard deps here
        return "unknown"


def _outcome_schema_version() -> int:
    """The ``OrderOutcome`` schema version (its field default; spine AD-12)."""
    default = OrderOutcome.model_fields["schema_version"].default
    return default if isinstance(default, int) else 1


def _normalize_intervals(
    intervals: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Canonicalize ``valid_intervals`` (spine AD-13): a non-empty, sorted,
    **merged** set of half-open ``[start, end)`` windows.

    Overlapping *and* contiguous (``end_i == start_{i+1}``) intervals are merged
    into one so there is no internal seam where ``expire_all`` (fired at an
    interval ``end``) would wrongly kill an order that is still inside the mask
    at the adjacent interval's ``start``.

    Raises:
        ValueError: no intervals, or any interval with ``start >= end``.
    """
    if not intervals:
        raise ValueError("valid_intervals must be non-empty (spine AD-13)")
    ordered = sorted((int(s), int(e)) for s, e in intervals)
    for s, e in ordered:
        if s >= e:
            raise ValueError(f"valid_interval [{s}, {e}) has start >= end")
    merged: list[list[int]] = [list(ordered[0])]
    for s, e in ordered[1:]:
        if s <= merged[-1][1]:  # overlap or touch -> extend
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return tuple((s, e) for s, e in merged)


def _ts_in_intervals(ts: int, intervals: Sequence[tuple[int, int]]) -> bool:
    """``True`` iff ``ts`` is inside the half-open union of ``intervals``."""
    return any(start <= ts < end for start, end in intervals)


def _validate_intent_log(
    intents: Sequence[OrderIntent], intervals: Sequence[tuple[int, int]]
) -> None:
    """Up-front intent-log validation (spine AD-2 / AD-13a) -> :class:`IntentLogError`.

    ``submit_ts_ns`` non-decreasing; every ``CANCEL`` / ``REPLACE`` references an
    ``order_id`` already seen as a ``SUBMIT``; no duplicate ``SUBMIT``
    ``order_id``; every ``submit_ts_ns`` inside the union of ``intervals``. Runs
    before any state is mutated, so a bad log produces no outcomes.
    """
    seen_submits: set[str] = set()
    last_ts: int | None = None
    for pos, intent in enumerate(intents):
        if last_ts is not None and intent.submit_ts_ns < last_ts:
            raise IntentLogError(
                f"intent log record {pos} ({intent.order_id!r}): submit_ts_ns "
                f"{intent.submit_ts_ns} decreases from {last_ts} (spine AD-2)"
            )
        last_ts = intent.submit_ts_ns
        if not _ts_in_intervals(intent.submit_ts_ns, intervals):
            raise IntentLogError(
                f"intent log record {pos} ({intent.order_id!r}): submit_ts_ns "
                f"{intent.submit_ts_ns} is outside every valid interval "
                f"(spine AD-13a)"
            )
        if intent.action is IntentAction.SUBMIT:
            if intent.order_id in seen_submits:
                raise IntentLogError(
                    f"intent log record {pos}: duplicate SUBMIT for order_id "
                    f"{intent.order_id!r}"
                )
            seen_submits.add(intent.order_id)
        else:
            ref = (
                intent.replaces_order_id
                if intent.action is IntentAction.REPLACE
                else intent.order_id
            )
            if ref not in seen_submits:
                raise IntentLogError(
                    f"intent log record {pos}: {intent.action.value} references "
                    f"order_id {ref!r} with no preceding SUBMIT (spine AD-2)"
                )


@dataclass(frozen=True)
class Manifest:
    """Run reproducibility manifest (spine AD-11 / AD-12 / Consistency Conventions).

    A frozen dataclass, not a Pydantic model: it is a run artifact, never a wire
    schema, and it is explicitly exempt from the AD-11 byte-identity guarantee
    (it carries tool versions). :meth:`to_dict` returns a JSON-safe ``dict``.
    ``sibling_run_id`` is ``None`` here -- the study layer sets it (spine AD-14).
    """

    config: dict[str, Any]
    seed: int
    valid_intervals: tuple[tuple[int, int], ...]
    degraded_days: tuple[str, ...]
    unseen_cm_count: int
    overcancel_count: int
    # Cold-start ghost-cross episodes tolerated by ``book._check_cross`` (a
    # cross wider than ``config.STALE_CROSS_MAX_TICKS``). A tolerance counter:
    # non-zero never fails a run, it tells the reader the window's book was
    # reconstructed with no opening snapshot.
    stale_cross_count: int
    max_transient_cross_ns: int
    last_ts_ns: int
    event_count: int
    intent_count: int
    oco_cascade_cancel_count: int  # OCO-cascade cancellations (events, AD-25)
    adverse_fill_count: int  # orders whose AD-28 window latched adverse (== # of
    #                          OrderOutcomes with adverse_selection == True)
    outcome_schema_version: int
    python_version: str
    databento_version: str
    sortedcontainers_version: str
    sibling_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe ``dict`` view (spine AD-12; manifest = JSON)."""
        return {
            "config": dict(self.config),
            "seed": self.seed,
            "valid_intervals": [list(iv) for iv in self.valid_intervals],
            "degraded_days": list(self.degraded_days),
            "unseen_cm_count": self.unseen_cm_count,
            "overcancel_count": self.overcancel_count,
            "stale_cross_count": self.stale_cross_count,
            "max_transient_cross_ns": self.max_transient_cross_ns,
            "last_ts_ns": self.last_ts_ns,
            "event_count": self.event_count,
            "intent_count": self.intent_count,
            "oco_cascade_cancel_count": self.oco_cascade_cancel_count,
            "adverse_fill_count": self.adverse_fill_count,
            "outcome_schema_version": self.outcome_schema_version,
            "python_version": self.python_version,
            "databento_version": self.databento_version,
            "sortedcontainers_version": self.sortedcontainers_version,
            "sibling_run_id": self.sibling_run_id,
        }


class SimRun:
    """One discrete-event simulation run (spine AD-20).

    Owns the book, the tracker, the queue model, the pending-effects heap and the
    manifest accumulator. Single-shot: :meth:`run` may be called once. A study
    drives exactly two ``SimRun``s (``PRIMARY`` + ``OPTIMISTIC``); this class
    never spawns a second pass itself (spine AD-14).
    """

    def __init__(
        self,
        config: SimConfig,
        valid_intervals: Sequence[tuple[int, int]],
        *,
        degraded_days: Sequence[str] = (),
    ) -> None:
        self.config = config
        if config.latency_ns < 0:
            raise ValueError(f"config.latency_ns must be >= 0, got {config.latency_ns}")
        self.valid_intervals: tuple[tuple[int, int], ...] = _normalize_intervals(
            valid_intervals
        )
        if isinstance(degraded_days, str):
            raise TypeError("degraded_days must be a sequence of strings, not a str")
        # sorted + de-duped so the manifest does not depend on caller order (AD-11)
        self.degraded_days: tuple[str, ...] = tuple(sorted(set(degraded_days)))
        self.book = Book()
        self.tracker = OrderTracker()
        self.queue_model = queue_model_for(config)
        # Sole entropy source for the run (spine AD-11); unused this slice.
        self._rng = random.Random(config.seed)
        self._iid: int | None = None
        self._event_count = 0
        self._intent_count = 0
        self._oco_cascade_cancel_count = 0
        # AD-28: bounded deferred adverse-selection checks (push order == list
        # order -- deterministic iteration, AD-11). `_max_deadline` gives the
        # run-end seal a `now_ns` >= every remaining deadline.
        self._adverse_checks: list[_AdverseCheck] = []
        self._adverse_fill_count = 0
        self._max_deadline = -1
        self._ran = False

    # -- public API ------------------------------------------------------

    def run(
        self,
        book_event_source: BookEventSource,
        intent_log: Iterable[OrderIntent],
    ) -> tuple[list[OrderOutcome], Manifest]:
        """Drive the loop to completion and return ``(outcomes, manifest)``.

        ``outcomes`` is submit-ordered (spine AD-12/24).

        Raises:
            IntentLogError: the intent log is not causally replayable, or the
                book stream is multi-instrument (before any state is mutated).
            InvariantViolation: a parity-verdict invariant was violated.
            RuntimeError: :meth:`run` was already called (single-shot).
            BookInconsistency: a structural book check failed (from ``book``).
            OrderStateError: an illegal tracker transition (e.g. an over-fill
                from ``fills.decide``, or an AD-28 ``set_adverse_selection`` on
                an order that is not ``FILLED`` / already finalized) -- a bug,
                propagated not caught (from ``orders``).
        """
        if self._ran:
            raise RuntimeError("SimRun.run() is single-shot (spine AD-14)")
        self._ran = True
        intents = list(intent_log)
        _validate_intent_log(intents, self.valid_intervals)
        self._loop(book_event_source, intents)
        # AD-28: seal every still-open adverse check before the tracker is
        # serialized (the order stays mutable "until run end, then serialized").
        # `_max_deadline + 1` is >= every remaining deadline so all of them seal;
        # `evaluate=False` -- no book delta at run end, only the latched `hit`
        # counts.
        self._step_adverse(self._max_deadline + 1, evaluate=False)
        self.book.check_invariants()
        outcomes = self.tracker.finalize()
        return outcomes, self._build_manifest()

    # -- helpers -------------------------------------------------------

    def _in_mask(self, ts: int) -> bool:
        return any(start <= ts < end for start, end in self.valid_intervals)

    # -- the tick loop (spine AD-20) --------------------------------------

    def _loop(self, source: BookEventSource, intents: Sequence[OrderIntent]) -> None:
        events: Iterator[BookEvent] = iter(source)
        ev_buf: BookEvent | None = next(events, None)
        ip = 0
        n_intents = len(intents)
        bounds: list[int] = sorted({b for iv in self.valid_intervals for b in iv})
        bp = 0
        # (effect_ts, tie-break seq, intent); seq keeps the intent out of every
        # heap comparison (it is not orderable).
        effects: list[tuple[int, int, OrderIntent]] = []
        effect_seq = 0
        # spine AD-22: an order's arrival tick is a wake point so its queue
        # position / arrival BBO snapshot at exactly `arrival_ts`, not at the
        # first later book event. Pushed on SUBMIT and on a price-change REPLACE.
        arrival_wakes: list[int] = []
        clock = -1  # monotonic guard (AD-1); `now_ns` never regresses

        while True:
            candidates: list[int] = []
            if ev_buf is not None:
                candidates.append(ev_buf.ts_event)
            if ip < n_intents:
                candidates.append(intents[ip].submit_ts_ns)
            if bp < len(bounds):
                candidates.append(bounds[bp])
            if effects:
                candidates.append(effects[0][0])
            if arrival_wakes:
                candidates.append(arrival_wakes[0])
            if not candidates:
                break
            now_ns = min(candidates)
            if now_ns < clock:
                raise InvariantViolation(
                    f"simulation clock regressed {clock} -> {now_ns} (spine AD-1)"
                )
            clock = now_ns
            in_mask = self._in_mask(now_ns)

            # -- STEP 1: fold every book delta at T (spine AD-20 step 1). Always,
            #    even outside the mask -- the book stays continuous (AD-13).
            events_before = self._event_count
            while ev_buf is not None and ev_buf.ts_event == now_ns:
                event = ev_buf
                if self._iid is None:
                    self._iid = event.instrument_id
                elif event.instrument_id != self._iid:
                    raise IntentLogError(
                        f"book-event stream carries >1 instrument_id "
                        f"({self._iid}, {event.instrument_id}) -- the intent log "
                        f"/ tracker carry none (spine Deferred: multi-instrument)"
                    )
                resting_before: RestingOrder | None = None
                if event.action in _CM_ACTIONS:
                    resting_before = self.book.order_by_id(
                        event.instrument_id, event.order_id
                    )
                apply_event(self.book, event)
                if event.action is MboAction.CLEAR and in_mask:
                    # prereg §2.2: an `R` is a halt/reset, expected only outside
                    # the mask. In-mask it leaves working passives with stale
                    # queue_ahead -- surfaced, not modelled (deferred-work).
                    logger.warning(
                        "book CLEAR (R) at %d is inside the session mask -- "
                        "working-order queue state may be stale",
                        now_ns,
                    )
                # spine AD-21: sim is the sole driver of the book->order seam.
                self.queue_model.observe_book_event(self.tracker, event, resting_before)
                self._event_count += 1
                if self._event_count % _PROGRESS_EVERY == 0:
                    logger.info(
                        "sim progress: %d book events folded (clock=%d)",
                        self._event_count,
                        now_ns,
                    )
                ev_buf = next(events, None)

            # -- Session-mask boundaries at T (spine AD-13). Expire every live
            #    order at an interval end; structural-check at every boundary.
            ends_here = any(end == now_ns for _, end in self.valid_intervals)
            starts_here = any(start == now_ns for start, _ in self.valid_intervals)
            if ends_here:
                self.tracker.expire_all(now_ns)
            if ends_here or starts_here:
                self.book.check_invariants()

            # -- STEP 2: intent records with submit_ts_ns == T (spine AD-20 step 2).
            if in_mask:
                while ip < n_intents and intents[ip].submit_ts_ns == now_ns:
                    intent = intents[ip]
                    ip += 1
                    self._intent_count += 1
                    if intent.action is IntentAction.SUBMIT:
                        self.tracker.submit(intent, self.config.latency_ns, now_ns)
                        heappush(arrival_wakes, now_ns + self.config.latency_ns)
                    elif intent.action in (IntentAction.CANCEL, IntentAction.REPLACE):
                        # CANCEL / REPLACE take one latency hop (spine AD-8).
                        heappush(
                            effects,
                            (now_ns + self.config.latency_ns, effect_seq, intent),
                        )
                        effect_seq += 1
                    else:  # pragma: no cover - IntentAction is a closed enum
                        raise IntentLogError(f"unknown intent action {intent.action!r}")
            elif ip < n_intents and intents[ip].submit_ts_ns == now_ns:
                # _validate_intent_log guarantees this cannot happen; a guard so a
                # mask-math bug fails loudly instead of spinning.
                raise IntentLogError(
                    f"intent {intents[ip].order_id!r} submit_ts_ns {now_ns} "
                    f"falls outside the session mask (spine AD-13a)"
                )

            # -- STEP 3: deferred CANCEL / REPLACE effects due by T (spine AD-20
            #    step 3). Drained regardless of the mask so a hop landing past an
            #    interval end cannot wedge the loop; the target is then already
            #    EXPIRED and the effect is dropped (a broker ignores a late
            #    cancel of a dead order).
            while effects and effects[0][0] <= now_ns:
                _effect_ts, _seq, effect_intent = heappop(effects)
                oid = effect_intent.order_id
                if self.tracker.terminal_state(oid) is not None:
                    logger.debug(
                        "deferred %s for %r dropped -- order already %s",
                        effect_intent.action.value,
                        oid,
                        self.tracker.terminal_state(oid),
                    )
                    continue
                if not in_mask:
                    raise InvariantViolation(
                        f"deferred {effect_intent.action.value} for live order "
                        f"{oid!r} lands at {now_ns}, outside the session mask "
                        f"(spine AD-13c)"
                    )
                if effect_intent.action is IntentAction.CANCEL:
                    self.tracker.cancel(oid, now_ns)
                else:
                    self.tracker.replace(effect_intent, self.config.latency_ns, now_ns)
                    # a price-change replace re-enters IN_FLIGHT with a fresh
                    # arrival tick -- make that a wake point too (spine AD-22).
                    if self.tracker.live_state(oid) is LiveState.IN_FLIGHT:
                        heappush(arrival_wakes, self.tracker.arrival_ts_ns(oid))

            if in_mask:
                self._step_arrivals(now_ns)
                self._step_fills(now_ns)

            # -- STEP 6: AD-28 deferred-check pass (the "deferred fill
            #    application" of AD-20's paradigm). Every tick, in *and* out of
            #    the mask -- a fill's 1 s window may cross an interval boundary
            #    and the book is continuous (AD-13). `evaluate` only when a book
            #    delta was folded this tick: the BBO can move only on a book
            #    event, so a non-book wake (an arrival, an interval bound) must
            #    not latch `hit` -- that would make the marker depend on
            #    unrelated timing (AD-11).
            self._step_adverse(now_ns, evaluate=self._event_count > events_before)

            while arrival_wakes and arrival_wakes[0] <= now_ns:
                heappop(arrival_wakes)
            while bp < len(bounds) and bounds[bp] <= now_ns:
                bp += 1

    # -- STEP 4 (spine AD-20 step 4, AD-22) --------------------------------

    def _step_arrivals(self, now_ns: int) -> None:
        activated = self.tracker.activate_arrivals(now_ns)
        if not activated:
            return
        # Before the first book event the book is genuinely empty (not
        # "unknown"): any iid keys an absent instrument, so `queue_ahead_size`
        # returns 0 and `snapshot_bbo` returns (None, None) -- correct for an
        # order that arrives with nothing resting ahead of it.
        iid = self._iid if self._iid is not None else 0
        for oid in activated:
            snap = self.tracker.snapshot(oid)
            if snap.kind is OrderKind.PASSIVE_LIMIT:
                if snap.limit_px_dbn is None:
                    raise InvariantViolation(
                        f"passive-limit order {oid!r} has no limit_px_dbn"
                    )
                book_side = BookSide.BID if snap.side is Side.BUY else BookSide.ASK
                # spine AD-22: called exactly once, at the arrival tick.
                ahead = self.queue_model.queue_ahead_size(
                    self.book,
                    iid,
                    book_side,
                    snap.limit_px_dbn,
                    snap.arrival_ts_ns,
                )
                # rank == ahead_size: queue_ahead_size returns a total, not a
                # per-order breakdown -- contracts-ahead is the rank proxy
                # (spec Design Notes; consistent with "every resting order
                # ahead of us").
                self.tracker.set_queue_position(oid, ahead, ahead)
            bid_dbn, ask_dbn = self.book.snapshot_bbo(iid)
            self.tracker.set_arrival_bbo(oid, bid_dbn, ask_dbn)

    # -- STEP 5 (spine AD-20 step 5) -------------------------------------

    def _step_fills(self, now_ns: int) -> None:
        if not self._in_mask(now_ns):  # spine AD-13c -- structurally unreachable
            raise InvariantViolation(
                f"fills.decide invoked at {now_ns}, outside the session mask "
                "(spine AD-13c)"
            )
        for fill_event in decide(self.book, self.tracker, now_ns, self.config):
            # AD-28 enqueue: read the snapshot while the order is still WORKING
            # (before `apply_fill`). A losing OCO leg whose fill this batch has
            # already cascade-cancelled is no longer WORKING -- skip it (its
            # fill is voided by `apply_fill` anyway).
            snap = (
                self.tracker.snapshot(fill_event.order_id)
                if self.tracker.live_state(fill_event.order_id) is LiveState.WORKING
                else None
            )
            cascaded = self.tracker.apply_fill(fill_event, now_ns)
            self._oco_cascade_cancel_count += len(cascaded)
            if cascaded:
                logger.debug(
                    "fill on %r cascaded OCO cancels: %s",
                    fill_event.order_id,
                    cascaded,
                )
            # Only a passive fill that actually completed the order is an
            # adverse-selection candidate (§2.1 passive-only; and
            # `set_adverse_selection` needs a FILLED order). Marketable /
            # marketable-limit fills enqueue nothing.
            if (
                snap is not None
                and snap.kind is OrderKind.PASSIVE_LIMIT
                and self.tracker.terminal_state(fill_event.order_id)
                is TerminalState.FILLED
            ):
                deadline_ns = fill_event.ts_ns + ADVERSE_SELECTION_WINDOW_NS
                self._adverse_checks.append(
                    _AdverseCheck(
                        order_id=fill_event.order_id,
                        price_dbn=fill_event.px_dbn,
                        side=snap.side,
                        deadline_ns=deadline_ns,
                        hit=False,
                    )
                )
                if deadline_ns > self._max_deadline:
                    self._max_deadline = deadline_ns

    # -- STEP 6: the AD-28 deferred-check pass ---------------------------

    def _step_adverse(self, now_ns: int, *, evaluate: bool) -> None:
        """Evaluate + seal the AD-28 adverse-selection checks (spine AD-28;
        predicate pinned by Alex 2026-08-30).

        *Evaluate* (only when ``evaluate`` -- i.e. a book delta was folded this
        tick; the BBO moves only on a book event, so a non-book wake must not
        latch, AD-11): for every open check whose window contains ``now_ns``
        strictly after the fill tick (``fill_ts < now_ns <= deadline_ns``),
        latch ``hit`` if the same-side quote is away from the fill price
        (BUY: ``best_bid_dbn < P``; SELL: ``best_ask_dbn > P``; a ``None``
        quote never triggers -- a quote that is not there did not move through
        a price).

        *Seal* (always): for every check with ``deadline_ns <= now_ns`` -- and
        for every remaining check at run end -- write ``adverse_selection =
        True`` on the tracker order iff it latched ``hit``, count it, drop it.

        O(open checks); open checks are bounded by the passive fills in the last
        1 s. Deterministic push-order iteration; the list is only rebuilt when a
        check actually seals.
        """
        if not self._adverse_checks:
            return
        if evaluate:
            iid = self._iid if self._iid is not None else 0
            best_bid = self.book.best_bid_dbn(iid)
            best_ask = self.book.best_ask_dbn(iid)
            for check in self._adverse_checks:
                if check.hit:
                    continue
                fill_ts = check.deadline_ns - ADVERSE_SELECTION_WINDOW_NS
                if not (fill_ts < now_ns <= check.deadline_ns):
                    continue
                if check.side is Side.BUY:
                    if best_bid is not None and best_bid < check.price_dbn:
                        check.hit = True
                elif best_ask is not None and best_ask > check.price_dbn:
                    check.hit = True

        # Push order == non-decreasing deadline (deadline = fill.ts_ns + a
        # constant; fills are appended in tick order). So if the oldest check's
        # deadline is still in the future, nothing seals -- keep the list as-is,
        # no realloc (this is the hot path over ~22.5M ticks).
        if self._adverse_checks[0].deadline_ns > now_ns:
            return
        survivors: list[_AdverseCheck] = []
        for check in self._adverse_checks:
            if check.deadline_ns <= now_ns:
                if check.hit:
                    self.tracker.set_adverse_selection(check.order_id, True)
                    self._adverse_fill_count += 1
            else:
                survivors.append(check)
        self._adverse_checks = survivors

    # -- manifest ------------------------------------------------------

    def _build_manifest(self) -> Manifest:
        return Manifest(
            config=self.config.model_dump(mode="json"),  # enum -> str, JSON-safe
            seed=self.config.seed,
            valid_intervals=self.valid_intervals,
            degraded_days=self.degraded_days,
            unseen_cm_count=self.book.unseen_cm_count,
            overcancel_count=self.book.overcancel_count,
            stale_cross_count=self.book.stale_cross_count,
            max_transient_cross_ns=self.book.max_transient_cross_ns,
            last_ts_ns=self.book.last_ts_ns,
            event_count=self._event_count,
            intent_count=self._intent_count,
            oco_cascade_cancel_count=self._oco_cascade_cancel_count,
            adverse_fill_count=self._adverse_fill_count,
            outcome_schema_version=_outcome_schema_version(),
            python_version=platform.python_version(),
            databento_version=_pkg_version("databento"),
            sortedcontainers_version=_pkg_version("sortedcontainers"),
            sibling_run_id=None,
        )


def simulate(
    book_event_source: BookEventSource,
    intent_log: Iterable[OrderIntent],
    config: SimConfig,
    valid_intervals: Sequence[tuple[int, int]],
    *,
    degraded_days: Sequence[str] = (),
) -> tuple[list[OrderOutcome], Manifest]:
    """Pure entry point (spine AD-2 / AD-5 / AD-11).

    Consumes a :class:`BookEventSource` and an iterable of
    :class:`~src.ticksim.orders.OrderIntent`; returns the submit-ordered
    :class:`~src.ticksim.orders.OrderOutcome` log and the run :class:`Manifest`.
    Same ``(source events, intent log, config, valid_intervals)`` =>
    byte-identical outcome log (the manifest is exempt).

    Args:
        book_event_source: one re-iterable L3 stream (``sim`` never merges
            multiple -- that is the caller's job).
        intent_log: the timestamped ``OrderIntent`` records (JSONL in production).
        config: the frozen ``SimConfig`` (``PRIMARY`` or ``OPTIMISTIC``).
        valid_intervals: half-open ``[start_ns, end_ns)`` session windows.
        degraded_days: Databento-``degraded`` day identifiers, recorded in the
            manifest (**not** auto-excluded -- spine AD-13).

    Raises:
        IntentLogError: the intent log is not causally replayable, or the book
            stream is multi-instrument.
        InvariantViolation: a simulator invariant the parity verdict depends on
            was violated.
        ValueError: ``config.latency_ns < 0``, or malformed ``valid_intervals``
            (empty, ``start >= end``).
        BookInconsistency: a structural book check failed (propagated from
            ``book``).
        OrderStateError: an illegal tracker transition (propagated from
            ``orders``) -- a bug, not an analyst-facing condition.
    """
    run = SimRun(config, valid_intervals, degraded_days=degraded_days)
    return run.run(book_event_source, intent_log)
