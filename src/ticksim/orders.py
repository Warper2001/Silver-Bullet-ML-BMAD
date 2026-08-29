"""Frozen simulator<->consumer schemas: ``OrderIntent``, ``FillEvent``, ``Fill``, ``OrderOutcome``.

Schema layer plus the one stateful engine the spine puts in this module
(AD-6, AD-8): :class:`OrderTracker` -- the sole authority on the order
lifecycle state machine ``intent -> in_flight -> working -> {filled | cancelled
| rejected | expired}``, the per-order queue counters ``fills.decide`` reads,
OCO / bracket groups (AD-25), and the :class:`OrderOutcome` list emitted at the
end. It is a plain mutable class alongside the frozen Pydantic schemas -- the
same "engine beside the models" pattern ``book._InstrumentBook`` uses beside
:class:`~src.ticksim.book.RestingOrder`. It is a leaf: it imports nothing from
``src.ticksim`` (AD-7); ``latency_ns`` is passed in as a plain ``int``, never a
``SimConfig``.

Every model is Pydantic v2 with ``ConfigDict(frozen=True, extra="forbid")``
(spine AD-10/12/19/23): these are JSONL wire records carrying ``schema_version``,
so an unknown key is a schema mismatch and must be rejected, not dropped. Every
numeric field is a strict ``int`` -- ns time, DBN 1e-9 fixed-point price, sizes,
counters -- with no ``float`` anywhere on the four schemas (spine AD-10). Money
never appears here (spine AD-24).

Field lists are pinned to the architecture spine:
  * ``OrderIntent``  -- AD-23
  * ``FillEvent``    -- AD-19
  * ``Fill``         -- AD-12
  * ``OrderOutcome`` -- AD-12

Cross-field / cross-order invariants (``terminal_state == FILLED`` <=> non-empty
``fills``; ``marketable`` => no queue position; ``arrival_ts_ns >= submit_ts_ns``;
non-crossed arrival BBO; cumulative fill size <= order size) are **not** enforced
here. Their single owner is ``OrderTracker`` / ``sim.py`` / ``book.py`` /
``parity/invariants.py`` per the spine; duplicating them in the schema layer
would create two owners.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import NoReturn

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "Side",
    "OrderKind",
    "IntentAction",
    "Leg",
    "TerminalState",
    "LiveState",
    "OrderIntent",
    "FillEvent",
    "Fill",
    "OrderOutcome",
    "OrderStateError",
    "OrderSnapshot",
    "OrderTracker",
]

logger = logging.getLogger(__name__)


class Side(str, Enum):
    """Order side. ``.value`` is the on-disk JSONL token."""

    BUY = "buy"
    SELL = "sell"


class OrderKind(str, Enum):
    """Frozen order-kind taxonomy (spine AD-12: ``kind`` is frozen)."""

    MARKETABLE = "marketable"
    MARKETABLE_LIMIT = "marketable_limit"
    PASSIVE_LIMIT = "passive_limit"


class IntentAction(str, Enum):
    """The three intent actions (spine AD-23).

    Replace convention (spine AD-23): a single record with ``action == REPLACE``
    that *reuses* ``order_id`` -- never cancel + new.
    """

    SUBMIT = "submit"
    CANCEL = "cancel"
    REPLACE = "replace"


class Leg(str, Enum):
    """Which leg of a round trip an order is (spine AD-12).

    ``trade_id`` links an ``ENTRY`` order to its ``EXIT`` order.
    """

    ENTRY = "entry"
    EXIT = "exit"


class TerminalState(str, Enum):
    """Terminal states of the order lifecycle (spine AD-8).

    Maps 1:1 to the terminal branch of the machine ``intent -> in_flight ->
    working -> {filled | cancelled | rejected | expired}`` that
    :class:`OrderTracker` drives. The live states are :class:`LiveState`.
    """

    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class LiveState(str, Enum):
    """The two non-terminal states an order held by :class:`OrderTracker` can be
    in (spine AD-8).

    ``IN_FLIGHT`` -- submitted, latency-delayed, not yet at the exchange.
    ``WORKING``   -- resting at the exchange, eligible to fill.
    """

    IN_FLIGHT = "in_flight"
    WORKING = "working"


class OrderIntent(BaseModel):
    """One line of the JSONL intent log the simulator consumes (spine AD-23).

    Frozen and versioned, parallel to :class:`OrderOutcome`. The intent log is
    JSONL, one record per line, with non-decreasing ``submit_ts_ns`` (ordering
    is enforced later by ``sim.py``, not here).

    Edge cases (documented, not enforced in this schema-only layer):
      * ``action == CANCEL`` may carry ``limit_px_dbn`` / ``size``; for a cancel
        those are ignorable. Not rejected here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(
        default=1,
        strict=True,
        ge=1,
        description="Schema version; any field change bumps it (spine AD-23). Starts at 1.",
    )
    action: IntentAction = Field(
        ..., description="submit / cancel / replace (spine AD-23)."
    )
    order_id: str = Field(
        ...,
        min_length=1,
        description="Producer-assigned order id. A replace reuses this id (spine AD-23).",
    )
    trade_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Opaque round-trip id; links an entry order to its exit order "
            "(spine AD-12)."
        ),
    )
    leg: Leg = Field(..., description="entry / exit (spine AD-23).")
    kind: OrderKind = Field(
        ..., description="marketable / marketable_limit / passive_limit (spine AD-23)."
    )
    side: Side = Field(..., description="buy / sell (spine AD-23).")
    size: int = Field(
        ..., strict=True, gt=0, description="Order size in contracts (spine AD-23)."
    )
    limit_px_dbn: int | None = Field(
        default=None,
        strict=True,
        gt=0,
        description=(
            "Limit price in DBN 1e-9 fixed-point units; ``None`` only for a "
            "pure ``MARKETABLE`` order (spine AD-23)."
        ),
    )
    submit_ts_ns: int = Field(
        ...,
        strict=True,
        ge=0,
        description=(
            "Submission timestamp in the GLBX ``ts_event`` ns epoch (spine "
            "AD-1, AD-23)."
        ),
    )
    replaces_order_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Target order id for ``action == replace`` -- must equal "
            "``order_id`` (spine AD-23); ``None`` for every other action."
        ),
    )
    oco_group_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "OCO / bracket group id linking entry + TP + SL (spine AD-23, "
            "AD-25). ``None`` for a standalone order."
        ),
    )

    @model_validator(mode="after")
    def _check_replace_and_limit(self) -> "OrderIntent":
        """Structural rules internal to a single record (spine AD-23).

        Order-independent (mode='after'). Cross-order and cross-field lifecycle
        invariants are owned elsewhere (see module docstring).
        """
        if self.action == IntentAction.REPLACE:
            if self.replaces_order_id is None:
                raise ValueError(
                    "action == replace requires replaces_order_id to be set"
                )
            if self.replaces_order_id != self.order_id:
                raise ValueError(
                    "action == replace must reuse order_id (spine AD-23): "
                    "replaces_order_id must equal order_id"
                )
        elif self.replaces_order_id is not None:
            raise ValueError("replaces_order_id must be None unless action == replace")
        if self.kind != OrderKind.MARKETABLE and self.limit_px_dbn is None:
            raise ValueError("a limit order (kind != marketable) requires limit_px_dbn")
        return self


class FillEvent(BaseModel):
    """The fill-engine return type (spine AD-19).

    Strictly **this-tick incremental** -- the new fill delta, never cumulative.
    Carries no queue-rank or adverse-selection field. Exactly these four fields.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    order_id: str = Field(
        ...,
        min_length=1,
        description="Order that (partially) filled (spine AD-19).",
    )
    px_dbn: int = Field(
        ...,
        strict=True,
        gt=0,
        description="Fill price, DBN 1e-9 fixed-point (spine AD-19).",
    )
    size: int = Field(
        ...,
        strict=True,
        gt=0,
        description="This-tick incremental fill size in contracts (spine AD-19).",
    )
    ts_ns: int = Field(
        ...,
        strict=True,
        ge=0,
        description="Fill timestamp, ns ``ts_event`` epoch (spine AD-19).",
    )


class Fill(BaseModel):
    """One realized fill inside an :class:`OrderOutcome` (spine AD-12)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    px_dbn: int = Field(
        ...,
        strict=True,
        gt=0,
        description="Fill price, DBN 1e-9 fixed-point (spine AD-12).",
    )
    size: int = Field(
        ...,
        strict=True,
        gt=0,
        description="Fill size in contracts (spine AD-12).",
    )
    ts_ns: int = Field(
        ...,
        strict=True,
        ge=0,
        description="Fill timestamp, ns ``ts_event`` epoch (spine AD-12).",
    )


class OrderOutcome(BaseModel):
    """The frozen, versioned fills contract -- one per order (spine AD-12).

    Consumers (``report.py``, ``parity/``, downstream) read fills **only** from
    this model and config/fees/multiplier **only** from the run manifest's
    ``SimConfig`` dump. Any field change bumps ``schema_version``.

    Fields deliberately *not* duplicated here (spine AD-12): consumers recover
    ``size`` / ``limit_px_dbn`` / ``oco_group_id`` by joining to the
    ``OrderIntent`` log on ``order_id``, and pair the two legs of a round trip
    by ``trade_id``.

    No monetary field ever enters this model (spine AD-24). Every numeric leaf
    is an integer (spine AD-10) -- asserted by the JSON audit test.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(
        default=1,
        strict=True,
        ge=1,
        description="Schema version; any field change bumps it (spine AD-12). Starts at 1.",
    )
    trade_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Opaque round-trip id; pairs this order's leg with the other "
            "(spine AD-12, AD-14)."
        ),
    )
    leg: Leg = Field(..., description="entry / exit (spine AD-12).")
    order_id: str = Field(
        ...,
        min_length=1,
        description="The order this outcome is for (spine AD-12).",
    )
    kind: OrderKind = Field(
        ...,
        description="marketable / marketable_limit / passive_limit (frozen; spine AD-12).",
    )
    side: Side = Field(..., description="buy / sell (spine AD-12).")
    submit_ts_ns: int = Field(
        ...,
        strict=True,
        ge=0,
        description="Intent submission ts, ns (spine AD-12).",
    )
    arrival_ts_ns: int = Field(
        ...,
        strict=True,
        ge=0,
        description="submit_ts_ns + latency_ns -- exchange-arrival ts, ns (spine AD-12).",
    )
    terminal_state: TerminalState = Field(
        ..., description="filled / cancelled / rejected / expired (spine AD-8, AD-12)."
    )
    fills: tuple[Fill, ...] = Field(
        default=(),
        description=(
            "Realized fills, this order only; empty if never filled (spine "
            "AD-12). An immutable tuple -- the schema layer cannot be mutated "
            "after construction."
        ),
    )
    queue_rank_at_submit: int | None = Field(
        default=None,
        strict=True,
        ge=0,
        description=(
            "Queue rank at the arrival tick, computed once (spine AD-22). "
            "``None`` for a marketable order (no passive queue position)."
        ),
    )
    queue_ahead_size_at_submit: int | None = Field(
        default=None,
        strict=True,
        ge=0,
        description=(
            "Total resting size ahead of us at the arrival tick, computed once "
            "(spine AD-22). ``None`` for a marketable order."
        ),
    )
    time_to_fill_ns: int | None = Field(
        default=None,
        strict=True,
        ge=0,
        description=(
            "arrival_ts_ns -> first/last fill latency, ns; ``None`` if the "
            "order never filled (spine AD-12)."
        ),
    )
    arrival_best_bid_dbn: int | None = Field(
        default=None,
        strict=True,
        description=(
            "Best bid snapshotted at the arrival tick after folding all "
            "same-ts book deltas (spine AD-12, AD-20). ``None`` if no bid."
        ),
    )
    arrival_best_ask_dbn: int | None = Field(
        default=None,
        strict=True,
        description=(
            "Best ask snapshotted at the arrival tick after folding all "
            "same-ts book deltas (spine AD-12, AD-20). ``None`` if no ask."
        ),
    )
    adverse_selection: bool = Field(
        default=False,
        strict=True,
        description=(
            "True iff a passive fill was followed within "
            "``config.ADVERSE_SELECTION_WINDOW_NS`` by a same-side quote move "
            "through our price (prereg §2.1; spine AD-28). False when not a "
            "passive fill or no such move. A JSON boolean, not a numeric leaf "
            "-- excluded from the integer audit (spine AD-10)."
        ),
    )


# ---------------------------------------------------------------------------
# OrderTracker -- the order-lifecycle state machine (spine AD-8, AD-25)
# ---------------------------------------------------------------------------


class OrderStateError(Exception):
    """An illegal order-lifecycle transition was attempted (spine AD-8).

    The typed-exception + raise-helper style mirrors ``book.BookInconsistency``
    / ``book._fail``: every guard in :class:`OrderTracker` funnels through
    :func:`_state_error`, which logs then raises.
    """


def _state_error(message: str) -> NoReturn:
    """Log then raise -- the one place an :class:`OrderStateError` is thrown."""
    logger.warning("OrderStateError: %s", message)
    raise OrderStateError(message)


@dataclass
class _TrackedOrder:
    """Mutable per-order record -- the plain-class analogue of
    ``book.RestingOrder`` (frozen there; mutable here because an order's state,
    fill list and queue counters all evolve in place).

    Not exported. :class:`OrderTracker` is the only thing that constructs or
    mutates one; consumers get an immutable :class:`OrderSnapshot` or the final
    :class:`OrderOutcome`.
    """

    intent: OrderIntent
    state: LiveState
    arrival_ts_ns: int
    filled_qty: int = 0
    fills: list[Fill] = field(default_factory=list)
    add_ts_ns: int | None = None
    queue_rank_at_submit: int | None = None
    queue_ahead_size_at_submit: int | None = None
    queue_ahead: int = 0
    cum_trade_vol_since_arrival: int = 0
    arrival_best_bid_dbn: int | None = None
    arrival_best_ask_dbn: int | None = None
    adverse_selection: bool = False
    terminal_state: TerminalState | None = None
    terminal_ts_ns: int | None = None
    time_to_fill_ns: int | None = None
    reject_reason: str | None = None
    queue_position_set: bool = False
    arrival_bbo_set: bool = False
    adverse_selection_set: bool = False
    # set to `now_ns` when this order is cancelled by an OCO cascade, so a
    # same-tick fill on the losing leg (both legs of an OCO crossing in one
    # `fills.decide` batch) is voided rather than raising (review: edge-case).
    oco_cancelled_at: int | None = None

    @property
    def is_live(self) -> bool:
        return self.terminal_state is None


@dataclass(frozen=True)
class OrderSnapshot:
    """Immutable view of a working order's queue counters + the fields
    ``fills.decide`` needs (spine AD-5: the fill engine is a pure function and
    must not be handed anything it could mutate).

    Carries, per the spec's acceptance criterion: ``queue_ahead``,
    ``cum_trade_vol_since_arrival``, ``queue_ahead_size_at_submit``,
    ``add_ts_ns``, ``size``, ``side``, ``limit_px_dbn`` -- plus ``kind`` (the
    queue model branches marketable vs passive on it) and a few more read-only
    fields a queue model may want. Fields the ``fills.decide`` signature turns
    out to need beyond these are added when that slice lands.
    """

    order_id: str
    side: Side
    kind: OrderKind
    size: int
    limit_px_dbn: int | None
    arrival_ts_ns: int
    add_ts_ns: int | None
    filled_qty: int
    queue_rank_at_submit: int | None
    queue_ahead_size_at_submit: int | None
    queue_ahead: int
    cum_trade_vol_since_arrival: int


class OrderTracker:
    """Sole authority on the order lifecycle (spine AD-8).

    Live states are :class:`LiveState` (``IN_FLIGHT`` / ``WORKING``); terminals
    map 1:1 to :class:`TerminalState`. Every transition goes through a method
    here and any illegal one raises :class:`OrderStateError`. Every
    :class:`OrderOutcome` field :meth:`finalize` emits is derived from a
    transition this tracker performed -- authored nowhere else.

    It is a leaf (spine AD-7): stdlib + pydantic only, no ``src.ticksim``
    import, no ``databento``, no ``SimConfig``. ``latency_ns`` is a plain
    ``int`` supplied by the caller (``sim.py``).

    Ordering: :meth:`finalize` (and every ``*_order_ids`` helper) iterates in
    submit order -- the order in which :meth:`submit` first saw each id.
    """

    def __init__(self) -> None:
        # insertion order == submit order (spine: finalize is submit-ordered)
        self._orders: dict[str, _TrackedOrder] = {}
        # oco_group_id -> the set of order ids registered in that group (AD-25)
        self._oco_groups: dict[str, set[str]] = {}
        self._finalized: bool = False
        # highest `now_ns` any transition has been called with; the clock only
        # moves forward (review: edge-case -- a backwards clock silently
        # produced negative durations and non-monotonic terminal timestamps).
        self._last_now_ns: int = 0

    # --- internal guards --------------------------------------------------

    def _require_open(self) -> None:
        """Every mutating transition funnels through here first: once
        :meth:`finalize` has run the tracker is sealed and any further
        transition would be silently absent from the emitted outcome list
        (review: edge-case).
        """
        if self._finalized:
            _state_error("tracker already finalized; no further transitions")

    def _advance_clock(self, now_ns: int) -> None:
        """Guard + record the monotonic simulation clock. `now_ns` may repeat
        (many transitions share a tick) but never move backwards.
        """
        if now_ns < self._last_now_ns:
            _state_error(f"now_ns went backwards: {now_ns} < last {self._last_now_ns}")
        self._last_now_ns = now_ns

    def _get(self, order_id: str) -> _TrackedOrder:
        order = self._orders.get(order_id)
        if order is None:
            _state_error(f"unknown order_id {order_id!r}")
        return order

    def _require_live(self, order_id: str) -> _TrackedOrder:
        order = self._get(order_id)
        if order.terminal_state is not None:
            _state_error(
                f"order {order_id!r} is terminal "
                f"({order.terminal_state.value}); expected a live order"
            )
        return order

    def _require_working(self, order_id: str) -> _TrackedOrder:
        order = self._require_live(order_id)
        if order.state is not LiveState.WORKING:
            _state_error(f"order {order_id!r} is {order.state.value}; expected working")
        return order

    # --- lifecycle transitions -----------------------------------------

    def submit(self, intent: OrderIntent, latency_ns: int, now_ns: int) -> None:
        """Create the order ``IN_FLIGHT`` (spine AD-8).

        ``arrival_ts_ns = intent.submit_ts_ns + latency_ns``. Registers the
        order in ``intent.oco_group_id`` if set (AD-25). A duplicate
        ``order_id`` or ``intent.action != SUBMIT`` raises. ``now_ns`` is
        accepted for call-site symmetry with the other transitions.
        """
        del now_ns  # not needed to create an IN_FLIGHT order; kept for symmetry
        self._require_open()
        if intent.action is not IntentAction.SUBMIT:
            _state_error(f"submit() needs action == submit, got {intent.action.value}")
        if latency_ns < 0:
            _state_error(f"latency_ns must be >= 0, got {latency_ns}")
        if intent.order_id in self._orders:
            _state_error(f"duplicate submit for order_id {intent.order_id!r}")
        self._orders[intent.order_id] = _TrackedOrder(
            intent=intent,
            state=LiveState.IN_FLIGHT,
            arrival_ts_ns=intent.submit_ts_ns + latency_ns,
        )
        if intent.oco_group_id is not None:
            self._oco_groups.setdefault(intent.oco_group_id, set()).add(intent.order_id)

    def activate_arrivals(self, now_ns: int) -> list[str]:
        """Transition every ``IN_FLIGHT`` order whose ``arrival_ts_ns <=
        now_ns`` to ``WORKING`` (spine AD-8). Returns the ids activated, in
        submit order.
        """
        self._require_open()
        self._advance_clock(now_ns)
        activated: list[str] = []
        for order_id, order in self._orders.items():
            if (
                order.terminal_state is None
                and order.state is LiveState.IN_FLIGHT
                and order.arrival_ts_ns <= now_ns
            ):
                order.state = LiveState.WORKING
                order.add_ts_ns = order.arrival_ts_ns
                activated.append(order_id)
        return activated

    def apply_fill(self, fill_event: FillEvent, now_ns: int) -> list[str]:
        """Apply one this-tick incremental fill to a ``WORKING`` order (spine
        AD-8, AD-19).

        ``filled_qty += fill_event.size`` and the :class:`Fill` is appended.
        A cumulative fill past the order size raises. When ``filled_qty``
        reaches the order size the order becomes ``FILLED``; if the filled
        order is an **EXIT** leg, every other live member of its OCO group is
        cancelled at ``now_ns`` (leg-aware cascade, spine AD-25 -- bookkeeping,
        no new intent). An ENTRY-leg fill cascades nothing.

        Returns the ids cancelled by the OCO cascade (``[]`` when the fill was
        partial, an entry-leg fill, or the order has no group) -- ``sim.py``
        records these in the outcome log. If both exit legs of an OCO cross in
        the same ``fills.decide`` batch, the fill on the leg the cascade
        already cancelled *this tick* is voided (returns ``[]``), not an error.

        Caller (``sim.py``) owns exactly-once delivery: :class:`FillEvent`s come
        straight from one ``fills.decide`` call per tick and are never replayed.
        """
        self._require_open()
        self._advance_clock(now_ns)
        target = self._get(fill_event.order_id)
        if (
            target.terminal_state is TerminalState.CANCELLED
            and target.oco_cancelled_at == now_ns
        ):
            return []  # losing leg of a same-tick OCO cross -- void the fill
        order = self._require_working(fill_event.order_id)
        new_filled = order.filled_qty + fill_event.size
        if new_filled > order.intent.size:
            _state_error(
                f"over-fill on {fill_event.order_id!r}: "
                f"{order.filled_qty} + {fill_event.size} > "
                f"order size {order.intent.size}"
            )
        order.fills.append(
            Fill(
                px_dbn=fill_event.px_dbn,
                size=fill_event.size,
                ts_ns=fill_event.ts_ns,
            )
        )
        order.filled_qty = new_filled
        if order.filled_qty == order.intent.size:
            order.terminal_state = TerminalState.FILLED
            order.terminal_ts_ns = now_ns
            order.time_to_fill_ns = now_ns - order.arrival_ts_ns
            return self._cascade_oco(fill_event.order_id, now_ns)
        return []

    def _cascade_oco(self, filled_order_id: str, now_ns: int) -> list[str]:
        """Leg-aware OCO cascade (spine AD-25).

        A bracket shares one ``oco_group_id`` across entry + TP + SL. Only an
        **EXIT**-leg fill closes the bracket: it cancels every other live
        member (the sibling exit, and any entry still unfilled). An
        **ENTRY**-leg fill cascades **nothing** -- the exits stay live so the
        position can be closed (and so Part A can replay the real exit fill).

        Reuses :meth:`cancel` so its guards apply; iterates sorted for
        determinism (spine AD-11).
        """
        filled = self._orders[filled_order_id]
        group_id = filled.intent.oco_group_id
        if group_id is None or filled.intent.leg is not Leg.EXIT:
            return []
        cascaded: list[str] = []
        for other_id in sorted(self._oco_groups.get(group_id, set())):
            if other_id == filled_order_id:
                continue
            other = self._orders.get(other_id)
            if other is not None and other.terminal_state is None:
                self.cancel(other_id, now_ns)
                other.oco_cancelled_at = now_ns
                cascaded.append(other_id)
        return cascaded

    def cancel(self, order_id: str, now_ns: int) -> None:
        """Transition a live (``IN_FLIGHT`` or ``WORKING``) order to
        ``CANCELLED`` at ``now_ns`` (spine AD-8).
        """
        self._require_open()
        self._advance_clock(now_ns)
        order = self._require_live(order_id)
        order.terminal_state = TerminalState.CANCELLED
        order.terminal_ts_ns = now_ns

    def reject(self, order_id: str, now_ns: int, reason: str) -> None:
        """Transition an ``IN_FLIGHT`` order to ``REJECTED`` at ``now_ns``
        (spine AD-8). Rejecting a ``WORKING`` or terminal order raises.
        """
        self._require_open()
        self._advance_clock(now_ns)
        order = self._get(order_id)
        if order.terminal_state is not None or order.state is not LiveState.IN_FLIGHT:
            current = (
                order.terminal_state.value
                if order.terminal_state is not None
                else order.state.value
            )
            _state_error(
                f"reject() needs an in_flight order; {order_id!r} is {current}"
            )
        order.terminal_state = TerminalState.REJECTED
        order.terminal_ts_ns = now_ns
        order.reject_reason = reason

    def replace(self, intent: OrderIntent, latency_ns: int, now_ns: int) -> None:
        """Apply an ``action == replace`` intent to a live order (spine AD-8).

        A size decrease at the same price keeps the order ``WORKING`` with its
        ``add_ts_ns`` and queue counters intact (priority preserved). Any price
        change -- and, per AD-8's "keeps priority *on a size decrease*", a size
        increase -- sends the order back to ``IN_FLIGHT`` with a fresh
        ``arrival_ts_ns = intent.submit_ts_ns + latency_ns`` (the replace
        message travels -- Ask-First resolved "yes") and its queue counters
        cleared (priority lost, spine AD-8).
        """
        del now_ns  # transition time is implied by the fresh arrival_ts_ns
        self._require_open()
        if intent.action is not IntentAction.REPLACE:
            _state_error(
                f"replace() needs action == replace, got {intent.action.value}"
            )
        if latency_ns < 0:
            _state_error(f"latency_ns must be >= 0, got {latency_ns}")
        order = self._require_live(intent.order_id)
        old = order.intent
        # a replace may only change price and/or size -- identity fields carry
        # into the OrderOutcome and the OCO registry (review: blind/edge-case).
        for f in ("order_id", "trade_id", "leg", "kind", "side", "oco_group_id"):
            if getattr(intent, f) != getattr(old, f):
                _state_error(
                    f"replace() may not change {f}: "
                    f"{getattr(old, f)!r} -> {getattr(intent, f)!r}"
                )
        if intent.size < order.filled_qty:
            _state_error(
                f"replace() size {intent.size} below already-filled "
                f"{order.filled_qty} on {intent.order_id!r}"
            )
        keeps_priority = (
            intent.limit_px_dbn == old.limit_px_dbn and intent.size <= old.size
        )
        order.intent = intent
        if keeps_priority:
            return
        order.state = LiveState.IN_FLIGHT
        order.arrival_ts_ns = intent.submit_ts_ns + latency_ns
        order.add_ts_ns = None
        order.queue_rank_at_submit = None
        order.queue_ahead_size_at_submit = None
        order.queue_ahead = 0
        order.cum_trade_vol_since_arrival = 0
        order.arrival_best_bid_dbn = None
        order.arrival_best_ask_dbn = None
        order.queue_position_set = False
        order.arrival_bbo_set = False

    def expire_all(self, now_ns: int) -> list[str]:
        """Force every live order (``IN_FLIGHT`` and ``WORKING``) to ``EXPIRED``
        at ``now_ns`` (spine AD-13(b)). The sim calls this at a
        ``valid_interval`` end. Returns the ids expired, in submit order.
        """
        self._require_open()
        self._advance_clock(now_ns)
        expired: list[str] = []
        for order_id, order in self._orders.items():
            if order.terminal_state is None:
                order.terminal_state = TerminalState.EXPIRED
                order.terminal_ts_ns = now_ns
                expired.append(order_id)
        return expired

    # --- sim-only setters (each callable once while the order is live) ---

    def set_queue_position(self, order_id: str, rank: int, ahead_size: int) -> None:
        """Write the arrival-tick queue position onto the order, once
        (spine AD-22). A second call, or a negative value, raises. The live
        ``queue_ahead`` counter is seeded from ``ahead_size``.
        """
        self._require_open()
        order = self._require_working(order_id)
        if order.queue_position_set:
            _state_error(f"set_queue_position already called for {order_id!r}")
        if rank < 0 or ahead_size < 0:
            _state_error(
                f"queue position must be non-negative, got "
                f"rank={rank}, ahead_size={ahead_size}"
            )
        order.queue_rank_at_submit = rank
        order.queue_ahead_size_at_submit = ahead_size
        order.queue_ahead = ahead_size
        order.queue_position_set = True

    def set_arrival_bbo(
        self, order_id: str, bid_dbn: int | None, ask_dbn: int | None
    ) -> None:
        """Write the BBO snapshotted at the order's arrival tick, once
        (spine AD-12, AD-20). A second call raises. ``None`` on a side means no
        quote there.
        """
        self._require_open()
        order = self._require_working(order_id)
        if order.arrival_bbo_set:
            _state_error(f"set_arrival_bbo already called for {order_id!r}")
        order.arrival_best_bid_dbn = bid_dbn
        order.arrival_best_ask_dbn = ask_dbn
        order.arrival_bbo_set = True

    def set_adverse_selection(self, order_id: str, value: bool) -> None:
        """Set the deferred ``adverse_selection`` marker (spine AD-28).

        Callable only on a ``FILLED`` order that has not yet been finalized --
        the tracker keeps that one field mutable past the terminal transition
        so ``sim.py``'s 1-second deferred check can write it.
        """
        order = self._get(order_id)
        if order.terminal_state is not TerminalState.FILLED:
            state = (
                order.terminal_state.value
                if order.terminal_state is not None
                else order.state.value
            )
            _state_error(
                f"set_adverse_selection needs a filled order; "
                f"{order_id!r} is {state}"
            )
        if self._finalized:
            _state_error(f"set_adverse_selection for {order_id!r} after finalize()")
        if order.adverse_selection_set:
            _state_error(
                f"set_adverse_selection already called for {order_id!r} "
                "(callable once, spine AD-28)"
            )
        order.adverse_selection = value
        order.adverse_selection_set = True

    # --- fills-only counter mutators (spine AD-21/22: rules live in fills.py;
    #     only the guarded state lives here) ------------------------------

    def add_trade_volume(self, order_id: str, qty: int) -> None:
        """Add ``qty`` to a working order's ``cum_trade_vol_since_arrival``
        (spine AD-21/22). ``fills.py`` decides *which* trades count; this only
        holds the running total.
        """
        self._require_open()
        if qty < 0:
            _state_error(f"add_trade_volume qty must be >= 0, got {qty}")
        order = self._require_working(order_id)
        order.cum_trade_vol_since_arrival += qty

    def decrement_queue_ahead(self, order_id: str, qty: int) -> None:
        """Decrement a working order's live ``queue_ahead`` by ``qty``, floored
        at 0 (spine AD-21/22). ``fills.py`` decides *when* to call this.
        """
        self._require_open()
        if qty < 0:
            _state_error(f"decrement_queue_ahead qty must be >= 0, got {qty}")
        order = self._require_working(order_id)
        order.queue_ahead = max(0, order.queue_ahead - qty)

    # --- reads ----------------------------------------------------------

    def snapshot(self, order_id: str) -> OrderSnapshot:
        """Immutable view of a ``WORKING`` order's counters + fields for
        ``fills.decide`` (spine AD-5). Raises if the order is not working.
        """
        order = self._require_working(order_id)
        return OrderSnapshot(
            order_id=order_id,
            side=order.intent.side,
            kind=order.intent.kind,
            size=order.intent.size,
            limit_px_dbn=order.intent.limit_px_dbn,
            arrival_ts_ns=order.arrival_ts_ns,
            add_ts_ns=order.add_ts_ns,
            filled_qty=order.filled_qty,
            queue_rank_at_submit=order.queue_rank_at_submit,
            queue_ahead_size_at_submit=order.queue_ahead_size_at_submit,
            queue_ahead=order.queue_ahead,
            cum_trade_vol_since_arrival=order.cum_trade_vol_since_arrival,
        )

    def live_order_ids(self) -> list[str]:
        """Ids of every non-terminal order, in submit order."""
        return [oid for oid, o in self._orders.items() if o.terminal_state is None]

    def working_order_ids(self) -> list[str]:
        """Ids of every ``WORKING`` order, in submit order."""
        return [
            oid
            for oid, o in self._orders.items()
            if o.terminal_state is None and o.state is LiveState.WORKING
        ]

    def in_flight_order_ids(self) -> list[str]:
        """Ids of every ``IN_FLIGHT`` order, in submit order."""
        return [
            oid
            for oid, o in self._orders.items()
            if o.terminal_state is None and o.state is LiveState.IN_FLIGHT
        ]

    def live_state(self, order_id: str) -> LiveState | None:
        """The order's :class:`LiveState`, or ``None`` if it is terminal."""
        order = self._get(order_id)
        return None if order.terminal_state is not None else order.state

    def terminal_state(self, order_id: str) -> TerminalState | None:
        """The order's :class:`TerminalState`, or ``None`` if it is still live."""
        return self._get(order_id).terminal_state

    def terminal_ts_ns(self, order_id: str) -> int | None:
        """The ``ts_event`` ns at which the order became terminal, or ``None``
        if it is still live.
        """
        return self._get(order_id).terminal_ts_ns

    def reject_reason(self, order_id: str) -> str | None:
        """The reason string passed to :meth:`reject`, or ``None`` if the order
        was not rejected. ``OrderOutcome`` (AD-12) has no reason field, so this
        reader is the only way ``sim.py`` recovers a rejection cause.
        """
        return self._get(order_id).reject_reason

    def oco_group_members(self, group_id: str) -> list[str]:
        """Sorted ids registered in ``group_id`` (empty for an unknown group)."""
        return sorted(self._oco_groups.get(group_id, set()))

    def arrival_ts_ns(self, order_id: str) -> int:
        """The order's current ``arrival_ts_ns`` (a price-change replace makes
        this a fresh value).
        """
        return self._get(order_id).arrival_ts_ns

    # --- terminal output ----------------------------------------------

    def finalize(self) -> list[OrderOutcome]:
        """Build one :class:`OrderOutcome` per order, in submit order (spine
        AD-8, AD-12). Every order must be terminal or this raises. Every field
        is read from a transition this tracker performed.
        """
        if self._finalized:
            _state_error("finalize() already called; the tracker is sealed")
        live = [oid for oid, o in self._orders.items() if o.terminal_state is None]
        if live:
            _state_error(f"finalize() with live orders: {live}")
        outcomes: list[OrderOutcome] = []
        for order_id, order in self._orders.items():
            terminal_state = order.terminal_state
            assert terminal_state is not None  # guarded by the `live` check
            outcomes.append(
                OrderOutcome(
                    trade_id=order.intent.trade_id,
                    leg=order.intent.leg,
                    order_id=order_id,
                    kind=order.intent.kind,
                    side=order.intent.side,
                    submit_ts_ns=order.intent.submit_ts_ns,
                    arrival_ts_ns=order.arrival_ts_ns,
                    terminal_state=terminal_state,
                    fills=tuple(order.fills),
                    queue_rank_at_submit=order.queue_rank_at_submit,
                    queue_ahead_size_at_submit=order.queue_ahead_size_at_submit,
                    time_to_fill_ns=order.time_to_fill_ns,
                    arrival_best_bid_dbn=order.arrival_best_bid_dbn,
                    arrival_best_ask_dbn=order.arrival_best_ask_dbn,
                    adverse_selection=order.adverse_selection,
                )
            )
        self._finalized = True  # seal only after every OrderOutcome built
        return outcomes
