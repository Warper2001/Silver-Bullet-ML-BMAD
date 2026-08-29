"""Frozen simulator<->consumer schemas: ``OrderIntent``, ``FillEvent``, ``Fill``, ``OrderOutcome``.

Contract layer only -- **no behaviour** (spine: this spec ships schemas, not the
engine). ``OrderTracker`` and OCO-group mechanics are a later spec and are
deliberately absent here.

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

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


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

    The full machine ``intent -> in_flight -> working -> {...}`` lives in the
    later ``OrderTracker`` spec; only the terminal set is part of this schema.
    """

    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


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
