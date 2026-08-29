"""Unit tests for ``src.ticksim.orders`` -- the four frozen schemas.

Covers the spec's I/O & Edge-Case Matrix rows and guards:
  * AD-10 -- every numeric leaf of an ``OrderOutcome`` is an ``int`` (no floats).
  * AD-12 / AD-19 / AD-23 -- exact field sets, no extra behavioural field.
  * AD-23 -- ``OrderIntent`` replace + limit-price structural rules.
  * frozen-ness (incl. the ``fills`` tuple) and ``extra="forbid"`` of all four
    models.
  * the schema layer carries **no** cross-field lifecycle behaviour (owned by
    OrderTracker / sim / book / parity per the spine).
"""

import dataclasses
import json
from typing import Any

import pytest
from pydantic import ValidationError

from src.ticksim.orders import (
    Fill,
    FillEvent,
    IntentAction,
    Leg,
    LiveState,
    OrderIntent,
    OrderKind,
    OrderOutcome,
    OrderSnapshot,
    OrderStateError,
    OrderTracker,
    Side,
    TerminalState,
)


def _valid_intent_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        action=IntentAction.SUBMIT,
        order_id="o1",
        trade_id="t1",
        leg=Leg.ENTRY,
        kind=OrderKind.PASSIVE_LIMIT,
        side=Side.BUY,
        size=2,
        limit_px_dbn=30_000_000_000,
        submit_ts_ns=1_000,
    )
    base.update(overrides)
    return base


def _full_outcome() -> OrderOutcome:
    """An ``OrderOutcome`` with *every* optional numeric field populated."""
    return OrderOutcome(
        trade_id="t1",
        leg=Leg.EXIT,
        order_id="o2",
        kind=OrderKind.PASSIVE_LIMIT,
        side=Side.SELL,
        submit_ts_ns=1_000,
        arrival_ts_ns=1_000 + 250_000_000,
        terminal_state=TerminalState.FILLED,
        fills=(
            Fill(px_dbn=30_000_250_000, size=1, ts_ns=2_000),
            Fill(px_dbn=30_000_250_000, size=1, ts_ns=3_000),
        ),
        queue_rank_at_submit=4,
        queue_ahead_size_at_submit=12,
        time_to_fill_ns=1_999_000,
        arrival_best_bid_dbn=30_000_000_000,
        arrival_best_ask_dbn=30_000_250_000,
        adverse_selection=True,
    )


def _first_error_loc(exc: pytest.ExceptionInfo[ValidationError]) -> tuple[Any, ...]:
    return exc.value.errors()[0]["loc"]


class TestEnumWireValues:
    """AD-23 / AD-12: enum ``.value`` is the on-disk JSONL token -- a rename
    must break a test here."""

    def test_side_values(self) -> None:
        assert Side.BUY.value == "buy"
        assert Side.SELL.value == "sell"

    def test_order_kind_values(self) -> None:
        assert OrderKind.MARKETABLE.value == "marketable"
        assert OrderKind.MARKETABLE_LIMIT.value == "marketable_limit"
        assert OrderKind.PASSIVE_LIMIT.value == "passive_limit"

    def test_intent_action_values(self) -> None:
        assert IntentAction.SUBMIT.value == "submit"
        assert IntentAction.CANCEL.value == "cancel"
        assert IntentAction.REPLACE.value == "replace"

    def test_leg_values(self) -> None:
        assert Leg.ENTRY.value == "entry"
        assert Leg.EXIT.value == "exit"

    def test_terminal_state_values(self) -> None:
        assert TerminalState.FILLED.value == "filled"
        assert TerminalState.CANCELLED.value == "cancelled"
        assert TerminalState.REJECTED.value == "rejected"
        assert TerminalState.EXPIRED.value == "expired"


class TestFieldSets:
    """AD-12 / AD-19 / AD-23 -- exact field lists, no extra behavioural field."""

    def test_order_intent_fields(self) -> None:
        assert set(OrderIntent.model_fields) == {
            "schema_version",
            "action",
            "order_id",
            "trade_id",
            "leg",
            "kind",
            "side",
            "size",
            "limit_px_dbn",
            "submit_ts_ns",
            "replaces_order_id",
            "oco_group_id",
        }

    def test_fill_event_fields(self) -> None:
        assert set(FillEvent.model_fields) == {"order_id", "px_dbn", "size", "ts_ns"}

    def test_fill_fields(self) -> None:
        assert set(Fill.model_fields) == {"px_dbn", "size", "ts_ns"}

    def test_order_outcome_fields(self) -> None:
        assert set(OrderOutcome.model_fields) == {
            "schema_version",
            "trade_id",
            "leg",
            "order_id",
            "kind",
            "side",
            "submit_ts_ns",
            "arrival_ts_ns",
            "terminal_state",
            "fills",
            "queue_rank_at_submit",
            "queue_ahead_size_at_submit",
            "time_to_fill_ns",
            "arrival_best_bid_dbn",
            "arrival_best_ask_dbn",
            "adverse_selection",
        }


class TestExtraForbid:
    """JSONL wire records: an unknown key is a schema mismatch, not dropped."""

    def test_order_intent_rejects_unknown_key(self) -> None:
        with pytest.raises(ValidationError):
            OrderIntent(**_valid_intent_kwargs(surprise=1))

    def test_order_outcome_rejects_unknown_key(self) -> None:
        with pytest.raises(ValidationError):
            OrderOutcome(
                trade_id="t1",
                leg=Leg.ENTRY,
                order_id="o1",
                kind=OrderKind.MARKETABLE,
                side=Side.BUY,
                submit_ts_ns=1,
                arrival_ts_ns=2,
                terminal_state=TerminalState.EXPIRED,
                pnl_usd=123,  # type: ignore[call-arg]
            )

    def test_fill_event_rejects_unknown_key(self) -> None:
        with pytest.raises(ValidationError):
            FillEvent(
                order_id="o1",
                px_dbn=1,
                size=1,
                ts_ns=1,
                queue_rank=0,  # type: ignore[call-arg]
            )

    def test_fill_rejects_unknown_key(self) -> None:
        with pytest.raises(ValidationError):
            Fill(px_dbn=1, size=1, ts_ns=1, side="buy")  # type: ignore[call-arg]


class TestOrderIntentStructuralRules:
    """AD-23 -- replace convention + limit-price rule (model_validator)."""

    def test_replace_with_matching_target_is_valid(self) -> None:
        intent = OrderIntent(
            **_valid_intent_kwargs(
                action=IntentAction.REPLACE, order_id="x", replaces_order_id="x"
            )
        )
        assert intent.action is IntentAction.REPLACE
        assert intent.replaces_order_id == "x"

    def test_replace_without_target_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OrderIntent(
                **_valid_intent_kwargs(
                    action=IntentAction.REPLACE, order_id="x", replaces_order_id=None
                )
            )

    def test_replace_with_mismatched_target_is_rejected(self) -> None:
        # AD-23: a replace *reuses* order_id.
        with pytest.raises(ValidationError):
            OrderIntent(
                **_valid_intent_kwargs(
                    action=IntentAction.REPLACE,
                    order_id="x",
                    replaces_order_id="y",
                )
            )

    def test_non_replace_with_target_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OrderIntent(
                **_valid_intent_kwargs(
                    action=IntentAction.SUBMIT, replaces_order_id="o1"
                )
            )

    def test_submit_without_target_is_valid(self) -> None:
        intent = OrderIntent(**_valid_intent_kwargs(action=IntentAction.SUBMIT))
        assert intent.replaces_order_id is None

    def test_limit_kind_without_price_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OrderIntent(
                **_valid_intent_kwargs(kind=OrderKind.PASSIVE_LIMIT, limit_px_dbn=None)
            )

    def test_marketable_order_allows_null_limit(self) -> None:
        intent = OrderIntent(
            **_valid_intent_kwargs(kind=OrderKind.MARKETABLE, limit_px_dbn=None)
        )
        assert intent.limit_px_dbn is None

    def test_cancel_with_price_and_size_is_valid(self) -> None:
        # Matrix row: cancel intent may carry price/size; ignorable, not enforced.
        intent = OrderIntent(
            **_valid_intent_kwargs(
                action=IntentAction.CANCEL,
                limit_px_dbn=30_000_000_000,
                size=5,
            )
        )
        assert intent.action is IntentAction.CANCEL


class TestOrderIntentSchema:
    def test_schema_version_defaults_to_one(self) -> None:
        assert OrderIntent(**_valid_intent_kwargs()).schema_version == 1

    def test_schema_version_zero_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OrderIntent(**_valid_intent_kwargs(schema_version=0))

    def test_oco_group_id_roundtrips(self) -> None:
        intent = OrderIntent(**_valid_intent_kwargs(oco_group_id="bracket-7"))
        assert intent.oco_group_id == "bracket-7"
        assert json.loads(intent.model_dump_json())["oco_group_id"] == "bracket-7"

    def test_empty_order_id_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OrderIntent(**_valid_intent_kwargs(order_id=""))

    def test_zero_size_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OrderIntent(**_valid_intent_kwargs(size=0))

    def test_enum_values_accept_raw_strings(self) -> None:
        intent = OrderIntent(
            **_valid_intent_kwargs(
                action="submit", leg="entry", kind="passive_limit", side="buy"
            )
        )
        assert intent.side is Side.BUY

    def test_float_size_is_rejected_and_names_field(self) -> None:
        with pytest.raises(ValidationError) as exc:
            OrderIntent(**_valid_intent_kwargs(size=2.0))
        assert _first_error_loc(exc) == ("size",)

    def test_frozen(self) -> None:
        intent = OrderIntent(**_valid_intent_kwargs())
        with pytest.raises(ValidationError):
            intent.size = 3  # type: ignore[misc]


class TestFillEventSchema:
    """AD-19 -- exactly four fields, strict ints, frozen."""

    def test_well_formed(self) -> None:
        ev = FillEvent(order_id="o1", px_dbn=30_000_000_000, size=1, ts_ns=5)
        assert ev.size == 1

    def test_float_price_rejected_and_names_field(self) -> None:
        with pytest.raises(ValidationError) as exc:
            FillEvent(order_id="o1", px_dbn=1.5, size=1, ts_ns=5)  # type: ignore[arg-type]
        assert _first_error_loc(exc) == ("px_dbn",)

    def test_zero_price_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FillEvent(order_id="o1", px_dbn=0, size=1, ts_ns=5)

    def test_negative_ts_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FillEvent(order_id="o1", px_dbn=1, size=1, ts_ns=-1)

    def test_frozen(self) -> None:
        ev = FillEvent(order_id="o1", px_dbn=30_000_000_000, size=1, ts_ns=5)
        with pytest.raises(ValidationError):
            ev.px_dbn = 1  # type: ignore[misc]


class TestFillSchema:
    """AD-12 -- ``Fill`` is ``{px_dbn, size, ts_ns}``, strict ints, frozen."""

    def test_float_into_int_field_rejected_and_names_field(self) -> None:
        # Matrix row: Fill(px_dbn=1.5, ...) -> ValidationError (strict int).
        with pytest.raises(ValidationError) as exc:
            Fill(px_dbn=1.5, size=1, ts_ns=5)  # type: ignore[arg-type]
        assert _first_error_loc(exc) == ("px_dbn",)

    def test_float_size_rejected_and_names_field(self) -> None:
        with pytest.raises(ValidationError) as exc:
            Fill(px_dbn=100, size=1.0, ts_ns=5)  # type: ignore[arg-type]
        assert _first_error_loc(exc) == ("size",)

    def test_frozen(self) -> None:
        f = Fill(px_dbn=100, size=1, ts_ns=5)
        with pytest.raises(ValidationError):
            f.size = 2  # type: ignore[misc]


class TestOrderOutcomeSchema:
    def test_minimal_unfilled_outcome(self) -> None:
        outcome = OrderOutcome(
            trade_id="t1",
            leg=Leg.ENTRY,
            order_id="o1",
            kind=OrderKind.MARKETABLE,
            side=Side.BUY,
            submit_ts_ns=1,
            arrival_ts_ns=2,
            terminal_state=TerminalState.EXPIRED,
        )
        assert outcome.fills == ()
        assert outcome.queue_rank_at_submit is None
        assert outcome.time_to_fill_ns is None
        assert outcome.adverse_selection is False

    def test_schema_version_defaults_to_one(self) -> None:
        assert _full_outcome().schema_version == 1

    def test_frozen_attribute(self) -> None:
        outcome = _full_outcome()
        with pytest.raises(ValidationError):
            outcome.terminal_state = TerminalState.CANCELLED  # type: ignore[misc]

    def test_fills_is_an_immutable_tuple(self) -> None:
        outcome = _full_outcome()
        assert isinstance(outcome.fills, tuple)
        # no append on a tuple
        with pytest.raises(AttributeError):
            outcome.fills.append(  # type: ignore[attr-defined]
                Fill(px_dbn=1, size=1, ts_ns=1)
            )
        # no item assignment on a tuple
        with pytest.raises(TypeError):
            outcome.fills[0] = Fill(px_dbn=1, size=1, ts_ns=1)  # type: ignore[index]
        # and each element is itself frozen
        with pytest.raises(ValidationError):
            outcome.fills[0].px_dbn = 0  # type: ignore[misc]
        # rebinding the whole field is blocked too
        with pytest.raises(ValidationError):
            outcome.fills = ()  # type: ignore[misc]

    def test_float_queue_field_rejected_and_names_field(self) -> None:
        with pytest.raises(ValidationError) as exc:
            OrderOutcome(
                trade_id="t1",
                leg=Leg.ENTRY,
                order_id="o1",
                kind=OrderKind.PASSIVE_LIMIT,
                side=Side.BUY,
                submit_ts_ns=1,
                arrival_ts_ns=2,
                terminal_state=TerminalState.FILLED,
                queue_ahead_size_at_submit=1.5,  # type: ignore[arg-type]
            )
        assert _first_error_loc(exc) == ("queue_ahead_size_at_submit",)

    def test_negative_queue_rank_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OrderOutcome(
                trade_id="t1",
                leg=Leg.ENTRY,
                order_id="o1",
                kind=OrderKind.PASSIVE_LIMIT,
                side=Side.BUY,
                submit_ts_ns=1,
                arrival_ts_ns=2,
                terminal_state=TerminalState.FILLED,
                queue_rank_at_submit=-1,
            )


class TestSchemaLayerHasNoCrossFieldBehaviour:
    """The four models are dumb containers. Lifecycle / book / parity invariants
    (``terminal_state == FILLED`` <=> ``fills``; ``marketable`` => no queue
    position; ``arrival_ts_ns >= submit_ts_ns``; non-crossed arrival BBO; Σfill
    <= size) are owned elsewhere per the spine and must NOT be duplicated here.
    """

    def test_filled_with_no_fills_constructs(self) -> None:
        outcome = OrderOutcome(
            trade_id="t1",
            leg=Leg.ENTRY,
            order_id="o1",
            kind=OrderKind.PASSIVE_LIMIT,
            side=Side.BUY,
            submit_ts_ns=10,
            arrival_ts_ns=2,  # deliberately < submit_ts_ns: not enforced here
            terminal_state=TerminalState.FILLED,
            fills=(),
        )
        assert outcome.terminal_state is TerminalState.FILLED
        assert outcome.fills == ()

    def test_marketable_with_queue_position_constructs(self) -> None:
        outcome = OrderOutcome(
            trade_id="t1",
            leg=Leg.ENTRY,
            order_id="o1",
            kind=OrderKind.MARKETABLE,
            side=Side.BUY,
            submit_ts_ns=1,
            arrival_ts_ns=2,
            terminal_state=TerminalState.FILLED,
            queue_rank_at_submit=5,
            arrival_best_bid_dbn=30_000_250_000,
            arrival_best_ask_dbn=30_000_000_000,  # crossed: not enforced here
        )
        assert outcome.queue_rank_at_submit == 5


class TestOrderOutcomeIntegerAudit:
    """AD-10 -- every numeric leaf of a serialized ``OrderOutcome`` is ``int``.

    No allow-list: walk the whole JSON tree. ``bool`` (``adverse_selection``) is
    a JSON boolean, not a number, and is explicitly not a numeric leaf.
    """

    @staticmethod
    def _numeric_leaves(node: Any, path: str = "$") -> list[tuple[str, Any]]:
        found: list[tuple[str, Any]] = []
        if isinstance(node, dict):
            for key, value in node.items():
                found.extend(
                    TestOrderOutcomeIntegerAudit._numeric_leaves(value, f"{path}.{key}")
                )
        elif isinstance(node, (list, tuple)):
            for i, value in enumerate(node):
                found.extend(
                    TestOrderOutcomeIntegerAudit._numeric_leaves(value, f"{path}[{i}]")
                )
        elif isinstance(node, bool):
            pass  # JSON boolean, not a numeric leaf
        elif isinstance(node, (int, float)):
            found.append((path, node))
        return found

    def test_every_numeric_leaf_is_int(self) -> None:
        payload = _full_outcome().model_dump(mode="json")
        leaves = self._numeric_leaves(payload)
        assert leaves, "expected at least one numeric leaf"
        offenders = [(p, v) for p, v in leaves if not isinstance(v, int)]
        assert offenders == [], f"non-int numeric leaves: {offenders}"

    def test_no_float_leaf_even_with_all_optionals_set(self) -> None:
        payload = _full_outcome().model_dump(mode="json")
        leaves = self._numeric_leaves(payload)
        assert all(not isinstance(v, float) for _, v in leaves)

    def test_round_trip_json_string(self) -> None:
        payload = json.loads(_full_outcome().model_dump_json())
        offenders = [
            (p, v) for p, v in self._numeric_leaves(payload) if not isinstance(v, int)
        ]
        assert offenders == []


# ---------------------------------------------------------------------------
# OrderTracker -- lifecycle state machine + OCO groups
# ---------------------------------------------------------------------------

P = 30_000_000_000  # a limit price in DBN 1e-9 fixed-point units


def _submit_intent(**overrides: Any) -> OrderIntent:
    base: dict[str, Any] = dict(
        action=IntentAction.SUBMIT,
        order_id="o1",
        trade_id="t1",
        leg=Leg.ENTRY,
        kind=OrderKind.PASSIVE_LIMIT,
        side=Side.SELL,
        size=2,
        limit_px_dbn=P,
        submit_ts_ns=1_000,
    )
    base.update(overrides)
    return OrderIntent(**base)


def _replace_intent(order_id: str, **overrides: Any) -> OrderIntent:
    base: dict[str, Any] = dict(
        action=IntentAction.REPLACE,
        order_id=order_id,
        replaces_order_id=order_id,
        trade_id="t1",
        leg=Leg.ENTRY,
        kind=OrderKind.PASSIVE_LIMIT,
        side=Side.SELL,
        size=2,
        limit_px_dbn=P,
        submit_ts_ns=1_000,
    )
    base.update(overrides)
    return OrderIntent(**base)


def _fill(order_id: str, size: int, ts_ns: int, px_dbn: int = P) -> FillEvent:
    return FillEvent(order_id=order_id, px_dbn=px_dbn, size=size, ts_ns=ts_ns)


class TestOrderTrackerSubmitAndArrival:
    """I/O rows: submit -> arrival; not yet arrived; duplicate submit."""

    def test_submit_then_activate_makes_working_with_arrival_ts(self) -> None:
        tracker = OrderTracker()
        tracker.submit(_submit_intent(submit_ts_ns=0), latency_ns=250, now_ns=0)
        assert tracker.in_flight_order_ids() == ["o1"]
        tracker.activate_arrivals(now_ns=250)
        assert tracker.working_order_ids() == ["o1"]
        assert tracker.live_state("o1") is LiveState.WORKING
        assert tracker.arrival_ts_ns("o1") == 250

    def test_not_yet_arrived_stays_in_flight(self) -> None:
        tracker = OrderTracker()
        tracker.submit(_submit_intent(submit_ts_ns=0), latency_ns=250, now_ns=0)
        tracker.activate_arrivals(now_ns=100)
        assert tracker.in_flight_order_ids() == ["o1"]
        assert tracker.working_order_ids() == []

    def test_duplicate_submit_raises(self) -> None:
        tracker = OrderTracker()
        tracker.submit(_submit_intent(), latency_ns=10, now_ns=0)
        with pytest.raises(OrderStateError):
            tracker.submit(_submit_intent(), latency_ns=10, now_ns=0)

    def test_submit_rejects_non_submit_action(self) -> None:
        tracker = OrderTracker()
        with pytest.raises(OrderStateError):
            tracker.submit(_replace_intent("o1"), latency_ns=10, now_ns=0)


class TestOrderTrackerFills:
    """I/O rows: partial then full fill; over-fill; fill a non-working order."""

    def _working(self, size: int = 4) -> OrderTracker:
        tracker = OrderTracker()
        tracker.submit(
            _submit_intent(size=size, submit_ts_ns=0), latency_ns=0, now_ns=0
        )
        tracker.activate_arrivals(now_ns=0)
        return tracker

    def test_partial_then_full_fill(self) -> None:
        tracker = self._working(size=4)
        tracker.apply_fill(_fill("o1", size=2, ts_ns=10), now_ns=10)
        assert tracker.terminal_state("o1") is None
        tracker.apply_fill(_fill("o1", size=2, ts_ns=20), now_ns=20)
        assert tracker.terminal_state("o1") is TerminalState.FILLED
        (outcome,) = tracker.finalize()
        assert len(outcome.fills) == 2
        assert outcome.time_to_fill_ns == 20  # now_ns - arrival_ts_ns (0)

    def test_over_fill_raises(self) -> None:
        tracker = self._working(size=3)
        with pytest.raises(OrderStateError):
            tracker.apply_fill(_fill("o1", size=4, ts_ns=10), now_ns=10)

    def test_fill_a_non_working_order_raises(self) -> None:
        tracker = OrderTracker()
        tracker.submit(_submit_intent(size=2), latency_ns=250, now_ns=0)
        # still IN_FLIGHT
        with pytest.raises(OrderStateError):
            tracker.apply_fill(_fill("o1", size=1, ts_ns=10), now_ns=10)

    def test_fill_unknown_order_raises(self) -> None:
        tracker = OrderTracker()
        with pytest.raises(OrderStateError):
            tracker.apply_fill(_fill("nope", size=1, ts_ns=10), now_ns=10)


class TestOrderTrackerOCOCascade:
    """I/O row: OCO cascade -- filling one member cancels the others."""

    def _group(self) -> OrderTracker:
        tracker = OrderTracker()
        for oid, leg in (("entry", Leg.ENTRY), ("tp", Leg.EXIT), ("sl", Leg.EXIT)):
            tracker.submit(
                _submit_intent(
                    order_id=oid, leg=leg, size=2, oco_group_id="grp", submit_ts_ns=0
                ),
                latency_ns=0,
                now_ns=0,
            )
        tracker.activate_arrivals(now_ns=0)
        return tracker

    def test_fill_tp_cancels_entry_and_sl_at_same_now(self) -> None:
        tracker = self._group()
        tracker.apply_fill(_fill("tp", size=2, ts_ns=500), now_ns=500)
        assert tracker.terminal_state("tp") is TerminalState.FILLED
        assert tracker.terminal_state("entry") is TerminalState.CANCELLED
        assert tracker.terminal_state("sl") is TerminalState.CANCELLED
        assert tracker.terminal_ts_ns("entry") == 500
        assert tracker.terminal_ts_ns("sl") == 500

    def test_group_membership_is_queryable(self) -> None:
        tracker = self._group()
        assert tracker.oco_group_members("grp") == ["entry", "sl", "tp"]
        assert tracker.oco_group_members("unknown") == []

    def test_partial_fill_does_not_cascade(self) -> None:
        tracker = self._group()
        tracker.apply_fill(_fill("tp", size=1, ts_ns=500), now_ns=500)
        assert tracker.terminal_state("tp") is None
        assert tracker.working_order_ids() == ["entry", "tp", "sl"]


class TestOrderTrackerCancelReplaceExpireReject:
    """I/O rows: cancel working; replace size-down same price; replace price
    change; expire at interval end; reject."""

    def _working(self, **intent_overrides: Any) -> OrderTracker:
        tracker = OrderTracker()
        tracker.submit(
            _submit_intent(submit_ts_ns=0, **intent_overrides),
            latency_ns=0,
            now_ns=0,
        )
        tracker.activate_arrivals(now_ns=0)
        return tracker

    def test_cancel_working(self) -> None:
        tracker = self._working()
        tracker.cancel("o1", now_ns=42)
        assert tracker.terminal_state("o1") is TerminalState.CANCELLED
        assert tracker.terminal_ts_ns("o1") == 42
        assert tracker.live_state("o1") is None

    def test_cancel_terminal_order_raises(self) -> None:
        tracker = self._working()
        tracker.cancel("o1", now_ns=42)
        with pytest.raises(OrderStateError):
            tracker.cancel("o1", now_ns=43)

    def test_replace_size_down_same_price_keeps_priority(self) -> None:
        tracker = self._working(size=5)
        tracker.set_queue_position("o1", rank=3, ahead_size=9)
        add_ts_before = tracker.snapshot("o1").add_ts_ns
        tracker.replace(
            _replace_intent("o1", size=3, limit_px_dbn=P, submit_ts_ns=2_000),
            latency_ns=250,
            now_ns=2_000,
        )
        snap = tracker.snapshot("o1")
        assert tracker.live_state("o1") is LiveState.WORKING
        assert snap.size == 3
        assert snap.add_ts_ns == add_ts_before
        assert snap.queue_rank_at_submit == 3
        assert snap.queue_ahead_size_at_submit == 9

    def test_replace_price_change_returns_to_in_flight_and_clears_queue(self) -> None:
        tracker = self._working(size=5)
        tracker.set_queue_position("o1", rank=3, ahead_size=9)
        tracker.replace(
            _replace_intent("o1", size=5, limit_px_dbn=P + 250_000, submit_ts_ns=2_000),
            latency_ns=250,
            now_ns=2_000,
        )
        assert tracker.in_flight_order_ids() == ["o1"]
        assert tracker.arrival_ts_ns("o1") == 2_250  # fresh submit_ts + latency
        tracker.activate_arrivals(now_ns=2_250)
        snap = tracker.snapshot("o1")
        assert snap.queue_rank_at_submit is None
        assert snap.queue_ahead_size_at_submit is None
        assert snap.queue_ahead == 0

    def test_replace_can_set_queue_position_again_after_price_change(self) -> None:
        tracker = self._working(size=5)
        tracker.set_queue_position("o1", rank=3, ahead_size=9)
        tracker.replace(
            _replace_intent("o1", size=5, limit_px_dbn=P + 250_000, submit_ts_ns=2_000),
            latency_ns=0,
            now_ns=2_000,
        )
        tracker.activate_arrivals(now_ns=2_000)
        tracker.set_queue_position("o1", rank=1, ahead_size=2)
        assert tracker.snapshot("o1").queue_ahead_size_at_submit == 2

    def test_expire_all_hits_working_and_in_flight(self) -> None:
        tracker = OrderTracker()
        tracker.submit(
            _submit_intent(order_id="w", submit_ts_ns=0), latency_ns=0, now_ns=0
        )
        tracker.submit(
            _submit_intent(order_id="f", submit_ts_ns=0), latency_ns=999, now_ns=0
        )
        tracker.activate_arrivals(now_ns=0)
        assert tracker.working_order_ids() == ["w"]
        assert tracker.in_flight_order_ids() == ["f"]
        tracker.expire_all(now_ns=7_000)
        assert tracker.terminal_state("w") is TerminalState.EXPIRED
        assert tracker.terminal_state("f") is TerminalState.EXPIRED
        assert tracker.terminal_ts_ns("w") == 7_000
        assert tracker.terminal_ts_ns("f") == 7_000

    def test_reject_from_in_flight(self) -> None:
        tracker = OrderTracker()
        tracker.submit(_submit_intent(), latency_ns=250, now_ns=0)
        tracker.reject("o1", now_ns=5, reason="broker nak")
        assert tracker.terminal_state("o1") is TerminalState.REJECTED

    def test_reject_from_working_raises(self) -> None:
        tracker = self._working()
        with pytest.raises(OrderStateError):
            tracker.reject("o1", now_ns=5, reason="too late")


class TestOrderTrackerSettersAndCounters:
    """I/O rows: set queue position twice; adverse on a non-filled order.
    Plus the counter mutators (spine AD-21/22) and the snapshot contract."""

    def _working(self, size: int = 4) -> OrderTracker:
        tracker = OrderTracker()
        tracker.submit(
            _submit_intent(size=size, submit_ts_ns=0), latency_ns=0, now_ns=0
        )
        tracker.activate_arrivals(now_ns=0)
        return tracker

    def test_set_queue_position_twice_raises(self) -> None:
        tracker = self._working()
        tracker.set_queue_position("o1", rank=2, ahead_size=5)
        with pytest.raises(OrderStateError):
            tracker.set_queue_position("o1", rank=1, ahead_size=1)

    def test_set_arrival_bbo_twice_raises(self) -> None:
        tracker = self._working()
        tracker.set_arrival_bbo("o1", bid_dbn=P, ask_dbn=P + 250_000)
        with pytest.raises(OrderStateError):
            tracker.set_arrival_bbo("o1", bid_dbn=None, ask_dbn=None)

    def test_adverse_on_non_filled_order_raises(self) -> None:
        tracker = self._working()
        tracker.cancel("o1", now_ns=10)
        with pytest.raises(OrderStateError):
            tracker.set_adverse_selection("o1", True)

    def test_adverse_on_filled_order_then_finalize(self) -> None:
        tracker = self._working(size=2)
        tracker.apply_fill(_fill("o1", size=2, ts_ns=10), now_ns=10)
        tracker.set_adverse_selection("o1", True)
        (outcome,) = tracker.finalize()
        assert outcome.adverse_selection is True

    def test_adverse_after_finalize_raises(self) -> None:
        tracker = self._working(size=2)
        tracker.apply_fill(_fill("o1", size=2, ts_ns=10), now_ns=10)
        tracker.finalize()
        with pytest.raises(OrderStateError):
            tracker.set_adverse_selection("o1", True)

    def test_snapshot_carries_the_acceptance_fields(self) -> None:
        tracker = self._working(size=6)
        tracker.set_queue_position("o1", rank=4, ahead_size=12)
        tracker.add_trade_volume("o1", 3)
        tracker.add_trade_volume("o1", 2)
        tracker.decrement_queue_ahead("o1", 5)
        snap = tracker.snapshot("o1")
        assert isinstance(snap, OrderSnapshot)
        assert snap.queue_ahead == 7  # 12 - 5
        assert snap.cum_trade_vol_since_arrival == 5
        assert snap.queue_ahead_size_at_submit == 12
        assert snap.add_ts_ns == 0
        assert snap.size == 6
        assert snap.side is Side.SELL
        assert snap.kind is OrderKind.PASSIVE_LIMIT
        assert snap.limit_px_dbn == P

    def test_snapshot_is_frozen(self) -> None:
        tracker = self._working()
        snap = tracker.snapshot("o1")
        with pytest.raises(dataclasses.FrozenInstanceError):
            snap.queue_ahead = 99  # type: ignore[misc]

    def test_decrement_queue_ahead_floors_at_zero(self) -> None:
        tracker = self._working()
        tracker.set_queue_position("o1", rank=1, ahead_size=2)
        tracker.decrement_queue_ahead("o1", 10)
        assert tracker.snapshot("o1").queue_ahead == 0

    def test_snapshot_of_non_working_order_raises(self) -> None:
        tracker = OrderTracker()
        tracker.submit(_submit_intent(), latency_ns=250, now_ns=0)
        with pytest.raises(OrderStateError):
            tracker.snapshot("o1")


class TestOrderTrackerFinalize:
    """I/O rows: finalize with a live order; finalize all terminal."""

    def test_finalize_with_a_live_order_raises(self) -> None:
        tracker = OrderTracker()
        tracker.submit(_submit_intent(submit_ts_ns=0), latency_ns=0, now_ns=0)
        tracker.activate_arrivals(now_ns=0)
        with pytest.raises(OrderStateError):
            tracker.finalize()

    def test_finalize_is_submit_ordered_and_one_per_order(self) -> None:
        tracker = OrderTracker()
        for oid in ("c", "a", "b"):
            tracker.submit(
                _submit_intent(order_id=oid, submit_ts_ns=0), latency_ns=0, now_ns=0
            )
        tracker.activate_arrivals(now_ns=0)
        tracker.cancel("c", now_ns=1)
        tracker.cancel("a", now_ns=1)
        tracker.cancel("b", now_ns=1)
        outcomes = tracker.finalize()
        assert [o.order_id for o in outcomes] == ["c", "a", "b"]

    def test_finalized_outcome_fields_are_all_populated(self) -> None:
        tracker = OrderTracker()
        tracker.submit(
            _submit_intent(order_id="o1", size=2, submit_ts_ns=100),
            latency_ns=250,
            now_ns=100,
        )
        tracker.activate_arrivals(now_ns=350)
        tracker.set_queue_position("o1", rank=2, ahead_size=6)
        tracker.set_arrival_bbo("o1", bid_dbn=P, ask_dbn=P + 250_000)
        tracker.apply_fill(_fill("o1", size=2, ts_ns=900), now_ns=900)
        (outcome,) = tracker.finalize()
        assert outcome.order_id == "o1"
        assert outcome.submit_ts_ns == 100
        assert outcome.arrival_ts_ns == 350
        assert outcome.terminal_state is TerminalState.FILLED
        assert outcome.queue_rank_at_submit == 2
        assert outcome.queue_ahead_size_at_submit == 6
        assert outcome.time_to_fill_ns == 550  # 900 - 350
        assert outcome.arrival_best_bid_dbn == P
        assert outcome.arrival_best_ask_dbn == P + 250_000
        assert len(outcome.fills) == 1


class TestOrderTrackerBracketLifecycle:
    """End-to-end: submit a 3-order bracket, arrive, partially fill the entry,
    then fill the TP -> the SL (and the still-working entry) cancel; finalize
    yields three coherent OrderOutcomes in submit order (spec Acceptance)."""

    def test_full_bracket(self) -> None:
        tracker = OrderTracker()
        legs = [
            ("entry", Leg.ENTRY, OrderKind.PASSIVE_LIMIT, P),
            ("tp", Leg.EXIT, OrderKind.PASSIVE_LIMIT, P - 500_000),
            ("sl", Leg.EXIT, OrderKind.MARKETABLE, None),
        ]
        for oid, leg, kind, px in legs:
            tracker.submit(
                _submit_intent(
                    order_id=oid,
                    trade_id="rt-1",
                    leg=leg,
                    kind=kind,
                    side=Side.SELL,
                    size=2,
                    limit_px_dbn=px,
                    oco_group_id="bracket-1",
                    submit_ts_ns=0,
                ),
                latency_ns=250,
                now_ns=0,
            )
        tracker.activate_arrivals(now_ns=250)
        assert tracker.working_order_ids() == ["entry", "tp", "sl"]

        # entry partially fills -- still working, no cascade
        tracker.apply_fill(_fill("entry", size=1, ts_ns=300), now_ns=300)
        assert tracker.terminal_state("entry") is None

        # tp fills fully -> entry + sl cancel at the identical now_ns
        tracker.apply_fill(
            _fill("tp", size=2, ts_ns=900, px_dbn=P - 500_000), now_ns=900
        )

        assert tracker.terminal_ts_ns("entry") == 900
        assert tracker.terminal_ts_ns("sl") == 900

        outcomes = tracker.finalize()
        assert [o.order_id for o in outcomes] == ["entry", "tp", "sl"]
        by_id = {o.order_id: o for o in outcomes}

        assert by_id["entry"].terminal_state is TerminalState.CANCELLED
        assert len(by_id["entry"].fills) == 1
        assert by_id["entry"].time_to_fill_ns is None

        assert by_id["tp"].terminal_state is TerminalState.FILLED
        assert by_id["tp"].fills[0].size == 2
        assert by_id["tp"].fills[0].px_dbn == P - 500_000
        assert by_id["tp"].time_to_fill_ns == 650  # 900 - 250

        assert by_id["sl"].terminal_state is TerminalState.CANCELLED
        assert by_id["sl"].fills == ()

        assert {o.trade_id for o in outcomes} == {"rt-1"}


class TestOrderTrackerReviewHardening:
    """Review-driven coverage (blind / edge-case / verification-gap passes):
    the finalized-tracker seal, the monotonic clock, replace-identity guards,
    the OCO cascade return value, and same-tick OCO crossing."""

    def _working(self, **kw: Any) -> OrderTracker:
        tracker = OrderTracker()
        tracker.submit(_submit_intent(submit_ts_ns=0, **kw), latency_ns=0, now_ns=0)
        tracker.activate_arrivals(now_ns=0)
        return tracker

    # --- finalized seal --------------------------------------------------

    def test_transitions_after_finalize_raise(self) -> None:
        tracker = self._working(size=2)
        tracker.cancel("o1", now_ns=10)
        tracker.finalize()
        with pytest.raises(OrderStateError):
            tracker.submit(_submit_intent(order_id="o2"), latency_ns=0, now_ns=20)
        with pytest.raises(OrderStateError):
            tracker.cancel("o1", now_ns=20)

    def test_double_finalize_raises(self) -> None:
        tracker = self._working(size=2)
        tracker.cancel("o1", now_ns=10)
        tracker.finalize()
        with pytest.raises(OrderStateError):
            tracker.finalize()

    # --- monotonic clock ----------------------------------------------

    def test_backwards_now_ns_raises(self) -> None:
        tracker = self._working(size=2)
        tracker.apply_fill(_fill("o1", size=1, ts_ns=500), now_ns=500)
        with pytest.raises(OrderStateError):
            tracker.apply_fill(_fill("o1", size=1, ts_ns=100), now_ns=100)

    def test_same_now_ns_across_transitions_is_allowed(self) -> None:
        tracker = OrderTracker()
        for oid in ("a", "b"):
            tracker.submit(
                _submit_intent(order_id=oid, submit_ts_ns=0), latency_ns=0, now_ns=0
            )
        tracker.activate_arrivals(now_ns=5)
        tracker.cancel("a", now_ns=5)
        tracker.cancel("b", now_ns=5)
        assert tracker.terminal_ts_ns("a") == 5

    # --- replace guards ---------------------------------------------

    def test_replace_may_not_change_identity_fields(self) -> None:
        tracker = self._working(size=5)
        with pytest.raises(OrderStateError):
            tracker.replace(
                _replace_intent("o1", side=Side.BUY, size=5, limit_px_dbn=P),
                latency_ns=0,
                now_ns=1_000,
            )

    def test_replace_below_filled_qty_raises(self) -> None:
        tracker = self._working(size=5)
        tracker.apply_fill(_fill("o1", size=3, ts_ns=10), now_ns=10)
        with pytest.raises(OrderStateError):
            tracker.replace(
                _replace_intent("o1", size=2, limit_px_dbn=P, submit_ts_ns=20),
                latency_ns=0,
                now_ns=20,
            )

    def test_replace_size_up_same_price_loses_priority(self) -> None:
        tracker = self._working(size=3)
        tracker.set_queue_position("o1", rank=2, ahead_size=4)
        tracker.replace(
            _replace_intent("o1", size=8, limit_px_dbn=P, submit_ts_ns=2_000),
            latency_ns=250,
            now_ns=2_000,
        )
        assert tracker.in_flight_order_ids() == ["o1"]
        assert tracker.arrival_ts_ns("o1") == 2_250
        tracker.activate_arrivals(now_ns=2_250)
        snap = tracker.snapshot("o1")
        assert snap.queue_rank_at_submit is None
        assert snap.queue_ahead == 0

    def test_replace_of_oco_member_keeps_group_and_cascades(self) -> None:
        tracker = OrderTracker()
        for oid in ("entry", "exit"):
            tracker.submit(
                _submit_intent(order_id=oid, size=2, oco_group_id="g", submit_ts_ns=0),
                latency_ns=0,
                now_ns=0,
            )
        tracker.activate_arrivals(now_ns=0)
        tracker.replace(
            _replace_intent(
                "entry",
                size=2,
                limit_px_dbn=P + 250_000,
                oco_group_id="g",
                submit_ts_ns=100,
            ),
            latency_ns=0,
            now_ns=100,
        )
        tracker.activate_arrivals(now_ns=100)
        cascaded = tracker.apply_fill(_fill("entry", size=2, ts_ns=200), now_ns=200)
        assert cascaded == ["exit"]
        assert tracker.terminal_state("exit") is TerminalState.CANCELLED

    # --- OCO cascade return + same-tick crossing --------------------

    def test_apply_fill_returns_cascaded_ids(self) -> None:
        tracker = OrderTracker()
        for oid, leg in (("tp", Leg.EXIT), ("sl", Leg.EXIT)):
            tracker.submit(
                _submit_intent(
                    order_id=oid, leg=leg, size=2, oco_group_id="g", submit_ts_ns=0
                ),
                latency_ns=0,
                now_ns=0,
            )
        tracker.activate_arrivals(now_ns=0)
        assert tracker.apply_fill(_fill("tp", size=1, ts_ns=1), now_ns=1) == []
        assert tracker.apply_fill(_fill("tp", size=1, ts_ns=2), now_ns=2) == ["sl"]

    def test_same_tick_oco_double_fill_voids_the_losing_leg(self) -> None:
        tracker = OrderTracker()
        for oid in ("tp", "sl"):
            tracker.submit(
                _submit_intent(
                    order_id=oid,
                    leg=Leg.EXIT,
                    size=2,
                    oco_group_id="g",
                    submit_ts_ns=0,
                ),
                latency_ns=0,
                now_ns=0,
            )
        tracker.activate_arrivals(now_ns=0)
        tracker.apply_fill(_fill("tp", size=2, ts_ns=500), now_ns=500)
        # sl was cascade-cancelled this same tick -- a stale fill for it is void
        assert tracker.apply_fill(_fill("sl", size=2, ts_ns=500), now_ns=500) == []
        assert tracker.terminal_state("sl") is TerminalState.CANCELLED

    # --- reject reason reader ---------------------------------------

    def test_reject_reason_is_readable(self) -> None:
        tracker = OrderTracker()
        tracker.submit(_submit_intent(size=2), latency_ns=250, now_ns=0)
        tracker.reject("o1", now_ns=5, reason="broker nak")
        assert tracker.reject_reason("o1") == "broker nak"
        tracker2 = self._working(size=2)
        assert tracker2.reject_reason("o1") is None

    # --- return-value ordering ------------------------------------

    def test_activate_and_expire_return_submit_ordered_ids(self) -> None:
        tracker = OrderTracker()
        for oid in ("c", "a", "b"):
            tracker.submit(
                _submit_intent(order_id=oid, submit_ts_ns=0), latency_ns=0, now_ns=0
            )
        assert tracker.activate_arrivals(now_ns=0) == ["c", "a", "b"]
        assert tracker.expire_all(now_ns=9) == ["c", "a", "b"]

    # --- setter guards -------------------------------------------

    def test_set_adverse_selection_twice_raises(self) -> None:
        tracker = self._working(size=2)
        tracker.apply_fill(_fill("o1", size=2, ts_ns=10), now_ns=10)
        tracker.set_adverse_selection("o1", True)
        with pytest.raises(OrderStateError):
            tracker.set_adverse_selection("o1", False)

    def test_negative_latency_and_counter_qty_raise(self) -> None:
        tracker = OrderTracker()
        with pytest.raises(OrderStateError):
            tracker.submit(_submit_intent(), latency_ns=-1, now_ns=0)
        working = self._working(size=4)
        with pytest.raises(OrderStateError):
            working.add_trade_volume("o1", -1)
        with pytest.raises(OrderStateError):
            working.decrement_queue_ahead("o1", -1)

    def test_finalize_emits_side_leg_kind_from_intent(self) -> None:
        tracker = OrderTracker()
        tracker.submit(
            _submit_intent(
                order_id="b",
                side=Side.BUY,
                leg=Leg.EXIT,
                kind=OrderKind.MARKETABLE,
                limit_px_dbn=None,
                size=2,
                submit_ts_ns=0,
            ),
            latency_ns=0,
            now_ns=0,
        )
        tracker.activate_arrivals(now_ns=0)
        tracker.apply_fill(_fill("b", size=2, ts_ns=1), now_ns=1)
        (outcome,) = tracker.finalize()
        assert outcome.side is Side.BUY
        assert outcome.leg is Leg.EXIT
        assert outcome.kind is OrderKind.MARKETABLE
