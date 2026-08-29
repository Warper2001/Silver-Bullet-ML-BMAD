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

import json
from typing import Any

import pytest
from pydantic import ValidationError

from src.ticksim.orders import (
    Fill,
    FillEvent,
    IntentAction,
    Leg,
    OrderIntent,
    OrderKind,
    OrderOutcome,
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
