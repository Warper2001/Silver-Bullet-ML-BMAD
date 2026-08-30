"""Unit tests for ``src.ticksim.parity.invariants`` (spine AD-16).

A holds-case and a raises-case per invariant, the kind-guard skips, the
``OrderOutcome`` consistency checks, and the ``check_order`` composition (incl.
one raises-case per invariant so a dropped call is caught) -- all from
hand-built ``OrderIntent`` / ``OrderOutcome`` fixtures (no ``SimRun``, no book).
"""

from __future__ import annotations

import pytest

from src.ticksim.orders import (
    Fill,
    IntentAction,
    Leg,
    OrderIntent,
    OrderKind,
    OrderOutcome,
    Side,
    TerminalState,
)
from src.ticksim.parity.invariants import (
    check_adverse_selection,
    check_fill_causality,
    check_fill_latency,
    check_no_price_improvement,
    check_order,
    check_partials_within_size,
    check_queue_position,
    check_time_to_fill,
    check_within_limit,
)
from src.ticksim.sim import InvariantViolation

LATENCY_NS = 250_000_000
SUBMIT_TS = 1_000_000_000
ARRIVAL_TS = SUBMIT_TS + LATENCY_NS
TICK = 250_000_000
ASK = 100_000_000_000
BID = ASK - TICK


def _intent(
    *,
    size: int = 3,
    limit_px_dbn: int | None = None,
    kind: OrderKind = OrderKind.MARKETABLE,
    side: Side = Side.BUY,
    order_id: str = "o1",
    trade_id: str = "t1",
) -> OrderIntent:
    return OrderIntent(
        action=IntentAction.SUBMIT,
        order_id=order_id,
        trade_id=trade_id,
        leg=Leg.ENTRY,
        kind=kind,
        side=side,
        size=size,
        limit_px_dbn=limit_px_dbn,
        submit_ts_ns=SUBMIT_TS,
    )


def _outcome(
    *,
    kind: OrderKind = OrderKind.MARKETABLE,
    side: Side = Side.BUY,
    order_id: str = "o1",
    trade_id: str = "t1",
    fills: tuple[tuple[int, int, int], ...] = (),
    terminal_state: TerminalState = TerminalState.FILLED,
    submit_ts_ns: int = SUBMIT_TS,
    arrival_ts_ns: int = ARRIVAL_TS,
    queue_rank_at_submit: int | None = None,
    queue_ahead_size_at_submit: int | None = None,
    arrival_best_bid_dbn: int | None = BID,
    arrival_best_ask_dbn: int | None = ASK,
    time_to_fill_ns: int | None = None,
    adverse_selection: bool = False,
) -> OrderOutcome:
    """``fills`` is a tuple of ``(px_dbn, size, ts_ns)``."""
    return OrderOutcome(
        trade_id=trade_id,
        leg=Leg.ENTRY,
        order_id=order_id,
        kind=kind,
        side=side,
        submit_ts_ns=submit_ts_ns,
        arrival_ts_ns=arrival_ts_ns,
        terminal_state=terminal_state,
        fills=tuple(Fill(px_dbn=px, size=sz, ts_ns=ts) for px, sz, ts in fills),
        queue_rank_at_submit=queue_rank_at_submit,
        queue_ahead_size_at_submit=queue_ahead_size_at_submit,
        arrival_best_bid_dbn=arrival_best_bid_dbn,
        arrival_best_ask_dbn=arrival_best_ask_dbn,
        time_to_fill_ns=time_to_fill_ns,
        adverse_selection=adverse_selection,
    )


def _passive(**kw: object) -> OrderOutcome:
    """A coherent passive-limit outcome: queue fields set, fills at the limit."""
    kw.setdefault("kind", OrderKind.PASSIVE_LIMIT)
    kw.setdefault("queue_rank_at_submit", 0)
    kw.setdefault("queue_ahead_size_at_submit", 0)
    return _outcome(**kw)  # type: ignore[arg-type]


# --- invariant 1: no price improvement --------------------------------------


def test_inv1_holds_buy_at_or_above_ask() -> None:
    oc = _outcome(fills=((ASK, 2, ARRIVAL_TS), (ASK + TICK, 1, ARRIVAL_TS + 1)))
    assert check_no_price_improvement(_intent(), oc) is None


def test_inv1_breach_buy_below_ask() -> None:
    oc = _outcome(fills=((ASK - TICK, 3, ARRIVAL_TS),))
    with pytest.raises(InvariantViolation) as exc:
        check_no_price_improvement(_intent(), oc)
    assert "invariant 1" in str(exc.value) and "o1" in str(exc.value)
    assert str(ASK - TICK) in str(exc.value) and str(ASK) in str(exc.value)


def test_inv1_holds_sell_at_or_below_bid() -> None:
    oc = _outcome(side=Side.SELL, fills=((BID, 3, ARRIVAL_TS),))
    assert check_no_price_improvement(_intent(side=Side.SELL), oc) is None


def test_inv1_breach_sell_above_bid() -> None:
    oc = _outcome(side=Side.SELL, fills=((BID + TICK, 3, ARRIVAL_TS),))
    with pytest.raises(InvariantViolation):
        check_no_price_improvement(_intent(side=Side.SELL), oc)


def test_inv1_skipped_for_passive() -> None:
    oc = _passive(side=Side.BUY, fills=((BID, 3, ARRIVAL_TS),))
    intent = _intent(kind=OrderKind.PASSIVE_LIMIT, limit_px_dbn=BID)
    assert check_no_price_improvement(intent, oc) is None


def test_inv1_unverifiable_when_arrival_quote_missing() -> None:
    oc = _outcome(fills=((ASK, 3, ARRIVAL_TS),), arrival_best_ask_dbn=None)
    with pytest.raises(InvariantViolation) as exc:
        check_no_price_improvement(_intent(), oc)
    assert "unverifiable" in str(exc.value)


def test_inv1_no_fills_is_noop() -> None:
    assert check_no_price_improvement(_intent(), _outcome(fills=())) is None


# --- invariant 2: within limit ---------------------------------------------


def test_inv2_passive_must_fill_exactly_at_limit() -> None:
    limit = ASK - 2 * TICK
    oc = _outcome(kind=OrderKind.PASSIVE_LIMIT, fills=((limit, 3, ARRIVAL_TS),))
    assert (
        check_within_limit(
            _intent(kind=OrderKind.PASSIVE_LIMIT, limit_px_dbn=limit), oc
        )
        is None
    )


def test_inv2_breach_passive_fill_not_at_limit() -> None:
    limit = ASK - 2 * TICK
    oc = _outcome(kind=OrderKind.PASSIVE_LIMIT, fills=((limit - TICK, 3, ARRIVAL_TS),))
    with pytest.raises(InvariantViolation) as exc:
        check_within_limit(
            _intent(kind=OrderKind.PASSIVE_LIMIT, limit_px_dbn=limit), oc
        )
    assert "invariant 2" in str(exc.value)


def test_inv2_marketable_limit_within() -> None:
    limit = ASK + 2 * TICK
    oc = _outcome(
        kind=OrderKind.MARKETABLE_LIMIT,
        fills=((ASK, 1, ARRIVAL_TS), (ASK + TICK, 2, ARRIVAL_TS)),
    )
    assert (
        check_within_limit(
            _intent(kind=OrderKind.MARKETABLE_LIMIT, limit_px_dbn=limit), oc
        )
        is None
    )


def test_inv2_breach_marketable_limit_through() -> None:
    limit = ASK
    oc = _outcome(kind=OrderKind.MARKETABLE_LIMIT, fills=((ASK + TICK, 3, ARRIVAL_TS),))
    with pytest.raises(InvariantViolation):
        check_within_limit(
            _intent(kind=OrderKind.MARKETABLE_LIMIT, limit_px_dbn=limit), oc
        )


def test_inv2_breach_sell_below_limit() -> None:
    limit = BID
    oc = _outcome(
        kind=OrderKind.MARKETABLE_LIMIT,
        side=Side.SELL,
        fills=((BID - TICK, 3, ARRIVAL_TS),),
    )
    with pytest.raises(InvariantViolation):
        check_within_limit(
            _intent(
                kind=OrderKind.MARKETABLE_LIMIT, side=Side.SELL, limit_px_dbn=limit
            ),
            oc,
        )


def test_inv2_skipped_for_marketable() -> None:
    assert (
        check_within_limit(_intent(), _outcome(fills=((ASK, 3, ARRIVAL_TS),))) is None
    )


# --- invariant 3: fill latency -------------------------------------------


def test_inv3_holds() -> None:
    oc = _outcome(fills=((ASK, 3, ARRIVAL_TS),))
    assert check_fill_latency(oc) is None


def test_inv3_breach_fill_before_arrival() -> None:
    oc = _outcome(fills=((ASK, 3, ARRIVAL_TS - 1),))
    with pytest.raises(InvariantViolation) as exc:
        check_fill_latency(oc)
    assert "invariant 3" in str(exc.value)


def test_inv3_size_down_replace_keeps_original_arrival() -> None:
    # a priority-preserving replace: outcome.submit_ts_ns is the replace's ts
    # (later), arrival_ts_ns is the original -- a fill between them is fine.
    replace_submit = ARRIVAL_TS + 10 * TICK
    oc = _outcome(
        kind=OrderKind.PASSIVE_LIMIT,
        submit_ts_ns=replace_submit,
        arrival_ts_ns=ARRIVAL_TS,
        fills=((ASK, 3, ARRIVAL_TS + TICK),),
        queue_rank_at_submit=0,
        queue_ahead_size_at_submit=0,
    )
    assert check_fill_latency(oc) is None  # would false-fail on submit+latency


# --- invariant 4: queue position ---------------------------------------


def test_inv4_holds_passive_with_both_fields() -> None:
    oc = _outcome(
        kind=OrderKind.PASSIVE_LIMIT,
        fills=((ASK, 3, ARRIVAL_TS),),
        queue_rank_at_submit=2,
        queue_ahead_size_at_submit=5,
    )
    assert check_queue_position(oc) is None


def test_inv4_holds_marketable_with_both_none() -> None:
    assert check_queue_position(_outcome(fills=((ASK, 3, ARRIVAL_TS),))) is None


def test_inv4_breach_filled_passive_missing_queue_field() -> None:
    oc = _outcome(
        kind=OrderKind.PASSIVE_LIMIT,
        fills=((ASK, 3, ARRIVAL_TS),),
        queue_rank_at_submit=2,
        queue_ahead_size_at_submit=None,
    )
    with pytest.raises(InvariantViolation) as exc:
        check_queue_position(oc)
    assert "invariant 4" in str(exc.value)


def test_inv4_in_flight_terminal_passive_may_have_none_queue() -> None:
    # a passive limit EXPIRED / REJECTED while IN_FLIGHT never worked -> None is OK
    for state in (
        TerminalState.EXPIRED,
        TerminalState.REJECTED,
        TerminalState.CANCELLED,
    ):
        oc = _outcome(kind=OrderKind.PASSIVE_LIMIT, fills=(), terminal_state=state)
        assert check_queue_position(oc) is None


def test_inv4_breach_marketable_with_queue_field() -> None:
    oc = _outcome(fills=((ASK, 3, ARRIVAL_TS),), queue_rank_at_submit=1)
    with pytest.raises(InvariantViolation):
        check_queue_position(oc)


# --- invariant 5a: partials within size --------------------------------


def test_inv5a_holds_filled_exact() -> None:
    oc = _outcome(fills=((ASK, 2, ARRIVAL_TS), (ASK, 1, ARRIVAL_TS + 1)))
    assert check_partials_within_size(_intent(size=3), oc) is None


def test_inv5a_holds_partial_then_expired() -> None:
    oc = _outcome(
        kind=OrderKind.PASSIVE_LIMIT,
        fills=((ASK, 2, ARRIVAL_TS),),
        terminal_state=TerminalState.EXPIRED,
    )
    assert check_partials_within_size(_intent(size=5), oc) is None


def test_inv5a_breach_oversize() -> None:
    oc = _outcome(fills=((ASK, 4, ARRIVAL_TS),))
    with pytest.raises(InvariantViolation) as exc:
        check_partials_within_size(_intent(size=3), oc)
    assert "invariant 5" in str(exc.value)


def test_inv5a_breach_filled_but_not_full() -> None:
    oc = _outcome(fills=((ASK, 2, ARRIVAL_TS),), terminal_state=TerminalState.FILLED)
    with pytest.raises(InvariantViolation):
        check_partials_within_size(_intent(size=3), oc)


def test_inv5a_breach_full_but_not_marked_filled() -> None:
    oc = _outcome(fills=((ASK, 3, ARRIVAL_TS),), terminal_state=TerminalState.CANCELLED)
    with pytest.raises(InvariantViolation) as exc:
        check_partials_within_size(_intent(size=3), oc)
    assert "not FILLED" in str(exc.value)


def test_inv5a_breach_rejected_with_fills() -> None:
    oc = _outcome(fills=((ASK, 1, ARRIVAL_TS),), terminal_state=TerminalState.REJECTED)
    with pytest.raises(InvariantViolation) as exc:
        check_partials_within_size(_intent(size=3), oc)
    assert "REJECTED" in str(exc.value)


# --- invariant 6: fill causality --------------------------------------


def test_inv6_holds_non_decreasing() -> None:
    oc = _outcome(
        fills=((ASK, 1, ARRIVAL_TS), (ASK, 1, ARRIVAL_TS), (ASK, 1, ARRIVAL_TS + 5))
    )
    assert check_fill_causality(oc) is None


def test_inv6_breach_out_of_order() -> None:
    oc = _outcome(fills=((ASK, 1, ARRIVAL_TS + 5), (ASK, 2, ARRIVAL_TS + 1)))
    with pytest.raises(InvariantViolation) as exc:
        check_fill_causality(oc)
    assert "invariant 6" in str(exc.value)


def test_inv6_breach_before_arrival() -> None:
    oc = _outcome(fills=((ASK, 3, ARRIVAL_TS - 10),))
    with pytest.raises(InvariantViolation):
        check_fill_causality(oc)


# --- OrderOutcome consistency: time_to_fill / adverse_selection --------


def test_time_to_fill_holds() -> None:
    oc = _outcome(fills=((ASK, 3, ARRIVAL_TS + 40),), time_to_fill_ns=40)
    assert check_time_to_fill(oc) is None


def test_time_to_fill_breach_wrong_value() -> None:
    oc = _outcome(fills=((ASK, 3, ARRIVAL_TS + 40),), time_to_fill_ns=99)
    with pytest.raises(InvariantViolation):
        check_time_to_fill(oc)


def test_time_to_fill_breach_present_but_not_filled() -> None:
    oc = _outcome(
        kind=OrderKind.PASSIVE_LIMIT,
        terminal_state=TerminalState.EXPIRED,
        time_to_fill_ns=10,
    )
    with pytest.raises(InvariantViolation):
        check_time_to_fill(oc)


def test_adverse_selection_holds_on_passive_fill() -> None:
    oc = _outcome(
        kind=OrderKind.PASSIVE_LIMIT,
        fills=((ASK, 3, ARRIVAL_TS),),
        adverse_selection=True,
    )
    assert check_adverse_selection(oc) is None


def test_adverse_selection_breach_on_marketable() -> None:
    oc = _outcome(fills=((ASK, 3, ARRIVAL_TS),), adverse_selection=True)
    with pytest.raises(InvariantViolation):
        check_adverse_selection(oc)


def test_adverse_selection_breach_on_unfilled() -> None:
    oc = _outcome(
        kind=OrderKind.PASSIVE_LIMIT,
        fills=(),
        terminal_state=TerminalState.EXPIRED,
        adverse_selection=True,
    )
    with pytest.raises(InvariantViolation):
        check_adverse_selection(oc)


# --- check_order composition -----------------------------------------


def _clean_marketable() -> tuple[OrderIntent, OrderOutcome]:
    return _intent(size=3), _outcome(
        fills=((ASK, 3, ARRIVAL_TS + 5),), time_to_fill_ns=5
    )


def _clean_passive() -> tuple[OrderIntent, OrderOutcome]:
    limit = ASK - 2 * TICK
    intent = _intent(size=3, kind=OrderKind.PASSIVE_LIMIT, limit_px_dbn=limit)
    oc = _outcome(
        kind=OrderKind.PASSIVE_LIMIT,
        fills=((limit, 3, ARRIVAL_TS + 7),),
        queue_rank_at_submit=4,
        queue_ahead_size_at_submit=4,
        time_to_fill_ns=7,
    )
    return intent, oc


def test_check_order_marketable_all_pass() -> None:
    assert check_order(*_clean_marketable()) is None


def test_check_order_passive_all_pass() -> None:
    assert check_order(*_clean_passive()) is None


def test_check_order_no_fills_all_pass() -> None:
    intent = _intent(size=3, kind=OrderKind.PASSIVE_LIMIT, limit_px_dbn=ASK - TICK)
    oc = _outcome(
        kind=OrderKind.PASSIVE_LIMIT, fills=(), terminal_state=TerminalState.CANCELLED
    )
    assert check_order(intent, oc) is None


def test_check_order_rejects_intent_outcome_mismatch() -> None:
    intent, oc = _clean_marketable()
    bad = _intent(size=3, order_id="OTHER")
    with pytest.raises(InvariantViolation) as exc:
        check_order(bad, oc)
    assert "join mismatch" in str(exc.value)


@pytest.mark.parametrize(
    "invariant, mutate",
    [
        (
            "1",
            lambda i, o: (
                i,
                _outcome(fills=((ASK - TICK, 3, ARRIVAL_TS),), time_to_fill_ns=0),
            ),
        ),
        (
            "2",
            lambda i, o: (
                _intent(size=3, kind=OrderKind.PASSIVE_LIMIT, limit_px_dbn=ASK),
                _outcome(
                    kind=OrderKind.PASSIVE_LIMIT,
                    fills=((ASK - TICK, 3, ARRIVAL_TS + 1),),
                    queue_rank_at_submit=0,
                    queue_ahead_size_at_submit=0,
                    time_to_fill_ns=1,
                ),
            ),
        ),
        (
            "3",
            lambda i, o: (i, _outcome(fills=((ASK, 3, ARRIVAL_TS - 1),))),
        ),
        (
            "4",
            lambda i, o: (
                _intent(size=3, kind=OrderKind.PASSIVE_LIMIT, limit_px_dbn=ASK),
                _outcome(
                    kind=OrderKind.PASSIVE_LIMIT,
                    fills=((ASK, 3, ARRIVAL_TS + 1),),
                    queue_rank_at_submit=None,
                    queue_ahead_size_at_submit=None,
                    time_to_fill_ns=1,
                ),
            ),
        ),
        (
            "5",
            lambda i, o: (_intent(size=2), _outcome(fills=((ASK, 5, ARRIVAL_TS),))),
        ),
        (
            "6",
            lambda i, o: (
                i,
                _outcome(fills=((ASK, 1, ARRIVAL_TS + 5), (ASK, 2, ARRIVAL_TS + 1))),
            ),
        ),
    ],
)
def test_check_order_raises_per_invariant(invariant: str, mutate) -> None:  # type: ignore[no-untyped-def]
    intent, oc = mutate(*_clean_marketable())
    with pytest.raises(InvariantViolation) as exc:
        check_order(intent, oc)
    assert f"invariant {invariant}" in str(exc.value)
