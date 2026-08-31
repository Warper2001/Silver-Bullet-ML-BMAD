"""Unit tests for ``src/ticksim/parity/part_b.py`` (spine AD-16, prereg §A8.2).

Every case builds a hand-rolled ``OrderIntent`` list + an in-memory
:class:`~src.ticksim.events.BookEventSource` and runs the *real* pipeline
(``sim.simulate`` folding a real book, ``invariants.check_order`` per pair) --
no ``.dbn.zst`` fixture. The doctored-outcome cases run a real ``simulate``
first, then monkeypatch ``part_b.simulate`` to return the doctored list so a
**real** ``check_*`` is what trips (no fake ``InvariantViolation`` messages).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import pytest

from src.ticksim.config import PART_B_MIN_ORDERS, PRIMARY
from src.ticksim.events import BookEvent, MboAction, MboSide
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
from src.ticksim.parity import part_b
from src.ticksim.parity.part_b import (
    PART_B_COVERAGE_NOTE,
    PART_B_WINDOW_PAD_NS,
    PartBError,
    Violation,
    run_part_b,
)
from src.ticksim.sim import IntentLogError, InvariantViolation, simulate

# --------------------------------------------------------------------------- #
# fixtures / builders
# --------------------------------------------------------------------------- #

IID = 7
TICK = 250_000_000
P = 20_000_000_000_000
BID_PX = P - TICK
ASK_PX = P + TICK
B = 1_700_000_000 * 1_000_000_000
LAT = 250_000_000  # PRIMARY latency (250 ms)
PAD = PART_B_WINDOW_PAD_NS


class ListSource:
    """Re-iterable in-memory :class:`BookEventSource` (spine AD-18)."""

    class_rank = 0

    def __init__(self, events: list[BookEvent]) -> None:
        self._events = list(events)

    def __iter__(self) -> Iterator[BookEvent]:
        return iter(self._events)


class CountingSource(ListSource):
    """A :class:`ListSource` that counts how many times it is iterated."""

    def __init__(self, events: list[BookEvent]) -> None:
        super().__init__(events)
        self.iter_count = 0

    def __iter__(self) -> Iterator[BookEvent]:
        self.iter_count += 1
        return super().__iter__()


def add(
    oid: int,
    side: MboSide,
    px: int,
    size: int,
    ts: int,
    seq: int,
    iid: int = IID,
) -> BookEvent:
    return BookEvent(
        action=MboAction.ADD,
        side=side,
        order_id=oid,
        price_dbn=px,
        size=size,
        ts_event=ts,
        sequence=seq,
        instrument_id=iid,
    )


def trade(px: int, size: int, ts: int, seq: int) -> BookEvent:
    return BookEvent(
        action=MboAction.TRADE,
        side=MboSide.ASK,
        order_id=0,
        price_dbn=px,
        size=size,
        ts_event=ts,
        sequence=seq,
        instrument_id=IID,
    )


def submit(
    oid: str,
    side: Side,
    kind: OrderKind,
    size: int,
    px: int | None,
    ts: int,
    *,
    trade_id: str | None = None,
) -> OrderIntent:
    return OrderIntent(
        action=IntentAction.SUBMIT,
        order_id=oid,
        trade_id=trade_id or f"tr-{oid}",
        leg=Leg.ENTRY,
        kind=kind,
        side=side,
        size=size,
        limit_px_dbn=px,
        submit_ts_ns=ts,
    )


def deep_book(ts: int = 0) -> list[BookEvent]:
    return [
        add(1, MboSide.BID, BID_PX, 10_000_000, ts=ts, seq=1),
        add(2, MboSide.ASK, ASK_PX, 10_000_000, ts=ts, seq=2),
    ]


def clean_marketable_batch(
    n: int, *, step: int = 1_000
) -> tuple[list[OrderIntent], ListSource]:
    """``n`` marketable orders, alternating BUY / SELL, each size 1, filling at
    the touch off a deep static book -- a genuinely clean Part B batch."""
    intents = [
        submit(
            f"o{i}",
            Side.BUY if i % 2 == 0 else Side.SELL,
            OrderKind.MARKETABLE,
            1,
            None,
            B + i * step,
        )
        for i in range(n)
    ]
    return intents, ListSource(deep_book())


def varied_batch(n: int) -> tuple[list[OrderIntent], ListSource]:
    """``n`` orders cycling kind (market / marketable_limit / passive_limit),
    side, and size (1-5) -- a closer proxy to the slice-2 generator's output.
    Marketable + marketable_limit fill at the touch off a deep static book; the
    inside-spread passive limits are cleared by one late mid print."""
    intents: list[OrderIntent] = []
    for i in range(n):
        side = Side.BUY if i % 2 == 0 else Side.SELL
        size = 1 + (i % 5)
        ts = B + i * 1_000
        if i % 3 == 0:
            intents.append(submit(f"o{i}", side, OrderKind.MARKETABLE, size, None, ts))
        elif i % 3 == 1:
            limit = ASK_PX + 3 * TICK if side is Side.BUY else BID_PX - 3 * TICK
            intents.append(
                submit(f"o{i}", side, OrderKind.MARKETABLE_LIMIT, size, limit, ts)
            )
        else:
            intents.append(submit(f"o{i}", side, OrderKind.PASSIVE_LIMIT, size, P, ts))
    late = B + (n - 1) * 1_000 + LAT + 1_000_000
    src = ListSource([*deep_book(), trade(P, 10_000_000, ts=late, seq=3)])
    return intents, src


def intervals_for(intents: Sequence[OrderIntent]) -> list[tuple[int, int]]:
    stamps = [i.submit_ts_ns for i in intents]
    return [(max(0, min(stamps) - PAD), max(stamps) + PAD)]


def _patch_simulate(
    monkeypatch: pytest.MonkeyPatch, outcomes: Sequence[OrderOutcome]
) -> None:
    monkeypatch.setattr(part_b, "simulate", lambda *a, **k: (list(outcomes), object()))


# --------------------------------------------------------------------------- #
# clean PASS -- low monkeypatched floor (kind mix) and a genuine 1000-order run
# --------------------------------------------------------------------------- #


def test_clean_kind_mix_passes_with_low_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(part_b, "PART_B_MIN_ORDERS", 3)
    # market BUY/SELL + passive_limit BUY/SELL (inside the spread) + a
    # marketable_limit BUY/SELL -- one of every kind, both sides.
    intents = [
        submit("m-buy", Side.BUY, OrderKind.MARKETABLE, 1, None, B + 0),
        submit("m-sell", Side.SELL, OrderKind.MARKETABLE, 1, None, B + 1_000),
        submit("p-buy", Side.BUY, OrderKind.PASSIVE_LIMIT, 2, P, B + 2_000),
        submit("p-sell", Side.SELL, OrderKind.PASSIVE_LIMIT, 2, P, B + 3_000),
        submit(
            "ml-buy",
            Side.BUY,
            OrderKind.MARKETABLE_LIMIT,
            1,
            ASK_PX + 2 * TICK,
            B + 4_000,
        ),
        submit(
            "ml-sell",
            Side.SELL,
            OrderKind.MARKETABLE_LIMIT,
            1,
            BID_PX - 2 * TICK,
            B + 5_000,
        ),
    ]
    src = ListSource(
        [
            *deep_book(),
            # one print at mid, well after every arrival -> clears both
            # inside-spread passive queues (queue_ahead == 0) and fills them.
            trade(P, 100, ts=B + LAT + 1_000_000, seq=3),
        ]
    )

    result = run_part_b(intents, src)

    assert result.verdict == "PASS"
    assert result.violations == ()
    assert result.n_orders == 6
    assert result.n_fill_events == 6
    assert result.coverage_note is PART_B_COVERAGE_NOTE
    assert "6 orders" in result.reason


def test_genuine_1000_order_clean_pass() -> None:
    intents, src = varied_batch(PART_B_MIN_ORDERS)
    assert len(intents) >= 1000
    kinds = {i.kind for i in intents}
    assert kinds == {
        OrderKind.MARKETABLE,
        OrderKind.MARKETABLE_LIMIT,
        OrderKind.PASSIVE_LIMIT,
    }
    assert {i.side for i in intents} == {Side.BUY, Side.SELL}
    assert {i.size for i in intents} == {1, 2, 3, 4, 5}

    result = run_part_b(intents, src)

    assert result.verdict == "PASS"
    assert result.violations == ()
    assert result.n_orders == PART_B_MIN_ORDERS
    assert result.n_fill_events == PART_B_MIN_ORDERS  # each order fills once
    assert result.coverage_note is PART_B_COVERAGE_NOTE


def test_source_consumed_exactly_once() -> None:
    intents, _ = clean_marketable_batch(4)
    src = CountingSource(deep_book())
    run_part_b(intents, src)
    assert src.iter_count == 1


# --------------------------------------------------------------------------- #
# real check_* breaches via doctored outcomes -- the invariants.py <-> part_b
# message coupling is pinned (parsed Violation.invariant asserted)
# --------------------------------------------------------------------------- #


def _real_run(n: int) -> tuple[list[OrderIntent], list[OrderOutcome], ListSource]:
    intents, src = clean_marketable_batch(n)
    outcomes, _ = simulate(src, intents, PRIMARY, intervals_for(intents))
    return intents, list(outcomes), src


def _one_marketable_buy(
    size: int = 3,
) -> tuple[list[OrderIntent], list[OrderOutcome], ListSource]:
    intents = [submit("o0", Side.BUY, OrderKind.MARKETABLE, size, None, B)]
    src = ListSource(deep_book())
    outcomes, _ = simulate(src, intents, PRIMARY, intervals_for(intents))
    return intents, list(outcomes), src


def _one_passive_buy() -> tuple[list[OrderIntent], list[OrderOutcome], ListSource]:
    intents = [submit("o0", Side.BUY, OrderKind.PASSIVE_LIMIT, 2, P, B)]
    src = ListSource([*deep_book(), trade(P, 100, ts=B + LAT + 1_000_000, seq=3)])
    outcomes, _ = simulate(src, intents, PRIMARY, intervals_for(intents))
    return intents, list(outcomes), src


def test_real_invariant_4_breach_recorded_and_sweep_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intents, outcomes, src = _real_run(5)
    # a marketable outcome that carries a queue rank genuinely trips invariant 4.
    outcomes[2] = outcomes[2].model_copy(update={"queue_rank_at_submit": 7})
    _patch_simulate(monkeypatch, outcomes)
    monkeypatch.setattr(part_b, "PART_B_MIN_ORDERS", 3)

    result = run_part_b(intents, src)

    assert result.verdict == "FAIL"
    assert len(result.violations) == 1
    (v,) = result.violations
    assert v.invariant == "4"
    assert v.order_id == "o2"
    assert "invariant 4" in v.message and "o2" in v.message
    assert "4=1" in result.reason
    # the four clean orders were still swept (no crash on the first breach).
    assert result.n_orders == 5


@pytest.mark.parametrize(
    "label, doctor",
    [
        pytest.param(
            "1",
            lambda o: {
                "fills": (Fill(px_dbn=ASK_PX - TICK, size=3, ts_ns=o.fills[0].ts_ns),)
            },
            id="inv1-buy-fill-below-arrival-ask",
        ),
        pytest.param(
            "3",
            lambda o: {
                "fills": (Fill(px_dbn=ASK_PX, size=3, ts_ns=o.arrival_ts_ns - 1),)
            },
            id="inv3-fill-before-arrival",
        ),
        pytest.param(
            "6",
            lambda o: {
                "fills": (
                    Fill(px_dbn=ASK_PX, size=2, ts_ns=o.arrival_ts_ns + 5),
                    Fill(px_dbn=ASK_PX, size=1, ts_ns=o.arrival_ts_ns + 1),
                )
            },
            id="inv6-fills-out-of-order",
        ),
        pytest.param(
            "adverse_selection",
            lambda o: {"adverse_selection": True},
            id="adverse-flag-on-marketable",
        ),
    ],
)
def test_real_check_star_breach_labels(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    doctor: Callable[[OrderOutcome], dict[str, object]],
) -> None:
    intents, outcomes, src = _one_marketable_buy(3)
    outcomes[0] = outcomes[0].model_copy(update=doctor(outcomes[0]))
    _patch_simulate(monkeypatch, outcomes)
    monkeypatch.setattr(part_b, "PART_B_MIN_ORDERS", 1)

    result = run_part_b(intents, src)

    assert result.verdict == "FAIL"
    assert len(result.violations) == 1
    assert result.violations[0].invariant == label
    assert result.violations[0].order_id == "o0"


def test_real_invariant_2_breach_on_passive_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intents, outcomes, src = _one_passive_buy()
    orig = outcomes[0].fills[0]
    # a passive_limit fill not exactly at the resting limit -> real invariant 2.
    outcomes[0] = outcomes[0].model_copy(
        update={"fills": (Fill(px_dbn=P - TICK, size=orig.size, ts_ns=orig.ts_ns),)}
    )
    _patch_simulate(monkeypatch, outcomes)
    monkeypatch.setattr(part_b, "PART_B_MIN_ORDERS", 1)

    result = run_part_b(intents, src)

    assert result.verdict == "FAIL"
    assert [v.invariant for v in result.violations] == ["2"]
    assert "invariant 2" in result.violations[0].message


def test_several_different_invariants_all_recorded_sorted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intents, outcomes, src = _real_run(5)
    outcomes[0] = outcomes[0].model_copy(update={"time_to_fill_ns": 123_456_789})
    outcomes[1] = outcomes[1].model_copy(update={"queue_rank_at_submit": 3})
    outcomes[3] = outcomes[3].model_copy(
        update={"terminal_state": TerminalState.CANCELLED}
    )
    _patch_simulate(monkeypatch, outcomes)
    monkeypatch.setattr(part_b, "PART_B_MIN_ORDERS", 3)

    result = run_part_b(intents, src)

    assert result.verdict == "FAIL"
    assert [(v.order_id, v.invariant) for v in result.violations] == [
        ("o0", "time_to_fill"),
        ("o1", "4"),
        ("o3", "5"),
    ]
    assert "4=1" in result.reason
    assert "5=1" in result.reason
    assert "time_to_fill=1" in result.reason


def test_label_is_unknown_for_an_unrecognised_prefix() -> None:
    assert part_b._invariant_label("something entirely new happened") == "unknown"
    assert part_b._invariant_label("invariant 2 (within limit) breached") == "2"
    assert (
        part_b._invariant_label("adverse_selection structural breach for order")
        == "adverse_selection"
    )


# --------------------------------------------------------------------------- #
# intent/outcome join mismatch -> PartBError
# --------------------------------------------------------------------------- #


def test_kind_mismatch_raises_part_b_error(monkeypatch: pytest.MonkeyPatch) -> None:
    intents, outcomes, src = _real_run(4)
    outcomes[1] = outcomes[1].model_copy(update={"kind": OrderKind.PASSIVE_LIMIT})
    _patch_simulate(monkeypatch, outcomes)

    with pytest.raises(PartBError, match="join mismatch"):
        run_part_b(intents, src)


def test_check_order_fault_on_malformed_outcome_raises_part_b_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a non-InvariantViolation raised inside the sweep (here: check_order gets an
    # object with no .kind) must surface as a clear PartBError, not a traceback.
    intents, _, src = _real_run(3)

    class _Broken:
        order_id = "o0"

        def __getattr__(self, name: str) -> object:
            raise AttributeError(name)

    monkeypatch.setattr(
        part_b,
        "simulate",
        lambda *a, **k: ([_Broken(), _Broken(), _Broken()], object()),
    )
    monkeypatch.setattr(part_b, "_pair_outcomes", lambda i, o: list(zip(i, o)))

    with pytest.raises(PartBError, match="check_order faulted"):
        run_part_b(intents, src)


# --------------------------------------------------------------------------- #
# simulate itself raises InvariantViolation -> Violation("", "sim", ...) + FAIL
# --------------------------------------------------------------------------- #


def test_sim_raised_invariant_violation_is_a_fail_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intents, src = clean_marketable_batch(4)

    def boom(*_a: object, **_k: object) -> object:
        raise InvariantViolation("invariant 6 (fill causality) breached mid-run")

    monkeypatch.setattr(part_b, "simulate", boom)
    monkeypatch.setattr(part_b, "PART_B_MIN_ORDERS", 3)

    result = run_part_b(intents, src)

    assert result.verdict == "FAIL"
    assert result.violations == (
        Violation("", "sim", "invariant 6 (fill causality) breached mid-run"),
    )
    assert "sim=1" in result.reason


def test_intent_log_error_from_simulate_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intents, src = clean_marketable_batch(2)

    def boom(*_a: object, **_k: object) -> object:
        raise IntentLogError("not causally replayable")

    monkeypatch.setattr(part_b, "simulate", boom)
    with pytest.raises(IntentLogError):
        run_part_b(intents, src)


# --------------------------------------------------------------------------- #
# order-count floor
# --------------------------------------------------------------------------- #


def test_below_min_orders_is_a_fail_even_when_all_hold() -> None:
    intents, src = clean_marketable_batch(5)
    result = run_part_b(intents, src)

    assert result.verdict == "FAIL"
    assert result.violations == ()
    assert "5" in result.reason and str(PART_B_MIN_ORDERS) in result.reason


def test_empty_intents_fail_and_still_consume_the_source() -> None:
    src = CountingSource(deep_book())
    result = run_part_b([], src)

    assert result.verdict == "FAIL"
    assert result.n_orders == 0
    assert src.iter_count == 1
    assert str(PART_B_MIN_ORDERS) in result.reason


def test_zero_orders_fail_even_with_a_zero_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(part_b, "PART_B_MIN_ORDERS", 0)
    src = CountingSource(deep_book())
    result = run_part_b([], src)

    assert result.verdict == "FAIL"  # a zero-order battery verified nothing
    assert src.iter_count == 1


# --------------------------------------------------------------------------- #
# structural faults -> PartBError
# --------------------------------------------------------------------------- #


def test_non_submit_intent_raises_part_b_error() -> None:
    intents, src = clean_marketable_batch(3)
    replace = OrderIntent(
        action=IntentAction.REPLACE,
        order_id="rp",
        trade_id="tr-rp",
        leg=Leg.ENTRY,
        kind=OrderKind.PASSIVE_LIMIT,
        side=Side.BUY,
        size=1,
        limit_px_dbn=BID_PX,
        submit_ts_ns=B + 10_000,
        replaces_order_id="rp",
    )
    with pytest.raises(PartBError, match="not SUBMIT"):
        run_part_b([*intents, replace], src)


def test_empty_order_id_raises_part_b_error() -> None:
    # '' is schema-forbidden on OrderIntent, but a non-validating construction
    # path (model_copy) can slip one through -- _validate_intents rejects it
    # because '' is the sim-raised Violation sentinel.
    intents, src = clean_marketable_batch(2)
    bad = intents[0].model_copy(update={"order_id": ""})
    with pytest.raises(PartBError, match="empty order_id"):
        run_part_b([bad, intents[1]], src)


def test_duplicate_order_id_raises_part_b_error() -> None:
    intents, src = clean_marketable_batch(3)
    dup = submit("o0", Side.BUY, OrderKind.MARKETABLE, 1, None, B + 10_000)
    with pytest.raises(PartBError, match="duplicate order_id"):
        run_part_b([*intents, dup], src)


def test_negative_pad_ns_raises_part_b_error() -> None:
    intents, src = clean_marketable_batch(2)
    with pytest.raises(PartBError, match="pad_ns must be >= 0"):
        run_part_b(intents, src, pad_ns=-1)


def test_count_mismatch_outcome_raises_part_b_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intents, outcomes, src = _real_run(4)
    _patch_simulate(monkeypatch, outcomes[:3])
    with pytest.raises(PartBError, match="incomplete run"):
        run_part_b(intents, src)


def test_duplicate_outcome_raises_part_b_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intents, outcomes, src = _real_run(4)
    _patch_simulate(monkeypatch, [*outcomes[:3], outcomes[0]])
    with pytest.raises(PartBError, match="two outcomes"):
        run_part_b(intents, src)


def test_foreign_outcome_raises_part_b_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intents, outcomes, src = _real_run(4)
    foreign = outcomes[0].model_copy(update={"order_id": "ZZZ"})
    _patch_simulate(monkeypatch, [*outcomes[:3], foreign])
    with pytest.raises(PartBError, match="unknown order_id"):
        run_part_b(intents, src)


# --------------------------------------------------------------------------- #
# multi-instrument source -> IntentLogError (from sim, propagated)
# --------------------------------------------------------------------------- #


def test_multi_instrument_source_raises_intent_log_error() -> None:
    intents, _ = clean_marketable_batch(2)
    src = ListSource(
        [
            *deep_book(),
            add(9, MboSide.ASK, ASK_PX, 50, ts=0, seq=3, iid=IID + 1),
        ]
    )
    with pytest.raises(IntentLogError):
        run_part_b(intents, src)


# --------------------------------------------------------------------------- #
# window math: n_fill_events partials, epoch clamp
# --------------------------------------------------------------------------- #


def test_n_fill_events_counts_partials_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(part_b, "PART_B_MIN_ORDERS", 1)
    # only 1 lot at the touch, the rest one tick worse -> a marketable BUY of 3
    # fills in two events (both at/above the arrival ask, so still clean).
    src = ListSource(
        [
            add(1, MboSide.BID, BID_PX, 1_000_000, ts=0, seq=1),
            add(2, MboSide.ASK, ASK_PX, 1, ts=0, seq=2),
            add(3, MboSide.ASK, ASK_PX + TICK, 1_000_000, ts=0, seq=3),
        ]
    )
    intents = [submit("o0", Side.BUY, OrderKind.MARKETABLE, 3, None, B)]

    result = run_part_b(intents, src)

    assert result.verdict == "PASS"
    assert result.n_orders == 1
    assert result.n_fill_events == 2


def test_interval_start_is_clamped_at_epoch_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # an order stamped within pad_ns of ts 0: lo - pad_ns is negative and must be
    # floored at 0 so sim gets no malformed interval (mirrors the part_a runner).
    monkeypatch.setattr(part_b, "PART_B_MIN_ORDERS", 1)
    intents = [submit("o0", Side.BUY, OrderKind.MARKETABLE, 1, None, 1_000)]
    src = ListSource(
        [
            add(1, MboSide.BID, BID_PX, 1_000_000, ts=0, seq=1),
            add(2, MboSide.ASK, ASK_PX, 1_000_000, ts=0, seq=2),
        ]
    )

    result = run_part_b(intents, src)  # no ValueError from sim

    assert result.verdict == "PASS"
    assert result.n_orders == 1


# --------------------------------------------------------------------------- #
# module surface
# --------------------------------------------------------------------------- #


def test_window_pad_constant_is_five_minutes() -> None:
    assert PART_B_WINDOW_PAD_NS == 5 * 60 * 1_000_000_000


def test_coverage_note_states_the_split_verbatim() -> None:
    note = PART_B_COVERAGE_NOTE
    assert "test_ticksim_fills.py" in note
    assert "test_ticksim_orders.py" in note
    assert "test_ticksim_sim.py" in note
    assert "construction guarantee" in note
    assert "ORIGINAL arrival" in note  # check_fill_latency vs literal prereg
    assert "NOT re-derived here" in note  # invariant 5 liquidity half
    assert "n_outcomes" not in note


def test_result_is_frozen() -> None:
    result = run_part_b(*clean_marketable_batch(2))
    with pytest.raises(Exception):
        result.verdict = "PASS"  # type: ignore[misc]


def test_result_has_no_n_outcomes_field() -> None:
    result = run_part_b(*clean_marketable_batch(2))
    assert not hasattr(result, "n_outcomes")
