"""Unit tests for ``src.ticksim.sim`` (spine AD-20 / AD-21 / AD-22 / AD-13 / AD-11).

One test per row of the spec's I/O & Edge-Case Matrix, plus the run-twice
byte-identity test (AD-11), a bracket end-to-end, and a mask-boundary expiry
test. Book-event sources are tiny hand-rolled :class:`_Source` doubles (a
``@dataclass`` with ``class_rank`` iterating a list of real
:class:`~src.ticksim.events.BookEvent`s); intent logs are ``OrderIntent`` lists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterator

import pytest

from src.ticksim.config import QueueModel, SimConfig
from src.ticksim.events import BookEvent, MboAction, MboSide
from src.ticksim.orders import (
    IntentAction,
    Leg,
    OrderIntent,
    OrderKind,
    Side,
    TerminalState,
)
from src.ticksim.sim import IntentLogError, Manifest, SimRun, simulate

IID = 1

CFG = SimConfig(
    queue_model=QueueModel.BACK_OF_QUEUE,
    latency_ns=100,
    exch_reg_fee_usd_cents=72,
    commission_usd_cents=58,
    seed=0,
    own_impact=False,
)
INTERVALS = [(0, 1_000_000)]


# --- source double + builders ----------------------------------------


@dataclass
class _Source:
    """A minimal :class:`~src.ticksim.events.BookEventSource` -- re-iterable
    (``iter(list)`` is fresh each call)."""

    events: list[BookEvent] = field(default_factory=list)
    class_rank: int = 0

    def __iter__(self) -> Iterator[BookEvent]:
        return iter(self.events)


def _ev(
    action: MboAction,
    side: MboSide,
    oid: int,
    px: int,
    size: int,
    ts: int,
    seq: int,
    iid: int = IID,
) -> BookEvent:
    return BookEvent(
        action=action,
        side=side,
        order_id=oid,
        price_dbn=px,
        size=size,
        ts_event=ts,
        sequence=seq,
        instrument_id=iid,
    )


def _add(
    oid: int, side: MboSide, px: int, size: int, ts: int, seq: int = 0, iid: int = IID
) -> BookEvent:
    return _ev(MboAction.ADD, side, oid, px, size, ts, seq, iid)


def _trade(px: int, size: int, ts: int, seq: int = 0) -> BookEvent:
    return _ev(MboAction.TRADE, MboSide.ASK, 0, px, size, ts, seq)


def _cancel_ev(
    oid: int, side: MboSide, px: int, size: int, ts: int, seq: int = 0
) -> BookEvent:
    return _ev(MboAction.CANCEL, side, oid, px, size, ts, seq)


def _submit(
    oid: str,
    side: Side,
    kind: OrderKind,
    size: int,
    px: int | None,
    ts: int,
    *,
    leg: Leg = Leg.ENTRY,
    trade_id: str | None = None,
    oco: str | None = None,
) -> OrderIntent:
    return OrderIntent(
        action=IntentAction.SUBMIT,
        order_id=oid,
        trade_id=trade_id or oid,
        leg=leg,
        kind=kind,
        side=side,
        size=size,
        limit_px_dbn=px,
        submit_ts_ns=ts,
        oco_group_id=oco,
    )


def _cancel_intent(oid: str, ts: int, *, trade_id: str | None = None) -> OrderIntent:
    return OrderIntent(
        action=IntentAction.CANCEL,
        order_id=oid,
        trade_id=trade_id or oid,
        leg=Leg.ENTRY,
        kind=OrderKind.PASSIVE_LIMIT,
        side=Side.BUY,
        size=1,
        limit_px_dbn=1,
        submit_ts_ns=ts,
    )


# --- I/O & Edge-Case Matrix ------------------------------------------


def test_marketable_entry_fills_at_arrival() -> None:
    src = _Source(
        [
            _add(1, MboSide.ASK, 100, 10, ts=1, seq=1),
            _trade(100, 1, ts=200, seq=2),  # a wake >= arrival (2 + 100)
        ]
    )
    intents = [_submit("o1", Side.BUY, OrderKind.MARKETABLE, 5, None, ts=2)]

    outcomes, manifest = simulate(src, intents, CFG, INTERVALS)

    assert len(outcomes) == 1
    (o,) = outcomes
    assert o.terminal_state is TerminalState.FILLED
    assert sum(f.size for f in o.fills) == 5
    assert o.fills[0].px_dbn == 100
    assert o.arrival_ts_ns == 2 + 100
    assert manifest.event_count == 2
    assert manifest.intent_count == 1


def test_passive_fill_after_queue_clears() -> None:
    src = _Source(
        [
            _add(1, MboSide.BID, 100, 3, ts=5, seq=1),  # 3 contracts ahead of us
            _trade(100, 1, ts=200, seq=2),  # wake to activate our order
            _trade(100, 10, ts=300, seq=3),  # volume through our price
        ]
    )
    intents = [_submit("p1", Side.BUY, OrderKind.PASSIVE_LIMIT, 5, 100, ts=10)]

    outcomes, _ = simulate(src, intents, CFG, INTERVALS)

    (o,) = outcomes
    assert o.terminal_state is TerminalState.FILLED
    assert o.queue_ahead_size_at_submit == 3
    assert o.queue_rank_at_submit == 3
    assert sum(f.size for f in o.fills) == 5
    assert all(f.px_dbn == 100 for f in o.fills)


def test_order_never_fills_interval_ends_expired() -> None:
    src = _Source(
        [
            _add(1, MboSide.BID, 100, 10, ts=5, seq=1),
            _trade(100, 1, ts=200, seq=2),  # activates our order; no through-volume
        ]
    )
    intents = [_submit("p1", Side.BUY, OrderKind.PASSIVE_LIMIT, 5, 100, ts=10)]

    outcomes, _ = simulate(src, intents, CFG, [(0, 500)])

    (o,) = outcomes
    assert o.terminal_state is TerminalState.EXPIRED
    assert o.fills == ()
    assert o.time_to_fill_ns is None


def test_cancel_takes_a_latency_hop() -> None:
    src = _Source([_add(1, MboSide.ASK, 100, 10, ts=1, seq=1)])
    intents = [
        _submit("o1", Side.BUY, OrderKind.PASSIVE_LIMIT, 5, 90, ts=10),
        _cancel_intent("o1", ts=20),
    ]

    run = SimRun(CFG, INTERVALS)
    outcomes, _ = run.run(src, intents)

    (o,) = outcomes
    assert o.terminal_state is TerminalState.CANCELLED
    # cancelled at submit_ts (20) + latency (100), never at 20
    assert run.tracker.terminal_ts_ns("o1") == 120


def test_bracket_exit_fill_cancels_entry_and_sibling() -> None:
    src = _Source(
        [
            _trade(50, 1, ts=200, seq=1),  # wake: activate all three legs
            _trade(110, 5, ts=300, seq=2),  # through the TP limit
        ]
    )
    intents = [
        _submit(
            "entry",
            Side.BUY,
            OrderKind.PASSIVE_LIMIT,
            5,
            100,
            ts=10,
            leg=Leg.ENTRY,
            trade_id="t1",
            oco="g1",
        ),
        _submit(
            "tp",
            Side.SELL,
            OrderKind.PASSIVE_LIMIT,
            5,
            110,
            ts=11,
            leg=Leg.EXIT,
            trade_id="t1",
            oco="g1",
        ),
        _submit(
            "sl",
            Side.SELL,
            OrderKind.PASSIVE_LIMIT,
            5,
            90,
            ts=12,
            leg=Leg.EXIT,
            trade_id="t1",
            oco="g1",
        ),
    ]

    outcomes, manifest = simulate(src, intents, CFG, INTERVALS)

    assert [o.order_id for o in outcomes] == ["entry", "tp", "sl"]
    by_id = {o.order_id: o for o in outcomes}
    assert by_id["tp"].terminal_state is TerminalState.FILLED
    assert by_id["entry"].terminal_state is TerminalState.CANCELLED
    assert by_id["sl"].terminal_state is TerminalState.CANCELLED
    assert manifest.event_count == 2
    assert manifest.intent_count == 3


def test_book_event_outside_mask_is_folded_not_traded() -> None:
    src = _Source(
        [
            _add(1, MboSide.ASK, 100, 10, ts=50, seq=1),  # outside [100, 1000)
            _trade(100, 1, ts=250, seq=2),  # wake, inside the mask
        ]
    )
    intents = [_submit("o1", Side.BUY, OrderKind.MARKETABLE, 5, None, ts=150)]

    outcomes, manifest = simulate(src, intents, CFG, [(100, 1000)])

    (o,) = outcomes
    # the outside-mask ADD was folded -> liquidity present -> the in-mask order fills
    assert o.terminal_state is TerminalState.FILLED
    assert manifest.event_count == 2


def test_non_decreasing_submit_ts_violation() -> None:
    src = _Source([])
    intents = [
        _submit("a", Side.BUY, OrderKind.MARKETABLE, 1, None, ts=100),
        _submit("b", Side.BUY, OrderKind.MARKETABLE, 1, None, ts=50),
    ]
    with pytest.raises(IntentLogError):
        simulate(src, intents, CFG, INTERVALS)


def test_cancel_of_unknown_order() -> None:
    src = _Source([])
    intents = [_cancel_intent("ghost", ts=100)]
    with pytest.raises(IntentLogError):
        simulate(src, intents, CFG, INTERVALS)


def test_submit_ts_outside_mask() -> None:
    src = _Source([])
    intents = [_submit("o1", Side.BUY, OrderKind.MARKETABLE, 1, None, ts=5000)]
    with pytest.raises(IntentLogError):
        simulate(src, intents, CFG, [(0, 1000)])


def test_more_than_one_instrument_in_book() -> None:
    src = _Source(
        [
            _add(1, MboSide.ASK, 100, 10, ts=1, seq=1, iid=1),
            _add(2, MboSide.ASK, 100, 10, ts=2, seq=2, iid=2),
        ]
    )
    with pytest.raises(IntentLogError):
        simulate(src, [], CFG, INTERVALS)


def test_determinism_run_twice_byte_identical() -> None:
    def build() -> tuple[_Source, list[OrderIntent]]:
        src = _Source(
            [
                _trade(50, 1, ts=200, seq=1),
                _trade(110, 5, ts=300, seq=2),
            ]
        )
        intents = [
            _submit(
                "entry",
                Side.BUY,
                OrderKind.PASSIVE_LIMIT,
                5,
                100,
                ts=10,
                leg=Leg.ENTRY,
                trade_id="t1",
                oco="g1",
            ),
            _submit(
                "tp",
                Side.SELL,
                OrderKind.PASSIVE_LIMIT,
                5,
                110,
                ts=11,
                leg=Leg.EXIT,
                trade_id="t1",
                oco="g1",
            ),
            _submit(
                "sl",
                Side.SELL,
                OrderKind.PASSIVE_LIMIT,
                5,
                90,
                ts=12,
                leg=Leg.EXIT,
                trade_id="t1",
                oco="g1",
            ),
        ]
        return src, intents

    src_a, intents_a = build()
    src_b, intents_b = build()
    out_a, _ = simulate(src_a, intents_a, CFG, INTERVALS)
    out_b, _ = simulate(src_b, intents_b, CFG, INTERVALS)

    assert [o.model_dump_json() for o in out_a] == [o.model_dump_json() for o in out_b]


def test_determinism_across_hash_seeds() -> None:
    # AD-11 forbids semantic dependence on dict/set iteration order; run the
    # same sim in two fresh interpreters with different PYTHONHASHSEED and diff
    # the serialized OrderOutcome log.
    import subprocess
    import sys

    script = (
        "import json,sys; sys.path.insert(0, %r);"
        "from tests.unit.test_ticksim_sim import "
        "_Source,_submit,_trade,_add,_cancel_ev,CFG,AINT;"
        "from src.ticksim.events import MboSide;"
        "from src.ticksim.orders import Side,OrderKind;"
        "from src.ticksim.sim import simulate;"
        # two passive BUYs fill in one trade, then the bid drops in-window: both
        # AD-28 checks latch and seal in the same tick -- exercises check-list
        # iteration order (AD-11) as well as OrderOutcome emit order.
        "src=_Source([_add(1,MboSide.BID,100,1,ts=5,seq=1),"
        "_add(2,MboSide.BID,99,99,ts=6,seq=2),"
        "_trade(100,50,ts=1000000,seq=3),"
        "_cancel_ev(1,MboSide.BID,100,1,ts=1500000,seq=4)]);"
        "i=[_submit('a',Side.BUY,OrderKind.PASSIVE_LIMIT,5,100,ts=10),"
        "_submit('b',Side.BUY,OrderKind.PASSIVE_LIMIT,5,100,ts=11)];"
        "o,m=simulate(src,i,CFG,AINT);"
        "assert m.adverse_fill_count==2 and all(x.adverse_selection for x in o);"
        "print(json.dumps([x.model_dump_json() for x in o]))"
    ) % str(__import__("pathlib").Path(__file__).resolve().parents[2])

    def run(seed: str) -> str:
        env = {**__import__("os").environ, "PYTHONHASHSEED": seed}
        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        ).stdout

    assert run("0") == run("12345")


def test_manifest_surfaces_book_counters() -> None:
    src = _Source([_cancel_ev(999, MboSide.BID, 100, 5, ts=100, seq=1)])

    outcomes, manifest = simulate(src, [], CFG, INTERVALS)

    assert outcomes == []
    assert manifest.unseen_cm_count > 0
    assert manifest.max_transient_cross_ns == 0
    assert manifest.stale_cross_count == 0
    assert manifest.last_ts_ns == 100
    d = manifest.to_dict()
    json.dumps(d)  # JSON-safe (spine AD-12)
    assert d["unseen_cm_count"] == manifest.unseen_cm_count
    assert d["stale_cross_count"] == manifest.stale_cross_count


# --- Acceptance Criteria ---------------------------------------------


def test_acceptance_bracket_shape_and_manifest_counts() -> None:
    src = _Source([_trade(50, 1, ts=200, seq=1), _trade(110, 5, ts=300, seq=2)])
    intents = [
        _submit(
            "entry",
            Side.BUY,
            OrderKind.PASSIVE_LIMIT,
            5,
            100,
            ts=10,
            leg=Leg.ENTRY,
            trade_id="t1",
            oco="g1",
        ),
        _submit(
            "tp",
            Side.SELL,
            OrderKind.PASSIVE_LIMIT,
            5,
            110,
            ts=11,
            leg=Leg.EXIT,
            trade_id="t1",
            oco="g1",
        ),
        _submit(
            "sl",
            Side.SELL,
            OrderKind.PASSIVE_LIMIT,
            5,
            90,
            ts=12,
            leg=Leg.EXIT,
            trade_id="t1",
            oco="g1",
        ),
    ]

    outcomes, manifest = simulate(src, intents, CFG, INTERVALS)

    assert len(outcomes) == 3
    assert {o.terminal_state for o in outcomes} == {
        TerminalState.FILLED,
        TerminalState.CANCELLED,
    }
    assert manifest.event_count == 2
    assert manifest.intent_count == 3
    assert isinstance(manifest, Manifest)
    assert manifest.config["queue_model"] == QueueModel.BACK_OF_QUEUE.value
    assert manifest.seed == 0
    assert manifest.sibling_run_id is None


def test_simrun_is_single_shot() -> None:
    run = SimRun(CFG, INTERVALS)
    run.run(_Source([]), [])
    with pytest.raises(RuntimeError):
        run.run(_Source([]), [])


# --- review-pass hardening (blind / edge-case / verification-gap) ------


def _replace_intent(
    oid: str, ts: int, *, px: int, size: int = 5, side: Side = Side.BUY
) -> OrderIntent:
    return OrderIntent(
        action=IntentAction.REPLACE,
        order_id=oid,
        replaces_order_id=oid,
        trade_id=oid,
        leg=Leg.ENTRY,
        kind=OrderKind.PASSIVE_LIMIT,
        side=side,
        size=size,
        limit_px_dbn=px,
        submit_ts_ns=ts,
    )


def test_duplicate_submit_order_id_raises() -> None:
    intents = [
        _submit("o1", Side.BUY, OrderKind.MARKETABLE, 5, None, ts=1),
        _submit("o1", Side.BUY, OrderKind.MARKETABLE, 5, None, ts=2),
    ]
    with pytest.raises(IntentLogError):
        simulate(_Source([]), intents, CFG, INTERVALS)


def test_replace_of_unseen_order_raises() -> None:
    with pytest.raises(IntentLogError):
        simulate(_Source([]), [_replace_intent("ghost", ts=5, px=100)], CFG, INTERVALS)


def test_price_change_replace_takes_a_latency_hop_and_re_arrives() -> None:
    # entry passive BUY @100 (queue 3 ahead); replace to @101 at ts=200; the
    # replace re-flights and re-activates at 200 + latency, snapshotting a fresh
    # queue position against the book at that arrival tick.
    src = _Source(
        [
            _add(1, MboSide.BID, 100, 3, ts=5, seq=1),
            _add(2, MboSide.BID, 101, 7, ts=6, seq=2),
            _trade(101, 20, ts=500, seq=3),
        ]
    )
    intents = [
        _submit("o1", Side.BUY, OrderKind.PASSIVE_LIMIT, 5, 100, ts=10),
        _replace_intent("o1", ts=200, px=101, size=5),
    ]
    outcomes, _ = simulate(src, intents, CFG, INTERVALS)
    (o,) = outcomes
    assert o.terminal_state is TerminalState.FILLED
    # queue position re-taken at the *replaced* price 101 -> 7 contracts ahead
    assert o.queue_ahead_size_at_submit == 7
    assert o.arrival_ts_ns == 200 + 100  # fresh hop, not 10 + 100
    assert all(f.px_dbn == 101 for f in o.fills)


def test_deferred_cancel_of_already_expired_order_is_dropped() -> None:
    # order expires at the interval end (500); a CANCEL submitted at 450 lands
    # at 450 + 100 = 550, after expiry -> silently dropped, no crash.
    src = _Source([_add(1, MboSide.BID, 100, 10, ts=5, seq=1)])
    intents = [
        _submit("o1", Side.BUY, OrderKind.PASSIVE_LIMIT, 5, 100, ts=10),
        _cancel_intent("o1", ts=450),
    ]
    run = SimRun(CFG, [(0, 500)])
    outcomes, _ = run.run(src, intents)
    (o,) = outcomes
    assert o.terminal_state is TerminalState.EXPIRED
    assert run.tracker.terminal_ts_ns("o1") == 500


def test_adjacent_intervals_merge_no_seam_expiry() -> None:
    # [(0,500),(500,1000)] must merge to [(0,1000)] -- an order live across 500
    # is NOT force-expired at the internal seam.
    src = _Source(
        [
            _add(1, MboSide.ASK, 100, 10, ts=5, seq=1),
            _trade(100, 1, ts=700, seq=2),  # wake after the old seam
        ]
    )
    intents = [_submit("o1", Side.BUY, OrderKind.MARKETABLE, 5, None, ts=10)]
    outcomes, manifest = simulate(src, intents, CFG, [(0, 500), (500, 1000)])
    (o,) = outcomes
    assert o.terminal_state is TerminalState.FILLED
    assert manifest.valid_intervals == ((0, 1000),)


def test_overlapping_and_malformed_intervals() -> None:
    assert SimRun(CFG, [(0, 300), (200, 600)]).valid_intervals == ((0, 600),)
    with pytest.raises(ValueError):
        SimRun(CFG, [(500, 100)])
    with pytest.raises(ValueError):
        SimRun(CFG, [])


def test_negative_latency_rejected() -> None:
    bad = CFG.model_copy(update={"latency_ns": -1})
    with pytest.raises(ValueError):
        SimRun(bad, INTERVALS)


def test_second_instrument_via_trade_only_is_caught() -> None:
    src = _Source(
        [
            _add(1, MboSide.BID, 100, 5, ts=1, seq=1),
            _ev(MboAction.TRADE, MboSide.ASK, 0, 4000, 1, ts=2, seq=2, iid=2),
        ]
    )
    with pytest.raises(IntentLogError):
        simulate(src, [], CFG, INTERVALS)


def test_marketable_limit_through_sim() -> None:
    src = _Source(
        [
            _add(1, MboSide.ASK, 100, 2, ts=1, seq=1),
            _add(2, MboSide.ASK, 125, 10, ts=2, seq=2),
            _trade(100, 1, ts=300, seq=3),
        ]
    )
    intents = [_submit("m1", Side.BUY, OrderKind.MARKETABLE_LIMIT, 5, 100, ts=5)]
    outcomes, _ = simulate(src, intents, CFG, INTERVALS)
    (o,) = outcomes
    # fills only the 2 @100 (its limit); remainder inert (fills.py walk-once)
    assert sum(f.size for f in o.fills) == 2
    assert o.terminal_state is TerminalState.EXPIRED  # remainder never fills


def test_bracket_cascade_lands_in_the_same_tick() -> None:
    # tp (SELL passive @90) + sl (SELL passive @80) in one OCO group; a trade
    # through 90 fills tp -> sl cancels in the SAME tick (AD-25).
    src = _Source([_trade(90, 50, ts=400, seq=1)])
    intents = [
        _submit(
            "tp", Side.SELL, OrderKind.PASSIVE_LIMIT, 2, 90, ts=5, oco="b", leg=Leg.EXIT
        ),
        _submit(
            "sl", Side.SELL, OrderKind.PASSIVE_LIMIT, 2, 80, ts=5, oco="b", leg=Leg.EXIT
        ),
    ]
    run = SimRun(CFG, INTERVALS)
    outcomes, _ = run.run(src, intents)
    by = {o.order_id: o for o in outcomes}
    assert by["tp"].terminal_state is TerminalState.FILLED
    assert by["sl"].terminal_state is TerminalState.CANCELLED
    # AD-25: the cascade cancels at the same tick as the TP fill
    assert run.tracker.terminal_ts_ns("sl") == by["tp"].fills[-1].ts_ns == 400


def test_manifest_to_dict_is_json_safe() -> None:
    _, manifest = simulate(_Source([]), [], CFG, INTERVALS)
    dumped = json.dumps(manifest.to_dict())
    assert '"queue_model": "back_of_queue"' in dumped
    assert manifest.to_dict()["oco_cascade_cancel_count"] == 0
    assert manifest.to_dict()["adverse_fill_count"] == 0


# --- AD-28 adverse-selection deferred-check queue --------------------
#
# One test per row of the spec's I/O & Edge-Case Matrix, plus the
# `manifest.adverse_fill_count` accumulator. The 1 s adverse window
# (ADVERSE_SELECTION_WINDOW_NS = 1e9) dwarfs the tiny INTERVALS above, so these
# use a wide `AINT` and ~1 ms fill timestamps; an adverse quote move is driven
# by a `_cancel_ev` that removes the best bid/ask level.

WINDOW_NS = 1_000_000_000
AINT = [(0, 5_000_000_000)]


def test_passive_buy_bid_drops_in_window_is_adverse() -> None:
    src = _Source(
        [
            _add(1, MboSide.BID, 100, 3, ts=5, seq=1),  # 3 contracts ahead @ P
            _add(2, MboSide.BID, 99, 100, ts=6, seq=2),  # a lower bid that stays
            _trade(100, 10, ts=1_000_000, seq=3),  # fills our 5 @100
            _cancel_ev(1, MboSide.BID, 100, 3, ts=1_500_000, seq=4),  # bid -> 99
        ]
    )
    intents = [_submit("p1", Side.BUY, OrderKind.PASSIVE_LIMIT, 5, 100, ts=10)]

    outcomes, manifest = simulate(src, intents, CFG, AINT)

    (o,) = outcomes
    assert o.terminal_state is TerminalState.FILLED
    assert o.adverse_selection is True
    assert manifest.adverse_fill_count == 1


def test_passive_buy_bid_drops_after_window_not_adverse() -> None:
    src = _Source(
        [
            _add(1, MboSide.BID, 100, 3, ts=5, seq=1),
            _add(2, MboSide.BID, 99, 100, ts=6, seq=2),
            _trade(100, 10, ts=1_000_000, seq=3),
            _cancel_ev(
                1, MboSide.BID, 100, 3, ts=1_000_000 + WINDOW_NS + 500_000, seq=4
            ),
        ]
    )
    intents = [_submit("p1", Side.BUY, OrderKind.PASSIVE_LIMIT, 5, 100, ts=10)]

    outcomes, manifest = simulate(src, intents, CFG, AINT)

    (o,) = outcomes
    assert o.terminal_state is TerminalState.FILLED
    assert o.adverse_selection is False
    assert manifest.adverse_fill_count == 0


def test_passive_buy_transient_dip_that_reverts_is_adverse() -> None:
    src = _Source(
        [
            _add(1, MboSide.BID, 100, 3, ts=5, seq=1),
            _add(2, MboSide.BID, 99, 100, ts=6, seq=2),
            _trade(100, 10, ts=1_000_000, seq=3),
            _cancel_ev(1, MboSide.BID, 100, 3, ts=1_300_000, seq=4),  # bid -> 99
            _add(3, MboSide.BID, 100, 8, ts=1_600_000, seq=5),  # bid back to 100
        ]
    )
    intents = [_submit("p1", Side.BUY, OrderKind.PASSIVE_LIMIT, 5, 100, ts=10)]

    outcomes, manifest = simulate(src, intents, CFG, AINT)

    (o,) = outcomes
    assert o.adverse_selection is True  # any point in the window counts
    assert manifest.adverse_fill_count == 1


def test_passive_sell_ask_rises_in_window_is_adverse() -> None:
    src = _Source(
        [
            _add(1, MboSide.ASK, 100, 3, ts=5, seq=1),  # 3 ahead @ P
            _add(2, MboSide.ASK, 101, 100, ts=6, seq=2),  # a higher ask that stays
            _trade(100, 10, ts=1_000_000, seq=3),  # fills our SELL 5 @100
            _cancel_ev(1, MboSide.ASK, 100, 3, ts=1_500_000, seq=4),  # ask -> 101
        ]
    )
    intents = [_submit("s1", Side.SELL, OrderKind.PASSIVE_LIMIT, 5, 100, ts=10)]

    outcomes, manifest = simulate(src, intents, CFG, AINT)

    (o,) = outcomes
    assert o.terminal_state is TerminalState.FILLED
    assert o.adverse_selection is True
    assert manifest.adverse_fill_count == 1


def test_passive_buy_quote_side_empties_not_adverse() -> None:
    src = _Source(
        [
            _add(1, MboSide.BID, 100, 3, ts=5, seq=1),  # the only bid
            _trade(100, 10, ts=1_000_000, seq=2),
            _cancel_ev(1, MboSide.BID, 100, 3, ts=1_500_000, seq=3),  # bid -> None
        ]
    )
    intents = [_submit("p1", Side.BUY, OrderKind.PASSIVE_LIMIT, 5, 100, ts=10)]

    outcomes, manifest = simulate(src, intents, CFG, AINT)

    (o,) = outcomes
    assert o.terminal_state is TerminalState.FILLED
    assert o.adverse_selection is False  # a None touch never triggers
    assert manifest.adverse_fill_count == 0


def test_marketable_fill_is_never_adverse() -> None:
    src = _Source(
        [
            _add(1, MboSide.ASK, 100, 100, ts=5, seq=1),  # ask to hit
            _add(2, MboSide.BID, 100, 5, ts=6, seq=2),  # bid @ fill px
            _add(3, MboSide.BID, 98, 100, ts=7, seq=3),  # a lower bid
            _cancel_ev(2, MboSide.BID, 100, 5, ts=1_500_000, seq=4),  # bid -> 98
        ]
    )
    intents = [_submit("m1", Side.BUY, OrderKind.MARKETABLE, 5, None, ts=10)]

    outcomes, manifest = simulate(src, intents, CFG, AINT)

    (o,) = outcomes
    assert o.terminal_state is TerminalState.FILLED
    assert o.adverse_selection is False  # marketable fills are never marked
    assert manifest.adverse_fill_count == 0


def test_adverse_window_crosses_interval_end_still_evaluated() -> None:
    src = _Source(
        [
            _add(1, MboSide.BID, 100, 3, ts=5, seq=1),
            _add(2, MboSide.BID, 99, 100, ts=6, seq=2),
            _trade(100, 10, ts=1_000_000, seq=3),  # fill in-mask
            _cancel_ev(1, MboSide.BID, 100, 3, ts=1_400_000, seq=4),  # past the end
        ]
    )
    intents = [_submit("p1", Side.BUY, OrderKind.PASSIVE_LIMIT, 5, 100, ts=10)]

    outcomes, manifest = simulate(src, intents, CFG, [(0, 1_200_000)])

    (o,) = outcomes
    assert o.terminal_state is TerminalState.FILLED
    assert o.adverse_selection is True  # book is continuous across the boundary
    assert manifest.adverse_fill_count == 1


def test_run_ends_before_window_closes_seals_latched_hit() -> None:
    # interval ends at 3_000_000 -- every wake precedes the ~1.002e9 deadline,
    # so the check is genuinely still open when _loop returns and is sealed by
    # run()'s `_step_adverse(_max_deadline + 1)` call, before finalize().
    src = _Source(
        [
            _add(1, MboSide.BID, 100, 3, ts=5, seq=1),
            _add(2, MboSide.BID, 99, 100, ts=6, seq=2),
            _trade(100, 10, ts=2_000_000, seq=3),
            _cancel_ev(
                1, MboSide.BID, 100, 3, ts=2_400_000, seq=4
            ),  # adverse, in-window
        ]
    )
    intents = [_submit("p1", Side.BUY, OrderKind.PASSIVE_LIMIT, 5, 100, ts=10)]

    outcomes, manifest = simulate(src, intents, CFG, [(0, 3_000_000)])

    (o,) = outcomes
    assert manifest.last_ts_ns == 2_400_000
    assert o.adverse_selection is True
    assert manifest.adverse_fill_count == 1


def test_adverse_sealed_in_loop_when_a_wake_crosses_the_deadline() -> None:
    # a real book event lands past the deadline -> the check seals INSIDE _loop
    # (not via the run-end sentinel); a later still-open check must not be
    # re-counted at run end.
    src = _Source(
        [
            _add(1, MboSide.BID, 100, 3, ts=5, seq=1),
            _add(2, MboSide.BID, 99, 100, ts=6, seq=2),
            _trade(100, 10, ts=1_000_000, seq=3),  # fill p1 @100
            _cancel_ev(
                1, MboSide.BID, 100, 3, ts=1_500_000, seq=4
            ),  # adverse in-window
            _trade(
                99, 1, ts=1_000_000 + WINDOW_NS + 10, seq=5
            ),  # wake past p1 deadline
        ]
    )
    intents = [_submit("p1", Side.BUY, OrderKind.PASSIVE_LIMIT, 5, 100, ts=10)]

    outcomes, manifest = simulate(src, intents, CFG, AINT)
    (o,) = outcomes
    assert o.adverse_selection is True
    assert manifest.adverse_fill_count == 1  # exactly once


def test_non_book_wake_does_not_latch_adverse() -> None:
    # the bid is below P for the whole window but the ONLY in-window wake is an
    # unrelated order arrival (no book delta) -> must NOT latch (AD-11: the
    # marker cannot depend on arrival timing).
    src = _Source(
        [
            _add(1, MboSide.BID, 99, 100, ts=5, seq=1),  # only bid: 99 < P=100
            _trade(100, 10, ts=1_000_000, seq=2),  # fills p1 @100 (bid already 99)
            _trade(
                99, 1, ts=1_000_000 + WINDOW_NS + 5, seq=3
            ),  # seal wake, past deadline
        ]
    )
    intents = [
        _submit("p1", Side.BUY, OrderKind.PASSIVE_LIMIT, 5, 100, ts=10),
        # p2 arrives at 1_400_000 + latency -- a non-book wake inside p1's window
        _submit("p2", Side.BUY, OrderKind.PASSIVE_LIMIT, 1, 50, ts=1_400_000),
    ]
    outcomes, manifest = simulate(src, intents, CFG, AINT)
    by = {o.order_id: o for o in outcomes}
    # bid 99 < 100 held all through p1's window, but only at the fill tick and an
    # arrival wake -- neither latches -> not adverse (matches "fill tick itself").
    assert by["p1"].adverse_selection is False
    assert manifest.adverse_fill_count == 0


def test_marketable_limit_fill_is_never_adverse() -> None:
    src = _Source(
        [
            _add(1, MboSide.ASK, 100, 10, ts=5, seq=1),
            _trade(100, 1, ts=1_000_000, seq=2),  # wake >= arrival; m1 walks @100
            _add(2, MboSide.BID, 90, 100, ts=1_500_000, seq=3),  # bid far below fill px
        ]
    )
    intents = [_submit("m1", Side.BUY, OrderKind.MARKETABLE_LIMIT, 5, 100, ts=10)]
    outcomes, manifest = simulate(src, intents, CFG, AINT)
    (o,) = outcomes
    assert o.terminal_state is TerminalState.FILLED
    assert o.adverse_selection is False
    assert manifest.adverse_fill_count == 0


def test_fill_tick_itself_adverse_is_not_marked() -> None:
    src = _Source(
        [
            _add(1, MboSide.BID, 99, 100, ts=5, seq=1),  # only bid: 99 < our P=100
            _trade(100, 10, ts=1_000_000, seq=2),  # fills our passive BUY @100
            _trade(100, 1, ts=1_000_000 + WINDOW_NS + 1, seq=3),  # tick past deadline
        ]
    )
    intents = [_submit("p1", Side.BUY, OrderKind.PASSIVE_LIMIT, 5, 100, ts=10)]

    outcomes, manifest = simulate(src, intents, CFG, AINT)

    (o,) = outcomes
    assert o.terminal_state is TerminalState.FILLED
    # bid < P only at the fill tick (strict now > fill_ts) and at a tick that is
    # already past the deadline (not evaluated) -> sealed False.
    assert o.adverse_selection is False
    assert manifest.adverse_fill_count == 0


def test_manifest_adverse_fill_count_counts_each_hit() -> None:
    src = _Source(
        [
            _add(1, MboSide.BID, 100, 1, ts=5, seq=1),
            _add(2, MboSide.BID, 99, 100, ts=6, seq=2),
            _trade(100, 50, ts=1_000_000, seq=3),  # fills both passive BUYs
            _cancel_ev(1, MboSide.BID, 100, 1, ts=1_500_000, seq=4),  # bid -> 99
        ]
    )
    intents = [
        _submit("p1", Side.BUY, OrderKind.PASSIVE_LIMIT, 5, 100, ts=10),
        _submit("p2", Side.BUY, OrderKind.PASSIVE_LIMIT, 5, 100, ts=11),
    ]

    outcomes, manifest = simulate(src, intents, CFG, AINT)

    assert all(o.terminal_state is TerminalState.FILLED for o in outcomes)
    assert all(o.adverse_selection is True for o in outcomes)
    assert manifest.adverse_fill_count == 2
