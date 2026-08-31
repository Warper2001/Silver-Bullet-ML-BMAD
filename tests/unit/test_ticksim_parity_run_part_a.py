"""Unit tests for ``src/ticksim/parity/part_a_runner.py`` (spine AD-17).

Every case runs against a hand-built in-memory :class:`BookEventSource` (a list
of :class:`BookEvent`s) so the runner is exercised end-to-end -- real
``sim.simulate`` folding a real book, real ``compare_fills`` / ``aggregate`` --
without any ``.dbn.zst`` fixture.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from src.ticksim.config import MNQ_TICK_DBN
from src.ticksim.events import BookEvent, MboAction, MboSide
from src.ticksim.orders import IntentAction, Leg, OrderIntent, OrderKind, Side
from src.ticksim.parity import part_a_runner
from src.ticksim.parity.part_a import (
    PartAError,
    ReconstructedTrade,
    RealFill,
)
from src.ticksim.parity.part_a_runner import (
    PART_A_WINDOW_PAD_NS,
    _touch_at,
    run_part_a,
)
from src.ticksim.sim import IntentLogError

# --------------------------------------------------------------------------- #
# fixtures / builders
# --------------------------------------------------------------------------- #

IID = 42004800
TICK = MNQ_TICK_DBN
P = 20_000_000_000_000  # 20000.0 in DBN 1e-9 fixed-point
BID_PX = P - TICK
ASK_PX = P + TICK

B = 1_700_000_000 * 1_000_000_000  # ns base (whole seconds -> exact)
LAT = 250_000_000  # PRIMARY latency (250 ms)
HOLD = 600 * 1_000_000_000  # 600 s hold


class ListSource:
    """Re-iterable in-memory :class:`BookEventSource` (spine AD-18)."""

    class_rank = 0

    def __init__(self, events: list[BookEvent]) -> None:
        self._events = list(events)

    def __iter__(self) -> Iterator[BookEvent]:
        return iter(self._events)


class OneShotSource:
    """A source that is only iterable once -- second pass raises."""

    class_rank = 0

    def __init__(self, events: list[BookEvent]) -> None:
        self._it: Iterator[BookEvent] | None = iter(list(events))

    def __iter__(self) -> Iterator[BookEvent]:
        if self._it is None:
            raise RuntimeError("source already consumed")
        it, self._it = self._it, None
        return it


def be(
    action: MboAction,
    side: MboSide,
    order_id: int,
    price_dbn: int,
    size: int,
    ts: int,
    seq: int,
) -> BookEvent:
    return BookEvent(
        action=action,
        side=side,
        order_id=order_id,
        price_dbn=price_dbn,
        size=size,
        ts_event=ts,
        sequence=seq,
        instrument_id=IID,
    )


def make_trade(
    *,
    trade_id: str,
    entry_side: Side,
    entry_submit: int,
    exit_submit: int,
    entry_real_dbn: int,
    exit_real_dbn: int,
    size: int = 1,
) -> ReconstructedTrade:
    exit_side = Side.SELL if entry_side is Side.BUY else Side.BUY
    oco = f"{trade_id}-oco"
    entry_oid, exit_oid = f"{trade_id}-e", f"{trade_id}-x"
    entry_i = OrderIntent(
        action=IntentAction.SUBMIT,
        order_id=entry_oid,
        trade_id=trade_id,
        leg=Leg.ENTRY,
        kind=OrderKind.MARKETABLE,
        side=entry_side,
        size=size,
        submit_ts_ns=entry_submit,
        oco_group_id=oco,
    )
    exit_i = OrderIntent(
        action=IntentAction.SUBMIT,
        order_id=exit_oid,
        trade_id=trade_id,
        leg=Leg.EXIT,
        kind=OrderKind.MARKETABLE,
        side=exit_side,
        size=size,
        submit_ts_ns=exit_submit,
        oco_group_id=oco,
    )
    real_fills = (
        RealFill(
            order_id=entry_oid,
            leg=Leg.ENTRY,
            side=entry_side,
            size=size,
            price_dbn=entry_real_dbn,
            ts_ns=entry_submit,
            fidelity="bar_reconstructed",
        ),
        RealFill(
            order_id=exit_oid,
            leg=Leg.EXIT,
            side=exit_side,
            size=size,
            price_dbn=exit_real_dbn,
            ts_ns=exit_submit,
            fidelity="bar_reconstructed",
        ),
    )
    return ReconstructedTrade(
        trade_id=trade_id,
        intents=(entry_i, exit_i),
        real_fills=real_fills,
        fidelity="bar_reconstructed",
    )


def both_sides_full_book() -> list[BookEvent]:
    """Deep bid + ask resting for the whole window -- both legs fill."""
    return [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 100, ts=B, seq=1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 100, ts=B, seq=2),
    ]


def one_source(events: list[BookEvent]) -> Callable[[ReconstructedTrade], ListSource]:
    src = ListSource(events)
    return lambda _trade: src


# --------------------------------------------------------------------------- #
# both legs filled
# --------------------------------------------------------------------------- #


def test_both_legs_filled_offsets_and_fail_on_small_n() -> None:
    # sim entry (BUY) fills at ASK_PX; real entry = ASK_PX - 1 tick -> +1.0
    # (sim paid a tick more). sim exit (SELL) fills at BID_PX; real exit =
    # BID_PX - 2 ticks -> -2.0 (sim received two ticks more -> better).
    trade = make_trade(
        trade_id="t1",
        entry_side=Side.BUY,
        entry_submit=B,
        exit_submit=B + HOLD,
        entry_real_dbn=ASK_PX - TICK,
        exit_real_dbn=BID_PX - 2 * TICK,
    )
    result = run_part_a([trade], one_source(both_sides_full_book()))

    assert result.stats.n == 2
    assert [e.signed_error_ticks for e in result.errors] == [1.0, -2.0]
    assert all(e.miss_reason is None for e in result.errors)
    assert result.unresolved_misses == 0
    assert result.verdict == "FAIL"  # N < PART_A_MIN_N
    assert "N=2" in result.reason


# --------------------------------------------------------------------------- #
# exit leg unfilled -> resolved from the touch (+1 tick of adverse slip)
# --------------------------------------------------------------------------- #


def test_exit_unfilled_resolved_from_touch_buy_exit() -> None:
    # short trade: entry SELL (fills off the bid), exit BUY. The ask is pulled
    # 100 ms after the real exit ts but 150 ms before the sim exit arrival, so
    # the sim exit misses and is priced from the touch that still existed at
    # real_ts_ns: ASK_PX + one tick of adverse slip (AD-17).
    trade = make_trade(
        trade_id="t1",
        entry_side=Side.SELL,
        entry_submit=B,
        exit_submit=B + HOLD,
        entry_real_dbn=BID_PX,
        exit_real_dbn=ASK_PX,
    )
    events = [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 100, ts=B, seq=1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 100, ts=B, seq=2),
        be(
            MboAction.CANCEL,
            MboSide.ASK,
            2,
            ASK_PX,
            100,
            ts=B + HOLD + 100_000_000,
            seq=3,
        ),
    ]
    result = run_part_a([trade], one_source(events))

    entry_err, exit_err = result.errors
    assert entry_err.signed_error_ticks == 0.0
    assert exit_err.miss_reason == "leg_unfilled"  # still identifiable
    assert exit_err.sim_terminal_state == "expired"
    assert exit_err.sim_vwap_dbn == ASK_PX + MNQ_TICK_DBN
    assert exit_err.signed_error_ticks == pytest.approx(1.0)  # BUY: paid a tick more
    assert result.unresolved_misses == 0


def test_exit_unfilled_resolved_from_touch_sell_exit() -> None:
    # long trade: exit SELL, priced from BID_PX - one tick of adverse slip.
    trade = make_trade(
        trade_id="t1",
        entry_side=Side.BUY,
        entry_submit=B,
        exit_submit=B + HOLD,
        entry_real_dbn=ASK_PX,
        exit_real_dbn=BID_PX,
    )
    events = [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 100, ts=B, seq=1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 100, ts=B, seq=2),
        be(
            MboAction.CANCEL,
            MboSide.BID,
            1,
            BID_PX,
            100,
            ts=B + HOLD + 100_000_000,
            seq=3,
        ),
    ]
    result = run_part_a([trade], one_source(events))

    exit_err = result.errors[1]
    assert exit_err.miss_reason == "leg_unfilled"
    assert exit_err.sim_vwap_dbn == BID_PX - MNQ_TICK_DBN
    assert result.unresolved_misses == 0


# --------------------------------------------------------------------------- #
# exit leg unfilled + no touch that side at exit_ts -> PartAError
# --------------------------------------------------------------------------- #


def test_exit_unfilled_no_touch_raises_part_a_error() -> None:
    # the ask is cancelled *before* the real exit ts, so _touch_at returns
    # (bid, None) and the BUY-exit miss cannot be priced.
    trade = make_trade(
        trade_id="t1",
        entry_side=Side.SELL,
        entry_submit=B,
        exit_submit=B + HOLD,
        entry_real_dbn=BID_PX,
        exit_real_dbn=ASK_PX,
    )
    events = [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 100, ts=B, seq=1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 100, ts=B, seq=2),
        be(MboAction.CANCEL, MboSide.ASK, 2, ASK_PX, 100, ts=B + HOLD - 1_000, seq=3),
    ]
    with pytest.raises(PartAError, match="un-priceable"):
        run_part_a([trade], one_source(events))


# --------------------------------------------------------------------------- #
# several trades -> exactly one aggregate call over the concatenated errors
# --------------------------------------------------------------------------- #


def test_multi_trade_single_aggregate_call(monkeypatch: pytest.MonkeyPatch) -> None:
    filled = make_trade(
        trade_id="t1",
        entry_side=Side.BUY,
        entry_submit=B,
        exit_submit=B + HOLD,
        entry_real_dbn=ASK_PX,
        exit_real_dbn=BID_PX,
    )
    missed = make_trade(
        trade_id="t2",
        entry_side=Side.BUY,
        entry_submit=B,
        exit_submit=B + HOLD,
        entry_real_dbn=ASK_PX,
        exit_real_dbn=BID_PX,
    )
    filled2 = make_trade(
        trade_id="t3",
        entry_side=Side.BUY,
        entry_submit=B,
        exit_submit=B + HOLD,
        entry_real_dbn=ASK_PX,
        exit_real_dbn=BID_PX,
    )
    missed_events = [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 100, ts=B, seq=1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 100, ts=B, seq=2),
        be(
            MboAction.CANCEL,
            MboSide.BID,
            1,
            BID_PX,
            100,
            ts=B + HOLD + 100_000_000,
            seq=3,
        ),
    ]
    per_trade = {
        "t1": ListSource(both_sides_full_book()),
        "t2": ListSource(missed_events),
        "t3": ListSource(both_sides_full_book()),
    }

    calls: list[int] = []
    real_aggregate = part_a_runner.aggregate

    def spy(errors: object) -> object:
        seq = list(errors)  # type: ignore[call-overload]
        calls.append(len(seq))
        return real_aggregate(seq)  # type: ignore[arg-type]

    monkeypatch.setattr(part_a_runner, "aggregate", spy)

    result = run_part_a(
        [filled, missed, filled2], lambda trade: per_trade[trade.trade_id]
    )

    assert calls == [6]  # one call, six errors (3 trades x 2 legs)
    assert result.stats.n == 6
    assert result.unresolved_misses == 0
    resolved_misses = [
        e
        for e in result.errors
        if e.miss_reason == "leg_unfilled" and e.signed_error_ticks is not None
    ]
    assert len(resolved_misses) == 1


# --------------------------------------------------------------------------- #
# window pad keeps a boundary order alive
# --------------------------------------------------------------------------- #


def test_window_pad_keeps_boundary_order_alive() -> None:
    trade = make_trade(
        trade_id="t1",
        entry_side=Side.BUY,
        entry_submit=B,
        exit_submit=B + HOLD,
        entry_real_dbn=ASK_PX,
        exit_real_dbn=BID_PX,
    )
    # default pad (5 min): the exit submit at hi == B + HOLD stays inside the
    # padded half-open window and the run completes.
    run_part_a([trade], one_source(both_sides_full_book()))

    # pad_ns=0: the window is [lo, hi) and the exit submit_ts == hi is outside
    # it, so sim rejects the intent log up front.
    with pytest.raises(IntentLogError):
        run_part_a([trade], one_source(both_sides_full_book()), pad_ns=0)


def test_negative_epoch_stamp_does_not_pass_a_negative_interval_bound() -> None:
    # a trade stamped inside the pad of the Unix epoch: lo - pad_ns would be
    # negative; run_part_a floors the interval start at 0.
    trade = make_trade(
        trade_id="t1",
        entry_side=Side.BUY,
        entry_submit=1_000,
        exit_submit=1_000 + HOLD,
        entry_real_dbn=ASK_PX,
        exit_real_dbn=BID_PX,
    )
    events = [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 100, ts=0, seq=1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 100, ts=0, seq=2),
    ]
    result = run_part_a([trade], one_source(events))  # no ValueError from sim
    assert result.stats.n == 2


# --------------------------------------------------------------------------- #
# _touch_at boundedness
# --------------------------------------------------------------------------- #


def test_touch_at_is_bounded_by_ts() -> None:
    events = [
        be(MboAction.ADD, MboSide.BID, 1, P, 10, ts=B, seq=1),
        be(MboAction.ADD, MboSide.ASK, 2, P + 4 * TICK, 10, ts=B, seq=2),
        # raises the bid, but it is stamped past the cutoff -> must not apply
        be(MboAction.MODIFY, MboSide.BID, 1, P + TICK, 10, ts=B + 5_000, seq=3),
    ]
    bid, ask = _touch_at(ListSource(events), B + 1_000)
    assert bid == P  # NOT P + TICK
    assert ask == P + 4 * TICK


def test_touch_at_applies_event_exactly_at_ts() -> None:
    events = [
        be(MboAction.ADD, MboSide.BID, 1, P, 10, ts=B, seq=1),
        be(MboAction.ADD, MboSide.BID, 2, P + TICK, 10, ts=B + 1_000, seq=2),
    ]
    bid, _ask = _touch_at(ListSource(events), B + 1_000)  # inclusive lower bound
    assert bid == P + TICK


def test_touch_at_empty_before_ts_returns_none_none() -> None:
    events = [be(MboAction.ADD, MboSide.BID, 1, P, 10, ts=B + 10_000, seq=1)]
    assert _touch_at(ListSource(events), B) == (None, None)


def test_touch_at_misordered_source_raises_part_a_error() -> None:
    # a window stream whose ts_event regresses -> BookReplay raises BookWalkError
    # which _touch_at translates to PartAError (fail loud, never truncate).
    events = [
        be(MboAction.ADD, MboSide.BID, 1, P, 10, ts=B + 1_000, seq=1),
        be(MboAction.ADD, MboSide.BID, 2, P + TICK, 10, ts=B, seq=2),  # goes back
    ]
    with pytest.raises(PartAError, match="ts_event regressed"):
        _touch_at(ListSource(events), B + 10_000)


def test_touch_at_multi_instrument_source_raises_part_a_error() -> None:
    # a second instrument_id inside the walk's cutoff -> PartAError (sim's own
    # multi-instrument guard is lazy and could miss an id past ts_ns).
    events = [
        be(MboAction.ADD, MboSide.BID, 1, P, 10, ts=B, seq=1),
        BookEvent(
            action=MboAction.ADD,
            side=MboSide.ASK,
            order_id=2,
            price_dbn=P + TICK,
            size=10,
            ts_event=B + 1_000,
            sequence=2,
            instrument_id=IID + 1,
        ),
    ]
    with pytest.raises(PartAError, match="multi-instrument"):
        _touch_at(ListSource(events), B + 10_000)


def test_touch_at_source_iterator_error_raises_part_a_error() -> None:
    # a window stream whose iterator blows up mid-scan (I/O, decode) -> the
    # _bookwalk non-StopIteration wrap -> BookWalkError -> PartAError.
    class BoomSource:
        class_rank = 0

        def __iter__(self) -> Iterator[BookEvent]:
            def _gen() -> Iterator[BookEvent]:
                yield be(MboAction.ADD, MboSide.BID, 1, P, 10, ts=B, seq=1)
                raise RuntimeError("window file truncated")

            return _gen()

    with pytest.raises(PartAError, match="window file truncated"):
        _touch_at(BoomSource(), B + 10_000)


# --------------------------------------------------------------------------- #
# propagation: non-re-iterable source, missing window
# --------------------------------------------------------------------------- #


def test_non_reiterable_source_propagates_runtime_error() -> None:
    # sim consumes the one shot; the _touch_at re-walk to price the miss raises.
    trade = make_trade(
        trade_id="t1",
        entry_side=Side.BUY,
        entry_submit=B,
        exit_submit=B + HOLD,
        entry_real_dbn=ASK_PX,
        exit_real_dbn=BID_PX,
    )
    events = [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 100, ts=B, seq=1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 100, ts=B, seq=2),
        be(
            MboAction.CANCEL,
            MboSide.BID,
            1,
            BID_PX,
            100,
            ts=B + HOLD + 100_000_000,
            seq=3,
        ),
    ]
    src = OneShotSource(events)
    with pytest.raises(RuntimeError):
        run_part_a([trade], lambda _t: src)


def test_source_for_error_propagates_unchanged() -> None:
    trade = make_trade(
        trade_id="t1",
        entry_side=Side.BUY,
        entry_submit=B,
        exit_submit=B + HOLD,
        entry_real_dbn=ASK_PX,
        exit_real_dbn=BID_PX,
    )

    def boom(_trade: ReconstructedTrade) -> ListSource:
        raise FileNotFoundError("missing window file")

    with pytest.raises(FileNotFoundError, match="missing window file"):
        run_part_a([trade], boom)


def test_pad_constant_is_five_minutes() -> None:
    assert PART_A_WINDOW_PAD_NS == 5 * 60 * 1_000_000_000


# --------------------------------------------------------------------------- #
# front-month filtering is the caller's job / multi-instrument -> IntentLogError
# --------------------------------------------------------------------------- #


def test_multi_instrument_source_raises_intent_log_error() -> None:
    trade = make_trade(
        trade_id="t1",
        entry_side=Side.BUY,
        entry_submit=B,
        exit_submit=B + HOLD,
        entry_real_dbn=ASK_PX,
        exit_real_dbn=BID_PX,
    )
    other = BookEvent(
        action=MboAction.ADD,
        side=MboSide.ASK,
        order_id=9,
        price_dbn=ASK_PX,
        size=50,
        ts_event=B,
        sequence=3,
        instrument_id=IID + 1,  # a second contract in the same stream
    )
    events = [*both_sides_full_book(), other]
    with pytest.raises(IntentLogError):
        run_part_a([trade], one_source(events))


# --------------------------------------------------------------------------- #
# config forwarding / one simulate call per trade
# --------------------------------------------------------------------------- #


def test_config_is_forwarded_to_simulate(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ticksim.config import OPTIMISTIC

    trade = make_trade(
        trade_id="t1",
        entry_side=Side.BUY,
        entry_submit=B,
        exit_submit=B + HOLD,
        entry_real_dbn=ASK_PX,
        exit_real_dbn=BID_PX,
    )
    seen: list[object] = []
    real_sim = part_a_runner.simulate

    def spy(src: object, intents: object, cfg: object, intervals: object) -> object:
        seen.append(cfg)
        return real_sim(src, intents, cfg, intervals)  # type: ignore[arg-type]

    monkeypatch.setattr(part_a_runner, "simulate", spy)
    run_part_a([trade], one_source(both_sides_full_book()), config=OPTIMISTIC)
    assert seen == [OPTIMISTIC]


def test_simulate_called_once_per_trade(monkeypatch: pytest.MonkeyPatch) -> None:
    trades = [
        make_trade(
            trade_id=f"t{i}",
            entry_side=Side.BUY,
            entry_submit=B,
            exit_submit=B + HOLD,
            entry_real_dbn=ASK_PX,
            exit_real_dbn=BID_PX,
        )
        for i in range(3)
    ]
    calls: list[int] = [0]
    real_sim = part_a_runner.simulate

    def spy(*a: object) -> object:
        calls[0] += 1
        return real_sim(*a)  # type: ignore[arg-type]

    monkeypatch.setattr(part_a_runner, "simulate", spy)
    run_part_a(trades, one_source(both_sides_full_book()))
    assert calls[0] == 3


# --------------------------------------------------------------------------- #
# empty trades / duplicate trade_id / non-leg_unfilled unresolved miss
# --------------------------------------------------------------------------- #


def test_empty_trades_returns_n0_fail_without_calling_source_for() -> None:
    called: list[int] = [0]

    def source_for(_t: ReconstructedTrade) -> ListSource:
        called[0] += 1
        return ListSource([])

    result = run_part_a([], source_for)
    assert result.stats.n == 0 and result.verdict == "FAIL"
    assert called[0] == 0


def test_duplicate_trade_id_raises() -> None:
    t = make_trade(
        trade_id="dup",
        entry_side=Side.BUY,
        entry_submit=B,
        exit_submit=B + HOLD,
        entry_real_dbn=ASK_PX,
        exit_real_dbn=BID_PX,
    )
    with pytest.raises(PartAError, match="duplicate trade_id"):
        run_part_a([t, t], one_source(both_sides_full_book()))


# --------------------------------------------------------------------------- #
# broker_fill fidelity path through the runner
# --------------------------------------------------------------------------- #


def test_broker_fill_only_sample_warns_and_grades() -> None:
    trade = make_trade(
        trade_id="t1",
        entry_side=Side.BUY,
        entry_submit=B,
        exit_submit=B + HOLD,
        entry_real_dbn=ASK_PX,
        exit_real_dbn=BID_PX,
    )
    trade = ReconstructedTrade(
        trade_id=trade.trade_id,
        intents=trade.intents,
        real_fills=tuple(
            RealFill(
                order_id=rf.order_id,
                leg=rf.leg,
                side=rf.side,
                size=rf.size,
                price_dbn=rf.price_dbn,
                ts_ns=rf.ts_ns,
                fidelity="broker_fill",
            )
            for rf in trade.real_fills
        ),
        fidelity="broker_fill",
    )
    result = run_part_a([trade], one_source(both_sides_full_book()))
    assert result.broker_fill_stats.n == 2
    assert result.warning is None  # broker_fill subset non-empty, agrees on FAIL


# --------------------------------------------------------------------------- #
# entry-leg unfilled miss (not just exit)
# --------------------------------------------------------------------------- #


def test_entry_leg_unfilled_miss_resolved_from_touch() -> None:
    # BUY entry, but the ask is gone by the time the sim entry arrives
    # (arrival = B + latency); the ask was still there at the real entry ts (B),
    # so the entry miss is priced from that touch + 1 tick adverse slip.
    trade = make_trade(
        trade_id="t1",
        entry_side=Side.BUY,
        entry_submit=B,
        exit_submit=B + HOLD,
        entry_real_dbn=ASK_PX,
        exit_real_dbn=BID_PX,
    )
    events = [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 100, ts=B, seq=1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 100, ts=B, seq=2),
        # ask pulled just after the real entry ts but before the sim arrival
        be(MboAction.CANCEL, MboSide.ASK, 2, ASK_PX, 100, ts=B + 1_000, seq=3),
    ]
    result = run_part_a([trade], one_source(events))
    entry_err = result.errors[0]
    assert entry_err.leg is Leg.ENTRY
    assert entry_err.miss_reason == "leg_unfilled"
    assert entry_err.sim_vwap_dbn == ASK_PX + MNQ_TICK_DBN
    assert all(e.signed_error_ticks is not None for e in result.errors)
