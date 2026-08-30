"""Unit tests for ``src.ticksim.report`` -- the AD-14 three-way P&L report.

One test per row of the spec's I/O & Edge-Case Matrix plus the Acceptance
Criteria and the review-1 hardening set. ``OrderOutcome`` fixtures are built by
:func:`_oc`; the manifest is the minimal ``Manifest.to_dict()`` slice
``report.py`` reads (spine AD-24).
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from src.ticksim.config import MNQ_TICK_DBN
from src.ticksim.orders import Fill, Leg, OrderKind, OrderOutcome, Side, TerminalState
from src.ticksim.report import (
    TICK_VALUE_CENTS,
    ModelPnL,
    OpenPosition,
    ReportError,
    RoundTrip,
    ThreeWayReport,
    build_report,
)

P = 100_000_000_000  # 100.00 index points, DBN 1e-9 fixed-point
TICK = MNQ_TICK_DBN  # 250_000_000 dbn == $0.50 == 50 cents
FEE_RT = 72 + 58  # per-contract round-turn fee, cents


def _manifest(
    exch_reg: int = 72, commission: int = 58, queue_model: str = "back_of_queue"
) -> dict:
    return {
        "config": {
            "exch_reg_fee_usd_cents": exch_reg,
            "commission_usd_cents": commission,
            "queue_model": queue_model,
        }
    }


_PRIMARY_M = _manifest()
_OPT_M = _manifest(queue_model="time_priority")


def _oc(
    trade_id: str,
    leg: Leg,
    side: Side,
    order_id: str,
    fills: list[tuple[int, int]],
    *,
    ts: int = 2_000,
    terminal_state: TerminalState = TerminalState.FILLED,
    adverse_selection: bool = False,
    kind: OrderKind = OrderKind.PASSIVE_LIMIT,
) -> OrderOutcome:
    """An ``OrderOutcome``; ``fills`` is a list of ``(px_dbn, size)``; all fills
    stamped at ``ts + i``."""
    return OrderOutcome(
        trade_id=trade_id,
        leg=leg,
        order_id=order_id,
        kind=kind,
        side=side,
        submit_ts_ns=1_000,
        arrival_ts_ns=1_000,
        terminal_state=terminal_state,
        fills=tuple(
            Fill(px_dbn=px, size=sz, ts_ns=ts + i) for i, (px, sz) in enumerate(fills)
        ),
        adverse_selection=adverse_selection,
    )


def _long_rt(
    trade_id: str, entry_px: int, exit_px: int, size: int, *, entry_ts: int = 2_000
) -> list[OrderOutcome]:
    return [
        _oc(
            trade_id,
            Leg.ENTRY,
            Side.BUY,
            f"{trade_id}-e",
            [(entry_px, size)],
            ts=entry_ts,
        ),
        _oc(
            trade_id,
            Leg.EXIT,
            Side.SELL,
            f"{trade_id}-x",
            [(exit_px, size)],
            ts=entry_ts + 500,
        ),
    ]


def _report(primary: list[OrderOutcome], optimistic: list[OrderOutcome] | None = None):
    return build_report(primary, _PRIMARY_M, optimistic or primary, _OPT_M)


# --- clean round trips + arithmetic ---------------------------------


def test_clean_long_round_trip() -> None:
    r = _report(_long_rt("T", P, P + 2 * TICK, 5))
    (rt,) = r.round_trips
    assert rt.direction == 1
    assert rt.matched_size == 5
    assert rt.net_primary_cents == (2 * TICK_VALUE_CENTS) * 5 - FEE_RT * 5
    assert rt.net_stressed_cents == rt.net_primary_cents - 100 * 5
    assert r.primary.net_cents == (rt.net_primary_cents,)
    assert r.stressed.net_cents == (rt.net_stressed_cents,)


def test_short_round_trip_mirrors_long() -> None:
    short = [
        _oc("T", Leg.ENTRY, Side.SELL, "T-e", [(P + 2 * TICK, 5)]),
        _oc("T", Leg.EXIT, Side.BUY, "T-x", [(P, 5)], ts=2_500),
    ]
    r = _report(short)
    (rt,) = r.round_trips
    assert rt.direction == -1
    assert rt.net_primary_cents == (2 * TICK_VALUE_CENTS) * 5 - FEE_RT * 5


def test_bracket_tp_fills_sl_cancelled() -> None:
    outcomes = [
        _oc("T", Leg.ENTRY, Side.BUY, "T-e", [(P, 3)]),
        _oc("T", Leg.EXIT, Side.SELL, "T-tp", [(P + 4 * TICK, 3)], ts=2_500),
        _oc(
            "T", Leg.EXIT, Side.SELL, "T-sl", [], terminal_state=TerminalState.CANCELLED
        ),
    ]
    r = _report(outcomes)
    (rt,) = r.round_trips
    assert rt.matched_size == 3
    assert rt.net_primary_cents == (4 * TICK_VALUE_CENTS) * 3 - FEE_RT * 3


@pytest.mark.parametrize("size", [1, 3, 5])
def test_model_b_is_model_a_minus_two_ticks(size: int) -> None:
    r = _report(_long_rt("T", P, P + 6 * TICK, size))
    (rt,) = r.round_trips
    assert rt.net_stressed_cents == rt.net_primary_cents - 2 * TICK_VALUE_CENTS * size


def test_round_trips_are_chronological_not_trade_id_order() -> None:
    # T1 entry later than T2 entry -> report order is [T2, T1] despite id order
    outcomes = _long_rt("T1", P, P + TICK, 1, entry_ts=9_000) + _long_rt(
        "T2", P, P + 2 * TICK, 1, entry_ts=3_000
    )
    r = _report(outcomes)
    assert [rt.trade_id for rt in r.round_trips] == ["T2", "T1"]
    assert r.primary.net_cents == (
        r.round_trips[0].net_primary_cents,
        r.round_trips[1].net_primary_cents,
    )


# --- optimistic model + population -----------------------------------


def test_optimistic_uses_optimistic_fills_and_fees() -> None:
    primary = _long_rt("T", P, P + 2 * TICK, 5)
    optimistic = _long_rt("T", P, P + 4 * TICK, 5)  # better exit
    r = build_report(
        primary, _PRIMARY_M, optimistic, _manifest(40, 30, "time_priority")
    )
    (rt,) = r.round_trips
    assert rt.net_optimistic_cents == (4 * TICK_VALUE_CENTS) * 5 - (40 + 30) * 5
    assert r.optimistic.net_cents == (rt.net_optimistic_cents,)


def test_optimistic_completes_a_primary_open_trade() -> None:
    primary = [
        _oc("T", Leg.ENTRY, Side.BUY, "T-e", [(P, 5)]),
        _oc("T", Leg.EXIT, Side.SELL, "T-x", [], terminal_state=TerminalState.EXPIRED),
    ]
    optimistic = _long_rt("T", P, P + 2 * TICK, 5)
    r = build_report(primary, _PRIMARY_M, optimistic, _OPT_M)
    assert [op.trade_id for op in r.incomplete] == ["T"]
    assert r.optimistic_only_completed == ("T",)
    assert r.primary.n == 0
    # (c) is the both-completed subset -> an optimistic-only trade is NOT in it
    assert r.optimistic.n == 0
    assert r.round_trips == ()  # primary didn't complete it


def test_optimistic_leaves_a_primary_round_trip_open() -> None:
    primary = _long_rt("T", P, P + 2 * TICK, 5)
    optimistic = [
        _oc("T", Leg.ENTRY, Side.BUY, "T-e", [(P, 5)]),
        _oc("T", Leg.EXIT, Side.SELL, "T-x", [], terminal_state=TerminalState.EXPIRED),
    ]
    r = build_report(primary, _PRIMARY_M, optimistic, _OPT_M)
    (rt,) = r.round_trips
    assert rt.net_optimistic_cents is None
    assert r.optimistic.n == 0
    assert r.primary.n == 1


# --- open / partial -------------------------------------------------


def test_incomplete_open_position_carries_exposure() -> None:
    outcomes = [
        _oc("T", Leg.ENTRY, Side.BUY, "T-e", [(P, 5)], ts=7_000),
        _oc("T", Leg.EXIT, Side.SELL, "T-x", [], terminal_state=TerminalState.EXPIRED),
    ]
    r = _report(outcomes)
    (op,) = r.incomplete
    assert op == OpenPosition("T", 5, P, 7_000)
    assert r.primary.n == 0


def test_partially_closed() -> None:
    outcomes = [
        _oc("T", Leg.ENTRY, Side.BUY, "T-e", [(P, 5)]),
        _oc("T", Leg.EXIT, Side.SELL, "T-x", [(P + 2 * TICK, 3)], ts=2_500),
    ]
    r = _report(outcomes)
    (rt,) = r.round_trips
    assert rt.matched_size == 3
    assert rt.net_primary_cents == (2 * TICK_VALUE_CENTS) * 3 - FEE_RT * 3
    assert r.partially_closed == (("T", 2),)


def test_neither_leg_fills_skipped_silently() -> None:
    skip = [
        _oc("T", Leg.ENTRY, Side.BUY, "T-e", [], terminal_state=TerminalState.EXPIRED),
        _oc(
            "T", Leg.EXIT, Side.SELL, "T-x", [], terminal_state=TerminalState.CANCELLED
        ),
    ]
    r = _report(skip)
    assert r.round_trips == () and r.incomplete == () and r.primary.n == 0


# --- errors --------------------------------------------------------


def test_exit_size_gt_entry_size_raises() -> None:
    with pytest.raises(ReportError):
        _report(
            [
                _oc("T", Leg.ENTRY, Side.BUY, "T-e", [(P, 3)]),
                _oc("T", Leg.EXIT, Side.SELL, "T-x", [(P + TICK, 5)], ts=2_500),
            ]
        )


def test_exit_fill_with_no_entry_fill_raises() -> None:
    with pytest.raises(ReportError):
        _report(
            [
                _oc(
                    "T",
                    Leg.ENTRY,
                    Side.BUY,
                    "T-e",
                    [],
                    terminal_state=TerminalState.EXPIRED,
                ),
                _oc("T", Leg.EXIT, Side.SELL, "T-x", [(P + TICK, 2)], ts=2_500),
            ]
        )


def test_mixed_entry_sides_raises() -> None:
    with pytest.raises(ReportError):
        _report(
            [
                _oc("T", Leg.ENTRY, Side.BUY, "T-e1", [(P, 2)]),
                _oc("T", Leg.ENTRY, Side.SELL, "T-e2", [(P, 2)]),
                _oc("T", Leg.EXIT, Side.SELL, "T-x", [(P + TICK, 4)], ts=2_500),
            ]
        )


def test_exit_same_side_as_entry_raises() -> None:
    with pytest.raises(ReportError):
        _report(
            [
                _oc("T", Leg.ENTRY, Side.BUY, "T-e", [(P, 3)]),
                _oc("T", Leg.EXIT, Side.BUY, "T-x", [(P + TICK, 3)], ts=2_500),
            ]
        )


def test_mismatched_trade_id_sets_raises() -> None:
    primary = _long_rt("T1", P, P + TICK, 1) + _long_rt("T2", P, P + TICK, 1)
    optimistic = _long_rt("T1", P, P + TICK, 1) + _long_rt("T3", P, P + TICK, 1)
    with pytest.raises(ReportError):
        build_report(primary, _PRIMARY_M, optimistic, _OPT_M)


def test_swapped_or_wrong_manifests_raises() -> None:
    outcomes = _long_rt("T", P, P + TICK, 1)
    with pytest.raises(ReportError):
        build_report(outcomes, _PRIMARY_M, outcomes, _PRIMARY_M)  # both back_of_queue


def test_manifest_missing_fee_key_raises() -> None:
    bad = {"config": {"exch_reg_fee_usd_cents": 72, "queue_model": "time_priority"}}
    with pytest.raises(ReportError):
        build_report(
            _long_rt("T", P, P + TICK, 1),
            _PRIMARY_M,
            _long_rt("T", P, P + TICK, 1),
            bad,
        )


def test_negative_fee_raises() -> None:
    bad = _manifest(exch_reg=-1, queue_model="time_priority")
    with pytest.raises(ReportError):
        build_report(
            _long_rt("T", P, P + TICK, 1),
            _PRIMARY_M,
            _long_rt("T", P, P + TICK, 1),
            bad,
        )


def test_duplicate_order_id_raises() -> None:
    dup = _long_rt("T", P, P + TICK, 1)
    dup.append(_oc("T", Leg.EXIT, Side.SELL, "T-x", [(P, 1)], ts=3_000))  # reused id
    with pytest.raises(ReportError):
        _report(dup)


# --- adverse / PF / determinism / to_dict --------------------------


def test_adverse_from_filled_leg_only() -> None:
    outcomes = [
        _oc("A", Leg.ENTRY, Side.BUY, "A-e", [(P, 1)], adverse_selection=True),
        _oc("A", Leg.EXIT, Side.SELL, "A-x", [(P + TICK, 1)], ts=2_500),
    ]
    # a CANCELLED (no-fill) leg flagged adverse must NOT taint the round trip
    clean = [
        _oc("C", Leg.ENTRY, Side.BUY, "C-e", [(P, 1)]),
        _oc("C", Leg.EXIT, Side.SELL, "C-tp", [(P + TICK, 1)], ts=2_500),
        _oc(
            "C",
            Leg.EXIT,
            Side.SELL,
            "C-sl",
            [],
            terminal_state=TerminalState.CANCELLED,
            adverse_selection=True,
        ),
    ]
    r = _report(outcomes + clean)
    by = {rt.trade_id: rt for rt in r.round_trips}
    assert by["A"].adverse is True
    assert by["C"].adverse is False


def test_profit_factor_inf_when_no_losers() -> None:
    r = _report(_long_rt("T1", P, P + 4 * TICK, 1) + _long_rt("T2", P, P + 4 * TICK, 1))
    assert r.primary.profit_factor == float("inf")


def test_empty_study() -> None:
    r = build_report([], _PRIMARY_M, [], _OPT_M)
    assert r.round_trips == ()
    assert r.primary.n == 0
    assert r.primary.mean_net_cents is None
    assert r.primary.profit_factor is None


def test_to_dict_json_serializable_and_populated() -> None:
    outcomes = (
        _long_rt("T1", P, P + 2 * TICK, 5)
        + [
            _oc("T2", Leg.ENTRY, Side.BUY, "T2-e", [(P, 5)]),
            _oc("T2", Leg.EXIT, Side.SELL, "T2-x", [(P + TICK, 3)], ts=2_500),
        ]
        + [
            _oc("T3", Leg.ENTRY, Side.BUY, "T3-e", [(P, 2)]),
            _oc(
                "T3",
                Leg.EXIT,
                Side.SELL,
                "T3-x",
                [],
                terminal_state=TerminalState.EXPIRED,
            ),
        ]
    )
    r = _report(outcomes)
    d = r.to_dict()
    dumped = json.loads(json.dumps(d))
    assert dumped["partially_closed"] == [["T2", 2]]
    assert dumped["config_primary"]["queue_model"] == "back_of_queue"
    assert dumped["incomplete"][0]["trade_id"] == "T3"
    assert (
        dumped["round_trips"][0]["net_primary_cents"]
        == r.round_trips[0].net_primary_cents
    )


def test_does_not_mutate_inputs() -> None:
    primary = _long_rt("T", P, P + TICK, 1)
    snapshot = list(primary)
    m = _manifest()
    build_report(primary, m, list(primary), _OPT_M)
    assert primary == snapshot
    assert m == _manifest()  # unchanged


def test_dataclasses_are_frozen() -> None:
    r = _report(_long_rt("T", P, P + TICK, 1))
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.primary.n = 99  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.round_trips[0].adverse = True  # type: ignore[misc]


def test_win_loss_boundary_with_real_fees() -> None:
    # gross = 1 tick = 50c on 1 contract; fees = 130c -> net -80c -> a loss
    r = _report(_long_rt("T", P, P + TICK, 1))
    (rt,) = r.round_trips
    assert rt.net_primary_cents == 50 - 130 == -80
    assert r.primary.losses == 1 and r.primary.wins == 0


def test_exports() -> None:
    assert issubclass(ReportError, Exception)
    for cls in (RoundTrip, OpenPosition, ModelPnL, ThreeWayReport):
        assert dataclasses.is_dataclass(cls)
