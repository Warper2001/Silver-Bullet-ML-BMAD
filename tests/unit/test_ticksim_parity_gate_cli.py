"""Unit tests for ``src/ticksim/parity/gate_cli.py`` (prereg §A8.2 / spine AD-26).

Every case drives :func:`run_parity_gate` with hand-built
:class:`~src.ticksim.parity.part_a.ReconstructedTrade`\\ s and in-memory
:class:`~src.ticksim.events.BookEventSource`\\ s -- real ``run_part_a`` /
``generate_synthetic_orders`` / ``run_part_b`` / ``preflight_integrity`` /
``gate.evaluate`` / ``gate.build_amendment_stub``, no ``.dbn.zst`` fixture.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator

import pytest

from src.ticksim.config import MNQ_TICK_DBN, OPTIMISTIC, PART_A_MIN_N
from src.ticksim.events import BookEvent, MboAction, MboSide
from src.ticksim.orders import IntentAction, Leg, OrderIntent, OrderKind, Side
from src.ticksim.parity import gate_cli
from src.ticksim.parity.gate_cli import (
    GateCliError,
    GateRun,
    WindowSpec,
    _trader_of,
    run_parity_gate,
)
from src.ticksim.parity.part_a import ReconstructedTrade, RealFill

IID = 42004800
TICK = MNQ_TICK_DBN
P = 20_000_000_000_000
BID_PX = P - TICK
ASK_PX = P + TICK
B = 1_700_000_000 * 1_000_000_000
W = 20_000_000_000  # 20 s -- wider than PART_B_MIN_ORDERS ns
HOLD = 10_000_000_000  # 10 s hold, mid-window


class ListSource:
    """Re-iterable in-memory :class:`BookEventSource` (spine AD-18)."""

    class_rank = 0

    def __init__(self, events: list[BookEvent]) -> None:
        self._events = list(events)

    def __iter__(self) -> Iterator[BookEvent]:
        return iter(self._events)


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


def clean_events(base: int) -> list[BookEvent]:
    """A deep two-sided book + one of each A/C/M/T/F, monotonic, non-crossed --
    serves Part A fills, the 1000-order Part B draw, and an OK integrity pass."""
    return [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 1_000_000, base - 3, 1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 1_000_000, base - 2, 2),
        be(MboAction.ADD, MboSide.BID, 3, BID_PX - 10 * TICK, 5, base - 1, 3),
        be(MboAction.MODIFY, MboSide.BID, 3, BID_PX - 10 * TICK, 3, base + 1, 4),
        be(MboAction.TRADE, MboSide.NONE, 0, BID_PX, 1, base + 2, 5),
        be(MboAction.FILL, MboSide.BID, 1, BID_PX, 1, base + 3, 6),
        be(MboAction.CANCEL, MboSide.BID, 3, BID_PX - 10 * TICK, 3, base + 4, 7),
    ]


def flagged_events(base: int) -> list[BookEvent]:
    """``clean_events`` minus the TRADE and FILL -> missing action classes T/F
    -> a FLAGGED integrity report (Part A / Part B still fine)."""
    return [ev for ev in clean_events(base) if str(ev.action) not in ("T", "F")]


def make_trade(
    *,
    trade_id: str,
    entry_submit: int,
    exit_submit: int,
    entry_real_dbn: int = ASK_PX,
    exit_real_dbn: int = BID_PX,
    entry_side: Side = Side.BUY,
) -> ReconstructedTrade:
    exit_side = Side.SELL if entry_side is Side.BUY else Side.BUY
    oco = f"{trade_id}-oco"
    e_oid, x_oid = f"{trade_id}-e", f"{trade_id}-x"
    entry_i = OrderIntent(
        action=IntentAction.SUBMIT,
        order_id=e_oid,
        trade_id=trade_id,
        leg=Leg.ENTRY,
        kind=OrderKind.MARKETABLE,
        side=entry_side,
        size=1,
        submit_ts_ns=entry_submit,
        oco_group_id=oco,
    )
    exit_i = OrderIntent(
        action=IntentAction.SUBMIT,
        order_id=x_oid,
        trade_id=trade_id,
        leg=Leg.EXIT,
        kind=OrderKind.MARKETABLE,
        side=exit_side,
        size=1,
        submit_ts_ns=exit_submit,
        oco_group_id=oco,
    )
    real_fills = (
        RealFill(
            order_id=e_oid,
            leg=Leg.ENTRY,
            side=entry_side,
            size=1,
            price_dbn=entry_real_dbn,
            ts_ns=entry_submit,
            fidelity="bar_reconstructed",
        ),
        RealFill(
            order_id=x_oid,
            leg=Leg.EXIT,
            side=exit_side,
            size=1,
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


def passing_trades(base: int, *, prefix: str = "yank") -> list[ReconstructedTrade]:
    """>= PART_A_MIN_N/2 trades whose sim fills equal their real fills -> Part A
    PASS (MAE / p90 / bias all 0, N floor cleared)."""
    n_trades = (PART_A_MIN_N + 1) // 2 + 1
    return [
        make_trade(
            trade_id=f"{prefix}-{i}",
            entry_submit=base + 1_000_000 + i * 1_000,
            exit_submit=base + HOLD,
        )
        for i in range(n_trades)
    ]


def one_window(base: int = B) -> dict[str, WindowSpec]:
    return {"w0": WindowSpec(lo_ns=base, hi_ns=base + W)}


def source_of(mapping: dict[str, list[BookEvent]]) -> Callable[[str], ListSource]:
    sources = {key: ListSource(events) for key, events in mapping.items()}
    return lambda key: sources[key]


# --------------------------------------------------------------------------- #
# clean PASS
# --------------------------------------------------------------------------- #


def test_clean_pass_gate_run() -> None:
    run = run_parity_gate(
        passing_trades(B),
        one_window(),
        "w0",
        source_of({"w0": clean_events(B)}),
        synthetic_seed=0,
        synthetic_n=1000,
        amendment_number=7,
        cycle_number=1,
    )
    assert isinstance(run, GateRun)
    assert run.part_a.verdict == "PASS", run.part_a.reason
    assert run.part_b.verdict == "PASS", run.part_b.reason
    assert run.verdict.verdict == "PASS"
    assert run.integrity_flagged is False
    assert len(run.integrity_reports) == 1
    assert run.integrity_reports[0][0] == "w0"
    assert run.integrity_reports[0][1].verdict == "OK"
    # acceptance: frozen SHA line + the verbatim Part B coverage note
    assert re.search(r"simulator commit: [0-9a-f]{40}", run.stub)
    from src.ticksim.parity.part_b import PART_B_COVERAGE_NOTE

    assert PART_B_COVERAGE_NOTE in run.stub


# --------------------------------------------------------------------------- #
# Part A FAIL
# --------------------------------------------------------------------------- #


def test_part_a_fail_miscalibrated() -> None:
    bad = [
        make_trade(
            trade_id=f"yank-{i}",
            entry_submit=B + 1_000_000 + i,
            exit_submit=B + HOLD,
            entry_real_dbn=ASK_PX - 8 * TICK,  # sim pays 8 ticks worse
        )
        for i in range((PART_A_MIN_N + 1) // 2 + 1)
    ]
    run = run_parity_gate(
        bad,
        one_window(),
        "w0",
        source_of({"w0": clean_events(B)}),
        synthetic_seed=0,
        synthetic_n=1000,
        amendment_number=1,
        cycle_number=1,
    )
    assert run.part_a.verdict == "FAIL"
    assert run.verdict.verdict == "FAIL"
    assert "miscalibrated" in run.verdict.reason
    assert "**FAIL**" in run.stub


# --------------------------------------------------------------------------- #
# Part B FAIL (structurally broken)
# --------------------------------------------------------------------------- #


def test_part_b_fail_structurally_broken() -> None:
    run = run_parity_gate(
        passing_trades(B),
        one_window(),
        "w0",
        source_of({"w0": clean_events(B)}),
        synthetic_seed=0,
        synthetic_n=5,  # below PART_B_MIN_ORDERS -> Part B FAIL
        amendment_number=1,
        cycle_number=1,
    )
    assert run.part_a.verdict == "PASS"
    assert run.part_b.verdict == "FAIL"
    assert run.verdict.verdict == "FAIL"
    assert "structurally broken" in run.verdict.reason


# --------------------------------------------------------------------------- #
# integrity FLAGGED -> flag set, verdict unchanged (CHECKPOINT 1a)
# --------------------------------------------------------------------------- #


def test_integrity_flagged_does_not_change_verdict() -> None:
    run = run_parity_gate(
        passing_trades(B),
        one_window(),
        "w0",
        source_of({"w0": flagged_events(B)}),
        synthetic_seed=0,
        synthetic_n=1000,
        amendment_number=3,
        cycle_number=2,
    )
    assert run.part_a.verdict == "PASS"
    assert run.part_b.verdict == "PASS"
    assert run.verdict.verdict == "PASS"  # unchanged
    assert run.integrity_flagged is True
    assert run.integrity_reports[0][1].verdict == "FLAGGED"
    # the stub's integrity section names the flagged window(s) and says the
    # verdict is unchanged -- it must NOT narrate a CLI exit code (this is a
    # library artifact; a FAIL+FLAGGED run rides exit 1, not 3).
    assert "integrity FLAGGED on window(s) w0" in run.stub
    assert "the parity verdict is unchanged per AD-26" in run.stub
    assert "exits 3" not in run.stub and "exit 3" not in run.stub


# --------------------------------------------------------------------------- #
# integrity string joins one labelled block per distinct window
# --------------------------------------------------------------------------- #


def test_integrity_string_joins_per_window_subheadings() -> None:
    windows = {
        "wA": WindowSpec(lo_ns=B, hi_ns=B + W),
        "wB": WindowSpec(lo_ns=B + W, hi_ns=B + 2 * W),
    }
    run = run_parity_gate(
        passing_trades(B),  # all in wA
        windows,
        "wB",  # synthetic over wB -> both windows touched
        source_of({"wA": clean_events(B), "wB": clean_events(B + W)}),
        synthetic_seed=0,
        synthetic_n=1000,
        amendment_number=1,
        cycle_number=1,
    )
    assert [key for key, _ in run.integrity_reports] == ["wA", "wB"]
    assert "window wA: integrity: OK" in run.stub
    assert "window wB: integrity: OK" in run.stub
    # flattened, not a raw multi-line block (build_amendment_stub rejects those)
    assert "\n### window" not in run.stub


# --------------------------------------------------------------------------- #
# _trader_of
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("trade_id", "expected"),
    [
        ("mimnb-abc123", "trader-mim-nb"),  # CSV reconstruct_mim_nb prefix
        ("mimnb-", "trader-mim-nb"),
        ("trader-mim-nb-2900", "trader-mim-nb"),  # DB-fallback prefix
        ("trader-yank-2900", "trader-yank"),
        ("yank-17", "trader-yank"),
    ],
)
def test_trader_of(trade_id: str, expected: str) -> None:
    trade = make_trade(trade_id=trade_id, entry_submit=B, exit_submit=B + HOLD)
    assert _trader_of(trade) == expected


# --------------------------------------------------------------------------- #
# build_amendment_stub called exactly once (spy)
# --------------------------------------------------------------------------- #


def test_build_amendment_stub_called_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    real = gate_cli.build_amendment_stub

    def spy(*a: object, **k: object) -> str:
        calls.append(1)
        return real(*a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(gate_cli, "build_amendment_stub", spy)
    run_parity_gate(
        passing_trades(B),
        one_window(),
        "w0",
        source_of({"w0": clean_events(B)}),
        synthetic_seed=0,
        synthetic_n=1000,
        amendment_number=1,
        cycle_number=1,
    )
    assert calls == [1]


# --------------------------------------------------------------------------- #
# window routing errors -> GateCliError
# --------------------------------------------------------------------------- #


def test_trade_matching_zero_windows_raises() -> None:
    trade = make_trade(trade_id="yank-0", entry_submit=B - 5_000, exit_submit=B + HOLD)
    with pytest.raises(GateCliError, match="contained by 0 of 1 windows"):
        run_parity_gate(
            [trade],
            one_window(),
            "w0",
            source_of({"w0": clean_events(B)}),
            synthetic_seed=0,
            synthetic_n=1000,
            amendment_number=1,
            cycle_number=1,
        )


def test_trade_whose_span_pokes_past_hi_ns_raises() -> None:
    # entry is inside [lo, hi) but the exit leg / real fill ts is past hi_ns --
    # _ClippedSource would truncate the book under run_part_a's exit pricing.
    trade = make_trade(
        trade_id="yank-0", entry_submit=B + 1_000, exit_submit=B + W + 5_000
    )
    with pytest.raises(GateCliError, match="contained by 0 of 1 windows"):
        run_parity_gate(
            [trade],
            one_window(),
            "w0",
            source_of({"w0": clean_events(B)}),
            synthetic_seed=0,
            synthetic_n=1000,
            amendment_number=1,
            cycle_number=1,
        )


def test_trade_matching_two_windows_raises() -> None:
    windows = {
        "wA": WindowSpec(lo_ns=B, hi_ns=B + 2 * W),
        "wB": WindowSpec(lo_ns=B, hi_ns=B + W),  # overlaps wA, still contains span
    }
    trade = make_trade(trade_id="yank-0", entry_submit=B + 10, exit_submit=B + HOLD)
    with pytest.raises(GateCliError, match="contained by 2 of 2 windows"):
        run_parity_gate(
            [trade],
            windows,
            "wA",
            source_of({"wA": clean_events(B), "wB": clean_events(B)}),
            synthetic_seed=0,
            synthetic_n=1000,
            amendment_number=1,
            cycle_number=1,
        )


def test_synthetic_window_absent_raises() -> None:
    with pytest.raises(GateCliError, match="not a key in windows"):
        run_parity_gate(
            [],
            one_window(),
            "missing",
            source_of({"w0": clean_events(B)}),
            synthetic_seed=0,
            synthetic_n=1000,
            amendment_number=1,
            cycle_number=1,
        )


def test_non_positive_amendment_number_raises_before_any_compute() -> None:
    calls: list[int] = []

    def spy(_key: str) -> ListSource:
        calls.append(1)
        return ListSource(clean_events(B))

    with pytest.raises(GateCliError, match="amendment_number must be > 0"):
        run_parity_gate(
            passing_trades(B),
            one_window(),
            "w0",
            spy,
            synthetic_seed=0,
            synthetic_n=1000,
            amendment_number=0,
            cycle_number=1,
        )
    assert calls == []  # rejected before Part A touched a source


# --------------------------------------------------------------------------- #
# --config optimistic forwarded to both runners
# --------------------------------------------------------------------------- #


def test_config_forwarded_to_both_runners(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []
    real_a, real_b = gate_cli.run_part_a, gate_cli.run_part_b

    def spy_a(trades: object, src: object, *, config: object = None) -> object:
        seen.append(("part_a", config))
        return real_a(trades, src, config=config)  # type: ignore[arg-type]

    def spy_b(intents: object, src: object, *, config: object = None) -> object:
        seen.append(("part_b", config))
        return real_b(intents, src, config=config)  # type: ignore[arg-type]

    monkeypatch.setattr(gate_cli, "run_part_a", spy_a)
    monkeypatch.setattr(gate_cli, "run_part_b", spy_b)
    run_parity_gate(
        passing_trades(B),
        one_window(),
        "w0",
        source_of({"w0": clean_events(B)}),
        synthetic_seed=0,
        synthetic_n=1000,
        amendment_number=1,
        cycle_number=1,
        config=OPTIMISTIC,
    )
    assert ("part_a", OPTIMISTIC) in seen
    assert ("part_b", OPTIMISTIC) in seen
