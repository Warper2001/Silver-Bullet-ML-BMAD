"""Unit tests for ``src/ticksim/parity/part_a.py`` (prereg §A8.2 Part A core)."""

from __future__ import annotations

import json

import pytest

from src.ticksim.config import (
    MNQ_TICK_DBN,
    PARITY_MAE_MAX_TICKS,
    PARITY_P90_MAX_TICKS,
    PARITY_SIGNED_BIAS_MAX_TICKS,
    PART_A_MIN_N,
)
from src.ticksim.orders import (
    Fill,
    Leg,
    OrderKind,
    OrderOutcome,
    Side,
    TerminalState,
)
from src.ticksim.parity.part_a import (
    FillError,
    PartAError,
    ReconstructedTrade,
    RealFill,
    _build_mim_trade,
    _parse_ts_ns,
    _PendingLeg,
    aggregate,
    compare_fills,
    reconstruct_mim_nb,
    reconstruct_trades_db_row,
)

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

T0 = "2026-06-17T14:00:00+00:00"
T1 = "2026-06-17T14:00:01+00:00"
T2 = "2026-06-17T14:00:02+00:00"
T3 = "2026-06-17T14:30:00+00:00"
T4 = "2026-06-17T14:30:01+00:00"
T5 = "2026-06-17T14:30:02+00:00"
T6 = "2026-06-17T14:30:03+00:00"

ENTRY_PX = "20000.25"
EXIT_PX = "20010.50"

TRADE_ID = "mimnb-E1"  # _build_mim_trade uses f"mimnb-{entry.order_id}"


def mrow(
    ts: str,
    event: str,
    order_id: str,
    otype: str,
    *,
    side: str = "0",
    size: str = "1",
    price: str = "",
    outcome: str = "",
) -> dict[str, object]:
    return {
        "ts_utc": ts,
        "event": event,
        "order_id": order_id,
        "otype": otype,
        "side": side,
        "size": size,
        "price": price,
        "outcome": outcome,
        "detail": "",
        "chain": "deadbeef",
    }


def standard_mim_lifecycle() -> list[dict[str, object]]:
    """entry PLACE->FILL @ X, stop PLACE->CANCEL, exit PLACE->FILL @ Y."""
    return [
        mrow(T0, "PLACE", "E1", "2", side="0"),
        mrow(T1, "FILL", "E1", "2", side="0", price=ENTRY_PX),
        mrow(T2, "PLACE", "S1", "4", side="1"),
        mrow(T3, "PLACE", "X1", "2", side="1"),
        mrow(T4, "FILL", "X1", "2", side="1", price=EXIT_PX),
        mrow(T5, "CANCEL", "S1", "4", side="1"),
    ]


def outcome(
    order_id: str,
    leg: Leg,
    side: Side,
    *,
    terminal: TerminalState = TerminalState.FILLED,
    fills: tuple[Fill, ...] = (),
    trade_id: str = TRADE_ID,
) -> OrderOutcome:
    return OrderOutcome(
        trade_id=trade_id,
        leg=leg,
        order_id=order_id,
        kind=OrderKind.MARKETABLE,
        side=side,
        submit_ts_ns=1,
        arrival_ts_ns=2,
        terminal_state=terminal,
        fills=fills,
    )


def fe(signed: float | None, fidelity: str = "broker_fill") -> FillError:
    return FillError(
        trade_id="t",
        order_id="o",
        leg=Leg.ENTRY,
        real_dbn=1_000,
        real_ts_ns=1,
        sim_vwap_dbn=None if signed is None else 1_000,
        signed_error_ticks=signed,
        miss_reason="leg_unfilled" if signed is None else None,
        sim_terminal_state="expired" if signed is None else None,
        fidelity=fidelity,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# reconstruct_mim_nb
# --------------------------------------------------------------------------- #


class TestReconstructMimNb:
    def test_empty(self) -> None:
        assert reconstruct_mim_nb([]) == []

    def test_standard_lifecycle_two_legs(self) -> None:
        trades = reconstruct_mim_nb(standard_mim_lifecycle())
        assert len(trades) == 1
        trade = trades[0]
        assert trade.fidelity == "broker_fill"
        assert len(trade.intents) == 2

        entry, exit_ = trade.intents
        assert entry.leg is Leg.ENTRY
        assert entry.kind is OrderKind.MARKETABLE
        assert entry.side is Side.BUY
        assert entry.limit_px_dbn is None
        assert exit_.leg is Leg.EXIT
        assert exit_.kind is OrderKind.MARKETABLE
        assert exit_.side is Side.SELL

        assert entry.oco_group_id == exit_.oco_group_id
        assert entry.oco_group_id is not None
        assert {i.order_id for i in trade.intents} == {"E1", "X1"}

        assert len(trade.real_fills) == 2
        rf_entry, rf_exit = trade.real_fills
        assert rf_entry.fidelity == "broker_fill"
        assert rf_entry.side is Side.BUY
        assert rf_entry.price_dbn == 20000_250_000_000
        assert rf_exit.price_dbn == 20010_500_000_000
        assert rf_exit.side is Side.SELL

    def test_price_not_tick_snapped_and_exact(self) -> None:
        rows = standard_mim_lifecycle()
        rows[1] = mrow(T1, "FILL", "E1", "2", side="0", price="20000.30")
        trade = reconstruct_mim_nb(rows)[0]
        assert trade.real_fills[0].price_dbn == 20000_300_000_000
        assert trade.real_fills[0].price_dbn % MNQ_TICK_DBN != 0

    def test_entry_cancelled_then_a_full_trade(self) -> None:
        rows = [
            mrow("2026-06-17T13:00:00+00:00", "PLACE", "E0", "2", side="0"),
            mrow("2026-06-17T13:00:05+00:00", "CANCEL", "E0", "2", side="0"),
            *standard_mim_lifecycle(),
        ]
        trades = reconstruct_mim_nb(rows)
        assert len(trades) == 1
        assert trades[0].intents[0].order_id == "E1"

    def test_rejected_and_fail_rows_dropped(self) -> None:
        rows = [
            mrow(T0, "REJECTED", "E9", "2", side="0"),
            mrow(T0, "PLACE", "FAIL", "2", side="0", outcome="REJECTED"),
            mrow(T0, "PLACE", "E8", "2", side="0", outcome="REJECTED"),
            *standard_mim_lifecycle(),
        ]
        assert len(reconstruct_mim_nb(rows)) == 1

    def test_fill_flagged_rejected_is_a_contradiction(self) -> None:
        rows = standard_mim_lifecycle()
        rows[1] = mrow(
            T1, "FILL", "E1", "2", side="0", price=ENTRY_PX, outcome="REJECTED"
        )
        with pytest.raises(PartAError, match="contradictory"):
            reconstruct_mim_nb(rows)

    def test_non_market_entry_raises(self) -> None:
        rows = standard_mim_lifecycle()
        rows[0] = mrow(T0, "PLACE", "E1", "1", side="0")
        with pytest.raises(PartAError, match="non-market"):
            reconstruct_mim_nb(rows)

    def test_protective_stop_fill_is_scored_as_the_exit(self) -> None:
        """A fired stop is a REAL exit fill and must be scored (2026-09-01).

        The original code raised here, on the spec's assumption that the stop
        "never fills in the real ledger". `data/mim_nb/orders.csv` falsifies
        that: 2026-07-29 and 2026-08-28 both carry an otype-4 FILL with a
        `pnl=` detail. Discarding those would silently drop real stop-out
        exits from Part A's sample.
        """
        rows = [
            mrow(T0, "PLACE", "E1", "2", side="0"),
            mrow(T0, "PLACE", "S1", "4", side="1", price="19990.00"),
            mrow(T1, "FILL", "E1", "2", side="0", price=ENTRY_PX),
            mrow(T3, "FILL", "S1", "4", side="1", price="19990.00"),
        ]
        trades = reconstruct_mim_nb(rows)
        assert len(trades) == 1
        exit_fill = trades[0].real_fills[-1]
        assert exit_fill.order_id == "S1"
        assert exit_fill.leg is Leg.EXIT
        assert exit_fill.price_dbn == 19990_000_000_000

    def test_unattributable_stop_fill_is_skipped_not_raised(self) -> None:
        """An otype-4 FILL with no live position (the real 2026-07-29
        `order_id='111'` row, which has no PLACE anywhere) cannot be attributed
        to a trade -- skipped and reported, never guessed at."""
        rows = standard_mim_lifecycle()
        rows.append(mrow(T3, "FILL", "111", "4", side="1", price="19990.00"))
        trades = reconstruct_mim_nb(rows)
        assert len(trades) == 1  # the standard trade survives

    def test_replace_event_raises(self) -> None:
        rows = standard_mim_lifecycle()
        rows.insert(2, mrow(T1, "REPLACE", "E1", "2", side="0"))
        with pytest.raises(PartAError, match="REPLACE"):
            reconstruct_mim_nb(rows)

    def test_modify_event_raises(self) -> None:
        rows = standard_mim_lifecycle()
        rows.insert(2, mrow(T1, "MODIFY", "E1", "2", side="0"))
        with pytest.raises(PartAError, match="MODIFY"):
            reconstruct_mim_nb(rows)

    def test_trailing_mid_trade_is_dropped_not_raised(self) -> None:
        """A ledger ending mid-trade is the bot still holding, or an exit that
        was never written. It has no comparable fill pair, so it is dropped
        with a warning (2026-09-01) -- one open trade at the tail must not cost
        Part A the other 20+ scoreable trades, which is what raising did on the
        real ledger."""
        rows = [
            mrow(T0, "PLACE", "E1", "2", side="0"),
            mrow(T1, "FILL", "E1", "2", side="0", price=ENTRY_PX),
        ]
        assert reconstruct_mim_nb(rows) == []

    def test_unfilled_entry_then_unbracketed_flatten_is_dropped(self) -> None:
        """The real 2026-06-11 shape: entry placed, never filled, then a
        cat-stop flatten that also never fills. Neither leg produced a fill, so
        there is nothing to compare -- dropped, not raised."""
        rows = [
            mrow(T0, "PLACE", "E1", "2", side="0"),
            mrow(T0, "PLACE", "S1", "4", side="1", price="19990.00"),
            mrow(T2, "CANCEL", "S1", "4", side="1"),
            mrow(T3, "PLACE", "X1", "2", side="1"),
        ]
        assert reconstruct_mim_nb(rows) == []

    def test_unclosed_trade_superseded_by_new_bracketed_entry(self) -> None:
        """The real 2026-06-24 -> 06-25 shape: a trade whose exit was never
        logged must NOT swallow the next day's entry as its exit. The bracket
        signal (an entry is followed by its otype-4 stop) separates them."""
        rows = [
            mrow(T0, "PLACE", "E1", "2", side="0"),
            mrow(T0, "PLACE", "S1", "4", side="1", price="19990.00"),
            mrow(T1, "FILL", "E1", "2", side="0", price=ENTRY_PX),
            # next trade, bracketed -- a NEW entry, not E1's exit
            mrow(T3, "PLACE", "E2", "2", side="0"),
            mrow(T3, "PLACE", "S2", "4", side="1", price="19980.00"),
            mrow(T4, "FILL", "E2", "2", side="0", price=ENTRY_PX),
            mrow(T5, "PLACE", "X2", "2", side="1"),
            mrow(T6, "FILL", "X2", "2", side="1", price=EXIT_PX),
        ]
        trades = reconstruct_mim_nb(rows)
        assert [t.trade_id for t in trades] == ["mimnb-E2"]

    def test_fill_without_order_id_raises(self) -> None:
        rows = standard_mim_lifecycle()
        rows[1] = mrow(T1, "FILL", "", "2", side="0", price=ENTRY_PX)
        with pytest.raises(PartAError, match="no order_id"):
            reconstruct_mim_nb(rows)

    def test_non_positive_price_raises(self) -> None:
        rows = standard_mim_lifecycle()
        rows[1] = mrow(T1, "FILL", "E1", "2", side="0", price="-5.0")
        with pytest.raises(PartAError, match="non-finite / non-positive"):
            reconstruct_mim_nb(rows)

    def test_unparseable_price_raises(self) -> None:
        rows = standard_mim_lifecycle()
        rows[4] = mrow(T4, "FILL", "X1", "2", side="1", price="not-a-number")
        with pytest.raises(PartAError, match="unparseable price"):
            reconstruct_mim_nb(rows)

    def test_bad_side_token_raises(self) -> None:
        rows = standard_mim_lifecycle()
        rows[0] = mrow(T0, "PLACE", "E1", "2", side="7")
        with pytest.raises(PartAError, match="bad side token"):
            reconstruct_mim_nb(rows)

    def test_exit_side_not_opposite_raises(self) -> None:
        rows = standard_mim_lifecycle()
        rows[3] = mrow(T3, "PLACE", "X1", "2", side="0")  # same side as entry
        rows[4] = mrow(T4, "FILL", "X1", "2", side="0", price=EXIT_PX)
        with pytest.raises(PartAError, match="same-side round trip"):
            reconstruct_mim_nb(rows)

    def test_unexpected_cancel_while_flat_raises(self) -> None:
        with pytest.raises(PartAError, match="unexpected CANCEL"):
            reconstruct_mim_nb([mrow(T0, "CANCEL", "Z1", "2", side="0")])

    def test_unexpected_fill_while_flat_raises(self) -> None:
        with pytest.raises(PartAError, match="unexpected FILL"):
            reconstruct_mim_nb([mrow(T0, "FILL", "Z1", "2", side="0", price="100.0")])

    def test_second_fill_on_one_leg_raises(self) -> None:
        rows = standard_mim_lifecycle()
        rows.insert(2, mrow(T2, "FILL", "E1", "2", side="0", price=ENTRY_PX))
        with pytest.raises(PartAError, match="unexpected FILL"):
            reconstruct_mim_nb(rows)

    def test_rows_are_timestamp_sorted(self) -> None:
        rows = list(reversed(standard_mim_lifecycle()))
        trades = reconstruct_mim_nb(rows)
        assert len(trades) == 1
        assert trades[0].intents[0].submit_ts_ns < trades[0].intents[1].submit_ts_ns

    def test_same_ts_place_before_fill(self) -> None:
        # PLACE and FILL share a timestamp; the event-rank tie-break must keep
        # PLACE first so the lifecycle does not invert.
        rows = [
            mrow(T1, "FILL", "E1", "2", side="0", price=ENTRY_PX),
            mrow(T1, "PLACE", "E1", "2", side="0"),
            mrow(T3, "PLACE", "X1", "2", side="1"),
            mrow(T3, "FILL", "X1", "2", side="1", price=EXIT_PX),
        ]
        trades = reconstruct_mim_nb(rows)
        assert len(trades) == 1

    def test_int_side_and_size_cells_not_collapsed(self) -> None:
        # a mapping carrying int 0 for side / int 1 for size must not raise a
        # misleading "bad token" error.
        rows = [
            {**mrow(T0, "PLACE", "E1", "2"), "side": 0, "size": 1},
            {**mrow(T1, "FILL", "E1", "2", price=ENTRY_PX), "side": 0, "size": 1},
            {**mrow(T3, "PLACE", "X1", "2"), "side": 1, "size": 1},
            {**mrow(T4, "FILL", "X1", "2", price=EXIT_PX), "side": 1, "size": 1},
        ]
        trade = reconstruct_mim_nb(rows)[0]
        assert trade.intents[0].side is Side.BUY
        assert trade.intents[0].size == 1


# --------------------------------------------------------------------------- #
# _parse_ts_ns normalization
# --------------------------------------------------------------------------- #


class TestParseTs:
    def test_z_suffix_space_and_naive_all_equal(self) -> None:
        a = _parse_ts_ns("2026-06-17T14:00:00Z", field_name="t", row_repr="r")
        b = _parse_ts_ns("2026-06-17 14:00:00+00:00", field_name="t", row_repr="r")
        c = _parse_ts_ns("2026-06-17T14:00:00", field_name="t", row_repr="r")
        assert a == b == c

    def test_fractional_seconds_truncated(self) -> None:
        # 7+ fractional digits are truncated to 6, so these collapse to the same
        # instant (and datetime.fromisoformat would reject the 9-digit form).
        a = _parse_ts_ns(
            "2026-06-17T14:00:00.1234561+00:00", field_name="t", row_repr="r"
        )
        b = _parse_ts_ns(
            "2026-06-17T14:00:00.1234569+00:00", field_name="t", row_repr="r"
        )
        assert a == b

    def test_unparseable_raises(self) -> None:
        with pytest.raises(PartAError, match="unparseable"):
            _parse_ts_ns("nonsense", field_name="t", row_repr="r")

    def test_empty_raises(self) -> None:
        with pytest.raises(PartAError, match="empty"):
            _parse_ts_ns("   ", field_name="t", row_repr="r")


# --------------------------------------------------------------------------- #
# reconstruct_trades_db_row
# --------------------------------------------------------------------------- #


def db_row(**over: object) -> dict[str, object]:
    row: dict[str, object] = {
        "trader_id": "yank",
        "timestamp": "2026-06-17T14:00:00+00:00",
        "symbol": "MNQU26",
        "direction": "S",
        "entry_price": 20000.0,
        "exit_price": 19990.0,
        "exit_reason": "bars_held",
        "exit_timestamp": "2026-06-17T15:00:00+00:00",
        "metadata": {"contracts": 2, "gap_size": 5.0, "backfill": True},
    }
    row.update(over)
    return row


class TestReconstructTradesDbRow:
    def test_two_marketable_legs_short(self) -> None:
        trade = reconstruct_trades_db_row(db_row())
        assert trade.fidelity == "bar_reconstructed"
        assert len(trade.intents) == 2
        entry, exit_ = trade.intents
        assert entry.kind is OrderKind.MARKETABLE
        assert exit_.kind is OrderKind.MARKETABLE
        assert entry.side is Side.SELL  # short -> sell to open
        assert exit_.side is Side.BUY
        assert entry.size == 2 and exit_.size == 2
        assert entry.oco_group_id == exit_.oco_group_id is not None
        assert all(rf.fidelity == "bar_reconstructed" for rf in trade.real_fills)
        assert trade.real_fills[0].price_dbn == 20000_000_000_000
        assert trade.real_fills[1].price_dbn == 19990_000_000_000

    def test_long_direction(self) -> None:
        trade = reconstruct_trades_db_row(db_row(direction="LONG"))
        assert trade.intents[0].side is Side.BUY
        assert trade.intents[1].side is Side.SELL

    def test_direction_numeric_tokens(self) -> None:
        assert (
            reconstruct_trades_db_row(db_row(direction="0")).intents[0].side is Side.BUY
        )
        assert (
            reconstruct_trades_db_row(db_row(direction="1")).intents[0].side
            is Side.SELL
        )

    def test_bars_held_fallback(self) -> None:
        row = db_row(exit_timestamp=None, metadata={"contracts": 1, "bars_held": 30})
        trade = reconstruct_trades_db_row(row)
        delta_ns = trade.intents[1].submit_ts_ns - trade.intents[0].submit_ts_ns
        assert delta_ns == 30 * 60 * 1_000_000_000

    def test_empty_exit_timestamp_default_bars_held(self) -> None:
        row = db_row(exit_timestamp="", metadata={"contracts": 1})
        trade = reconstruct_trades_db_row(row)
        delta_ns = trade.intents[1].submit_ts_ns - trade.intents[0].submit_ts_ns
        assert delta_ns == 60 * 60 * 1_000_000_000

    def test_metadata_json_string(self) -> None:
        row = db_row(metadata=json.dumps({"contracts": 3, "backfill": True}))
        trade = reconstruct_trades_db_row(row)
        assert trade.intents[0].size == 3

    def test_metadata_none(self) -> None:
        trade = reconstruct_trades_db_row(db_row(metadata=None))
        assert trade.intents[0].size == 1  # default contracts

    def test_metadata_non_mapping_non_string_raises(self) -> None:
        with pytest.raises(PartAError, match="neither a mapping nor a JSON string"):
            reconstruct_trades_db_row(db_row(metadata=123))

    def test_integral_float_contracts(self) -> None:
        trade = reconstruct_trades_db_row(db_row(metadata={"contracts": 2.0}))
        assert trade.intents[0].size == 2

    def test_non_integral_float_contracts_raises(self) -> None:
        with pytest.raises(PartAError, match="non-integral"):
            reconstruct_trades_db_row(db_row(metadata={"contracts": 2.5}))

    def test_sqlite_id_used_in_trade_id(self) -> None:
        a = reconstruct_trades_db_row(db_row(id=41))
        b = reconstruct_trades_db_row(db_row(id=42))
        assert a.trade_id != b.trade_id
        assert a.trade_id == "yank-41"

    def test_exit_ts_not_after_entry_ts_raises(self) -> None:
        row = db_row(exit_timestamp="2026-06-17T13:00:00+00:00")
        with pytest.raises(PartAError, match="causally corrupt"):
            reconstruct_trades_db_row(row)

    def test_bad_direction_raises(self) -> None:
        with pytest.raises(PartAError, match="bad direction"):
            reconstruct_trades_db_row(db_row(direction="sideways"))

    def test_non_positive_price_raises(self) -> None:
        with pytest.raises(PartAError, match="non-finite / non-positive"):
            reconstruct_trades_db_row(db_row(entry_price=0.0))

    def test_bad_contracts_raises(self) -> None:
        with pytest.raises(PartAError, match="contracts"):
            reconstruct_trades_db_row(db_row(metadata={"contracts": "lots"}))


# --------------------------------------------------------------------------- #
# _build_mim_trade guard (causally-corrupt bracket — unreachable via the
# timestamp-sorted walk, so exercised directly)
# --------------------------------------------------------------------------- #


def test_build_mim_trade_rejects_causally_corrupt_bracket() -> None:
    entry = _PendingLeg(
        "E1", 100, Side.BUY, 1, fill_px_dbn=1, fill_ts_ns=100, fill_size=1
    )
    bad_exit = _PendingLeg(
        "X1", 100, Side.SELL, 1, fill_px_dbn=1, fill_ts_ns=100, fill_size=1
    )
    with pytest.raises(PartAError, match="causally corrupt"):
        _build_mim_trade(entry, bad_exit)


# --------------------------------------------------------------------------- #
# compare_fills
# --------------------------------------------------------------------------- #


class TestCompareFills:
    def _trade(self) -> ReconstructedTrade:
        return reconstruct_mim_nb(standard_mim_lifecycle())[0]

    def _match(self, trade: ReconstructedTrade) -> list[OrderOutcome]:
        rf_entry, rf_exit = trade.real_fills
        return [
            outcome(
                "E1",
                Leg.ENTRY,
                Side.BUY,
                fills=(Fill(px_dbn=rf_entry.price_dbn, size=1, ts_ns=1),),
            ),
            outcome(
                "X1",
                Leg.EXIT,
                Side.SELL,
                fills=(Fill(px_dbn=rf_exit.price_dbn, size=1, ts_ns=1),),
            ),
        ]

    def test_exact_match_zero_error(self) -> None:
        trade = self._trade()
        errors = compare_fills(self._match(trade), trade)
        assert [e.signed_error_ticks for e in errors] == [0.0, 0.0]
        assert all(e.real_ts_ns > 0 for e in errors)

    def test_sim_buy_one_tick_above_real_is_positive(self) -> None:
        trade = self._trade()
        outcomes = self._match(trade)
        outcomes[0] = outcome(
            "E1",
            Leg.ENTRY,
            Side.BUY,
            fills=(
                Fill(
                    px_dbn=trade.real_fills[0].price_dbn + MNQ_TICK_DBN,
                    size=1,
                    ts_ns=1,
                ),
            ),
        )
        errors = compare_fills(outcomes, trade)
        assert errors[0].signed_error_ticks == pytest.approx(1.0)

    def test_sim_sell_one_tick_below_real_is_positive(self) -> None:
        trade = self._trade()
        outcomes = self._match(trade)
        outcomes[1] = outcome(
            "X1",
            Leg.EXIT,
            Side.SELL,
            fills=(
                Fill(
                    px_dbn=trade.real_fills[1].price_dbn - MNQ_TICK_DBN,
                    size=1,
                    ts_ns=1,
                ),
            ),
        )
        errors = compare_fills(outcomes, trade)
        assert errors[1].signed_error_ticks == pytest.approx(1.0)

    def test_size_weighted_vwap(self) -> None:
        trade = self._trade()
        outcomes = self._match(trade)
        base = trade.real_fills[0].price_dbn
        outcomes[0] = outcome(
            "E1",
            Leg.ENTRY,
            Side.BUY,
            fills=(
                Fill(px_dbn=base, size=1, ts_ns=1),
                Fill(px_dbn=base + 4 * MNQ_TICK_DBN, size=3, ts_ns=2),
            ),
        )
        errors = compare_fills(outcomes, trade)
        assert errors[0].sim_vwap_dbn == base + 3 * MNQ_TICK_DBN
        assert errors[0].signed_error_ticks == pytest.approx(3.0)

    def test_leg_unfilled_miss(self) -> None:
        trade = self._trade()
        outcomes = self._match(trade)
        outcomes[1] = outcome("X1", Leg.EXIT, Side.SELL, terminal=TerminalState.EXPIRED)
        errors = compare_fills(outcomes, trade)
        assert errors[1].miss_reason == "leg_unfilled"
        assert errors[1].signed_error_ticks is None
        assert errors[1].sim_vwap_dbn is None
        assert errors[1].sim_terminal_state == "expired"

    def test_missing_outcome_raises(self) -> None:
        trade = self._trade()
        with pytest.raises(PartAError, match="no OrderOutcome"):
            compare_fills(self._match(trade)[:1], trade)

    def test_duplicate_outcome_raises(self) -> None:
        trade = self._trade()
        dup = self._match(trade)[0]
        with pytest.raises(PartAError, match="duplicate OrderOutcome"):
            compare_fills([dup, dup], trade)

    def test_side_mismatch_raises(self) -> None:
        trade = self._trade()
        outcomes = self._match(trade)
        outcomes[0] = outcome(
            "E1",
            Leg.ENTRY,
            Side.SELL,
            fills=(Fill(px_dbn=trade.real_fills[0].price_dbn, size=1, ts_ns=1),),
        )
        with pytest.raises(PartAError, match="side mismatch"):
            compare_fills(outcomes, trade)

    def test_foreign_trade_id_is_filtered_out(self) -> None:
        trade = self._trade()
        outcomes = self._match(trade)
        # relabel both to a different trade_id -> they no longer join
        outcomes = [
            outcome(
                o.order_id,
                o.leg,
                o.side,
                fills=o.fills,
                trade_id="some-other-trade",
            )
            for o in outcomes
        ]
        with pytest.raises(PartAError, match="no OrderOutcome"):
            compare_fills(outcomes, trade)

    def test_leftover_outcome_for_this_trade_is_ignored(self) -> None:
        trade = self._trade()
        outcomes = self._match(trade)
        outcomes.append(
            outcome(
                "LEFTOVER",
                Leg.EXIT,
                Side.SELL,
                fills=(Fill(px_dbn=1, size=1, ts_ns=1),),
            )
        )
        errors = compare_fills(outcomes, trade)
        assert len(errors) == 2


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #


class TestAggregate:
    def test_known_vector(self) -> None:
        errors = [fe(v) for v in (0.0, 1.0, -2.0, 3.0, -4.0)]
        result = aggregate(errors)
        assert result.stats.n == 5
        assert result.stats.mae_ticks == pytest.approx(2.0)
        assert result.stats.p90_ticks == pytest.approx(3.6)
        assert result.stats.signed_bias_ticks == pytest.approx(-0.4)
        assert result.verdict == "FAIL"  # N < 28

    def test_n_shortfall_reason(self) -> None:
        result = aggregate([fe(0.0) for _ in range(5)])
        assert result.verdict == "FAIL"
        assert "N=5" in result.reason and str(PART_A_MIN_N) in result.reason

    def test_n_floor_only_no_spurious_disagreement_warning(self) -> None:
        # sub-28 sample, quality perfect on both the full set and the broker_fill
        # subset -> FAIL on N alone, and NO "subset disagrees" warning.
        result = aggregate([fe(0.0) for _ in range(10)])
        assert result.verdict == "FAIL"
        assert "N=10" in result.reason
        assert result.warning is None

    def test_pass_when_all_bounds_met(self) -> None:
        result = aggregate([fe(0.0) for _ in range(PART_A_MIN_N)])
        assert result.verdict == "PASS"
        assert result.stats.mae_ticks == 0.0
        assert result.stats.signed_bias_ticks == 0.0
        assert result.warning is None
        assert result.unresolved_misses == 0

    def test_mae_over_tolerance(self) -> None:
        vals = [1.5 if i % 2 == 0 else -1.5 for i in range(PART_A_MIN_N)]
        result = aggregate([fe(v) for v in vals])
        assert result.verdict == "FAIL"
        assert "MAE" in result.reason
        assert result.stats.mae_ticks > PARITY_MAE_MAX_TICKS

    def test_p90_over_tolerance(self) -> None:
        vals = [0.0] * (PART_A_MIN_N - 4) + [3.0, -3.0, 3.0, -3.0]
        result = aggregate([fe(v) for v in vals])
        assert result.verdict == "FAIL"
        assert "p90" in result.reason
        assert result.stats.p90_ticks > PARITY_P90_MAX_TICKS
        assert result.stats.mae_ticks <= PARITY_MAE_MAX_TICKS

    def test_signed_bias_over_tolerance(self) -> None:
        result = aggregate([fe(0.5) for _ in range(PART_A_MIN_N)])
        assert result.verdict == "FAIL"
        assert "signed bias" in result.reason
        assert abs(result.stats.signed_bias_ticks) > PARITY_SIGNED_BIAS_MAX_TICKS

    def test_unresolved_miss_blocks_pass(self) -> None:
        errors = [fe(0.0) for _ in range(PART_A_MIN_N - 1)] + [fe(None)]
        result = aggregate(errors)
        assert result.stats.n == PART_A_MIN_N
        assert result.verdict == "FAIL"
        assert "unresolved" in result.reason
        assert result.unresolved_misses == 1

    def test_broker_fill_subset_empty_warning(self) -> None:
        result = aggregate(
            [fe(0.0, fidelity="bar_reconstructed") for _ in range(PART_A_MIN_N)]
        )
        assert result.verdict == "PASS"
        assert result.warning is not None
        assert "empty" in result.warning
        assert result.broker_fill_stats.n == 0

    def test_broker_fill_subset_disagreement_warning(self) -> None:
        errors = [fe(2.5, fidelity="broker_fill"), fe(2.5, fidelity="broker_fill")]
        errors += [
            fe(0.0, fidelity="bar_reconstructed") for _ in range(PART_A_MIN_N - 2)
        ]
        result = aggregate(errors)
        assert result.verdict == "PASS"  # pooled stats within tolerance
        assert result.broker_fill_stats.mae_ticks > PARITY_MAE_MAX_TICKS
        assert result.warning is not None
        assert "disagree" in result.warning

    def test_no_warning_on_agreement(self) -> None:
        errors = [fe(0.0, fidelity="broker_fill") for _ in range(PART_A_MIN_N)]
        result = aggregate(errors)
        assert result.verdict == "PASS"
        assert result.warning is None
        assert result.broker_fill_stats.n == PART_A_MIN_N


# --------------------------------------------------------------------------- #
# ReconstructedTrade.__post_init__
# --------------------------------------------------------------------------- #


class TestReconstructedTradePostInit:
    def test_zero_real_fills_rejected(self) -> None:
        trade = reconstruct_mim_nb(standard_mim_lifecycle())[0]
        with pytest.raises(PartAError, match="zero real_fills"):
            ReconstructedTrade(
                trade_id="x",
                intents=trade.intents,
                real_fills=(),
                fidelity="broker_fill",
            )

    def test_no_intents_rejected(self) -> None:
        rf = RealFill(
            order_id="o",
            leg=Leg.ENTRY,
            side=Side.BUY,
            size=1,
            price_dbn=1_000,
            ts_ns=1,
            fidelity="broker_fill",
        )
        with pytest.raises(PartAError, match="no intents"):
            ReconstructedTrade(
                trade_id="x", intents=(), real_fills=(rf,), fidelity="broker_fill"
            )

    def test_intents_must_share_oco_group(self) -> None:
        trade = reconstruct_mim_nb(standard_mim_lifecycle())[0]
        bad_exit = trade.intents[1].model_copy(update={"oco_group_id": "other"})
        with pytest.raises(PartAError, match="oco_group_id"):
            ReconstructedTrade(
                trade_id="x",
                intents=(trade.intents[0], bad_exit),
                real_fills=trade.real_fills,
                fidelity="broker_fill",
            )

    def test_intents_must_share_trade_id(self) -> None:
        trade = reconstruct_mim_nb(standard_mim_lifecycle())[0]
        bad_exit = trade.intents[1].model_copy(update={"trade_id": "other-trade"})
        with pytest.raises(PartAError, match="multiple trade_ids"):
            ReconstructedTrade(
                trade_id="x",
                intents=(trade.intents[0], bad_exit),
                real_fills=trade.real_fills,
                fidelity="broker_fill",
            )

    def test_real_fill_without_matching_intent_rejected(self) -> None:
        trade = reconstruct_mim_nb(standard_mim_lifecycle())[0]
        orphan = RealFill(
            order_id="NOPE",
            leg=Leg.EXIT,
            side=Side.SELL,
            size=1,
            price_dbn=1_000,
            ts_ns=1,
            fidelity="broker_fill",
        )
        with pytest.raises(PartAError, match="no matching"):
            ReconstructedTrade(
                trade_id="x",
                intents=trade.intents,
                real_fills=(trade.real_fills[0], orphan),
                fidelity="broker_fill",
            )

    def test_intents_must_be_non_decreasing(self) -> None:
        trade = reconstruct_mim_nb(standard_mim_lifecycle())[0]
        with pytest.raises(PartAError, match="non-decreasing"):
            ReconstructedTrade(
                trade_id="x",
                intents=(trade.intents[1], trade.intents[0]),
                real_fills=trade.real_fills,
                fidelity="broker_fill",
            )
