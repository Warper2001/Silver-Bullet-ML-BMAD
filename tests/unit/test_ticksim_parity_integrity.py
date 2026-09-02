"""Unit tests for ``src/ticksim/parity/integrity.py`` (seal §5 / prereg §A9.3).

Every case runs against a hand-built in-memory
:class:`~src.ticksim.events.BookEventSource` (a list of
:class:`~src.ticksim.events.BookEvent`\\ s). Covers every row of the spec's
I/O & Edge-Case matrix, the ``format_integrity`` OK / FLAGGED shapes, and each
``verdict`` flag condition in isolation.
"""

from __future__ import annotations

from collections.abc import Iterator

from src.ticksim.config import MAX_TRANSIENT_CROSS_NS, MNQ_TICK_DBN
from src.ticksim.events import BookEvent, MboAction, MboSide
from src.ticksim.parity.integrity import (
    IntegrityReport,
    format_integrity,
    preflight_integrity,
)

IID = 42004800
OTHER_IID = 42004801
TICK = MNQ_TICK_DBN
BID_PX = 100 * TICK
ASK_PX = 101 * TICK
WARMUP_NS = 60_000_000_000


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
    *,
    instrument_id: int = IID,
) -> BookEvent:
    return BookEvent(
        action=action,
        side=side,
        order_id=order_id,
        price_dbn=price_dbn,
        size=size,
        ts_event=ts,
        sequence=seq,
        instrument_id=instrument_id,
    )


def clean_events() -> list[BookEvent]:
    """A monotonic window with one of each A/C/M/T/F and no crossed market."""
    return [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 5, 1_000, 1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 5, 2_000, 2),
        be(MboAction.MODIFY, MboSide.BID, 1, BID_PX, 3, 3_000, 3),
        be(MboAction.TRADE, MboSide.NONE, 0, BID_PX, 1, 4_000, 4),
        be(MboAction.FILL, MboSide.BID, 1, BID_PX, 1, 5_000, 5),
        be(MboAction.CANCEL, MboSide.BID, 1, BID_PX, 3, 6_000, 6),
    ]


# --------------------------------------------------------------------------- #
# I/O & Edge-Case matrix
# --------------------------------------------------------------------------- #


def test_clean_rth_window_is_ok() -> None:
    report = preflight_integrity(ListSource(clean_events()))
    assert report.verdict == "OK"
    assert report.flags == ()
    assert report.missing_actions == ()
    assert report.actions_seen == frozenset({"A", "C", "M", "T", "F"})
    assert report.persistent_cross_count == 0
    assert report.transient_cross_count == 0
    assert report.n_events == 6
    assert report.n_trades == 1
    assert report.instrument_id == IID
    assert report.session_low_dbn == BID_PX
    assert report.session_high_dbn == BID_PX


def test_ts_regression_is_flagged_with_an_example() -> None:
    events = clean_events()
    events.insert(2, be(MboAction.ADD, MboSide.ASK, 9, ASK_PX, 1, 1_500, 99))
    report = preflight_integrity(ListSource(events))
    assert report.ts_regressions == 1
    assert report.ts_regression_examples == ((2_000, 1_500),)
    assert report.verdict == "FLAGGED"
    assert "ts regressions" in report.flags


def test_transient_cross_stays_ok() -> None:
    events = [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 5, 1_000, 1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 5, 2_000, 2),
        be(MboAction.ADD, MboSide.ASK, 3, BID_PX, 5, 3_000, 3),  # crosses
        be(MboAction.CANCEL, MboSide.ASK, 3, BID_PX, 5, 3_000 + 20_000_000, 4),
        be(MboAction.MODIFY, MboSide.BID, 1, BID_PX, 3, 30_000_000, 5),
        be(MboAction.TRADE, MboSide.NONE, 0, BID_PX, 1, 31_000_000, 6),
        be(MboAction.FILL, MboSide.BID, 1, BID_PX, 1, 32_000_000, 7),
    ]
    report = preflight_integrity(ListSource(events))
    assert report.transient_cross_count == 1
    assert report.persistent_cross_count == 0
    assert report.verdict == "OK"


def _persistent_cross_events() -> list[BookEvent]:
    resolve_ts = 3_000 + 200_000_000
    return [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 5, 1_000, 1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 5, 2_000, 2),
        be(MboAction.ADD, MboSide.ASK, 3, BID_PX, 5, 3_000, 3),  # crosses
        be(MboAction.CANCEL, MboSide.ASK, 3, BID_PX, 5, resolve_ts, 4),
        be(MboAction.MODIFY, MboSide.BID, 1, BID_PX, 2, resolve_ts + 1_000, 5),
        be(MboAction.TRADE, MboSide.NONE, 0, BID_PX, 1, resolve_ts + 2_000, 6),
        be(MboAction.FILL, MboSide.ASK, 1, BID_PX, 1, resolve_ts + 3_000, 7),
    ]


def test_persistent_cross_is_flagged() -> None:
    report = preflight_integrity(ListSource(_persistent_cross_events()))
    assert report.persistent_cross_count == 1
    assert report.persistent_crosses[0][0] == 3_000
    assert report.persistent_crosses[0][2] >= 200_000_000
    assert report.transient_cross_count == 0
    assert report.book_inconsistencies == 0
    assert report.verdict == "FLAGGED"
    assert "persistent cross" in report.flags


def test_cross_open_at_stream_end_is_its_own_flag() -> None:
    events = [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 5, 1_000, 1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 5, 2_000, 2),
        be(MboAction.MODIFY, MboSide.BID, 1, BID_PX, 3, 3_000, 3),
        be(MboAction.TRADE, MboSide.NONE, 0, BID_PX, 1, 4_000, 4),
        be(MboAction.FILL, MboSide.BID, 1, BID_PX, 1, 5_000, 5),
        be(MboAction.CANCEL, MboSide.ASK, 2, ASK_PX, 5, 6_000, 6),
        be(
            MboAction.ADD, MboSide.ASK, 3, BID_PX, 5, 7_000, 7
        ),  # crosses, never resolves
    ]
    report = preflight_integrity(ListSource(events))
    # not folded into either count -- its own flag (amended spec)
    assert report.transient_cross_count == 0
    assert report.persistent_cross_count == 0
    assert report.unresolved_cross_at_end is True
    assert report.verdict == "FLAGGED"
    assert "unresolved cross at end" in report.flags


def test_multi_event_persistent_cross_no_book_inconsistency() -> None:
    # cross opens at 3_000; events land WHILE crossed past 50 ms; book.apply_event
    # _fail()s on each, but those are the cross -- must not count as book
    # inconsistencies (verification-gap gap).
    resolve_ts = 3_000 + 300_000_000
    events = [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 5, 1_000, 1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 5, 2_000, 2),
        be(MboAction.ADD, MboSide.ASK, 3, BID_PX, 5, 3_000, 3),  # crosses
        be(MboAction.MODIFY, MboSide.BID, 1, BID_PX, 4, 3_000 + 80_000_000, 4),
        be(MboAction.MODIFY, MboSide.BID, 1, BID_PX, 3, 3_000 + 150_000_000, 5),
        be(MboAction.TRADE, MboSide.NONE, 0, BID_PX, 1, 3_000 + 220_000_000, 6),
        be(MboAction.CANCEL, MboSide.ASK, 3, BID_PX, 5, resolve_ts, 7),  # resolves
        be(MboAction.FILL, MboSide.BID, 1, BID_PX, 1, resolve_ts + 1_000, 8),
    ]
    report = preflight_integrity(ListSource(events))
    assert report.persistent_cross_count == 1
    assert report.persistent_crosses[0][2] >= 300_000_000
    assert report.book_inconsistencies == 0
    assert report.flags == ("persistent cross",)
    assert report.verdict == "FLAGGED"


def test_ts_cascade_does_not_inflate_book_inconsistencies() -> None:
    # 100, 200, 50, 60: the 50 and 60 events are both below max_ts_seen (200);
    # apply_event raises for both, but they are the regression -- not counted.
    events = [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 5, 100, 1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 5, 200, 2),
        be(MboAction.MODIFY, MboSide.BID, 1, BID_PX, 3, 50, 3),
        be(MboAction.MODIFY, MboSide.ASK, 2, ASK_PX, 4, 60, 4),
        be(MboAction.TRADE, MboSide.NONE, 0, BID_PX, 1, 300, 5),
        be(MboAction.FILL, MboSide.BID, 1, BID_PX, 1, 400, 6),
        be(MboAction.CANCEL, MboSide.BID, 1, BID_PX, 3, 500, 7),
    ]
    report = preflight_integrity(ListSource(events))
    assert report.ts_regressions == 2
    assert report.book_inconsistencies == 0
    assert "ts regressions" in report.flags
    assert "book inconsistencies" not in report.flags


def test_mid_iteration_source_error_is_counted_not_raised() -> None:
    class BoomSource:
        class_rank = 0

        def __iter__(self) -> Iterator[BookEvent]:
            yield be(MboAction.ADD, MboSide.BID, 1, BID_PX, 5, 1_000, 1)
            yield be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 5, 2_000, 2)
            raise ValueError("unknown MBO action code 'X'")

    report = preflight_integrity(BoomSource())
    assert report.malformed_events >= 1
    assert report.verdict == "FLAGGED"
    assert "malformed events" in report.flags


def test_large_inter_event_gap_is_flagged() -> None:
    events = [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 5, 1_000, 1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 5, 2_000, 2),
        # 10-minute forward jump
        be(MboAction.MODIFY, MboSide.BID, 1, BID_PX, 3, 2_000 + 600_000_000_000, 3),
        be(MboAction.TRADE, MboSide.NONE, 0, BID_PX, 1, 2_000 + 600_000_000_001, 4),
        be(MboAction.FILL, MboSide.BID, 1, BID_PX, 1, 2_000 + 600_000_000_002, 5),
        be(MboAction.CANCEL, MboSide.BID, 1, BID_PX, 3, 2_000 + 600_000_000_003, 6),
    ]
    report = preflight_integrity(ListSource(events))
    assert report.gaps_over_threshold >= 1
    assert report.largest_gap_ns >= 600_000_000_000
    assert report.verdict == "FLAGGED"
    assert "large inter-event gaps" in " ".join(report.flags) or any(
        "gap" in f for f in report.flags
    )


def test_foreign_event_not_folded_into_actions_seen() -> None:
    events = clean_events()
    # a foreign instrument's TRADE -- must NOT let the primary window's
    # missing-T check pass
    events = [e for e in events if str(e.action) != "T"]
    events.insert(
        3,
        be(
            MboAction.TRADE,
            MboSide.NONE,
            0,
            BID_PX,
            1,
            3_500,
            50,
            instrument_id=OTHER_IID,
        ),
    )
    report = preflight_integrity(ListSource(events))
    assert "T" not in report.actions_seen
    assert report.missing_actions == ("T",)
    assert report.foreign_instrument_events == 1


def test_missing_action_class_is_flagged() -> None:
    events = [ev for ev in clean_events() if str(ev.action) != "T"]
    report = preflight_integrity(ListSource(events))
    assert report.missing_actions == ("T",)
    assert report.verdict == "FLAGGED"
    assert any("missing actions" in f for f in report.flags)


def test_post_warmup_unknown_ref_counts_as_book_inconsistency() -> None:
    ts0 = WARMUP_NS + 10_000_000_000
    events = [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 5, 1_000, 1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 5, 2_000, 2),
        be(MboAction.MODIFY, MboSide.BID, 1, BID_PX, 3, 3_000, 3),
        be(MboAction.TRADE, MboSide.NONE, 0, BID_PX, 1, 4_000, 4),
        be(MboAction.FILL, MboSide.BID, 1, BID_PX, 1, 5_000, 5),
        be(
            MboAction.CANCEL, MboSide.BID, 999, BID_PX, 5, ts0, 6
        ),  # unknown, past warm-up
    ]
    report = preflight_integrity(ListSource(events))
    assert report.book_inconsistencies >= 1
    assert report.warmup_unknown_ref == 0
    assert report.verdict == "FLAGGED"
    assert "book inconsistencies" in report.flags


def test_stale_cold_start_cross_is_counted_and_rendered() -> None:
    """A cross wider than ``STALE_CROSS_MAX_TICKS`` (a pre-window ghost the cold
    reconstruction never saw ``A``-dded) is surfaced as ``stale_cross_count``,
    which the ``format_integrity`` block renders. It is not a flag reason of its
    own -- the preflight's own BBO state machine still reports the episode as a
    persistent cross, exactly as before."""
    events = clean_events()
    # a resting bid 60 ticks above the ask, never cancelled
    events.append(be(MboAction.ADD, MboSide.BID, 99, ASK_PX + 60 * TICK, 5, 7_000, 7))
    events.append(be(MboAction.ADD, MboSide.ASK, 98, ASK_PX + 900 * TICK, 5, 8_000, 8))
    report = preflight_integrity(ListSource(events))
    assert report.stale_cross_count == 1
    assert "- stale (cold-start) cross episodes tolerated: 1" in format_integrity(
        report
    )


def test_clean_window_reports_no_stale_crosses() -> None:
    report = preflight_integrity(ListSource(clean_events()))
    assert report.stale_cross_count == 0
    assert report.verdict == "OK"
    assert "- stale (cold-start) cross episodes tolerated: 0" in format_integrity(
        report
    )


def test_warmup_unknown_ref_does_not_flag() -> None:
    events = clean_events()
    events.append(be(MboAction.CANCEL, MboSide.BID, 999, BID_PX, 5, 7_000, 7))
    report = preflight_integrity(ListSource(events))
    assert report.warmup_unknown_ref >= 1
    assert report.book_inconsistencies == 0
    assert report.verdict == "OK"


def test_second_instrument_is_flagged() -> None:
    events = clean_events()
    events.insert(
        3,
        be(
            MboAction.ADD,
            MboSide.BID,
            50,
            BID_PX,
            5,
            3_500,
            50,
            instrument_id=OTHER_IID,
        ),
    )
    report = preflight_integrity(ListSource(events))
    assert report.foreign_instrument_events >= 1
    assert report.instrument_id == IID
    assert report.verdict == "FLAGGED"
    assert "foreign instrument_id" in report.flags


def test_off_book_trade_is_flagged() -> None:
    events = [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 5, 1_000, 1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 5, 2_000, 2),
        be(MboAction.TRADE, MboSide.NONE, 0, ASK_PX + 10 * TICK, 1, 3_000, 3),
        be(MboAction.MODIFY, MboSide.BID, 1, BID_PX, 3, 4_000, 4),
        be(MboAction.FILL, MboSide.BID, 1, BID_PX, 1, 5_000, 5),
        be(MboAction.CANCEL, MboSide.BID, 1, BID_PX, 3, 6_000, 6),
    ]
    report = preflight_integrity(ListSource(events))
    assert report.trades_off_book == 1
    assert report.session_high_dbn == ASK_PX + 10 * TICK
    assert report.verdict == "FLAGGED"
    assert "off-book trades" in report.flags


def test_trade_just_inside_tolerance_is_not_off_book() -> None:
    events = [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 5, 1_000, 1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 5, 2_000, 2),
        be(MboAction.TRADE, MboSide.NONE, 0, ASK_PX + 2 * TICK, 1, 3_000, 3),
        be(MboAction.MODIFY, MboSide.BID, 1, BID_PX, 3, 4_000, 4),
        be(MboAction.FILL, MboSide.BID, 1, BID_PX, 1, 5_000, 5),
        be(MboAction.CANCEL, MboSide.BID, 1, BID_PX, 3, 6_000, 6),
    ]
    report = preflight_integrity(ListSource(events))
    assert report.trades_off_book == 0
    assert report.verdict == "OK"


def test_degraded_day_is_recorded_but_does_not_flag() -> None:
    report = preflight_integrity(
        ListSource(clean_events()), degraded_days=("2026-07-30",)
    )
    assert report.degraded_days == ("2026-07-30",)
    assert report.verdict == "OK"
    assert "2026-07-30" in format_integrity(report)


def test_empty_source() -> None:
    report = preflight_integrity(ListSource([]))
    assert report.n_events == 0
    assert report.instrument_id is None
    assert report.verdict == "FLAGGED"
    assert report.flags == ("no events",)
    # format still renders
    assert format_integrity(report).startswith("integrity: FLAGGED (no events)")


def test_one_sided_book_trade_not_counted() -> None:
    events = [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 5, 1_000, 1),
        be(MboAction.TRADE, MboSide.NONE, 0, BID_PX + 99 * TICK, 1, 2_000, 2),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 5, 3_000, 3),
        be(MboAction.MODIFY, MboSide.BID, 1, BID_PX, 3, 4_000, 4),
        be(MboAction.FILL, MboSide.BID, 1, BID_PX, 1, 5_000, 5),
        be(MboAction.CANCEL, MboSide.BID, 1, BID_PX, 3, 6_000, 6),
    ]
    report = preflight_integrity(ListSource(events))
    assert report.trades_off_book == 0  # only-bid book at the trade instant


# --------------------------------------------------------------------------- #
# format_integrity
# --------------------------------------------------------------------------- #


def test_format_integrity_ok_shape_is_deterministic_and_ascii() -> None:
    report = preflight_integrity(ListSource(clean_events()))
    first = format_integrity(report)
    second = format_integrity(report)
    assert first == second
    assert first.isascii()
    assert first.splitlines()[0] == "integrity: OK"
    assert "- events: 6" in first
    assert "- transient crosses: 0" in first
    assert "- actions seen: A,C,F,M,T" in first
    assert "- missing actions: (none)" in first
    assert "- degraded days: (none)" in first
    assert "##" not in first
    assert "—" not in first  # no em-dash


def test_format_integrity_flagged_shape() -> None:
    report = preflight_integrity(ListSource(_persistent_cross_events()))
    text = format_integrity(report)
    assert format_integrity(report) == text
    assert text.isascii()
    assert text.splitlines()[0] == "integrity: FLAGGED (persistent cross)"
    assert "- persistent cross examples:" in text
    assert "duration=200000000" in text
    assert "##" not in text


def test_format_integrity_flagged_lists_ts_regression_examples() -> None:
    events = clean_events()
    events.insert(2, be(MboAction.ADD, MboSide.ASK, 9, ASK_PX, 1, 1_500, 99))
    report = preflight_integrity(ListSource(events))
    text = format_integrity(report)
    assert "- ts regression examples:" in text
    assert "prev_max=2000 ts=1500" in text
    assert "- window: 1000 .. 6000" in text
    assert "- bbo cross rate: 0.0000%" in text


# --------------------------------------------------------------------------- #
# verdict flag conditions in isolation
# --------------------------------------------------------------------------- #


def test_each_flag_condition_sets_flagged() -> None:
    # ts regression
    ev = clean_events()
    ev.insert(2, be(MboAction.ADD, MboSide.ASK, 9, ASK_PX, 1, 1_500, 99))
    assert preflight_integrity(ListSource(ev)).verdict == "FLAGGED"

    # persistent cross
    assert (
        preflight_integrity(ListSource(_persistent_cross_events())).verdict == "FLAGGED"
    )

    # foreign instrument
    ev = clean_events()
    ev.insert(
        3,
        be(
            MboAction.ADD,
            MboSide.BID,
            50,
            BID_PX,
            5,
            3_500,
            50,
            instrument_id=OTHER_IID,
        ),
    )
    assert preflight_integrity(ListSource(ev)).verdict == "FLAGGED"

    # missing action
    ev = [e for e in clean_events() if str(e.action) != "F"]
    assert preflight_integrity(ListSource(ev)).verdict == "FLAGGED"

    # empty
    assert preflight_integrity(ListSource([])).verdict == "FLAGGED"


def test_transient_cross_and_degraded_day_never_flag_together() -> None:
    events = [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 5, 1_000, 1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 5, 2_000, 2),
        be(MboAction.ADD, MboSide.ASK, 3, BID_PX, 5, 3_000, 3),
        be(
            MboAction.CANCEL,
            MboSide.ASK,
            3,
            BID_PX,
            5,
            3_000 + MAX_TRANSIENT_CROSS_NS,
            4,
        ),
        be(MboAction.MODIFY, MboSide.BID, 1, BID_PX, 3, 60_000_000, 5),
        be(MboAction.TRADE, MboSide.NONE, 0, BID_PX, 1, 61_000_000, 6),
        be(MboAction.FILL, MboSide.BID, 1, BID_PX, 1, 62_000_000, 7),
    ]
    report = preflight_integrity(
        ListSource(events), degraded_days=("2026-05-24", "2026-07-30")
    )
    assert report.transient_cross_count == 1
    assert report.degraded_days == ("2026-05-24", "2026-07-30")
    assert report.verdict == "OK"


def test_report_is_frozen() -> None:
    report = preflight_integrity(ListSource(clean_events()))
    assert isinstance(report, IntegrityReport)
    try:
        report.n_events = 99  # type: ignore[misc]
    except Exception as exc:  # FrozenInstanceError
        assert type(exc).__name__ == "FrozenInstanceError"
    else:  # pragma: no cover
        raise AssertionError("IntegrityReport must be frozen")
