"""Unit tests for ``src/ticksim/cli.py`` -- the ``simulate`` sub-command,
``FrontMonthSource``, ``detect_front_month``, ``_read_intents``, ``_span_interval``.

Every ``main(["simulate", ...])`` case monkeypatches ``cli.DbnMboSource`` to a
hand-built in-memory list of :class:`BookEvent`s and writes to ``tmp_path`` --
no ``.dbn.zst`` fixture is touched. The I/O & Edge-Case matrix rows in
``spec-ticksim-cli-simulate.md`` are each exercised for their stated exit code
and a stderr message.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.ticksim import cli
from src.ticksim.config import MNQ_TICK_DBN
from src.ticksim.events import BookEvent, MboAction, MboSide
from src.ticksim.orders import (
    IntentAction,
    Leg,
    OrderIntent,
    OrderKind,
    OrderOutcome,
    OrderStateError,
    Side,
)
from src.ticksim.sim import IntentLogError, InvariantViolation

# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #

IID = 42004800
SPREAD_IID = 42004801
TICK = MNQ_TICK_DBN
P = 20_000_000_000_000  # 20000.0 in DBN 1e-9 fixed-point
BID_PX = P - TICK
ASK_PX = P + TICK

B = 1_700_000_000 * 1_000_000_000  # ns base (whole seconds -> exact)
HOLD = 600 * 1_000_000_000  # 600 s


def be(
    action: MboAction,
    side: MboSide,
    order_id: int,
    price_dbn: int,
    size: int,
    ts: int,
    seq: int,
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


def both_sides_book(instrument_id: int = IID) -> list[BookEvent]:
    return [
        be(
            MboAction.ADD,
            MboSide.BID,
            1,
            BID_PX,
            100,
            ts=B,
            seq=1,
            instrument_id=instrument_id,
        ),
        be(
            MboAction.ADD,
            MboSide.ASK,
            2,
            ASK_PX,
            100,
            ts=B,
            seq=2,
            instrument_id=instrument_id,
        ),
    ]


class ReiterableSource:
    """Re-iterable in-memory :class:`BookEventSource` (spine AD-18)."""

    class_rank = 0

    def __init__(self, events: list[BookEvent]) -> None:
        self._events = list(events)

    def __iter__(self) -> Iterator[BookEvent]:
        return iter(self._events)


def one_shot(events: list[BookEvent]):
    """A bare generator -- ``iter(x) is iter(x)``, so not re-iterable (AD-18)."""
    return (ev for ev in events)


def marketable(order_id: str, leg: Leg, side: Side, submit_ts_ns: int) -> OrderIntent:
    return OrderIntent(
        action=IntentAction.SUBMIT,
        order_id=order_id,
        trade_id="t1",
        leg=leg,
        kind=OrderKind.MARKETABLE,
        side=side,
        size=1,
        submit_ts_ns=submit_ts_ns,
    )


def round_trip_intents() -> list[OrderIntent]:
    return [
        marketable("t1-e", Leg.ENTRY, Side.BUY, B),
        marketable("t1-x", Leg.EXIT, Side.SELL, B + HOLD),
    ]


def write_intents(path: Path, intents: list[OrderIntent]) -> Path:
    path.write_text("".join(i.model_dump_json() + "\n" for i in intents))
    return path


@pytest.fixture
def patched_source(monkeypatch: pytest.MonkeyPatch):
    """Return a setter: ``patched_source(events)`` makes ``cli.DbnMboSource(any)``
    yield ``events`` (re-iterable)."""

    def _set(events: list[BookEvent]) -> None:
        monkeypatch.setattr(cli, "DbnMboSource", lambda _path: ReiterableSource(events))

    return _set


def _dbn_file(tmp_path: Path) -> Path:
    path = tmp_path / "window.mbo.dbn.zst"
    path.write_bytes(b"")  # only existence is checked; content is monkeypatched
    return path


def _simulate_argv(dbn: Path, intents: Path, out_dir: Path, **extra: str) -> list[str]:
    argv = [
        "simulate",
        "--dbn",
        str(dbn),
        "--intents",
        str(intents),
        "--out-outcomes",
        str(out_dir / "outcomes.jsonl"),
        "--out-manifest",
        str(out_dir / "manifest.json"),
    ]
    for key, value in extra.items():
        argv += [f"--{key.replace('_', '-')}", value]
    return argv


# --------------------------------------------------------------------------- #
# FrontMonthSource
# --------------------------------------------------------------------------- #


class TestFrontMonthSource:
    def test_filters_to_one_instrument(self) -> None:
        events = both_sides_book(IID) + both_sides_book(SPREAD_IID)
        src = cli.FrontMonthSource(ReiterableSource(events), IID)
        got = list(src)
        assert got == both_sides_book(IID)
        assert all(ev.instrument_id == IID for ev in got)

    def test_is_reiterable(self) -> None:
        src = cli.FrontMonthSource(ReiterableSource(both_sides_book()), IID)
        assert list(src) == list(src)
        assert len(list(src)) == 2

    def test_class_rank_forwarded_from_inner(self) -> None:
        inner = ReiterableSource([])
        inner.class_rank = 7  # type: ignore[assignment]
        assert cli.FrontMonthSource(inner, IID).class_rank == 7

    def test_no_matching_instrument_yields_empty(self) -> None:
        src = cli.FrontMonthSource(ReiterableSource(both_sides_book(IID)), 999)
        assert list(src) == []

    def test_rejects_one_shot_inner(self) -> None:
        with pytest.raises(TypeError, match="re-iterable"):
            cli.FrontMonthSource(one_shot(both_sides_book()), IID)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# detect_front_month
# --------------------------------------------------------------------------- #


class TestDetectFrontMonth:
    def test_modal_id(self) -> None:
        events = [
            be(MboAction.ADD, MboSide.BID, i, BID_PX, 1, ts=B + i, seq=i)
            for i in range(1, 25)
        ] + [
            be(
                MboAction.ADD,
                MboSide.ASK,
                100,
                ASK_PX,
                1,
                ts=B,
                seq=100,
                instrument_id=SPREAD_IID,
            )
        ]
        assert cli.detect_front_month(ReiterableSource(events)) == IID

    def test_empty_stream_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            cli.detect_front_month(ReiterableSource([]))

    def test_tie_raises(self) -> None:
        events = [
            be(MboAction.ADD, MboSide.BID, 1, BID_PX, 1, ts=B, seq=1),
            be(
                MboAction.ADD,
                MboSide.ASK,
                2,
                ASK_PX,
                1,
                ts=B,
                seq=2,
                instrument_id=SPREAD_IID,
            ),
        ]
        with pytest.raises(ValueError, match="ambiguous"):
            cli.detect_front_month(ReiterableSource(events))

    def test_rejects_one_shot_source(self) -> None:
        with pytest.raises(TypeError, match="re-iterable"):
            cli.detect_front_month(one_shot(both_sides_book()))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# _read_intents
# --------------------------------------------------------------------------- #


class TestReadIntents:
    def test_happy_path_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "intents.jsonl"
        lines = [i.model_dump_json() for i in round_trip_intents()]
        path.write_text(lines[0] + "\n\n  \n" + lines[1] + "\n")
        got = cli._read_intents(path)
        assert [i.order_id for i in got] == ["t1-e", "t1-x"]

    def test_malformed_line_names_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "intents.jsonl"
        good = round_trip_intents()[0].model_dump_json()
        path.write_text(good + "\n" + "{not valid json\n")
        with pytest.raises(cli._CliError, match="line 2"):
            cli._read_intents(path)

    def test_schema_violation_names_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "intents.jsonl"
        path.write_text('{"action": "submit", "order_id": "x"}\n')
        with pytest.raises(cli._CliError, match="line 1"):
            cli._read_intents(path)

    def test_empty_file_raises_no_intents(self, tmp_path: Path) -> None:
        path = tmp_path / "intents.jsonl"
        path.write_text("\n  \n")
        with pytest.raises(cli._CliError, match="no intents"):
            cli._read_intents(path)

    def test_unreadable_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(cli._CliError, match="cannot read"):
            cli._read_intents(tmp_path / "does-not-exist.jsonl")

    def test_non_utf8_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "intents.jsonl"
        path.write_bytes(b"\xff\xfe not utf-8 \x80\n")
        with pytest.raises(cli._CliError, match="cannot read"):
            cli._read_intents(path)


# --------------------------------------------------------------------------- #
# _span_interval
# --------------------------------------------------------------------------- #


class TestSpanInterval:
    def test_pads_each_side_plus_one_ns_on_end(self) -> None:
        intents = round_trip_intents()
        pad = 5 * 60 * 1_000_000_000
        lo, hi = cli._span_interval(intents, pad)
        assert lo == B - pad
        assert hi == B + HOLD + pad + 1

    def test_start_clamped_at_zero(self) -> None:
        intents = [marketable("a", Leg.ENTRY, Side.BUY, 10)]
        lo, hi = cli._span_interval(intents, 1_000)
        assert lo == 0
        assert hi == 1_011

    def test_single_intent_zero_pad_is_a_valid_window(self) -> None:
        intents = [marketable("a", Leg.ENTRY, Side.BUY, B)]
        lo, hi = cli._span_interval(intents, 0)
        assert (lo, hi) == (B, B + 1)
        assert lo < hi  # not a degenerate [x, x) that sim would reject


# --------------------------------------------------------------------------- #
# main(["simulate", ...]) -- happy paths
# --------------------------------------------------------------------------- #


class TestSimulateHappyPath:
    def test_explicit_iid_round_trips(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        patched_source(both_sides_book())
        dbn = _dbn_file(tmp_path)
        intents = write_intents(tmp_path / "i.jsonl", round_trip_intents())
        rc = cli.main(_simulate_argv(dbn, intents, tmp_path, instrument_id=str(IID)))
        assert rc == 0

        outcomes_text = (tmp_path / "outcomes.jsonl").read_text().splitlines()
        assert len(outcomes_text) == 2
        parsed = [OrderOutcome.model_validate_json(line) for line in outcomes_text]
        assert {o.order_id for o in parsed} == {"t1-e", "t1-x"}
        assert all(o.terminal_state.value == "filled" for o in parsed)

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert "seed" in manifest
        assert manifest["intent_count"] == 2

        out = capsys.readouterr().out
        assert "config:     primary" in out
        assert "outcomes:   2" in out

    def test_summary_lines_present(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        patched_source(both_sides_book())
        dbn = _dbn_file(tmp_path)
        intents = write_intents(tmp_path / "i.jsonl", round_trip_intents())
        assert (
            cli.main(_simulate_argv(dbn, intents, tmp_path, instrument_id=str(IID)))
            == 0
        )
        out = capsys.readouterr().out
        assert "fills:      2 fill events" in out
        assert "terminal:   filled=2" in out
        assert "seed=0 oco_cascade_cancels=0 adverse_fills=0" in out
        assert f"outcomes -> {tmp_path / 'outcomes.jsonl'}" in out
        assert f"manifest -> {tmp_path / 'manifest.json'}" in out

    def test_auto_detect_logs_percentage(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        events = both_sides_book(IID) + [
            be(
                MboAction.ADD,
                MboSide.BID,
                9,
                BID_PX,
                1,
                ts=B,
                seq=9,
                instrument_id=SPREAD_IID,
            )
        ]
        patched_source(events)
        dbn = _dbn_file(tmp_path)
        intents = write_intents(tmp_path / "i.jsonl", round_trip_intents())
        rc = cli.main(_simulate_argv(dbn, intents, tmp_path))
        assert rc == 0
        err = capsys.readouterr().err
        assert f"detected front-month instrument_id={IID}" in err
        assert "%" in err

    def test_config_optimistic(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        patched_source(both_sides_book())
        dbn = _dbn_file(tmp_path)
        intents = write_intents(tmp_path / "i.jsonl", round_trip_intents())
        rc = cli.main(
            _simulate_argv(
                dbn, intents, tmp_path, instrument_id=str(IID), config="optimistic"
            )
        )
        assert rc == 0
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["config"]["queue_model"] == "time_priority"
        assert "config:     optimistic" in capsys.readouterr().out

    def test_out_path_in_missing_dir_is_created(
        self, tmp_path: Path, patched_source
    ) -> None:
        patched_source(both_sides_book())
        dbn = _dbn_file(tmp_path)
        intents = write_intents(tmp_path / "i.jsonl", round_trip_intents())
        nested = tmp_path / "a" / "b" / "c"
        rc = cli.main(_simulate_argv(dbn, intents, nested, instrument_id=str(IID)))
        assert rc == 0
        assert (nested / "outcomes.jsonl").is_file()
        assert (nested / "manifest.json").is_file()

    def test_default_span_recorded_in_manifest(
        self, tmp_path: Path, patched_source
    ) -> None:
        patched_source(both_sides_book())
        dbn = _dbn_file(tmp_path)
        intents = write_intents(tmp_path / "i.jsonl", round_trip_intents())
        assert (
            cli.main(_simulate_argv(dbn, intents, tmp_path, instrument_id=str(IID)))
            == 0
        )
        pad = int(cli._DEFAULT_PAD_MINUTES * cli._NS_PER_MINUTE)
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["valid_intervals"] == [[B - pad, B + HOLD + pad + 1]]

    def test_pad_minutes_changes_the_span(self, tmp_path: Path, patched_source) -> None:
        patched_source(both_sides_book())
        dbn = _dbn_file(tmp_path)
        intents = write_intents(tmp_path / "i.jsonl", round_trip_intents())
        rc = cli.main(
            _simulate_argv(
                dbn, intents, tmp_path, instrument_id=str(IID), pad_minutes="10"
            )
        )
        assert rc == 0
        pad = 10 * cli._NS_PER_MINUTE
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["valid_intervals"] == [[B - pad, B + HOLD + pad + 1]]

    def test_explicit_interval_overrides_span(
        self, tmp_path: Path, patched_source
    ) -> None:
        patched_source(both_sides_book())
        dbn = _dbn_file(tmp_path)
        intents = write_intents(tmp_path / "i.jsonl", round_trip_intents())
        rc = cli.main(
            [
                *_simulate_argv(dbn, intents, tmp_path, instrument_id=str(IID)),
                "--interval",
                str(B - 1000),
                str(B + HOLD + 10 * HOLD),
            ]
        )
        assert rc == 0
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["valid_intervals"] == [[B - 1000, B + HOLD + 10 * HOLD]]

    def test_degraded_day_recorded(self, tmp_path: Path, patched_source) -> None:
        patched_source(both_sides_book())
        dbn = _dbn_file(tmp_path)
        intents = write_intents(tmp_path / "i.jsonl", round_trip_intents())
        rc = cli.main(
            [
                *_simulate_argv(dbn, intents, tmp_path, instrument_id=str(IID)),
                "--degraded-day",
                "2026-07-30",
                "--degraded-day",
                "2026-05-24",
            ]
        )
        assert rc == 0
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["degraded_days"] == ["2026-05-24", "2026-07-30"]  # sim sorted

    def test_degraded_day_bad_format_warns_but_records(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        patched_source(both_sides_book())
        dbn = _dbn_file(tmp_path)
        intents = write_intents(tmp_path / "i.jsonl", round_trip_intents())
        rc = cli.main(
            [
                *_simulate_argv(dbn, intents, tmp_path, instrument_id=str(IID)),
                "--degraded-day",
                "may 24th",
            ]
        )
        assert rc == 0
        assert "not YYYY-MM-DD" in capsys.readouterr().err
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["degraded_days"] == ["may 24th"]

    def test_rerun_is_idempotent(self, tmp_path: Path, patched_source) -> None:
        patched_source(both_sides_book())
        dbn = _dbn_file(tmp_path)
        intents = write_intents(tmp_path / "i.jsonl", round_trip_intents())
        argv = _simulate_argv(dbn, intents, tmp_path, instrument_id=str(IID))
        assert cli.main(argv) == 0
        first_outcomes = (tmp_path / "outcomes.jsonl").read_text()
        first_manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert cli.main(argv) == 0
        assert (tmp_path / "outcomes.jsonl").read_text() == first_outcomes
        second_manifest = json.loads((tmp_path / "manifest.json").read_text())
        for key in ("valid_intervals", "seed", "intent_count", "event_count"):
            assert second_manifest[key] == first_manifest[key]

    def test_verbose_emits_info_to_stderr(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        patched_source(both_sides_book())
        dbn = _dbn_file(tmp_path)
        intents = write_intents(tmp_path / "i.jsonl", round_trip_intents())
        argv = _simulate_argv(dbn, intents, tmp_path, instrument_id=str(IID))
        argv.insert(1, "-v")  # after the sub-command name
        rc = cli.main(argv)
        assert rc == 0
        err = capsys.readouterr().err
        assert "INFO" in err
        assert "src.ticksim.cli" in err


# --------------------------------------------------------------------------- #
# main(["simulate", ...]) -- I/O & edge-case matrix error rows
# --------------------------------------------------------------------------- #


class TestSimulateErrors:
    def test_auto_detect_tie_exit_1(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        events = [
            be(MboAction.ADD, MboSide.BID, 1, BID_PX, 1, ts=B, seq=1),
            be(
                MboAction.ADD,
                MboSide.ASK,
                2,
                ASK_PX,
                1,
                ts=B,
                seq=2,
                instrument_id=SPREAD_IID,
            ),
        ]
        patched_source(events)
        dbn = _dbn_file(tmp_path)
        intents = write_intents(tmp_path / "i.jsonl", round_trip_intents())
        rc = cli.main(_simulate_argv(dbn, intents, tmp_path))
        assert rc == 1
        assert "ambiguous" in capsys.readouterr().err

    def test_empty_dbn_stream_auto_detect_exit_1(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        patched_source([])
        dbn = _dbn_file(tmp_path)
        intents = write_intents(tmp_path / "i.jsonl", round_trip_intents())
        rc = cli.main(_simulate_argv(dbn, intents, tmp_path))
        assert rc == 1
        assert capsys.readouterr().err.strip()

    def test_empty_dbn_stream_explicit_iid_exit_1(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        patched_source([])
        dbn = _dbn_file(tmp_path)
        intents = write_intents(tmp_path / "i.jsonl", round_trip_intents())
        rc = cli.main(_simulate_argv(dbn, intents, tmp_path, instrument_id=str(IID)))
        assert rc == 1
        assert "no book events" in capsys.readouterr().err

    def test_missing_dbn_file_exit_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        intents = write_intents(tmp_path / "i.jsonl", round_trip_intents())
        rc = cli.main(
            _simulate_argv(
                tmp_path / "nope.dbn.zst", intents, tmp_path, instrument_id=str(IID)
            )
        )
        assert rc == 1
        assert "no such DBN file" in capsys.readouterr().err

    def test_malformed_intent_line_exit_1(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        patched_source(both_sides_book())
        dbn = _dbn_file(tmp_path)
        path = tmp_path / "i.jsonl"
        good = round_trip_intents()[0].model_dump_json()
        path.write_text(good + "\n{bad\n" + good + "\n")  # malformed line 2
        rc = cli.main(_simulate_argv(dbn, path, tmp_path, instrument_id=str(IID)))
        assert rc == 1
        assert "line 2" in capsys.readouterr().err

    def test_empty_intent_file_exit_1(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        patched_source(both_sides_book())
        dbn = _dbn_file(tmp_path)
        path = tmp_path / "i.jsonl"
        path.write_text("\n\n")
        rc = cli.main(_simulate_argv(dbn, path, tmp_path, instrument_id=str(IID)))
        assert rc == 1
        assert "no intents" in capsys.readouterr().err

    def test_non_causal_intent_log_exit_1(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        patched_source(both_sides_book())
        dbn = _dbn_file(tmp_path)
        cancel_first = OrderIntent(
            action=IntentAction.CANCEL,
            order_id="ghost",
            trade_id="t1",
            leg=Leg.ENTRY,
            kind=OrderKind.MARKETABLE,
            side=Side.BUY,
            size=1,
            submit_ts_ns=B,
        )
        path = write_intents(tmp_path / "i.jsonl", [cancel_first])
        rc = cli.main(_simulate_argv(dbn, path, tmp_path, instrument_id=str(IID)))
        assert rc == 1
        err = capsys.readouterr().err
        assert "simulator fault" in err and "SUBMIT" in err

    def test_multi_instrument_after_filter_exit_1(
        self,
        tmp_path: Path,
        patched_source,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # a FrontMonthSource bug / an unfiltered stream -> sim raises IntentLogError
        patched_source(both_sides_book())

        def _boom(*_a: object, **_k: object) -> object:
            raise IntentLogError("book-event stream carries >1 instrument_id (a, b)")

        monkeypatch.setattr(cli, "simulate", _boom)
        dbn = _dbn_file(tmp_path)
        intents = write_intents(tmp_path / "i.jsonl", round_trip_intents())
        rc = cli.main(_simulate_argv(dbn, intents, tmp_path, instrument_id=str(IID)))
        assert rc == 1
        assert "instrument_id" in capsys.readouterr().err

    def test_simulator_invariant_violation_exit_1(
        self,
        tmp_path: Path,
        patched_source,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        patched_source(both_sides_book())

        def _boom(*_a: object, **_k: object) -> object:
            raise InvariantViolation("no price improvement invariant broke")

        monkeypatch.setattr(cli, "simulate", _boom)
        dbn = _dbn_file(tmp_path)
        intents = write_intents(tmp_path / "i.jsonl", round_trip_intents())
        rc = cli.main(_simulate_argv(dbn, intents, tmp_path, instrument_id=str(IID)))
        assert rc == 1
        assert "invariant broke" in capsys.readouterr().err

    def test_book_inconsistency_exit_1(
        self,
        tmp_path: Path,
        patched_source,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from src.ticksim.book import BookInconsistency

        def _boom(*_a: object, **_k: object) -> object:
            raise BookInconsistency("persistent crossed book")

        monkeypatch.setattr(cli, "simulate", _boom)
        patched_source(both_sides_book())
        dbn = _dbn_file(tmp_path)
        intents = write_intents(tmp_path / "i.jsonl", round_trip_intents())
        rc = cli.main(_simulate_argv(dbn, intents, tmp_path, instrument_id=str(IID)))
        assert rc == 1
        assert "simulator fault" in capsys.readouterr().err

    def test_order_state_error_propagates_not_exit_1(
        self, tmp_path: Path, patched_source, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*_a: object, **_k: object) -> object:
            raise OrderStateError("illegal working->working transition")

        monkeypatch.setattr(cli, "simulate", _boom)
        patched_source(both_sides_book())
        dbn = _dbn_file(tmp_path)
        intents = write_intents(tmp_path / "i.jsonl", round_trip_intents())
        with pytest.raises(OrderStateError):
            cli.main(_simulate_argv(dbn, intents, tmp_path, instrument_id=str(IID)))

    def test_wiring_bug_propagates_not_exit_1(
        self, tmp_path: Path, patched_source, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*_a: object, **_k: object) -> object:
            raise KeyError("cli<->sim wiring bug")

        monkeypatch.setattr(cli, "simulate", _boom)
        patched_source(both_sides_book())
        dbn = _dbn_file(tmp_path)
        intents = write_intents(tmp_path / "i.jsonl", round_trip_intents())
        with pytest.raises(KeyError):
            cli.main(_simulate_argv(dbn, intents, tmp_path, instrument_id=str(IID)))

    def test_atomic_write_failure_leaves_nothing(
        self,
        tmp_path: Path,
        patched_source,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        patched_source(both_sides_book())
        real_write = cli._write_tmp

        def _selective(path: Path, text: str) -> None:
            if "manifest" in path.name:
                raise OSError("disk full")
            real_write(path, text)

        monkeypatch.setattr(cli, "_write_tmp", _selective)
        dbn = _dbn_file(tmp_path)
        intents = write_intents(tmp_path / "i.jsonl", round_trip_intents())
        rc = cli.main(_simulate_argv(dbn, intents, tmp_path, instrument_id=str(IID)))
        assert rc == 1
        assert "cannot write" in capsys.readouterr().err
        assert not (tmp_path / "outcomes.jsonl").exists()
        assert not (tmp_path / "manifest.json").exists()
        assert list(tmp_path.glob("*.tmp")) == []


# --------------------------------------------------------------------------- #
# main -- argparse validation (exit 2)
# --------------------------------------------------------------------------- #


class TestArgValidation:
    def test_negative_pad_minutes_exit_2(self) -> None:
        assert (
            cli.main(
                [
                    "simulate",
                    "--dbn",
                    "d",
                    "--intents",
                    "i",
                    "--out-outcomes",
                    "o",
                    "--out-manifest",
                    "m",
                    "--pad-minutes",
                    "-1",
                ]
            )
            == 2
        )

    def test_nonfinite_pad_minutes_exit_2(self) -> None:
        assert (
            cli.main(
                [
                    "simulate",
                    "--dbn",
                    "d",
                    "--intents",
                    "i",
                    "--out-outcomes",
                    "o",
                    "--out-manifest",
                    "m",
                    "--pad-minutes",
                    "inf",
                ]
            )
            == 2
        )

    def test_negative_instrument_id_exit_2(self) -> None:
        assert (
            cli.main(
                [
                    "simulate",
                    "--dbn",
                    "d",
                    "--intents",
                    "i",
                    "--out-outcomes",
                    "o",
                    "--out-manifest",
                    "m",
                    "--instrument-id",
                    "-5",
                ]
            )
            == 2
        )

    def test_backwards_interval_exit_2(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = cli.main(
            [
                "simulate",
                "--dbn",
                "d",
                "--intents",
                "i",
                "--out-outcomes",
                "o",
                "--out-manifest",
                "m",
                "--interval",
                "1000",
                "1000",
            ]
        )
        assert rc == 2
        assert "not a valid [start, end) window" in capsys.readouterr().err

    def test_out_paths_identical_exit_2(self, tmp_path: Path) -> None:
        same = str(tmp_path / "both.json")
        rc = cli.main(
            [
                "simulate",
                "--dbn",
                "d",
                "--intents",
                "i",
                "--out-outcomes",
                same,
                "--out-manifest",
                same,
            ]
        )
        assert rc == 2

    def test_output_collides_with_input_exit_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dbn = str(tmp_path / "w.dbn.zst")
        rc = cli.main(
            [
                "simulate",
                "--dbn",
                dbn,
                "--intents",
                str(tmp_path / "i.jsonl"),
                "--out-outcomes",
                dbn,
                "--out-manifest",
                str(tmp_path / "m.json"),
            ]
        )
        assert rc == 2
        assert "must not be the same file" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# main -- dispatcher / usage
# --------------------------------------------------------------------------- #


class TestDispatcher:
    def test_no_command_returns_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert cli.main([]) == 2
        assert capsys.readouterr().err.strip()

    def test_unknown_command_returns_2(self) -> None:
        assert cli.main(["frobnicate"]) == 2

    def test_missing_required_arg_returns_2(self) -> None:
        assert cli.main(["simulate", "--dbn", "x.dbn.zst"]) == 2

    def test_simulate_help_returns_0(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert cli.main(["simulate", "--help"]) == 0
        assert "simulate" in capsys.readouterr().out

    def test_all_exports(self) -> None:
        assert set(cli.__all__) == {"FrontMonthSource", "detect_front_month", "main"}
