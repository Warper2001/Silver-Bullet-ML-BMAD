"""Unit tests for ``src/ticksim/cli.py`` -- the ``report`` sub-command.

The PRIMARY + OPTIMISTIC pair the ``report`` command consumes is built by
running ``cli.main(["simulate", ...])`` twice against a monkeypatched in-memory
``cli.DbnMboSource`` (a two-sided book + a marketable round-trip intent log),
writing four files to ``tmp_path``. Those four files are then fed to
``cli.main(["report", ...])``. Every row of the spec's I/O & Edge-Case matrix is
exercised for its stated exit code + stderr message.
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
    Fill,
    IntentAction,
    Leg,
    OrderIntent,
    OrderKind,
    OrderOutcome,
    Side,
    TerminalState,
)
from src.ticksim.report import TICK_VALUE_CENTS

# --------------------------------------------------------------------------- #
# builders (mirror test_ticksim_cli.py)
# --------------------------------------------------------------------------- #

IID = 42004800
TICK = MNQ_TICK_DBN
P = 20_000_000_000_000
BID_PX = P - TICK
ASK_PX = P + TICK
B = 1_700_000_000 * 1_000_000_000
HOLD = 600 * 1_000_000_000


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


def both_sides_book() -> list[BookEvent]:
    return [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 100, ts=B, seq=1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 100, ts=B, seq=2),
    ]


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


def entry_only_intents() -> list[OrderIntent]:
    """An entry that fills but no exit -> one OpenPosition, zero round trips."""
    return [marketable("t1-e", Leg.ENTRY, Side.BUY, B)]


class ReiterableSource:
    class_rank = 0

    def __init__(self, events: list[BookEvent]) -> None:
        self._events = list(events)

    def __iter__(self) -> Iterator[BookEvent]:
        return iter(self._events)


@pytest.fixture
def patched_source(monkeypatch: pytest.MonkeyPatch):
    def _set(events: list[BookEvent]) -> None:
        monkeypatch.setattr(cli, "DbnMboSource", lambda _path: ReiterableSource(events))

    return _set


def _dbn_file(tmp_path: Path) -> Path:
    path = tmp_path / "window.mbo.dbn.zst"
    path.write_bytes(b"")
    return path


def _make_pair(
    tmp_path: Path,
    patched_source,
    *,
    intent_log: list[OrderIntent] | None = None,
) -> dict[str, Path]:
    """Run ``simulate`` twice (primary + optimistic) and return the 4 paths."""
    patched_source(both_sides_book())
    dbn = _dbn_file(tmp_path)
    intents = tmp_path / "intents.jsonl"
    intents.write_text(
        "".join(
            i.model_dump_json() + "\n"
            for i in (intent_log if intent_log is not None else round_trip_intents())
        )
    )
    paths = {
        "primary_outcomes": tmp_path / "primary_outcomes.jsonl",
        "primary_manifest": tmp_path / "primary_manifest.json",
        "optimistic_outcomes": tmp_path / "optimistic_outcomes.jsonl",
        "optimistic_manifest": tmp_path / "optimistic_manifest.json",
    }
    for config, oc_key, mf_key in (
        ("primary", "primary_outcomes", "primary_manifest"),
        ("optimistic", "optimistic_outcomes", "optimistic_manifest"),
    ):
        rc = cli.main(
            [
                "simulate",
                "--dbn",
                str(dbn),
                "--intents",
                str(intents),
                "--config",
                config,
                "--instrument-id",
                str(IID),
                "--out-outcomes",
                str(paths[oc_key]),
                "--out-manifest",
                str(paths[mf_key]),
            ]
        )
        assert rc == 0, f"simulate --config {config} failed"
    return paths


def _report_argv(paths: dict[str, Path], out: Path, **override: Path) -> list[str]:
    merged = {**paths, **override}
    return [
        "report",
        "--primary-outcomes",
        str(merged["primary_outcomes"]),
        "--primary-manifest",
        str(merged["primary_manifest"]),
        "--optimistic-outcomes",
        str(merged["optimistic_outcomes"]),
        "--optimistic-manifest",
        str(merged["optimistic_manifest"]),
        "--out",
        str(out),
    ]


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #


class TestReportHappyPath:
    def test_writes_report_json_and_exits_0(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = _make_pair(tmp_path, patched_source)
        out = tmp_path / "report.json"
        capsys.readouterr()  # drop the simulate summaries

        rc = cli.main(_report_argv(paths, out))
        assert rc == 0

        doc = json.loads(out.read_text())
        assert {"round_trips", "primary", "stressed", "optimistic"} <= doc.keys()
        for model in ("primary", "stressed", "optimistic"):
            assert "n" in doc[model] and "net_cents" in doc[model]
        assert len(doc["round_trips"]) == 1
        assert doc["primary"]["n"] == 1

    def test_net_cents_arithmetic_and_stress_transform(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # both_sides_book(): a marketable BUY entry fills at the ask (P + TICK),
        # a marketable SELL exit fills at the bid (P - TICK) -> a 2-tick loss
        # (-100 cents) minus the $1.30 round-turn fee = -230 cents.
        paths = _make_pair(tmp_path, patched_source)
        out = tmp_path / "report.json"
        capsys.readouterr()
        assert cli.main(_report_argv(paths, out)) == 0

        doc = json.loads(out.read_text())
        (rt,) = doc["round_trips"]
        assert rt["matched_size"] == 1
        assert rt["net_primary_cents"] == -2 * TICK_VALUE_CENTS - (72 + 58)
        assert (
            rt["net_stressed_cents"]
            == rt["net_primary_cents"] - 2 * TICK_VALUE_CENTS * rt["matched_size"]
        )
        assert doc["primary"]["net_cents"] == [rt["net_primary_cents"]]
        assert doc["primary"]["sum_net_cents"] == rt["net_primary_cents"]
        assert doc["stressed"]["net_cents"] == [rt["net_stressed_cents"]]

    def test_stdout_summary_lines(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = _make_pair(tmp_path, patched_source)
        out = tmp_path / "report.json"
        capsys.readouterr()

        assert cli.main(_report_argv(paths, out)) == 0
        text = capsys.readouterr().out
        assert "round trips: 1" in text
        assert "incomplete: 0 open positions" in text
        assert "partially closed: 0" in text
        assert "optimistic-only completed: 0" in text
        assert "primary     n=1" in text
        assert "stressed    n=1" in text
        assert "optimistic  n=1" in text
        assert "win_rate=" in text and "profit_factor=" in text
        assert "optimistic n=1 (primary n=1)" in text
        # both models completed the same trades -> no both-completed-subset note
        assert "note: optimistic P&L is over" not in text
        assert f"report -> {out}" in text

    def test_out_path_in_missing_dir_is_created(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = _make_pair(tmp_path, patched_source)
        out = tmp_path / "nope" / "deeper" / "r.json"
        capsys.readouterr()
        assert cli.main(_report_argv(paths, out)) == 0
        assert out.is_file()

    def test_rerun_overwrites(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = _make_pair(tmp_path, patched_source)
        out = tmp_path / "report.json"
        capsys.readouterr()
        assert cli.main(_report_argv(paths, out)) == 0
        first = out.read_text()
        assert cli.main(_report_argv(paths, out)) == 0
        assert out.read_text() == first
        assert list(tmp_path.glob("*.tmp")) == []

    def test_zero_round_trip_pair_exits_0(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = _make_pair(tmp_path, patched_source, intent_log=entry_only_intents())
        out = tmp_path / "report.json"
        capsys.readouterr()
        assert cli.main(_report_argv(paths, out)) == 0

        doc = json.loads(out.read_text())
        assert doc["round_trips"] == []
        for model in ("primary", "stressed", "optimistic"):
            assert doc[model]["n"] == 0
            assert doc[model]["profit_factor"] is None  # inf/None -> null in JSON
            assert doc[model]["mean_net_cents"] is None

        text = capsys.readouterr().out
        assert "round trips: 0" in text
        assert "incomplete: 1 open position" in text
        for name in ("primary", "stressed", "optimistic"):
            assert f"{name:<11} n=0 " in text
        assert "win_rate=n/a" in text
        assert "profit_factor=n/a" in text

    def test_open_position_reported(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = _make_pair(tmp_path, patched_source, intent_log=entry_only_intents())
        out = tmp_path / "report.json"
        capsys.readouterr()
        assert cli.main(_report_argv(paths, out)) == 0

        doc = json.loads(out.read_text())
        assert len(doc["incomplete"]) == 1
        assert doc["incomplete"][0]["open_size"] == 1
        assert len(doc["round_trips"]) == 0
        assert "incomplete: 1 open position" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# I/O & Edge-Case matrix -- error rows
# --------------------------------------------------------------------------- #


class TestReportErrors:
    def test_malformed_outcome_line_exit_1(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = _make_pair(tmp_path, patched_source)
        good = paths["primary_outcomes"].read_text().splitlines()
        paths["primary_outcomes"].write_text(
            good[0] + "\n" + good[1] + "\n{ not valid json\n"
        )
        capsys.readouterr()
        rc = cli.main(_report_argv(paths, tmp_path / "r.json"))
        assert rc == 1
        assert "line 3" in capsys.readouterr().err

    def test_empty_outcome_file_exit_1(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = _make_pair(tmp_path, patched_source)
        paths["optimistic_outcomes"].write_text("\n   \n")
        capsys.readouterr()
        rc = cli.main(_report_argv(paths, tmp_path / "r.json"))
        assert rc == 1
        assert "no outcomes" in capsys.readouterr().err

    def test_manifest_not_an_object_exit_1(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = _make_pair(tmp_path, patched_source)
        paths["primary_manifest"].write_text("[1, 2, 3]")
        capsys.readouterr()
        rc = cli.main(_report_argv(paths, tmp_path / "r.json"))
        assert rc == 1
        assert "JSON object" in capsys.readouterr().err

    def test_manifest_unreadable_exit_1(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = _make_pair(tmp_path, patched_source)
        paths["primary_manifest"].unlink()
        capsys.readouterr()
        rc = cli.main(_report_argv(paths, tmp_path / "r.json"))
        assert rc == 1
        assert "cannot read manifest" in capsys.readouterr().err

    def test_outcomes_unreadable_exit_1(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = _make_pair(tmp_path, patched_source)
        paths["primary_outcomes"].unlink()
        capsys.readouterr()
        rc = cli.main(_report_argv(paths, tmp_path / "r.json"))
        assert rc == 1
        assert "cannot read outcome log" in capsys.readouterr().err

    def test_non_utf8_outcomes_exit_1(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = _make_pair(tmp_path, patched_source)
        paths["primary_outcomes"].write_bytes(b"\xff\xfe not utf-8 \x80\n")
        capsys.readouterr()
        rc = cli.main(_report_argv(paths, tmp_path / "r.json"))
        assert rc == 1
        assert "cannot read outcome log" in capsys.readouterr().err

    def test_manifest_not_valid_json_exit_1(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = _make_pair(tmp_path, patched_source)
        paths["optimistic_manifest"].write_text("{ not json")
        capsys.readouterr()
        rc = cli.main(_report_argv(paths, tmp_path / "r.json"))
        assert rc == 1
        assert "not valid JSON" in capsys.readouterr().err

    def test_not_a_primary_optimistic_pair_exit_1(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = _make_pair(tmp_path, patched_source)
        capsys.readouterr()
        # copy the PRIMARY manifest over a distinct path and pass it as
        # --optimistic-manifest: the two paths clear the distinctness gate, but
        # build_report sees (primary=back_of_queue, optimistic=back_of_queue)
        # and raises ReportError.
        dupe = tmp_path / "primary_manifest_copy.json"
        dupe.write_text(paths["primary_manifest"].read_text())
        rc = cli.main(
            _report_argv(paths, tmp_path / "r.json", optimistic_manifest=dupe)
        )
        assert rc == 1
        assert "swapped or wrong configs" in capsys.readouterr().err

    def test_duplicate_order_id_exit_1(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = _make_pair(tmp_path, patched_source)
        lines = paths["primary_outcomes"].read_text().splitlines()
        paths["primary_outcomes"].write_text("\n".join(lines + [lines[0]]) + "\n")
        capsys.readouterr()
        rc = cli.main(_report_argv(paths, tmp_path / "r.json"))
        assert rc == 1
        assert "duplicate order_id" in capsys.readouterr().err

    def test_mixed_entry_sides_exit_1(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = _make_pair(tmp_path, patched_source)
        # two FILLED entry legs of the same trade_id on opposite sides
        bad = [
            OrderOutcome(
                trade_id="t1",
                leg=Leg.ENTRY,
                order_id="t1-e",
                kind=OrderKind.MARKETABLE,
                side=Side.BUY,
                submit_ts_ns=B,
                arrival_ts_ns=B,
                terminal_state=TerminalState.FILLED,
                fills=(Fill(px_dbn=ASK_PX, size=1, ts_ns=B + 1),),
            ),
            OrderOutcome(
                trade_id="t1",
                leg=Leg.ENTRY,
                order_id="t1-e2",
                kind=OrderKind.MARKETABLE,
                side=Side.SELL,
                submit_ts_ns=B,
                arrival_ts_ns=B,
                terminal_state=TerminalState.FILLED,
                fills=(Fill(px_dbn=BID_PX, size=1, ts_ns=B + 1),),
            ),
        ]
        paths["primary_outcomes"].write_text(
            "".join(o.model_dump_json() + "\n" for o in bad)
        )
        capsys.readouterr()
        rc = cli.main(_report_argv(paths, tmp_path / "r.json"))
        assert rc == 1
        assert "mixed side" in capsys.readouterr().err

    def test_out_collides_with_input_exit_2(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = _make_pair(tmp_path, patched_source)
        capsys.readouterr()
        rc = cli.main(_report_argv(paths, paths["primary_manifest"]))
        assert rc == 2
        assert "must be different paths" in capsys.readouterr().err

    def test_out_collides_with_primary_outcomes_exit_2(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = _make_pair(tmp_path, patched_source)
        capsys.readouterr()
        rc = cli.main(_report_argv(paths, paths["primary_outcomes"]))
        assert rc == 2

    def test_input_input_collision_exit_2(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = _make_pair(tmp_path, patched_source)
        capsys.readouterr()
        rc = cli.main(
            _report_argv(
                paths,
                tmp_path / "r.json",
                optimistic_outcomes=paths["primary_outcomes"],
            )
        )
        assert rc == 2
        assert "must be different paths" in capsys.readouterr().err

    def test_out_unwritable_exit_1(
        self,
        tmp_path: Path,
        patched_source,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        paths = _make_pair(tmp_path, patched_source)

        def _boom(path: Path, text: str) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(cli, "_write_tmp", _boom)
        out = tmp_path / "r.json"
        capsys.readouterr()
        rc = cli.main(_report_argv(paths, out))
        assert rc == 1
        assert "cannot write output" in capsys.readouterr().err
        assert not out.exists()
        assert list(tmp_path.glob("*.tmp")) == []

    def test_missing_required_arg_exit_2(self) -> None:
        assert (
            cli.main(
                [
                    "report",
                    "--primary-outcomes",
                    "po",
                    "--primary-manifest",
                    "pm",
                    "--optimistic-outcomes",
                    "oo",
                    "--out",
                    "r.json",
                ]
            )
            == 2
        )

    def test_report_help_exit_0(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert cli.main(["report", "--help"]) == 0
        assert "report" in capsys.readouterr().out

    def test_verbose_emits_info_to_stderr(
        self, tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = _make_pair(tmp_path, patched_source)
        capsys.readouterr()
        argv = _report_argv(paths, tmp_path / "r.json")
        argv.insert(1, "-v")  # after the sub-command name
        assert cli.main(argv) == 0
        err = capsys.readouterr().err
        assert "INFO" in err
        assert "src.ticksim.cli" in err


# --------------------------------------------------------------------------- #
# import-edge / exports
# --------------------------------------------------------------------------- #


def test_cli_exports_unchanged() -> None:
    assert set(cli.__all__) == {"FrontMonthSource", "detect_front_month", "main"}
