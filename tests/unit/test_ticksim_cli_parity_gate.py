"""Unit tests for ``src/ticksim/cli.py`` -- the ``parity-gate`` sub-command.

Drives ``cli.main(["parity-gate", ...])`` against a monkeypatched in-memory
``cli.DbnMboSource``, a hand-built ``--windows`` JSON and the two broker-accurate
Part A sources: a tiny ``orders.csv`` and a tiny ``projectx_fills.json``
(``data/trades.db`` is no longer a Part A source -- its timestamps are bar
stamps, not fill times). The Part A N-floor and the Part B order-count floor are
patched down so the plumbing (reconstruction, per-leg routing, source wiring,
atomic write, exit codes, arg validation) is what is under test -- the numeric
verdicts have their own coverage in ``test_ticksim_parity_gate_cli.py`` /
``test_ticksim_parity_*``.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.ticksim import cli
from src.ticksim.config import MNQ_TICK_DBN
from src.ticksim.events import BookEvent, MboAction, MboSide

IID = 42004800
TICK = MNQ_TICK_DBN
P = 20_000_000_000_000
BID_PX = P - TICK
ASK_PX = P + TICK

ENTRY_ISO = "2026-06-22T14:00:00.000Z"
EXIT_ISO = "2026-06-22T14:05:00.000Z"


def _ns(iso: str) -> int:
    """ISO-8601 -> integer ns since the epoch (no float drift at ~1.7e18 ns)."""
    moment = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    delta = moment - dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    return (
        delta.days * 86_400 * 1_000_000_000
        + delta.seconds * 1_000_000_000
        + delta.microseconds * 1_000
    )


ENTRY_NS = _ns(ENTRY_ISO)
LO_NS = ENTRY_NS - 60_000_000_000
HI_NS = ENTRY_NS + 3_600_000_000_000


class ReiterableSource:
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


def clean_events() -> list[BookEvent]:
    base = ENTRY_NS
    return [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 1_000_000, base - 3, 1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 1_000_000, base - 2, 2),
        be(MboAction.ADD, MboSide.BID, 3, BID_PX - 10 * TICK, 5, base - 1, 3),
        be(MboAction.MODIFY, MboSide.BID, 3, BID_PX - 10 * TICK, 3, base + 1, 4),
        be(MboAction.TRADE, MboSide.NONE, 0, BID_PX, 1, base + 2, 5),
        be(MboAction.FILL, MboSide.BID, 1, BID_PX, 1, base + 3, 6),
        be(MboAction.CANCEL, MboSide.BID, 3, BID_PX - 10 * TICK, 3, base + 4, 7),
    ]


def flagged_events() -> list[BookEvent]:
    return [ev for ev in clean_events() if str(ev.action) not in ("T", "F")]


@pytest.fixture
def low_floors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the Part A N-floor + the Part B order-count floor down so a 1-trade
    / 25-order run can PASS."""
    monkeypatch.setattr("src.ticksim.parity.part_a.PART_A_MIN_N", 2)
    monkeypatch.setattr("src.ticksim.parity.part_b.PART_B_MIN_ORDERS", 10)


@pytest.fixture
def patched_source(monkeypatch: pytest.MonkeyPatch):
    def _set(events: list[BookEvent]) -> None:
        monkeypatch.setattr(cli, "DbnMboSource", lambda _p: ReiterableSource(events))

    return _set


def _dbn_file(tmp_path: Path) -> Path:
    path = tmp_path / "window.mbo.dbn.zst"
    path.write_bytes(b"")
    return path


def _windows_json(tmp_path: Path, dbn: Path, *, key: str = "w0") -> Path:
    path = tmp_path / "windows.json"
    path.write_text(
        json.dumps(
            {
                key: {
                    "dbn": str(dbn),
                    "instrument_id": IID,
                    "lo_ns": LO_NS,
                    "hi_ns": HI_NS,
                    "degraded_days": [],
                }
            }
        )
    )
    return path


ENTRY_DT = dt.datetime.fromisoformat(ENTRY_ISO.replace("Z", "+00:00"))


def _iso_at(seconds: int) -> str:
    """ISO-8601 stamp ``seconds`` after ``ENTRY_ISO`` (ProjectX export format)."""
    return (ENTRY_DT + dt.timedelta(seconds=seconds)).isoformat()


def _px_fill(
    *,
    order_id: object,
    at_seconds: int,
    side: int,
    price_dbn: int,
    pnl: float | None,
    size: int = 1,
) -> dict[str, object]:
    """One ProjectX fill object (``profitAndLoss is None`` == an opening fill)."""
    return {
        "id": 1_000_000 + at_seconds,
        "accountId": 26556101,
        "contractId": "CON.F.US.MNQ.U26",
        "creationTimestamp": _iso_at(at_seconds),
        "price": round(price_dbn / 1e9, 2),
        "profitAndLoss": pnl,
        "fees": 0.36,
        "commissions": 0.25,
        "side": side,
        "size": size,
        "voided": False,
        "orderId": order_id,
    }


def _projectx_json(
    tmp_path: Path,
    *,
    pairs: int = 1,
    bad_price: bool = False,
    unpaired_open: bool = False,
    fills: list[dict[str, object]] | None = None,
    name: str = "projectx_fills.json",
) -> Path:
    """A ProjectX fill export: ``pairs`` long round trips, buy @ ask -> sell @ bid.

    ``bad_price`` moves the opening fill 8 ticks away from the book's ask, so the
    simulator's fill is 8 ticks off the "real" one -> Part A FAIL.
    """
    path = tmp_path / name
    if fills is None:
        entry_px = ASK_PX - (8 * TICK if bad_price else 0)
        fills = []
        for i in range(pairs):
            open_at = i * 600
            fills.append(
                _px_fill(
                    order_id=900_000 + 2 * i,
                    at_seconds=open_at,
                    side=0,
                    price_dbn=entry_px,
                    pnl=None,
                )
            )
            fills.append(
                _px_fill(
                    order_id=900_001 + 2 * i,
                    at_seconds=open_at + 300,
                    side=1,
                    price_dbn=BID_PX,
                    pnl=-2.5,
                )
            )
        if unpaired_open:
            fills.append(
                _px_fill(
                    order_id=999_999,
                    at_seconds=pairs * 600,
                    side=0,
                    price_dbn=ASK_PX,
                    pnl=None,
                )
            )
    path.write_text(json.dumps(fills))
    return path


def _mim_nb_csv(tmp_path: Path) -> Path:
    """A one-round-trip mim-nb order-lifecycle CSV -> `mimnb-e1` trade."""
    path = tmp_path / "orders.csv"
    path.write_text(
        "ts_utc,event,order_id,otype,side,size,price\n"
        f"{ENTRY_ISO},PLACE,e1,2,0,1,{ASK_PX / 1e9:.2f}\n"
        f"{ENTRY_ISO},FILL,e1,2,0,1,{ASK_PX / 1e9:.2f}\n"
        f"{EXIT_ISO},PLACE,x1,2,1,1,\n"
        f"{EXIT_ISO},FILL,x1,2,1,1,{BID_PX / 1e9:.2f}\n"
    )
    return path


def _argv(
    windows: Path,
    out: Path,
    projectx: Path,
    *,
    orders_csv: Path | None = None,
    synthetic_window: str = "w0",
    synthetic_n: str = "25",
    extra: list[str] | None = None,
) -> list[str]:
    argv = [
        "parity-gate",
        "--projectx-fills",
        str(projectx),
        "--windows",
        str(windows),
        "--synthetic-window",
        synthetic_window,
        "--synthetic-n",
        synthetic_n,
        "--amendment-number",
        "7",
        "--cycle-number",
        "1",
        "--out",
        str(out),
    ]
    if orders_csv is not None:
        argv += ["--orders-csv", str(orders_csv)]
    else:
        # No CSV: these fixtures' ProjectX exports are yank-only by
        # construction, so the attribution guard is explicitly asserted rather
        # than tripped. The guard itself is covered by
        # `test_projectx_without_orders_csv_fails_closed`, which strips the flag.
        argv += [
            "--orders-csv",
            str(windows.parent / "no_such_orders.csv"),
            "--projectx-yank-only",
        ]
    return argv + (extra or [])


# --------------------------------------------------------------------------- #
# happy path -> stub written, exit 0
# --------------------------------------------------------------------------- #


def test_clean_run_writes_stub_exit_0(
    tmp_path: Path,
    patched_source,
    low_floors: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patched_source(clean_events())
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    out = tmp_path / "amendment.md"

    rc = cli.main(_argv(windows, out, _projectx_json(tmp_path)))
    assert rc == 0, capsys.readouterr()

    text = out.read_text()
    assert text.startswith("# Amendment 7 -- Parity gate result (cycle 1)")
    assert "**PASS**" in text
    stdout = capsys.readouterr().out
    assert "verdict:           PASS" in stdout
    assert "integrity_flagged: False" in stdout
    assert str(out) in stdout


# --------------------------------------------------------------------------- #
# FAIL -> exit 1
# --------------------------------------------------------------------------- #


def test_part_a_fail_exit_1(
    tmp_path: Path,
    patched_source,
    low_floors: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patched_source(clean_events())
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    out = tmp_path / "amendment.md"

    rc = cli.main(_argv(windows, out, _projectx_json(tmp_path, bad_price=True)))
    assert rc == 1
    assert "**FAIL**" in out.read_text()
    captured = capsys.readouterr()
    assert "reason:" in captured.err  # the verdict reason is echoed to stderr
    assert "miscalibrated" in captured.err


# --------------------------------------------------------------------------- #
# integrity FLAGGED but PASS -> exit 3
# --------------------------------------------------------------------------- #


def test_integrity_flagged_pass_exit_3(
    tmp_path: Path,
    patched_source,
    low_floors: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patched_source(flagged_events())
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    out = tmp_path / "amendment.md"

    rc = cli.main(_argv(windows, out, _projectx_json(tmp_path)))
    assert rc == 3, capsys.readouterr()
    text = out.read_text()
    assert "**PASS**" in text
    assert "integrity FLAGGED on window(s) w0" in text
    assert "exits 3" not in text and "exit 3" not in text  # no CLI-code narration
    assert "integrity_flagged: True" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# error matrix
# --------------------------------------------------------------------------- #


def test_reconstruction_error_exit_1(
    tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
) -> None:
    patched_source(clean_events())
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    orders = tmp_path / "orders.csv"
    # a FILL row on an order the lifecycle never PLACEd -> reconstruct_mim_nb raises
    orders.write_text(
        "ts_utc,event,order_id,otype,side,size,price\n"
        f"{ENTRY_ISO},FILL,o-ghost,2,0,1,20000.25\n"
    )
    out = tmp_path / "amendment.md"

    rc = cli.main(_argv(windows, out, _projectx_json(tmp_path), orders_csv=orders))
    assert rc == 1
    assert "error:" in capsys.readouterr().err
    assert not out.exists()


def test_missing_window_dbn_exit_1(
    tmp_path: Path, low_floors: None, capsys: pytest.CaptureFixture[str]
) -> None:
    # the .dbn.zst named in --windows does not exist
    windows = _windows_json(tmp_path, tmp_path / "absent.dbn.zst")
    out = tmp_path / "amendment.md"
    rc = cli.main(_argv(windows, out, _projectx_json(tmp_path)))
    assert rc == 1
    assert "no such DBN file" in capsys.readouterr().err


def test_synthetic_window_not_a_key_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    out = tmp_path / "amendment.md"
    rc = cli.main(
        _argv(windows, out, _projectx_json(tmp_path), synthetic_window="nope")
    )
    assert rc == 2
    assert "not a key in --windows" in capsys.readouterr().err


def test_frozen_sha_failure_no_sha_exit_1(
    tmp_path: Path,
    patched_source,
    low_floors: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patched_source(clean_events())
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    out = tmp_path / "amendment.md"
    monkeypatch.setenv("PATH", "")  # `git` unreachable -> gate.frozen_sha() fails
    rc = cli.main(_argv(windows, out, _projectx_json(tmp_path)))
    assert rc == 1
    assert not out.exists()  # never a SHA-less stub


def test_out_collides_with_input_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    rc = cli.main(_argv(windows, windows, _projectx_json(tmp_path)))
    assert rc == 2
    assert "--out must not be the same file" in capsys.readouterr().err


def test_out_collides_with_window_dbn_exit_2(tmp_path: Path) -> None:
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    rc = cli.main(_argv(windows, dbn, _projectx_json(tmp_path)))
    assert rc == 2


def test_help_exits_0() -> None:
    assert cli.main(["parity-gate", "--help"]) == 0


def test_non_positive_amendment_number_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    out = tmp_path / "amendment.md"
    argv = _argv(windows, out, _projectx_json(tmp_path))
    argv[argv.index("--amendment-number") + 1] = "0"
    rc = cli.main(argv)
    assert rc == 2
    assert "--amendment-number must be > 0" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# the two broker-accurate sources: wiring, de-dup, per-leg source provenance
# --------------------------------------------------------------------------- #


def test_both_sources_feed_part_a_per_trader_section(
    tmp_path: Path,
    patched_source,
    low_floors: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """orders.csv (mim-nb) + projectx_fills.json (yank) -> 4 scored legs, both
    traders in the per-trader table, every leg's source in the provenance note."""
    patched_source(clean_events())
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    out = tmp_path / "amendment.md"
    orders = _mim_nb_csv(tmp_path)

    rc = cli.main(_argv(windows, out, _projectx_json(tmp_path), orders_csv=orders))
    assert rc in (0, 3), capsys.readouterr()
    text = out.read_text()
    assert text.startswith("# Amendment 7 --")  # provenance appended, not prepended
    assert "| trader-mim-nb |" in text
    assert "| trader-yank |" in text
    # 1 mim-nb round trip + 1 yank round trip, split per leg -> N = 4
    assert "- sample N: 4" in text
    assert "## Part A fill sources" in text
    assert "`orders.csv` (2 legs): mimnb-e1#ENTRY, mimnb-e1#EXIT" in text
    assert "`projectx` (2 legs): yank-900000#ENTRY, yank-900000#EXIT" in text
    assert "trades.db" in text  # the note says why the DB is not a source


def test_projectx_fills_already_in_orders_csv_are_skipped(
    tmp_path: Path,
    patched_source,
    low_floors: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A ProjectX fill whose ``orderId`` is an ``orders.csv`` order is mim-nb's:
    already scored from the richer lifecycle ledger, so it must not be counted a
    second time as a yank leg."""
    patched_source(clean_events())
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    out = tmp_path / "amendment.md"
    orders = _mim_nb_csv(tmp_path)  # order ids e1 / x1
    projectx = _projectx_json(
        tmp_path,
        fills=[
            _px_fill(order_id="e1", at_seconds=0, side=0, price_dbn=ASK_PX, pnl=None),
            _px_fill(order_id="x1", at_seconds=300, side=1, price_dbn=BID_PX, pnl=-2.5),
        ],
    )

    rc = cli.main(_argv(windows, out, projectx, orders_csv=orders))
    assert rc in (0, 3), capsys.readouterr()
    text = out.read_text()
    assert "`orders.csv` (2 legs)" in text
    assert "`projectx`" not in text  # every ProjectX fill was a mim-nb order
    assert "| trader-yank |" not in text


def test_unpaired_trailing_open_is_dropped_not_fatal(
    tmp_path: Path,
    patched_source,
    low_floors: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The bot is still holding: the trailing opening fill has no close, so it is
    dropped + logged -- the closed round trip is still scored."""
    patched_source(clean_events())
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    out = tmp_path / "amendment.md"

    rc = cli.main(_argv(windows, out, _projectx_json(tmp_path, unpaired_open=True)))
    captured = capsys.readouterr()
    assert rc in (0, 3), captured
    assert "unpaired opening fill" in captured.err
    assert "`projectx` (2 legs)" in out.read_text()


def test_projectx_without_orders_csv_fails_closed(
    tmp_path: Path,
    patched_source,
    low_floors: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--projectx-fills without --orders-csv must fail closed, not mislabel.

    A ProjectX fill carries no trader field -- both bots trade the same
    contractId on the same accountId -- so orders.csv is the ONLY thing that
    separates them. Without it every mim-nb fill in the export is relabelled
    ``yank-*``, and the open->close pairing (which walks per contractId) lets
    the two bots' fills interleave and cross-pair. On the real 14-fill export
    that is silent: 7 "yank" trades, 5 of them actually mim-nb, and the sealed
    stub's per-trader breakdown is wrong. Regression for that.
    """
    patched_source(clean_events())
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    out = tmp_path / "amendment.md"

    argv = [
        a
        for a in _argv(windows, out, _projectx_json(tmp_path))
        if a != "--projectx-yank-only"
    ]
    rc = cli.main(argv)
    captured = capsys.readouterr()
    assert rc == 1, captured
    assert "carry no trader field" in captured.err
    assert not out.exists()  # no sealed artifact written from ambiguous input


def test_mixed_projectx_export_is_split_by_orders_csv(
    tmp_path: Path,
    patched_source,
    low_floors: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With orders.csv present, a ProjectX export holding BOTH bots' fills is
    attributed correctly: the mim-nb orderIds are filtered out of the yank set
    (they are scored from the richer CSV lifecycle) and both traders appear."""
    patched_source(clean_events())
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    out = tmp_path / "amendment.md"
    orders = _mim_nb_csv(tmp_path)

    # export = the mim-nb round trip (order_ids e1/x1, as in the CSV) PLUS a
    # genuine yank round trip -- exactly the real export's shape.
    mixed = [
        _px_fill(order_id="e1", at_seconds=0, side=0, price_dbn=ASK_PX, pnl=None),
        _px_fill(order_id="x1", at_seconds=300, side=1, price_dbn=BID_PX, pnl=-2.5),
        _px_fill(order_id=900_000, at_seconds=600, side=0, price_dbn=ASK_PX, pnl=None),
        _px_fill(order_id=900_001, at_seconds=900, side=1, price_dbn=BID_PX, pnl=-2.5),
    ]
    projectx = _projectx_json(tmp_path, fills=mixed)

    rc = cli.main(_argv(windows, out, projectx, orders_csv=orders))
    captured = capsys.readouterr()
    assert rc in (0, 1, 3), captured
    text = out.read_text()
    assert "| trader-mim-nb |" in text
    assert "| trader-yank |" in text
    # the mim-nb fills must NOT have been reconstructed a second time as yank
    assert "yank-e1" not in text


def test_stop_out_exit_timed_from_broker_record_reaches_the_stub(
    tmp_path: Path,
    patched_source,
    low_floors: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A mim-nb round trip whose exit is a FIRED otype-4 stop: the stub's Part A
    fill-sources section names the stop-out exit leg as timed from a broker
    record, and carries the "market leg timed by PLACE ts" note (§A8.2 cycle 2,
    verification-gap R1)."""
    patched_source(clean_events())
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    out = tmp_path / "amendment.md"

    stop_fill_iso = _iso_at(1800)  # +30 min, inside the window
    orders = tmp_path / "orders.csv"
    orders.write_text(
        "ts_utc,event,order_id,otype,side,size,price\n"
        f"{ENTRY_ISO},PLACE,e1,2,0,1,{ASK_PX / 1e9:.2f}\n"
        f"{ENTRY_ISO},PLACE,s1,4,1,1,{BID_PX / 1e9:.2f}\n"
        f"{_iso_at(3)},FILL,e1,2,0,1,{ASK_PX / 1e9:.2f}\n"
        f"{stop_fill_iso},FILL,s1,4,1,1,{BID_PX / 1e9:.2f}\n"
    )
    # ProjectX carries the true execution instant of the stop-out (orderId s1);
    # both orderIds are in orders.csv, so neither is (double-)counted as yank.
    projectx = _projectx_json(
        tmp_path,
        fills=[
            _px_fill(order_id="e1", at_seconds=0, side=0, price_dbn=ASK_PX, pnl=None),
            _px_fill(
                order_id="s1", at_seconds=1798, side=1, price_dbn=BID_PX, pnl=-100.0
            ),
        ],
    )

    rc = cli.main(_argv(windows, out, projectx, orders_csv=orders))
    captured = capsys.readouterr()
    assert rc in (0, 3), captured
    text = out.read_text()
    assert "## Part A fill sources" in text
    assert "market** leg is timed by its PLACE ts" in text
    assert "Stop-out exit legs timed from broker records" in text
    assert "mimnb-e1#EXIT" in text
    assert "| trader-yank |" not in text  # every ProjectX fill was a mim-nb order


def test_stop_out_exit_dropped_when_no_broker_ts_reaches_the_stub(
    tmp_path: Path,
    patched_source,
    low_floors: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The DROP branch (sibling of the timed branch above): a fired otype-4 stop
    whose orderId is NOT in the ProjectX export. The stub must disclose the
    dropped exit leg -- a sealed artifact silently omitting an un-graded real
    fill is the "restrict-to-favorable-subset without disclosure" failure the
    design is written to prevent (review round 2)."""
    patched_source(clean_events())
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    out = tmp_path / "amendment.md"

    orders = tmp_path / "orders.csv"
    orders.write_text(
        "ts_utc,event,order_id,otype,side,size,price\n"
        f"{ENTRY_ISO},PLACE,e1,2,0,1,{ASK_PX / 1e9:.2f}\n"
        f"{ENTRY_ISO},PLACE,s1,4,1,1,{BID_PX / 1e9:.2f}\n"
        f"{_iso_at(3)},FILL,e1,2,0,1,{ASK_PX / 1e9:.2f}\n"
        f"{_iso_at(1800)},FILL,s1,4,1,1,{BID_PX / 1e9:.2f}\n"
    )
    # ProjectX has the entry but NOT the stop-out order id -> no ts for s1.
    projectx = _projectx_json(
        tmp_path,
        fills=[
            _px_fill(order_id="e1", at_seconds=0, side=0, price_dbn=ASK_PX, pnl=None)
        ],
    )

    rc = cli.main(_argv(windows, out, projectx, orders_csv=orders))
    captured = capsys.readouterr()
    assert rc in (0, 1, 3), captured
    text = out.read_text()
    assert "Dropped, not graded" in text
    assert "s1" in text and "mimnb-e1" in text  # the trade + dropped exit named
    assert "mimnb-e1#ENTRY" in text  # entry leg kept and still graded


def test_yank_only_run_omits_the_mim_nb_timing_sentence(
    tmp_path: Path,
    patched_source,
    low_floors: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Matrix row 6: a yank-only Part A sample must NOT carry the "mim-nb market
    leg timed by its PLACE ts" sentence in the sealed stub (review round 2)."""
    patched_source(clean_events())
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    out = tmp_path / "amendment.md"
    # No orders.csv -> only yank legs from ProjectX.
    rc = cli.main(_argv(windows, out, _projectx_json(tmp_path)))
    assert rc in (0, 1, 3), capsys.readouterr()
    text = out.read_text()
    assert "timed by its PLACE ts" not in text
    assert "| trader-yank |" in text


def test_both_sources_absent_fails_on_the_n_floor(
    tmp_path: Path,
    patched_source,
    low_floors: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No CSV and no ProjectX file -> zero trades -> Part A FAIL (n == 0), never
    a silent PASS on no data."""
    patched_source(clean_events())
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    out = tmp_path / "amendment.md"

    rc = cli.main(_argv(windows, out, tmp_path / "no_such_projectx.json"))
    captured = capsys.readouterr()
    assert rc == 1, captured
    assert "no yank trades in this Part A sample" in captured.err
    text = out.read_text()
    assert "**FAIL**" in text
    assert "N=0" in text
    assert "Part A fill sources" not in text  # nothing to record


def test_malformed_projectx_json_exit_1(
    tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
) -> None:
    patched_source(clean_events())
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    out = tmp_path / "amendment.md"
    projectx = tmp_path / "projectx_fills.json"
    projectx.write_text("{not json")

    rc = cli.main(_argv(windows, out, projectx))
    assert rc == 1
    assert "not valid JSON" in capsys.readouterr().err
    assert not out.exists()


def test_projectx_close_without_open_exit_1(
    tmp_path: Path, patched_source, capsys: pytest.CaptureFixture[str]
) -> None:
    """A closing fill with no opening fill pending is a corrupt export -> handled
    error, never a guessed pairing."""
    patched_source(clean_events())
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    out = tmp_path / "amendment.md"
    projectx = _projectx_json(
        tmp_path,
        fills=[
            _px_fill(order_id=1, at_seconds=300, side=1, price_dbn=BID_PX, pnl=-2.5)
        ],
    )

    rc = cli.main(_argv(windows, out, projectx))
    assert rc == 1
    err = capsys.readouterr().err
    assert "yank reconstruction" in err
    assert "no preceding opening fill" in err
    assert not out.exists()


# --------------------------------------------------------------------------- #
# _ClippedSource drops out-of-window events
# --------------------------------------------------------------------------- #


def test_clipped_source_drops_out_of_window_events() -> None:
    lo, hi = 1_000, 2_000
    stamps = [lo - 1, lo, 1_500, hi - 1, hi, hi + 1]
    evs = [
        be(MboAction.ADD, MboSide.BID, i, BID_PX, 1, ts, i)
        for i, ts in enumerate(stamps)
    ]
    clipped = cli._ClippedSource(ReiterableSource(evs), lo, hi)
    got = [e.ts_event for e in clipped]
    assert got == [lo, 1_500, hi - 1]  # [lo, hi) half-open
    assert [e.ts_event for e in clipped] == got  # re-iterable


def test_clipped_source_shrinks_integrity_event_count(
    tmp_path: Path, patched_source, low_floors: None, capsys: pytest.CaptureFixture[str]
) -> None:
    # add 2 events outside [LO_NS, HI_NS); the integrity pass (surveys the whole
    # source) must not see them -- 7 in-window events, not 9.
    events = clean_events() + [
        be(MboAction.ADD, MboSide.BID, 90, BID_PX, 1, LO_NS - 1, 90),
        be(MboAction.ADD, MboSide.BID, 91, BID_PX, 1, HI_NS + 1, 91),
    ]
    patched_source(events)
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    out = tmp_path / "amendment.md"
    rc = cli.main(_argv(windows, out, _projectx_json(tmp_path)))
    assert rc in (0, 3), capsys.readouterr()
    text = out.read_text()
    assert "- events: 7 /" in text  # the 2 out-of-window events were clipped
    assert "- events: 9 /" not in text


# --------------------------------------------------------------------------- #
# degraded_days reaches the written stub
# --------------------------------------------------------------------------- #


def test_degraded_days_reaches_the_stub(
    tmp_path: Path, patched_source, low_floors: None, capsys: pytest.CaptureFixture[str]
) -> None:
    patched_source(clean_events())
    dbn = _dbn_file(tmp_path)
    windows = tmp_path / "windows.json"
    windows.write_text(
        json.dumps(
            {
                "w0": {
                    "dbn": str(dbn),
                    "instrument_id": IID,
                    "lo_ns": LO_NS,
                    "hi_ns": HI_NS,
                    "degraded_days": ["2026-07-30"],
                }
            }
        )
    )
    out = tmp_path / "amendment.md"
    rc = cli.main(_argv(windows, out, _projectx_json(tmp_path)))
    assert rc in (0, 3), capsys.readouterr()
    assert "degraded days: 2026-07-30" in out.read_text()


def test_window_key_with_template_marker_rejected_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dbn = _dbn_file(tmp_path)
    windows = tmp_path / "windows.json"
    windows.write_text(
        json.dumps(
            {
                "w0\n## H": {
                    "dbn": str(dbn),
                    "instrument_id": IID,
                    "lo_ns": LO_NS,
                    "hi_ns": HI_NS,
                }
            }
        )
    )
    out = tmp_path / "amendment.md"
    rc = cli.main(
        _argv(windows, out, _projectx_json(tmp_path), synthetic_window="w0\n## H")
    )
    assert rc == 2
    assert "newline or '## '" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# BrokenPipeError on the summary must not downgrade the exit code
# --------------------------------------------------------------------------- #


def test_broken_pipe_preserves_fail_exit_code(
    tmp_path: Path,
    patched_source,
    low_floors: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patched_source(clean_events())
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    out = tmp_path / "amendment.md"

    real_print = print

    def boom(*a: object, **k: object) -> None:
        if a and isinstance(a[0], str) and a[0].startswith("verdict:"):
            raise BrokenPipeError
        real_print(*a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr("builtins.print", boom)
    # neutralise the handler's fd redirect so it can't clobber pytest's capture
    monkeypatch.setattr(cli.os, "dup2", lambda *_a, **_k: 0)

    rc = cli.main(_argv(windows, out, _projectx_json(tmp_path, bad_price=True)))
    assert rc == 1  # FAIL preserved through the broken pipe, not a clean 0
    assert out.exists()  # stub was already written


# --------------------------------------------------------------------------- #
# --out overwrite guard
# --------------------------------------------------------------------------- #


def test_out_exists_refused_without_force_exit_2(
    tmp_path: Path, patched_source, low_floors: None, capsys: pytest.CaptureFixture[str]
) -> None:
    patched_source(clean_events())
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    out = tmp_path / "amendment.md"
    out.write_text("previous cycle stub\n")
    rc = cli.main(_argv(windows, out, _projectx_json(tmp_path)))
    assert rc == 2
    assert "already exists" in capsys.readouterr().err
    assert out.read_text() == "previous cycle stub\n"  # untouched


def test_out_exists_overwritten_with_force(
    tmp_path: Path, patched_source, low_floors: None, capsys: pytest.CaptureFixture[str]
) -> None:
    patched_source(clean_events())
    dbn = _dbn_file(tmp_path)
    windows = _windows_json(tmp_path, dbn)
    out = tmp_path / "amendment.md"
    out.write_text("previous cycle stub\n")
    rc = cli.main(_argv(windows, out, _projectx_json(tmp_path), extra=["--force"]))
    assert rc in (0, 3), capsys.readouterr()
    assert "previous cycle stub" not in out.read_text()


# --------------------------------------------------------------------------- #
# _source_for_factory memoises one source per window key
# --------------------------------------------------------------------------- #


def test_source_for_factory_opens_each_dbn_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_parity_gate`` calls ``source_for`` many times per key (once per Part
    A trade, again per unfilled leg, once for Part B, once for integrity). Without
    the per-key cache each call would spin up a fresh ``DBNStore.from_file``
    decompression pass over a ~0.5 GB/hour window."""
    opened: list[str] = []

    def _spy(path: str) -> ReiterableSource:
        opened.append(path)
        return ReiterableSource(clean_events())

    monkeypatch.setattr(cli, "DbnMboSource", _spy)

    dbn_a = tmp_path / "wA.mbo.dbn.zst"
    dbn_b = tmp_path / "wB.mbo.dbn.zst"
    dbn_a.write_bytes(b"")
    dbn_b.write_bytes(b"")
    entries = {
        "wA": cli._WindowEntry(str(dbn_a), IID, LO_NS, HI_NS, ()),
        "wB": cli._WindowEntry(str(dbn_b), IID, LO_NS, HI_NS, ()),
    }
    source_for = cli._source_for_factory(entries)

    first_a = source_for("wA")
    for _ in range(9):
        assert source_for("wA") is first_a  # same object, no re-open
    source_for("wB")
    source_for("wB")

    assert opened == [str(dbn_a), str(dbn_b)]  # exactly one open per key
    # the cached source is still re-iterable (spine AD-18) -- a re-fold must
    # yield the same events, not an exhausted iterator.
    assert list(first_a) == list(first_a) != []


def test_source_for_factory_missing_dbn_raises_cli_error(tmp_path: Path) -> None:
    entries = {
        "w0": cli._WindowEntry(str(tmp_path / "gone.zst"), IID, LO_NS, HI_NS, ())
    }
    with pytest.raises(cli._CliError, match="no such DBN file"):
        cli._source_for_factory(entries)("w0")


# --------------------------------------------------------------------------- #
# a BOM'd orders.csv reconstructs (utf-8-sig)
# --------------------------------------------------------------------------- #


def test_orders_csv_with_utf8_bom_is_reconstructed(
    tmp_path: Path, patched_source, low_floors: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """An ``orders.csv`` saved by Excel carries a UTF-8 BOM; read as plain
    ``utf-8`` the first header becomes ``"\\ufeffts_utc"`` and
    ``reconstruct_mim_nb`` fails with a misleading missing-column error."""
    patched_source(clean_events())
    csv_path = _mim_nb_csv(tmp_path)
    csv_path.write_bytes(b"\xef\xbb\xbf" + csv_path.read_bytes())
    assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")

    out = tmp_path / "amendment.md"
    rc = cli.main(
        _argv(
            _windows_json(tmp_path, _dbn_file(tmp_path)),
            out,
            _projectx_json(tmp_path),
            orders_csv=csv_path,
        )
    )
    captured = capsys.readouterr()
    assert rc != 2, captured
    assert "mim-nb reconstruction" not in captured.err  # not a header/read fault
    # the real proof: the stub was written and carries the reconstructed trade.
    assert "trader-mim-nb" in out.read_text()
