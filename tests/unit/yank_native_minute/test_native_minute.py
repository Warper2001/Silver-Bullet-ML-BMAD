"""Native-format, timing, publication and frozen-adapter contract checks."""

import copy
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "minute_cli", ROOT / "src/cli/check_yank_native_minute.py"
)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
runner = cli.load_runner()
b = sys.modules["_yank_native_minute.builder"]
r = sys.modules["_yank_native_minute.replay"]


def record(action="T", recv=None, event=None, price=20_000_000_000, size=2, flags=128):
    a = np.zeros(1, dtype=b.DTYPE)
    values = dict(
        length=14,
        rtype=160,
        publisher=1,
        id=b.INSTRUMENT,
        action=ord(action),
        recv=recv if recv is not None else b.START + 1,
        event=event if event is not None else b.START,
        price=price,
        size=size,
        flags=flags,
        seq=9,
    )
    for k, v in values.items():
        a[k] = v
    return a


def build(records, partitions=None):
    data = np.concatenate(records)
    builder = b.Builder()
    offset = 0
    for size in partitions or [len(data)]:
        builder.consume(data[offset : offset + size], "synthetic", offset)
        offset += size
    assert offset == len(data)
    bars, delayed = builder.finish()
    return builder, bars, delayed


def test_trade_fill_snapshot_none_and_zero_volume():
    builder, bars, _ = build(
        [record(), record("F"), record(flags=32 | 128), record("N"), record(size=0)]
    )
    assert bars[0]["ohlcv"] == [20_000_000_000] * 4 + [2]
    assert bars[0]["trade_count"] == 2
    assert builder.counts["F_records"] == 1
    assert builder.counts["snapshot_records"] == 1
    assert builder.counts["zero_size_T_records"] == 1
    zero, _, _ = build([record(size=0)])
    assert b.coverage(zero, [])[0]["coverage"] == "TRADED"
    assert b.coverage(zero, [])[1]["ohlcv"] is None


@pytest.mark.parametrize("partition", [[4], [1, 3], [2, 2], [1, 1, 1, 1]])
def test_complete_event_cross_chunks_and_minute(partition):
    t = b.START + b.MINUTE - 1
    records = [
        record(recv=t, flags=0),
        record("F", recv=t, flags=0),
        record("A", recv=t + 3),
        record(recv=t + 4),
    ]
    builder, bars, delayed = build(records, partition)
    assert bars[0]["availability_ns"] == t + 3
    assert bars[0]["completion_ref"]["record_index"] == 2
    assert len(delayed) == 1
    assert runner.canonical(bars) == runner.canonical(build(records)[1])


def test_native_order_ohlc_not_exchange_sort():
    _, bars, _ = build(
        [
            record(event=b.START + 3, price=21_000_000_000),
            record(event=b.START + 1, price=19_000_000_000),
        ]
    )
    assert bars[0]["ohlcv"] == [
        21_000_000_000,
        21_000_000_000,
        19_000_000_000,
        19_000_000_000,
        4,
    ]


@pytest.mark.parametrize(
    "changes", [dict(event=0), dict(recv=(1 << 64) - 1), dict(flags=128 | 8)]
)
def test_invalid_trade_time_holds(changes):
    builder, bars, _ = build([record(**changes)])
    assert "INVALID_TRADE_TIMESTAMP" in builder.holds
    assert not bars


def test_bad_earlier_terminator_cannot_hide_behind_good_one():
    builder, bars, _ = build([record(flags=0), record("F", flags=128 | 8), record()])
    assert "INVALID_TRADE_EVENT_TERMINATOR_TIME" in builder.holds
    assert bars[0]["incomplete_event"]


def test_capture_regression_across_chunks():
    builder, _, _ = build([record(recv=b.START + 2), record(recv=b.START + 1)], [1, 1])
    assert "NONMONOTONE_NATIVE_CAPTURE" in builder.holds


def test_pending_terminator_remains_unassessable(tmp_path):
    builder = b.Builder()
    builder.consume(record(flags=0), "synthetic", 0)
    assert builder.pending == [b.START]
    bars, _ = builder.finish()
    assert "MISSING_TRADE_EVENT_TERMINATOR" in builder.holds
    assert bars[0]["incomplete_event"]


@pytest.mark.parametrize(
    "raw",
    [
        b"DBN",
        b"DBN\x02" + struct.pack("<I", 100),
        b"DBN\x03" + struct.pack("<I", 5),
        b"DBN\x03" + struct.pack("<I", 100) + b"x",
    ],
)
def test_metadata_corruption(raw):
    with pytest.raises(ValueError):
        b.metadata(io.BytesIO(raw), "mbo", "20250519")


def test_truncated_fixed_record_and_arbitrary_byte_reads():
    with pytest.raises(ValueError):
        list(b.chunks(io.BytesIO(record().tobytes()[:-1])))

    class ShortRead(io.BytesIO):
        def read(self, n=-1):
            return super().read(min(n, 13))

    assert (
        b"".join(a.tobytes() for a in b.chunks(ShortRead(record().tobytes()), 1))
        == record().tobytes()
    )


@pytest.mark.parametrize(
    "field,value",
    [("length", 13), ("rtype", 1), ("id", 1), ("action", ord("Z")), ("price", 1)],
)
def test_invalid_schema_action_price(field, value):
    a = record()
    a[field] = value
    with pytest.raises(ValueError):
        build([a])


def test_coverage_status_transition_and_unknown():
    builder, _, _ = build([record()])
    builder.end = b.START + 3 * b.MINUTE
    status = dict(
        ts_recv_ns=b.START + b.MINUTE,
        ts_event_ns=b.START,
        is_trading=False,
        file="status",
        record_index=0,
        initial_state=False,
    )
    rows = b.coverage(builder, [status])
    assert rows[0]["status_at_end"] is None
    assert rows[1]["coverage"] == "NO_TRADE_OBSERVED_NONTRADING"
    assert rows[1]["status_transitions"] == [status]
    assert rows[2]["status_transitions"] == []


def test_publisher_collision_and_byte_determinism(tmp_path):
    artifacts = {"a.json": runner.canonical({"z": 1, "a": 2})}
    for name in ("a", "b"):
        runner.publish(tmp_path / name, artifacts)
    assert [(p.name, p.read_bytes()) for p in sorted((tmp_path / "a").iterdir())] == [
        (p.name, p.read_bytes()) for p in sorted((tmp_path / "b").iterdir())
    ]
    with pytest.raises(ValueError):
        runner.publish(tmp_path / "a", artifacts)


def test_publisher_interrupt_no_success_residue(tmp_path, monkeypatch):
    def interrupt(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(runner, "no_replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.publish(tmp_path / "result", {"a": b"x"})
    assert list(tmp_path.iterdir()) == []


def test_publisher_racing_empty_directory_preserved(tmp_path, monkeypatch):
    rename = runner.no_replace

    def race(source, target):
        target.mkdir()
        rename(source, target)

    monkeypatch.setattr(runner, "no_replace", race)
    with pytest.raises(OSError):
        runner.publish(tmp_path / "result", {"a": b"x"})
    assert list((tmp_path / "result").iterdir()) == []
    assert len(list(tmp_path.iterdir())) == 1


def test_source_and_model_hash_failure(tmp_path, monkeypatch):
    pins = json.loads((runner.HERE / "pins.json").read_text())
    pins["sources"]["src/research/yank_signals/model.py"] = "0" * 64
    with pytest.raises(ValueError, match="source hash"):
        r.load_frozen(pins)
    fake = tmp_path / "model"
    fake.write_bytes(b"tamper")
    monkeypatch.setattr(runner, "MODEL", fake)
    with pytest.raises(ValueError, match="model pin"):
        runner.verify_pins(
            {"files": [], "sources": {}, "model_sha256": "0" * 64, "versions": {}}
        )


def test_frozen_config_cash_and_h1_m15_boundaries():
    pins = json.loads((runner.HERE / "pins.json").read_text())
    engine, model, reference, reconcile = r.load_frozen(pins)
    from datetime import datetime, timezone, timedelta

    start = datetime(2025, 5, 19, 0, 0, tzinfo=timezone.utc)
    bars = [
        engine.Bar(start + timedelta(minutes=i), 100, 101, 99, 100, 2)
        for i in range(121)
    ]
    result = engine.ReplayEngine().run(bars)
    assert reconcile(result)["passed"] is True
    assert engine.ReplayEngine().config is reference.FROZEN_CONFIG
    assert reference.FROZEN_CONFIG.max_pending_bars == 240
    assert reference.lr_regime([1.0] * 1949) == "WARMUP"
    assert all(
        e["completed_timestamp"] < e["timestamp"]
        for e in result["events"]
        if e["kind"] in ("H1", "M15")
    )
    result["summary"]["cash"] = "49999"
    with pytest.raises(ValueError, match="cash"):
        reconcile(result)


def test_hold_prevents_frozen_import(monkeypatch):
    monkeypatch.setattr(r, "load_frozen", lambda _: pytest.fail("unsafe replay loaded"))
    assert r.replay([], {}, {"MISSING_TRADE_EVENT_TERMINATOR"})["outcome"] == "HOLD"


def test_cli_only_output_argument():
    with pytest.raises(SystemExit):
        cli.main(["--input", "other"])


def synthetic_native_bars():
    pins = json.loads((runner.HERE / "pins.json").read_text())
    engine, _, _, _ = r.load_frozen(pins)
    import ast
    from datetime import datetime, timedelta, timezone

    tree = ast.parse(
        (r.REPLAY_ROOT / "src/research/yank_signals/evidence.py").read_bytes()
    )
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "synthetic_bars"
    ]
    ns = dict(Bar=engine.Bar, datetime=datetime, timedelta=timedelta, timezone=timezone)
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "frozen-synthetic", "exec"), ns
    )
    return [
        dict(
            start_ns=int(bar.timestamp.timestamp()) * 1_000_000_000,
            end_ns=int(bar.timestamp.timestamp()) * 1_000_000_000 + b.MINUTE,
            availability_ns=int(bar.timestamp.timestamp()) * 1_000_000_000 + b.MINUTE,
            ohlcv=[
                int(getattr(bar, f) * 1_000_000_000)
                for f in ("open", "high", "low", "close")
            ]
            + [bar.volume],
        )
        for bar in ns["synthetic_bars"]()
    ]


def test_both_arms_next_bar_replay_cash_and_cold_start():
    bars = synthetic_native_bars()
    result = r.replay(bars, json.loads((runner.HERE / "pins.json").read_text()), set())
    assert result["outcome"] == "REPLAY_COMPLETE"
    assert set(result["arms"]) == {"no-ml", "ml050"}
    plain = result["arms"]["no-ml"]
    assert plain["summary"]["historical_pnl"] == 6.0
    assert plain["independent_reconciliation"]["passed"]
    assert plain["trades"][0]["bars_held"] == 60
    # Adapter annotations must preserve every original gate/accounting field.
    pins = json.loads((runner.HERE / "pins.json").read_text())
    engine, _, _, _ = r.load_frozen(pins)
    from datetime import datetime, timezone

    converted = [
        engine.Bar(
            datetime.fromtimestamp(raw["start_ns"] // 1_000_000_000, timezone.utc),
            *(price / 1_000_000_000 for price in raw["ohlcv"][:4]),
            raw["ohlcv"][4],
        )
        for raw in bars
    ]
    original = engine.ReplayEngine(classification="DEVELOPMENT_REPLAY").run(converted)
    for collection in ("gates", "events", "trades"):
        assert len(original[collection]) == len(plain[collection])
        for before, after in zip(original[collection], plain[collection]):
            assert all(after[key] == value for key, value in before.items())
    assert all(
        plain["terminal"][key] == value for key, value in original["terminal"].items()
    )
    assert plain["summary"] == original["summary"]
    for arm in result["arms"].values():
        assert arm["gates"][0]["cold_start"]["lr_full_history"] is False
        assert arm["gates"][0]["cold_start"]["adr_full_history"] is False
        assert (
            arm["gates"][0]["cold_start"]["volatility_positive_atr_observations"] == 0
        )
        for e in arm["events"]:
            if e["kind"] == "FILL":
                assert e["signal_time"] < e["timestamp"]
                assert e["timing_qualification"].startswith("MODELED_WITHIN_BAR")
    order = next(g for g in plain["gates"] if g["outcome"] == "ORDER")
    dangerous = copy.deepcopy(bars)
    dangerous[
        next(
            i
            for i, bar in enumerate(bars)
            if bar["start_ns"] == order["interval_start_ns"]
        )
    ]["availability_ns"] += 1
    held = r.replay(
        dangerous, json.loads((runner.HERE / "pins.json").read_text()), set()
    )
    assert held["outcome"] == "HOLD"
    assert held["arms"] == {}
    assert held["reasons"] == ["ORDER_AVAILABILITY_OVERLAPS_NEXT_BAR"]


def test_non_order_delay_does_not_fabricate_unsafe_order():
    bars = synthetic_native_bars()
    bars[0]["availability_ns"] += 1
    result = r.replay(bars, json.loads((runner.HERE / "pins.json").read_text()), set())
    assert result["outcome"] == "REPLAY_COMPLETE"
    for arm in result["arms"].values():
        assert arm["gates"][0]["outcome"] != "ORDER"
        assert arm["gates"][0]["decision_available_ns"] == bars[0]["availability_ns"]


def test_output_cannot_add_files_under_pinned_inputs():
    with pytest.raises(ValueError, match="protected"):
        runner.run(runner.DATA / "new-pilot-output")


def test_acquisition_and_dependency_tamper(tmp_path, monkeypatch):
    source = tmp_path / "native"
    source.write_bytes(b"changed")
    monkeypatch.setattr(runner, "DATA", tmp_path)
    with pytest.raises(ValueError, match="acquisition pin"):
        runner.verify_pins(
            {"files": [{"file": "native", "bytes": 7, "sha256": "0" * 64}]}
        )
    with pytest.raises(ValueError, match="dependency pin"):
        runner.verify_pins(
            {
                "files": [],
                "sources": {},
                "model_sha256": runner.sha(runner.MODEL),
                "versions": {"numpy": "0.0.0"},
            }
        )


def test_coverage_marks_missing_source_day():
    builder = b.Builder(start=b.START, end=b.START + 2 * b.MINUTE)
    builder.files = [
        {"file": "native/job/glbx-mdp3-20250519.mbo.dbn.zst", "records": 0}
    ]
    row = b.coverage(builder, [])[0]
    assert row["mbo_source_file_present"] is True
    builder.files = []
    row = b.coverage(builder, [])[0]
    assert row["mbo_source_file_present"] is False
    assert row["coverage"] == "NO_TRADE_STATUS_UNKNOWN"


def test_real_auxiliary_decode_boolean_status_and_definition(tmp_path):
    definition = next(runner.DATA.glob("native/*/*20250519.definition.dbn.zst"))
    status = next(runner.DATA.glob("native/*/*20250519.status.dbn.zst"))
    ds = b.read_auxiliary(
        definition, str(definition.relative_to(runner.DATA)), "definition"
    )
    ss = b.read_auxiliary(status, str(status.relative_to(runner.DATA)), "status")
    assert ds[0]["tick_nanos"] == 250_000_000
    assert ds[0]["point_value_nanos"] == 2_000_000_000
    assert ss[0]["is_trading"] is True
    assert ss[0]["initial_state"] is True
    builder = b.Builder(
        start=b.START + 21 * 60 * b.MINUTE, end=b.START + 22 * 60 * b.MINUTE
    )
    rows = b.coverage(builder, ss)
    assert rows[0]["coverage"] == "NO_TRADE_MIXED_STATUS"
    assert all(row["coverage"] == "NO_TRADE_OBSERVED_NONTRADING" for row in rows[1:])
    with b.zstandard.ZstdDecompressor().stream_reader(status.open("rb")) as f:
        raw = f.read()
    truncated = tmp_path / status.name
    truncated.write_bytes(b.zstandard.ZstdCompressor().compress(raw[:-1]))
    with pytest.raises(ValueError, match="truncated auxiliary"):
        b.read_auxiliary(truncated, truncated.name, "status")


def test_earlier_delayed_bar_blocks_later_order_at_prefix_watermark():
    bars = synthetic_native_bars()
    bars[0]["availability_ns"] = bars[-1]["end_ns"] + b.MINUTE
    result = r.replay(bars, json.loads((runner.HERE / "pins.json").read_text()), set())
    assert result["outcome"] == "HOLD"
    assert result["arms"] == {}
    assert result["reasons"] == ["ORDER_AVAILABILITY_OVERLAPS_NEXT_BAR"]


def test_fresh_process_private_loader_never_imports_live_initializers():
    import subprocess

    code = """
import importlib.abc,importlib.util,json,pathlib,sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if any(fullname==p or fullname.startswith(p+'.') for p in ('src.research','src.data','src.detection')):
            raise AssertionError('forbidden initializer: '+fullname)
sys.meta_path.insert(0,Guard())
root=pathlib.Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('minute_cli',root/'src/cli/check_yank_native_minute.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
runner=cli.load_runner()
replay=sys.modules['_yank_native_minute.replay']
_,model,_,_=replay.load_frozen(json.loads((runner.HERE/'pins.json').read_text()))
assert model.PinnedModel(replay.MODEL).error is None
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT)], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("partitions", [[4], [1, 3], [2, 1, 1], [1, 1, 1, 1]])
def test_trade_digest_preserves_every_raw_native_byte(partitions):
    import hashlib

    records = [
        record(),
        record("F"),
        record(price=21_000_000_000),
        record(price=19_000_000_000),
    ]
    for i, rec in enumerate(records):
        rec["order_id"] = 0x1234567800000000 + i
        rec["channel"] = 8 + i
        rec["side"] = ord("A")
        rec["ts_in_delta"] = -123456 - i
    raw = np.concatenate(records).tobytes()
    expected = hashlib.sha256(raw[:56] + raw[112:]).hexdigest()
    _, bars, _ = build(records, partitions)
    assert bars[0]["trade_sha256"] == expected
    assert sum(field[0].itemsize for field in b.DTYPE.fields.values()) == 56


@pytest.mark.parametrize(
    "halt_offset,classification",
    [(30_000_000_000, "MIXED"), (0, "NONTRADING"), (b.MINUTE, "TRADING")],
)
def test_status_classification_covers_full_half_open_interval(
    halt_offset, classification
):
    builder = b.Builder(start=b.START, end=b.START + b.MINUTE)
    states = [
        dict(ts_recv_ns=b.START - 1, is_trading=True, file="status", record_index=0),
        dict(
            ts_recv_ns=b.START + halt_offset,
            is_trading=False,
            file="status",
            record_index=1,
        ),
    ]
    row = b.coverage(builder, states)[0]
    assert row["interval_status"] == classification
    assert row["status_at_start"]["is_trading"] is (halt_offset != 0)
    assert row["status_at_end"]["is_trading"] is (halt_offset == b.MINUTE)
    assert (
        row["coverage"]
        == {
            "MIXED": "NO_TRADE_MIXED_STATUS",
            "NONTRADING": "NO_TRADE_OBSERVED_NONTRADING",
            "TRADING": "NO_TRADE_OBSERVED_TRADING",
        }[classification]
    )


@pytest.mark.parametrize(
    "mutation", ["lookalike", "wrong_symbol", "wrong_start", "wrong_end", "extra"]
)
def test_metadata_mapping_requires_exact_daily_identity(monkeypatch, mutation):
    from datetime import date
    from types import SimpleNamespace

    mapping = {
        "MNQM5": [
            {
                "start_date": date(2025, 5, 19),
                "end_date": date(2025, 5, 20),
                "symbol": "42009475",
            }
        ]
    }
    meta = SimpleNamespace(
        schema="mbo",
        dataset="GLBX.MDP3",
        symbols=["MNQM5"],
        start=b.START,
        end=b.START + 1440 * b.MINUTE,
        ts_out=False,
        partial=[],
        not_found=[],
        mappings=mapping,
    )
    monkeypatch.setattr(
        b.dbn,
        "DBNDecoder",
        lambda: SimpleNamespace(write_and_decode=lambda raw: [meta]),
    )
    raw = b"DBN\x03" + struct.pack("<I", 100) + b"x" * 100
    b.metadata(io.BytesIO(raw), "mbo", "20250519")
    if mutation == "lookalike":
        mapping["MNQM5"][0]["symbol"] = "1420094750"
    elif mutation == "wrong_symbol":
        mapping["OTHER"] = mapping.pop("MNQM5")
    elif mutation == "extra":
        mapping["MNQM5"].append(dict(mapping["MNQM5"][0]))
    else:
        mapping["MNQM5"][0][
            "start_date" if mutation == "wrong_start" else "end_date"
        ] = date(2025, 5, 21)
    with pytest.raises(ValueError, match="metadata identity"):
        b.metadata(io.BytesIO(raw), "mbo", "20250519")


def test_terminal_mark_uses_delayed_history_watermark_without_orders():
    bars = synthetic_native_bars()[:2]
    bars[0]["availability_ns"] = bars[-1]["end_ns"] + b.MINUTE
    result = r.replay(bars, json.loads((runner.HERE / "pins.json").read_text()), set())
    assert result["outcome"] == "REPLAY_COMPLETE"
    for arm in result["arms"].values():
        assert all(g["outcome"] != "ORDER" for g in arm["gates"])
        assert arm["terminal"]["mark_available_ns"] == bars[0]["availability_ns"]


@pytest.mark.parametrize(
    "case,reason",
    [
        ("unknown", "NO_TRADE_STATUS_UNKNOWN"),
        ("missing_trading", "ABSENT_MBO_FILE_WITHOUT_OBSERVED_NONTRADING_STATUS"),
        ("missing_late_halt", "ABSENT_MBO_FILE_WITHOUT_OBSERVED_NONTRADING_STATUS"),
    ],
)
def test_runner_coverage_holds_preserve_real_replay_and_publication(
    tmp_path, monkeypatch, case, reason
):
    root = tmp_path / "repo"
    here = root / "src/research/yank_native_minute"
    here.mkdir(parents=True)
    (root / "src/cli").mkdir()
    (root / "src/cli/check_yank_native_minute.py").write_text("")
    (root / "pyproject.toml").write_text("")
    status_name = "native/job/glbx-mdp3-20250519.status.dbn.zst"
    pins = {"files": [{"file": status_name}], "versions": {}}
    (here / "pins.json").write_text(json.dumps(pins))
    builder = b.Builder(start=b.START, end=b.START + 2 * b.MINUTE)
    builder.consume(record(), "native", 0)
    if case == "unknown":
        builder.files = [
            {"file": "native/job/glbx-mdp3-20250519.mbo.dbn.zst", "records": 1}
        ]
    states = (
        []
        if case == "unknown"
        else [
            dict(ts_recv_ns=b.START, is_trading=True, file=status_name, record_index=0)
        ]
    )
    if case == "missing_late_halt":
        states.append(
            dict(
                ts_recv_ns=b.START + b.MINUTE + 30_000_000_000,
                is_trading=False,
                file=status_name,
                record_index=1,
            )
        )
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "HERE", here)
    monkeypatch.setattr(runner, "Builder", lambda: builder)
    monkeypatch.setattr(runner, "verify_pins", lambda pins: None)
    monkeypatch.setattr(runner, "read_auxiliary", lambda *args: states)
    output = root / "output"
    report = runner.run(output)
    assert report["status"] == "HOLD_DATA_CHECKS"
    assert reason in report["hold_reasons"]
    saved = json.loads((output / "replay.json").read_text())
    assert saved["outcome"] == "HOLD" and saved["arms"] == {}
    assert reason in saved["reasons"]
    assert (output / "manifest.json").is_file()
