"""Native measurement, calendar, availability and filesystem contracts."""

import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "native_audit", ROOT / "tools/valentini_native_audit.py"
)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
b = a.load_builder()


def record(action="T", recv=None, event=None, price=20_000_000_000, size=2, flags=128):
    result = np.zeros(1, dtype=b.DTYPE)
    for name, value in dict(
        length=14,
        rtype=160,
        publisher=1,
        id=b.INSTRUMENT,
        action=ord(action),
        recv=b.START + 1 if recv is None else recv,
        event=b.START if event is None else event,
        price=price,
        size=size,
        flags=flags,
        seq=9,
    ).items():
        result[name] = value
    return result


def build(records, partitions=None, end=None):
    rows = np.concatenate(records)
    builder = a.histogram_builder(b.START, end or b.START + 5 * b.MINUTE)
    offset = 0
    for size in partitions or [len(rows)]:
        builder.consume(rows[offset : offset + size], "fixture", offset)
        offset += size
    assert offset == len(rows)
    bars, delayed = builder.finish()
    return builder, bars, delayed


def calendar(start="2025-05-19T00:00:00Z", end="2025-05-19T00:03:00Z", breaks=None):
    return dict(
        schema_version=1,
        instrument="MNQM5",
        timezone="America/New_York",
        sources=[
            dict(id="official", url="https://example.test/calendar", date="2025-01-01")
        ],
        unresolved_evidence=[],
        sessions=[
            dict(
                name="one",
                start=start,
                end=end,
                verification="VERIFIED",
                sources=["official"],
                unresolved_evidence=[],
                breaks=breaks or [],
                boundary_status_exceptions=[],
            )
        ],
    )


def coverage(bars, session):
    present = {row["start_ns"] for row in bars}
    return [
        dict(
            start_ns=m,
            end_ns=m + b.MINUTE,
            mbo_source_file_present=True,
            interval_status="TRADING" if m in session["minutes"] else "NONTRADING",
            coverage="TRADED" if m in present else "NO_TRADE",
            status_transitions=[],
            status_at_end=dict(is_trading=m in session["minutes"]),
        )
        for m in range(session["start"], session["end"] + b.MINUTE, b.MINUTE)
    ]


def compare(records, cal=None):
    builder, bars, delayed = build(records)
    sessions = a.calendar_sessions(cal or calendar())
    ledger, observed = a.classify_sessions(sessions, bars, coverage(bars, sessions[0]))
    snapshots = a.compare_profiles(sessions, ledger, bars, builder.histograms)
    return ledger, snapshots, observed


@pytest.mark.parametrize("partitions", [[6], [1, 2, 3], [1] * 6])
def test_histograms_chunk_order_duplicates_snapshot_and_fill(partitions):
    records = [
        record(),
        record(),
        record("F"),
        record(flags=160),
        record(price=21_000_000_000),
        record(size=0),
    ]
    builder, bars, _ = build(records, partitions)
    assert builder.histograms == {b.START: {80: 4, 84: 2}}
    assert bars[0]["ohlcv"] == [
        20_000_000_000,
        21_000_000_000,
        20_000_000_000,
        20_000_000_000,
        6,
    ]
    assert bars[0]["trade_count"] == 4
    expected = hashlib.sha256(
        b"".join(records[i].tobytes() for i in (0, 1, 4, 5))
    ).hexdigest()
    assert bars[0]["trade_sha256"] == expected
    assert bars == build(records)[1]


def test_conservation_above_uint32_and_exact_area():
    builder, bars, _ = build([record(size=2**32 - 1), record(size=2**32 - 1)])
    assert builder.histograms[b.START][80] == 2 * (2**32 - 1) == bars[0]["ohlcv"][4]
    assert a.native_area({1: 7 * 10**18, 2: 3 * 10**18}) == (1, 1, 1)
    assert a.native_area({1: 0}) is None
    assert a.native_area({1: 2, 2: 2, 3: 2}) == (1, 3, 1)


def test_equal_and_concentrated_distributions():
    records = [
        record(price=p, recv=b.START + 1 + i)
        for i, p in enumerate((20_000_000_000, 20_250_000_000, 20_500_000_000))
    ]
    records += [
        record(recv=b.START + b.MINUTE + 1),
        record(recv=b.START + 2 * b.MINUTE + 1),
    ]
    ledger, snapshots, _ = compare(records)
    assert ledger[0]["eligible"]
    assert snapshots[0]["status"] == "EMPTY_PREFIX"
    assert snapshots[1]["signed_proxy_minus_native_ticks"] == [0, 0, 0]
    records[0]["size"] = 100
    _, snapshots, _ = compare(records)
    assert snapshots[1]["native_ticks"] == [80, 80, 80]
    assert snapshots[1]["proxy_ticks"] == [80, 82, 80]
    summary = a.summarize(snapshots)
    assert summary["comparison_count"] == 2
    assert summary["levels"]["VAH"]["denominator"] == 2
    assert summary["levels"]["VAH"]["absolute_ticks"]["maximum"] == 2


def test_halt_preserves_profile_and_next_session_resets():
    pause = dict(
        start="2025-05-19T00:01:00Z", end="2025-05-19T00:02:00Z", sources=["official"]
    )
    cal = calendar(breaks=[pause])
    second = copy.deepcopy(cal["sessions"][0])
    second.update(
        name="two", start="2025-05-19T00:04:00Z", end="2025-05-19T00:06:00Z", breaks=[]
    )
    cal["sessions"].append(second)
    builder, bars, _ = build(
        [record(recv=b.START + m * b.MINUTE + 1) for m in (0, 2, 4, 5)],
        end=b.START + 6 * b.MINUTE,
    )
    sessions = a.calendar_sessions(cal)
    cov = sum((coverage(bars, s) for s in sessions), [])
    ledger, _ = a.classify_sessions(sessions, bars, cov)
    snapshots = a.compare_profiles(sessions, ledger, bars, builder.histograms)
    assert all(row["eligible"] for row in ledger)
    assert [s["prior_bar_count"] for s in snapshots] == [0, 1, 0, 1]
    assert snapshots[1]["cumulative_volume"] == 2


@pytest.mark.parametrize("partitions", [[4], [1, 1, 1, 1], [2, 2]])
def test_delayed_last_excluded_then_available_no_retroactivity(partitions):
    records = [
        record(recv=b.START + b.MINUTE - 1, flags=0),
        record("F", recv=b.START + b.MINUTE + 1),
        record(recv=b.START + b.MINUTE + 2),
        record(recv=b.START + 2 * b.MINUTE + 1),
    ]
    builder, bars, delayed = build(records, partitions)
    sessions = a.calendar_sessions(calendar())
    ledger, _ = a.classify_sessions(sessions, bars, coverage(bars, sessions[0]))
    snapshots = a.compare_profiles(sessions, ledger, bars, builder.histograms)
    assert len(delayed) == 1
    assert [s["status"] for s in snapshots] == [
        "EMPTY_PREFIX",
        "UNAVAILABLE_PREFIX",
        "COMPARED",
    ]
    assert snapshots[2]["prior_bar_count"] == 2
    # Prefix availability must include every earlier event, not just the last bar.
    bars[0]["availability_ns"] = b.START + 4 * b.MINUTE
    snapshots = a.compare_profiles(sessions, ledger, bars, builder.histograms)
    assert snapshots[2]["status"] == "UNAVAILABLE_PREFIX"


@pytest.mark.parametrize(
    "case", ["missing", "holiday", "conflict", "incomplete", "source", "range"]
)
def test_session_exclusions_cannot_compare(case):
    cal = calendar()
    builder, bars, _ = build(
        [record(recv=b.START + i * b.MINUTE + 1) for i in range(3)]
    )
    if case == "holiday":
        cal["sessions"][0].update(
            verification="UNVERIFIED", unresolved_evidence=["holiday boundary unknown"]
        )
    sessions = a.calendar_sessions(cal)
    cov = coverage(bars, sessions[0])
    if case == "missing":
        bars.pop(1)
    if case == "conflict":
        cov[1]["interval_status"] = "MIXED"
    if case == "incomplete":
        bars[1]["incomplete_event"] = True
    if case == "source":
        cov[1]["mbo_source_file_present"] = False
    if case == "range":
        cov.pop()
    ledger, observed = a.classify_sessions(sessions, bars, cov)
    assert not ledger[0]["eligible"] and ledger[0]["exclusions"]
    assert not a.compare_profiles(sessions, ledger, bars, builder.histograms)
    assert all(row["classification"] == "EXCLUDED_SESSION" for row in observed)


def test_every_observed_bar_assigned_including_outside_calendar():
    _, _, observed = compare(
        [record(recv=b.START + i * b.MINUTE + 1) for i in range(4)]
    )
    assert len(observed) == 4 and observed[-1]["classification"] == "OUTSIDE_CALENDAR"


@pytest.mark.parametrize(
    "case",
    [
        "overlap",
        "duplicate",
        "naive",
        "offset",
        "seconds",
        "break_overlap",
        "citation",
        "unresolved",
        "exception_interior",
    ],
)
def test_calendar_rejects_ambiguous_malformed_boundaries(case):
    cal = calendar()
    row = cal["sessions"][0]
    if case == "overlap":
        cal["sessions"].append(dict(row, name="two"))
    if case == "duplicate":
        cal["sessions"].append(dict(row))
    if case == "naive":
        row["start"] = "2025-05-19T00:00:00"
    if case == "offset":
        row["start"] = "2025-05-19T00:00:00-04:00"
    if case == "seconds":
        row["start"] = "2025-05-19T00:00:01Z"
    if case == "break_overlap":
        row["breaks"] = [
            dict(start=row["start"], end=row["end"], sources=["official"])
        ] * 2
    if case == "citation":
        row["sources"] = ["unknown"]
    if case == "unresolved":
        row["unresolved_evidence"] = ["not proven"]
    if case == "exception_interior":
        row["boundary_status_exceptions"] = [
            dict(
                minute="2025-05-19T00:01:00Z",
                expected_trading=True,
                reason="test",
                sources=["official"],
                transition_refs=[{}],
            )
        ]
    with pytest.raises(ValueError):
        a.calendar_sessions(cal)


def test_dst_is_explicit_utc_not_implicit_fixed_offset():
    winter = a.calendar_sessions(
        calendar("2025-03-07T23:00:00Z", "2025-03-08T22:00:00Z")
    )[0]
    summer = a.calendar_sessions(
        calendar("2025-03-09T22:00:00Z", "2025-03-10T21:00:00Z")
    )[0]
    assert len(winter["minutes"]) == len(summer["minutes"]) == 1380
    assert a.utc(winter["start"]).hour == 23 and a.utc(summer["start"]).hour == 22


def test_boundary_exception_requires_exact_native_exchange_event_evidence():
    transition = dict(
        file="status",
        record_index=3,
        ts_recv_ns=b.START + 100,
        ts_event_ns=b.START,
        is_trading=True,
    )
    row = dict(
        start_ns=b.START,
        interval_status="MIXED",
        status_transitions=[transition],
        status_at_end=transition,
    )
    refs = [
        {
            key: transition[key]
            for key in ("file", "record_index", "ts_recv_ns", "ts_event_ns")
        }
    ]
    exc = dict(transition_refs=refs)
    assert not a.status_matches(row, True, None)
    assert a.status_matches(row, True, exc)
    for field in ("record_index", "ts_event_ns", "ts_recv_ns"):
        mutated = copy.deepcopy(row)
        mutated["status_transitions"][0][field] += 1
        assert not a.status_matches(mutated, True, exc)


def test_malformed_native_identity_tick_and_truncation():
    for field, value in (
        ("id", 1),
        ("price", 1),
        ("length", 13),
        ("publisher", 2),
        ("action", ord("Z")),
    ):
        rec = record()
        rec[field] = value
        with pytest.raises(ValueError):
            build([rec])
    with pytest.raises(ValueError):
        list(b.chunks(io.BytesIO(record().tobytes()[:-1])))


def test_reconciliation_rejects_saved_bars_counts_and_histogram_tampering():
    builder, bars, delayed = build([record()])
    cov = []
    saved = {
        "bars.jsonl": copy.deepcopy(bars),
        "coverage.jsonl": cov,
        "definitions.json": [],
        "status.json": [],
        "delayed-events.json": delayed,
        "exchange-diagnostic.jsonl": [
            builder.exchange[m] for m in sorted(builder.exchange)
        ],
        "report.json": dict(
            counts=dict(builder.counts), files=builder.files, hold_reasons=[]
        ),
    }
    assert a.reconcile(builder, bars, delayed, [], [], cov, saved)["exact"]
    builder.histograms[b.START][80] += 1
    with pytest.raises(ValueError, match="conservation"):
        a.reconcile(builder, bars, delayed, [], [], cov, saved)
    builder.histograms[b.START][80] -= 1
    saved["bars.jsonl"][0]["ohlcv"][0] += 1
    with pytest.raises(ValueError, match="reconciliation"):
        a.reconcile(builder, bars, delayed, [], [], cov, saved)


def test_filesystem_refusals_do_not_modify_input(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    value = source / "input"
    value.write_text("untouched")
    for output in (value, source, source / "new"):
        with pytest.raises(ValueError):
            a.output_path(output, [source])
    symbolic = tmp_path / "symlink"
    symbolic.symlink_to(value)
    with pytest.raises(ValueError):
        a.safe_input(symbolic)
    hard = tmp_path / "hardlink"
    hard.hardlink_to(value)
    with pytest.raises(ValueError):
        a.safe_input(hard)
    assert value.read_text() == "untouched"
    with pytest.raises(ValueError):
        a.safe_input(tmp_path / "sealed_holdout" / "any")


def test_atomic_publication_deterministic_and_collision(tmp_path):
    artifacts = {"report.json": a.canonical_json({"z": 1, "a": 2}).encode()}
    for name in ("a", "b"):
        a.publish(tmp_path / name, artifacts)
    assert (tmp_path / "a/report.json").read_bytes() == (
        tmp_path / "b/report.json"
    ).read_bytes()
    with pytest.raises(OSError):
        a.publish(tmp_path / "a", {"report.json": b"clobber"})
    assert (tmp_path / "a/report.json").read_bytes() == artifacts["report.json"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a", "b"]


def test_corrupt_pins_no_success_output(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    (root / "file").write_text("bad")
    with pytest.raises(ValueError, match="pin mismatch"):
        a.verify_sources(root, {"files": [dict(file="file", bytes=3, sha256="0" * 64)]})
    saved = tmp_path / "saved"
    saved.mkdir()
    (saved / "manifest.json").write_text(
        json.dumps({"artifacts": {"bars.jsonl": "0" * 64}})
    )
    (saved / "bars.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="reconstruction pin mismatch"):
        a.load_reconstruction(saved)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "raw", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}']
)
def test_strict_json(raw):
    with pytest.raises(ValueError):
        a.strict_value(raw)


def test_zero_comparison_summary_is_not_pass():
    result = a.summarize([])
    assert result["comparison_count"] == 0
    assert result["levels"]["VAL"]["absolute_ticks"] is None
    assert result["levels"]["VAL"]["exact_agreement_fraction"] is None


def test_private_loader_forbids_live_runner_and_replay_imports():
    code = """
import importlib.abc,importlib.util,pathlib,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.startswith(('src.research','src.data','src.detection','_yank_native_minute')):
   raise AssertionError(fullname)
sys.meta_path.insert(0,Guard())
root=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(root))
spec=importlib.util.spec_from_file_location('audit',root/'tools/valentini_native_audit.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
a.histogram_builder()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT)], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr


def test_full_publication_contract_determinism_and_failure(tmp_path, monkeypatch):
    """Exercise the CLI orchestration with bounded, independently saved fixtures."""
    source = tmp_path / "source"
    source.mkdir()
    reconstruction = tmp_path / "reconstruction"
    reconstruction.mkdir()
    calendar_file = tmp_path / "calendar.json"
    cal = calendar()
    excluded = copy.deepcopy(cal["sessions"][0])
    excluded.update(
        name="two",
        start="2025-05-19T00:04:00Z",
        end="2025-05-19T00:06:00Z",
        verification="UNVERIFIED",
        unresolved_evidence=["calendar not verified"],
    )
    cal["sessions"].append(excluded)
    calendar_file.write_text(json.dumps(cal))
    names = [
        f"glbx-mdp3-20250519.{schema}.dbn.zst"
        for schema in ("mbo", "status", "definition")
    ]
    for name in names:
        (source / name).write_bytes(b"fixture")
    pins = {
        "files": [
            dict(file=name, bytes=7, sha256=hashlib.sha256(b"fixture").hexdigest())
            for name in names
        ]
    }
    real_read = a.read_json
    monkeypatch.setattr(
        a, "read_json", lambda path: pins if path == a.PINS_PATH else real_read(path)
    )
    statuses = [
        dict(
            file=names[1],
            record_index=0,
            ts_recv_ns=b.START,
            ts_event_ns=b.START,
            is_trading=True,
        ),
        dict(
            file=names[1],
            record_index=1,
            ts_recv_ns=b.START + 3 * b.MINUTE,
            ts_event_ns=b.START + 3 * b.MINUTE,
            is_trading=False,
        ),
    ]
    monkeypatch.setattr(
        b,
        "read_auxiliary",
        lambda path, name, schema: statuses if schema == "status" else [],
    )
    original_factory = a.histogram_builder

    def fixture_builder():
        builder = original_factory(b.START, b.START + 4 * b.MINUTE)
        for i in range(3):
            builder.consume(record(recv=b.START + i * b.MINUTE + 1), names[0], i)
        builder.files = [dict(file=names[0], records=3)]
        builder.read_mbo = lambda *args, **kwargs: None
        return builder

    expected = fixture_builder()
    bars, delayed = expected.finish()
    cov = b.coverage(expected, statuses)
    saved = {
        "bars.jsonl": bars,
        "coverage.jsonl": cov,
        "definitions.json": [],
        "status.json": statuses,
        "delayed-events.json": delayed,
        "exchange-diagnostic.jsonl": [
            expected.exchange[m] for m in sorted(expected.exchange)
        ],
        "report.json": dict(
            counts=dict(expected.counts),
            files=expected.files,
            hold_reasons=[],
            source_pins=copy.deepcopy(pins),
        ),
    }
    monkeypatch.setattr(a, "histogram_builder", fixture_builder)
    monkeypatch.setattr(a, "load_reconstruction", lambda root: (saved, {}))
    for name in ("first", "second"):
        report = a.run(source, reconstruction, calendar_file, tmp_path / name)
        assert report["profile_summary"]["comparison_count"] == 2
        assert report["market_evaluation"] == "NOT_ADMITTED"
        assert report["session_profile_summaries"]["one"] == report["profile_summary"]
        assert report["session_profile_summaries"]["two"]["comparison_count"] == 0
    first, second = tmp_path / "first", tmp_path / "second"
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {
        p.name: p.read_bytes() for p in second.iterdir()
    }
    manifest = json.loads((first / "manifest.json").read_text())
    for name, digest in manifest["artifacts"].items():
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == digest
    saved["bars.jsonl"][0]["ohlcv"][0] += 1
    with pytest.raises(ValueError, match="reconciliation"):
        a.run(source, reconstruction, calendar_file, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    saved["bars.jsonl"][0]["ohlcv"][0] -= 1
    # Valid reconstruction but zero admitted sessions still publishes evidence.
    cal = calendar()
    cal["sessions"][0].update(
        verification="UNVERIFIED", unresolved_evidence=["holiday"]
    )
    calendar_file.write_text(json.dumps(cal))
    args = [
        "--source-root",
        str(source),
        "--reconstruction",
        str(reconstruction),
        "--calendar",
        str(calendar_file),
        "--output-dir",
        str(tmp_path / "excluded"),
    ]
    assert a.main(args) == 3
    report = json.loads((tmp_path / "excluded/report.json").read_text())
    assert report["measurement_status"] == "NO_ELIGIBLE_COMPARISONS"
    assert report["session_profile_summaries"]["one"]["comparison_count"] == 0
    # Joint raw/pin modification must fail before any native decode.
    (source / names[0]).write_bytes(b"changed")
    pins["files"][0]["sha256"] = hashlib.sha256(b"changed").hexdigest()
    monkeypatch.setattr(a, "histogram_builder", lambda: pytest.fail("decode started"))
    with pytest.raises(ValueError, match="pins differ from saved"):
        a.run(source, reconstruction, calendar_file, tmp_path / "pins-changed")
    assert not (tmp_path / "pins-changed").exists()


@pytest.mark.parametrize(
    "fraction", [".000000001", ".0000000001", ",000000001", ".100000000"]
)
def test_minute_boundary_rejects_submicrosecond_fraction(fraction):
    with pytest.raises(ValueError, match="minute aligned"):
        a.minute_ns("2025-05-19T00:00:00" + fraction + "Z")
    assert a.minute_ns("2025-05-19T00:00:00.000000000Z") == b.START


@pytest.mark.parametrize(
    "field,value",
    [
        ("instrument", "NQM5"),
        ("timezone", "UTC"),
        ("instrument", None),
        ("timezone", None),
    ],
)
def test_calendar_rejects_unsupported_instrument_timezone(field, value):
    cal = calendar()
    cal[field] = value
    with pytest.raises(ValueError, match="instrument/timezone"):
        a.calendar_sessions(cal)


@pytest.mark.parametrize("missing", [True, False])
def test_session_requires_corroborated_closing_boundary(missing):
    builder, bars, _ = build(
        [record(recv=b.START + i * b.MINUTE + 1) for i in range(3)]
    )
    sessions = a.calendar_sessions(calendar())
    cov = coverage(bars, sessions[0])
    if missing:
        cov.pop()
        reason = "MISSING_CLOSING_EVIDENCE"
    else:
        # An invented early end inside a continuing TRADING interval is excluded.
        cov[-1]["interval_status"] = "TRADING"
        cov[-1]["status_at_end"]["is_trading"] = True
        reason = "CLOSING_STATUS_CONFLICT"
    ledger, _ = a.classify_sessions(sessions, bars, cov)
    assert ledger[0]["exclusions"][reason] == [sessions[0]["end"]]
    assert not a.compare_profiles(sessions, ledger, bars, builder.histograms)


def test_missing_source_during_scheduled_break_excludes_session():
    pause = dict(
        start="2025-05-19T00:01:00Z", end="2025-05-19T00:02:00Z", sources=["official"]
    )
    sessions = a.calendar_sessions(calendar(breaks=[pause]))
    builder, bars, _ = build([record(recv=b.START + i * b.MINUTE + 1) for i in (0, 2)])
    cov = coverage(bars, sessions[0])
    cov[1]["mbo_source_file_present"] = False
    ledger, _ = a.classify_sessions(sessions, bars, cov)
    assert ledger[0]["exclusions"] == {"MISSING_SOURCE_FILE": [b.START + b.MINUTE]}
    assert not a.compare_profiles(sessions, ledger, bars, builder.histograms)


@pytest.mark.parametrize("changed", [None, "opening", "closing"])
def test_sourced_opening_and_closing_exceptions_reach_classifier(changed):
    cal = calendar()
    builder, bars, _ = build(
        [record(recv=b.START + i * b.MINUTE + 100) for i in range(3)]
    )
    row = cal["sessions"][0]
    transitions = {}
    for label, timestamp, trading, index in (
        ("opening", row["start"], True, 0),
        ("closing", row["end"], False, 1),
    ):
        minute = a.minute_ns(timestamp)
        transition = dict(
            file="status",
            record_index=index,
            ts_recv_ns=minute + 10,
            ts_event_ns=minute,
            is_trading=trading,
        )
        transitions[minute] = transition
        refs = [
            {
                key: transition[key]
                for key in ("file", "record_index", "ts_recv_ns", "ts_event_ns")
            }
        ]
        if changed == label:
            refs[0]["record_index"] += 1
        row["boundary_status_exceptions"].append(
            dict(
                minute=timestamp,
                expected_trading=trading,
                sources=["official"],
                reason="exact boundary event",
                transition_refs=refs,
            )
        )
    sessions = a.calendar_sessions(cal)
    cov = coverage(bars, sessions[0])
    for evidence in cov:
        if evidence["start_ns"] in transitions:
            transition = transitions[evidence["start_ns"]]
            evidence.update(
                interval_status="MIXED",
                status_transitions=[transition],
                status_at_end=transition,
            )
    ledger, _ = a.classify_sessions(sessions, bars, cov)
    assert ledger[0]["eligible"] is (changed is None)
    summary = a.summarize(
        a.compare_profiles(sessions, ledger, bars, builder.histograms)
    )
    assert summary["comparison_count"] == (2 if changed is None else 0)
    if changed:
        reason = (
            "CALENDAR_STATUS_CONFLICT"
            if changed == "opening"
            else "CLOSING_STATUS_CONFLICT"
        )
        assert reason in ledger[0]["exclusions"]


def test_cached_decoder_cannot_claim_different_source_bytes(tmp_path, monkeypatch):
    cached = a.load_builder()
    replacement = tmp_path / "builder.py"
    replacement.write_bytes(a.BUILDER_PATH.read_bytes() + b"\n# changed\n")
    monkeypatch.setattr(a, "BUILDER_PATH", replacement)
    with pytest.raises(ValueError, match="Cached native decoder source hash mismatch"):
        a.load_builder()
    replacement.write_bytes(b"raise AssertionError('must not execute')\n")
    with pytest.raises(ValueError, match="source hash mismatch"):
        a.load_builder()
    assert sys.modules["_valentini_native_builder"] is cached


def test_output_parent_swap_cannot_redirect_into_input(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    sentinel = source / "sentinel"
    sentinel.write_bytes(b"untouched")
    target = a.output_path(parent / "output", [source])
    moved = tmp_path / "moved-parent"
    actual_mkdir = os.mkdir
    held = []
    actual_directory_fd = a.directory_fd

    def capture_fd(path):
        fd = actual_directory_fd(path)
        held.append(fd)
        return fd

    def swap_before_stage(path, *args, **kwargs):
        if str(path).startswith(".valentini-native-"):
            parent.rename(moved)
            parent.symlink_to(source, target_is_directory=True)
        return actual_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(a, "directory_fd", capture_fd)
    monkeypatch.setattr(a.os, "mkdir", swap_before_stage)
    a.publish(target, {"report.json": b"{}"})
    assert sorted(p.name for p in source.iterdir()) == ["sentinel"]
    assert sentinel.read_bytes() == b"untouched"
    assert (moved / "output/report.json").read_bytes() == b"{}"
    with pytest.raises(OSError):
        os.fstat(held[0])
    # A swap before directory traversal is rejected outright.
    with pytest.raises(OSError):
        a.publish(parent / "second", {"report.json": b"{}"})
    assert sorted(p.name for p in source.iterdir()) == ["sentinel"]


def test_summary_signed_absolute_quantiles_have_hand_calculated_values():
    rows = [
        dict(
            status="COMPARED",
            signed_proxy_minus_native_ticks=[value] * 3,
            absolute_ticks=[abs(value)] * 3,
        )
        for value in (-4, -1, 0, 3, 8)
    ]
    summary = a.summarize(rows)
    for level in summary["levels"].values():
        assert level["denominator"] == 5
        assert level["exact_agreement_count"] == 1
        assert level["exact_agreement_fraction"] == 0.2
        signed = level["signed_ticks"]
        absolute = level["absolute_ticks"]
        assert signed["mean"] == 1.2 and signed["median"] == 0
        assert absolute["mean"] == 3.2 and absolute["median"] == 3
        assert signed["quantiles"] == pytest.approx(
            {"0.05": -3.4, "0.25": -1, "0.75": 3, "0.95": 7}
        )
        assert absolute["quantiles"] == pytest.approx(
            {"0.05": 0.2, "0.25": 1, "0.75": 4, "0.95": 7.2}
        )
