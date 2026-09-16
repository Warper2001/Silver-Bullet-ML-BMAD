"""Synthetic Phase A contracts; never read the bound market input or credentials."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import pytest

from tools import mnq_wick_short_calibration as cal
from tools import mnq_wick_short_power_gate as gate


def raw_session(day: date) -> list[dict[str, Any]]:
    start = datetime.combine(day, time(9, 31), tzinfo=gate.NY)
    return [
        {
            "TimeStamp": (start + timedelta(minutes=i)).isoformat(),
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Contract": "MNQH25",
        }
        for i in range(390)
    ]


def set_bar(
    rows: list[dict[str, Any]],
    slot: int,
    opening: float,
    high: float,
    low: float,
    close: float,
) -> None:
    for i in range(slot * 5, slot * 5 + 5):
        rows[i].update(Open=opening, High=high, Low=low, Close=close)


def parse(tmp_path: Path, rows: list[dict[str, Any]]) -> tuple[list[gate.Minute], int]:
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(rows))
    return gate.load_minutes(path)


def synthetic_sample(tmp_path: Path) -> list[gate.Minute]:
    first = raw_session(date(2025, 1, 2))
    set_bar(first, 0, 100, 103, 100, 101)  # bullish qualifying signal
    set_bar(first, 1, 202, 206, 200, 200)  # valid bearish next signal; +$4
    set_bar(first, 2, 300, 304, 299, 303)  # next reference interval; -$6
    set_bar(first, 75, 100, 103, 100, 101)  # final qualifying slot
    set_bar(first, 76, 400, 403, 400, 401)  # qualifying wick, excluded slot
    set_bar(first, 77, 500, 503, 500, 501)  # qualifying wick, excluded slot
    second = raw_session(date(2025, 1, 3))  # eligible zero-signal session
    third = raw_session(date(2025, 1, 6))
    set_bar(third, 0, 101, 105, 100, 100)  # valid bearish signal
    set_bar(third, 1, 110, 112, 108, 109)  # +$2 reference
    return parse(tmp_path, first + second + third)[0]


def original_report(counts: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "rth_sessions_seen",
        "eligible_sessions",
        "excluded_sessions",
        "exclusion_reasons",
        "post_cutoff_records_skipped",
    )
    return {
        "verdict": "POWER_UNDETERMINED",
        "evaluation_allowed": False,
        "eligibility": {key: counts[key] for key in keys},
        "signal_frequency": {
            key: counts[key]
            for key in (
                "total_signals",
                "sessions_with_signal",
                "signals_per_eligible_session",
            )
        },
    }


@pytest.mark.parametrize(
    "opening,high,low,close,expected",
    [
        (100, 103, 100, 101, True),
        (101, 103, 100, 100, True),
        (100, 111, 99, 101, True),
        (101, 111, 99, 100, True),
        (100, 103, 100, 100, False),
        (100, 102.99, 100, 101, False),
        (100, 111, 98.99, 101, False),
    ],
)
def test_valid_ohlc_wick_boundaries_both_colours(
    opening: float, high: float, low: float, close: float, expected: bool
) -> None:
    record = {"Open": opening, "High": high, "Low": low, "Close": close}
    gate.finite_ohlc(record)
    assert (
        gate.is_wick_short(gate.Bar(date(2025, 1, 2), 0, opening, high, low, close))
        is expected
    )


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("Open", None, "malformed"),
        ("High", float("nan"), "non-finite"),
        ("Low", float("inf"), "non-finite"),
        ("High", 90, "ordering"),
        ("Low", 102, "ordering"),
        ("TimeStamp", "2025-01-02T09:31:00", "offset"),
        ("TimeStamp", "nonsense", "invalid"),
        ("Contract", "", "Contract"),
    ],
)
def test_input_malformed_rows_refuse(
    tmp_path: Path, field: str, value: Any, error: str
) -> None:
    rows = raw_session(date(2025, 1, 2))
    rows[0][field] = value
    with pytest.raises(gate.GateError, match=error):
        parse(tmp_path, rows)


def test_missing_ohlc_and_duplicate_offsets_refuse(tmp_path: Path) -> None:
    rows = raw_session(date(2025, 1, 2))
    del rows[0]["Close"]
    with pytest.raises(gate.GateError, match="malformed Close"):
        parse(tmp_path, rows)
    rows = raw_session(date(2025, 1, 2))
    duplicate = dict(rows[0], TimeStamp="2025-01-02T14:31:00Z")
    with pytest.raises(gate.GateError, match="duplicate retained"):
        parse(tmp_path, rows + [duplicate])


def test_cutoff_equality_and_offset_are_excluded_before_ohlc(tmp_path: Path) -> None:
    rows = raw_session(date(2025, 1, 2))
    rows += [
        {"TimeStamp": "2026-03-01T00:00:00Z"},
        {"TimeStamp": "2026-02-28T19:00:00-05:00"},
        dict(rows[0], TimeStamp="2026-02-28T23:59:00Z"),
    ]
    minutes, skipped = parse(tmp_path, rows)
    assert skipped == 2 and len(minutes) == 391
    assert minutes[-1].timestamp == datetime(2026, 2, 28, 23, 59, tzinfo=timezone.utc)


def test_dst_labels_coverage_and_primary_exclusions(tmp_path: Path) -> None:
    before, after = date(2025, 3, 7), date(2025, 3, 10)
    short = raw_session(date(2025, 7, 3))[:210]
    mixed = raw_session(date(2025, 3, 11))
    mixed[50]["Contract"] = "MNQM25"
    both = raw_session(date(2025, 3, 12))[:-1]
    both[50]["Contract"] = "MNQM25"
    minutes, _ = parse(
        tmp_path, raw_session(before) + raw_session(after) + short + mixed + both
    )
    ledger, bars, _, counts = cal.eligible_ledgers(minutes, 7)
    assert counts["exclusion_reasons"] == {
        "INCOMPLETE_RTH_MINUTES": 2,
        "MIXED_CONTRACT": 1,
    }
    assert counts["rth_sessions_seen"] == 5 and counts["eligible_sessions"] == 2
    assert [bar.slot for bar in bars[before]] == list(range(78))
    lookup = {row.timestamp.astimezone(gate.NY): row for row in minutes}
    first = cal.component_minutes(before, 0, lookup)
    last = cal.component_minutes(after, 77, lookup)
    assert first[0]["minute_label_utc"] == "2025-03-07T14:31:00+00:00"
    assert last[-1]["minute_label_utc"] == "2025-03-10T20:00:00+00:00"
    assert first[-1]["minute_label_local"].endswith("09:35:00-05:00")
    assert last[0]["minute_label_local"].endswith("15:56:00-04:00")
    excluded = {row["session_id"]: row for row in ledger if not row["eligible"]}
    assert excluded["2025-03-12"]["exclusion_reasons"] == [
        "INCOMPLETE_RTH_MINUTES",
        "MIXED_CONTRACT",
    ]
    assert len(excluded["2025-07-03"]["missing_minute_labels"]) == 180
    assert all(row["signal_count"] is None for row in excluded.values())


def test_actual_adjacent_mapping_signs_costs_and_zero_session(tmp_path: Path) -> None:
    minutes = synthetic_sample(tmp_path)
    eligibility, bars, signals, counts = cal.eligible_ledgers(minutes, 0)
    assert counts["total_signals"] == 4 and counts["sessions_with_signal"] == 2
    assert counts["sessions_without_signal"] == 1
    assert signals[date(2025, 1, 2)] == [0, 1, 75]
    cal.reconcile_counts(counts, original_report(counts))
    outcomes, sessions = cal.aligned_ledgers(minutes, bars, signals)
    assert [row["gross_dollars"] for row in outcomes] == [4, -6, -2, 2]
    assert outcomes[0]["next_open"] == 202 and outcomes[0]["next_close"] == 200
    assert outcomes[1]["next_open"] == 300 and outcomes[1]["next_close"] == 303
    assert (
        outcomes[0]["reference_interval_end_local"]
        == outcomes[1]["reference_interval_start_local"]
    )
    assert outcomes[0]["reference_component_minutes"][-1][
        "minute_label_local"
    ].endswith("09:40:00-05:00")
    assert outcomes[1]["reference_component_minutes"][0]["minute_label_local"].endswith(
        "09:41:00-05:00"
    )
    assert outcomes[0]["next_close"] != outcomes[1]["next_open"]
    assert outcomes[2]["reference_interval_end_local"].endswith("15:55:00-05:00")
    for row in outcomes:
        assert row["body"] > 0 and row["upper_wick"] >= 2 * row["body"]
        assert row["lower_wick"] <= 0.1 * row["upper_wick"]
        for cost in (1.22, 2.22, 3.22):
            assert row["net_dollars"][f"{cost:.2f}"] == pytest.approx(
                row["gross_dollars"] - cost
            )
    assert [row["signal_count"] for row in sessions] == [3, 0, 1]
    assert [row["gross_total_dollars"] for row in sessions] == [-4, 0, 2]
    assert sessions[1]["net_total_dollars"] == {"1.22": 0.0, "2.22": 0.0, "3.22": 0.0}
    assert outcomes[0]["session_net_total_dollars"]["1.22"] == pytest.approx(-7.66)
    assert all(row["eligible"] for row in eligibility)


def test_cluster_finite_correction_t_and_zero_residual_cluster() -> None:
    result = cal.cluster_interval([0, 2, 4, 6, 8], ["a", "a", "b", "c", "c"])
    assert result["N"] == 5 and result["G"] == 3
    assert result["degrees_of_freedom"] == 2
    assert result["variance"] == pytest.approx(1.5 * (36 + 0 + 36) / 25)
    assert result["se"] == pytest.approx(math.sqrt(4.32))
    assert result["t_critical"] == pytest.approx(4.302652729696142)
    assert result["interval"] == pytest.approx(
        [
            4 - 4.302652729696142 * math.sqrt(4.32),
            4 + 4.302652729696142 * math.sqrt(4.32),
        ]
    )
    assert result["cluster_sizes"] == [
        {"label": "a", "count": 2},
        {"label": "b", "count": 1},
        {"label": "c", "count": 2},
    ]
    assert result["largest_cluster_fraction"] == 0.4
    assert result["cluster_share_hhi"] == pytest.approx(0.36)


def test_independent_week_interval_envelope_and_cost_invariance() -> None:
    days = [
        date(2025, 1, 2),
        date(2025, 1, 2),
        date(2025, 1, 3),
        date(2025, 1, 6),
        date(2025, 1, 6),
    ]
    gross = cal.estimand([0, 2, 4, 6, 8], days)
    # Weekly residual sums: -6 and +6; N=5, G=2, variance=5.76.
    assert gross["iso_week_clustered"]["variance"] == pytest.approx(5.76)
    assert gross["iso_week_clustered"]["degrees_of_freedom"] == 1
    week_ci = [4 - 12.706204736432095 * 2.4, 4 + 12.706204736432095 * 2.4]
    assert gross["iso_week_clustered"]["interval"] == pytest.approx(week_ci)
    assert gross["envelope"]["interval"] == pytest.approx(week_ci)
    assert gross["descriptive"]["sample_sd"] == pytest.approx(math.sqrt(10))
    for cost in (1.22, 2.22, 3.22):
        net = cal.estimand([value - cost for value in [0, 2, 4, 6, 8]], days)
        assert net["descriptive"]["mean"] == pytest.approx(4 - cost)
        assert net["descriptive"]["sample_sd"] == pytest.approx(math.sqrt(10))
        for key in ("session_clustered", "iso_week_clustered"):
            assert net[key]["se"] == pytest.approx(gross[key]["se"])
            assert net[key]["interval"] == pytest.approx(
                [x - cost for x in gross[key]["interval"]]
            )


@pytest.mark.parametrize(
    "values,labels,reason",
    [
        ([], [], "FEWER_THAN_TWO_OBSERVATIONS"),
        ([1], ["a"], "FEWER_THAN_TWO_OBSERVATIONS"),
        ([1, 2], ["a", "a"], "FEWER_THAN_TWO_NONEMPTY_CLUSTERS"),
        ([0.1, 0.1, 0.1], ["a", "b", "c"], "CONSTANT_OBSERVATIONS_ZERO_VARIANCE"),
        ([0, 2, 0, 2], ["a", "a", "b", "b"], "INVALID_OR_NONPOSITIVE_CLUSTER_VARIANCE"),
        ([float("nan"), 1], ["a", "b"], "NONFINITE_OBSERVATION"),
        ([float("inf"), 1], ["a", "b"], "NONFINITE_OBSERVATION"),
        ([1e308, -1e308], ["a", "b"], "INVALID_OR_NONPOSITIVE_CLUSTER_VARIANCE"),
    ],
)
def test_invalid_uncertainty_is_null_with_reason(
    values: list[float], labels: list[str], reason: str
) -> None:
    result = cal.cluster_interval(values, labels)
    assert result["status"] == "UNASSESSABLE" and result["reason"] == reason
    assert all(
        result[field] is None for field in ("variance", "se", "interval", "t_critical")
    )
    json.dumps(result, allow_nan=False)


def test_invalid_companion_preserves_session_interval_and_constant_sd() -> None:
    result = cal.estimand(
        [1, 2, 3], [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)]
    )
    assert result["session_clustered"]["interval"] is not None
    assert result["iso_week_clustered"]["interval"] is None
    assert result["envelope"]["status"] == "UNASSESSABLE"
    assert cal.descriptive([0.1, 0.1, 0.1])["sample_sd"] == 0
    assert cal.descriptive([])["mean"] is None
    assert cal.descriptive([1])["sample_sd"] is None
    assert cal.descriptive([1, float("nan")])["mean"] is None
    assert cal.descriptive([1e308, 1e308])["mean"] is None
    with pytest.raises(cal.CalibrationError, match="length mismatch"):
        cal.cluster_interval([1, 2], ["a"])


@pytest.mark.parametrize("values", [[0.1, 0.3, 0.1, 0.3], [0.78, 2.78, 0.78, 2.78]])
def test_decimal_balanced_clusters_have_no_invented_variance(
    values: list[float],
) -> None:
    result = cal.cluster_interval(values, ["a", "a", "b", "b"])
    assert result["status"] == "UNASSESSABLE"
    assert result["reason"] == "INVALID_OR_NONPOSITIVE_CLUSTER_VARIANCE"
    assert result["variance"] is None and result["se"] is None


def test_session_totals_frequency_denominators_and_iso_year(tmp_path: Path) -> None:
    minutes = synthetic_sample(tmp_path)
    _, bars, signals, _ = cal.eligible_ledgers(minutes, 0)
    outcomes, sessions = cal.aligned_ledgers(minutes, bars, signals)
    result = cal.summarize(outcomes, sessions)
    frequency = result["per_eligible_session"]["signal_count"]
    assert frequency["descriptive"]["observation_count"] == 3
    assert frequency["descriptive"]["mean"] == pytest.approx(4 / 3)
    assert frequency["session_clustered"]["G"] == 3
    assert result["per_signal"]["gross_dollars"]["session_clustered"]["G"] == 2
    session_gross = result["per_eligible_session"]["gross_total_dollars"]
    assert session_gross["descriptive"]["mean"] == pytest.approx(-2 / 3)
    assert session_gross["descriptive"]["sample_sd"] == pytest.approx(math.sqrt(28 / 3))
    net = result["per_eligible_session"]["net_total_dollars"]["1.22"]
    assert net["descriptive"]["total"] == pytest.approx(-6.88)
    assert net["descriptive"]["mean"] == pytest.approx(-6.88 / 3)
    assert net["session_clustered"]["se"] != pytest.approx(
        session_gross["session_clustered"]["se"]
    )
    assert cal.iso_week(date(2024, 12, 30)) == "2025-W01"
    assert cal.iso_week(date(2021, 1, 1)) == "2020-W53"


def configure_synthetic_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    (tmp_path / ".git").write_text("synthetic worktree marker")
    (tmp_path / "_bmad-output").mkdir()
    synthetic_sample(tmp_path)
    source = tmp_path / "synthetic.json"
    minutes, skipped = gate.load_minutes(source)
    counts = cal.eligible_ledgers(minutes, skipped)[3]
    bound_report = tmp_path / "gate-report.json"
    bound_report.write_text(json.dumps(original_report(counts)))
    monkeypatch.setattr(cal, "ROOT", tmp_path)
    monkeypatch.setattr(cal, "AUTHORIZED_ROOT", tmp_path)
    monkeypatch.setattr(cal, "INPUT", source)
    monkeypatch.setattr(cal, "INPUT_SHA256", cal.sha256(source))
    monkeypatch.setattr(cal, "GATE_REPORT", "gate-report.json")
    monkeypatch.setattr(
        cal, "verify_provenance", lambda root: {"run_revision": "synthetic"}
    )
    monkeypatch.setattr(
        cal, "runtime_provenance", lambda: {"python_version": "synthetic"}
    )
    return (
        source,
        tmp_path / "_bmad-output/mnq-wick-short-calibration-phase-a-synthetic",
    )


def assert_not_called(*args: Any, **kwargs: Any) -> Any:
    pytest.fail("operation must not be reached")


def test_synthetic_cli_publication_provenance_and_no_old_gate_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = configure_synthetic_cli(tmp_path, monkeypatch)
    for method in (
        "run",
        "shifted_outcomes",
        "summarize_shifts",
        "cluster_se",
        "publish",
    ):
        monkeypatch.setattr(gate, method, assert_not_called)
    assert cal.main(["--input", str(source), "--output-dir", str(output)]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["sample_role"] == "calibration-development-only"
    assert report["original_gate_verdict"] == "POWER_UNDETERMINED"
    assert report["evaluation_allowed"] is False
    assert report["counts_reconciled_before_alignment"] is True
    assert report["cost_scenarios_dollars"] == [1.22, 2.22, 3.22]
    assert report["provenance"]["input"]["sha256"] == cal.sha256(source)
    assert report["started_at_utc"] <= report["completed_at_utc"]
    manifest = json.loads((output / "manifest.json").read_text())
    for name, digest in manifest["output_sha256"].items():
        assert cal.sha256(output / name) == digest
    complete = json.loads((output / "COMPLETE.json").read_text())
    assert complete["manifest_sha256"] == cal.sha256(output / "manifest.json")
    assert "Calibration only" in (output / "report.md").read_text()


@pytest.mark.parametrize(
    "kind", ["collision", "prohibited", "symlink", "sealed", "alternate"]
)
def test_destination_and_input_refusal_before_market_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    source, output = configure_synthetic_cli(tmp_path, monkeypatch)
    if kind == "collision":
        output.mkdir()
        (output / "keep.txt").write_text("preserve")
    elif kind == "prohibited":
        output = tmp_path / "logs/run"
    elif kind == "symlink":
        output.symlink_to(tmp_path, target_is_directory=True)
    elif kind == "sealed":
        source = tmp_path / "data/sealed_holdout/never-read.json"
    else:
        source = tmp_path / "another-input.json"
    monkeypatch.setattr(cal, "sha256", assert_not_called)
    monkeypatch.setattr(gate, "load_minutes", assert_not_called)
    assert cal.main(["--input", str(source), "--output-dir", str(output)]) == 2
    if kind == "collision":
        assert (output / "keep.txt").read_text() == "preserve"


def test_count_mismatch_prevents_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = configure_synthetic_cli(tmp_path, monkeypatch)
    report = json.loads((tmp_path / "gate-report.json").read_text())
    report["signal_frequency"]["total_signals"] += 1
    (tmp_path / "gate-report.json").write_text(json.dumps(report))
    monkeypatch.setattr(cal, "aligned_ledgers", assert_not_called)
    assert cal.main(["--input", str(source), "--output-dir", str(output)]) == 2
    assert not (output / "COMPLETE.json").exists()
    assert "reconciliation" in (output / "FAILED.json").read_text()


def test_input_hash_mismatch_prevents_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = configure_synthetic_cli(tmp_path, monkeypatch)
    monkeypatch.setattr(cal, "INPUT_SHA256", "0" * 64)
    monkeypatch.setattr(gate, "load_minutes", assert_not_called)
    assert cal.main(["--input", str(source), "--output-dir", str(output)]) == 2
    assert "input hash mismatch" in (output / "FAILED.json").read_text()


def test_malformed_input_prevents_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = configure_synthetic_cli(tmp_path, monkeypatch)
    source.write_text("[]")
    monkeypatch.setattr(cal, "INPUT_SHA256", cal.sha256(source))
    monkeypatch.setattr(cal, "aligned_ledgers", assert_not_called)
    assert cal.main(["--input", str(source), "--output-dir", str(output)]) == 2
    assert not (output / "report.json").exists()


def test_provenance_change_prevents_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = configure_synthetic_cli(tmp_path, monkeypatch)
    revisions = iter([{"revision": "one"}, {"revision": "two"}])
    monkeypatch.setattr(cal, "verify_provenance", lambda root: next(revisions))
    monkeypatch.setattr(cal, "aligned_ledgers", assert_not_called)
    assert cal.main(["--input", str(source), "--output-dir", str(output)]) == 2
    assert "changed before alignment" in (output / "FAILED.json").read_text()


@pytest.mark.parametrize("kind", ["valid", "pin", "canonical", "uncommitted", "staged"])
def test_committed_and_canonical_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    (tmp_path / "pin.md").write_bytes(b"frozen")
    (tmp_path / "runner.py").write_bytes(b"implementation")
    monkeypatch.setattr(
        cal, "PINS", {"pin.md": ("canonical", hashlib.sha256(b"frozen").hexdigest())}
    )
    monkeypatch.setattr(cal, "IMPLEMENTATION", ("runner.py",))
    monkeypatch.setattr(cal, "git_audit", lambda root: {"status_porcelain": ""})

    def fake_git(root: Path, *args: str) -> bytes:
        if args[0] == "rev-parse":
            return b"head\n"
        if args[0] == "show":
            if args[1].endswith(":pin.md"):
                return b"changed" if kind == "canonical" else b"frozen"
            return b"implementation"
        if args[0] == "diff":
            return b"staged diff" if kind == "staged" else b""
        if args[0] == "log":
            return b"implementation-revision\n"
        return b""

    monkeypatch.setattr(cal, "git", fake_git)
    if kind == "pin":
        (tmp_path / "pin.md").write_bytes(b"modified")
    if kind == "uncommitted":
        (tmp_path / "runner.py").write_bytes(b"modified")
    if kind == "valid":
        provenance = cal.verify_provenance(tmp_path)
        assert provenance["canonical_artifacts"]["pin.md"]["revision"] == "canonical"
        assert (
            provenance["implementation"]["runner.py"]["revision"]
            == "implementation-revision"
        )
    else:
        with pytest.raises(cal.CalibrationError, match="mismatch|uncommitted|staged"):
            cal.verify_provenance(tmp_path)


def test_publication_rejects_nonempty_and_partial_write_has_no_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = configure_synthetic_cli(tmp_path, monkeypatch)
    original_open = Path.open

    def failed_open(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if path.name == "sessions.jsonl" and mode == "xb":
            raise OSError("synthetic disk failure")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failed_open)
    assert cal.main(["--input", str(source), "--output-dir", str(output)]) == 2
    assert not (output / "COMPLETE.json").exists()
    assert (output / "FAILED.json").exists()
    with pytest.raises(cal.CalibrationError, match="empty directory"):
        cal.publish(output, {}, [], [], [])


def test_git_guard_precedes_git_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[Sequence[str]] = []

    def fake_command(root: Path, args: Sequence[str]) -> bytes:
        commands.append(args)
        if args == ["log", "--format=%H", "origin/main..HEAD"]:
            return b"a\nb\n"
        return b""

    monkeypatch.setattr(cal, "_git_command", fake_command)
    cal.git(tmp_path, "rev-parse", "HEAD")
    assert commands[:3] == [
        ["status", "--porcelain"],
        ["log", "--format=%H", "origin/main..HEAD"],
        ["log", "--format=%H", "HEAD..origin/main"],
    ]
    assert commands[3] == ("rev-parse", "HEAD")
