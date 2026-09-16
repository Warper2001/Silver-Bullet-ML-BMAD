"""Synthetic contract tests for the outcome-blind wick-short gate."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pytest

from tools import mnq_wick_short_power_gate as gate


def minute_rows(day: date, contract: str = "MNQM26") -> list[dict[str, str]]:
    start = datetime.combine(day, time(9, 31), tzinfo=gate.NY)
    rows = []
    for index in range(390):
        stamp = start + timedelta(minutes=index)
        rows.append(
            {
                "TimeStamp": stamp.astimezone(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "Open": "100",
                "High": "101",
                "Low": "99",
                "Close": "100",
                "Contract": contract,
            }
        )
    return rows


def make_input(path: Path, days: int = 12) -> Path:
    start = date(2025, 1, 2)
    rows = [
        row
        for offset in range(days)
        for row in minute_rows(start + timedelta(days=offset))
    ]
    path.write_text(json.dumps(rows))
    return path


def wick_bar(opening: float, high: float, low: float, close: float) -> gate.Bar:
    return gate.Bar(date(2025, 1, 2), 0, opening, high, low, close)


@pytest.mark.parametrize(
    "bar", [wick_bar(100, 103, 100, 101), wick_bar(101, 104, 101, 100)]
)
def test_wick_accepts_both_colours_and_boundaries(bar: gate.Bar) -> None:
    assert gate.is_wick_short(bar)


def test_wick_rejects_zero_body_and_boundary_failures() -> None:
    assert not gate.is_wick_short(wick_bar(100, 103, 100, 100))
    assert not gate.is_wick_short(wick_bar(100, 102.99, 100, 101))
    assert not gate.is_wick_short(wick_bar(100, 103, 99.79, 101))


def test_complete_sessions_require_exact_minutes_and_single_contract() -> None:
    day = date(2025, 3, 10)  # first Monday after US DST change
    raw = minute_rows(day)
    parsed, _ = gate.load_minutes(write_rows(raw))
    eligible, excluded = gate.sessionize(parsed)
    assert list(eligible) == [day]
    assert not excluded
    incomplete = raw[:-1]
    parsed, _ = gate.load_minutes(write_rows(incomplete))
    assert gate.sessionize(parsed)[1] == {"INCOMPLETE_RTH_MINUTES": 1}
    mixed = minute_rows(day)
    mixed[20]["Contract"] = "MNQU25"
    parsed, _ = gate.load_minutes(write_rows(mixed))
    assert gate.sessionize(parsed)[1] == {"MIXED_CONTRACT": 1}


def write_rows(rows: list[dict[str, str]]) -> Path:
    import tempfile

    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    handle.write(json.dumps(rows))
    handle.close()
    return Path(handle.name)


def test_cutoff_and_duplicate_rejection(tmp_path: Path) -> None:
    before = minute_rows(date(2025, 1, 2))
    after = minute_rows(date(2026, 3, 2))
    path = tmp_path / "input.json"
    path.write_text(json.dumps(before + after))
    parsed, skipped = gate.load_minutes(path)
    assert len(parsed) == 390 and skipped == 390
    path.write_text(json.dumps(before + [before[0]]))
    with pytest.raises(gate.GateError, match="duplicate"):
        gate.load_minutes(path)


def test_malformed_ohlc_is_refused(tmp_path: Path) -> None:
    rows = minute_rows(date(2025, 1, 2))
    rows[0]["High"] = "99"  # below the open and close
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(rows))
    with pytest.raises(gate.GateError, match="OHLC"):
        gate.load_minutes(path)


def test_adjacent_signal_holding_slots_are_nonoverlapping() -> None:
    bars = [gate.Bar(date(2025, 1, 2), slot, 100, 103, 100, 101) for slot in range(78)]
    slots = gate.signal_slots(bars)
    assert slots == list(range(76))
    assert all((left + 1) <= right for left, right in zip(slots, slots[1:]))


def test_identity_pairing_is_refused() -> None:
    sessions = [date(2025, 1, 2), date(2025, 1, 3)]
    bars = {day: [wick_bar(100, 103, 100, 101) for _ in range(78)] for day in sessions}
    with pytest.raises(gate.GateError, match="identity"):
        gate.shifted_outcomes(sessions, bars, {day: [0] for day in sessions}, 0)


def test_cluster_errors_and_mde_match_independent_arithmetic() -> None:
    values = [1.0, 3.0, 5.0, 7.0]
    session_se = gate.cluster_se(values, ["a", "a", "b", "b"])
    week_se = gate.cluster_se(values, ["w1", "w2", "w3", "w4"])
    assert session_se == pytest.approx(math.sqrt(32) / 4)
    assert week_se == pytest.approx(math.sqrt(20) / 4)
    assert max(session_se, week_se) == pytest.approx(session_se)
    assert gate.Z_SUM * session_se == pytest.approx(
        (1.6448536269514722 + 0.8416212335729143) * math.sqrt(32) / 4
    )


def test_valid_cli_is_terminal_and_never_promotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_input(tmp_path / "input.json")
    plan = tmp_path / "plan.md"
    plan.write_text("synthetic frozen plan")
    monkeypatch.setattr(gate, "PLAN_SHA256", gate.sha256(plan))
    # Give every session a valid first-bar wick and distinct next-bar movement.
    payload = json.loads(source.read_text())
    for row_index in range(0, len(payload), 390):
        payload[row_index].update(Open="100", High="103", Low="100", Close="101")
        for offset in range(1, 5):
            payload[row_index + offset].update(
                Open="101", High="101", Low="101", Close="101"
            )
        movement = (row_index // 390) % 3
        for offset in range(5, 10):
            payload[row_index + offset].update(
                Open="100",
                High=str(100 + movement),
                Low="99",
                Close=str(100 + movement),
            )
    source.write_text(json.dumps(payload))
    output = tmp_path / "out"
    assert (
        gate.main(
            [
                "--input",
                str(source),
                "--committed-plan",
                str(plan),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads((output / "report.json").read_text())
    assert report["verdict"] == "POWER_UNDETERMINED"
    assert report["evaluation_allowed"] is False
    assert report["signal_frequency"]["actual_signal_times_counted_only"] is True
    assert report["conditional_detectability"]["valid_shift_count"] == 3


def test_bad_plan_or_output_collision_refuses(tmp_path: Path) -> None:
    source = make_input(tmp_path / "input.json")
    plan = tmp_path / "plan.md"
    plan.write_text("wrong")
    output = tmp_path / "out"
    assert (
        gate.main(
            [
                "--input",
                str(source),
                "--committed-plan",
                str(plan),
                "--output-dir",
                str(output),
            ]
        )
        == 2
    )
    output.mkdir()
    assert (
        gate.main(
            [
                "--input",
                str(source),
                "--committed-plan",
                str(plan),
                "--output-dir",
                str(output),
            ]
        )
        == 2
    )
