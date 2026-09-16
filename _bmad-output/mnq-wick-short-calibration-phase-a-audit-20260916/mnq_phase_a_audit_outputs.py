"""Audit persisted-ledger consistency, not an independent raw-source rescan."""

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from fractions import Fraction
from zoneinfo import ZoneInfo

from mnq_phase_a_independent_audit import assert_number, reference_summary

if not __debug__:
    raise SystemExit("Audit refuses optimized execution: assertions are required.")

SCHEMA = "mnq-wick-short-calibration-phase-a/v1"
ROLE = "calibration-development-only"
COSTS = ("1.22", "2.22", "3.22")
FILES = {
    "report.json",
    "report.md",
    "eligibility.jsonl",
    "outcomes.jsonl",
    "sessions.jsonl",
}
QUANTILES = {"0", "0.01", "0.05", "0.25", "0.5", "0.75", "0.95", "0.99", "1"}
NY = ZoneInfo("America/New_York")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def week(day):
    year, number, _ = date.fromisoformat(day).isocalendar()
    return f"{year:04d}-W{number:02d}"


def local_stamp(value):
    stamp = datetime.fromisoformat(value)
    assert stamp.tzinfo is not None
    assert stamp.isoformat() == stamp.astimezone(NY).isoformat()
    return stamp


def check_interval(actual, expected):
    assert isinstance(actual, list) and len(actual) == 2
    for value, reference in zip(actual, expected):
        assert_number(value, reference, "interval endpoint")


def audit_metric(metric, values, days, gross_values=None, cost=0, exact_values=None):
    desc = metric["descriptive"]
    assert desc["observation_count"] == len(values)
    assert_number(desc["total"], sum(values), "total")
    results = []
    for key, groups in [
        ("session_clustered", days),
        ("iso_week_clustered", [week(day) for day in days]),
    ]:
        expected = reference_summary(
            values if gross_values is None else gross_values, groups
        )
        if exact_values is not None:
            expected = reference_summary(exact_values, groups)
        if gross_values is not None:
            expected["mean"] -= cost
            if expected["ci"] is not None:
                expected["ci"] = [value - cost for value in expected["ci"]]
        actual = metric[key]
        assert actual["N"] == expected["n"] and actual["G"] == expected["g"]
        assert_number(desc["mean"], expected["mean"], "mean")
        assert_number(desc["sample_sd"], expected["sd"], "sd")
        assert_number(actual["variance"], expected["variance"], "variance")
        sizes = Counter(groups)
        pairs = [(row["label"], row["count"]) for row in actual["cluster_sizes"]]
        assert len(pairs) == len(sizes) and dict(pairs) == dict(sizes)
        assert_number(
            actual["largest_cluster_fraction"],
            max(sizes.values()) / len(values),
            "largest cluster",
        )
        assert_number(
            actual["cluster_share_hhi"],
            sum((n / len(values)) ** 2 for n in sizes.values()),
            "cluster concentration",
        )
        assert actual["degrees_of_freedom"] == (
            expected["g"] - 1 if expected["g"] >= 2 else None
        )
        assert_number(actual["se"], expected["se"], "SE")
        assert_number(actual["t_critical"], expected["t_critical"], "t critical")
        if expected["ci"] is None:
            assert actual["status"] == "UNASSESSABLE" and actual["interval"] is None
            assert isinstance(actual["reason"], str) and actual["reason"]
        else:
            assert actual["status"] == "ASSESSABLE" and actual["reason"] is None
            check_interval(actual["interval"], expected["ci"])
        results.append(expected["ci"])
    envelope = metric["envelope"]
    assert envelope["role"] == "sensitivity-envelope-only"
    if all(result is not None for result in results):
        expected = [min(r[0] for r in results), max(r[1] for r in results)]
        assert envelope["status"] == "ASSESSABLE" and envelope["reason"] is None
        check_interval(envelope["interval"], expected)
    else:
        assert envelope["status"] == "UNASSESSABLE" and envelope["interval"] is None
        assert envelope["reason"] == "BOTH_GROUPINGS_REQUIRED"
    frequencies = Counter(values)
    ecdf = desc["distribution"]["ecdf"]
    assert [(r["value"], r["count"]) for r in ecdf] == sorted(frequencies.items())
    cumulative = 0
    for row in ecdf:
        cumulative += row["count"]
        assert_number(
            row["cumulative_probability"], cumulative / len(values), "ECDF probability"
        )
    quantiles = desc["distribution"]["quantiles"]
    assert set(quantiles) == QUANTILES
    ordered = sorted(values)
    for quantile, actual in quantiles.items():
        index = float(quantile) * (len(ordered) - 1)
        low = int(index)
        high = min(low + 1, len(ordered) - 1)
        expected = ordered[low] + (index - low) * (ordered[high] - ordered[low])
        assert_number(actual, expected, "quantile")


def audit_eligibility(rows):
    assert len({row["session_id"] for row in rows}) == len(rows)
    for row in rows:
        day = date.fromisoformat(row["session_id"])
        assert row["sample_role"] == ROLE and row["iso_week"] == week(row["session_id"])
        start = datetime.combine(day, time(9, 31), tzinfo=NY)
        expected = {start + timedelta(minutes=i) for i in range(390)}
        missing = {local_stamp(value) for value in row["missing_minute_labels"]}
        extra = {local_stamp(value) for value in row["unexpected_minute_labels"]}
        assert len(missing) == len(row["missing_minute_labels"]) and missing <= expected
        assert (
            len(extra) == len(row["unexpected_minute_labels"]) and not extra & expected
        )
        observed = expected - missing | extra
        assert row["expected_minute_count"] == 390
        assert row["observed_minute_count"] == len(observed)
        assert local_stamp(row["first_minute_label_local"]) == min(observed)
        assert local_stamp(row["last_minute_label_local"]) == max(observed)
        assert all(
            stamp.date() == day and time(9, 31) <= stamp.time() <= time(16)
            for stamp in observed
        )
        contracts = row["contract_minute_counts"]
        assert row["contracts"] == sorted(contracts)
        assert all(
            isinstance(n, int) and not isinstance(n, bool) and n > 0
            for n in contracts.values()
        )
        assert sum(contracts.values()) == len(observed)
        reasons = []
        if missing or extra:
            reasons.append("INCOMPLETE_RTH_MINUTES")
        if len(contracts) != 1:
            reasons.append("MIXED_CONTRACT")
        assert row["exclusion_reasons"] == reasons
        assert row["primary_exclusion_reason"] == (reasons[0] if reasons else None)
        assert row["eligible"] is (not reasons)
        if reasons:
            assert row["signal_count"] is None
            assert row["signal_count_null_reason"] == "INELIGIBLE_SESSION"
        else:
            assert row["signal_count_null_reason"] is None


def audit(output):
    complete = json.loads((output / "COMPLETE.json").read_text())
    assert set(complete) == {
        "schema_version",
        "status",
        "evaluation_allowed",
        "manifest_sha256",
    }
    assert (
        complete["schema_version"] == SCHEMA
        and complete["status"] == "COMPLETE_CALIBRATION_ONLY"
    )
    assert complete["evaluation_allowed"] is False
    assert (
        complete["manifest_sha256"]
        == hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert set(manifest) == {
        "schema_version",
        "sample_role",
        "evaluation_allowed",
        "output_sha256",
    }
    assert manifest["schema_version"] == SCHEMA and manifest["sample_role"] == ROLE
    assert (
        manifest["evaluation_allowed"] is False
        and set(manifest["output_sha256"]) == FILES
    )
    for name, digest in manifest["output_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest, name
    report = json.loads((output / "report.json").read_text())
    assert (
        report["schema_version"] == SCHEMA
        and report["status"] == "COMPLETE_CALIBRATION_ONLY"
    )
    assert report["phase"] == "A" and report["sample_role"] == ROLE
    assert report["original_gate_verdict"] == "POWER_UNDETERMINED"
    assert (
        report["evaluation_allowed"] is False
        and report["confirmation_authorized"] is False
    )
    assert report["counts_reconciled_before_alignment"] is True
    assert report["cost_scenarios_dollars"] == [1.22, 2.22, 3.22]
    assert report["point_value_dollars"] == 2
    eligibility = read_rows(output / "eligibility.jsonl")
    outcomes = read_rows(output / "outcomes.jsonl")
    sessions = read_rows(output / "sessions.jsonl")
    assert (len(eligibility), len(outcomes), len(sessions)) == (576, 585, 515)
    audit_eligibility(eligibility)
    assert len({row["signal_id"] for row in outcomes}) == len(outcomes)
    assert len({row["session_id"] for row in sessions}) == len(sessions)
    eligible = {row["session_id"]: row for row in eligibility if row["eligible"]}
    assert set(eligible) == {row["session_id"] for row in sessions}
    exclusions = Counter(
        row["primary_exclusion_reason"] for row in eligibility if not row["eligible"]
    )
    assert exclusions == {"INCOMPLETE_RTH_MINUTES": 23, "MIXED_CONTRACT": 38}
    counts = {
        "rth_sessions_seen": len(eligibility),
        "eligible_sessions": len(sessions),
        "excluded_sessions": len(eligibility) - len(sessions),
        "exclusion_reasons": dict(exclusions),
        "post_cutoff_records_skipped": 62277,
        "total_signals": len(outcomes),
        "sessions_with_signal": sum(row["signal_count"] > 0 for row in sessions),
        "sessions_without_signal": sum(row["signal_count"] == 0 for row in sessions),
        "signals_per_eligible_session": len(outcomes) / len(sessions),
    }
    assert report["counts"] == counts
    assert (
        counts["sessions_with_signal"] == 351
        and counts["sessions_without_signal"] == 164
    )
    by_day = defaultdict(list)
    seen_minutes = {}
    for row in outcomes:
        session = row["session_id"]
        assert row["sample_role"] == ROLE and row["iso_week"] == week(session)
        assert row["contract"] == eligible[session]["contracts"][0]
        slot = row["signal_slot"]
        assert (
            isinstance(slot, int)
            and 0 <= slot <= 75
            and row["following_slot"] == slot + 1
        )
        assert row["signal_id"] == f"{session}/slot-{slot:02d}"
        label = local_stamp(row["signal_bar_label_local"])
        assert (
            label.hour * 60 + label.minute == 9 * 60 + 35 + 5 * slot
            and label.second == label.microsecond == 0
        )
        assert local_stamp(row["reference_interval_start_local"]) == label
        assert local_stamp(row["reference_interval_end_local"]) == label + timedelta(
            minutes=5
        )
        assert label.date().isoformat() == session
        for key, expected in [
            ("signal_bar_label_utc", label),
            ("reference_interval_start_utc", label),
            ("reference_interval_end_utc", label + timedelta(minutes=5)),
        ]:
            assert datetime.fromisoformat(row[key]) == expected
        for key, offset in [
            ("signal_component_minutes", -4),
            ("reference_component_minutes", 1),
        ]:
            assert len(row[key]) == 5
            for index, minute in enumerate(row[key]):
                stamp = local_stamp(minute["minute_label_local"])
                assert stamp == label + timedelta(minutes=offset + index)
                assert datetime.fromisoformat(minute["minute_label_utc"]) == stamp
                assert datetime.fromisoformat(minute["source_record_id"]) == stamp
                assert minute["contract"] == row["contract"]
                assert stamp.astimezone(timezone.utc) < datetime(
                    2026, 3, 1, tzinfo=timezone.utc
                )
                assert all(
                    math.isfinite(minute[k]) for k in ("open", "high", "low", "close")
                )
                assert (
                    minute["high"]
                    >= max(minute["open"], minute["close"])
                    >= min(minute["open"], minute["close"])
                    >= minute["low"]
                )
                if stamp in seen_minutes:
                    assert seen_minutes[stamp] == minute
                seen_minutes[stamp] = minute
        parts = row["signal_component_minutes"]
        opening, close = parts[0]["open"], parts[-1]["close"]
        high, low = max(x["high"] for x in parts), min(x["low"] for x in parts)
        assert row["signal_bar_ohlc"] == {
            "open": opening,
            "high": high,
            "low": low,
            "close": close,
        }
        assert row["body"] == abs(close - opening) > 0
        assert row["upper_wick"] == high - max(opening, close) >= 2 * row["body"]
        assert row["lower_wick"] == min(opening, close) - low <= 0.1 * row["upper_wick"]
        assert row["next_open"] == row["reference_component_minutes"][0]["open"]
        assert row["next_close"] == row["reference_component_minutes"][-1]["close"]
        assert_number(
            row["gross_dollars"],
            2 * (row["next_open"] - row["next_close"]),
            "short dollars",
        )
        assert set(row["net_dollars"]) == set(COSTS)
        for cost, value in row["net_dollars"].items():
            assert_number(value, row["gross_dollars"] - float(cost), "net dollars")
        by_day[session].append(row)
    for row in sessions:
        day = row["session_id"]
        assert row["sample_role"] == ROLE and row["iso_week"] == week(day)
        assert row["contract"] == eligible[day]["contracts"][0]
        signals = by_day[day]
        assert row["signal_count"] == eligible[day]["signal_count"] == len(signals)
        gross = sum(r["gross_dollars"] for r in signals)
        assert_number(row["gross_total_dollars"], gross, "session gross")
        assert set(row["net_total_dollars"]) == set(COSTS)
        for cost, value in row["net_total_dollars"].items():
            assert_number(value, gross - float(cost) * len(signals), "session net")
        for signal in signals:
            assert signal["session_signal_count"] == row["signal_count"]
            assert signal["session_gross_total_dollars"] == row["gross_total_dollars"]
            assert signal["session_net_total_dollars"] == row["net_total_dollars"]
    signal_days = [row["session_id"] for row in outcomes]
    session_days = [row["session_id"] for row in sessions]
    stats = report["statistics"]
    gross_values = [r["gross_dollars"] for r in outcomes]
    audit_metric(stats["per_signal"]["gross_dollars"], gross_values, signal_days)
    audit_metric(
        stats["per_eligible_session"]["signal_count"],
        [r["signal_count"] for r in sessions],
        session_days,
    )
    audit_metric(
        stats["per_eligible_session"]["gross_total_dollars"],
        [r["gross_total_dollars"] for r in sessions],
        session_days,
    )
    assert set(stats["per_signal"]["net_dollars"]) == set(COSTS)
    assert set(stats["per_eligible_session"]["net_total_dollars"]) == set(COSTS)
    for cost in COSTS:
        audit_metric(
            stats["per_signal"]["net_dollars"][cost],
            [r["net_dollars"][cost] for r in outcomes],
            signal_days,
            gross_values,
            float(cost),
        )
        exact = [
            Fraction(str(r["gross_total_dollars"])) - Fraction(cost) * r["signal_count"]
            for r in sessions
        ]
        audit_metric(
            stats["per_eligible_session"]["net_total_dollars"][cost],
            [r["net_total_dollars"][cost] for r in sessions],
            session_days,
            exact_values=exact,
        )
    print(
        "INDEPENDENT AUDIT PASSED: exact publication inventory, persisted-ledger "
        "counts/contracts/coverage, monetary arithmetic and all nine estimands. "
        "No independent raw-source rescan was performed."
    )


if __name__ == "__main__":
    audit(Path(sys.argv[1]))
