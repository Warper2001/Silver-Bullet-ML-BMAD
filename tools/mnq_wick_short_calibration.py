"""Registered Phase A historical reference calibration; never a strategy verdict.

The production CLI accepts only the registered container, in the designated
worktree, after verifying committed implementation and all canonical pins.
Synthetic tests exercise the pure helpers without reading market data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Hashable, Sequence

import numpy as np
from scipy import stats

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import mnq_wick_short_power_gate as gate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AUTHORIZED_ROOT = Path(
    "/root/Silver-Bullet-ML-BMAD/.claude/worktrees/"
    "mnq-wick-short-calibration-phase-a"
)
INPUT = Path("/root/mnq_historical.json")
INPUT_SHA256 = "e7aed8ba786436ba80f4b081d3b7a4ee97b06bd3e8cc347c035547ec57dcb924"
REGISTRATION = "_bmad-output/preregistration_mnq_wick_short_calibration_20260916.md"
REGISTRATION_REVISION = "3167d5354d2b71af413e9e2404b69efa82df96fc"
REGISTRATION_SHA256 = "429dcb566b6b03a7565fc389d9c0aa048099e48718f20024bc52cd4919135a84"
GATE_REVISION = "a04fde464d499bb9ca8f9a663657a984bd28b0f5"
RESULT_REVISION = "8492ae12212d85e0b420240c36fa5742a8cae289"
GATE_REPORT = "_bmad-output/mnq-wick-short-power-20260916/report.json"
PINS = {
    REGISTRATION: (REGISTRATION_REVISION, REGISTRATION_SHA256),
    "_bmad-output/preregistration_mnq_wick_short_power_20260916.md": (
        GATE_REVISION,
        "ed3fad3c455100ca4b12a918fa658b7a717f011a4dfdfd0cf292c1e66784daff",
    ),
    "tools/mnq_wick_short_power_gate.py": (
        GATE_REVISION,
        "adb16fda09e0324ca30180589faf39e7c8580d46c8032e806e8556125244b0da",
    ),
    "_bmad-output/specs/spec-mnq-wick-short-power/SPEC.md": (
        GATE_REVISION,
        "0a93788286199fc420bb72ec1faf7c568722f527e1ea5dd8c9e7a6f434f1384d",
    ),
    "_bmad-output/specs/spec-mnq-wick-short-power/mechanics.md": (
        GATE_REVISION,
        "d943106f4bf940992937cd7993ffd516a265d22ed04f467962684cd077d74e72",
    ),
    "_bmad-output/specs/spec-mnq-wick-short-power/data-handling.md": (
        GATE_REVISION,
        "35574935330b56c91cfaa9d3a7896f02b08c11f290c71beabb69997ed958526a",
    ),
    "_bmad-output/specs/spec-mnq-wick-short-power/statistical-assumptions.md": (
        GATE_REVISION,
        "b1f1f34ec3fb175d05bf53446a5f4adb8c0ced34771f8db33d1ca31cae74de10",
    ),
    GATE_REPORT: (
        RESULT_REVISION,
        "4b33288c6abfefe2d0a36f93365e7841784e29f7c3630a1a13694b3b40930546",
    ),
    "_bmad-output/mnq-wick-short-power-20260916/report.md": (
        RESULT_REVISION,
        "4c557946a7fb798a7ef10d2c158cede6e38bb7968e5720a0767c5aba04d3779c",
    ),
}
IMPLEMENTATION = (
    "tools/mnq_wick_short_calibration.py",
    "tests/unit/test_mnq_wick_short_calibration.py",
    "tests/unit/test_mnq_wick_short_power_gate.py",
    "pyproject.toml",
)
SCHEMA = "mnq-wick-short-calibration-phase-a/v1"
ROLE = "calibration-development-only"
LIMITATIONS = [
    "Calibration only: all historical rows were previously exposed; none are "
    "untouched confirmation evidence. No profitability or future-edge verdict.",
    "Historical next-open/close prices do not establish timely signal "
    "availability, executable fills, routing latency or measured costs.",
    "Costs are the frozen $1.22 fee plus $0/$1/$2 adverse execution assumptions; "
    "they are not reconstructed historical charges or measured future costs.",
    "Approximate Student-t intervals assume finite variance, adequate cluster "
    "information, dependence represented within clusters and independence "
    "across clusters. Weekly clustering does not resolve cross-week dependence, "
    "regime change, feed errors or prior design exposure.",
    "The outer interval is a sensitivity envelope, not a third confidence "
    "procedure; neither clustering nor the envelope guarantees 95% coverage.",
    "Calendar/contract transition flags are descriptive observations only; "
    "they introduce no eligibility or trading filters. Calendar closures are "
    "not inferred from absent dates in this input.",
    "Original POWER_UNDETERMINED and evaluation_allowed=false remain binding. "
    "No fresh power verdict, Phase B, confirmation, or trading is performed.",
]


class CalibrationError(ValueError):
    """The registered calculation cannot proceed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_command(root: Path, args: Sequence[str]) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, check=False
    )
    if result.returncode:
        raise CalibrationError(
            f"git verification failed: {result.stderr.decode().strip()}"
        )
    return result.stdout


def git_audit(root: Path) -> dict[str, Any]:
    """Required local status/divergence audit before further Git operations."""
    status = _git_command(root, ["status", "--porcelain"]).decode()
    ahead = _git_command(root, ["log", "--format=%H", "origin/main..HEAD"])
    behind = _git_command(root, ["log", "--format=%H", "HEAD..origin/main"])
    return {
        "status_porcelain": status,
        "ahead_of_origin_main": len(ahead.splitlines()),
        "behind_origin_main": len(behind.splitlines()),
    }


def git(root: Path, *args: str) -> bytes:
    git_audit(root)
    return _git_command(root, args)


def verify_provenance(root: Path) -> dict[str, Any]:
    audit = git_audit(root)
    head = git(root, "rev-parse", "HEAD").decode().strip()
    artifacts = {}
    for relative, (revision, expected) in PINS.items():
        path = root / relative
        if path.is_symlink() or sha256(path) != expected:
            raise CalibrationError(f"pinned artifact hash mismatch: {relative}")
        canonical = git(root, "show", f"{revision}:{relative}")
        if hashlib.sha256(canonical).hexdigest() != expected:
            raise CalibrationError(f"canonical artifact mismatch: {relative}")
        git(root, "merge-base", "--is-ancestor", revision, head)
        if git(root, "show", f"HEAD:{relative}") != canonical:
            raise CalibrationError(f"HEAD differs from canonical: {relative}")
        artifacts[relative] = {"sha256": expected, "revision": revision}
    implementation = {}
    for relative in IMPLEMENTATION:
        path = root / relative
        committed = git(root, "show", f"HEAD:{relative}")
        if path.is_symlink() or path.read_bytes() != committed:
            raise CalibrationError(f"uncommitted implementation: {relative}")
        if git(root, "diff", "--cached", "HEAD", "--", relative):
            raise CalibrationError(f"staged implementation change: {relative}")
        revision_bytes = git(root, "log", "-1", "--format=%H", "--", relative)
        implementation[relative] = {
            "sha256": sha256(path),
            "revision": revision_bytes.decode().strip(),
        }
    return {
        "run_revision": head,
        "git_audit": audit,
        "canonical_artifacts": artifacts,
        "implementation": implementation,
    }


def runtime_provenance() -> dict[str, Any]:
    dependencies = {}
    for name in ("numpy", "scipy", "pytest"):
        distribution = importlib.metadata.distribution(name)
        files = distribution.files
        if files is None:
            raise CalibrationError(f"dependency manifest absent: {name}")
        inventory = hashlib.sha256()
        count = 0
        for entry in sorted(files, key=str):
            # Installed files, including extension libraries, bind actual bytes.
            path = Path(str(distribution.locate_file(entry)))
            if path.is_file():
                inventory.update(str(entry).encode() + b"\0")
                inventory.update(sha256(path).encode() + b"\n")
                count += 1
        dependencies[name] = {
            "version": distribution.version,
            "installed_files_sha256": inventory.hexdigest(),
            "hashed_file_count": count,
        }
    return {
        "python_version": platform.python_version(),
        "interpreter": sys.executable,
        "interpreter_resolved": str(Path(sys.executable).resolve()),
        "interpreter_sha256": sha256(Path(sys.executable)),
        "platform": platform.platform(),
        "dependencies": dependencies,
    }


def reserve_output(root: Path, destination: Path) -> Path:
    """Reserve an exclusive direct child before any market-container access."""
    if root.resolve() != AUTHORIZED_ROOT or not (root / ".git").is_file():
        raise CalibrationError("production execution requires the authorized worktree")
    parent = root / "_bmad-output"
    if parent.is_symlink() or parent.resolve() != parent:
        raise CalibrationError("output parent must be the worktree research directory")
    absolute = Path(os.path.abspath(destination))
    if absolute.parent != parent or not absolute.name.startswith(
        "mnq-wick-short-calibration-phase-a-"
    ):
        raise CalibrationError("prohibited output destination")
    if absolute.is_symlink() or absolute.exists():
        raise CalibrationError("output directory already exists")
    absolute.mkdir(exist_ok=False)
    return absolute


def validate_input_path(path: Path) -> None:
    if path != INPUT or path.is_symlink() or path.resolve() != INPUT:
        raise CalibrationError("only the bound /root/mnq_historical.json is allowed")


def validate_input_hash(path: Path) -> None:
    if sha256(path) != INPUT_SHA256:
        raise CalibrationError("bound input hash mismatch")


def iso_week(day: date) -> str:
    year, week = gate.week_key(day)
    return f"{year:04d}-W{week:02d}"


def eligible_ledgers(minutes: Sequence[gate.Minute], skipped: int) -> tuple[
    list[dict[str, Any]],
    dict[date, list[gate.Bar]],
    dict[date, list[int]],
    dict[str, Any],
]:
    """Count eligibility/signals only; do not pair aligned outcomes here."""
    eligible, exclusions = gate.sessionize(minutes)
    buckets: dict[date, list[gate.Minute]] = defaultdict(list)
    for minute in minutes:
        local = minute.timestamp.astimezone(gate.NY)
        if time(9, 31) <= local.time() <= time(16):
            buckets[local.date()].append(minute)
    bars = {day: gate.bars_for_session(day, rows) for day, rows in eligible.items()}
    signals = {day: gate.signal_slots(group) for day, group in bars.items()}
    ledger = []
    previous_contracts: list[str] | None = None
    for day, rows in sorted(buckets.items()):
        observed = {row.timestamp.astimezone(gate.NY) for row in rows}
        expected = gate.expected_rth_minutes(day)
        contracts = sorted({row.contract for row in rows})
        reasons = []
        if observed != expected:
            reasons.append("INCOMPLETE_RTH_MINUTES")
        if len(contracts) != 1:
            reasons.append("MIXED_CONTRACT")
        flags = []
        if day.weekday() >= 5:
            flags.append("WEEKEND_RTH_DATE")
        if observed != expected:
            flags.append("COVERAGE_ANOMALY_POSSIBLE_SHORT_SESSION_OR_DATA_GAP")
        if len(contracts) != 1:
            flags.append("MULTIPLE_CONTRACTS_OBSERVED")
        if previous_contracts is not None and contracts != previous_contracts:
            flags.append("CONTRACT_SET_CHANGED_FROM_PREVIOUS_OBSERVED_DATE")
        ledger.append(
            {
                "session_id": day.isoformat(),
                "iso_week": iso_week(day),
                "contracts": contracts,
                "contract_minute_counts": dict(Counter(row.contract for row in rows)),
                "eligible": day in eligible,
                "primary_exclusion_reason": reasons[0] if reasons else None,
                "exclusion_reasons": reasons,
                "expected_minute_count": 390,
                "observed_minute_count": len(rows),
                "missing_minute_labels": sorted(
                    x.isoformat() for x in expected - observed
                ),
                "unexpected_minute_labels": sorted(
                    x.isoformat() for x in observed - expected
                ),
                "first_minute_label_local": min(observed).isoformat(),
                "last_minute_label_local": max(observed).isoformat(),
                "signal_count": len(signals[day]) if day in eligible else None,
                "signal_count_null_reason": (
                    None if day in eligible else "INELIGIBLE_SESSION"
                ),
                "calendar_roll_audit_flags": flags,
                "calendar_audit_role": "observations-only; no additional filters",
                "sample_role": ROLE,
            }
        )
        previous_contracts = contracts
    total = sum(len(slots) for slots in signals.values())
    with_signal = sum(bool(slots) for slots in signals.values())
    counts = {
        "rth_sessions_seen": len(buckets),
        "eligible_sessions": len(eligible),
        "excluded_sessions": sum(exclusions.values()),
        "exclusion_reasons": exclusions,
        "post_cutoff_records_skipped": skipped,
        "total_signals": total,
        "sessions_with_signal": with_signal,
        "sessions_without_signal": len(eligible) - with_signal,
        "signals_per_eligible_session": total / len(eligible) if eligible else None,
    }
    return ledger, bars, signals, counts


def reconcile_counts(counts: dict[str, Any], original: dict[str, Any]) -> None:
    expected = dict(original["eligibility"])
    for key in (
        "total_signals",
        "sessions_with_signal",
        "signals_per_eligible_session",
    ):
        expected[key] = original["signal_frequency"][key]
    expected["sessions_without_signal"] = (
        expected["eligible_sessions"] - expected["sessions_with_signal"]
    )
    if original["verdict"] != "POWER_UNDETERMINED" or original["evaluation_allowed"]:
        raise CalibrationError("original gate boundary mismatch")
    if counts != expected:
        discrepancies = {
            key: {"observed": counts.get(key), "bound_gate": value}
            for key, value in expected.items()
            if counts.get(key) != value
        }
        raise CalibrationError(f"gate count reconciliation mismatch: {discrepancies}")


def component_minutes(
    day: date, slot: int, lookup: dict[datetime, gate.Minute]
) -> list[dict[str, Any]]:
    start = datetime.combine(day, time(9, 31), tzinfo=gate.NY)
    result = []
    for offset in range(5):
        local = start + timedelta(minutes=slot * 5 + offset)
        row = lookup[local]
        result.append(
            {
                "source_record_id": row.timestamp.isoformat(),
                "minute_label_utc": row.timestamp.isoformat(),
                "minute_label_local": local.isoformat(),
                "contract": row.contract,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
            }
        )
    return result


def aligned_ledgers(
    minutes: Sequence[gate.Minute],
    bars: dict[date, list[gate.Bar]],
    signals: dict[date, list[int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair each counted signal with its own following bar; include zero sessions."""
    lookup = {minute.timestamp.astimezone(gate.NY): minute for minute in minutes}
    outcomes = []
    sessions = []
    for day in sorted(bars):
        session_outcomes = []
        for slot in signals[day]:
            bar, following = bars[day][slot], bars[day][slot + 1]
            signal_components = component_minutes(day, slot, lookup)
            following_components = component_minutes(day, slot + 1, lookup)
            label = datetime.combine(day, time(9, 35), tzinfo=gate.NY) + timedelta(
                minutes=slot * 5
            )
            gross = gate.POINT_VALUE * (following.open - following.close)
            row = {
                "signal_id": f"{day.isoformat()}/slot-{slot:02d}",
                "session_id": day.isoformat(),
                "iso_week": iso_week(day),
                "contract": signal_components[0]["contract"],
                "signal_slot": slot,
                "following_slot": slot + 1,
                "signal_bar_label_local": label.isoformat(),
                "signal_bar_label_utc": label.astimezone(timezone.utc).isoformat(),
                "reference_interval_start_local": label.isoformat(),
                "reference_interval_end_local": (
                    label + timedelta(minutes=5)
                ).isoformat(),
                "reference_interval_start_utc": label.astimezone(
                    timezone.utc
                ).isoformat(),
                "reference_interval_end_utc": (label + timedelta(minutes=5))
                .astimezone(timezone.utc)
                .isoformat(),
                "signal_component_minutes": signal_components,
                "reference_component_minutes": following_components,
                "body": abs(bar.close - bar.open),
                "upper_wick": bar.high - max(bar.open, bar.close),
                "lower_wick": min(bar.open, bar.close) - bar.low,
                "signal_bar_ohlc": {
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                },
                "next_open": following.open,
                "next_close": following.close,
                "gross_dollars": gross,
                "net_dollars": {f"{cost:.2f}": gross - cost for cost in gate.COSTS},
                "sample_role": ROLE,
                "price_role": "historical-reference; execution availability unverified",
            }
            session_outcomes.append(row)
        count = len(session_outcomes)
        gross_total = math.fsum(row["gross_dollars"] for row in session_outcomes)
        session_row = {
            "session_id": day.isoformat(),
            "iso_week": iso_week(day),
            "contract": component_minutes(day, 0, lookup)[0]["contract"],
            "signal_count": count,
            "gross_total_dollars": gross_total,
            "net_total_dollars": {
                f"{cost:.2f}": gross_total - cost * count for cost in gate.COSTS
            },
            "sample_role": ROLE,
        }
        for row in session_outcomes:
            row["session_signal_count"] = count
            row["session_gross_total_dollars"] = gross_total
            row["session_net_total_dollars"] = session_row["net_total_dollars"]
        outcomes.extend(session_outcomes)
        sessions.append(session_row)
    return outcomes, sessions


def descriptive(values: Sequence[float]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "observation_count": len(values),
        "total": None,
        "mean": None,
        "sample_sd": None,
        "distribution": None,
        "null_reasons": {},
    }
    reason = "NO_OBSERVATIONS" if not values else None
    if any(not math.isfinite(value) for value in values):
        reason = "NONFINITE_OBSERVATION"
    if reason:
        result["null_reasons"] = {
            field: reason for field in ("total", "mean", "sample_sd", "distribution")
        }
        return result
    try:
        total = math.fsum(values)
        mean = total / len(values)
        variance = (
            math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
            if len(values) > 1
            else None
        )
        if variance is not None and not math.isfinite(variance):
            raise OverflowError
    except (OverflowError, ValueError):
        result["null_reasons"] = {
            field: "INVALID_NUMERIC_ARITHMETIC"
            for field in ("total", "mean", "sample_sd", "distribution")
        }
        return result
    result.update(total=total, mean=mean)
    if len(values) < 2:
        result["null_reasons"]["sample_sd"] = "FEWER_THAN_TWO_OBSERVATIONS"
    else:
        assert variance is not None
        result["sample_sd"] = 0.0 if min(values) == max(values) else math.sqrt(variance)
    frequencies = Counter(values)
    cumulative = 0
    ecdf = []
    for value, count in sorted(frequencies.items()):
        cumulative += count
        ecdf.append(
            {
                "value": value,
                "count": count,
                "cumulative_probability": cumulative / len(values),
            }
        )
    result["distribution"] = {
        "quantile_method": "linear; descriptive only, not trading thresholds",
        "quantiles": {
            str(q): float(np.quantile(values, q))
            for q in (0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1)
        },
        "ecdf": ecdf,
    }
    return result


def cluster_interval(
    values: Sequence[float],
    labels: Sequence[Hashable],
    exact_values: Sequence[Fraction] | None = None,
) -> dict[str, Any]:
    if len(values) != len(labels):
        raise CalibrationError("observation/cluster label length mismatch")
    if exact_values is not None and len(exact_values) != len(values):
        raise CalibrationError("exact monetary observation length mismatch")
    groups: dict[Hashable, list[float]] = defaultdict(list)
    for value, label in zip(values, labels):
        groups[label].append(value)
    n, g = len(values), len(groups)
    result: dict[str, Any] = {
        "status": "UNASSESSABLE",
        "reason": None,
        "N": n,
        "G": g,
        "degrees_of_freedom": g - 1 if g >= 2 else None,
        "variance": None,
        "se": None,
        "t_critical": None,
        "interval": None,
        "cluster_sizes": [
            {"label": str(label), "count": len(group)}
            for label, group in groups.items()
        ],
        "largest_cluster_fraction": max(map(len, groups.values())) / n if n else None,
        "cluster_share_hhi": (
            math.fsum((len(group) / n) ** 2 for group in groups.values()) if n else None
        ),
    }
    if n < 2:
        result["reason"] = "FEWER_THAN_TWO_OBSERVATIONS"
    elif g < 2:
        result["reason"] = "FEWER_THAN_TWO_NONEMPTY_CLUSTERS"
    elif any(not math.isfinite(value) for value in values):
        result["reason"] = "NONFINITE_OBSERVATION"
    elif (
        min(exact_values) == max(exact_values)
        if exact_values is not None
        else min(values) == max(values)
    ):
        # Exact equality must precede floating-point centering (e.g. 0.1).
        result["reason"] = "CONSTANT_OBSERVATIONS_ZERO_VARIANCE"
    else:
        try:
            # Exact rational arithmetic on the supplied binary floats avoids
            # inventing positive variance from cancellation in balanced clusters.
            # No tolerance or hand-set variance threshold is needed.
            numbers = (
                list(map(Fraction, values)) if exact_values is None else exact_values
            )
            exact_mean = sum(numbers, Fraction()) / n
            mean = float(exact_mean)
            exact_groups: dict[Hashable, list[Fraction]] = defaultdict(list)
            for number, label in zip(numbers, labels):
                exact_groups[label].append(number)
            residuals = [
                sum(group, Fraction()) - len(group) * exact_mean
                for group in exact_groups.values()
            ]
            variance = float(
                Fraction(g, g - 1) * sum(value**2 for value in residuals) / n**2
            )
            if not math.isfinite(variance) or variance <= 0:
                raise ValueError
            se = math.sqrt(variance)
            critical = float(stats.t.ppf(0.975, g - 1))
            interval = [mean - critical * se, mean + critical * se]
            if not all(math.isfinite(value) for value in interval + [critical, se]):
                raise ValueError
        except (ValueError, OverflowError):
            result["reason"] = "INVALID_OR_NONPOSITIVE_CLUSTER_VARIANCE"
        else:
            result.update(
                status="ASSESSABLE",
                variance=variance,
                se=se,
                t_critical=critical,
                interval=interval,
            )
    return result


def estimand(
    values: Sequence[float],
    days: Sequence[date],
    exact_values: Sequence[Fraction] | None = None,
) -> dict[str, Any]:
    session = cluster_interval(values, days, exact_values)
    week = cluster_interval(values, [iso_week(day) for day in days], exact_values)
    envelope: dict[str, Any] = {
        "status": "UNASSESSABLE",
        "interval": None,
        "reason": "BOTH_GROUPINGS_REQUIRED",
        "role": "sensitivity-envelope-only",
    }
    if session["interval"] is not None and week["interval"] is not None:
        envelope.update(
            status="ASSESSABLE",
            reason=None,
            interval=[
                min(session["interval"][0], week["interval"][0]),
                max(session["interval"][1], week["interval"][1]),
            ],
        )
    return {
        "descriptive": descriptive(values),
        "session_clustered": session,
        "iso_week_clustered": week,
        "envelope": envelope,
    }


def signal_cost_estimand(
    gross: dict[str, Any], values: Sequence[float], cost: float
) -> dict[str, Any]:
    """Translate gross inference before rounded subtraction can change centering."""
    result = deepcopy(gross)
    result["descriptive"] = descriptive(values)
    desc = result["descriptive"]
    original = gross["descriptive"]
    desc["sample_sd"] = original["sample_sd"]
    if original["mean"] is not None:
        desc["mean"] = original["mean"] - cost
    if original["total"] is not None:
        desc["total"] = original["total"] - cost * original["observation_count"]
    for grouping in ("session_clustered", "iso_week_clustered", "envelope"):
        interval = gross[grouping]["interval"]
        if interval is not None:
            result[grouping]["interval"] = [value - cost for value in interval]
    return result


def summarize(
    outcomes: list[dict[str, Any]], sessions: list[dict[str, Any]]
) -> dict[str, Any]:
    signal_days = [date.fromisoformat(row["session_id"]) for row in outcomes]
    session_days = [date.fromisoformat(row["session_id"]) for row in sessions]
    gross = estimand([row["gross_dollars"] for row in outcomes], signal_days)
    return {
        "per_signal": {
            "gross_dollars": gross,
            "net_dollars": {
                f"{cost:.2f}": signal_cost_estimand(
                    gross, [row["net_dollars"][f"{cost:.2f}"] for row in outcomes], cost
                )
                for cost in gate.COSTS
            },
        },
        "per_eligible_session": {
            "signal_count": estimand(
                [float(row["signal_count"]) for row in sessions], session_days
            ),
            "gross_total_dollars": estimand(
                [row["gross_total_dollars"] for row in sessions], session_days
            ),
            "net_total_dollars": {
                f"{cost:.2f}": estimand(
                    [row["net_total_dollars"][f"{cost:.2f}"] for row in sessions],
                    session_days,
                    [
                        Fraction(row["gross_total_dollars"])
                        - Fraction(f"{cost:.2f}") * row["signal_count"]
                        for row in sessions
                    ],
                )
                for cost in gate.COSTS
            },
        },
    }


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MNQ wick-short Phase A historical calibration",
        "",
        "Calibration only. Original gate: **POWER_UNDETERMINED**; "
        "`evaluation_allowed=false`.",
        "",
        f"Sample role: `{ROLE}`. Completed: {report['completed_at_utc']}.",
        "",
        "Counts reconciled to the original gate **before** aligned outcomes: "
        + json.dumps(report["counts"], sort_keys=True),
        "",
        "All eligible zero-signal sessions are included in session estimands; "
        "excluded sessions are separate.",
        "",
        "| Estimand | N | Total | Mean | Sample SD | Session G / df / SE / 95% CI "
        "| Week G / df / SE / 95% CI | Envelope |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    metrics = [("Gross $/signal", report["statistics"]["per_signal"]["gross_dollars"])]
    metrics += [
        (f"Net $/signal (cost ${cost})", metric)
        for cost, metric in report["statistics"]["per_signal"]["net_dollars"].items()
    ]
    session = report["statistics"]["per_eligible_session"]
    metrics += [
        ("Signals/eligible session", session["signal_count"]),
        ("Gross $/eligible session", session["gross_total_dollars"]),
    ]
    metrics += [
        (f"Net $/eligible session (cost ${cost})", metric)
        for cost, metric in session["net_total_dollars"].items()
    ]
    for label, metric in metrics:
        desc = metric["descriptive"]
        intervals = []
        for grouping in ("session_clustered", "iso_week_clustered"):
            item = metric[grouping]
            intervals.append(
                f"{item['G']} / {item['degrees_of_freedom']} / {item['se']} / "
                f"{item['interval'] or item['reason']}"
            )
        envelope = metric["envelope"]
        lines.append(
            f"| {label} | {desc['observation_count']} | {desc['total']} | "
            f"{desc['mean']} | {desc['sample_sd']} | {intervals[0]} | "
            f"{intervals[1]} | {envelope['interval'] or envelope['reason']} |"
        )
    lines += [
        "",
        "Distributions, ECDFs, cluster sizes/concentration, null reasons and "
        "complete provenance are in `report.json`; individual observations are "
        "in the three JSONL ledgers.",
        "",
    ]
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines += [
        "",
        "Output byte hashes are in `manifest.json`, including this report. The "
        "manifest's own hash is in `COMPLETE.json`; its final Git commit binds "
        "both without a self-referential hash.",
        "",
    ]
    return "\n".join(lines)


def publish(
    output: Path,
    report: dict[str, Any],
    eligibility: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
) -> None:
    """Publish only inside a reserved empty directory; COMPLETE is written last."""
    if not output.is_dir() or any(output.iterdir()):
        raise CalibrationError("publication requires a reserved empty directory")
    contents = {
        "report.json": json_bytes(report),
        "report.md": markdown(report).encode(),
    }
    for name, rows in (
        ("eligibility.jsonl", eligibility),
        ("outcomes.jsonl", outcomes),
        ("sessions.jsonl", sessions),
    ):
        contents[name] = b"".join(
            (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
            for row in rows
        )
    manifest = {
        "schema_version": SCHEMA,
        "sample_role": ROLE,
        "evaluation_allowed": False,
        "output_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in contents.items()
        },
    }
    contents["manifest.json"] = json_bytes(manifest)
    for name, data in contents.items():
        with (output / name).open("xb") as stream:
            stream.write(data)
    completion = {
        "schema_version": SCHEMA,
        "status": "COMPLETE_CALIBRATION_ONLY",
        "evaluation_allowed": False,
        "manifest_sha256": sha256(output / "manifest.json"),
    }
    temporary = output / ".COMPLETE.json.tmp"
    with temporary.open("xb") as stream:
        stream.write(json_bytes(completion))
        stream.flush()
        os.fsync(stream.fileno())
    # A hard link publishes closed, complete bytes atomically and refuses an
    # existing destination. Unlike replace(), it cannot overwrite a marker.
    os.link(temporary, output / "COMPLETE.json")
    try:
        temporary.unlink()
    except OSError:
        pass  # Publication already succeeded; leftover temporary is harmless.


def execute(input_path: Path, output: Path) -> None:
    started = utc_now()
    validate_input_path(input_path)
    reserved = reserve_output(ROOT, output)
    try:
        provenance = verify_provenance(ROOT)
        runtime = runtime_provenance()
        validate_input_hash(input_path)
        print(
            "Provenance verified; loading registered pre-cutoff observations.",
            flush=True,
        )
        minutes, skipped = gate.load_minutes(input_path)
        validate_input_hash(input_path)
        eligibility, bars, signals, counts = eligible_ledgers(minutes, skipped)
        original = json.loads((ROOT / GATE_REPORT).read_text())
        reconcile_counts(counts, original)
        print("Eligibility and signal counts reconciled before alignment.", flush=True)
        if verify_provenance(ROOT) != provenance:
            raise CalibrationError("provenance changed before alignment")
        outcomes, sessions = aligned_ledgers(minutes, bars, signals)
        statistics = summarize(outcomes, sessions)
        if verify_provenance(ROOT) != provenance:
            raise CalibrationError("provenance changed before publication")
        validate_input_hash(input_path)
        report = {
            "schema_version": SCHEMA,
            "status": "COMPLETE_CALIBRATION_ONLY",
            "phase": "A",
            "sample_role": ROLE,
            "original_gate_verdict": "POWER_UNDETERMINED",
            "evaluation_allowed": False,
            "confirmation_authorized": False,
            "started_at_utc": started,
            "completed_at_utc": utc_now(),
            "output_directory": str(reserved),
            "counts": counts,
            "counts_reconciled_before_alignment": True,
            "statistics": statistics,
            "cost_scenarios_dollars": list(gate.COSTS),
            "point_value_dollars": gate.POINT_VALUE,
            "limitations": LIMITATIONS,
            "provenance": {
                **provenance,
                "runtime": runtime,
                "input": {
                    "path": str(input_path),
                    "sha256": INPUT_SHA256,
                    "canonical_git_revision": None,
                    "canonical_git_revision_null_reason": (
                        "EXTERNAL_INPUT_HASH_BOUND_BY_ORIGINAL_RESULT"
                    ),
                    "hash_recorded_in_revision": RESULT_REVISION,
                    "cutoff_exclusive_utc": gate.CUTOFF.isoformat(),
                },
            },
        }
        publish(reserved, report, eligibility, outcomes, sessions)
    except Exception as exc:
        # Preserve any partial files; a failure never writes COMPLETE.json.
        failure = {
            "schema_version": SCHEMA,
            "status": "FAILED",
            "evaluation_allowed": False,
            "original_gate_verdict": "POWER_UNDETERMINED",
            "started_at_utc": started,
            "failed_at_utc": utc_now(),
            "reason": str(exc),
        }
        with (reserved / "FAILED.json").open("xb") as stream:
            stream.write(json_bytes(failure))
        raise
    try:
        print(f"Completed calibration only: {reserved}", flush=True)
    except (OSError, ValueError):
        pass  # Successful publication is authoritative if the log stream fails.


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        execute(args.input, args.output_dir)
    except (CalibrationError, gate.GateError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "FAILED",
                    "evaluation_allowed": False,
                    "original_gate_verdict": "POWER_UNDETERMINED",
                    "reason": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
