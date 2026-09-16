"""Outcome-blind MNQ wick-short detectability gate.

Actual wick events supply only timing. Their following-bar movement is never
paired with them: outcomes are from a different, circularly shifted session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
NY = ZoneInfo("America/New_York")
CUTOFF = datetime(2026, 3, 1, tzinfo=timezone.utc)
PLAN_SHA256 = "0a93788286199fc420bb72ec1faf7c568722f527e1ea5dd8c9e7a6f434f1384d"
REGISTRATION_SHA256 = "ed3fad3c455100ca4b12a918fa658b7a717f011a4dfdfd0cf292c1e66784daff"
MIN_SHIFT = 5
POINT_VALUE = 2.0
COSTS = (1.22, 2.22, 3.22)
Z_SUM = float(stats.norm.ppf(0.95) + stats.norm.ppf(0.80))


class GateError(ValueError):
    """A valid gate cannot be constructed from the supplied inputs."""


@dataclass(frozen=True)
class Minute:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    contract: str


@dataclass(frozen=True)
class Bar:
    session: date
    slot: int
    open: float
    high: float
    low: float
    close: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise GateError("TimeStamp must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GateError("invalid TimeStamp") from exc
    if parsed.tzinfo is None:
        raise GateError("TimeStamp must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def finite_ohlc(record: dict[str, Any]) -> tuple[float, float, float, float]:
    values: list[float] = []
    for field in ("Open", "High", "Low", "Close"):
        try:
            value = float(record[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise GateError(f"malformed {field}") from exc
        if not math.isfinite(value):
            raise GateError(f"non-finite {field}")
        values.append(value)
    opening, high, low, close = values
    if high < max(opening, close) or low > min(opening, close) or high < low:
        raise GateError("invalid OHLC ordering")
    return opening, high, low, close


def load_minutes(path: Path) -> tuple[list[Minute], int]:
    """Load pre-cutoff records; malformed retained rows and duplicates refuse."""
    if not path.is_file():
        raise GateError("input must be a regular JSON file")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError("input is not valid JSON") from exc
    if not isinstance(payload, list):
        raise GateError("input JSON must be an array")
    minutes: list[Minute] = []
    seen: set[datetime] = set()
    skipped = 0
    for record in payload:
        if not isinstance(record, dict):
            raise GateError("each input row must be an object")
        stamp = parse_timestamp(record.get("TimeStamp"))
        if stamp >= CUTOFF:
            skipped += 1
            continue
        if stamp in seen:
            raise GateError("duplicate retained timestamp")
        seen.add(stamp)
        opening, high, low, close = finite_ohlc(record)
        contract = record.get("Contract")
        if not isinstance(contract, str) or not contract:
            raise GateError("malformed Contract")
        minutes.append(Minute(stamp, opening, high, low, close, contract))
    if not minutes:
        raise GateError("no pre-cutoff observations")
    return sorted(minutes, key=lambda row: row.timestamp), skipped


def expected_rth_minutes(session: date) -> set[datetime]:
    start = datetime.combine(session, time(9, 31), tzinfo=NY)
    return {start + timedelta(minutes=offset) for offset in range(390)}


def sessionize(
    minutes: Iterable[Minute],
) -> tuple[dict[date, list[Minute]], dict[str, int]]:
    buckets: dict[date, list[Minute]] = defaultdict(list)
    for row in minutes:
        local = row.timestamp.astimezone(NY)
        if time(9, 31) <= local.timetz().replace(tzinfo=None) <= time(16, 0):
            buckets[local.date()].append(row)
    eligible: dict[date, list[Minute]] = {}
    exclusions: Counter[str] = Counter()
    for session, rows in sorted(buckets.items()):
        actual = {row.timestamp.astimezone(NY) for row in rows}
        if actual != expected_rth_minutes(session):
            exclusions["INCOMPLETE_RTH_MINUTES"] += 1
            continue
        if len({row.contract for row in rows}) != 1:
            exclusions["MIXED_CONTRACT"] += 1
            continue
        eligible[session] = rows
    return eligible, dict(exclusions)


def bars_for_session(session: date, rows: Sequence[Minute]) -> list[Bar]:
    by_local = {row.timestamp.astimezone(NY): row for row in rows}
    start = datetime.combine(session, time(9, 31), tzinfo=NY)
    bars: list[Bar] = []
    for slot in range(78):
        group = [
            by_local[start + timedelta(minutes=slot * 5 + offset)]
            for offset in range(5)
        ]
        bars.append(
            Bar(
                session,
                slot,
                group[0].open,
                max(x.high for x in group),
                min(x.low for x in group),
                group[-1].close,
            )
        )
    return bars


def is_wick_short(bar: Bar) -> bool:
    body = abs(bar.close - bar.open)
    upper = bar.high - max(bar.open, bar.close)
    lower = min(bar.open, bar.close) - bar.low
    return body > 0 and upper >= 2 * body and lower <= 0.1 * upper


def signal_slots(bars: Sequence[Bar]) -> list[int]:
    return [bar.slot for bar in bars if bar.slot <= 75 and is_wick_short(bar)]


def cluster_se(values: Sequence[float], labels: Sequence[object]) -> float:
    if len(values) < 2 or len(values) != len(labels):
        raise GateError("insufficient paired observations")
    average = float(np.mean(values))
    totals: dict[object, float] = defaultdict(float)
    for value, label in zip(values, labels):
        totals[label] += value - average
    if len(totals) < 2:
        raise GateError("insufficient clusters")
    result = math.sqrt(sum(total * total for total in totals.values())) / len(values)
    if not math.isfinite(result) or result <= 0:
        raise GateError("unusable clustered variance")
    return result


def week_key(session: date) -> tuple[int, int]:
    iso = session.isocalendar()
    return iso.year, iso.week


def shifted_outcomes(
    sessions: Sequence[date],
    bars: dict[date, Sequence[Bar]],
    signals: dict[date, Sequence[int]],
    shift: int,
) -> tuple[list[float], list[date]]:
    count = len(sessions)
    if count == 0 or shift % count == 0:
        raise GateError("identity pairing is forbidden")
    values: list[float] = []
    origins: list[date] = []
    for index, original in enumerate(sessions):
        source = sessions[(index + shift) % count]
        for slot in signals[original]:
            # This slot belongs to the shifted session, never the original outcome.
            following = bars[source][slot + 1]
            values.append(POINT_VALUE * (following.open - following.close))
            origins.append(original)
    return values, origins


def summarize_shifts(
    sessions: Sequence[date],
    bars: dict[date, Sequence[Bar]],
    signals: dict[date, Sequence[int]],
) -> dict[str, Any]:
    if len(sessions) < 10:
        raise GateError(
            "insufficient eligible sessions for five-session circular shifts"
        )
    rows: list[dict[str, float | int]] = []
    for shift in range(MIN_SHIFT, len(sessions) - MIN_SHIFT + 1):
        values, origins = shifted_outcomes(sessions, bars, signals, shift)
        session_error = cluster_se(values, origins)
        week_error = cluster_se(values, [week_key(day) for day in origins])
        selected = max(session_error, week_error)
        rows.append(
            {
                "shift": shift,
                "session_cluster_se_dollars": session_error,
                "week_cluster_se_dollars": week_error,
                "selected_se_dollars": selected,
            }
        )
    selected = np.array([float(row["selected_se_dollars"]) for row in rows])
    median = float(np.median(selected))
    return {
        "valid_shift_count": len(rows),
        "shift_details": rows,
        "selected_se_dollars": {
            "median": median,
            "p10": float(np.percentile(selected, 10)),
            "p90": float(np.percentile(selected, 90)),
        },
        "mde_net_dollars_per_trade": median * Z_SUM,
        "mde_net_points_per_trade": median * Z_SUM / POINT_VALUE,
    }


def output_report(
    input_path: Path, plan_path: Path, results: dict[str, Any]
) -> dict[str, Any]:
    mde = results["dispersion"]["mde_net_dollars_per_trade"]
    costs = [
        {
            "round_trip_cost_dollars": cost,
            "required_gross_dollars_per_trade": mde + cost,
            "required_gross_points_per_trade": (mde + cost) / POINT_VALUE,
        }
        for cost in COSTS
    ]
    return {
        "gate": "mnq_wick_short_detectability_v1",
        "verdict": "POWER_UNDETERMINED",
        "evaluation_allowed": False,
        "eligibility": results["eligibility"],
        "signal_frequency": results["signals"],
        "conditional_detectability": results["dispersion"],
        "cost_scenarios": costs,
        "limitations": [
            "Actual signal outcomes were not calculated or paired.",
            "Normal approximation and transferred dispersion are assumptions, "
            "not validated power.",
            "Costs are scenarios: published fee plus zero, two, or four adverse "
            "ticks; they are not measured fills or verified account charges.",
            "No independently supported expected effect or calibrated execution "
            "model exists.",
        ],
        "hashes": {
            "input_sha256": sha256(input_path),
            "code_sha256": sha256(Path(__file__)),
            "committed_plan_sha256": sha256(plan_path),
            "registration_sha256": REGISTRATION_SHA256,
        },
    }


def markdown(report: dict[str, Any]) -> str:
    se = report["conditional_detectability"]["selected_se_dollars"]
    mde = report["conditional_detectability"]
    lines = [
        "# MNQ wick-short detectability gate",
        "",
        "**Verdict:** POWER_UNDETERMINED; `evaluation_allowed=false`.",
        "",
        "Eligible sessions: "
        f"{report['eligibility']['eligible_sessions']} of "
        f"{report['eligibility']['rth_sessions_seen']}; signals: "
        f"{report['signal_frequency']['total_signals']}.",
        "",
        "Transferred-dispersion SE ($/trade): "
        f"median {se['median']:.4f}, p10 {se['p10']:.4f}, "
        f"p90 {se['p90']:.4f} across "
        f"{report['conditional_detectability']['valid_shift_count']} shifts.",
        "Approximate minimum detectable net edge: "
        f"${mde['mde_net_dollars_per_trade']:.4f}/trade "
        f"({mde['mde_net_points_per_trade']:.4f} points).",
        "",
        "## Cost scenarios",
        "",
        "| Cost ($) | Required gross ($/trade) | Required gross points |",
        "|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['round_trip_cost_dollars']:.2f} | "
        f"{row['required_gross_dollars_per_trade']:.4f} | "
        f"{row['required_gross_points_per_trade']:.4f} |"
        for row in report["cost_scenarios"]
    )
    lines += [
        "",
        "These are outcome-blind shifted-session sensitivity calculations, not "
        "actual strategy performance or validated execution power.",
        "",
    ]
    return "\n".join(lines)


def publish(output_dir: Path, report: dict[str, Any]) -> None:
    if output_dir.exists():
        raise GateError("output directory must not already exist")
    output_dir.mkdir(parents=True)
    try:
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        (output_dir / "report.md").write_text(markdown(report))
    except Exception:
        for child in output_dir.iterdir():
            child.unlink()
        output_dir.rmdir()
        raise


def run(input_path: Path, plan_path: Path) -> dict[str, Any]:
    if sha256(plan_path) != PLAN_SHA256:
        raise GateError("committed plan hash mismatch")
    registration = (
        ROOT / "_bmad-output/preregistration_mnq_wick_short_power_20260916.md"
    )
    if sha256(registration) != REGISTRATION_SHA256:
        raise GateError("registration hash mismatch")
    minutes, skipped = load_minutes(input_path)
    eligible, exclusions = sessionize(minutes)
    sessions = sorted(eligible)
    bars = {
        session: bars_for_session(session, rows) for session, rows in eligible.items()
    }
    signals = {session: signal_slots(bars[session]) for session in sessions}
    total_signals = sum(len(value) for value in signals.values())
    if total_signals < 2:
        raise GateError("insufficient actual signal observations")
    dispersion = summarize_shifts(sessions, bars, signals)
    return {
        "eligibility": {
            "rth_sessions_seen": len(eligible) + sum(exclusions.values()),
            "eligible_sessions": len(sessions),
            "excluded_sessions": sum(exclusions.values()),
            "exclusion_reasons": exclusions,
            "post_cutoff_records_skipped": skipped,
        },
        "signals": {
            "total_signals": total_signals,
            "sessions_with_signal": sum(bool(value) for value in signals.values()),
            "signals_per_eligible_session": total_signals / len(sessions),
            "actual_signal_times_counted_only": True,
        },
        "dispersion": dispersion,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--committed-plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = output_report(
            args.input,
            args.committed_plan,
            run(args.input, args.committed_plan),
        )
        publish(args.output_dir, report)
    except (GateError, OSError) as exc:
        print(
            json.dumps(
                {
                    "verdict": "POWER_UNDETERMINED",
                    "evaluation_allowed": False,
                    "error": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
