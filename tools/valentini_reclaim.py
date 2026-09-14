"""Research-only Globex reclaim mechanics; market evaluation is deliberately closed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

TICK = 0.25
DOLLARS_PER_POINT = 2.0
AREA_FRACTION = 0.70
VERSION = "valentini-terminal-v1"
KNOWN_UNSUITABLE = "3f20ec70885cdee6b48e6c5c7ed3254dd4cc8ce7bd8533696c5e461c75fb7822"
REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


class Rejected(ValueError):
    """Input or admission rejected before performance execution."""


@dataclass(frozen=True)
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Session:
    """Independent UTC schedule: inclusive start, exclusive end, minute-start labels."""

    name: str
    start: datetime
    end: datetime
    breaks: tuple[tuple[datetime, datetime], ...] = ()


@dataclass(frozen=True)
class Signal:
    session: str
    break_time: datetime
    reclaim_time: datetime
    val: float
    vah: float
    excursion_low: float
    stop: float
    target: float


@dataclass(frozen=True)
class Trade:
    signal: Signal
    entry_time: datetime
    entry: float
    exit_time: datetime | None
    exit_bar_time: datetime
    exit: float
    reason: str
    contracts: int
    gross_dollars: float
    commission_dollars: float
    net_dollars: float


@dataclass(frozen=True)
class Costs:
    """Slippage in price points per market fill; commission dollars per side."""

    slippage_points: float
    commission_per_side: float

    def validate(self) -> None:
        for value in (self.slippage_points, self.commission_per_side):
            if not math.isfinite(value) or value < 0:
                raise Rejected("Costs must be finite and nonnegative")


@dataclass(frozen=True)
class Simulation:
    signals: tuple[Signal, ...]
    trades: tuple[Trade, ...]


def _utc_minute(value: datetime) -> bool:
    return (
        value.tzinfo is not None
        and value.utcoffset() == timedelta(0)
        and value.second == 0
        and value.microsecond == 0
    )


def _finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise Rejected(f"Nonfinite derived {label}")
    return value


def _tick_index(price: float) -> int:
    if not math.isfinite(price) or price <= 0:
        raise Rejected("Price must be finite and positive")
    tick = round(_finite(price / TICK, "tick index"))
    if not math.isclose(price / TICK, tick, rel_tol=0, abs_tol=1e-8):
        raise Rejected("Off-tick price")
    return tick


def validate_bar(bar: Bar) -> None:
    for price in (bar.open, bar.high, bar.low, bar.close):
        _tick_index(price)
    if not bar.low <= min(bar.open, bar.close) <= max(bar.open, bar.close) <= bar.high:
        raise Rejected("Invalid OHLC ordering")
    if not math.isfinite(bar.volume) or bar.volume < 0:
        raise Rejected("Volume must be finite and nonnegative")


def scheduled_minutes(session: Session) -> tuple[datetime, ...]:
    if not _utc_minute(session.start) or not _utc_minute(session.end):
        raise Rejected("Schedule requires UTC minute boundaries")
    if session.end <= session.start:
        raise Rejected("Empty or reversed schedule")
    prior = session.start
    excluded: set[datetime] = set()
    for start, end in session.breaks:
        if not _utc_minute(start) or not _utc_minute(end):
            raise Rejected("Breaks require UTC minute boundaries")
        if not session.start <= start < end <= session.end or start < prior:
            raise Rejected("Invalid or overlapping scheduled breaks")
        excluded.update(
            start + timedelta(minutes=i)
            for i in range(int((end - start).total_seconds() / 60))
        )
        prior = end
    minutes = tuple(
        session.start + timedelta(minutes=i)
        for i in range(int((session.end - session.start).total_seconds() / 60))
        if session.start + timedelta(minutes=i) not in excluded
    )
    if not minutes:
        raise Rejected("Schedule contains no tradable minutes")
    return minutes


def validate_session(bars: Sequence[Bar], session: Session) -> None:
    expected = scheduled_minutes(session)
    if len(bars) != len(expected):
        raise Rejected("Session rejected: missing or extra minute")
    for bar, timestamp in zip(bars, expected):
        validate_bar(bar)
        if not _utc_minute(bar.timestamp) or bar.timestamp != timestamp:
            raise Rejected(
                "Session rejected: duplicate, gap, order, or schedule mismatch"
            )


class Profile:
    """Uniform OHLCV allocation, retaining empty ticks between traded levels."""

    def __init__(self) -> None:
        self.volumes: dict[int, float] = {}
        self.total = 0.0

    def add(self, bar: Bar) -> None:
        validate_bar(bar)
        low, high = _tick_index(bar.low), _tick_index(bar.high)
        total = _finite(self.total + bar.volume, "profile volume")
        amount = bar.volume / (high - low + 1)
        for level in range(low, high + 1):
            self.volumes[level] = _finite(
                self.volumes.get(level, 0.0) + amount, "tick volume"
            )
        self.total = total

    def area(self) -> tuple[float, float, float] | None:
        try:
            total = _finite(math.fsum(self.volumes.values()), "profile volume")
        except OverflowError as exc:
            raise Rejected("Nonfinite derived profile volume") from exc
        if total == 0:
            return None
        poc = min(self.volumes, key=lambda level: (-self.volumes[level], level))
        low = high = poc
        covered = self.volumes[poc]
        bottom, top = min(self.volumes), max(self.volumes)
        while covered < AREA_FRACTION * total and (low > bottom or high < top):
            left = self.volumes.get(low - 1, 0.0) if low > bottom else -1.0
            right = self.volumes.get(high + 1, 0.0) if high < top else -1.0
            if left >= right:
                low -= 1
                covered += left
            else:
                high += 1
                covered += right
        _finite(covered, "covered profile volume")
        return low * TICK, high * TICK, poc * TICK


def simulate_session(bars: Sequence[Bar], session: Session, costs: Costs) -> Simulation:
    """Pure synthetic-fixture engine. Admission CLI never calls this function."""
    validate_session(bars, session)  # Reject the whole session before any signal.
    costs.validate()
    commission = _finite(2 * costs.commission_per_side, "round-trip commission")
    profile = Profile()
    signals: list[Signal] = []
    trades: list[Trade] = []
    setup: tuple[datetime, float, float, float] | None = None
    pending: Signal | None = None
    position: tuple[Signal, datetime, float] | None = None
    previous: Bar | None = None
    for index, bar in enumerate(bars):
        if previous is not None and bar.timestamp - previous.timestamp != timedelta(
            minutes=1
        ):
            pending = None
            setup = None
            previous = None  # No adjacent-volume comparison across a scheduled halt.
        consumed = False
        if pending is not None:
            consumed = True
            entry = _finite(bar.open + costs.slippage_points, "entry fill")
            if (
                pending.stop < bar.open < pending.target
                and pending.stop < entry < pending.target
            ):
                position = (pending, bar.timestamp, entry)
            pending = None
        if position is not None:
            consumed = True
            signal, entry_time, entry = position
            fill: float | None = None
            reason = ""
            exit_time: datetime | None = None
            if bar.open <= signal.stop:
                fill = bar.open - costs.slippage_points
                reason, exit_time = "stop", bar.timestamp
            elif bar.open > signal.target:
                fill = signal.target
                reason, exit_time = "target", bar.timestamp
            elif bar.low <= signal.stop:
                fill = min(signal.stop, bar.open) - costs.slippage_points
                reason = "stop"
            elif bar.high > signal.target:
                fill, reason = signal.target, "target"
            elif index == len(bars) - 1:
                fill = bar.close - costs.slippage_points
                reason = "session_end"
                exit_time = bar.timestamp + timedelta(minutes=1)
            if fill is not None:
                _finite(fill, "exit fill")
                gross = _finite((fill - entry) * DOLLARS_PER_POINT, "gross dollars")
                net = _finite(gross - commission, "net dollars")
                trades.append(
                    Trade(
                        signal,
                        entry_time,
                        entry,
                        exit_time,
                        bar.timestamp,
                        fill,
                        reason,
                        1,
                        gross,
                        commission,
                        net,
                    )
                )
                position = None
        if not consumed and setup is not None:
            consumed = True
            break_time, val, vah, low = setup
            low = min(low, bar.low)
            setup = (break_time, val, vah, low)
            if bar.close >= val:
                if (
                    bar.close <= vah
                    and previous is not None
                    and bar.volume > previous.volume
                ):
                    pending = Signal(
                        session.name,
                        break_time,
                        bar.timestamp,
                        val,
                        vah,
                        low,
                        low - TICK,
                        vah,
                    )
                    signals.append(pending)
                setup = None  # First reclaim succeeds or cancels permanently.
        if not consumed and previous is not None:
            area = profile.area()
            if area is not None:
                val, vah, _ = area
                if (
                    val <= previous.close <= vah
                    and bar.close < val
                    and bar.volume < previous.volume
                ):
                    setup = (bar.timestamp, val, vah, bar.low)
        profile.add(bar)  # Only completed preceding bars form the decision profile.
        previous = bar
    # Unentered signals are recorded as confirmations, never as trades.
    return Simulation(tuple(signals), tuple(trades))


def simulate_sessions(
    bars: Sequence[Bar], sessions: Sequence[Session], costs: Costs
) -> Simulation:
    """Validate an independent full-session schedule before simulating any session."""
    costs.validate()
    chunks: list[tuple[Session, list[Bar]]] = []
    offset = 0
    previous_end: datetime | None = None
    names: set[str] = set()
    for session in sessions:
        if session.name in names or (
            previous_end is not None and session.start < previous_end
        ):
            raise Rejected(
                "Duplicate session name or overlapping/out-of-order schedule"
            )
        names.add(session.name)
        count = len(scheduled_minutes(session))
        chunk = list(bars[offset : offset + count])
        validate_session(chunk, session)
        chunks.append((session, chunk))
        offset += count
        previous_end = session.end
    if offset != len(bars):
        raise Rejected("Bars outside supplied sessions")
    signals: list[Signal] = []
    trades: list[Trade] = []
    for session, chunk in chunks:
        result = simulate_session(chunk, session, costs)
        signals.extend(result.signals)
        trades.extend(result.trades)
    return Simulation(tuple(signals), tuple(trades))


def null_mde(
    panel: Sequence[Sequence[float]],
    pairings: Sequence[Sequence[int]],
    *,
    effect: float | None,
    alpha: float,
    power: float,
    dependence_factors: Sequence[float],
) -> dict[str, Any]:
    """Synthetic pairing calibration: cell i,j pairs setup session i to path j.

    Never reads the matched diagonal. Pairings must be complete derangements.
    Normal-approximation MDE is illustrative, not independently validated power.
    """
    size = len(panel)
    if size < 2 or any(len(row) != size for row in panel) or len(pairings) < 2:
        raise Rejected("Need square panel and at least two derangements")
    if not 0 < alpha < 0.5 or not 0.5 < power < 1:
        raise Rejected("Invalid alpha or desired power")
    if effect is not None and (not math.isfinite(effect) or effect <= 0):
        raise Rejected("Effect must be externally supplied, finite and positive")
    values: list[float] = []
    for pairing in pairings:
        if sorted(pairing) != list(range(size)) or any(
            i == j for i, j in enumerate(pairing)
        ):
            raise Rejected("Identity/fixed points and non-permutations are forbidden")
        cells = [panel[i][j] for i, j in enumerate(pairing)]
        if any(not math.isfinite(value) for value in cells):
            raise Rejected("Nonfinite off-diagonal outcomes")
        values.append(statistics.mean(cells))
    if not dependence_factors or any(
        not math.isfinite(f) or f < 1 for f in dependence_factors
    ):
        raise Rejected("Dependence sensitivity requires finite variance factors >= 1")
    if len({tuple(pairing) for pairing in pairings}) < 2:
        raise Rejected("Degenerate calibration: need distinct derangements")
    spread = _finite(statistics.stdev(values), "null spread")
    if spread <= 0:
        raise Rejected("Degenerate calibration: zero null spread")
    normal = statistics.NormalDist()
    multiplier = normal.inv_cdf(1 - alpha) + normal.inv_cdf(power)
    sensitivity = []
    for factor in dependence_factors:
        mde = multiplier * spread * math.sqrt(factor)
        sensitivity.append(
            {
                "variance_inflation": factor,
                "mde": mde,
                "effect_exceeds_mde": None if effect is None else effect >= mde,
            }
        )
    result = {
        "null_means": values,
        "null_mean": statistics.mean(values),
        "null_sd": spread,
        "alpha_one_sided": alpha,
        "desired_power": power,
        "effect": effect,
        "dependence_sensitivity": sensitivity,
        "verdict": "SYNTHETIC_CALIBRATION_ONLY",
    }
    canonical_json(result)  # Reject overflow as well as nonfinite inputs.
    return result


def derangements(size: int, draws: int, seed: int) -> tuple[tuple[int, ...], ...]:
    if size < 2 or draws < 2:
        raise Rejected("Need at least two sessions and draws")
    rng = random.Random(seed)
    result: list[tuple[int, ...]] = []
    while len(result) < draws:
        pairing = list(range(size))
        rng.shuffle(pairing)
        if all(i != j for i, j in enumerate(pairing)):
            result.append(tuple(pairing))
    return tuple(result)


def protected_path(raw: str | Path) -> Path:
    def check(candidate: Path) -> None:
        lowered = {part.lower() for part in candidate.parts}
        if lowered & {"sealed_holdout", "trades.db", ".env", ".access_token"}:
            raise Rejected("Protected research input/output path")

    requested = Path(raw).expanduser().absolute()
    check(requested)
    # Resolve one link at a time: Path.resolve() hides protected intermediate hops.
    pending = list(requested.parts[1:])
    resolved = Path(requested.anchor)
    links = 0
    while pending:
        part = pending.pop(0)
        if part == "..":
            resolved = resolved.parent
            continue
        candidate = resolved / part
        check(candidate)
        if candidate.is_symlink():
            links += 1
            if links > 40:
                raise Rejected("Symlink cycle or excessive indirection")
            target = candidate.readlink()
            check(target)
            if target.is_absolute():
                resolved = Path(target.anchor)
                pending = list(target.parts[1:]) + pending
            else:
                pending = list(target.parts) + pending
        else:
            resolved = candidate
    check(resolved)
    return resolved


def same_file(left: Path, right: Path) -> bool:
    """Recognize both normalized names and existing hard-link aliases."""
    return left == right or (left.exists() and right.exists() and left.samefile(right))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with protected_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_hash() -> str:
    return sha256_file(__file__)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def audit_csv(raw: str | Path) -> dict[str, Any]:
    path = protected_path(raw)
    before = sha256_file(path)
    count = valid = invalid = duplicates = gaps = reversals = 0
    missing_minutes = 0
    seen: set[str] = set()
    first: datetime | None = None
    last: datetime | None = None
    previous: datetime | None = None
    examples: list[dict[str, Any]] = []
    issues: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, strict=True)
        columns = reader.fieldnames or []
        missing_columns = sorted(REQUIRED_COLUMNS - set(columns))
        if len(columns) != len(set(columns)):
            issues.add("duplicate_column_names")
        for row in reader:
            count += 1
            try:
                if (
                    missing_columns
                    or None in row
                    or any(value is None for value in row.values())
                ):
                    raise Rejected("Missing schema column or malformed CSV row")
                timestamp = _parse_timestamp(row["timestamp"])
                bar = Bar(
                    timestamp,
                    *(
                        float(row[key])
                        for key in ("open", "high", "low", "close", "volume")
                    ),
                )
                validate_bar(bar)
                if not _utc_minute(timestamp):
                    raise Rejected("Timestamp is not explicit UTC minute boundary")
                key = timestamp.isoformat()
                if key in seen:
                    duplicates += 1
                seen.add(key)
                if previous is not None:
                    delta = (timestamp - previous).total_seconds()
                    if delta <= 0:
                        reversals += 1
                    elif delta != 60:
                        gaps += 1
                        missing_minutes += max(int(delta / 60) - 1, 0)
                previous = timestamp
                first = timestamp if first is None else min(first, timestamp)
                last = timestamp if last is None else max(last, timestamp)
                valid += 1
            except (ValueError, TypeError, OverflowError) as exc:
                invalid += 1
                if len(examples) < 10:
                    examples.append({"csv_row": count + 1, "reason": str(exc)})
    after = sha256_file(path)
    if before != after:
        raise Rejected("Input changed during audit")
    if before == KNOWN_UNSUITABLE:
        issues.add("known_dollar_aggregated_not_fixed_minute")
    if count == 0:
        issues.add("empty_input")
    if missing_columns:
        issues.add("missing_schema_columns")
    if invalid:
        issues.add("invalid_rows_reject_affected_sessions")
    if duplicates or reversals:
        issues.add("duplicate_or_out_of_order_rows")
    if gaps:
        issues.add("gaps_require_independent_session_calendar")
    return {
        "path": str(path),
        "requested_path": str(raw),
        "sha256": before,
        "bytes": path.stat().st_size,
        "columns": columns,
        "missing_columns": missing_columns,
        "row_count": count,
        "valid_rows": valid,
        "invalid_rows": invalid,
        "invalid_examples": examples,
        "first_timestamp": None if first is None else first.isoformat(),
        "last_timestamp": None if last is None else last.isoformat(),
        "duplicate_timestamps": duplicates,
        "nonincreasing_timestamps": reversals,
        "gap_intervals": gaps,
        "missing_clock_minutes": missing_minutes,
        "findings": sorted(issues),
        "evidence": {
            "timestamp_label_semantics": "UNKNOWN",
            "session_calendar": "UNKNOWN",
            "contract_provenance": "UNKNOWN",
            "fixed_minute_source_provenance": "UNKNOWN",
        },
        "admitted": False,
    }


def make_audit(paths: Sequence[str | Path]) -> dict[str, Any]:
    if not paths:
        raise Rejected("At least one CSV is required")
    # Check every path before opening any input.
    resolved = [protected_path(path) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise Rejected("Duplicate input file or symlink alias")
    return {
        "kind": "valentini_audit",
        "version": VERSION,
        "code_sha256": code_hash(),
        "inputs": [audit_csv(path) for path in paths],
        "admitted": False,
        "performance_computed": False,
    }


def read_json(path: str | Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise Rejected(f"Nonfinite JSON constant: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise Rejected(f"Duplicate JSON key: {key}")
            result[key] = item
        return result

    with protected_path(path).open(encoding="utf-8") as stream:
        value = json.load(
            stream, parse_constant=reject_constant, object_pairs_hook=unique_object
        )
    if not isinstance(value, dict):
        raise Rejected("JSON object required")
    canonical_json(value)
    return value


def validate_audit(audit: Mapping[str, Any]) -> None:
    if (
        audit.get("kind") != "valentini_audit"
        or audit.get("version") != VERSION
        or audit.get("code_sha256") != code_hash()
    ):
        raise Rejected("Audit schema/version/code hash mismatch; rerun audit")
    inputs = audit.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise Rejected("Audit has no input records")
    for record in inputs:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise Rejected("Invalid audit input record")
        requested = record.get("requested_path", record["path"])
        if not isinstance(requested, str):
            raise Rejected("Invalid requested input path")
        protected_path(requested)
        if sha256_file(record["path"]) != record.get("sha256"):
            raise Rejected("Audit/input hash mismatch")


def make_power(
    audit_path: str | Path, evidence_path: str | Path | None
) -> dict[str, Any]:
    audit = read_json(audit_path)
    validate_audit(audit)
    evidence = {} if evidence_path is None else read_json(evidence_path)
    blockers = [
        "Independent timestamp-label, fixed-minute-source, calendar, and contract "
        "evidence has not been validated by this version."
    ]
    for record in audit["inputs"]:
        blockers.extend(
            f"{record['path']}: {finding}" for finding in record.get("findings", [])
        )
    power_blockers = []
    if not evidence.get("transferable_effect"):
        power_blockers.append(
            "No transferable, independently justified effect estimate."
        )
    if not evidence.get("independent_calibration"):
        power_blockers.append("No independent dependence-aware calibration.")
    power_blockers.append(
        "This terminal-only gate cannot validate or promote submitted evidence."
    )
    return {
        "kind": "valentini_power",
        "version": VERSION,
        "verdict": "DATA_UNSUITABLE",
        "power_status": "POWER_UNDETERMINED",
        "audit_sha256": sha256_file(audit_path),
        "evidence_sha256": (
            None if evidence_path is None else sha256_file(evidence_path)
        ),
        "code_sha256": code_hash(),
        "input_hashes": [
            {"path": row["path"], "sha256": row["sha256"]} for row in audit["inputs"]
        ],
        "data_blockers": blockers,
        "power_blockers": power_blockers,
        "performance_computed": False,
        "null_computed": False,
    }


def evaluate(
    audit_path: str | Path,
    gate_path: str | Path,
    evidence_path: str | Path | None = None,
) -> None:
    """Admission before performance reads. No simulator or ledger path exists here."""
    audit = read_json(audit_path)
    gate = read_json(gate_path)
    if audit.get("kind") != "valentini_audit" or audit.get("version") != VERSION:
        raise Rejected("Audit schema/version mismatch")
    if gate.get("kind") != "valentini_power" or gate.get("version") != VERSION:
        raise Rejected("Gate schema/version mismatch")
    if (
        gate.get("code_sha256") != code_hash()
        or audit.get("code_sha256") != code_hash()
    ):
        raise Rejected("Gate/audit code hash mismatch")
    if gate.get("audit_sha256") != sha256_file(audit_path):
        raise Rejected("Gate/audit hash mismatch")
    evidence_hash = None if evidence_path is None else sha256_file(evidence_path)
    if gate.get("evidence_sha256") != evidence_hash:
        raise Rejected("Gate/evidence hash mismatch")
    if gate.get("verdict") != "POWERED":
        raise Rejected("Evaluation refused: gate is not POWERED")
    # Even internally consistent user-authored hashes cannot authorize promotion.
    raise Rejected(
        "Evaluation refused: this version cannot independently establish "
        "POWERED promotion"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser(
        "audit", help="Metadata only; never compute signals or returns"
    )
    audit_parser.add_argument("--csv", action="append", required=True)
    audit_parser.add_argument("--output", required=True)
    power_parser = commands.add_parser(
        "power", help="Terminal-only admission and power report"
    )
    power_parser.add_argument("--audit", required=True)
    power_parser.add_argument("--evidence")
    power_parser.add_argument("--output", required=True)
    evaluate_parser = commands.add_parser(
        "evaluate", help="Fail-closed market evaluation firewall"
    )
    evaluate_parser.add_argument("--audit", required=True)
    evaluate_parser.add_argument("--gate", required=True)
    evaluate_parser.add_argument("--evidence")
    args = parser.parse_args(argv)
    try:
        output = protected_path(args.output) if args.command != "evaluate" else None
        inputs = (
            args.csv
            if args.command == "audit"
            else [args.audit] + ([args.evidence] if args.evidence else [])
        )
        if output is not None and any(
            same_file(output, protected_path(path)) for path in inputs
        ):
            raise Rejected("Output would overwrite an input")
        if args.command == "audit":
            result = make_audit(args.csv)
        elif args.command == "power":
            result = make_power(args.audit, args.evidence)
            assert output is not None
            if any(
                same_file(output, protected_path(row["path"]))
                for row in result["input_hashes"]
            ):
                raise Rejected("Output would overwrite audited market data")
        else:
            evaluate(args.audit, args.gate, args.evidence)
            return 2
        assert output is not None
        output.write_text(canonical_json(result), encoding="utf-8")
        print(
            canonical_json(
                {
                    "output": str(output),
                    "verdict": result.get("verdict", "NOT_ADMITTED"),
                }
            ),
            end="",
        )
        return 0
    except (OSError, ValueError, TypeError, KeyError, csv.Error) as exc:
        print(
            canonical_json({"error": str(exc), "performance_computed": False}),
            file=sys.stderr,
            end="",
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
