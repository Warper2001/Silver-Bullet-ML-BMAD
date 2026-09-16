"""Read-only source/ledger audit for the registered MNQ wick-short calibration.

It verifies the source container before decoding it, then checks the published
Phase A r2 ledgers against exact raw RTH rows.  It does not import or invoke
the calibration runner and it never publishes a partial result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
INPUT = Path("/root/mnq_historical.json")
INPUT_SHA256 = "e7aed8ba786436ba80f4b081d3b7a4ee97b06bd3e8cc347c035547ec57dcb924"
CUTOFF = datetime(2026, 3, 1, tzinfo=timezone.utc)
NY = ZoneInfo("America/New_York")
ROLL_DATE = date(2025, 3, 17)
CME_ROLL_URL = "https://www.cmegroup.com/trading/equity-index/rolldates.html"
R2 = ROOT / "_bmad-output/mnq-wick-short-calibration-phase-a-20260916-r2"
PINNED_PHASE_A = {
    "mnq-wick-short-calibration-phase-a-20260916": {
        "COMPLETE.json": (
            "a9947d0c3c0c772f29d635a3277aea845f592e58ab5d5b7977629b0c5409ca5f"
        ),
        "manifest.json": (
            "bc2ddd9d95e8c95658fdda72259e266dd6c62d14e9f65005205e8d3200521779"
        ),
        "report.json": (
            "1aa42110f376910751da6e462f4e37b087d0e9556ecf0d544a31c061d5c43086"
        ),
    },
    "mnq-wick-short-calibration-phase-a-20260916-r2": {
        "COMPLETE.json": (
            "e4a9d4e10964546a077b5d82270ae825826ac95859f937263909aa7cd895a3f7"
        ),
        "manifest.json": (
            "6aeb7869658fbfc2ad57ec7385ad82175ef9e925af8ab3eb88dfeecf6835726e"
        ),
        "report.json": (
            "b0eafae80426cd95c7d934c9e9c02bc842d2a27b834323f23a1cba2cf280c6bc"
        ),
    },
}


class AuditError(ValueError):
    """A bound input or published artifact failed provenance or reconciliation."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_stamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise AuditError("raw source row has no ISO-8601 TimeStamp")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise AuditError("raw source row has invalid TimeStamp") from error
    if stamp.tzinfo is None:
        raise AuditError("raw source row has offset-free TimeStamp")
    return stamp.astimezone(timezone.utc)


def expected_minutes(day: date) -> set[datetime]:
    start = datetime.combine(day, time(9, 31), tzinfo=NY)
    return {start + timedelta(minutes=offset) for offset in range(390)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot decode published ledger: {path}") from error


def verify_phase_artifacts(root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for directory, pins in PINNED_PHASE_A.items():
        path = root / "_bmad-output" / directory
        try:
            hashes = {name: sha256(path / name) for name in pins}
            manifest = json.loads((path / "manifest.json").read_text())
            output_hashes = manifest["output_sha256"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise AuditError(f"malformed Phase A publication: {directory}") from error
        if hashes != pins:
            raise AuditError(f"published Phase A hash mismatch: {directory}")
        if not isinstance(output_hashes, dict):
            raise AuditError(f"malformed Phase A manifest: {directory}")
        for name, digest in output_hashes.items():
            try:
                actual = sha256(path / name)
            except OSError as error:
                raise AuditError(
                    f"missing Phase A manifest output: {directory}/{name}"
                ) from error
            if not isinstance(digest, str) or actual != digest:
                raise AuditError(f"manifest output hash mismatch: {directory}/{name}")
        result[directory] = hashes
    return result


def load_raw_after_hash(path: Path) -> dict[datetime, dict[str, Any]]:
    """Decode only after the registered whole-file hash is verified."""
    if path != INPUT or path.is_symlink() or path.resolve() != INPUT:
        raise AuditError("only the registered /root/mnq_historical.json is allowed")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise AuditError("bound input cannot be read") from error
    if sha256_bytes(data) != INPUT_SHA256:
        raise AuditError("bound input hash mismatch")
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as error:
        raise AuditError("bound input cannot be decoded") from error
    if not isinstance(payload, list):
        raise AuditError("bound input JSON is not an array")
    rows: dict[datetime, dict[str, Any]] = {}
    for raw in payload:
        if not isinstance(raw, dict):
            raise AuditError("raw source row is not an object")
        stamp = parse_stamp(raw.get("TimeStamp"))
        if stamp >= CUTOFF:
            continue
        if stamp in rows:
            raise AuditError("duplicate retained raw timestamp")
        try:
            numbers = {
                name: float(raw[name]) for name in ("Open", "High", "Low", "Close")
            }
        except (KeyError, TypeError, ValueError) as error:
            raise AuditError("raw source row has malformed OHLC") from error
        if not all(math.isfinite(value) for value in numbers.values()):
            raise AuditError("raw source row has non-finite OHLC")
        contract = raw.get("Contract")
        if not isinstance(contract, str) or not contract:
            raise AuditError("raw source row has malformed Contract")
        local = stamp.astimezone(NY)
        if time(9, 31) <= local.time() <= time(16):
            rows[stamp] = {
                "stamp": stamp,
                "local": local,
                "contract": contract,
                **numbers,
            }
    return rows


def source_sessions(
    rows: dict[datetime, dict[str, Any]],
) -> dict[date, list[dict[str, Any]]]:
    result: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for row in rows.values():
        result[row["local"].date()].append(row)
    return result


def audit_eligibility(
    ledger: list[dict[str, Any]], sessions: dict[date, list[dict[str, Any]]]
) -> dict[str, int]:
    try:
        session_ids = {row["session_id"] for row in ledger}
    except (KeyError, TypeError) as error:
        raise AuditError("malformed eligibility ledger session ID") from error
    if len(session_ids) != len(ledger):
        raise AuditError("eligibility ledger has duplicate session IDs")
    try:
        ledger_days = {date.fromisoformat(row["session_id"]) for row in ledger}
    except (KeyError, TypeError, ValueError) as error:
        raise AuditError("malformed eligibility ledger session ID") from error
    if ledger_days != set(sessions):
        raise AuditError("source and eligibility session sets differ")
    summary = {
        "observed_rth_sessions": len(ledger),
        "eligible_single_contract_complete_sessions": 0,
        "incomplete_sessions": 0,
        "mixed_contract_sessions": 0,
        "incomplete_and_mixed_sessions": 0,
    }
    for item in ledger:
        try:
            day = date.fromisoformat(item["session_id"])
            source = sessions[day]
            labels = {row["local"] for row in source}
            expected = expected_minutes(day)
            contracts = sorted({row["contract"] for row in source})
            counts: dict[str, int] = defaultdict(int)
            for row in source:
                counts[row["contract"]] += 1
            missing = sorted(stamp.isoformat() for stamp in expected - labels)
            unexpected = sorted(stamp.isoformat() for stamp in labels - expected)
            reasons = []
            if labels != expected:
                reasons.append("INCOMPLETE_RTH_MINUTES")
            if len(contracts) != 1:
                reasons.append("MIXED_CONTRACT")
            one_contract = len(contracts) == 1
            complete = not reasons
            if (
                item["contracts"] != contracts
                or item["contract_minute_counts"] != dict(counts)
                or item["observed_minute_count"] != len(source)
                or item["missing_minute_labels"] != missing
                or item["unexpected_minute_labels"] != unexpected
                or item["exclusion_reasons"] != reasons
                or item["primary_exclusion_reason"] != (reasons[0] if reasons else None)
                or item["eligible"] != complete
            ):
                raise AuditError(f"raw/ledger eligibility mismatch: {day}")
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, AuditError):
                raise
            raise AuditError("malformed eligibility ledger fields") from error
        if complete and one_contract:
            summary["eligible_single_contract_complete_sessions"] += 1
        else:
            if not complete:
                summary["incomplete_sessions"] += 1
            if not one_contract:
                summary["mixed_contract_sessions"] += 1
            if not complete and not one_contract:
                summary["incomplete_and_mixed_sessions"] += 1
    return summary


def same_number(a: object, b: object) -> bool:
    return (
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and float(a) == float(b)
    )


def audit_outcomes(
    outcomes: list[dict[str, Any]], raw: dict[datetime, dict[str, Any]]
) -> None:
    for outcome in outcomes:
        try:
            signal_id = outcome["signal_id"]
            day = outcome["session_id"]
            contract = outcome["contract"]
            label = datetime.fromisoformat(outcome["signal_bar_label_local"])
            if label.tzinfo is None or label.astimezone(NY).date().isoformat() != day:
                raise AuditError(f"malformed outcome session label: {signal_id}")
            label = label.astimezone(NY)
            signal = outcome["signal_component_minutes"]
            reference = outcome["reference_component_minutes"]
            if len(signal) != 5 or len(reference) != 5:
                raise AuditError(
                    f"outcome does not have two five-minute windows: {signal_id}"
                )
            expected_signal = [
                label - timedelta(minutes=4 - offset) for offset in range(5)
            ]
            expected_reference = [
                label + timedelta(minutes=1 + offset) for offset in range(5)
            ]
            all_stamps: list[datetime] = []
            for window, expected in (
                (signal, expected_signal),
                (reference, expected_reference),
            ):
                for component, expected_local in zip(window, expected):
                    stamp = parse_stamp(component["source_record_id"])
                    source = raw.get(stamp)
                    if source is None:
                        raise AuditError(
                            f"outcome component absent from raw source: {signal_id}"
                        )
                    if stamp.astimezone(NY) != expected_local:
                        raise AuditError(
                            f"outcome component window is not contiguous: {signal_id}"
                        )
                    if component["minute_label_local"] != expected_local.isoformat():
                        raise AuditError(
                            f"outcome component local label mismatch: {signal_id}"
                        )
                    if (
                        source["local"].date().isoformat() != day
                        or source["contract"] != contract
                    ):
                        raise AuditError(
                            "outcome component crosses source session/contract: "
                            f"{signal_id}"
                        )
                    if component["contract"] != contract or any(
                        not same_number(component[key], source[key.title()])
                        for key in ("open", "high", "low", "close")
                    ):
                        raise AuditError(
                            "outcome component values differ from raw source: "
                            f"{signal_id}"
                        )
                    all_stamps.append(stamp)
            if len(set(all_stamps)) != 10:
                raise AuditError(
                    f"outcome component minutes are duplicated: {signal_id}"
                )
            if (
                outcome["reference_interval_start_local"] != label.isoformat()
                or outcome["reference_interval_end_local"]
                != (label + timedelta(minutes=5)).isoformat()
            ):
                raise AuditError(f"outcome reference interval mismatch: {signal_id}")
            if datetime.fromisoformat(
                outcome["signal_bar_label_utc"]
            ) != label.astimezone(timezone.utc):
                raise AuditError(f"outcome UTC signal label mismatch: {signal_id}")
            expected_ohlc = {
                "open": signal[0]["open"],
                "high": max(row["high"] for row in signal),
                "low": min(row["low"] for row in signal),
                "close": signal[-1]["close"],
            }
            if outcome["signal_bar_ohlc"] != expected_ohlc:
                raise AuditError(f"outcome signal OHLC mismatch: {signal_id}")
            body = abs(expected_ohlc["close"] - expected_ohlc["open"])
            upper = expected_ohlc["high"] - max(
                expected_ohlc["open"], expected_ohlc["close"]
            )
            lower = (
                min(expected_ohlc["open"], expected_ohlc["close"])
                - expected_ohlc["low"]
            )
            if not all(
                same_number(outcome[key], value)
                for key, value in (
                    ("body", body),
                    ("upper_wick", upper),
                    ("lower_wick", lower),
                )
            ):
                raise AuditError(f"outcome derived geometry mismatch: {signal_id}")
            gross = 2 * (float(reference[0]["open"]) - float(reference[-1]["close"]))
            if (
                not same_number(outcome["next_open"], reference[0]["open"])
                or not same_number(outcome["next_close"], reference[-1]["close"])
                or not same_number(outcome["gross_dollars"], gross)
            ):
                raise AuditError(f"outcome reference arithmetic mismatch: {signal_id}")
        except AuditError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise AuditError("malformed outcome ledger fields") from error


def pre_roll_summary(
    ledger: list[dict[str, Any]], outcomes: list[dict[str, Any]]
) -> dict[str, Any]:
    contaminated = [
        row
        for row in ledger
        if date.fromisoformat(row["session_id"]) < ROLL_DATE
        and row["eligible"]
        and row["contracts"] == ["MNQM25"]
    ]
    dates = {row["session_id"] for row in contaminated}
    selected = [
        row
        for row in outcomes
        if row["session_id"] in dates and row["contract"] == "MNQM25"
    ]
    return {
        "classification": "CONTAMINATED_DIAGNOSTIC_ONLY",
        "session_count": len(contaminated),
        "session_ids": sorted(dates),
        "outcome_count": len(selected),
        "gross_dollars": math.fsum(float(row["gross_dollars"]) for row in selected),
        "net_dollars": {
            cost: math.fsum(float(row["net_dollars"][cost]) for row in selected)
            for cost in ("1.22", "2.22", "3.22")
        },
    }


def march_april_contract_table(
    ledger: list[dict[str, Any]], outcomes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    totals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for outcome in outcomes:
        totals[outcome["session_id"]].append(outcome)
    result = []
    for row in ledger:
        day = date.fromisoformat(row["session_id"])
        if day.year == 2025 and day.month in (3, 4):
            day_outcomes = totals[row["session_id"]]
            result.append(
                {
                    "session_id": row["session_id"],
                    "contracts": row["contracts"],
                    "eligible": row["eligible"],
                    "exclusion_reasons": row["exclusion_reasons"],
                    "recorded_outcome_count": len(day_outcomes),
                    "recorded_gross_dollars": math.fsum(
                        float(item["gross_dollars"]) for item in day_outcomes
                    ),
                }
            )
    return result


def render_markdown(report: dict[str, Any]) -> str:
    pre = report["pre_roll_mnqm25"]
    integrity = report["source_session_integrity"]
    rows = "\n".join(f"- {day}" for day in pre["session_ids"]) or "- None"
    return "\n".join(
        [
            "# MNQ wick-short source audit",
            "",
            "**Decision:** SOURCE/LEDGER RECONCILIATION COMPLETE; pre-roll MNQM25 is "
            "flagged as contaminated for diagnosis only. This does not adopt an "
            "exclusion, "
            "change eligibility, rerun calibration, or make a strategy/edge verdict.",
            "",
            "The registered external input and both immutable Phase A publication "
            "hash chains "
            "matched before the source was decoded. Every r2 outcome's ten RTH "
            "components "
            "mapped to raw rows from its recorded session and exact contract.",
            "",
            "Raw session integrity: "
            f"{integrity['observed_rth_sessions']} observed RTH sessions; "
            f"{integrity['eligible_single_contract_complete_sessions']} complete "
            "single-contract sessions; "
            f"{integrity['incomplete_sessions']} incomplete; "
            f"{integrity['mixed_contract_sessions']} mixed-contract.",
            "",
            "## March 2025 lead-month check",
            "",
            "CME's Equity Index roll calendar identifies March 17, 2025 as the "
            "lead-month "
            f"transition. [CME Equity Index roll dates]({CME_ROLL_URL})",
            "",
            "Complete single-contract MNQM25 RTH sessions before that date: "
            f"{pre['session_count']}; recorded outcomes: {pre['outcome_count']}; "
            f"gross recorded dollars: ${pre['gross_dollars']:.2f}.",
            "",
            "Affected session IDs:",
            rows,
            "",
            "Mixed or incomplete days were reported separately by the ledger and were "
            "never used as substitute bars. Contract labels prove source labeling "
            "only; "
            "they do not "
            "prove execution, front-month truth, or a future edge.",
            "",
            "See `report.json` for immutable artifact hashes and the full "
            "deterministic evidence.",
            "",
        ]
    )


def publish(output: Path, report: dict[str, Any]) -> None:
    if output.exists():
        raise AuditError(f"audit publication already exists: {output}")
    output.mkdir(parents=True)
    try:
        (output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        (output / "decision.md").write_text(render_markdown(report))
        commands = "\n".join(
            [
                "# Commands",
                "",
                "```bash",
                f"{sys.executable} tools/audit_mnq_wick_short_source.py --output "
                "_bmad-output/mnq-wick-short-source-audit-YYYYMMDD-new-run",
                "```",
                "",
                "The audit is read-only: it hashes the registered source before "
                "decoding "
                "it and "
                "does not call the calibration runner.",
                "",
            ]
        )
        (output / "commands.md").write_text(commands)
    except Exception:
        for child in output.iterdir():
            child.unlink()
        output.rmdir()
        raise


def audit(input_path: Path = INPUT, phase_a_dir: Path = R2) -> dict[str, Any]:
    hashes = verify_phase_artifacts(ROOT)
    if phase_a_dir.resolve() != R2.resolve():
        raise AuditError("only the published corrected r2 ledger is allowed")
    raw = load_raw_after_hash(input_path)
    eligibility = read_jsonl(phase_a_dir / "eligibility.jsonl")
    outcomes = read_jsonl(phase_a_dir / "outcomes.jsonl")
    integrity = audit_eligibility(eligibility, source_sessions(raw))
    audit_outcomes(outcomes, raw)
    return {
        "schema_version": "mnq-wick-short-source-audit/v1",
        "status": "COMPLETE_DIAGNOSTIC_ONLY",
        "evaluation_allowed": False,
        "source": {
            "path": str(input_path),
            "sha256": INPUT_SHA256,
            "decoded_after_hash_verification": True,
        },
        "phase_a_hashes": hashes,
        "r2_ledger": {
            "eligibility_rows": len(eligibility),
            "outcome_rows": len(outcomes),
            "outcomes_mapped_to_raw_same_contract_rth_components": len(outcomes),
        },
        "source_session_integrity": integrity,
        "roll_calendar": {
            "lead_month_transition": ROLL_DATE.isoformat(),
            "source": CME_ROLL_URL,
        },
        "pre_roll_mnqm25": pre_roll_summary(eligibility, outcomes),
        "march_april_session_contract_evidence": march_april_contract_table(
            eligibility, outcomes
        ),
        "limitations": [
            "Diagnostic only; no exclusion selected after outcomes is adopted as a "
            "strategy filter.",
            "Contract labels establish source labels, not trade execution, front-month "
            "accuracy, or a future edge.",
            "No sealed holdout, strategy mechanics, costs, eligibility, Phase B, "
            "orders, "
            "services, or trade records were accessed or changed.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--phase-a-dir", type=Path, default=R2)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="a new, non-existent directory for this immutable audit publication",
    )
    args = parser.parse_args(argv)
    try:
        report = audit(args.input, args.phase_a_dir)
        publish(args.output, report)
    except AuditError as error:
        print(f"audit refused: {error}", file=sys.stderr)
        return 2
    print(f"Audit complete: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
