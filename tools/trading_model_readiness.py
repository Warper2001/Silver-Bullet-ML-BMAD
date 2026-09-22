"""Offline, descriptive trading-model readiness audit; never permits strategy tests.

No trader imports, broker access, model downloads, fitted parameters or returns.
Run with explicitly named unsealed CSVs and a fresh output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    ROOT.parents[2]
    if ROOT.parent.name == "worktrees" and ROOT.parent.parent.name == ".claude"
    else ROOT
)
APPROVED_INPUTS = frozenset(
    DATA_ROOT / name
    for name in (
        "data/mim_x/mnq_1min_by_contract.csv",
        "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv",
        "_bmad-output/diagnostics_gap_fade_splice_20260916/"
        "mnq_1min_2025_frontmonth.csv",
        ".claude/worktrees/gapfade-splice-sensitivity/_bmad-output/"
        "diagnostics_gap_fade_splice_20260916/mnq_1min_2025_frontmonth.csv",
    )
)
REGISTRATION = "_bmad-output/preregistration_trading_model_readiness.md"
REGISTRATION_COMMIT = "ff7fbeb74491704ea0a5c36e5ac1aff2d5cf633f"
REGISTRATION_HASH = "cf88f0870acb63999c6e025247daf4cce0ce3a51c82f13ec5453b536490ad4f0"
NY = ZoneInfo("America/New_York")
FIELDS = ("timestamp", "open", "high", "low", "close", "volume")
ALIASES = {"timestamp": "timestamp", "totalvolume": "volume"}
FULL_RTH = (1 << 390) - 1
FULL_GROUP = (1 << 15) - 1
DATA_GAPS = [
    "SOURCE_PROVENANCE_NOT_ADMITTED",
    "CAUSAL_FRONTMONTH_SELECTION_NOT_ADMITTED",
    "BAR_LABEL_AND_DECISION_TIME_AVAILABILITY_UNVERIFIED",
    "HISTORICAL_EXCHANGE_CALENDAR_UNVERIFIED",
    "FILL_AND_COST_EVIDENCE_UNVERIFIED",
    "UNTOUCHED_EVALUATION_INTERVAL_NOT_REGISTERED",
    "PRETRAINING_EXPOSURE_NOT_AUDITED",
]


class AuditError(ValueError):
    """The audit cannot safely process the requested inputs or destination."""


def safe_path(path: Path) -> Path:
    for candidate in (path.absolute(), path.resolve()):
        if any(part.lower() == "sealed_holdout" for part in candidate.parts):
            raise AuditError("sealed_holdout access is prohibited")
    return path.resolve()


def digest(path: Path) -> str:
    path = safe_path(path)
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.second or parsed.microsecond:
            raise ValueError
        normalized = parsed.astimezone(timezone.utc)
        if normalized.second or normalized.microsecond:
            raise ValueError
        return normalized
    except (ValueError, AttributeError, OverflowError) as exc:
        raise AuditError("timestamp must be an aware whole ISO minute") from exc


def schema(names: Sequence[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in names:
        key = ALIASES.get(name.lower(), name.lower())
        if key in mapping:
            raise AuditError("ambiguous duplicate column names")
        mapping[key] = name
    if not set(FIELDS) <= mapping.keys():
        raise AuditError("CSV requires timestamp, open, high, low, close, volume")
    return mapping


def grid_summary(
    masks: dict[tuple[str, str], int], duplicates: set[tuple[str, str]]
) -> dict[str, Any]:
    """Both grids are hypotheses, not exchange-calendar or source validation."""
    result: dict[str, Any] = {}
    for label, opening in (("start", 570), ("end", 571)):
        days: set[str] = set()
        complete_days: set[str] = set()
        groups = complete = duplicate_groups = 0
        for key, mask in masks.items():
            regular = (mask >> opening) & FULL_RTH
            if not regular:
                continue
            days.add(key[0])
            if key in duplicates:
                duplicate_groups += 1
                continue
            groups += sum(
                ((regular >> slot) & FULL_GROUP) == FULL_GROUP
                for slot in range(0, 390, 15)
            )
            if regular == FULL_RTH:
                complete += 1
                complete_days.add(key[0])
        result[label] = {
            "observed_weekday_rth_dates": len(days),
            "full_regular_grid_contract_days": complete,
            "dates_with_any_full_regular_grid": len(complete_days),
            "candidate_15min_contract_groups": groups,
            "duplicate_contract_days_excluded_from_grid_counts": duplicate_groups,
            "qualification": "DESCRIPTIVE_ONLY_NOT_ADMITTED_OR_INDEPENDENT",
        }
    return result


def inspect_csv(path: Path) -> dict[str, Any]:
    path = safe_path(path)
    result: dict[str, Any] = {
        "path": str(path),
        "data_gate": "HOLD_DATA",
        "contract_identity": "UNASSESSABLE",
    }
    if not path.exists():
        return dict(result, status="MISSING", sha256=None)
    if not path.is_file() or path.suffix.lower() != ".csv":
        raise AuditError("inputs must be regular CSV files")
    before = digest(path)
    result.update(sha256=before, bytes=path.stat().st_size)
    started = time.perf_counter()
    counts: Counter[str] = Counter()
    contracts: set[str] = set()
    masks: dict[tuple[str, str], int] = defaultdict(int)
    seen_masks: dict[tuple[str, str], int] = defaultdict(int)
    duplicate_days: set[tuple[str, str]] = set()
    invalid_reasons: Counter[str] = Counter()
    invalid_examples: list[dict[str, Any]] = []
    contracts_per_day: dict[str, set[str]] = defaultdict(set)
    last: dict[str, datetime] = {}
    first_stamp: datetime | None = None
    last_stamp: datetime | None = None
    error: str | None = None
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream, strict=True)
            names = reader.fieldnames or []
            mapping = schema(names)
            result["columns"] = names
            result["contract_identity"] = (
                "OBSERVED_NOT_AUTHENTICATED" if "contract" in mapping else "ABSENT"
            )
            for record in reader:
                counts["rows"] += 1
                try:
                    if None in record or any(
                        value is None for value in record.values()
                    ):
                        raise AuditError("row width differs from header")
                    stamp = timestamp(record[mapping["timestamp"]])
                    contract = (
                        record[mapping["contract"]].strip()
                        if "contract" in mapping
                        else "UNKNOWN"
                    )
                    if "contract" in mapping and not re.fullmatch(
                        r"MNQ[HMUZ]\d{2}", contract
                    ):
                        raise AuditError("unrecognized MNQ quarterly contract")
                    # Identity must be counted even when the prices are invalid.
                    local = stamp.astimezone(NY)
                    day = local.date().isoformat()
                    contracts_per_day[day].add(contract)
                    contracts.add(contract)
                    key = (day, contract)
                    bit = 1 << (local.hour * 60 + local.minute)
                    if contract in last and stamp <= last[contract]:
                        counts["nonincreasing_within_contract_rows"] += 1
                    last[contract] = stamp
                    if local.weekday() < 5:
                        if seen_masks[key] & bit:
                            counts["duplicate_weekday_contract_minutes"] += 1
                            duplicate_days.add(key)
                        seen_masks[key] |= bit
                    else:
                        counts["weekend_rows"] += 1
                    values = [float(record[mapping[field]]) for field in FIELDS[1:]]
                    opening, high, low, close, volume = values
                    if not all(math.isfinite(value) for value in values):
                        raise AuditError("non-finite numeric value")
                    if min(opening, high, low, close) <= 0 or volume < 0:
                        raise AuditError("invalid price or volume")
                    if high < max(opening, close, low) or low > min(
                        opening, close, high
                    ):
                        raise AuditError("inconsistent OHLC")
                except (AuditError, ValueError, TypeError, OverflowError) as exc:
                    counts["invalid_rows"] += 1
                    reason = (
                        str(exc)
                        if isinstance(exc, AuditError)
                        else "invalid numeric or temporal value"
                    )
                    invalid_reasons[reason] += 1
                    if len(invalid_examples) < 12:
                        invalid_examples.append(
                            {"line_end": reader.line_num, "reason": reason}
                        )
                    continue
                counts["valid_rows"] += 1
                first_stamp = stamp if first_stamp is None else min(first_stamp, stamp)
                last_stamp = stamp if last_stamp is None else max(last_stamp, stamp)
                if local.weekday() < 5:
                    masks[key] |= bit
    except (AuditError, csv.Error, UnicodeError) as exc:
        error = str(exc)
    elapsed = time.perf_counter() - started
    if digest(path) != before:
        raise AuditError("input changed during audit; discard this run")
    result.update(
        status=(
            "INVALID_SCHEMA"
            if error
            else ("EMPTY" if not counts["rows"] else "AUDITED")
        ),
        counts={
            key: counts[key]
            for key in (
                "rows",
                "valid_rows",
                "invalid_rows",
                "nonincreasing_within_contract_rows",
                "duplicate_weekday_contract_minutes",
                "weekend_rows",
            )
        },
        first_valid_timestamp=first_stamp.isoformat() if first_stamp else None,
        last_valid_timestamp=last_stamp.isoformat() if last_stamp else None,
        observed_dates=len(contracts_per_day),
        observed_weekday_dates=sum(
            datetime.fromisoformat(day).weekday() < 5 for day in contracts_per_day
        ),
        dates_with_multiple_contracts=(
            sum(len(items) > 1 for items in contracts_per_day.values())
            if result["contract_identity"] == "OBSERVED_NOT_AUTHENTICATED"
            else None
        ),
        invalid_row_reasons=dict(sorted(invalid_reasons.items())),
        invalid_row_examples=invalid_examples,
        invalid_row_examples_limit=12,
        contracts=sorted(contracts),
        grid_hypotheses=grid_summary(masks, duplicate_days),
        measurement={
            "audit_seconds": elapsed,
            "rows_per_second": counts["rows"] / elapsed if elapsed > 0 else None,
        },
        structural_quality=(
            "INVALID"
            if error
            or counts["invalid_rows"]
            or counts["nonincreasing_within_contract_rows"]
            or counts["duplicate_weekday_contract_minutes"]
            or not counts["valid_rows"]
            else "STRUCTURAL_CHECKS_ONLY"
        ),
    )
    if error:
        result["schema_error"] = error
    return result


def compute_readiness() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    gpus: list[str] = []
    status = "NVIDIA_TOOL_UNAVAILABLE"
    if executable:
        try:
            probe = subprocess.run(
                [executable, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if probe.returncode == 0:
                gpus = probe.stdout.strip().splitlines()
                status = "DEVICE_QUERY_SUCCEEDED" if gpus else "NO_DEVICES_REPORTED"
            else:
                status = "DEVICE_QUERY_FAILED"
        except (OSError, subprocess.TimeoutExpired):
            status = "DEVICE_QUERY_FAILED"
    return {
        "python": platform.python_version(),
        "interpreter": sys.executable,
        "reported_logical_cpus": os.cpu_count(),
        "packages_available": {
            name: importlib.util.find_spec(name) is not None
            for name in ("numpy", "pandas", "scipy", "torch", "transformers")
        },
        "nvidia_probe": status,
        "devices": gpus,
        "gpu_training_benchmark": "NOT_MEASURED",
        "training_gpu_hours": None,
        "training_cost": None,
        "qualification": (
            "CSV audit timing is CPU preprocessing only; no training extrapolation."
        ),
        "fp16_parameter_storage_gib": {
            str(count): count * 2 / (1024**3)
            for count in (
                4_100_000,
                24_700_000,
                102_300_000,
                4_000_000_000,
                8_000_000_000,
            )
        },
        "storage_qualification": (
            "Weights only; excludes activations, gradients, optimizer, KV cache "
            "and all framework overhead. Not a VRAM requirement or fit prediction."
        ),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Trading-model readiness",
        "",
        "**HOLD — strategy testing is not authorized by this audit.**",
        "",
        f"Runner-supplied revision (not Git-verified): `{report['source_revision']}`.",
        f"Readiness preregistration: `{REGISTRATION_COMMIT}`.",
        "",
        "## Dataset observations",
        "",
    ]
    for item in report["datasets"]:
        multiple = item.get("dates_with_multiple_contracts")
        multiple_text = f"{multiple:,}" if multiple is not None else "UNASSESSABLE"
        lines.extend(
            [
                f"### {Path(item['path']).name}",
                "",
                f"Path: `{item['path']}`.",
                f"Status: {item['status']}; gate: HOLD_DATA.",
            ]
        )
        if "counts" in item:
            lines.extend(
                [
                    f"Rows: {item['counts']['rows']:,}; structurally valid rows: "
                    f"{item['counts']['valid_rows']:,}; invalid rows: "
                    f"{item['counts']['invalid_rows']:,}.",
                    f"Observed weekday dates: {item['observed_weekday_dates']:,}; "
                    "dates with multiple contracts: "
                    f"{multiple_text}.",
                    f"Contract identity: {item['contract_identity']}; "
                    f"structural quality: {item['structural_quality']}.",
                    f"Range: {item['first_valid_timestamp']} "
                    f"to {item['last_valid_timestamp']}.",
                    "CPU CSV audit: "
                    f"{item['measurement']['audit_seconds']:.3f} seconds, "
                    f"{item['measurement']['rows_per_second']:.0f} "
                    "rows/second (hashing excluded).",
                    "",
                ]
            )
            for label, grid in item["grid_hypotheses"].items():
                lines.append(
                    f"- {label}-label hypothesis: "
                    f"{grid['dates_with_any_full_regular_grid']:,} "
                    "dates with any full regular grid; "
                    f"{grid['candidate_15min_contract_groups']:,} "
                    "candidate contract-groups."
                )
            if item["invalid_row_reasons"]:
                lines.extend(["", "Invalid row reasons:", ""])
                lines.extend(
                    f"- {reason}: {count}"
                    for reason, count in item["invalid_row_reasons"].items()
                )
        lines.extend(["", f"SHA-256: `{item['sha256']}`.", ""])
    lines.extend(
        [
            "These are overlapping, previously researched development inputs. "
            "Counts cannot be added across files, contracts or windows. "
            "A full regular-length weekday grid is not authenticated calendar "
            "completeness or decision-time availability.",
            "",
            "## Gates",
            "",
            "Data: HOLD_DATA. Missing evidence:",
            "",
            *[f"- {gap}" for gap in report["data_gaps"]],
            "",
            "Power: UNASSESSABLE. No registered economic effect, eligible untouched "
            "evaluation sample, dependence-adjusted variance or comparison "
            "allocation. No strategy returns were computed.",
            "",
            "## Compute",
            "",
            f"Python {report['compute']['python']}; "
            f"{report['compute']['reported_logical_cpus']} reported logical CPUs; "
            f"NVIDIA probe: {report['compute']['nvidia_probe']}.",
            "Package visibility: `"
            + json.dumps(report["compute"]["packages_available"], sort_keys=True)
            + "`.",
            "GPU training throughput, training hours and training cost: NOT MEASURED. "
            "Preprocessing throughput cannot substitute for a model benchmark.",
            "",
            "## Next requirements",
            "",
            "Establish source/causal-roll/calendar/availability/cost provenance; "
            "register evaluation data and effect-size justification; run the "
            "experiment-specific power gate. Then benchmark the pinned model in "
            "an isolated training environment before budgeting training. "
            "No data acquisition, dependency installation, model training or "
            "deployment was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def run(inputs: Sequence[Path], output: Path, revision: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise AuditError("source revision must be a full canonical Git SHA")
    if not inputs:
        raise AuditError("at least one explicit input is required")
    resolved = [safe_path(path) for path in inputs]
    if len(set(resolved)) != len(resolved):
        raise AuditError("duplicate input paths or aliases")
    if any(path not in APPROVED_INPUTS for path in resolved):
        raise AuditError("input is not in the registered development-data allowlist")
    destination = safe_path(output)
    prohibited = {"data", "logs", "models", ".git", ".venv", ".venv-research"}
    if any(part.lower() in prohibited for part in destination.parts):
        raise AuditError("output cannot be in data, live, model or environment paths")
    if destination.exists() or any(
        path.is_relative_to(destination) for path in resolved
    ):
        raise AuditError("output must be a new directory separate from inputs")
    if digest(ROOT / REGISTRATION) != REGISTRATION_HASH:
        raise AuditError("readiness preregistration hash mismatch")
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": revision,
        "source_revision_verification": "RUNNER_SUPPLIED_NOT_GIT_VERIFIED",
        "input_scope": "READINESS_REGISTRATION_ALLOWLIST",
        "completion_marker": "COMPLETE.json",
        "audit_sha256": digest(Path(__file__)),
        "preregistration_commit": REGISTRATION_COMMIT,
        "preregistration_sha256": REGISTRATION_HASH,
        "strategy_testing_authorized": False,
        "data_gate": "HOLD_DATA",
        "data_gaps": DATA_GAPS,
        "power_gate": {
            "status": "UNASSESSABLE",
            "strategy_test_permitted": False,
            "reason": (
                "No registered effect, eligible evaluation population, "
                "dependence model or multiplicity allocation."
            ),
        },
        "datasets": [inspect_csv(path) for path in resolved],
        "compute": compute_readiness(),
    }
    # Render before creating outputs; only the final marker certifies completion.
    report_json = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    report_markdown = markdown(report)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "report.json").write_text(report_json, encoding="utf-8")
    (destination / "report.md").write_text(report_markdown, encoding="utf-8")
    completion = {
        name: digest(destination / name) for name in ("report.json", "report.md")
    }
    (destination / "COMPLETE.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args(argv)
    try:
        run(args.input, args.output_dir, args.source_revision)
    except (AuditError, OSError) as exc:
        print(f"Audit refused: {exc}", file=sys.stderr)
        return 1
    print(
        "Readiness audit complete: HOLD_DATA, UNASSESSABLE power; "
        "no strategy test permitted."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
