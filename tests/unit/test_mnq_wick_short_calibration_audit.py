"""Synthetic, rehashed-corruption regressions for the independent artifact audit."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import subprocess
import sys
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any
from zoneinfo import ZoneInfo

import pytest

AUDIT_DIR = Path(__file__).resolve().parents[2] / (
    "_bmad-output/mnq-wick-short-calibration-phase-a-audit-20260916"
)
COSTS = ("1.22", "2.22", "3.22")
ROLE = "calibration-development-only"
SCHEMA = "mnq-wick-short-calibration-phase-a/v1"
NY = ZoneInfo("America/New_York")


@pytest.fixture(scope="module")
def audit_module() -> ModuleType:
    sys.path.insert(0, str(AUDIT_DIR))
    try:
        return importlib.import_module("mnq_phase_a_audit_outputs")
    finally:
        sys.path.pop(0)


def metric(module: ModuleType, values: list[float], days: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, labels in [
        ("session_clustered", days),
        ("iso_week_clustered", [module.week(d) for d in days]),
    ]:
        reference = module.reference_summary(values, labels)
        sizes = Counter(labels)
        result[key] = {
            "N": len(values),
            "G": len(sizes),
            "degrees_of_freedom": len(sizes) - 1,
            "status": "ASSESSABLE" if reference["ci"] else "UNASSESSABLE",
            "reason": None if reference["ci"] else "UNASSESSABLE_FIXTURE",
            "variance": reference["variance"],
            "se": reference["se"],
            "t_critical": reference["t_critical"],
            "interval": reference["ci"],
            "cluster_sizes": [
                {"label": label, "count": n} for label, n in sizes.items()
            ],
            "largest_cluster_fraction": max(sizes.values()) / len(values),
            "cluster_share_hhi": sum((n / len(values)) ** 2 for n in sizes.values()),
        }
    intervals = [
        result[key]["interval"] for key in ("session_clustered", "iso_week_clustered")
    ]
    result["envelope"] = {
        "role": "sensitivity-envelope-only",
        "status": "ASSESSABLE",
        "reason": None,
        "interval": [min(r[0] for r in intervals), max(r[1] for r in intervals)],
    }
    counts = Counter(values)
    cumulative = 0
    ecdf = []
    for value, count in sorted(counts.items()):
        cumulative += count
        ecdf.append(
            {
                "value": value,
                "count": count,
                "cumulative_probability": cumulative / len(values),
            }
        )
    ordered = sorted(values)
    quantiles = {}
    for key in ("0", "0.01", "0.05", "0.25", "0.5", "0.75", "0.95", "0.99", "1"):
        position = float(key) * (len(values) - 1)
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        quantiles[key] = ordered[lower] + (position - lower) * (
            ordered[upper] - ordered[lower]
        )
    result["descriptive"] = {
        "observation_count": len(values),
        "total": sum(values),
        "mean": sum(values) / len(values),
        "sample_sd": reference["sd"],
        "distribution": {"ecdf": ecdf, "quantiles": quantiles},
    }
    return result


@pytest.fixture(scope="module")
def synthetic_bundle(audit_module: ModuleType) -> dict[str, Any]:
    eligibility, outcomes, sessions = [], [], []
    for index in range(576):
        day = date(2024, 1, 2) + timedelta(days=index)
        day_id = day.isoformat()
        start = datetime.combine(day, time(9, 31), tzinfo=NY)
        final = start + timedelta(minutes=389)
        eligible = index < 515
        count = 2 if index < 234 else 1 if index < 351 else 0
        short = 515 <= index < 538
        mixed = index >= 538
        reason = (
            "INCOMPLETE_RTH_MINUTES" if short else "MIXED_CONTRACT" if mixed else None
        )
        eligibility.append(
            {
                "session_id": day_id,
                "iso_week": audit_module.week(day_id),
                "contracts": ["T", "U"] if mixed else ["T"],
                "contract_minute_counts": (
                    {"T": 195, "U": 195} if mixed else {"T": 389 if short else 390}
                ),
                "eligible": eligible,
                "primary_exclusion_reason": reason,
                "exclusion_reasons": [reason] if reason else [],
                "expected_minute_count": 390,
                "observed_minute_count": 389 if short else 390,
                "missing_minute_labels": [final.isoformat()] if short else [],
                "unexpected_minute_labels": [],
                "first_minute_label_local": start.isoformat(),
                "last_minute_label_local": (
                    final - timedelta(minutes=1) if short else final
                ).isoformat(),
                "signal_count": count if eligible else None,
                "signal_count_null_reason": None if eligible else "INELIGIBLE_SESSION",
                "sample_role": ROLE,
            }
        )
        if not eligible:
            continue
        rows = []
        for slot in range(0, count * 2, 2):
            label = start + timedelta(minutes=4 + slot * 5)
            following_close = 200 + index % 7 - 3
            parts = []
            for is_following in (False, True):
                group = []
                for minute_index in range(5):
                    stamp = label + timedelta(
                        minutes=(1 if is_following else -4) + minute_index
                    )
                    ohlc = (
                        {
                            "open": 200,
                            "high": max(200, following_close) + 1,
                            "low": min(200, following_close) - 1,
                            "close": following_close,
                        }
                        if is_following
                        else {"open": 100, "high": 103, "low": 100, "close": 101}
                    )
                    group.append(
                        {
                            **ohlc,
                            "contract": "T",
                            "source_record_id": stamp.astimezone(
                                timezone.utc
                            ).isoformat(),
                            "minute_label_utc": stamp.astimezone(
                                timezone.utc
                            ).isoformat(),
                            "minute_label_local": stamp.isoformat(),
                        }
                    )
                parts.append(group)
            gross = float(2 * (200 - following_close))
            rows.append(
                {
                    "signal_id": f"{day_id}/slot-{slot:02d}",
                    "session_id": day_id,
                    "iso_week": audit_module.week(day_id),
                    "contract": "T",
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
                    "signal_component_minutes": parts[0],
                    "reference_component_minutes": parts[1],
                    "body": 1,
                    "upper_wick": 2,
                    "lower_wick": 0,
                    "signal_bar_ohlc": {
                        "open": 100,
                        "high": 103,
                        "low": 100,
                        "close": 101,
                    },
                    "next_open": 200,
                    "next_close": following_close,
                    "gross_dollars": gross,
                    "net_dollars": {cost: gross - float(cost) for cost in COSTS},
                    "sample_role": ROLE,
                }
            )
        gross_total = sum(row["gross_dollars"] for row in rows)
        totals = {cost: gross_total - float(cost) * count for cost in COSTS}
        for row in rows:
            row.update(
                session_signal_count=count,
                session_gross_total_dollars=gross_total,
                session_net_total_dollars=totals,
            )
        outcomes.extend(rows)
        sessions.append(
            {
                "session_id": day_id,
                "iso_week": audit_module.week(day_id),
                "contract": "T",
                "signal_count": count,
                "gross_total_dollars": gross_total,
                "net_total_dollars": totals,
                "sample_role": ROLE,
            }
        )
    signal_days = [row["session_id"] for row in outcomes]
    session_days = [row["session_id"] for row in sessions]
    statistics = {
        "per_signal": {
            "gross_dollars": metric(
                audit_module, [r["gross_dollars"] for r in outcomes], signal_days
            ),
            "net_dollars": {
                cost: metric(
                    audit_module,
                    [r["net_dollars"][cost] for r in outcomes],
                    signal_days,
                )
                for cost in COSTS
            },
        },
        "per_eligible_session": {
            "signal_count": metric(
                audit_module, [r["signal_count"] for r in sessions], session_days
            ),
            "gross_total_dollars": metric(
                audit_module, [r["gross_total_dollars"] for r in sessions], session_days
            ),
            "net_total_dollars": {
                cost: metric(
                    audit_module,
                    [r["net_total_dollars"][cost] for r in sessions],
                    session_days,
                )
                for cost in COSTS
            },
        },
    }
    report = {
        "schema_version": SCHEMA,
        "status": "COMPLETE_CALIBRATION_ONLY",
        "phase": "A",
        "sample_role": ROLE,
        "original_gate_verdict": "POWER_UNDETERMINED",
        "evaluation_allowed": False,
        "confirmation_authorized": False,
        "counts_reconciled_before_alignment": True,
        "cost_scenarios_dollars": [1.22, 2.22, 3.22],
        "point_value_dollars": 2,
        "counts": {
            "rth_sessions_seen": 576,
            "eligible_sessions": 515,
            "excluded_sessions": 61,
            "exclusion_reasons": {"INCOMPLETE_RTH_MINUTES": 23, "MIXED_CONTRACT": 38},
            "post_cutoff_records_skipped": 62277,
            "total_signals": 585,
            "sessions_with_signal": 351,
            "sessions_without_signal": 164,
            "signals_per_eligible_session": 585 / 515,
        },
        "statistics": statistics,
    }
    return {
        "report.json": report,
        "report.md": "Synthetic calibration only",
        "eligibility.jsonl": eligibility,
        "outcomes.jsonl": outcomes,
        "sessions.jsonl": sessions,
    }


def write_bundle(path: Path, bundle: dict[str, Any], corruption: str = "") -> None:
    hashes = {}
    for name, value in bundle.items():
        content = (
            "\n".join(json.dumps(row) for row in value)
            if name.endswith(".jsonl")
            else value if name.endswith(".md") else json.dumps(value)
        )
        (path / name).write_text(content)
        hashes[name] = hashlib.sha256(content.encode()).hexdigest()
    if corruption == "manifest_inventory":
        del hashes["report.json"]
    manifest = {
        "schema_version": "bad" if corruption == "manifest_schema" else SCHEMA,
        "sample_role": ROLE,
        "evaluation_allowed": False,
        "output_sha256": hashes,
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    complete = {
        "schema_version": "bad" if corruption == "complete_schema" else SCHEMA,
        "status": (
            "FAILED" if corruption == "complete_status" else "COMPLETE_CALIBRATION_ONLY"
        ),
        "evaluation_allowed": False,
        "manifest_sha256": hashlib.sha256(
            (path / "manifest.json").read_bytes()
        ).hexdigest(),
    }
    (path / "COMPLETE.json").write_text(json.dumps(complete))


def test_synthetic_bundle_passes(
    audit_module: ModuleType, synthetic_bundle: dict[str, Any], tmp_path: Path
) -> None:
    write_bundle(tmp_path, synthetic_bundle)
    audit_module.audit(tmp_path)


@pytest.mark.parametrize(
    "corruption",
    [
        "manifest_inventory",
        "manifest_schema",
        "complete_status",
        "complete_schema",
        "report_status",
        "report_schema",
        "interval",
        "envelope",
        "quantiles",
        "cluster_labels",
        "cluster_fraction",
        "cluster_hhi",
        "t_critical",
        "ecdf",
        "eligibility_count",
        "eligibility_contract",
        "minute_count",
        "coverage",
        "report_count",
        "report_exclusions",
        "report_cutoff",
        "outcome_contract",
    ],
)
def test_rehashed_corruptions_refuse(
    audit_module: ModuleType,
    synthetic_bundle: dict[str, Any],
    tmp_path: Path,
    corruption: str,
) -> None:
    bundle = copy.deepcopy(synthetic_bundle)
    report = bundle["report.json"]
    metric_row = report["statistics"]["per_signal"]["gross_dollars"]
    cluster = metric_row["session_clustered"]
    if corruption in ("report_status", "report_schema"):
        report["status" if corruption == "report_status" else "schema_version"] = "bad"
    elif corruption == "interval":
        cluster["interval"] = []
    elif corruption == "envelope":
        metric_row["envelope"]["interval"] = metric_row["envelope"]["interval"][:1]
    elif corruption == "quantiles":
        del metric_row["descriptive"]["distribution"]["quantiles"]["0.99"]
    elif corruption == "cluster_labels":
        cluster["cluster_sizes"][0]["label"] = "wrong-day"
    elif corruption in ("cluster_fraction", "cluster_hhi", "t_critical"):
        field = {
            "cluster_fraction": "largest_cluster_fraction",
            "cluster_hhi": "cluster_share_hhi",
            "t_critical": "t_critical",
        }[corruption]
        cluster[field] = 123
    elif corruption == "ecdf":
        metric_row["descriptive"]["distribution"]["ecdf"][0][
            "cumulative_probability"
        ] = 1
    elif corruption == "eligibility_count":
        bundle["eligibility.jsonl"][0]["signal_count"] += 1
    elif corruption == "eligibility_contract":
        bundle["eligibility.jsonl"][0]["contracts"] = ["wrong-contract"]
        bundle["eligibility.jsonl"][0]["contract_minute_counts"] = {
            "wrong-contract": 390
        }
    elif corruption == "minute_count":
        bundle["eligibility.jsonl"][0]["observed_minute_count"] = 389
    elif corruption == "coverage":
        bundle["eligibility.jsonl"][515]["missing_minute_labels"] = []
    elif corruption == "report_count":
        report["counts"]["total_signals"] = 584
    elif corruption == "report_exclusions":
        report["counts"]["exclusion_reasons"]["MIXED_CONTRACT"] = 39
    elif corruption == "report_cutoff":
        report["counts"]["post_cutoff_records_skipped"] = 0
    elif corruption == "outcome_contract":
        bundle["outcomes.jsonl"][0]["contract"] = "wrong-contract"
    write_bundle(tmp_path, bundle, corruption)
    with pytest.raises(AssertionError):
        audit_module.audit(tmp_path)


@pytest.mark.parametrize(
    "script", ["mnq_phase_a_audit_outputs.py", "mnq_phase_a_independent_audit.py"]
)
def test_audit_scripts_refuse_optimized_execution(script: str, tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-O", str(AUDIT_DIR / script), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "refuses optimized execution" in result.stderr


def test_archived_oracle_selfcheck_is_location_independent(tmp_path: Path) -> None:
    for name in ("mnq_phase_a_independent_audit.py", "independent-oracles.json"):
        (tmp_path / name).write_bytes((AUDIT_DIR / name).read_bytes())
    result = subprocess.run(
        [sys.executable, str(tmp_path / "mnq_phase_a_independent_audit.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "four cluster oracles" in result.stdout
