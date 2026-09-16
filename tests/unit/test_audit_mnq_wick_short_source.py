from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/audit_mnq_wick_short_source.py"


def load_module():
    spec = importlib.util.spec_from_file_location("audit_mnq_wick_short_source", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def raw_row(stamp: str, contract: str = "MNQM25") -> dict[str, object]:
    return {
        "TimeStamp": stamp,
        "Contract": contract,
        "Open": 100,
        "High": 101,
        "Low": 99,
        "Close": 100,
    }


def test_changed_bound_source_refuses_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    source = tmp_path / "source.json"
    source.write_text("this is not JSON")
    monkeypatch.setattr(module, "INPUT", source)
    monkeypatch.setattr(module, "INPUT_SHA256", "0" * 64)
    with pytest.raises(module.AuditError, match="hash mismatch"):
        module.load_raw_after_hash(source)


def test_raw_component_mapping_refuses_duplicate_components() -> None:
    module = load_module()
    stamp = datetime(2025, 3, 14, 14, 31, tzinfo=timezone.utc)
    component = {
        "source_record_id": stamp.isoformat(),
        "contract": "MNQM25",
        "open": 100,
        "high": 101,
        "low": 99,
        "close": 100,
    }
    outcome = {
        "signal_id": "2025-03-14/slot-00",
        "session_id": "2025-03-14",
        "contract": "MNQM25",
        "signal_component_minutes": [component] * 5,
        "reference_component_minutes": [component] * 5,
        "signal_bar_label_local": stamp.astimezone(module.NY).isoformat(),
    }
    raw = {
        stamp: {
            "stamp": stamp,
            "local": stamp.astimezone(module.NY),
            "contract": "MNQM25",
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
        }
    }
    with pytest.raises(module.AuditError, match="window is not contiguous"):
        module.audit_outcomes([outcome], raw)


def test_pre_roll_classification_counts_ledger_outcomes() -> None:
    module = load_module()
    ledger = [
        {"session_id": "2025-03-14", "eligible": True, "contracts": ["MNQM25"]},
        {"session_id": "2025-03-17", "eligible": True, "contracts": ["MNQM25"]},
        {"session_id": "2025-03-13", "eligible": False, "contracts": ["MNQM25"]},
    ]
    outcomes = [
        {
            "session_id": "2025-03-14",
            "contract": "MNQM25",
            "gross_dollars": -4,
            "net_dollars": {"1.22": -5.22, "2.22": -6.22, "3.22": -7.22},
        }
    ]
    result = module.pre_roll_summary(ledger, outcomes)
    assert result["classification"] == "CONTAMINATED_DIAGNOSTIC_ONLY"
    assert result["session_ids"] == ["2025-03-14"]
    assert result["outcome_count"] == 1 and result["gross_dollars"] == -4


def test_eligibility_reconciles_contract_counts_labels_and_reasons() -> None:
    module = load_module()
    complete_day = date(2025, 3, 3)
    incomplete_day = date(2025, 3, 4)
    mixed_day = date(2025, 3, 5)

    def rows(day: date, contracts: list[str]) -> list[dict[str, object]]:
        return [
            {"local": stamp, "contract": contract}
            for stamp, contract in zip(sorted(module.expected_minutes(day)), contracts)
        ]

    full = ["MNQM25"] * 390
    sessions = {
        complete_day: rows(complete_day, full),
        incomplete_day: rows(incomplete_day, full[:-1]),
        mixed_day: rows(mixed_day, ["MNQH25", *full[1:]]),
    }
    ledger = [
        {
            "session_id": complete_day.isoformat(),
            "contracts": ["MNQM25"],
            "contract_minute_counts": {"MNQM25": 390},
            "observed_minute_count": 390,
            "missing_minute_labels": [],
            "unexpected_minute_labels": [],
            "exclusion_reasons": [],
            "primary_exclusion_reason": None,
            "eligible": True,
        },
        {
            "session_id": incomplete_day.isoformat(),
            "contracts": ["MNQM25"],
            "contract_minute_counts": {"MNQM25": 389},
            "observed_minute_count": 389,
            "missing_minute_labels": [
                max(module.expected_minutes(incomplete_day)).isoformat()
            ],
            "unexpected_minute_labels": [],
            "exclusion_reasons": ["INCOMPLETE_RTH_MINUTES"],
            "primary_exclusion_reason": "INCOMPLETE_RTH_MINUTES",
            "eligible": False,
        },
        {
            "session_id": mixed_day.isoformat(),
            "contracts": ["MNQH25", "MNQM25"],
            "contract_minute_counts": {"MNQH25": 1, "MNQM25": 389},
            "observed_minute_count": 390,
            "missing_minute_labels": [],
            "unexpected_minute_labels": [],
            "exclusion_reasons": ["MIXED_CONTRACT"],
            "primary_exclusion_reason": "MIXED_CONTRACT",
            "eligible": False,
        },
    ]
    assert module.audit_eligibility(ledger, sessions)["observed_rth_sessions"] == 3


def test_publish_writes_complete_diagnostic_artifacts(tmp_path: Path) -> None:
    module = load_module()
    output = tmp_path / "audit"
    module.publish(
        output,
        {
            "pre_roll_mnqm25": {
                "session_count": 0,
                "outcome_count": 0,
                "gross_dollars": 0.0,
                "session_ids": [],
            },
            "source_session_integrity": {
                "observed_rth_sessions": 0,
                "eligible_single_contract_complete_sessions": 0,
                "incomplete_sessions": 0,
                "mixed_contract_sessions": 0,
            },
        },
    )
    assert (output / "report.json").is_file()
    assert (output / "decision.md").is_file()
    assert (output / "commands.md").is_file()
    with pytest.raises(module.AuditError, match="already exists"):
        module.publish(output, {"pre_roll_mnqm25": {"session_ids": []}})


def test_main_requires_a_fresh_publication_destination() -> None:
    module = load_module()
    with pytest.raises(SystemExit) as result:
        module.main([])
    assert result.value.code == 2


def test_changed_report_hash_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    directory = tmp_path / "_bmad-output" / "published"
    directory.mkdir(parents=True)
    for name in ("COMPLETE.json", "manifest.json", "report.json"):
        (directory / name).write_text(json.dumps({"output_sha256": {}}))
    monkeypatch.setattr(
        module,
        "PINNED_PHASE_A",
        {
            "published": {
                "COMPLETE.json": hashlib.sha256(
                    (directory / "COMPLETE.json").read_bytes()
                ).hexdigest(),
                "manifest.json": hashlib.sha256(
                    (directory / "manifest.json").read_bytes()
                ).hexdigest(),
                "report.json": "f" * 64,
            }
        },
    )
    with pytest.raises(module.AuditError, match="published Phase A hash mismatch"):
        module.verify_phase_artifacts(tmp_path)


def test_audit_orchestrates_raw_outcome_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    phase = tmp_path / "r2"
    phase.mkdir()
    calls: list[str] = []
    monkeypatch.setattr(module, "R2", phase)
    monkeypatch.setattr(module, "verify_phase_artifacts", lambda root: {})
    monkeypatch.setattr(module, "load_raw_after_hash", lambda path: {})
    monkeypatch.setattr(module, "read_jsonl", lambda path: [])
    monkeypatch.setattr(
        module, "audit_eligibility", lambda ledger, raw: calls.append("eligibility")
    )
    monkeypatch.setattr(
        module, "audit_outcomes", lambda outcomes, raw: calls.append("outcomes")
    )
    monkeypatch.setattr(module, "pre_roll_summary", lambda ledger, outcomes: {})
    monkeypatch.setattr(
        module, "march_april_contract_table", lambda ledger, outcomes: []
    )
    module.audit(module.INPUT, phase)
    assert calls == ["eligibility", "outcomes"]
