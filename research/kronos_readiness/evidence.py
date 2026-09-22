"""Strict documentary inputs and fresh, manifested evidence artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.kronos_evaluation_preflight import EVIDENCE as ORIGINAL_EVIDENCE

EVIDENCE = {
    **ORIGINAL_EVIDENCE,
    "docs/reports/kronos-evaluation-preflight/run-20260922-reviewed/report.json": "0de7d8192620fdfea7f52054a587aefa93eb6b5223bd3df2ec6f75f07fac93cf",
}
ROOT = Path(__file__).resolve().parents[2]
FROZEN = {
    "research/kronos_replay/engine.py": "d7c925c05299593a3cd9450df67e1eccb3c0c873e059b693a160bda8cc40984a",
    "research/kronos_replay/providers.py": "1862e3214632618dbef6fbcdbb2c0f45807db2bdbbeae5801f090b8b5eaa5f75",
    "_bmad-output/preregistration_kronos_readiness_20260922.md": "d09ca2cc3a686f4ff28565455afe1ab3238123098d5f016fee9ecc5ff63e4fd5",
}

from tools.kronos_inference_pilot import check_destination
from tools.trading_model_readiness import AuditError, safe_path
from . import FLAGS


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def documentary(path: Path) -> bytes:
    """Reject data paths and aliases before opening any documentary input."""
    absolute = path.absolute()
    resolved = safe_path(absolute)
    if (
        absolute != resolved
        or any(
            p.lower() in {"data", "logs", "models", ".git", ".venv", ".venv-research"}
            for p in absolute.parts
        )
        or path.suffix.lower() not in {".json", ".md", ".txt", ".html", ".pdf"}
    ):
        raise AuditError("only non-aliased documentary report/source paths permitted")
    return resolved.read_bytes()


def load_json(path: Path) -> dict[str, Any]:
    result = json.loads(documentary(path))
    if not isinstance(result, dict):
        raise AuditError("document must be a JSON object")
    return result


def sources(pack: dict[str, Any], base: Path) -> list[dict[str, Any]]:
    """Fingerprint validity is distinct from substantive evidence sufficiency."""
    result = []
    for entry in pack.get("sources", []):
        row = dict(entry)
        row["verification"] = "UNRESOLVED"
        try:
            if not all(
                isinstance(entry.get(k), str) and entry[k].strip()
                for k in ("path", "sha256", "date", "url", "claim")
            ):
                raise AuditError("source path/hash/date/url/claim required")
            datetime.fromisoformat(entry["date"].replace("Z", "+00:00"))
            path = Path(entry["path"])
            if not path.is_absolute():
                path = base / path
            if sha(documentary(path)) != entry["sha256"]:
                raise AuditError("source fingerprint mismatch")
            row["verification"] = "HASH_VERIFIED_ONLY"
        except (OSError, ValueError, TypeError):
            row["reason"] = "missing, invalid or changed documentary source"
        result.append(row)
    return result


def fixed_reports(root: Path) -> list[dict[str, Any]]:
    result = []
    for relative, expected in EVIDENCE.items():
        row: dict[str, Any] = {
            "path": relative,
            "sha256": expected,
            "verification": "UNRESOLVED",
        }
        try:
            payload = documentary(root / relative)
            if sha(payload) != expected:
                raise AuditError("documentary report fingerprint mismatch")
            if not isinstance(json.loads(payload), dict):
                raise AuditError("documentary report is not an object")
            row["verification"] = "HASH_VERIFIED_ONLY"
        except (OSError, ValueError):
            row["reason"] = "missing, invalid or changed fixed documentary report"
        result.append(row)
    return result


def new_output(output: Path) -> Path:
    for name, expected in FROZEN.items():
        if sha((ROOT / name).read_bytes()) != expected:
            raise AuditError("frozen mechanics or preregistration fingerprint changed")
    output = check_destination(output)
    output.mkdir(parents=True, exist_ok=False)
    return output


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        stream.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    path.chmod(0o444)


def finish(output: Path, report: dict[str, Any]) -> dict[str, Any]:
    report = {**report, **FLAGS}
    write_json(output / "report.json", report)
    with (output / "report.md").open("x") as stream:
        stream.write(
            "# Kronos readiness\n\n"
            + str(report["status"])
            + "\n\nStrategy testing permitted: false. Trading authorized: false.\n\n"
        )
        if report.get("reason"):
            stream.write(report["reason"] + "\n\n")
        if report.get("power_verdict"):
            stream.write(
                "Actual power: "
                + report["power_verdict"]
                + ". Admitted sessions: 0.\n\n"
            )
        for key, next_evidence in report.get("blockers", {}).items():
            stream.write(f"- **{key} — unresolved:** {next_evidence}\n")
        if report.get("blockers"):
            from .power import planning

            stream.write(
                "\nConditional normal planning only; per-test alpha .025, marginal power .90, joint lower bound .80.\n\n| Sessions | SE inflation | Detectable standardized effect |\n| ---: | ---: | ---: |\n"
            )
            for row in planning()["scenarios"]:
                stream.write(
                    f"| {row['sessions']} | {row['se_inflation']} | {row['detectable_standardized_effect']:.4f} |\n"
                )
        if "three_seed_decision_seconds" in report:
            stream.write(
                f"Startup seconds: {report['startup_seconds']}. Three-seed decision seconds: {report['three_seed_decision_seconds']}. No latency adopted.\n"
            )
        if "bar_requests" in report:
            stream.write(
                f"\nBar requests: {report['bar_requests']}. Current observations cannot authenticate history.\n"
            )
        stream.write(
            "\nFull evidence and descriptive measurements are in report.json; COMPLETE.json fingerprints all outputs.\n"
        )
    (output / "report.md").chmod(0o444)
    manifest = {
        str(p.relative_to(output)): sha(p.read_bytes())
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    code = {
        str(p.relative_to(ROOT)): sha(p.read_bytes())
        for p in sorted((ROOT / "research/kronos_readiness").glob("*.py"))
    }
    write_json(
        output / "COMPLETE.json",
        {**FLAGS, "sha256": manifest, "frozen_inputs": FROZEN, "code_sha256": code},
    )
    return report


BLOCKERS = {
    "historical_acquisition": "Original dated request/response records, identities and hashes binding each historical file to its acquisition.",
    "minute_convention": "Provider documentation and acquisition-linked evidence of open/end labels and timezone for historical minutes.",
    "completion_and_arrival": "Historical bar completion, first availability and revisions; current probe is descriptive only.",
    "calendar": "Dated exchange RTH sessions, holidays, early closes and DST with verified UTC boundaries for every evaluated session.",
    "contract_selection": "Causal per-session front-contract selection using contemporaneous volume, expiry and same-contract prior close.",
    "research_exposure": "Untouched population and auditable local research exposure plus model training-exposure limitations.",
    "costs": "Separate documented commission, exchange/regulatory costs, independently validated slippage and adopted latency protocol.",
    "economic_power": "Independent useful dollar effects for K and K-M, variances and covariance, eligible N and dependence justification.",
}


def assess(root: Path, pack_path: Path, output: Path) -> dict[str, Any]:
    from .power import planning
    from .protocol import candidate

    pack = load_json(pack_path)
    register = {
        **FLAGS,
        "fixed_reports": fixed_reports(root),
        "sources": sources(pack, pack_path.parent),
        "interpretation": "Hashes establish archived identity only, never historical admission.",
    }
    register["categories"] = {
        key: {
            "status": (
                "PARTIAL"
                if any(
                    r.get("category") == key
                    and r["verification"] == "HASH_VERIFIED_ONLY"
                    for r in register["sources"]
                )
                else "UNRESOLVED"
            ),
            "source_claims": [
                r for r in register["sources"] if r.get("category") == key
            ],
            "fixed_documentary_reports": [
                r
                for r in register["fixed_reports"]
                if r["verification"] == "HASH_VERIFIED_ONLY"
            ],
            "limits_and_next_evidence": value,
        }
        for key, value in BLOCKERS.items()
    }
    output = new_output(output)
    write_json(output / "evidence-register.json", register)
    write_json(output / "candidate-protocol.json", candidate())
    write_json(output / "power.json", planning())
    return finish(
        output,
        {
            "status": "HOLD_EVALUATION",
            "power_verdict": "UNASSESSABLE",
            "admitted_sessions": 0,
            "actual_eligible_sessions": None,
            "blockers": BLOCKERS,
            "evidence": register,
            "current_observations_authenticate_history": False,
        },
    )
