"""Offline blocked-report identity checks; never economic admission."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

PACKET = Path(__file__).resolve().parent
REPO = PACKET.parents[2]

EXPECTED_INPUTS = frozenset(
    (
        "AGENTS.md",
        "docs/kronos-prospective-collection.md",
        "docs/kronos-replay.md",
        "docs/reports/kronos-calibration-entry/README.md",
        "docs/reports/kronos-calibration-entry/decision.json",
        "docs/reports/kronos-calibration-entry/economics.md",
        (
            "docs/reports/kronos-calibration-entry/inputs/"
            "08-11-candidate-protocol.json"
        ),
        "docs/reports/kronos-calibration-entry/routes.md",
        "docs/reports/kronos-design/run-20260923-reviewed/report.md",
        "docs/reports/kronos-design/sources-20260922/bounded-inventory.md",
        "docs/reports/kronos-evaluation-preflight/README.md",
        (
            "docs/reports/kronos-evaluation-preflight/"
            "run-20260922-reviewed/report.json"
        ),
        "docs/reports/kronos-readiness/run-20260922-reviewed/report.md",
        "research/kronos_readiness/design.py",
        "research/kronos_readiness/power.py",
        "research/kronos_replay/__main__.py",
        "research/kronos_replay/engine.py",
        "research/kronos_replay/providers.py",
        "research/mim_comparison/evidence/reference-audit.md",
        "research/mim_comparison/evidence/source-manifest.json",
    )
)
EXPECTED_CONSTRAINTS = {
    "active_hours_ceiling": 8,
    "capital_usd": 30000,
    "loss_budget_is_strategy_stop": False,
    "loss_from_start_assessment_usd": 5000,
    "market_data_downloads_allowed": False,
    "objective": (
        "Any positive net profit and improvement " "on four-bar momentum"
    ),
    "purchase_budget_usd": 0,
    "service_changes_allowed": False,
    "trading_allowed": False,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_local(base: Path, name: str) -> bytes:
    relative = Path(name)
    require(
        not relative.is_absolute()
        and bool(relative.parts)
        and ".." not in relative.parts,
        f"invalid relative path: {name}",
    )
    current = base
    for part in relative.parts:
        current = current / part
        require(not current.is_symlink(), f"symlink refused: {name}")
    return current.read_bytes()


def verify(packet: Path, repo: Path) -> int:
    manifest = json.loads(read_local(packet, "COMPLETE.json"))
    require(
        manifest["kind"] == "BLOCKED_DOCUMENTARY_PACKET", "wrong packet kind"
    )
    for flag in ("strategy_test_permitted", "trading_authorized"):
        require(manifest[flag] is False, f"manifest permission drift: {flag}")
    actual = {
        str(p.relative_to(packet))
        for p in packet.rglob("*")
        if p.is_file() and p != packet / "COMPLETE.json"
    }
    require(actual == set(manifest["sha256"]), "packet membership mismatch")
    for name, expected in manifest["sha256"].items():
        require(
            hashlib.sha256(read_local(packet, name)).hexdigest() == expected,
            f"packet hash mismatch: {name}",
        )
    decision = json.loads(read_local(packet, "decision.json"))
    require(decision["decision"] == "BACKTEST_BLOCKED", "decision drift")
    require(
        decision["recommendation"] == "PARK_PENDING_EVIDENCE",
        "recommendation drift",
    )
    for flag in (
        "strategy_test_permitted",
        "trading_authorized",
        "collection_authorized",
        "historical_inference_run",
        "historical_scoring_run",
    ):
        require(decision[flag] is False, f"decision permission drift: {flag}")
    require(decision["actual_power"] == "UNASSESSABLE", "power drift")
    require(decision["admitted_sessions"] == 0, "admission drift")
    require(
        decision["eligible_sessions"] is None, "unknown eligible count drift"
    )
    require(
        decision["measured_results"] is None, "unexpected performance results"
    )
    require(
        set(decision["requirements"])
        == {
            "historical_population",
            "execution_assumptions",
            "statistical_admission",
        }
        and all(
            x["status"] == "NOT_MET" for x in decision["requirements"].values()
        ),
        "requirement drift",
    )
    require(
        decision["operator_constraints"] == EXPECTED_CONSTRAINTS,
        "operator constraint drift",
    )
    design = decision["statistical_design"]
    require(design["one_sided_alpha_each"] == 0.025, "alpha drift")
    require(design["marginal_power_each"] == 0.90, "power target drift")
    require(design["planning_effects"] is None, "unsupported effect")
    require(
        design["joint_nuisance_evidence"] is None,
        "unsupported nuisance inputs",
    )
    register = json.loads(read_local(packet, "input-register.json"))
    require(
        set(register["sha256"]) == EXPECTED_INPUTS,
        "dependency membership mismatch",
    )
    for name, expected in register["sha256"].items():
        # Metadata only. Never follow data paths named inside documents.
        require(
            name == "AGENTS.md"
            or (
                name.startswith(
                    (
                        "docs/",
                        "research/kronos_",
                        "research/mim_comparison/evidence/",
                    )
                )
                and Path(name).suffix in {".md", ".json", ".py"}
            ),
            f"non-documentary input refused: {name}",
        )
        require(
            hashlib.sha256(read_local(repo, name)).hexdigest() == expected,
            f"input hash mismatch: {name}",
        )
    return len(actual)


def refusal_checks() -> None:
    """Check corruption, missing files and permissions on temporary copies."""
    with tempfile.TemporaryDirectory(
        prefix="kronos-blocked-verify-"
    ) as directory:
        target = Path(directory) / "packet"
        shutil.copytree(
            PACKET, target, ignore=shutil.ignore_patterns("__pycache__")
        )
        decision_path = target / "decision.json"
        original = decision_path.read_bytes()
        for mutation in (original + b" ", None):
            if mutation is None:
                decision_path.unlink()
            else:
                decision_path.write_bytes(mutation)
            try:
                verify(target, REPO)
            except (ValueError, FileNotFoundError):
                pass
            else:
                raise ValueError("corrupt/missing packet accepted")
            decision_path.write_bytes(original)
        decision = json.loads(original)
        decision["strategy_test_permitted"] = True
        decision_path.write_text(json.dumps(decision))
        manifest_path = target / "COMPLETE.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["sha256"]["decision.json"] = hashlib.sha256(
            decision_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        try:
            verify(target, REPO)
        except ValueError as error:
            require("permission drift" in str(error), "wrong refusal reason")
        else:
            raise ValueError("permission change accepted after rehash")

        for field, value, reason in (
            ("operator_constraints", {}, "operator constraint drift"),
        ):
            decision = json.loads(original)
            decision[field] = value
            decision_path.write_text(json.dumps(decision))
            manifest["sha256"]["decision.json"] = hashlib.sha256(
                decision_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            try:
                verify(target, REPO)
            except ValueError as error:
                require(reason in str(error), "wrong constraint refusal")
            else:
                raise ValueError("constraint mutation accepted")
        shutil.rmtree(target)
        shutil.copytree(PACKET, target)
        register_path = target / "input-register.json"
        register_path.write_text('{"sha256": {}}')
        manifest = json.loads((target / "COMPLETE.json").read_text())
        manifest["sha256"]["input-register.json"] = hashlib.sha256(
            register_path.read_bytes()
        ).hexdigest()
        (target / "COMPLETE.json").write_text(json.dumps(manifest))
        try:
            verify(target, REPO)
        except ValueError as error:
            require(
                "dependency membership" in str(error), "wrong register refusal"
            )
        else:
            raise ValueError("empty dependency register accepted")
        shutil.rmtree(target)
        shutil.copytree(PACKET, target)
        (target / "nested").mkdir()
        (target / "nested/COMPLETE.json").write_text("{}")
        try:
            verify(target, REPO)
        except ValueError as error:
            require("membership" in str(error), "wrong membership refusal")
        else:
            raise ValueError("nested manifest accepted")


if __name__ == "__main__":
    count = verify(PACKET, REPO)
    refusal_checks()
    print(
        f"PASS: {count} packet hashes, documentary/code identities "
        "and refusal checks; BACKTEST BLOCKED"
    )
