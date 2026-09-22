"""Outcome-blind Kronos evaluation preflight; never reads prices or runs a strategy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import NormalDist
from typing import Any, Sequence
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.trading_model_readiness import AuditError, digest, safe_path  # noqa: E402

AUDIT = "docs/reports/trading-model-feasibility/run-20260922-reviewed/report.json"
PILOT = "docs/reports/kronos-inference-pilot/run-20260922-verified/report.json"
EVIDENCE = {
    AUDIT: "ec6a2a65db2ec6c95ec6b5fe3bfb39d12bf89c6a8b9160b865193612d0463ce2",
    PILOT: "97fd8cb6d7926beffef933889194b3dff16fbe56b02c855efbef0dddb3f4c7cc",
}
REVISIONS = {
    "model_revision": {
        "repo": "NeoQuasar/Kronos-small",
        "sha": "901c26c1332695a2a8f243eb2f37243a37bea320",
        "date": "2025-09-09T14:10:26+00:00",
    },
    "tokenizer_revision": {
        "repo": "NeoQuasar/Kronos-Tokenizer-base",
        "sha": "0e0117387f39004a9016484a186a908917e22426",
        "date": "2025-09-09T14:10:02+00:00",
    },
}


def read_evidence(relative: str) -> dict[str, Any]:
    """Only the fixed immutable documentary inputs are permitted."""
    if relative not in EVIDENCE:
        raise AuditError("not an approved documentary input")
    payload = safe_path(ROOT / relative).read_bytes()
    if hashlib.sha256(payload).hexdigest() != EVIDENCE[relative]:
        raise AuditError("documentary evidence fingerprint changed")
    document = json.loads(payload)
    if not isinstance(document, dict):
        raise AuditError("expected documentary JSON object")
    return document


def weekday_ceiling(start: date, end: date) -> int:
    """Inclusive weekday count, not an exchange calendar or eligible sample."""
    if start > end:
        return 0
    weeks, remainder = divmod((end - start).days + 1, 7)
    return weeks * 5 + sum((start.weekday() + d) % 7 < 5 for d in range(remainder))


def normal_scenario(
    days: int,
    annual_sharpe: float,
    se_inflation: float = 1.0,
    comparisons: int = 1,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> dict[str, Any]:
    """Hypothetical one-sided known-variance mean test, not estimated strategy power.

    Daily effect = annual_sharpe / sqrt(252). Standard error inflation multiplies
    sigma/sqrt(n). Bonferroni allocates alpha across the hypothetical family.
    Actual heavy tails, dependence and unknown variance require a separate gate.
    """
    if type(days) is not int or days < 0:
        raise AuditError("days must be a nonnegative integer")
    if type(comparisons) is not int or comparisons < 1:
        raise AuditError("comparisons must be a positive integer")
    if not all(
        math.isfinite(x) for x in (annual_sharpe, se_inflation, alpha, target_power)
    ):
        raise AuditError("scenario parameters must be finite")
    if annual_sharpe <= 0 or se_inflation < 1 or not 0 < alpha < 0.5:
        raise AuditError("invalid effect, inflation or alpha")
    if not 0.5 < target_power < 1:
        raise AuditError("target power must be between one half and one")
    normal = NormalDist()
    tail = alpha / comparisons
    if not 0 < tail < 1 or 1 - tail == 1:
        raise AuditError("comparison allocation exceeds numerical precision")
    critical = normal.inv_cdf(1 - tail)
    z_power = normal.inv_cdf(target_power)
    required = 252 * ((critical + z_power) * se_inflation / annual_sharpe) ** 2
    return {
        "hypothetical_annual_net_sharpe": annual_sharpe,
        "se_inflation": se_inflation,
        "comparisons": comparisons,
        "family_alpha": alpha,
        "target_power": target_power,
        "required_daily_observations": math.ceil(required),
        "required_252_day_years": required / 252,
        "assumed_daily_observations": days,
        "hypothetical_power": (
            normal.cdf(annual_sharpe * math.sqrt(days / 252) / se_inflation - critical)
            if days
            else None
        ),
    }


def build_report(audit: dict[str, Any], pilot: dict[str, Any]) -> dict[str, Any]:
    for key, record in REVISIONS.items():
        if pilot[key] != record["sha"]:
            raise AuditError("pilot checkpoint differs from documented revision")
    matches = [d for d in audit["datasets"] if d.get("sha256") == pilot["input_sha256"]]
    if len(matches) != 1:
        raise AuditError("pilot input does not uniquely match audited evidence")
    data = matches[0]
    ny = ZoneInfo("America/New_York")
    boundary = max(datetime.fromisoformat(r["date"]) for r in REVISIONS.values())
    start = max(
        boundary.astimezone(ny).date() + timedelta(days=1),
        datetime.fromisoformat(data["first_valid_timestamp"]).astimezone(ny).date(),
    )
    last = datetime.fromisoformat(data["last_valid_timestamp"]).astimezone(ny)
    # Only ordinary full RTH sessions are counted in this deliberately generous ceiling.
    end = last.date() - timedelta(days=int((last.hour, last.minute) < (16, 0)))
    ceiling = min(
        weekday_ceiling(start, end),
        data["grid_hypotheses"]["end"]["dates_with_any_full_regular_grid"],
    )
    return {
        "status": "HOLD_EVALUATION",
        "power_verdict": "UNASSESSABLE",
        "strategy_test_permitted": False,
        "training_permitted": False,
        "admitted_untouched_sessions": 0,
        "actual_eligible_sessions": None,
        "actual_strategy_power": None,
        "evidence_hashes": EVIDENCE,
        "checkpoint_revision_metadata": REVISIONS,
        "data_sha256": data["sha256"],
        "candidate_start_date": start.isoformat(),
        "candidate_end_date": end.isoformat(),
        "post_revision_weekday_ceiling": ceiling,
        "data_gate_from_audit": audit["data_gate"],
        "data_gaps_from_audit": audit["data_gaps"],
        "missing_economic_inputs": [
            "ADMITTED_UNTOUCHED_EVALUATION_POPULATION",
            "FROZEN_FORECAST_TO_POSITION_POLICY_AND_EXECUTION_COSTS",
            "JUSTIFIED_TARGET_EFFECT",
            "REGISTERED_DEPENDENCE_AND_MULTIPLICITY_MODEL",
        ],
        "planning_scenarios": [
            normal_scenario(ceiling, sr, inflation, family)
            for sr in (0.5, 1.0, 1.5, 2.0)
            for inflation in (1.0, 1.5)
            for family in (1, 3)
        ],
        "qualifications": [
            "Planning scenarios are unsealed assumptions, not adopted thresholds or Kronos estimates.",
            "252 days/year is a scaling convention, not a historical exchange calendar.",
            "Post-revision weekday ceiling includes holidays and unknown gaps; not effective N.",
            "Revision dates come from public commit metadata, not verified training cutoffs.",
            "Later dates reduce temporal pretraining concerns but do not erase local research exposure.",
            "Zero admitted sessions means none certified here, not proof that none could qualify.",
            "Three seeds and overlapping forecast bars do not multiply market evidence.",
            "Normal known-variance calculations do not replace a dependence-aware strategy power gate.",
            "No prices, forecasts, returns, PnL, credentials or sealed data were read.",
        ],
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# Kronos evaluation preflight",
        "",
        "**HOLD_EVALUATION — power UNASSESSABLE. No strategy test permitted.**",
        "",
        f"Candidate post-revision interval: {report['candidate_start_date']} through "
        f"{report['candidate_end_date']}; at most {report['post_revision_weekday_ceiling']} "
        "weekdays before exchange-calendar, data and research-exposure exclusions.",
        "",
        "Admitted untouched sessions: 0; actual eligible count and strategy power: unknown.",
        "",
        "## Illustrative planning, not a measured edge",
        "",
        "One-sided normal known-variance test; illustrative family alpha 5%, target power 80%, "
        "one comparison, no SE inflation. Net daily-return Sharpe annualized with sqrt(252). "
        "These are sensitivity assumptions, not adopted admission thresholds.",
        "",
        "| Assumed annual net Sharpe | Required days | 252-day years | Power at weekday ceiling |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in report["planning_scenarios"]:
        if row["se_inflation"] == 1 and row["comparisons"] == 1:
            power = row["hypothetical_power"]
            power_text = "unavailable" if power is None else f"{power:.1%}"
            lines.append(
                f"| {row['hypothetical_annual_net_sharpe']:.1f} | "
                f"{row['required_daily_observations']} | "
                f"{row['required_252_day_years']:.2f} | {power_text} |"
            )
    lines += [
        "",
        "Full sensitivity (SE inflation and comparison families) is in report.json.",
        "",
        "## Qualifications",
        "",
    ]
    lines += [f"- {item}" for item in report["qualifications"]]
    return "\n".join(lines) + "\n"


def run(output: Path) -> dict[str, Any]:
    output = safe_path(output)
    allowed = ROOT.resolve() / "docs/reports/kronos-evaluation-preflight"
    if output == allowed or not output.is_relative_to(allowed) or output.exists():
        raise AuditError(
            "use a fresh child directory of docs/reports/kronos-evaluation-preflight"
        )
    report = build_report(read_evidence(AUDIT), read_evidence(PILOT))
    report["script_sha256"] = digest(Path(__file__))
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (output / "report.md").write_text(render(report))
    (output / "COMPLETE.json").write_text(
        json.dumps(
            {name: digest(output / name) for name in ("report.json", "report.md")},
            indent=2,
        )
        + "\n"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.output_dir)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "power_verdict",
                    "strategy_test_permitted",
                    "post_revision_weekday_ceiling",
                )
            }
        )
    )
    # Unlike the descriptive readiness audit, the preflight is explicitly a gate.
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
