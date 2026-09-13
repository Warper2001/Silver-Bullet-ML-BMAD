"""Metadata-only commodity carry feasibility packet."""

from __future__ import annotations

import json

import pandas as pd

from .artifacts import BASE, PINNED_INPUTS, input_path

ROOTS = ("CL", "NG", "RB", "HO", "HG", "ZC", "ZW", "ZS", "ZM", "ZL", "LE", "HE")
VENUES = ("Topstep", "TradeStation_SIM", "future_self_funded")


def build_carry() -> dict[str, object]:
    contracts = pd.read_csv(
        input_path("data/commodity_curve/coverage-pilot-2025-20260906-v3/contracts.csv")
    )
    observed_roots = set(contracts.canonical_root.astype(str))
    if observed_roots != set(ROOTS):
        raise ValueError("Carry evidence does not contain the frozen 12-root universe")
    units = json.loads(
        input_path(
            "data/commodity_curve/unit-audit-2025-20260907/exchange-unit-facts.json"
        ).read_text()
    )
    if {row["root"] for row in units} != set(ROOTS):
        raise ValueError("Official unit evidence is incomplete")
    calendar = json.loads(
        input_path(
            "data/commodity_curve/calendar-audit-2025-20260907/source-facts.json"
        ).read_text()
    )
    official_venues = json.loads((BASE / "official_venue_evidence.json").read_text())
    venue_providers = {row["provider"] for row in official_venues["sources"]}
    if venue_providers != {"Topstep", "TradeStation"}:
        raise ValueError("Bound official venue evidence is incomplete")
    topstep_facts = [
        fact
        for row in official_venues["sources"]
        if row["provider"] == "Topstep"
        for fact in row["facts"]
    ]
    tradestation_facts = [
        fact
        for row in official_venues["sources"]
        if row["provider"] == "TradeStation"
        for fact in row["facts"]
    ]
    if not any("cannot be held" in fact for fact in topstep_facts):
        raise ValueError("Topstep overnight restriction is not source-bound")
    if not any("overnight margin" in fact for fact in tradestation_facts):
        raise ValueError("TradeStation overnight margin evidence is missing")
    development = json.loads(
        input_path(
            "data/commodity_curve/development-jan-aug-2025-monthly-v2/report.json"
        ).read_text()
    )
    if (
        set(development["roots"]) != set(ROOTS)
        or development["rows"] != 95537
        or development["official_publication_rows"] != 0
        or development["research_status"] != "HOLD-DATA"
    ):
        raise ValueError("Bound development facts do not support carry classifications")

    rows = []
    for root in ROOTS:
        for venue in VENUES:
            if venue == "Topstep":
                classification = "UNAVAILABLE_OVERNIGHT"
                account_gap = "combine path requires intraday liquidation; monthly overnight outright book is unavailable"
                product, overnight = "usable", "unavailable"
            elif venue == "TradeStation_SIM":
                classification = "VALIDATION_ONLY"
                account_gap = "SIM can validate mechanics but supplies no funded capital, binding overnight budget, or production fee/margin seal"
                product = overnight = "requires verification/acquisition"
            else:
                classification = "REQUIREMENTS_ONLY"
                account_gap = "capital, permissions, margin, fees, integer sizing, limits and overnight drawdown budget are unspecified"
                product = overnight = "requires verification/acquisition"
            rows.append(
                {
                    "root": root,
                    "venue": venue,
                    "path_classification": classification,
                    "account_evidence": account_gap,
                    "product_permission_status": product,
                    "overnight_status": overnight,
                    "integer_sizing_status": "requires verification/acquisition",
                    "margin_status": "requires verification/acquisition",
                    "cost_quote_status": "requires verification/acquisition",
                    "observed_staging_status": "usable",
                    "publication_lineage_status": "requires verification/acquisition",
                    "calendar_status": "requires verification/acquisition",
                    "history_power_status": "requires verification/acquisition",
                    "data_evidence": "2025 staging exists; original-publication lineage, effective calendars, inactive-contract coverage and executable quotes/costs remain unresolved",
                    "power_evidence": "eight development month ends; promotion protocol requires at least five years of untouched reserved history",
                    "park_reasons": "PARK_ACCOUNT|PARK_DATA|PARK_POWER",
                    "returns_calculated": False,
                    "questions_sent": False,
                }
            )
    matrix = pd.DataFrame(rows)

    evidence = []
    for row in units:
        evidence.append(
            {
                "category": "official_exchange_unit_rule",
                "root_scope": row["root"],
                "source": row["source_url"],
                "access_date_utc": row["access_date_utc"],
                "status": "OBSERVED_WITHOUT_HISTORICAL_EFFECTIVE_DATE",
                "local_binding": "data/commodity_curve/unit-audit-2025-20260907/exchange-unit-facts.json",
            }
        )
    for row in calendar["observations"]:
        roots = row.get("roots", [row.get("root", "ALL")])
        evidence.append(
            {
                "category": "official_exchange_calendar_or_rule",
                "root_scope": ",".join(roots),
                "source": row["source_url"],
                "access_date_utc": calendar["access_date_utc"],
                "status": "INCOMPLETE_EFFECTIVE_DATE_OR_PUBLICATION_LINEAGE",
                "local_binding": "data/commodity_curve/calendar-audit-2025-20260907/source-facts.json",
            }
        )
    evidence.extend(
        [
            {
                "category": "vendor_contract_snapshot",
                "root_scope": "ALL",
                "source": "TradeStation API response normalized in existing pilot",
                "access_date_utc": "2026-09-06",
                "status": "IDENTITY_INDICATIVE_NOT_EXECUTABILITY",
                "local_binding": "data/commodity_curve/coverage-pilot-2025-20260906-v3/contracts.csv",
            },
            {
                "category": "vendor_publication_lineage",
                "root_scope": "ALL",
                "source": "Databento GLBX.MDP3 local evidence and prepared clarification",
                "access_date_utc": "2026-09-07",
                "status": "SOURCE_CLARIFICATION_REQUIRED_UNSENT",
                "local_binding": "docs/commodity-curve-databento-fields-inquiry-20260907.md",
            },
        ]
    )
    for source in official_venues["sources"]:
        evidence.append(
            {
                "category": "official_venue_rule",
                "root_scope": "ALL",
                "source": source["url"],
                "access_date_utc": official_venues["observed_at_utc"],
                "status": "OBSERVED_CURRENT_PUBLIC_RULE_REQUIRES_ACCOUNT_CONFIRMATION",
                "local_binding": "research/pf_improvement/official_venue_evidence.json",
            }
        )
    verdict = {
        "overall": [
            "PARK_ACCOUNT",
            "PARK_DATA",
            "PARK_POWER",
            "SOURCE_CLARIFICATION_REQUIRED",
        ],
        "matrix_rows": 36,
        "roots": 12,
        "venues": 3,
        "questions_prepared_not_sent": True,
        "purchase_authorized": False,
        "strategy_returns_calculated": False,
        "preserved_prior_verdicts": {
            "TSC-1": "FAIL",
            "TSMOM-1": "FAIL",
            "COT": "FAIL",
            "XSMOM-1": "UNDERPOWERED",
            "VRP-1": "UNDERPOWERED",
        },
        "bound_evidence_files": len([p for p in PINNED_INPUTS if "commodity" in p]),
    }
    questions = """# Prepared source and account questions — not sent

No vendor contact, subscription, purchase, service change, or data acquisition is authorized by this packet.

## Data source

1. Identify an existing product that supplies historical first-notice, first-intent, last-trade, delivery and versioned session-calendar fields for the frozen 12-root universe.
2. Explain whether reconstructed GLBX.MDP3 send times preserve original CME SendingTime and how initial publications, corrections, recovery messages and restatements are distinguished.
3. Identify coverage for listed contracts whose settlements are absent from MDP, with publication/revision history, entitlements, license limits and a non-binding scoped quote.

## Account paths

1. For each venue, identify allowed roots, overnight holding rules, current initial/maintenance margin, all fees, product permissions and liquidation constraints.
2. For any self-funded path, state capital, gross and sector limits, maximum overnight loss/drawdown budget, currency treatment and integer-contract policy.
3. Treat TradeStation SIM as mechanics validation only unless a separately identified funded account supplies the complete effective-dated account manifest.

## Power and seal

Document at least five years of untouched reserved history, prior access, exact split dates and a carry-specific immutable seal before any strategy-return calculation. Eight development month ends remain `PARK_POWER`.
"""
    return {
        "matrix": matrix,
        "evidence": pd.DataFrame(evidence),
        "verdict": verdict,
        "questions": questions,
    }
