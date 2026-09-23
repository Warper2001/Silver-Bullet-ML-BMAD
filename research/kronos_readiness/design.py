"""Offline prospective planning. No collector, forecast or admission engine."""

from __future__ import annotations

import base64
import calendar as month_calendar
from datetime import date, datetime, timedelta, timezone
import json
import math
import os
import stat
from pathlib import Path
import re
from typing import Any
from zoneinfo import ZoneInfo

from . import FLAGS
from . import evidence
from .power import dollar_scenario, standardized
from .protocol import candidate
from tools.trading_model_readiness import AuditError

DESIGN_FROZEN = {
    "_bmad-output/preregistration_kronos_design_20260922.md": (
        "a60a7b528fc84e90d43162b1bbb325cf27fdfdbd74cfeed93657f59d30786cbb"
    ),
}

BLOCKERS = {
    "economics": (
        (
            "Independent useful K and K-M dollar effects, dated "
            "before sample-size selection; own-strategy variances, "
            "covariance and dependence evidence, not another "
            "strategy's estimates."
        )
    ),
    "calendar": (
        (
            "Reviewed CME MNQ RTH calendar with UTC/DST boundaries, "
            "holidays, early closes and coverage over the entire "
            "selected horizon; proxy weekdays are insufficient."
        )
    ),
    "operations": (
        (
            "Validated endpoint entitlements, "
            "completion/arrival/revision semantics, quote coverage, "
            "request budget, clock-error bounds, latency and "
            "outage/recovery coverage for proposed cadence."
        )
    ),
    "costs": (
        (
            "Independently supported commissions, fees, slippage "
            "and latency; freeze before evaluation without "
            "consulting concealed outcomes."
        )
    ),
    "population": (
        (
            "Separate preregistered calibration and untouched "
            "evaluation populations; power gate before performance "
            "calibration and another before evaluation; "
            "contamination/exposure audit."
        )
    ),
    "collection": (
        (
            "Separate collection and shadow-forecast admission, "
            "isolated auth, operational review and preregistration. "
            "This command does not implement admission assessment."
        )
    ),
    "contract": (
        (
            "Causal volume-derived front-contract decisions, "
            "same-contract history and resets; independently "
            "justify warmup and roll deductions."
        )
    ),
}
SCENARIO_KEYS = {
    "name",
    "roll_dates",
    "outage_dates",
    "incomplete_dates",
    "calibration_sessions",
}
MAX_DOCUMENT_BYTES = 8 * 1024 * 1024

ECONOMIC_KEYS = {
    "k_effect",
    "incremental_effect",
    "k_variance",
    "m_variance",
    "covariance",
}


def checked_path(path: Path) -> Path:
    absolute = path.absolute()
    forbidden = {
        "data",
        "market",
        "market_data",
        "market-data",
        "logs",
        "models",
        "cache",
        ".git",
        ".venv",
        ".venv-research",
        "sealed_holdout",
    }
    for part in absolute.parts:
        lower = part.lower()
        if (
            lower in forbidden
            or lower.startswith(".env")
            or any(s in lower for s in ("credential", "token", "secret"))
        ):
            raise AuditError("forbidden documentary path")
    if absolute != absolute.resolve():
        raise AuditError("documentary aliases prohibited")
    return absolute


def read_document(path: Path) -> bytes:
    """Bound explicit documentary reads; identity does not certify content."""
    path = checked_path(path)
    if path.suffix.lower() not in {".json", ".md", ".txt", ".html", ".pdf"}:
        raise AuditError("documentary extension required")

    def validate(info: os.stat_result) -> None:
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size > MAX_DOCUMENT_BYTES
        ):
            raise AuditError("single-link bounded regular document required")

    before = path.stat()
    validate(before)
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        validate(opened)
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise AuditError("document changed before opening")
        payload = stream.read(MAX_DOCUMENT_BYTES + 1)
        after = os.fstat(stream.fileno())
        validate(after)
        if len(payload) > MAX_DOCUMENT_BYTES or (
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise AuditError("oversized or changing document")
    return payload


def keys(obj: Any, required: set[str]) -> None:
    if not isinstance(obj, dict) or set(obj) != required:
        raise AuditError("unexpected or missing schema fields")


def iso_date(value: Any) -> date:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}", value
    ):
        raise AuditError("ISO date required")
    return date.fromisoformat(value)


def load_pack(path: Path) -> tuple[dict[str, Any], dict[str, bytes], bytes]:
    payload = read_document(path)

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AuditError("duplicate JSON key")
            result[key] = value
        return result

    pack = json.loads(
        payload,
        object_pairs_hook=unique,
        parse_constant=evidence.reject_constant,
    )
    if isinstance(pack, dict) and set(FLAGS) & set(pack):
        if any(pack.get(k) is not False for k in FLAGS):
            raise AuditError("both permission flags must be false")
        pack = {k: v for k, v in pack.items() if k not in FLAGS}
    keys(
        pack,
        {"schema_version", "sources", "calendar", "scenarios", "economic"},
    )
    if type(pack["schema_version"]) is not int or pack["schema_version"] != 1:
        raise AuditError("unsupported pack schema")
    if not isinstance(pack["sources"], list):
        raise AuditError("sources must be a list")
    snapshots = {}
    for source in pack["sources"]:
        keys(source, {"id", "path", "sha256", "date", "url", "claim"})
        if not all(isinstance(v, str) and v.strip() for v in source.values()):
            raise AuditError("source fields must be nonempty strings")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", source["id"]):
            raise AuditError("invalid source id")
        if source["id"] in snapshots:
            raise AuditError("duplicate source id")
        iso_date(source["date"])
        source_path = Path(source["path"])
        content = read_document(
            checked_path(
                source_path
                if source_path.is_absolute()
                else path.parent / source_path
            )
        )
        if evidence.sha(content) != source["sha256"]:
            raise AuditError("source fingerprint mismatch")
        snapshots[source["id"]] = content
    scenarios = pack["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise AuditError("nonempty scenarios required")
    names = set()
    for row in scenarios:
        keys(row, SCENARIO_KEYS)
        if (
            not isinstance(row["name"], str)
            or not row["name"].strip()
            or row["name"] in names
        ):
            raise AuditError("unique scenario names required")
        names.add(row["name"])
        if (
            type(row["calibration_sessions"]) is not int
            or row["calibration_sessions"] < 0
        ):
            raise AuditError("nonnegative calibration session count required")
        for key in ("roll_dates", "outage_dates", "incomplete_dates"):
            if not isinstance(row[key], list) or len(set(row[key])) != len(
                row[key]
            ):
                raise AuditError("unique scenario dates required")
            for value in row[key]:
                iso_date(value)
    cal = pack["calendar"]
    if cal is not None:
        keys(
            cal,
            {
                "coverage_start",
                "coverage_end",
                "source_ids",
                "closed_dates",
                "early_close_dates",
            },
        )
        first, last = iso_date(cal["coverage_start"]), iso_date(
            cal["coverage_end"]
        )
        if (
            first > last
            or not isinstance(cal["source_ids"], list)
            or not cal["source_ids"]
        ):
            raise AuditError("calendar coverage and provenance required")
        if any(s not in snapshots for s in cal["source_ids"]):
            raise AuditError("calendar source missing")
        for key in ("closed_dates", "early_close_dates"):
            if not isinstance(cal[key], list) or len(set(cal[key])) != len(
                cal[key]
            ):
                raise AuditError("unique calendar dates required")
            if any(not first <= iso_date(d) <= last for d in cal[key]):
                raise AuditError("calendar date outside coverage")
        if set(cal["closed_dates"]) & set(cal["early_close_dates"]):
            raise AuditError("conflicting calendar dates")
    econ = pack["economic"]
    if econ is not None:
        keys(econ, ECONOMIC_KEYS | {"se_inflation", "independent_evidence"})
        keys(econ["independent_evidence"], ECONOMIC_KEYS | {"dependence"})
        if any(
            v not in snapshots for v in econ["independent_evidence"].values()
        ):
            raise AuditError("economic source missing")
        conditional_dollars(1, econ)
    return pack, snapshots, payload


def conditional_dollars(n: int, economic: dict[str, Any]) -> dict[str, Any]:
    for key in ECONOMIC_KEYS | {"se_inflation"}:
        value = economic[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise AuditError("finite int/float economic inputs required")
    values = dict(economic)
    inflation = values.pop("se_inflation")
    return dollar_scenario(n, inflation, **values)


def add_months(start: date, months: int) -> date:
    year, month = divmod(start.year * 12 + start.month - 1 + months, 12)
    return date(
        year,
        month + 1,
        min(start.day, month_calendar.monthrange(year, month + 1)[1]),
    )


def planning_start(now: datetime, cal: dict[str, Any] | None) -> date:
    """Next full assumed RTH; supplied exclusions remain assertions."""
    local = now.astimezone(ZoneInfo("America/New_York"))
    day = local.date()
    if (local.hour, local.minute) >= (9, 30):
        day += timedelta(days=1)
    excluded = (
        set()
        if cal is None
        else set(cal["closed_dates"] + cal["early_close_dates"])
    )
    while day.weekday() >= 5 or day.isoformat() in excluded:
        day += timedelta(days=1)
    return day


def scenarios(pack: dict[str, Any], start: date) -> list[dict[str, Any]]:
    result = []
    for months in (3, 6, 12):
        end = add_months(start, months)
        days = [start + timedelta(days=i) for i in range((end - start).days)]
        weekdays = sum(d.weekday() < 5 for d in days)
        cal = pack["calendar"]
        covered = (
            cal is not None
            and iso_date(cal["coverage_start"]) <= start
            and iso_date(cal["coverage_end"]) >= end - timedelta(days=1)
        )
        for assumptions in pack["scenarios"]:
            context = 0
            warmup_cause = "initial"
            warmup_breakdown = dict.fromkeys(("initial", "contract", "gap"), 0)
            last_calibration = None
            first_evaluation = None
            collection_n = 0
            ledger = []
            deductions = dict.fromkeys(
                ("holiday", "outage", "incomplete", "warmup_or_horizon"), 0
            )
            pending_reset = False
            for day in days:
                stamp = day.isoformat()
                # Resets during closures apply before the next session.
                if stamp in assumptions["roll_dates"]:
                    pending_reset = True
                if day.weekday() >= 5:
                    continue
                reset = pending_reset
                if reset:
                    context = 0
                    warmup_cause = "contract"
                    pending_reset = False
                closed = bool(
                    cal
                    and iso_date(cal["coverage_start"])
                    <= day
                    <= iso_date(cal["coverage_end"])
                    and stamp in cal["closed_dates"]
                )
                early = bool(
                    cal
                    and iso_date(cal["coverage_start"])
                    <= day
                    <= iso_date(cal["coverage_end"])
                    and stamp in cal["early_close_dates"]
                )
                bars = 0 if closed else 14 if early else 26
                flags = []
                if closed:
                    flags.append("holiday")
                if stamp in assumptions["outage_dates"]:
                    flags.append("outage")
                if stamp in assumptions["incomplete_dates"]:
                    flags.append("incomplete")
                before = context
                earliest_decision_bar = max(1, 128 - context)
                eligible = not flags and earliest_decision_bar + 4 <= bars
                if not flags and not eligible:
                    flags.append("warmup_or_horizon")
                primary = flags[0] if flags else None
                role = "excluded"
                if primary:
                    deductions[primary] += 1
                    if primary == "warmup_or_horizon":
                        warmup_breakdown[warmup_cause] += 1
                else:
                    collection_n += 1
                    if collection_n <= assumptions["calibration_sessions"]:
                        role = "hypothetical_calibration"
                        last_calibration = stamp
                    else:
                        role = "hypothetical_evaluation"
                        if first_evaluation is None:
                            first_evaluation = stamp
                session_warmup_cause = warmup_cause
                # Missing bars cannot complete context. Restart warmup
                # after gaps and preserve excluded records.
                if "outage" in flags or "incomplete" in flags:
                    context = 0
                    warmup_cause = "gap"
                elif not closed:
                    context = min(128, context + bars)
                ledger.append(
                    {
                        "date": stamp,
                        "full_15m_bars": bars,
                        "early_close": early,
                        "contract_reset": reset,
                        "context_before": before,
                        "context_after": context,
                        "first_possible_decision_bar": earliest_decision_bar,
                        "conditional_collection_eligible": eligible,
                        "causes_retained": flags,
                        "primary_deduction": primary,
                        "warmup_cause": (
                            session_warmup_cause
                            if primary == "warmup_or_horizon"
                            else None
                        ),
                        "hypothetical_population_role": role,
                    }
                )
            allocation = min(collection_n, assumptions["calibration_sessions"])
            n = collection_n - allocation
            result.append(
                {
                    **FLAGS,
                    "months": months,
                    "start_inclusive": start.isoformat(),
                    "end_exclusive": end.isoformat(),
                    "scenario": assumptions["name"],
                    "calendar_days": len(days),
                    "weekday_ceiling": weekdays,
                    "calendar_basis": (
                        "SOURCE_ASSERTED_NOT_ADMITTED"
                        if covered
                        else "PARTIAL_OR_WEEKDAY_ASSUMPTION"
                    ),
                    "calendar_coverage_complete": covered,
                    "assumptions": assumptions,
                    "deductions": deductions,
                    "warmup_breakdown_subset_of_deductions": warmup_breakdown,
                    "last_hypothetical_calibration_date": last_calibration,
                    "first_hypothetical_evaluation_date": first_evaluation,
                    "calibration_freeze_requirement": (
                        "Calibration analysis and separate preregistration "
                        "freeze must complete before the first evaluation "
                        "cutoff. Allocation alone does not establish this; "
                        "otherwise defer evaluation and recalculate power."
                    ),
                    "session_ledger": ledger,
                    "conditional_collection_sessions": collection_n,
                    "calibration_allocation_sessions": allocation,
                    "conditional_evaluation_sessions": n,
                    "calibration_allocation_satisfied": allocation
                    == assumptions["calibration_sessions"],
                    "measured_eligible_sessions": None,
                    "admitted_sessions": 0,
                    "standardized_sensitivities": (
                        [standardized(n, i) for i in (1.0, 1.5, 2.0)]
                        if n
                        else []
                    ),
                    "dollar_power": (
                        conditional_dollars(n, pack["economic"])
                        if n and pack["economic"]
                        else None
                    ),
                }
            )
    return result


def causal_check(
    *,
    received: datetime,
    completed: datetime,
    cutoff: datetime,
    forecast_start: datetime,
    forecast_complete: datetime,
    fill_open: datetime,
    scheduled_flatten: datetime,
    clock_error_seconds: float = 0,
    complete: bool = True,
    exposed: bool = False,
    missing_flatten: bool = False,
    incomplete_session: bool = False,
    unresolved_exposure: bool = False,
) -> dict[str, Any]:
    """Uninterrupted synthetic minute schedule; no production eligibility.

    Every whole UTC minute is available until the explicit scheduled flatten.
    Missing actual opens require a different admitted schedule, never silently
    accepting a later fill in this synthetic invariant.
    """
    stamps = (
        received,
        completed,
        cutoff,
        forecast_start,
        forecast_complete,
        fill_open,
        scheduled_flatten,
    )
    if any(t.tzinfo is None or t.utcoffset() is None for t in stamps):
        raise AuditError("aware timestamps required")
    if not math.isfinite(clock_error_seconds) or clock_error_seconds < 0:
        raise AuditError("finite nonnegative clock bound required")
    (
        received,
        completed,
        cutoff,
        forecast_start,
        forecast_complete,
        fill_open,
        scheduled_flatten,
    ) = tuple(t.astimezone(timezone.utc) for t in stamps)
    error = timedelta(seconds=clock_error_seconds)
    reasons = []
    if completed + error > received - error:
        reasons.append("completion not known at receipt")
    if fill_open.second or fill_open.microsecond:
        reasons.append("fill is not a minute open")
    if (
        not complete
        or completed + error > cutoff - error
        or received + error > cutoff - error
    ):
        reasons.append("not completed and received by cutoff")
    if (
        forecast_start - error < cutoff + error
        or forecast_complete - error < forecast_start + error
    ):
        reasons.append("ambiguous forecast ordering")
    if (
        fill_open - error
        <= max(received, completed, forecast_complete) + error
    ):
        reasons.append("fill not strictly after availability")
    availability = max(received, completed, forecast_complete)
    strict_boundary = availability + 2 * error
    first_open = strict_boundary.replace(second=0, microsecond=0) + timedelta(
        minutes=1
    )
    if fill_open != first_open:
        reasons.append("not first eligible synthetic minute open")
    if (
        max(completed, forecast_complete) + error >= scheduled_flatten - error
        or fill_open + error >= scheduled_flatten - error
    ):
        reasons.append("completion or fill reaches scheduled flatten")
    if exposed:
        reasons.append("exposed population quarantined")
    if missing_flatten or incomplete_session or unresolved_exposure:
        reasons.append("incomplete session or unresolved position retained")
    return {
        **FLAGS,
        "synthetic_causal_order_valid": not reasons,
        "reasons": reasons,
        "record_retained": True,
    }


def protocol() -> dict[str, Any]:
    return {
        **FLAGS,
        "status": "PROPOSED_ONLY",
        "frozen_candidate": candidate(),
        "availability_precedence": (
            "Prospective availability uses actual input receipt/completion "
            "and actual forecast completion with conservative clock bounds. "
            "The nested candidate modeled-latency field is historical replay "
            "metadata only and cannot override this prospective rule."
        ),
        "cost_evidence": {
            key: {"status": "UNRESOLVED", "adopted_usd": None}
            for key in (
                "commission",
                "clearing",
                "exchange",
                "regulatory",
                "slippage",
                "operating_costs",
            )
        },
        "calibration_timeline": (
            "Zero reservation assumes separately powered "
            "independent calibration before the horizon; positive "
            "reservation is hypothetical allocation only, requiring "
            "its own preregistration and power gate."
        ),
        "populations": {
            "calibration": (
                (
                    "Separate preregistration and power gate before "
                    "performance calibration; label all exposure. No "
                    "calibration outcomes enter untouched evaluation."
                )
            ),
            "evaluation": (
                (
                    "Prospective contemporaneous forecasts only; freeze "
                    "dates, economic effects, model/rules/costs and "
                    "dependence plan before collection. No optional "
                    "stopping or favorable-session selection."
                )
            ),
        },
        "causality": [
            (
                (
                    "At each cutoff hash exact versions and only use "
                    "explicitly completed observations received by cutoff; "
                    "revisions are new observations for later decisions "
                    "only."
                )
            ),
            (
                (
                    "Record request start, response receipt, provider "
                    "event/completion times, input availability, forecast "
                    "actual start and completion with UTC and monotonic "
                    "clocks and uncertainty bounds."
                )
            ),
            (
                (
                    "Fill only at first minute open strictly after "
                    "conservative availability including actual forecast "
                    "completion. Clock overlap blocks causal eligibility."
                )
            ),
            (
                (
                    "Preserve scheduled final-minute flatten and contract "
                    "reset unchanged; delayed forecast cannot move flatten. "
                    "Missing flatten, outages and unresolved exposure "
                    "remain incomplete without invented prices."
                )
            ),
            (
                (
                    "Recovery retains original event time and new receipt "
                    "time; it cannot backdate availability or reconstruct a "
                    "missed contemporaneous forecast."
                )
            ),
        ],
        "proposed_operations": {
            "endpoints": [
                "GET /v3/marketdata/symbols/{explicit_contract}",
                "GET /v3/marketdata/barcharts/{explicit_contract}",
                "GET /v3/marketdata/quotes/{explicit_contract}",
            ],
            "cadence": (
                (
                    "Proposed bar polling every 5 seconds; one metadata "
                    "request per session/roll; quote polling each second "
                    "during RTH. Coverage and entitlements UNVALIDATED."
                )
            ),
            "retrieval_depth": (
                (
                    "Proposed last 3 one-minute bars; separate bounded "
                    "recovery up to 128 completed same-contract 15-minute "
                    "bars. Validate provider limits and ability to "
                    "reconstruct warmup without future knowledge."
                )
            ),
            "budget": (
                (
                    "390-minute full session ceiling: 4680 bar + 23400 "
                    "quote + 1 metadata requests = 28081, excluding bounded "
                    "recovery; preregister provider-approved recovery quota "
                    "before launch. Stop on 401/403/429; no catch-up bursts."
                )
            ),
            "recovery": (
                (
                    "Append-only raw observations and revisions; "
                    "sequence/checksum gaps and outages retained; backoff "
                    "within reviewed budget; require known-flat state for "
                    "reset, otherwise quarantine incomplete session."
                )
            ),
            "clock": (
                (
                    "Measure UTC-to-monotonic mapping and independent clock "
                    "error bound at startup and during session. Unknown "
                    "bound or discontinuity blocks causal eligibility; no "
                    "assumed zero production uncertainty."
                )
            ),
            "quote_fields": [
                "contract",
                "bid",
                "ask",
                "bid_size",
                "ask_size",
                "provider_event_time",
                "receipt_utc",
                "receipt_monotonic",
                "sequence",
                "request_id",
                "raw_payload_hash",
            ],
            "quote_checks": (
                (
                    "Validate identity, timestamps, spread/crossing, "
                    "nonnegative sizes, stale and out-of-order quotes, gaps "
                    "and receipt latency. Thresholds require independent "
                    "protocol evidence; no quote slippage admitted here."
                )
            ),
            "authentication": (
                (
                    "Future dedicated research auth/cache outside shared "
                    "trading state; never read or refresh shared "
                    "credentials. This command accepts no auth and performs "
                    "no network calls."
                )
            ),
        },
        "concealment": (
            (
                "Encrypted/access-separated forecast and outcome "
                "archive; operator integrity reports show only counts, "
                "hashes, receipt coverage, clocks and error categories, "
                "never forecasts, targets, positions or outcomes. Log "
                "every human/tool exposure and quarantine affected "
                "population before further evaluation."
            )
        ),
        "decision": (
            (
                "No horizon selected for launch. Compare 3/6/12 months "
                "conditionally; two one-sided tests at alpha .025, "
                "marginal power .90, joint lower bound .80. Independent "
                "economic effects must precede sample-size selection; "
                "separate future dependence-aware gate required."
            )
        ),
    }


def run(
    pack_path: Path,
    output: Path,
    start: str | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    # Verify frozen identities before reading a caller-supplied document.
    for name, digest in {**evidence.FROZEN, **DESIGN_FROZEN}.items():
        if evidence.sha((evidence.ROOT / name).read_bytes()) != digest:
            raise AuditError("frozen mechanics fingerprint changed")
    checked_path(output)
    if output.exists():
        raise AuditError("fresh destination required")
    pack, snapshots, pack_bytes = load_pack(pack_path)
    generated = now or datetime.now(timezone.utc)
    day = (
        iso_date(start)
        if start
        else planning_start(generated, pack["calendar"])
    )
    rows = scenarios(pack, day)
    output = evidence.new_output(output)
    register = {
        **FLAGS,
        "pack": pack,
        "pack_sha256": evidence.sha(pack_bytes),
        "pack_original_bytes_base64": base64.b64encode(pack_bytes).decode(
            "ascii"
        ),
        "design_frozen_inputs": DESIGN_FROZEN,
        "verification": "HASH_VERIFIED_ONLY_NOT_EVIDENCE_SUFFICIENCY",
    }
    evidence.write_json(output / "evidence.json", register)
    for source_id, payload in snapshots.items():
        evidence.write_json(
            output / f"source-{source_id}.json",
            {
                **FLAGS,
                "original_sha256": evidence.sha(payload),
                "original_bytes_base64": base64.b64encode(payload).decode(
                    "ascii"
                ),
            },
        )
    evidence.write_json(output / "scenarios.json", {**FLAGS, "horizons": rows})
    evidence.write_json(output / "protocol.json", protocol())
    with (output / "comparison.md").open("x") as stream:
        stream.write(
            (
                (
                    "# Kronos prospective evaluation "
                    "design\n\nstrategy_test_permitted=false; "
                    "trading_authorized=false\n\nPARK_PENDING_EVIDENCE. All "
                    "session counts are conditional, never measured or "
                    "admitted. Dated calendar is an assertion, not "
                    "admission. Warmup uses 128 completed same-contract "
                    "15-minute bars and leaves four horizon bars; gaps "
                    "reset context conservatively. Calendar sessions have "
                    "26 bars, early closes 14. Deductions are disjoint with "
                    "all causes retained. Calibration allocation is "
                    "hypothetical, not powered calibration; zero allocation "
                    "requires calibration separately sourced before this "
                    "horizon.\n\n|Months|Scenario|Weekday ceiling|Conditional "
                    "evaluation N|Detectable effect (SE inflation 1 / 1.5 / "
                    "2)|\n|---:|---|---:|---:|---|\n"
                )
            )
        )
        for row in rows:
            effects = (
                " / ".join(
                    f"{r['detectable_standardized_effect']:.4f}"
                    for r in row["standardized_sensitivities"]
                )
                or "no sessions"
            )
            stream.write(
                f"|{row['months']}|{row['scenario']}|"
                f"{row['weekday_ceiling']}|"
                f"{row['conditional_evaluation_sessions']}|{effects}|\n"
            )
        stream.write(
            (
                (
                    "\nSee scenarios.json for separate deductions, "
                    "assumptions and calendar coverage; protocol.json for "
                    "proposed recovery, concealment and operational "
                    "requirements. No horizon is sufficient on evidence "
                    "currently assessed.\n"
                )
            )
        )
    return evidence.finish(
        output,
        {
            "status": "PARK_PENDING_EVIDENCE",
            "reason": (
                (
                    "Documentary planning only. No collection, forecasting, "
                    "scoring or trading admitted; no automated "
                    "evidence-sufficiency assessment is implemented.\n\n"
                    "**This run: [3/6/12-month horizon comparison]"
                    "(comparison.md), with full accounting in "
                    "[scenarios.json](scenarios.json).** The inherited "
                    "sessions/SE-inflation table below is GENERIC readiness "
                    "planning only; it is not this run's horizon analysis."
                )
            ),
            "generated_at": generated.isoformat(),
            "planning_start": day.isoformat(),
            "next_full_assumed_rth": planning_start(
                generated, pack["calendar"]
            ).isoformat(),
            "start_basis": (
                "EXPLICIT_HYPOTHETICAL"
                if start
                else "NEXT_FULL_RTH_ASSUMED_CALENDAR_PENDING_VERIFICATION"
            ),
            "power_verdict": "UNASSESSABLE",
            "admitted_sessions": 0,
            "blockers": BLOCKERS,
            "horizons": [
                {k: v for k, v in row.items() if k != "session_ledger"}
                for row in rows
            ],
            "session_ledger_artifact": "scenarios.json",
        },
    )
