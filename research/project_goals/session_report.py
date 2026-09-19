"""Bounded API-export accounting, never invented account reset provenance."""

from collections import defaultdict
from datetime import timedelta
import json
import re
from zoneinfo import ZoneInfo
from .common import input_path, inventory, timestamp, known_epoch
from .portfolio import marked_curves, portfolio_report

ASSUMPTIONS = [
    "Broker account/date trade export completeness requires independent confirmation.",
    "Each session starts flat only as an explicit conditional accounting assumption, not account reset proof.",
    "Exact broker contract IDs identify marks; API OHLC exports remain conditional source evidence.",
]
ET = ZoneInfo("America/New_York")


def session_report(fills, mark_paths, calendar=None, operating_cost_monthly=0.0):
    mark_paths = list(mark_paths)
    inputs = mark_paths + ([calendar] if calendar else [])
    inventory_before = inventory(inputs)
    if not fills:
        return dict(status="INSUFFICIENT_DATA", reason="no attributable fills"), []
    identities = {
        (str(f.get("account")), str(f.get("epoch")))
        for f in fills
        if known_epoch(f.get("epoch"))
    }
    if len(identities) > 1:
        return (
            dict(
                status="INSUFFICIENT_DATA",
                reason="split multiple actual account epochs into separate reports",
            ),
            [],
        )
    unknown_epoch = any(not known_epoch(f.get("epoch")) for f in fills)
    byday = defaultdict(list)
    for fill in fills:
        day = str(timestamp(fill["timestamp"]).astimezone(ET).date())
        byday[day].append(fill)
    # A file may span dates; several files may describe different contracts on one date.
    # Keep conflicting duplicate rows: marked_curves must reject them, not overwrite them.
    daily_marks = defaultdict(list)
    for path in mark_paths:
        envelope = json.loads(input_path(path).read_text())
        if not envelope["response"]["success"]:
            raise ValueError("unsuccessful bar export")
        request = envelope["request"]
        if (
            request["unit"] != 2
            or request["unitNumber"] != 1
            or request["includePartialBar"]
        ):
            raise ValueError("completed one-minute bars required")
        contract = request["contractId"]
        if not re.fullmatch(r"CON\.F\.US\.MNQ\.[HMUZ]\d{2}", contract):
            raise ValueError(
                "unsupported contract identity; validated MNQ contracts only"
            )
        for bar in envelope["response"]["bars"]:
            close_time = timestamp(bar["t"]) + timedelta(minutes=1)
            daily_marks[str(close_time.astimezone(ET).date())].append(
                dict(
                    timestamp=close_time.isoformat(),
                    contract=contract,
                    close=bar["c"],
                    multiplier=2.0,
                    corrected_contract=True,
                )
            )
    if set(byday) - set(daily_marks):
        return (
            dict(
                status="INSUFFICIENT_DATA",
                missing_mark_sessions=sorted(set(byday) - set(daily_marks)),
            ),
            [],
        )
    observed = {}
    if calendar:
        record = json.loads(input_path(calendar).read_text())
        from .scheduler import prefix

        prefix(
            record["source"],
            dict(size=record["source_prefix_bytes"], hash=record["sha256"]),
        )
        observed = {r["session"]: r for r in record["sessions"]}
        if len(observed) != len(record["sessions"]):
            raise ValueError("duplicate observed calendar dates")
        if set(byday) - set(observed):
            raise ValueError("fill sessions missing from observed calendar")
    sessions, per_session, all_rows = {}, {}, []
    offsets = defaultdict(float)
    dates = set(byday) if unknown_epoch else set(byday) | set(observed)
    for day in sorted(dates):
        fday = byday.get(day, [])
        if not fday:
            record = observed[day]
            for time in sorted(
                {record["first_timestamp"], record["last_timestamp"]}, key=timestamp
            ):
                for strategy in ("MIM", "YANK"):
                    all_rows.append(
                        dict(
                            timestamp=time,
                            strategy=strategy,
                            cumulative_net=offsets[strategy],
                            gross_notional=0.0,
                            baseline_units=1.0 if strategy == "MIM" else 2.0,
                        )
                    )
            sessions[day] = dict(
                status="CONDITIONAL_ZERO_NO_FILLS_FLAT_CARRY",
                observed_rows=record["observed_rows"],
            )
            continue
        marks = sorted(daily_marks[day], key=lambda r: timestamp(r["timestamp"]))
        coverage = dict(
            start=marks[0]["timestamp"],
            end=marks[-1]["timestamp"],
            initial_flat=False,
            complete_export=False,
            strategies=["MIM", "YANK"],
            baseline_units={"MIM": 1, "YANK": 2},
            grid_seconds=60,
        )
        status, rows = marked_curves(fday, marks, coverage, assumptions=ASSUMPTIONS)
        sessions[day] = status
        if status["status"] == "INSUFFICIENT_DATA" or status["final_open_positions"]:
            return (
                dict(
                    status="INSUFFICIENT_DATA",
                    sessions=sessions,
                    reason="missing marks or nonflat session boundary",
                ),
                rows,
            )
        if unknown_epoch:
            # No reset identity: do not bridge any balance, drawdown or recovery across dates.
            per_session[day] = portfolio_report(rows, operating_cost_monthly=0.0)
            for row in rows:
                row["accounting_scope"] = "PER_SESSION_CONDITIONAL_UNKNOWN_EPOCH"
            all_rows.extend(rows)
            continue
        for row in rows:
            row["cumulative_net"] += offsets[row["strategy"]]
        for strategy in coverage["strategies"]:
            offsets[strategy] = next(
                r["cumulative_net"] for r in reversed(rows) if r["strategy"] == strategy
            )
        all_rows.extend(rows)
    if unknown_epoch:
        report = dict(
            status="PER_SESSION_CONDITIONAL_UNKNOWN_EPOCH",
            per_session=per_session,
            reason="Epoch unknown: no cross-session equity, drawdown, recovery or flat-carry inference.",
            operating_cost_monthly=operating_cost_monthly,
            operating_expense_allocation="UNKNOWN; not charged independently to each session",
        )
    else:
        report = portfolio_report(
            all_rows, operating_cost_monthly=operating_cost_monthly
        )
        if report["status"] != "INSUFFICIENT_DATA":
            report["status"] = "CONDITIONAL_DESCRIPTIVE"
    report.update(
        assumptions=ASSUMPTIONS,
        sessions=sessions,
        inventory=inventory_before,
        inputs_unchanged_during_read=inventory_before == inventory(inputs),
        sample_scope=(
            "Separate sessions only; no cross-session estimate"
            if unknown_epoch
            else (
                "Observed calendar including conditional flat-carry zero dates"
                if calendar
                else "Active-fill sessions only"
            )
        ),
        no_confirmatory_claim=True,
    )
    if not report["inputs_unchanged_during_read"]:
        report["status"] = "INPUT_CHANGED_RETRY"
    return report, all_rows
