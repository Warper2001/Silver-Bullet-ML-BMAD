"""Research evidence CLI. Outputs remain separate from read-only source data."""

import argparse
import json
from pathlib import Path
from .common import input_path, read_csv, write_csv, write_json, inventory
from .audit import audit, attributed_snapshot
from .portfolio import marked_curves, portfolio_report, fill_cost_sweep
from .accounts import sensitivity
from .power import sweep, standalone_sweep, evaluate
from .reconciliation import pair_fills, compare_records, realized_economics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--root", required=True, type=Path)
    a.add_argument("--fills", type=Path)
    a.add_argument("--attribution", type=Path)
    a.add_argument("--ledger-export", type=Path)
    a.add_argument("--account")
    a.add_argument("--epoch-start")
    r = sub.add_parser("report")
    r.add_argument("--broker-marks", type=Path)
    r.add_argument("--session-calendar", type=Path)
    for name in ("fills", "marks", "coverage", "account-sessions", "account-config"):
        r.add_argument("--" + name, type=Path)
    r.add_argument("--operating-cost-monthly", type=float, default=0.0)
    p = sub.add_parser("power")
    p.add_argument("--paired", type=Path)
    p.add_argument("--standalone", type=Path)
    e = sub.add_parser("evaluate")
    for name in ("data", "registration", "power", "calibration", "repo"):
        e.add_argument("--" + name, required=True, type=Path)
    for command in (a, r, p, e):
        command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    # Never write into an input directory/file, production data, or sealed holdout.
    if "sealed_holdout" in out.parts or "data" in out.parts:
        parser.error("outputs must be research artifacts outside source data")
    inputs = [
        value
        for key, value in vars(args).items()
        if isinstance(value, Path) and key != "output"
    ]
    for value in inputs:
        value = input_path(value)
        if value.is_file() and (out == value or out in value.parents):
            parser.error("output overlaps an input")
    out.mkdir(parents=True, exist_ok=False)
    if args.command == "audit":
        result, rows = audit(args.root, args.account, args.epoch_start)
        write_csv(out / "original_snapshot_fills.csv", rows)
        if args.fills:
            if not args.attribution:
                parser.error("--fills requires --attribution")
            result["fresh_broker_snapshot"], rows, canonical = attributed_snapshot(
                args.fills,
                args.attribution,
                read_csv(args.root / "data/mim_nb/orders.csv"),
                epoch_start=args.epoch_start,
                account=args.account,
            )
            write_csv(out / "attributed_fills.csv", rows)
            write_csv(out / "canonical_fills.csv", canonical)
            pairs, unmatched = pair_fills(rows)
            write_csv(out / "broker_roundtrip_parts.csv", pairs)
            write_json(out / "unmatched_broker_fills.json", unmatched)
            ledger = (
                json.loads(input_path(args.ledger_export).read_text())["rows"]
                if args.ledger_export
                else []
            )
            links = compare_records(
                pairs, read_csv(args.root / "data/mim_nb/trades.csv"), ledger
            )
            write_json(out / "strategy_ledger_reconciliation.json", links)
            from collections import Counter

            result["fresh_broker_snapshot"]["reconciliation_status_counts"] = dict(
                Counter(r["status"] for r in links)
            )
            result["fresh_broker_snapshot"]["unmatched_broker_fill_parts"] = len(
                unmatched
            )
            result["fresh_broker_snapshot"]["realized_economics"] = realized_economics(
                rows
            )

    elif args.command == "report":
        paths = [
            v
            for v in (
                args.fills,
                args.marks,
                args.coverage,
                args.account_sessions,
                args.account_config,
                args.session_calendar,
            )
            if v
        ]
        before = inventory(paths)
        if args.broker_marks and args.fills:
            from .session_report import session_report

            fills = read_csv(args.fills)
            paths.extend(sorted(args.broker_marks.glob("*.json")))
            before = inventory(paths)
            result, rows = session_report(
                fills,
                sorted(args.broker_marks.glob("*.json")),
                args.session_calendar,
                operating_cost_monthly=args.operating_cost_monthly,
            )
            write_csv(out / "marked_equity.csv", rows)
            result["cost_sweeps"] = {}
            if result["status"] in (
                "CONDITIONAL_DESCRIPTIVE",
                "PER_SESSION_CONDITIONAL_UNKNOWN_EPOCH",
            ):
                for extra in (0.5, 1.0, 2.0):
                    stressed = [
                        dict(
                            f,
                            actual_cost=float(f["actual_cost"])
                            + extra * abs(float(f["signed_quantity"])),
                        )
                        for f in fills
                    ]
                    result["cost_sweeps"][str(extra)] = session_report(
                        stressed,
                        sorted(args.broker_marks.glob("*.json")),
                        args.session_calendar,
                        operating_cost_monthly=args.operating_cost_monthly,
                    )[0]
        elif all((args.fills, args.marks, args.coverage)):
            fills, marks = read_csv(args.fills), read_csv(args.marks)
            coverage = json.loads(input_path(args.coverage).read_text())
            status, rows = marked_curves(fills, marks, coverage)
            result = dict(
                marked_equity=status,
                portfolio=portfolio_report(
                    rows, operating_cost_monthly=args.operating_cost_monthly
                ),
            )
            write_csv(out / "marked_equity.csv", rows)
            if status["status"] == "DESCRIPTIVE_ONLY":
                result["cost_sweep"] = fill_cost_sweep(
                    fills,
                    marks,
                    coverage,
                    (0.0, 0.5, 1.0, 2.0),
                    operating_cost_monthly=args.operating_cost_monthly,
                )
        else:
            result = dict(
                status="INSUFFICIENT_DATA",
                reason="attributable complete fill coverage and corrected matching marks required",
            )
        if args.account_sessions:
            sessions = json.loads(input_path(args.account_sessions).read_text())
            config = (
                json.loads(input_path(args.account_config).read_text())
                if args.account_config
                else None
            )
            result["account_scenarios"] = sensitivity(sessions, config)
        else:
            result["account_model"] = dict(
                status="AWAITING_INTRADAY_PATH",
                self_funded_capital=10000,
                automated_live_available=False,
                engineering="implemented",
            )
        result["inventory"] = before
        result["inputs_unchanged_during_read"] = before == inventory(paths)
        if not result["inputs_unchanged_during_read"]:
            result["status"] = "INPUT_CHANGED_RETRY"
    elif args.command == "power":
        result = sweep(args.paired)
        if args.standalone:
            result["standalone"] = standalone_sweep(args.standalone)
        write_csv(out / "power_sweep.csv", result["sweep"])
        if args.standalone:
            write_csv(out / "standalone_sweep.csv", result["standalone"]["sweep"])
    else:
        result = evaluate(
            args.data, args.registration, args.power, args.calibration, args.repo
        )
    result["computational_sources"] = inventory(Path(__file__).parent.glob("*.py"))
    result["command_input_inventory"] = inventory([p for p in inputs if p.is_file()])
    write_json(out / "evidence.json", result)
    lines = [
        "# " + args.command.title() + " evidence",
        "",
        "Status: **" + result.get("status", "See component statuses") + "**.",
        "",
        "This is a research artifact. Deployment and strategy changes are not authorized.",
        "",
    ]
    if args.command == "audit":
        lines += [
            "Original snapshot: "
            + str(result["fills"])
            + " fills; coverage remains partial.",
            "Shared SIM files are observations of one account and are not added together.",
            "",
        ]
        fresh = result.get("fresh_broker_snapshot", {})
        if fresh:
            lines += [
                "| Strategy | Fills | Gross reported | Actual costs | Net reported |",
                "|---|---:|---:|---:|---:|",
            ]
            for name, row in fresh["strategy_totals"].items():
                lines.append(
                    "| "
                    + name
                    + " | "
                    + str(row["fills"])
                    + " | "
                    + str(row["gross_reported"])
                    + " | "
                    + str(row["actual_cost"])
                    + " | "
                    + str(row["net_reported"])
                    + " |"
                )
            lines += [
                "",
                "Independent starting inventory, complete coverage and account reset history require a statement.",
                "Actual broker P&L already includes execution prices; modeled slippage must not be subtracted twice.",
                "",
            ]
    elif args.command == "power":
        lines += [
            "Paired corrected session evidence: "
            + result["status"]
            + ". No confirmatory efficacy conclusion.",
            "Standalone calibration uses exposed trade P&L; it does not establish portfolio benefit.",
            "",
        ]
        standalone = result.get("standalone", {})
        if standalone:
            lines += [
                "| Detectable effect | Required future trades (IID diagnostic) |",
                "|---:|---:|",
            ]
            unique = {
                r["effect_usd"]: r["required_future_trades"]
                for r in standalone["sweep"]
                if r["block_trades"] == 1
            }
            lines += [
                "| " + str(k) + " | " + str(v) + " |" for k, v in sorted(unique.items())
            ]
            lines += [
                "",
                "Calendar horizon depends on future eligible-session/trade rates, which are not yet established.",
                "",
            ]
    else:
        lines += [
            result.get(
                "reason",
                "See portfolio, path and gate component statuses in the evidence file.",
            ),
            "",
        ]
    lines += [
        "Reproducible details: `evidence.json`; tabular evidence is retained alongside it.",
        "Missing costs, marks, attribution and coverage remain unknown.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))
    write_json(
        out / "output_inventory.json",
        inventory(p for p in out.iterdir() if p.is_file()),
    )
    print(
        json.dumps(
            dict(
                command=args.command,
                status=result.get("status", "SEE_COMPONENT_STATUSES"),
                output=str(out),
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
