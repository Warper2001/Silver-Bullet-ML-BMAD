"""Broker-fill evidence; explicit IDs only, no speculative trade matching."""

import json
import csv
import re
from datetime import timezone
from zoneinfo import ZoneInfo
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from .common import input_path, inventory, number, read_csv, timestamp


def reconcile(fills, orders=(), epoch_start=None, account=None):
    """Retain partial fills individually; deduplicate by account and broker fill ID.

    Epoch exclusion applies to totals, never deletes raw evidence. Strategy orders
    cannot establish account attribution when their CSV has no account identity.
    """
    seen, rows, issues = {}, [], []
    order_ids = {str(row.get("order_id")) for row in orders if row.get("order_id")}
    for raw in fills:
        row = dict(raw)
        acct, fid = row.get("accountId"), row.get("id")
        key = (acct, fid)
        if acct is None or fid is None:
            issues.append("missing_account_or_fill_id")
        elif key in seen:
            if seen[key] != raw:
                issues.append("conflicting_duplicate_fill")
            continue
        else:
            seen[key] = raw
        time = timestamp(row["creationTimestamp"])
        fees, commission = number(row.get("fees")), number(row.get("commissions"))
        cost = (
            fees + commission if fees is not None and commission is not None else None
        )
        pnl = number(row.get("profitAndLoss"))
        quantity = number(row.get("size"))
        if quantity is None or quantity <= 0 or not quantity.is_integer():
            issues.append("invalid_quantity")
        if type(row.get("side")) is not int or row.get("side") not in (0, 1):
            issues.append("unknown_side")
        if not row.get("contractId"):
            issues.append("missing_contract")
        price = number(row.get("price"))
        if price is None or price <= 0:
            issues.append("invalid_price")
        included = not row.get("voided", False)
        reason = "included"
        if row.get("voided", False):
            reason = "voided"
        elif account is not None and str(acct) != str(account):
            included, reason = False, "other_account"
        elif epoch_start is not None and time < timestamp(epoch_start):
            included, reason = False, "before_epoch"
        rows.append(
            dict(
                fill_id=fid,
                account=acct,
                contract=row.get("contractId"),
                timestamp=time.astimezone(timezone.utc).isoformat(),
                order_id=row.get("orderId"),
                side=row.get("side"),
                quantity=quantity,
                price=price,
                gross_realized_pnl=pnl,
                actual_fees=fees,
                actual_commissions=commission,
                actual_total_cost=cost,
                included=included,
                exclusion_reason=reason,
                order_id_observed=str(row.get("orderId")) in order_ids,
                strategy_attribution="unknown",
                account_epoch=epoch_start,
            )
        )
    rows.sort(
        key=lambda row: (
            timestamp(row["timestamp"]),
            str(row["account"]),
            str(row["fill_id"]),
        )
    )
    for row in rows:
        row["evidence_valid"] = not issues
        row["validation_issues"] = sorted(set(issues))
    active = [row for row in rows if row["included"]]
    missing_cost = sum(row["actual_total_cost"] is None for row in active)
    realized = [
        row["gross_realized_pnl"]
        for row in active
        if row["gross_realized_pnl"] is not None
    ]
    known_cost = sum(row["actual_total_cost"] or 0 for row in active)
    # Null P&L on an entry is not imputed to zero; only broker-reported realized
    # P&L values contribute to this specifically named snapshot aggregate.
    report = dict(
        status="INVALID_EVIDENCE" if issues else "PARTIAL_COVERAGE",
        issues=sorted(set(issues)),
        fills=len(active),
        retained_records=len(rows),
        accounts=sorted({str(row["account"]) for row in active}),
        contracts=sorted({str(row["contract"]) for row in active}),
        first_timestamp=active[0]["timestamp"] if active else None,
        last_timestamp=active[-1]["timestamp"] if active else None,
        reported_realized_records=len(realized),
        reported_gross_realized_sum=sum(realized) if realized and not issues else None,
        known_cost_subtotal=known_cost if not issues else None,
        missing_cost_records=missing_cost,
        snapshot_net_reported_realized_less_all_fill_costs=(
            sum(realized) - known_cost
            if realized and not missing_cost and not issues
            else None
        ),
        unmatched_order_ids=sum(not row["order_id_observed"] for row in active),
        strategy_attribution="unknown; accountless strategy CSV cannot prove joins",
        completeness="not certified; requires broker export start/end and account reset register",
        modeled_slippage="not measured; requires contemporaneous intended prices",
    )
    return report, rows


def realtime_ledger(path):
    path = input_path(path)
    if not path.exists():
        return dict(status="MISSING", rows=0)
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        cols = {row[1] for row in db.execute("PRAGMA table_info(trades)")}
        if "write_mode" not in cols:
            return dict(status="UNKNOWN_REALTIME_MODE", rows=None)
        rows = [
            dict(row)
            for row in db.execute("SELECT * FROM trades WHERE write_mode='realtime'")
        ]
    missing, invalid = Counter(), 0
    for row in rows:
        try:
            timestamp(row["timestamp"])  # mixed ISO8601 shapes accepted explicitly
        except (ValueError, KeyError, TypeError):
            invalid += 1
        metadata = row.get("metadata")
        try:
            metadata = (
                json.loads(metadata) if isinstance(metadata, str) else metadata or {}
            )
        except (TypeError, ValueError):
            metadata = {}
        for field in ("quantity", "account", "symbol"):
            if row.get(field) is None and metadata.get(field) is None:
                missing[field] += 1
    return dict(
        status="PARTIAL_COVERAGE",
        rows=len(rows),
        missing_provenance=dict(missing),
        invalid_timestamps=invalid,
        pnl_not_combined_with_broker="no proven identity join",
    )


def shared_equity(paths):
    streams = []
    for path in paths:
        if not Path(path).exists():
            streams.append(dict(path=str(path), status="MISSING"))
            continue
        rows = read_csv(path)
        streams.append(
            dict(
                path=str(path),
                rows=len(rows),
                first=rows[0] if rows else None,
                last=rows[-1] if rows else None,
            )
        )
    return dict(
        status="SHARED_ACCOUNT_OBSERVATIONS_ONLY",
        streams=streams,
        warning="Both files observe the same SIM account. Never add or attribute these curves.",
    )


def audit(root, account=None, epoch_start=None):
    root = input_path(root)
    paths = [
        root / "data/mim_nb" / name
        for name in (
            "projectx_fills.json",
            "orders.csv",
            "trades.csv",
            "sessions.csv",
            "state.json",
        )
    ]
    paths += [
        root / "data/trades.db",
        root / "data/ts_sim_mirror/mim_invvol_equity.csv",
        root / "data/ts_sim_mirror/yank_invvol_equity.csv",
    ]
    before = inventory(paths)
    fills = json.loads(paths[0].read_text()) if paths[0].exists() else []
    result, rows = reconcile(
        fills, read_csv(paths[1]) if paths[1].exists() else [], epoch_start, account
    )
    result["ledger"] = realtime_ledger(paths[5])
    result["sim_observations"] = shared_equity(paths[6:])
    result["inventory"] = before
    result["inputs_unchanged_during_read"] = before == inventory(paths)
    if not result["inputs_unchanged_during_read"]:
        result["status"] = "INPUT_CHANGED_RETRY"
    return result, rows


def broker_export(path):
    data = json.loads(input_path(path).read_text())
    if isinstance(data, list):
        return data
    response = data.get("response", data)
    if response.get("success") is False:
        raise ValueError("broker export unsuccessful")
    return response["trades"]


def attributed_snapshot(
    fills_path, attribution_path, orders=(), epoch_start=None, account=None
):
    fills = broker_export(fills_path)
    mapping = json.loads(input_path(attribution_path).read_text())
    sources, limits = {}, {}
    for record in mapping["orders"].values():
        record["source"] = str(input_path(record["source"]))
        path = record["source"]
        identity = (record["sha256"], record.get("source_prefix_bytes"))
        if identity[1] is not None and (
            type(identity[1]) is not int or identity[1] <= 0
        ):
            raise ValueError("invalid source prefix length")
        if path in sources and (sources[path], limits[path]) != identity:
            raise ValueError("inconsistent attribution source identity")
        sources[path], limits[path] = identity
    from .common import digest

    for path, expected in sources.items():
        if limits[path] is not None:
            from .scheduler import prefix

            prefix(path, dict(size=limits[path], hash=expected))
        elif digest(path) != expected:
            raise ValueError("attribution source changed: " + path)
    evidence, headers = {}, {}
    for path in sources:
        requested = {
            v["line"] for v in mapping["orders"].values() if v["source"] == path
        }
        evidence[path] = {}
        consumed = 0
        with input_path(path).open("rb") as stream:
            for n, line in enumerate(stream, 1):
                if n == 1:
                    headers[path] = line.decode()
                consumed += len(line)
                if limits[path] is not None and consumed > limits[path]:
                    break
                if n in requested:
                    evidence[path][n] = line.decode()
                if n >= max(requested):
                    break
    result, rows = reconcile(fills, orders, epoch_start, account)
    if result["issues"]:
        result.update(
            strategy_totals={},
            economic_sensitivity=None,
            monthly_reported_realized_less_all_fill_costs=None,
        )
        return result, rows, []
    totals = defaultdict(
        lambda: dict(fills=0, gross_reported=0.0, actual_cost=0.0, missing_costs=0)
    )
    canonical = []
    for row in rows:
        record = mapping["orders"].get(str(row["order_id"]))
        if record:
            line = evidence[record["source"]].get(record["line"], "")
            if record["strategy"] == "MIM":
                parsed = list(csv.DictReader([headers[record["source"]], line]))
                observed_id = parsed[0].get("order_id") if len(parsed) == 1 else None
            elif record["strategy"] == "YANK":
                match = re.search(
                    r"ProjectX (?:entry limit|market close) #(\d+)(?!\d)", line
                )
                observed_id = match.group(1) if match else None
            else:
                raise ValueError("unsupported attribution source format")
            if observed_id != str(row["order_id"]):
                raise ValueError("order ID absent from exact cited attribution field")
            row["strategy_attribution"] = record["strategy"]
            row["attribution_source"] = record["source"] + ":" + str(record["line"])
            row["attribution_sha256"] = record["sha256"]
        strategy = row["strategy_attribution"]
        total = totals[strategy]
        if row["included"]:
            total["fills"] += 1
            total["gross_reported"] += row["gross_realized_pnl"] or 0.0
            total["actual_cost"] += row["actual_total_cost"] or 0.0
            total["missing_costs"] += row["actual_total_cost"] is None
        if not row["included"]:
            continue
        canonical.append(
            dict(
                id=row["fill_id"],
                timestamp=row["timestamp"],
                account=row["account"],
                epoch=row["account_epoch"],
                strategy=strategy,
                contract=row["contract"],
                signed_quantity=row["quantity"] * (1 if row["side"] == 0 else -1),
                price=row["price"],
                actual_cost=row["actual_total_cost"],
            )
        )
    for total in totals.values():
        total["net_reported"] = (
            total["gross_reported"] - total["actual_cost"]
            if not total["missing_costs"]
            else None
        )
    result["strategy_totals"] = dict(totals)
    monthly = defaultdict(float)
    turnover = defaultdict(float)
    cost_complete = all(
        row["actual_total_cost"] is not None for row in rows if row["included"]
    )
    for row in rows:
        if not row["included"]:
            continue
        month = (
            timestamp(row["timestamp"])
            .astimezone(ZoneInfo("America/New_York"))
            .strftime("%Y-%m")
        )
        monthly[month] += (row["gross_realized_pnl"] or 0.0) - (
            row["actual_total_cost"] or 0.0
        )
        turnover[month] += row["quantity"]
    result["monthly_reported_realized_less_all_fill_costs"] = (
        dict(monthly) if cost_complete else None
    )
    result["economic_sensitivity"] = (
        [
            dict(
                extra_modeled_slippage_per_filled_contract=cost,
                monthly_operating_assumption=operating,
                monthly_net={
                    month: value - cost * turnover[month] - operating
                    for month, value in monthly.items()
                },
            )
            for cost in (0.0, 0.5, 1.0, 2.0)
            for operating in (0.0, 29.0, 78.0, 116.0)
        ]
        if cost_complete
        else None
    )
    result["economic_scope"] = (
        "Partial snapshot cash P&L; months may be incomplete. "
        "Actual prices already include realized slippage. Stress is additional hypothetical degradation; "
        "operating assumptions are not invoices. This is not marked-equity drawdown or earning capacity."
    )

    result["inventory"] = inventory([fills_path, attribution_path])
    result["source_prefix_inventory"] = [
        dict(path=p, bytes=limits[p], sha256=h) for p, h in sources.items()
    ]
    result["coverage"] = (
        "broker export snapshot; independent flat-start and completeness statement unavailable"
    )
    return result, rows, canonical
