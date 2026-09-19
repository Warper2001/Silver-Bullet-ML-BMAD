"""Deterministic advisory reconciliation from immutable broker captures."""

import json
from typing import Any
import math
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from .capture import immutable, sha
from .common import timestamp
from .scheduler import atomic

ET = ZoneInfo("America/New_York")


def session_bounds(day: str) -> tuple[datetime, datetime]:
    day = date.fromisoformat(str(day))
    return (
        datetime.combine(day - timedelta(days=1), time(18), ET),
        datetime.combine(day, time(17), ET),
    )


def scheduled_days(now: datetime) -> list[str]:
    """Weekday 08:30 refresh and 16:10 report, plus restart catchup, DST-aware."""
    local = now.astimezone(ET)
    previous = local.date() - timedelta(days=1)
    while previous.weekday() >= 5:
        previous -= timedelta(days=1)
    days = [previous.isoformat()]  # catchup is safe: reports are content addressed
    if local.weekday() < 5 and local.time().replace(tzinfo=None) >= time(16, 10):
        days.append(local.date().isoformat())
    return days


def response_rows(
    snapshot: dict[str, Any], endpoint: str
) -> list[dict[str, Any]] | None:
    from .capture import ENDPOINTS

    requests = [r for r in snapshot["requests"] if r["endpoint"] == endpoint]
    if not requests or any(not r.get("ok") for r in requests):
        return None
    rows = []
    for request in requests:
        payload = request.get("response", {}).get(ENDPOINTS[endpoint])
        if not isinstance(payload, list) or any(
            not isinstance(row, dict) for row in payload
        ):
            return None
        rows.extend(payload)
    return rows


def account_observed(snapshot: dict[str, Any]) -> datetime:
    requests = [
        r
        for r in snapshot["requests"]
        if r["endpoint"] == "Account/search" and r.get("ok")
    ]
    return (
        timestamp(requests[0].get("observed_at", snapshot["observed_at"]))
        if requests
        else timestamp(snapshot["observed_at"])
    )


def balance(snapshot: dict[str, Any], account: str) -> float | None:
    rows = response_rows(snapshot, "Account/search")
    found = [r for r in rows or [] if str(r.get("id")) == str(account)]
    try:
        value = float(found[0]["balance"]) if len(found) == 1 else None
        return value if value is not None and math.isfinite(value) else None
    except (KeyError, TypeError, ValueError):
        return None


def flat_reasons(snapshot: dict[str, Any], account: str) -> list[str]:
    reasons = []
    if str(snapshot["account"]) != str(account) or balance(snapshot, account) is None:
        reasons.append("account_or_balance_unknown")
    for endpoint in ("Order/searchOpen", "Position/searchOpen"):
        rows = response_rows(snapshot, endpoint)
        if rows is None or rows:
            reasons.append("broker_not_proven_flat:" + endpoint)
    for strategy in ("MIM", "YANK"):
        before = snapshot["local_before"].get(strategy, {})
        after = snapshot["local_after"].get(strategy, {})
        if not isinstance(before, dict) or not isinstance(after, dict):
            reasons.append(strategy + ":unknown_state")
            continue
        producer = after.get("producer", {})
        if not isinstance(producer, dict):
            reasons.append(strategy + ":unknown_producer")
            continue
        if (
            before != after
            or not after.get("stable")
            or not producer.get("identity")
            or not producer.get("matches_source")
            or str(producer.get("account")) != str(account)
        ):
            reasons.append(strategy + ":unstable_or_unproven_producer")
        try:
            published = numeric(after.get("mtime_ns")) / 1e9
            producer_start = timestamp(producer["started_at"]).timestamp()
            if published < producer_start:
                reasons.append(strategy + ":state_predates_producer")
        except (KeyError, TypeError, ValueError):
            reasons.append(strategy + ":state_provenance_unknown")
        state = after.get("state")
        if not isinstance(state, dict):
            reasons.append(strategy + ":unknown_state")
        elif strategy == "MIM":
            if (
                type(state.get("position")) not in (int, float)
                or state.get("position") != 0
                or "cat_stop_id" not in state
                or state.get("cat_stop_id") is not None
                or not state.get("saved_at")
            ):
                reasons.append("MIM:not_individually_flat")
        elif (
            set(state) != {"daily_pnl", "daily_halted", "last_trading_date"}
            or not isinstance(state.get("daily_pnl"), (int, float))
            or type(state.get("daily_halted")) is not bool
            or not state.get("last_trading_date")
        ):
            reasons.append("YANK:not_individually_flat")
    return reasons


def observation(
    snapshots: list[tuple[str, dict[str, Any]]], account: str
) -> dict[str, Any]:
    previous = None
    for key, snapshot in snapshots:
        if not flat_reasons(snapshot, account):
            if previous:
                old_key, old = previous
                elapsed = (
                    timestamp(snapshot["started_at"]) - timestamp(old["observed_at"])
                ).total_seconds()
                if (
                    0 <= elapsed <= 180
                    and balance(old, account) == balance(snapshot, account)
                    and all(
                        old["local_after"][name]["producer"]
                        == snapshot["local_before"][name]["producer"]
                        for name in ("MIM", "YANK")
                    )
                ):
                    return dict(
                        status="STARTED",
                        account=str(account),
                        observed_at=snapshot["observed_at"],
                        balance=balance(snapshot, account),
                        source_hashes=[old_key, key],
                        account_reset_epoch=None,
                    )
            previous = (key, snapshot)
        else:
            previous = None
    return dict(
        status="PENDING_FLAT_EVIDENCE",
        account_reset_epoch=None,
        reasons=(
            flat_reasons(snapshots[-1][1], account) if snapshots else ["no_snapshots"]
        ),
    )


def covering(
    snapshot: dict[str, Any], endpoint: str, start: datetime, end: datetime
) -> bool:
    windows = []
    for request in snapshot["requests"]:
        if request["endpoint"] != endpoint or not request.get("ok"):
            continue
        body = request["response"]
        field = "trades" if endpoint == "Trade/search" else "orders"
        # Conservative cap guard: API has no count/pagination completeness contract here.
        if len(body[field]) >= 1000 or any(
            body.get(k) for k in ("hasMore", "nextPage", "nextPageToken")
        ):
            continue
        windows.append(
            (
                timestamp(request["request"]["startTimestamp"]),
                timestamp(request["request"]["endTimestamp"]),
            )
        )
    cursor = start
    for left, right in sorted(windows):
        if left <= cursor:
            cursor = max(cursor, right)
    return cursor >= end


def economics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    issues, seen = [], {}
    for row in trades:
        key = str(row.get("id"))
        if row.get("id") is None:
            issues.append("missing_fill_id")
        elif key in seen and seen[key] != row:
            issues.append("conflicting_fill:" + key)
        seen[key] = row
    valid, gross, fees, commissions = [], 0.0, 0.0, 0.0
    invalid = bool(issues)
    for key, raw in sorted(seen.items()):
        row = dict(raw)
        if row.get("voided") is True:
            continue
        try:
            if (
                row.get("voided") is not False
                or type(row.get("side")) is not int
                or row["side"] not in (0, 1)
                or not row.get("contractId")
                or row.get("orderId") is None
                or not float(row["size"]).is_integer()
                or float(row["size"]) <= 0
                or float(row["price"]) <= 0
            ):
                raise ValueError()
            values = [float(row["size"]), float(row["price"])]
            if row.get("profitAndLoss") is not None:
                values.append(float(row["profitAndLoss"]))
            if not all(math.isfinite(v) for v in values):
                raise ValueError()
            row["size"] = int(float(row["size"]))
            row["price"] = float(row["price"])
            if row.get("profitAndLoss") is None:
                issues.append("unknown_profitAndLoss:" + key)
                invalid = True
            else:
                gross += float(row["profitAndLoss"])
            valid.append(row)
        except (KeyError, TypeError, ValueError):
            issues.append("invalid_fill:" + key)
            invalid = True
            continue
        for field in ("fees", "commissions"):
            try:
                value = float(row[field])
                if not math.isfinite(value):
                    raise ValueError()
                if field == "fees" and fees is not None:
                    fees += value
                if field == "commissions" and commissions is not None:
                    commissions += value
            except (KeyError, TypeError, ValueError):
                issues.append("unknown_" + field + ":" + key)
                if field == "fees":
                    fees = None
                else:
                    commissions = None
    gross = None if invalid else gross
    costs = (
        fees + commissions
        if fees is not None and commissions is not None and not invalid
        else None
    )
    return dict(
        gross=gross,
        fees=fees,
        commissions=commissions,
        costs=costs,
        net=gross - costs if gross is not None and costs is not None else None,
        issues=issues,
        fills=valid,
    )


def scheduled_events(now: datetime) -> list[tuple[str, str]]:
    local = now.astimezone(ET)
    result = []
    # Latest seven calendar days catch up after a restart; explicit CLI handles older gaps.
    for offset in range(7, -1, -1):
        day = local.date() - timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        prior = day - timedelta(days=1)
        while prior.weekday() >= 5:
            prior -= timedelta(days=1)
        for label, at, target in [
            ("refresh", time(8, 30), prior),
            ("close", time(16, 10), day),
        ]:
            if datetime.combine(day, at, ET) <= local:
                result.append((day.isoformat() + "-" + label, target.isoformat()))
    return result


# Content-addressed source cache: every unchanged blob is parsed/verified once per process.
# Stat signatures also detect on-disk corruption or replacement of cached evidence.
_CACHE: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}


def load_artifact(path: Path) -> dict[str, Any]:
    stat = path.stat()
    signature = (stat.st_mtime_ns, stat.st_size)
    cached = _CACHE.get(str(path))
    if cached and cached[0] == signature:
        return cached[1]
    value = json.loads(path.read_text())
    if sha(value) != path.stem:
        raise ValueError("artifact hash mismatch")
    if len(_CACHE) > 2048:
        _CACHE.clear()
    _CACHE[str(path)] = (signature, value)
    return value


def load_history(output: Path, until: datetime) -> list[tuple[str, dict[str, Any]]]:
    result = []
    for path in (output / "snapshots").glob("*.json"):
        raw = load_artifact(path)
        if timestamp(raw["observed_at"]) > until:
            continue
        data = dict(raw)
        for field in ("local_before", "local_after"):
            if field + "_hash" in data:
                data[field] = load_artifact(
                    output / "sources" / (data[field + "_hash"] + ".json")
                )
        result.append((path.stem, data))
    return sorted(result, key=lambda pair: (pair[1]["observed_at"], pair[0]))


def numeric(value: Any, *, integer: bool = False) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError("unknown number")
    number = float(value)
    if not math.isfinite(number) or (
        integer and (number < 0 or not number.is_integer())
    ):
        raise ValueError("invalid number")
    return number


def account_interval(snapshot: dict[str, Any]) -> tuple[datetime, datetime]:
    requests = [
        r
        for r in snapshot["requests"]
        if r["endpoint"] == "Account/search" and r.get("ok")
    ]
    if not requests:
        return timestamp(snapshot["started_at"]), timestamp(snapshot["observed_at"])
    row = requests[0]
    return timestamp(row.get("started_at", snapshot["started_at"])), account_observed(
        snapshot
    )


def select_history(
    snapshots: list[tuple[str, dict[str, Any]]],
    endpoint: str,
    start: datetime,
    end: datetime,
) -> tuple[str, dict[str, Any]] | None:
    for pair in reversed(snapshots):
        if covering(pair[1], endpoint, start, end):
            return pair
    return None


def scoped_rows(
    snapshot: dict[str, Any],
    endpoint: str,
    account: str,
    start: datetime,
    end: datetime,
    issues: list[str],
    referenced_orders: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows = response_rows(snapshot, endpoint)
    if rows is None:
        issues.append("missing_response:" + endpoint)
        return []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            issues.append("malformed_record:" + endpoint)
            continue
        try:
            times = [timestamp(row["creationTimestamp"])]
            if endpoint == "Order/search" and row.get("updateTimestamp"):
                times.append(timestamp(row["updateTimestamp"]))
        except (KeyError, TypeError, ValueError, AttributeError):
            issues.append("malformed_timestamp:" + endpoint)
            continue
        in_window = any(start <= when < end for when in times)
        referenced = endpoint == "Order/search" and str(row.get("id")) in (
            referenced_orders or set()
        )
        if not in_window and not referenced:
            continue
        if str(row.get("accountId")) != str(account):
            issues.append("missing_or_wrong_account:" + endpoint)
            continue
        out.append(dict(row))
    return out


def position_inventory(
    snapshot: dict[str, Any], account: str, issues: list[str]
) -> dict[str, float] | None:
    rows = response_rows(snapshot, "Position/searchOpen")
    if rows is None:
        issues.append("positions_unknown")
        return None
    result: dict[str, float] = {}
    for row in rows:
        try:
            if (
                str(row.get("accountId")) != str(account)
                or not row.get("contractId")
                or type(row.get("type")) is not int
                or row["type"] not in (1, 2)
            ):
                raise ValueError()
            quantity = numeric(row.get("size"), integer=True)
            contract = row["contractId"]
            result[contract] = result.get(contract, 0) + quantity * (
                1 if row["type"] == 1 else -1
            )
        except (KeyError, TypeError, ValueError):
            issues.append("invalid_position_identity_or_quantity")
            return None
    return result


def reconcile_day(output: Path, account: str, day: str) -> dict[str, Any]:
    output = Path(output)
    start, end = session_bounds(day)
    # Corrections are bounded to the collector's eight-day rolling retention window.
    snapshots = load_history(output, end + timedelta(days=8))
    matched = [
        (key, s) for key, s in snapshots if str(s.get("account")) == str(account)
    ]
    issues: list[str] = []
    observed = [
        (key, s)
        for key, s in matched
        if timestamp(s["started_at"]) <= end + timedelta(minutes=5)
    ]
    closing = [
        (key, s)
        for key, s in observed
        if end <= timestamp(s["started_at"]) <= end + timedelta(minutes=5)
    ]
    exposure = closing[0] if closing else (observed[-1] if observed else None)
    horizon = timestamp(exposure[1]["observed_at"]) if exposure else end
    relevant = [
        (key, s) for key, s in snapshots if timestamp(s["observed_at"]) <= horizon
    ]
    local = [(key, s) for key, s in relevant if str(s.get("account")) == str(account)]
    obs = observation(local, account)
    used = set(obs.get("source_hashes", []))
    if obs["status"] != "STARTED":
        issues.append("observation_not_started")
    elif timestamp(obs["observed_at"]) > start:
        issues.append("observation_started_after_session_open")
    query_end = min(end, horizon)
    selected = {
        endpoint: select_history(matched, endpoint, start, query_end)
        for endpoint in ("Trade/search", "Order/search")
    }
    if query_end < end or any(pair is None for pair in selected.values()):
        issues.append("incomplete_session_coverage")
    data: dict[str, list[dict[str, Any]]] = {}
    for endpoint, pair in selected.items():
        if pair:
            used.add(pair[0])
            data[endpoint] = scoped_rows(
                pair[1],
                endpoint,
                account,
                start,
                query_end,
                issues,
                {str(row.get("orderId")) for row in data.get("Trade/search", [])},
            )
        else:
            data[endpoint] = []
    trades, orders = data["Trade/search"], data["Order/search"]
    # Strategy claims are monotonic sets, keyed by exact broker account/order/contract.
    claims: dict[tuple[str, str, str], dict[str, set[str]]] = {}

    def claim(strategy: str, sub: dict[str, Any], source: str) -> None:
        if (
            str(sub.get("accountId")) != str(account)
            or sub.get("orderId") is None
            or not sub.get("contractId")
        ):
            return
        ident = (str(account), str(sub["orderId"]), str(sub["contractId"]))
        claims.setdefault(ident, {}).setdefault(strategy, set()).add(source)
        used.add(source)

    for source, evidence in local:
        after = evidence.get("local_after", {})
        for sub in after.get("mim_submissions", []):
            if isinstance(sub, dict):
                claim("MIM", sub, source)
        for item in after.get("mim_fills", []):
            if isinstance(item, dict) and isinstance(item.get("submission"), dict):
                claim("MIM", item["submission"], source)
        yank = after.get("YANK", {})
        if not isinstance(yank, dict):
            issues.append("malformed_yank_state")
            continue
        producer, state = yank.get("producer", {}), yank.get("state")
        if not isinstance(state, dict) or not isinstance(producer, dict):
            issues.append("malformed_yank_state")
            continue
        if str(producer.get("account")) == str(account) and producer.get(
            "matches_source"
        ):
            for field in ("sim_entry_order_id", "sim_tp_order_id", "sim_sl_order_id"):
                oid = state.get(field)
                contracts = {
                    r.get("contractId")
                    for r in orders
                    if oid is not None and str(r.get("id")) == str(oid)
                }
                if len(contracts) == 1 and None not in contracts:
                    claim(
                        "YANK",
                        dict(
                            accountId=account,
                            orderId=oid,
                            contractId=next(iter(contracts)),
                        ),
                        source,
                    )
        for sub in after.get("yank_orders", {}).get("orders", []):
            if not isinstance(sub, dict) or str(sub.get("accountId")) != str(account):
                continue
            try:
                proof = load_artifact(
                    output / "order-sources" / (sub["source_hash"] + ".json")
                )
                line = proof["text"].splitlines()[int(sub["line"]) - 1]
                from .order_evidence import ORDER

                match = ORDER.search(line)
                if (
                    not match
                    or match.group(2) != str(sub.get("orderId"))
                    or str(proof["producer"]["account"]) != str(account)
                ):
                    raise ValueError("invalid order proof")
            except (OSError, ValueError, KeyError, TypeError, IndexError):
                issues.append("invalid_yank_order_source")
                continue
            contracts = {
                r.get("contractId")
                for r in orders
                if str(r.get("id")) == str(sub["orderId"])
            }
            if len(contracts) == 1 and None not in contracts:
                claim(
                    "YANK",
                    dict(
                        accountId=account,
                        orderId=sub["orderId"],
                        contractId=next(iter(contracts)),
                    ),
                    source,
                )
                ident = (str(account), str(sub["orderId"]), str(next(iter(contracts))))
                claims[ident]["YANK"].add(sub["source_hash"] + ":" + str(sub["line"]))
                used.add(sub["source_hash"])
    econ = economics(trades)
    issues.extend(econ["issues"])
    by_order: dict[tuple[str, str], float] = {}
    inventory: dict[str, float] = {}
    for fill in econ["fills"]:
        ident = (str(fill["orderId"]), fill["contractId"])
        owners = claims.get((str(account), *ident), {})
        fill["attribution_claims"] = {
            name: sorted(hashes) for name, hashes in sorted(owners.items())
        }
        fill["strategy_attribution"] = (
            next(iter(owners))
            if len(owners) == 1
            else ("ambiguous" if owners else "unknown")
        )
        if len(owners) != 1:
            issues.append("unknown_or_ambiguous_strategy_attribution")
        by_order[ident] = by_order.get(ident, 0) + fill["size"]
        contract = fill["contractId"]
        inventory[contract] = inventory.get(contract, 0) + fill["size"] * (
            1 if fill["side"] == 0 else -1
        )
        found = [r for r in orders if (str(r.get("id")), r.get("contractId")) == ident]
        if not found or any(
            type(r.get("side")) is not int or r.get("side") != fill["side"]
            for r in found
        ):
            issues.append("order_fill_side_unreconciled:" + ident[0])
    unique_orders: dict[tuple[str, str], dict[str, Any]] = {}
    for order in orders:
        ident = (str(order.get("id")), str(order.get("contractId")))
        if ident in unique_orders and unique_orders[ident] != order:
            issues.append("conflicting_order:" + ident[0])
        unique_orders[ident] = order
    for ident in set(unique_orders) | set(by_order):
        try:
            order = unique_orders[ident]
            quantity = numeric(order.get("fillVolume"), integer=True)
            if (
                not order.get("contractId")
                or order.get("id") is None
                or quantity != by_order.get(ident, 0)
            ):
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            issues.append("filled_order_quantity_unreconciled:" + ident[0])
    opening = [
        (key, s)
        for key, s in local
        if start - timedelta(minutes=5) <= timestamp(s["observed_at"]) <= start
    ]
    open_pair = opening[-1] if opening else None
    close_pair = closing[0] if closing else None
    if not open_pair or flat_reasons(open_pair[1], account):
        issues.append("opening_inventory_unproven")
    if not close_pair or flat_reasons(close_pair[1], account):
        issues.append("closing_inventory_unproven")
    if open_pair and close_pair:
        used.update([open_pair[0], close_pair[0]])
        initial = position_inventory(open_pair[1], account, issues)
        final = position_inventory(close_pair[1], account, issues)
        if initial is not None and final is not None:
            for contract in set(initial) | set(final) | set(inventory):
                if initial.get(contract, 0) + inventory.get(contract, 0) != final.get(
                    contract, 0
                ):
                    issues.append("signed_inventory_mismatch:" + contract)
    bridge = None
    if obs["status"] == "STARTED":
        since = timestamp(obs["observed_at"])
        sequence = [
            (key, s)
            for key, s in local
            if since <= timestamp(s["observed_at"]) <= horizon
        ]
        bridge = dict(
            status="RECONCILED", segments=0, residuals=[], uncertain_boundaries=[]
        )
        for (old_key, old), (new_key, new) in zip(sequence, sequence[1:]):
            used.update([old_key, new_key])
            old_start, left = account_interval(old)
            new_start, right = account_interval(new)
            # Prefer the latest corrected records covering this bridge, not stale first-seen fills.
            corrected = select_history(matched, "Trade/search", old_start, right)
            if corrected is None:
                bridge["status"] = "INCOMPLETE"
                continue
            used.add(corrected[0])
            segment_issues: list[str] = []
            rows = scoped_rows(
                corrected[1], "Trade/search", account, old_start, right, segment_issues
            )
            boundary = any(
                old_start <= timestamp(r["creationTimestamp"]) < left
                or new_start <= timestamp(r["creationTimestamp"]) < right
                for r in rows
            )
            if boundary:
                bridge["uncertain_boundaries"].append([old_key, new_key])
                bridge["status"] = "INCOMPLETE"
                continue
            amounts = economics(
                [r for r in rows if left <= timestamp(r["creationTimestamp"]) < right]
            )
            before, after = balance(old, account), balance(new, account)
            if (
                segment_issues
                or amounts["net"] is None
                or before is None
                or after is None
            ):
                bridge["status"] = "INCOMPLETE"
                continue
            residual = round(after - before - amounts["net"], 8)
            bridge["segments"] += 1
            if residual:
                bridge["residuals"].append(
                    dict(source_hashes=[old_key, new_key], residual=residual)
                )
        if bridge["residuals"]:
            obs = dict(obs, status="INTERRUPTED", reason="unexplained_balance_delta")
            issues.append("balance_continuity_interrupted")
        if bridge["status"] != "RECONCILED":
            issues.append("balance_bridge_incomplete")
        for source, record in relevant:
            if timestamp(record["observed_at"]) <= since:
                continue
            if str(record.get("account")) != str(account):
                obs = dict(
                    obs, status="INTERRUPTED", reason="account_identity_transition"
                )
                issues.append("account_identity_transition")
                used.add(source)
            elif (
                response_rows(record, "Account/search") is not None
                and balance(record, account) is None
            ):
                obs = dict(
                    obs, status="INTERRUPTED", reason="account_disappeared_or_invalid"
                )
                issues.append("account_disappeared_or_invalid")
                used.add(source)
            for where in ("local_before", "local_after"):
                for strategy in ("MIM", "YANK"):
                    evidence = record.get(where, {}).get(strategy, {})
                    producer = (
                        evidence.get("producer", {})
                        if isinstance(evidence, dict)
                        else {}
                    )
                    producer_account = (
                        producer.get("account") if isinstance(producer, dict) else None
                    )
                    if producer_account is not None and str(producer_account) != str(
                        account
                    ):
                        obs = dict(
                            obs,
                            status="INTERRUPTED",
                            reason="producer_account_identity_transition",
                        )
                        issues.append(
                            "producer_account_identity_transition:" + strategy
                        )
                        used.add(source)
                    elif (
                        not isinstance(producer, dict)
                        or not producer.get("identity")
                        or not producer.get("matches_source")
                        or producer_account is None
                    ):
                        issues.append("producer_identity_unknown:" + strategy)
                        bridge["status"] = "INCOMPLETE"
                        used.add(source)
    result = dict(
        schema=2,
        account=str(account),
        session_date=str(day),
        status="INCOMPLETE" if issues else "COMPLETE",
        issues=sorted(set(issues)),
        observation=obs,
        account_reset_epoch=None,
        source_hashes=sorted(used),
        balance_bridge=bridge,
        gross_reported=econ["gross"],
        actual_fees=econ["fees"],
        actual_commissions=econ["commissions"],
        actual_costs=econ["costs"],
        net_total=econ["net"] if not issues else None,
        observed_net=econ["net"],
        fills=econ["fills"],
        signed_fill_inventory=inventory,
        zero_trade_session=(not trades and not issues),
        exposure=(
            dict(
                observed_at=exposure[1]["observed_at"],
                broker_positions=response_rows(exposure[1], "Position/searchOpen"),
                broker_open_orders=response_rows(exposure[1], "Order/searchOpen"),
                bot_states={
                    name: exposure[1]["local_after"].get(name, {})
                    for name in ("MIM", "YANK")
                },
            )
            if exposure
            else None
        ),
        note="Advisory; actual execution prices, no double slippage. Reset epoch unknown; natural traded-session acceptance pending.",
    )
    version = immutable(output / "reports" / str(day), result)
    latest_path = output / "latest_report.json"
    try:
        latest = json.loads(latest_path.read_text())
    except (OSError, ValueError):
        latest = {}
    # A refresh of an older day preserves its version without regressing health's
    # pointer to the most recent session already reported for this account.
    if latest.get("account") != str(account) or str(day) >= latest.get(
        "session_date", ""
    ):
        atomic(
            latest_path,
            dict(
                report_hash=version,
                account=str(account),
                session_date=str(day),
                status=result["status"],
                issues=result["issues"],
                observation=obs["status"],
            ),
        )
    immutable(output / "observations", obs)
    update_observation(output, obs, horizon)
    return result


def update_observation(
    output: Path, observed: dict[str, Any], horizon: datetime
) -> None:
    """Historical refresh cannot erase newer account interruption; preserve start identity."""
    path = output / "observation.json"
    try:
        previous = json.loads(path.read_text())
    except (OSError, ValueError):
        previous = {}
    if observed.get("status") == "PENDING_FLAT_EVIDENCE" and previous:
        # The capture loop owns current pending reasons. An old report with no
        # snapshots cannot replace live producer-provenance findings.
        return
    prior_horizon = (
        timestamp(previous["evaluated_through"])
        if previous.get("evaluated_through")
        else None
    )
    same_start = not previous.get("observed_at") or previous.get(
        "observed_at"
    ) == observed.get("observed_at")
    if previous.get("status") == "INTERRUPTED" and (
        not same_start or prior_horizon and horizon < prior_horizon
    ):
        return
    if observed.get("status") == "PENDING_FLAT_EVIDENCE" and previous.get(
        "observed_at"
    ):
        return
    if (
        previous.get("reason")
        in (
            "account_identity_transition",
            "producer_account_identity_transition",
            "account_disappeared_or_invalid",
        )
        and previous.get("status") == "INTERRUPTED"
    ):
        # Identity loss is not repaired by an economic correction or a later return.
        return
    current = dict(observed, evaluated_through=horizon.isoformat())
    atomic(path, current)
    immutable(output / "observations", current)


def refresh_corrections(
    output: Path, account: str, snapshot: dict[str, Any], limit: int = 2
) -> list[str]:
    """Refresh changed broker evidence for already reported days, bounded per cycle."""
    now = timestamp(snapshot["observed_at"])
    completed = []
    for offset in range(8):
        day = (now.astimezone(ET).date() - timedelta(days=offset)).isoformat()
        if not (output / "reports" / day).exists():
            continue
        start, end = session_bounds(day)
        signatures: dict[str, Any] = {}
        for endpoint in ("Trade/search", "Order/search"):
            problems: list[str] = []
            rows = scoped_rows(
                snapshot,
                endpoint,
                account,
                start,
                end,
                problems,
                {
                    str(row.get("orderId"))
                    for row in signatures.get("Trade/search", {}).get("rows", [])
                },
            )
            signatures[endpoint] = dict(
                rows=sorted(rows, key=lambda row: json.dumps(row, sort_keys=True)),
                issues=problems,
                covered=covering(snapshot, endpoint, start, end),
            )
        digest = sha(signatures)
        receipt = output / "corrections" / str(account) / (day + ".json")
        try:
            previous = json.loads(receipt.read_text())
        except (OSError, ValueError):
            previous = {}
        if previous.get("signature") == digest:
            continue
        reconcile_day(output, account, day)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        atomic(receipt, dict(signature=digest, day=day))
        completed.append(day)
        if len(completed) >= limit:
            break
    return completed
