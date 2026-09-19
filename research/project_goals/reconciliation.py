"""Explicit fill pairing and separately labelled strategy/ledger inference."""

from collections import defaultdict, deque
from zoneinfo import ZoneInfo
from .common import timestamp, known_epoch
from .portfolio import metrics, path_risk


def pair_fills(rows, multipliers=None):
    """FIFO within account/epoch/contract/strategy; report unexplained exits intact.

    Pairing is an accounting reconstruction conditional on export coverage, never
    evidence that start inventory was zero. Partial fills allocate fees by quantity.
    """
    multipliers = multipliers or {"CON.F.US.MNQ.U26": 2.0, "CON.F.US.MNQ.Z26": 2.0}
    if any(row.get("evidence_valid") is False for row in rows):
        return [], [dict(row, reason="invalid_evidence_batch") for row in rows]
    books = defaultdict(deque)
    pairs = []
    unmatched = []
    for row in sorted(
        rows, key=lambda r: (timestamp(r["timestamp"]), str(r["fill_id"]))
    ):
        if not row["included"]:
            continue
        key = (
            row["account"],
            row["account_epoch"],
            row["contract"],
            row["strategy_attribution"],
            (
                None
                if known_epoch(row["account_epoch"])
                else str(
                    timestamp(row["timestamp"])
                    .astimezone(ZoneInfo("America/New_York"))
                    .date()
                )
            ),
        )
        if row["quantity"] is None or row["side"] not in (0, 1) or row["price"] is None:
            unmatched.append(dict(row, reason="invalid_fill_provenance"))
            continue
        book = books[key]
        remaining = row["quantity"]
        sign = 1 if row["side"] == 0 else -1
        if not book and row["gross_realized_pnl"] is not None:
            unmatched.append(dict(row, reason="realized_exit_without_observed_entry"))
            continue
        while remaining > 0 and book and book[0]["sign"] != sign:
            lot = book[0]
            qty = min(remaining, lot["remaining"])
            entry = lot["row"]
            multiplier = multipliers.get(row["contract"])
            gross = (
                (row["price"] - entry["price"]) * lot["sign"] * qty * multiplier
                if multiplier
                else None
            )
            cost = (
                (
                    entry["actual_total_cost"] / entry["quantity"]
                    + row["actual_total_cost"] / row["quantity"]
                )
                * qty
                if entry["actual_total_cost"] is not None
                and row["actual_total_cost"] is not None
                else None
            )
            pairs.append(
                dict(
                    account=key[0],
                    epoch=key[1],
                    contract=key[2],
                    strategy=key[3],
                    entry_fill_id=entry["fill_id"],
                    exit_fill_id=row["fill_id"],
                    entry_order_id=entry["order_id"],
                    exit_order_id=row["order_id"],
                    entry_timestamp=entry["timestamp"],
                    exit_timestamp=row["timestamp"],
                    direction=lot["sign"],
                    quantity=qty,
                    entry_price=entry["price"],
                    exit_price=row["price"],
                    reconstructed_gross=gross,
                    allocated_actual_cost=cost,
                    reconstructed_net=(
                        gross - cost if gross is not None and cost is not None else None
                    ),
                    basis=(
                        "FIFO conditional on export completeness; no synthetic entry"
                        if known_epoch(key[1])
                        else "PER_SESSION_CONDITIONAL_UNKNOWN_EPOCH; no cross-session pairing"
                    ),
                )
            )
            remaining -= qty
            lot["remaining"] -= qty
            if lot["remaining"] == 0:
                book.popleft()
        if remaining > 0:
            book.append(dict(row=row, remaining=remaining, sign=sign))
    for book in books.values():
        for lot in book:
            unmatched.append(
                dict(
                    lot["row"],
                    unmatched_quantity=lot["remaining"],
                    reason="open_or_missing_exit",
                )
            )
    return pairs, unmatched


def compare_records(pairs, strategy_rows, ledger_rows):
    """Inferred links require uniqueness; unmatched records are retained as evidence."""
    groups = defaultdict(list)
    for pair in pairs:
        groups[
            (
                pair["account"],
                pair["epoch"],
                pair["strategy"],
                pair["entry_order_id"],
                pair["exit_order_id"],
            )
        ].append(pair)
    trades = []
    for key, parts in groups.items():
        first = parts[0]
        trades.append(
            dict(
                first,
                quantity=sum(p["quantity"] for p in parts),
                reconstructed_gross=(
                    sum(p["reconstructed_gross"] for p in parts)
                    if all(p["reconstructed_gross"] is not None for p in parts)
                    else None
                ),
            )
        )
    outputs = []
    used = set()
    for index, row in enumerate(strategy_rows):
        candidates = []
        for n, trade in enumerate(trades):
            et = timestamp(trade["entry_timestamp"]).astimezone(
                ZoneInfo("America/New_York")
            )
            if (
                trade["strategy"] == "MIM"
                and str(et.date()) == row["day"]
                and et.strftime("%H:%M") == row["entry_t"]
                and trade["direction"] == int(row["dir"])
            ):
                candidates.append(n)
        linked = trades[candidates[0]] if len(candidates) == 1 else None
        if linked:
            used.add(candidates[0])
        outputs.append(
            dict(
                source="strategy_csv",
                source_row=index + 2,
                original=row,
                status=(
                    "UNIQUE_INFERRED_DATE_ENTRY_MINUTE_SIDE"
                    if linked
                    else ("AMBIGUOUS" if candidates else "UNMATCHED")
                ),
                candidates=candidates,
                broker_trade=linked,
                reported_minus_broker_gross=(
                    float(row["pnl_usd"]) - linked["reconstructed_gross"]
                    if linked and linked["reconstructed_gross"] is not None
                    else None
                ),
                warning="inferred comparison, not authoritative identity; quantity differences may explain gaps",
            )
        )
    for row in ledger_rows:
        if row.get("write_mode") != "realtime":
            continue
        strategy = {"trader-mim-nb": "MIM", "trader-yank": "YANK"}.get(
            row.get("trader_id")
        )
        if strategy is None:
            continue
        try:
            minute = timestamp(row["timestamp"]).replace(second=0, microsecond=0)
            candidates = [
                n
                for n, t in enumerate(trades)
                if t["strategy"] == strategy
                and timestamp(t["exit_timestamp"]).replace(second=0, microsecond=0)
                == minute
                and t["direction"] == (1 if row["direction"] in ("L", "LONG") else -1)
            ]
        except (ValueError, KeyError, TypeError):
            candidates = []
        linked = trades[candidates[0]] if len(candidates) == 1 else None
        outputs.append(
            dict(
                source="realtime_ledger",
                source_id=row.get("id"),
                original=row,
                status=(
                    "UNIQUE_INFERRED_EXIT_MINUTE_SIDE_STRATEGY"
                    if linked
                    else ("AMBIGUOUS" if candidates else "UNMATCHED")
                ),
                candidates=candidates,
                broker_trade=linked,
                reported_minus_broker_gross=(
                    float(row["pnl"]) - linked["reconstructed_gross"]
                    if linked
                    and linked["reconstructed_gross"] is not None
                    and row.get("pnl") is not None
                    else None
                ),
                warning="missing account/quantity metadata cannot be invented",
            )
        )
    return outputs


def realized_economics(rows):
    if any(row.get("evidence_valid") is False for row in rows):
        return dict(status="INVALID_EVIDENCE")
    valid = sorted(
        [r for r in rows if r["included"]], key=lambda r: timestamp(r["timestamp"])
    )
    if any(r["actual_total_cost"] is None for r in valid):
        return dict(status="UNKNOWN_COSTS")
    daily = defaultdict(float)
    running = 0.0
    series = []
    for row in valid:
        value = (row["gross_realized_pnl"] or 0.0) - row["actual_total_cost"]
        running += value
        series.append((row["timestamp"], running))
        day = str(
            timestamp(row["timestamp"]).astimezone(ZoneInfo("America/New_York")).date()
        )
        daily[day] += value
    epochs = {(r["account"], r["account_epoch"]) for r in valid}
    if any(not known_epoch(r["account_epoch"]) for r in valid) or len(epochs) > 1:
        return dict(
            status="PER_SESSION_CONDITIONAL_ONLY",
            cashflow_by_observed_day=dict(daily),
            reason="Unknown or multiple account epochs; no cross-session drawdown, recovery or capital curve.",
        )
    return dict(
        status="PARTIAL_REALIZED_CASHFLOW_ONLY",
        cashflow_by_observed_day=dict(daily),
        observed_day_statistics=metrics(list(daily.values())),
        realized_path=path_risk(series) if series else None,
        capital_sensitivity={
            str(c): metrics(list(daily.values()), c) for c in (5000.0, 10000.0, 20000.0)
        },
        limitation="Missing zero-trade sessions and intraday marks. Sharpe on observed days is descriptive only; realized drawdown omits unrealized excursions and is not a risk bound.",
    )
