"""Exact-first reconciliation of existing MIM and GAP execution records."""

from __future__ import annotations

import csv
import hashlib
import json

import numpy as np
import pandas as pd

from .artifacts import input_path

ET = "America/New_York"
ID_NAMESPACE = "pf-improvement-v1"


def pseudonym(kind: str, value: object) -> str | None:
    if value is None or pd.isna(value) or str(value) == "FAIL":
        return None
    return hashlib.sha256(f"{ID_NAMESPACE}|{kind}|{value}".encode()).hexdigest()[:20]


def _side(value: object) -> str:
    number = int(float(value))
    if number == 0:
        return "BUY"
    if number == 1:
        return "SELL"
    raise ValueError("Unknown execution side")


KNOWN_CHAIN_SCARS = {"data/gap_fade/decisions.csv": 30}


def _chain_check(relative: str) -> tuple[int, int | None, str | None]:
    with input_path(relative).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "chain" not in reader.fieldnames:
            return 0, None, None
        fields = [field for field in reader.fieldnames if field != "chain"]
        head, rows, first_bad, first_key = "GENESIS", 0, None, None
        for row in reader:
            rows += 1
            payload = "|".join(str(row.get(field, "")) for field in fields)
            expected = hashlib.sha256((head + "|" + payload).encode()).hexdigest()[:16]
            stored = row.get("chain")
            if stored != expected and first_bad is None:
                first_bad, first_key = rows, row.get(fields[0])
            head = stored if stored != expected else expected
        return rows, first_bad, first_key


def _chain_row(
    name: str,
    relative: str,
    frame: pd.DataFrame,
    time_column: str | None,
    era: str,
    duplicate_key: str | None = None,
) -> dict[str, object]:
    chain = frame["chain"] if "chain" in frame else pd.Series(dtype=str)
    timestamps = (
        pd.to_datetime(frame[time_column], utc=True, errors="coerce")
        if time_column
        else pd.Series(dtype="datetime64[ns, UTC]")
    )
    rows, bad_row, bad_key = _chain_check(relative)
    duplicates = (
        sorted(
            frame.loc[frame[duplicate_key].duplicated(False), duplicate_key]
            .astype(str)
            .unique()
        )
        if duplicate_key and duplicate_key in frame
        else []
    )
    known_break = KNOWN_CHAIN_SCARS.get(relative) == bad_row
    known_duplicate = relative == "data/gap_fade/trades.csv" and duplicates == [
        "2026-06-25"
    ]
    status = "VALID"
    if bad_row is not None:
        status = "KNOWN_SCAR" if known_break else "BROKEN_UNREGISTERED"
    if duplicates:
        status = "KNOWN_SCAR" if known_duplicate and status == "VALID" else status
    return {
        "source": name,
        "configuration_era": era,
        "rows": len(frame),
        "chain_present": int(chain.notna().sum()),
        "chain_missing": int(chain.isna().sum()) if len(chain) else len(frame),
        "duplicate_chain_values": int(chain.dropna().duplicated().sum()),
        "duplicate_business_keys": "|".join(duplicates) if duplicates else None,
        "first_chain_break_row": bad_row,
        "first_chain_break_key": bad_key,
        "chain_valid_from_genesis": bad_row is None,
        "integrity_status": status,
        "first_observation_utc": (
            timestamps.min().isoformat()
            if len(timestamps) and timestamps.notna().any()
            else None
        ),
        "last_observation_utc": (
            timestamps.max().isoformat()
            if len(timestamps) and timestamps.notna().any()
            else None
        ),
        "chain_interpretation": "ordered SHA-256 prefix; integrity does not establish completeness",
    }


def _era(strategy: str, day: str) -> str:
    if strategy == "GAP":
        return "pre_ledger_hardening" if day < "2026-08-21" else "ledger_hardened"
    if day < "2026-06-25":
        return "initial_live"
    if day < "2026-07-07":
        return "catstop250_dll500"
    if day < "2026-07-29":
        return "dll1000_ops_hardening"
    return "current_risk_mechanics"


def _mim_records() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    orders = pd.read_csv(input_path("data/mim_nb/orders.csv"), dtype={"order_id": str})
    local_trades = pd.read_csv(input_path("data/mim_nb/trades.csv"))
    decisions = pd.read_csv(input_path("data/mim_nb/decisions.csv"))
    fills = sorted(
        json.loads(input_path("data/mim_nb/projectx_fills.json").read_text()),
        key=lambda row: row["creationTimestamp"],
    )
    if any(row.get("voided") for row in fills):
        raise ValueError("Voided broker fill cannot enter reconciliation")

    local_fills = orders[(orders.event == "FILL") & (orders.order_id != "FAIL")].copy()
    local_fill_ids = set(local_fills.order_id.dropna())
    broker_order_ids = {str(row["orderId"]) for row in fills}
    for broker in fills:
        order_id = str(broker["orderId"])
        matches = local_fills[local_fills.order_id == order_id]
        if len(matches) > 1:
            raise ValueError("Duplicate local fill for saved broker order ID")
        if len(matches) == 1:
            local = matches.iloc[0]
            if (
                _side(local.side) != _side(broker["side"])
                or int(local["size"]) != int(broker["size"])
                or not np.isclose(
                    float(local.price), float(broker["price"]), atol=1e-8, rtol=0
                )
            ):
                raise ValueError("Exact order ID conflicts on side, size, or price")
    events: list[dict[str, object]] = []
    for row in orders.itertuples(index=False):
        events.append(
            {
                "strategy": "MIM",
                "venue": "ProjectX",
                "source": "local_order_log",
                "event_timestamp_utc": row.ts_utc,
                "event_type": row.event,
                "order_id": pseudonym("projectx-order", row.order_id),
                "fill_id": None,
                "side": _side(row.side) if pd.notna(row.side) else None,
                "size": int(row.size),
                "price": float(row.price) if pd.notna(row.price) else None,
                "costs": None,
                "exact_order_match": row.order_id in broker_order_ids,
                "evidence_status": "local_record",
            }
        )
    for row in fills:
        events.append(
            {
                "strategy": (
                    "MIM"
                    if str(row["orderId"]) in local_fill_ids
                    else "UNATTRIBUTED_SHARED_ACCOUNT"
                ),
                "venue": "ProjectX",
                "source": "saved_broker_export",
                "event_timestamp_utc": pd.Timestamp(
                    row["creationTimestamp"]
                ).isoformat(),
                "event_type": "FILL",
                "order_id": pseudonym("projectx-order", row["orderId"]),
                "fill_id": pseudonym("projectx-fill", row["id"]),
                "side": _side(row["side"]),
                "size": int(row["size"]),
                "price": float(row["price"]),
                "costs": float(row["fees"]) + float(row["commissions"]),
                "exact_order_match": str(row["orderId"]) in local_fill_ids,
                "evidence_status": (
                    "exact_order_id"
                    if str(row["orderId"]) in local_fill_ids
                    else "unmatched_broker_candidate"
                ),
            }
        )

    broker_pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    open_fill: dict[str, object] | None = None
    for row in fills:
        if open_fill is None:
            open_fill = row
            continue
        if (
            row["accountId"] != open_fill["accountId"]
            or row["contractId"] != open_fill["contractId"]
            or int(row["size"]) != int(open_fill["size"])
            or int(row["side"]) == int(open_fill["side"])
        ):
            raise ValueError("Ambiguous saved-export round trip")
        broker_pairs.append((open_fill, row))
        open_fill = None
    if open_fill is not None:
        raise ValueError("Unpaired saved broker fill")

    round_trips: list[dict[str, object]] = []
    matched_local_indices: set[int] = set()
    for open_fill, row in broker_pairs:
        direction = 1 if int(open_fill["side"]) == 0 else -1
        gross = (
            (float(row["price"]) - float(open_fill["price"]))
            * direction
            * 2.0
            * int(row["size"])
        )
        if row.get("profitAndLoss") is None or not np.isclose(
            gross, float(row["profitAndLoss"]), atol=1e-8, rtol=0
        ):
            raise ValueError("Saved-export P&L does not reconcile to fills")
        costs = sum(
            float(x.get("fees") or 0) + float(x.get("commissions") or 0)
            for x in (open_fill, row)
        )
        exact = (
            str(open_fill["orderId"]) in local_fill_ids
            and str(row["orderId"]) in local_fill_ids
        )
        entry_et = pd.Timestamp(open_fill["creationTimestamp"]).tz_convert(ET)
        exit_et = pd.Timestamp(row["creationTimestamp"]).tz_convert(ET)
        day = entry_et.strftime("%Y-%m-%d")
        modeled = None
        strategy = "UNATTRIBUTED_SHARED_ACCOUNT"
        attribution = "no_local_order_ids"
        if exact:
            candidates = local_trades[
                (local_trades.day == day)
                & (local_trades.dir == direction)
                & (local_trades.entry_t == entry_et.strftime("%H:%M"))
                & (local_trades.exit_t == exit_et.strftime("%H:%M"))
            ]
            if len(candidates) != 1:
                raise ValueError("Exact MIM broker pair lacks one unique local trade")
            local_index = int(candidates.index[0])
            matched_local_indices.add(local_index)
            modeled = float(candidates.iloc[0].pnl_usd)
            strategy = "MIM"
            attribution = "exact_orders_and_unique_local_trade"
        round_trips.append(
            {
                "strategy": strategy,
                "venue": "ProjectX",
                "day_et": day,
                "configuration_era": _era("MIM", day),
                "entry_order_id": pseudonym("projectx-order", open_fill["orderId"]),
                "exit_order_id": pseudonym("projectx-order", row["orderId"]),
                "entry_fill_id": pseudonym("projectx-fill", open_fill["id"]),
                "exit_fill_id": pseudonym("projectx-fill", row["id"]),
                "direction": direction,
                "size": int(row["size"]),
                "entry_timestamp_utc": pd.Timestamp(
                    open_fill["creationTimestamp"]
                ).isoformat(),
                "exit_timestamp_utc": pd.Timestamp(
                    row["creationTimestamp"]
                ).isoformat(),
                "entry_price": float(open_fill["price"]),
                "exit_price": float(row["price"]),
                "broker_gross": gross,
                "complete_costs": True,
                "costs": costs,
                "broker_net": gross - costs,
                "modeled_gross": modeled,
                "signed_difference": gross - modeled if modeled is not None else None,
                "join_method": (
                    "exact_order_id" if exact else "unmatched_broker_candidate"
                ),
                "strategy_side_size_validated": exact,
                "causal_decomposition_available": False,
                "record_only_error": False,
                "attribution_status": attribution,
            }
        )

    for local_index, local in local_trades.iterrows():
        if local_index in matched_local_indices:
            continue
        direction = int(local.dir)
        day = str(local.day)
        modeled = float(local.pnl_usd)
        round_trips.append(
            {
                "strategy": "MIM",
                "venue": "ProjectX",
                "day_et": day,
                "configuration_era": _era("MIM", day),
                "entry_order_id": None,
                "exit_order_id": None,
                "entry_fill_id": None,
                "exit_fill_id": None,
                "direction": direction,
                "size": 1,
                "entry_timestamp_utc": None,
                "exit_timestamp_utc": None,
                "entry_price": None,
                "exit_price": None,
                "broker_gross": None,
                "complete_costs": False,
                "costs": None,
                "broker_net": None,
                "modeled_gross": modeled,
                "signed_difference": None,
                "join_method": "no_saved_broker_pair",
                "strategy_side_size_validated": False,
                "causal_decomposition_available": False,
                "record_only_error": False,
                "attribution_status": "local_reference_trade_only",
            }
        )

    coverage = [
        _chain_row(
            "mim_decisions",
            "data/mim_nb/decisions.csv",
            decisions,
            "ts_et",
            "mixed_history_local_schema",
        ),
        _chain_row(
            "mim_orders", "data/mim_nb/orders.csv", orders, "ts_utc", "local_id_logging"
        ),
        _chain_row(
            "mim_trades",
            "data/mim_nb/trades.csv",
            local_trades,
            None,
            "mixed_history_trade_log",
        ),
        {
            "source": "mim_saved_broker_export",
            "configuration_era": "saved_export_2026-08-13_through_2026-08-28",
            "rows": len(fills),
            "chain_present": 0,
            "chain_missing": len(fills),
            "duplicate_chain_values": 0,
            "duplicate_business_keys": None,
            "first_chain_break_row": None,
            "first_chain_break_key": None,
            "chain_valid_from_genesis": None,
            "integrity_status": "NOT_APPLICABLE",
            "first_observation_utc": pd.Timestamp(
                fills[0]["creationTimestamp"]
            ).isoformat(),
            "last_observation_utc": pd.Timestamp(
                fills[-1]["creationTimestamp"]
            ).isoformat(),
            "chain_interpretation": "broker IDs present; no local hash chain",
        },
    ]
    return pd.DataFrame(events), pd.DataFrame(round_trips), coverage


def _gap_records() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    fills = pd.read_csv(
        input_path("data/gap_fade/fills.csv"), dtype={"entry_id": str, "exit_id": str}
    )
    decisions = pd.read_csv(input_path("data/gap_fade/decisions.csv"))
    trades = pd.read_csv(input_path("data/gap_fade/trades.csv"))
    events = []
    trips = []
    for row in fills.itertuples(index=False):
        direction = 1 if row.dir == "L" else -1
        gross = (
            (float(row.exit_exec) - float(row.entry_exec))
            * direction
            * 2.0
            * int(row.qty)
        )
        if not np.isclose(gross, float(row.realized_pnl_usd), atol=1e-8, rtol=0):
            raise ValueError("GAP fill ledger gross mismatch")
        for role, oid, price, side in (
            (
                "ENTRY",
                row.entry_id,
                row.entry_exec,
                "BUY" if direction == 1 else "SELL",
            ),
            ("EXIT", row.exit_id, row.exit_exec, "SELL" if direction == 1 else "BUY"),
        ):
            events.append(
                {
                    "strategy": "GAP",
                    "venue": "TradeStation_SIM",
                    "source": "fill_reconciliation_log",
                    "event_timestamp_utc": None,
                    "event_type": role + "_FILL",
                    "order_id": pseudonym("tradestation-order", oid),
                    "fill_id": None,
                    "side": side,
                    "size": int(row.qty),
                    "price": float(price),
                    "costs": None,
                    "exact_order_match": True,
                    "evidence_status": "exact_order_id_gross_only",
                }
            )
        trips.append(
            {
                "strategy": "GAP",
                "venue": "TradeStation_SIM",
                "day_et": row.date_et,
                "configuration_era": _era("GAP", str(row.date_et)),
                "entry_order_id": pseudonym("tradestation-order", row.entry_id),
                "exit_order_id": pseudonym("tradestation-order", row.exit_id),
                "entry_fill_id": None,
                "exit_fill_id": None,
                "direction": direction,
                "size": int(row.qty),
                "entry_timestamp_utc": None,
                "exit_timestamp_utc": None,
                "entry_price": float(row.entry_exec),
                "exit_price": float(row.exit_exec),
                "broker_gross": gross,
                "complete_costs": False,
                "costs": None,
                "broker_net": None,
                "modeled_gross": float(row.modeled_pnl_usd),
                "signed_difference": float(row.delta_usd),
                "join_method": "exact_order_id",
                "strategy_side_size_validated": True,
                "causal_decomposition_available": False,
                "record_only_error": False,
                "attribution_status": "exact_fill_log_ids_gross_only",
            }
        )
    coverage = [
        _chain_row(
            "gap_decisions",
            "data/gap_fade/decisions.csv",
            decisions,
            None,
            "mixed_history_decision_log",
        ),
        _chain_row(
            "gap_fills",
            "data/gap_fade/fills.csv",
            fills,
            None,
            "id_reconciliation_gross_only",
        ),
        _chain_row(
            "gap_trades",
            "data/gap_fade/trades.csv",
            trades,
            None,
            "mixed_history_trade_log",
            "date_et",
        ),
    ]
    return pd.DataFrame(events), pd.DataFrame(trips), coverage


def classify_gate(
    *,
    complete_current_coverage: bool,
    recurring_current_economic_defect: bool,
    exact_causality: bool,
    unchanged_intended_behavior: bool,
) -> str:
    """Apply the declared three-state gate without estimating missing evidence."""
    if (
        recurring_current_economic_defect
        and exact_causality
        and complete_current_coverage
        and unchanged_intended_behavior
    ):
        return "CURRENT_REPAIR_CANDIDATE"
    if (
        complete_current_coverage
        and exact_causality
        and not recurring_current_economic_defect
        and unchanged_intended_behavior
    ):
        return "NO_REPAIRABLE_MECHANISM"
    return "INSUFFICIENT_CAUSAL_EVIDENCE"


def build_execution() -> dict[str, object]:
    mim_events, mim_trips, mim_coverage = _mim_records()
    gap_events, gap_trips, gap_coverage = _gap_records()
    events = pd.DataFrame.from_records(
        mim_events.to_dict("records") + gap_events.to_dict("records")
    )
    trips = pd.DataFrame.from_records(
        mim_trips.to_dict("records") + gap_trips.to_dict("records")
    )
    coverage = pd.DataFrame(mim_coverage + gap_coverage)
    attributed_mim = mim_trips[mim_trips.strategy == "MIM"]
    exact_mim = int((attributed_mim.join_method == "exact_order_id").sum())
    saved_pairs = int((mim_trips.join_method != "no_saved_broker_pair").sum())
    unattributed_pairs = int(
        (mim_trips.strategy == "UNATTRIBUTED_SHARED_ACCOUNT").sum()
    )
    chain_findings = coverage[coverage.integrity_status == "BROKEN_UNREGISTERED"]
    strategy_trips = trips[trips.strategy.isin(["MIM", "GAP"])]
    current_trips = strategy_trips[
        strategy_trips.configuration_era.isin(
            ["current_risk_mechanics", "ledger_hardened"]
        )
    ]
    complete_current_coverage = bool(
        len(current_trips)
        and current_trips.join_method.eq("exact_order_id").all()
        and current_trips.complete_costs.all()
        and current_trips.entry_timestamp_utc.notna().all()
        and current_trips.exit_timestamp_utc.notna().all()
    )
    exact_causality = bool(
        len(current_trips) and current_trips.causal_decomposition_available.all()
    )
    unchanged_intended_behavior = bool(
        len(current_trips)
        and current_trips.strategy_side_size_validated.all()
        and current_trips.attribution_status.eq(
            "exact_orders_and_unique_local_trade"
        ).all()
    )
    causal_current = current_trips[
        current_trips.causal_decomposition_available
        & current_trips.complete_costs
        & current_trips.signed_difference.notna()
    ]
    recurring_current_economic_defect = bool(
        len(causal_current) >= 2 and (causal_current.signed_difference < 0).all()
    )
    gate_verdict = classify_gate(
        complete_current_coverage=complete_current_coverage,
        recurring_current_economic_defect=recurring_current_economic_defect,
        exact_causality=exact_causality,
        unchanged_intended_behavior=unchanged_intended_behavior,
    )
    reasons = [
        f"MIM saved export contains {saved_pairs} complete shared-account round trips; {exact_mim} have both order IDs and one unique local MIM trade, while {unattributed_pairs} remain unattributed.",
        "No executable quote series binds decision-to-arrival and arrival-to-fill movement for MIM.",
        "GAP has exact entry/exit order IDs and gross differences, but no fill timestamps, executable quotes, or complete costs.",
        f"{len(chain_findings)} MIM source chains have unregistered historical breaks; chain validity does not establish record completeness.",
        "Local trade rows do not carry broker order IDs, so missing broker observations remain explicit rather than being repaired by inferred price matching.",
    ]
    gate = {
        "verdict": gate_verdict,
        "stops_later_stages": gate_verdict == "CURRENT_REPAIR_CANDIDATE",
        "current_repair_specification": None,
        "complete_current_coverage": complete_current_coverage,
        "exact_causality": exact_causality,
        "recurring_current_economic_defect": recurring_current_economic_defect,
        "unchanged_intended_behavior": unchanged_intended_behavior,
        "reasons": reasons,
        "mim_local_reference_trades": len(attributed_mim),
        "mim_saved_shared_account_round_trips": saved_pairs,
        "mim_unattributed_shared_account_round_trips": unattributed_pairs,
        "mim_exact_order_round_trips": exact_mim,
        "gap_round_trips": len(gap_trips),
        "broker_net_round_trips_all_shared_account": int(
            trips.broker_net.notna().sum()
        ),
        "broker_net_round_trips_attributed_to_mim": int(
            attributed_mim.broker_net.notna().sum()
        ),
        "causally_decomposable_round_trips": int(
            trips.causal_decomposition_available.sum()
        ),
        "recoverable_dollars": None,
    }
    return {"events": events, "round_trips": trips, "coverage": coverage, "gate": gate}
