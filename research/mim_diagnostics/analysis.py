"""Accounting of recorded executions only; no signal or strategy engine."""

import math
import numpy as np
import pandas as pd
from research.mim_comparison.data import load
from .artifacts import permitted, read_csv
from research.mim_lifecycle.analysis import (
    select_inputs,
    validate_session,
    trace_trade,
    timestamp,
)

ATOL = 1e-8
DIMENSIONS = ("direction", "exit_reason", "year", "entry_order", "entry_half_hour")


def close(a, b, label):
    if not np.allclose(a, b, rtol=0, atol=ATOL):
        raise ValueError(f"{label} reconciliation failed")


def half_hour(value):
    """Completed labels 09:31..10:00 map to [09:30,10:00)."""
    t = timestamp(value)
    minute = t.hour * 60 + t.minute
    if not 571 <= minute <= 960:
        raise ValueError("Event outside cash-session completed-minute grid")
    start = 570 + ((minute - 571) // 30) * 30
    return f"{start//60:02d}:{start%60:02d}-{(start+30)//60:02d}:{(start+30)%60:02d}"


def reconstruct(ledger, daily, expected):
    """Replay execution accounting in CSV row order, never simulate decisions."""
    required = {
        "day",
        "contract",
        "event_timestamp",
        "modeled_fill_timestamp",
        "fill_time_basis",
        "fill",
        "quantity",
        "position_after",
        "costs",
        "reason",
        "source_row",
    }
    if not required <= set(ledger):
        raise ValueError("Missing ledger columns")
    if ledger.drop(columns="source_row").duplicated().any():
        raise ValueError("Duplicate execution rows")
    if not set(zip(ledger.day, ledger.contract)) <= set(zip(daily.day, daily.contract)):
        raise ValueError("Execution missing daily join")
    events, trades = [], []
    for (day, contract), group in ledger.groupby(["day", "contract"], sort=True):
        p, entry, prior_direction, sequence = 0, None, None, 0
        last = None
        for raw in group.to_dict("records"):
            t = timestamp(raw["event_timestamp"])
            if t.strftime("%Y-%m-%d") != day or (last is not None and t < last):
                raise ValueError("Nonchronological execution order or day mismatch")
            last = t
            after, qty = raw["position_after"], raw["quantity"]
            if after not in (-1, 0, 1) or after == p or qty != abs(after - p):
                raise ValueError("Invalid position transition/turnover")
            if not np.isfinite([raw["fill"], raw["costs"]]).all() or raw["fill"] <= 0:
                raise ValueError("Invalid execution numeric")
            close(raw["costs"], qty * 1.12, "Event fee")
            basis = raw["fill_time_basis"]
            if basis == "intrabar_unknown_within_event_minute":
                if (
                    raw["reason"] != "CAT_STOP"
                    or after != 0
                    or pd.notna(raw["modeled_fill_timestamp"])
                ):
                    raise ValueError("Invalid uncertain stop timestamp")
            elif basis in ("bar_open", "session_close_proxy"):
                fill_t = timestamp(raw["modeled_fill_timestamp"])
                target = t - pd.Timedelta(minutes=1) if basis == "bar_open" else t
                if fill_t != target:
                    raise ValueError("Fill/event timestamp mismatch")
            else:
                raise ValueError("Unknown fill time basis")
            if p:
                if raw["reason"] not in ("REVERSAL", "CAT_STOP", "EOD_CLOSE_PROXY"):
                    raise ValueError("Unexpected primary exit label")
                gross = p * (raw["fill"] - entry["entry_fill"]) * 2
                trade = dict(
                    entry,
                    exit_event_timestamp=raw["event_timestamp"],
                    exit_fill_timestamp=raw["modeled_fill_timestamp"],
                    exit_fill_time_basis=basis,
                    exit_fill=raw["fill"],
                    exit_reason=raw["reason"],
                    gross=gross,
                    costs=2.24,
                    net=gross - 2.24,
                )
                trades.append(trade)
            else:
                gross = 0.0
            transition = (
                "true_reversal"
                if p and after
                else (
                    "opposite_entry_after_flat"
                    if after
                    and prior_direction is not None
                    and after != prior_direction
                    else "entry_from_flat" if after else "exit_to_flat"
                )
            )
            events.append(
                dict(
                    raw,
                    source_row=raw["source_row"],
                    position_before=p,
                    transition=transition,
                    realized_gross=gross,
                    realized_net=gross - raw["costs"],
                    event_half_hour=half_hour(t),
                )
            )
            if after:
                sequence += 1
                entry = dict(
                    trade_id=f"{day}-{sequence:02d}",
                    day=day,
                    contract=contract,
                    direction=after,
                    quantity=1,
                    entry_event_timestamp=raw["event_timestamp"],
                    entry_fill_timestamp=raw["modeled_fill_timestamp"],
                    entry_fill=raw["fill"],
                    entry_original_reason=raw["reason"],
                    entry_transition=transition,
                    year=int(day[:4]),
                    entry_order="first" if sequence == 1 else "subsequent",
                    entry_half_hour=half_hour(t),
                )
                prior_direction = after
            p = after
        if p:
            raise ValueError("Unclosed session position")
    result = pd.DataFrame(trades)
    if len(result) != len(expected):
        raise ValueError("Reconstructed trade count mismatch")
    keys = ["day", "contract", "entry_event_timestamp"]
    merged = result.merge(
        expected,
        on=keys,
        suffixes=("", "_source"),
        validate="one_to_one",
        how="outer",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Missing source trade join")
    for col in (
        "direction",
        "quantity",
        "entry_fill",
        "exit_fill",
        "gross",
        "costs",
        "net",
    ):
        close(merged[col], merged[col + "_source"], col)
    for col in (
        "entry_fill_timestamp",
        "exit_event_timestamp",
        "exit_fill_timestamp",
        "exit_fill_time_basis",
        "exit_reason",
    ):
        if not merged[col].fillna("").eq(merged[col + "_source"].fillna("")).all():
            raise ValueError(f"Trade {col} mismatch")
    return pd.DataFrame(events), result


def attribution(trades, accrual):
    rows = []
    total_losses = float(-trades.net.clip(upper=0).sum())
    for dimension in DIMENSIONS:
        for key, g in trades.groupby(dimension, sort=True):
            loss = float(-g.net.clip(upper=0).sum())
            rows.append(
                dict(
                    basis="entry_cohort",
                    dimension=dimension,
                    group=str(key),
                    count=len(g),
                    gross=float(g.gross.sum()),
                    costs=float(g.costs.sum()),
                    net=float(g.net.sum()),
                    losing_trade_dollars=loss,
                    losing_trade_share=loss / total_losses if total_losses else 0,
                    exposure_minutes_upper=float(g.exposure_minutes_upper.sum()),
                    exposure_minutes_lower=float(g.exposure_minutes_lower.sum()),
                )
            )
    for dimension in (*DIMENSIONS, "accrual_half_hour"):
        for key, g in accrual.groupby(dimension, sort=True):
            # Signed actual minute contributions of eventual losing trades.
            loss = float(-g.eventual_losers_net_contribution.sum())
            rows.append(
                dict(
                    basis="minute_accrual",
                    dimension=dimension,
                    group=str(key),
                    count=int(g.entry_count.sum()),
                    gross=float(g.gross.sum()),
                    costs=float(g.costs.sum()),
                    net=float(g.net.sum()),
                    losing_trade_dollars=loss,
                    losing_trade_share=loss / total_losses if total_losses else 0,
                    exposure_minutes_upper=float(g.exposure_minutes_upper.sum()),
                    exposure_minutes_lower=float(g.exposure_minutes_lower.sum()),
                )
            )
    result = pd.DataFrame(rows)
    buckets = [
        f"{m//60:02d}:{m%60:02d}-{(m+30)//60:02d}:{(m+30)%60:02d}"
        for m in range(570, 960, 30)
    ]
    for basis, dimension in [
        ("entry_cohort", "entry_half_hour"),
        ("minute_accrual", "entry_half_hour"),
        ("minute_accrual", "accrual_half_hour"),
    ]:
        present = set(
            result.loc[
                (result.basis == basis) & (result.dimension == dimension), "group"
            ]
        )
        for bucket in buckets:
            if bucket not in present:
                row = {c: 0.0 for c in result.columns}
                row.update(basis=basis, dimension=dimension, group=bucket)
                result.loc[len(result)] = row
    return result.sort_values(["basis", "dimension", "group"]).reset_index(drop=True)


def drawdown(frame, column, clock):
    values = frame[column].to_numpy(float)
    peaks = np.maximum.accumulate(np.r_[0.0, values])[1:]
    depth = values - peaks
    trough = int(np.argmin(depth))
    start = next((i for i in range(trough - 1, -1, -1) if depth[i] == 0), -1)
    recovery = next((i for i in range(trough + 1, len(values)) if depth[i] == 0), None)
    # Longest underwater sample run, including unresolved final episode.
    longest = current = 0
    for d in depth:
        current = current + 1 if d < 0 else 0
        longest = max(longest, current)
    return dict(
        depth=float(-depth[trough]),
        peak_timestamp=str(frame.iloc[start][clock]) if start >= 0 else "initial_zero",
        trough_timestamp=str(frame.iloc[trough][clock]),
        recovery_timestamp=(
            str(frame.iloc[recovery][clock]) if recovery is not None else None
        ),
        peak_to_trough_samples=trough - start,
        peak_to_recovery_samples=recovery - start if recovery is not None else None,
        longest_underwater_samples=longest,
        unrecovered_at_end=bool(depth[-1] < 0),
    )


def analyze(source, data):
    raw_daily = read_csv(permitted(source / "daily.csv"))
    raw_trades = read_csv(permitted(source / "trades.csv"))
    expected, daily = select_inputs(raw_trades, raw_daily)
    raw_ledger = read_csv(permitted(source / "ledger.csv"))
    raw_ledger["source_row"] = np.arange(len(raw_ledger))
    ledger = raw_ledger[
        (raw_ledger.arm == "A")
        & (raw_ledger.delay == 2)
        & (raw_ledger.cost_scenario == 2.24)
    ].copy()
    if len(ledger) != 1595:
        raise ValueError("Pinned primary execution count mismatch")
    events, trades = reconstruct(ledger, daily, expected)
    bars = load(permitted(data), "end")
    keys = set(zip(daily.day, daily.contract))
    selected = bars[pd.MultiIndex.from_frame(bars[["day", "contract"]]).isin(keys)]
    sessions = {
        k: validate_session(g, *k)
        for k, g in selected.groupby(["day", "contract"], sort=True)
    }
    if set(sessions) != keys:
        raise ValueError("Missing selected session bar join")
    grids = []
    for (day, contract), group in sessions.items():
        grid = group[["day", "contract", "timestamp", "close"]].copy()
        grids.append(grid)
    minute = pd.concat(grids, ignore_index=True)
    contributions = []
    excursions = []
    for t in trades.to_dict("records"):
        b = sessions[t["day"], t["contract"]]
        rows, excursion = trade_rows(t, b)
        contributions.extend(rows)
        excursions.append(excursion)
    accrual = pd.DataFrame(contributions)
    excursions = pd.DataFrame(excursions)
    exp = (
        accrual.groupby("trade_id")[
            ["exposure_minutes_upper", "exposure_minutes_lower"]
        ]
        .sum()
        .reset_index()
    )
    trades = trades.merge(exp, on="trade_id", validate="one_to_one").merge(
        excursions, on="trade_id", validate="one_to_one"
    )
    agg = (
        accrual.groupby(["day", "contract", "timestamp"])[
            [
                "gross",
                "costs",
                "net",
                "exposure_minutes_upper",
                "exposure_minutes_lower",
            ]
        ]
        .sum()
        .reset_index()
    )
    minute = minute.merge(
        agg, on=["day", "contract", "timestamp"], how="left", validate="one_to_one"
    )
    cols = ["gross", "costs", "net", "exposure_minutes_upper", "exposure_minutes_lower"]
    minute[cols] = minute[cols].fillna(0)
    position = {}
    for (day, contract), group in events.groupby(["day", "contract"], sort=True):
        for row in group.itertuples():
            position[day, contract, timestamp(row.event_timestamp)] = row.position_after
    minute["position_after"] = [
        position.get((r.day, r.contract, r.timestamp), np.nan)
        for r in minute.itertuples()
    ]
    minute["position_after"] = (
        minute.groupby("day").position_after.ffill().fillna(0).astype(int)
    )
    minute["session_marked_gross"] = minute.groupby("day").gross.cumsum()
    minute["session_marked_net"] = minute.groupby("day").net.cumsum()
    minute["marked_equity_net"] = minute.net.cumsum()
    minute["accrual_half_hour"] = [half_hour(t) for t in minute.timestamp]
    realized = events.copy()
    realized["timestamp"] = pd.to_datetime(
        realized.event_timestamp, utc=True
    ).dt.tz_convert("America/New_York")
    rg = (
        realized.groupby(["day", "contract", "timestamp"])
        .realized_gross.sum()
        .reset_index()
    )
    minute = minute.merge(
        rg, on=["day", "contract", "timestamp"], how="left", validate="one_to_one"
    )
    minute["realized_gross"] = minute.realized_gross.fillna(0)
    minute["session_realized_gross"] = minute.groupby("day").realized_gross.cumsum()
    minute["session_costs"] = minute.groupby("day").costs.cumsum()
    minute["session_realized_net"] = (
        minute.session_realized_gross - minute.session_costs
    )
    minute["unrealized_gross"] = (
        minute.session_marked_gross - minute.session_realized_gross
    )
    parts = attribution(trades, accrual)
    for table in (
        minute,
        trades,
        events.rename(columns={"realized_gross": "gross", "realized_net": "net"}),
    ):
        for field in ("gross", "costs", "net"):
            aligned = table.groupby("day")[field].sum().reindex(daily.day, fill_value=0)
            close(aligned.to_numpy(), daily[field].to_numpy(), f"daily {field}")
    close(
        minute.groupby("day").exposure_minutes_upper.sum().reindex(daily.day),
        daily.exposure_contract_minutes,
        "Exposure",
    )
    for _, g in parts.groupby(["basis", "dimension"]):
        for field in ("gross", "costs", "net"):
            close(g[field].sum(), daily[field].sum(), f"Partition {field}")
        close(g["count"].sum(), len(trades), "Partition counts")
    daily["marked_equity_net"] = daily.net.cumsum()
    top = daily.sort_values(["net", "day"], ascending=[False, True]).head(67)
    close(top.net.sum(), 39552.92, "Best 67")
    exit_pins = {
        "CAT_STOP": (71, -35331.04),
        "EOD_CLOSE_PROXY": (723, 59824.48),
        "REVERSAL": (7, -2603.68),
    }
    for label, (count, net) in exit_pins.items():
        g = trades[trades.exit_reason == label]
        if len(g) != count:
            raise ValueError("Pinned exit count mismatch")
        close(g.net.sum(), net, "Pinned exit payoff")
    scenarios = recorded_scenarios(raw_daily)
    summary = dict(
        descriptive_only=True,
        primary=dict(arm="A", delay=2, roundtrip_cost=2.24, quantity=1, point_value=2),
        sessions=len(daily),
        trades=len(trades),
        events=len(events),
        flat_sessions=int((daily.costs == 0).sum()),
        gross=float(daily.gross.sum()),
        costs=float(daily.costs.sum()),
        net=float(daily.net.sum()),
        best_67_net=float(top.net.sum()),
        remainder_net=float(daily.net.sum() - top.net.sum()),
        best_67_share=float(top.net.sum() / daily.net.sum()),
        best_67_days=top[["day", "net"]].to_dict("records"),
        top_5_percent_trades_net=float(
            trades.nlargest(math.ceil(len(trades) * 0.05), "net").net.sum()
        ),
        losing_trade_dollars=float(-trades.net.clip(upper=0).sum()),
        exposure_minutes_upper=float(minute.exposure_minutes_upper.sum()),
        exposure_minutes_lower=float(minute.exposure_minutes_lower.sum()),
        uncertain_stop_trades=int(trades.coverage.eq("incomplete_stop_interval").sum()),
        transitions=events.transition.value_counts().to_dict(),
        drawdown_minute=drawdown(minute, "marked_equity_net", "timestamp"),
        drawdown_daily=drawdown(daily, "marked_equity_net", "day"),
    )
    return dict(
        events=events,
        trades=trades,
        minute=minute,
        accrual=accrual,
        daily=daily,
        partitions=parts,
        scenarios=scenarios,
        summary=summary,
    )


def trade_rows(t, b):
    contributions, excursions = [], []
    observations, life, _ = trace_trade(t, b, checkpoints=())
    excursions.append(
        dict(
            trade_id=t["trade_id"],
            mfe=life["mfe"],
            mae=life["mae"],
            sampled_mfe=max(r["gross_mark"] for r in observations),
            sampled_mae=-min(r["gross_mark"] for r in observations),
            coverage=(
                "incomplete_stop_interval"
                if life["stop_time_uncertain"]
                else "complete_recorded_bar_path"
            ),
            duration_min_lower=life["duration_min_lower"],
            duration_min_upper=life["duration_min_upper"],
            valid_completed_closes=life["valid_completed_closes"],
        )
    )
    begin = timestamp(t["entry_event_timestamp"])
    end = timestamp(t["exit_event_timestamp"])
    held = b[(b.timestamp >= begin) & (b.timestamp <= end)].copy()
    if held.empty:
        raise ValueError("Trade has no minute attribution interval")
    cumulative = (held.close.to_numpy(float) - t["entry_fill"]) * t["direction"] * 2
    cumulative[-1] = t["gross"]  # never sample stop-minute post-exit close
    gross = np.diff(np.r_[0.0, cumulative])
    cost = np.zeros(len(held))
    cost[0] += 1.12
    cost[-1] += 1.12
    exp_upper = np.ones(len(held))
    exp_lower = np.ones(len(held))
    if t["exit_fill_time_basis"] == "bar_open":
        exp_upper[-1] = exp_lower[-1] = 0
    elif t["exit_reason"] == "CAT_STOP":
        exp_lower[-1] = 0
    for i, row in enumerate(held.itertuples()):
        contributions.append(
            dict(
                trade_id=t["trade_id"],
                day=t["day"],
                contract=t["contract"],
                timestamp=row.timestamp,
                **{d: t[d] for d in DIMENSIONS},
                accrual_half_hour=half_hour(row.timestamp),
                gross=gross[i],
                costs=cost[i],
                net=gross[i] - cost[i],
                entry_count=int(i == 0),
                exposure_minutes_upper=exp_upper[i],
                exposure_minutes_lower=exp_lower[i],
                eventual_losers_net_contribution=(
                    (gross[i] - cost[i]) if t["net"] < 0 else 0.0
                ),
            )
        )
    return contributions, excursions[0]


def recorded_scenarios(raw_daily):
    return (
        raw_daily[raw_daily.arm == "A"]
        .groupby(["delay", "cost"], sort=True)
        .agg(
            sessions=("day", "size"),
            gross=("gross", "sum"),
            costs=("costs", "sum"),
            net=("net", "sum"),
        )
        .reset_index()
    )
