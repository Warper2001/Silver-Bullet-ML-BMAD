from __future__ import annotations

import numpy as np
import pandas as pd

from research.mim_robustness.engine import Arm
from research.mim_robustness.study import trade_ledger, reconcile_accounting

from .config import CONFIG
from .engine import simulate
from .statistics import one_sided_ci, performance, power_from_null


def simulate_sessions(prepared, threshold=None, external=None, rules=None):
    rules = CONFIG if rules is None else rules
    daily_rows, event_rows, decision_rows = [], [], []
    arm = Arm("A")
    for session, sigma, feature_values in prepared:
        result, events, decisions = simulate(
            session,
            sigma,
            arm,
            delay=int(rules["delay"]),
            cost=float(rules["round_trip_cost"]),
            quantity=int(rules["quantity"]),
            feature_values=feature_values,
            giveback_threshold=threshold,
            external_trigger_minutes=(external or {}).get(session.day),
        )
        daily_rows.append(result)
        event_rows.extend(dict(row, delay=int(rules["delay"])) for row in events)
        decision_rows.extend(dict(row, delay=int(rules["delay"])) for row in decisions)
    daily = pd.DataFrame(daily_rows)
    ledger = pd.DataFrame(event_rows)
    decisions = pd.DataFrame(decision_rows)
    trades = trade_ledger(ledger, decisions)
    reconcile_accounting(daily, ledger, trades)
    return daily, ledger, decisions, trades


def reconcile_baseline(prepared, source_daily):
    daily, ledger, decisions, trades = simulate_sessions(prepared)
    expected = source_daily.sort_values("day").reset_index(drop=True)
    actual = daily.sort_values("day").reset_index(drop=True)
    for column in ("gross", "costs", "net", "turnover"):
        if not np.allclose(actual[column], expected[column], atol=1e-8, rtol=0):
            raise ValueError(f"Baseline {column} does not reconcile")
    if (
        len(trades) != CONFIG["baseline_trades"]
        or not np.isclose(trades.net.sum(), CONFIG["baseline_net"], atol=1e-8, rtol=0)
        or trades.exit_reason.value_counts().to_dict()
        != {
            "EOD_CLOSE_PROXY": 723,
            "CAT_STOP": 71,
            "REVERSAL": 7,
        }
    ):
        raise ValueError("Baseline trade reconciliation failed")
    return daily, ledger, decisions, trades


def _donor_trigger_sets(marks: pd.DataFrame, threshold: float) -> dict[str, set[int]]:
    frame = marks.loc[marks.mfe.gt(0)].copy()
    frame["ratio"] = frame.giveback / frame.mfe
    frame = frame.loc[frame.ratio.ge(threshold)]
    timestamps = pd.to_datetime(frame.mark_timestamp, utc=True).dt.tz_convert(
        "America/New_York"
    )
    frame["minute"] = timestamps.dt.hour * 60 + timestamps.dt.minute
    return {
        str(day): set(group.minute.astype(int))
        for day, group in frame.groupby("day", sort=False)
    }


def _shift_mapping(days: list[str], triggers: dict[str, set[int]], shift: int):
    if not shift or shift % len(days) == 0:
        raise AssertionError("FIREWALL VIOLATION: identity pairing requested")
    return {
        recipient: set(triggers.get(days[(index + shift) % len(days)], set()))
        for index, recipient in enumerate(days)
    }


def run_power(prepared, source, family: pd.DataFrame):
    baseline, _, _, baseline_trades = reconcile_baseline(prepared, source["daily"])
    days = baseline.day.astype(str).tolist()
    null_rows = []
    for threshold in family.threshold:
        donor = _donor_trigger_sets(source["marks"], float(threshold))
        for shift in CONFIG["null_shifts"]:
            external = _shift_mapping(days, donor, shift)
            candidate, _, _, _ = simulate_sessions(prepared, external=external)
            delta = candidate.net.to_numpy(float) - baseline.net.to_numpy(float)
            null_rows.extend(
                {
                    "threshold": float(threshold),
                    "shift": int(shift),
                    "session_index": index,
                    "delta": float(value),
                }
                for index, value in enumerate(delta)
            )
    null = pd.DataFrame(null_rows)
    expected_rows = len(family) * len(CONFIG["null_shifts"]) * len(days)
    if (
        len(null) != expected_rows
        or null[["threshold", "shift", "session_index"]].duplicated().any()
        or not null.delta.ne(0).any()
    ):
        raise ValueError("Mismatched null construction is incomplete or degenerate")
    baseline_stats = performance(baseline_trades, baseline)
    retained_wins = CONFIG["profit_retention"] * baseline_stats["winning_dollars"]
    target_losses = retained_wins / CONFIG["target_pf"]
    target_net = retained_wins - target_losses
    minimum_effect = (target_net - baseline_stats["net"]) / baseline_stats["sessions"]
    power_table, verdict = power_from_null(null, minimum_effect)
    return {
        "baseline_daily": baseline,
        "baseline_trades": baseline_trades,
        "baseline": baseline_stats,
        "family": family,
        "null": null,
        "power": power_table,
        "verdict": verdict,
    }


def run_sweep(prepared, source_daily, family: pd.DataFrame):
    baseline, _, _, baseline_trades = reconcile_baseline(prepared, source_daily)
    baseline_stats = performance(baseline_trades, baseline)
    rows, payloads = [], {}
    for index, threshold in enumerate(sorted(family.threshold)):
        daily, ledger, decisions, trades = simulate_sessions(
            prepared, threshold=float(threshold)
        )
        stats = performance(trades, daily)
        delta = daily.net.to_numpy(float) - baseline.net.to_numpy(float)
        cis = {
            str(block): one_sided_ci(delta, block, CONFIG["seed"] + index * 100 + block)
            for block in CONFIG["bootstrap_blocks"]
        }
        qualifies = bool(
            (stats["pf_infinite"] or (stats["pf"] or 0) >= CONFIG["target_pf"])
            and stats["net"] >= CONFIG["profit_retention"] * baseline_stats["net"]
            and stats["max_drawdown"] <= baseline_stats["max_drawdown"]
            and stats["top_5pct_trade_fraction"] is not None
            and stats["top_5pct_trade_fraction"]
            <= baseline_stats["top_5pct_trade_fraction"]
            and all(value["lower95"] > 0 for value in cis.values())
        )
        rows.append(
            {
                "threshold": float(threshold),
                **stats,
                "qualifies": qualifies,
                **{
                    f"paired_lower95_block_{b}": cis[str(b)]["lower95"]
                    for b in CONFIG["bootstrap_blocks"]
                },
            }
        )
        payloads[float(threshold)] = (daily, ledger, decisions, trades)
    table = pd.DataFrame(rows)
    qualifying = table.loc[table.qualifies].sort_values("threshold", ascending=False)
    selected = None if qualifying.empty else float(qualifying.iloc[0].threshold)
    verdict = "DEVELOPMENT_PASS" if selected is not None else "DEVELOPMENT_FAIL"
    return {
        "baseline": baseline_stats,
        "table": table,
        "selected_threshold": selected,
        "verdict": verdict,
        "selected_payload": payloads.get(selected),
    }
