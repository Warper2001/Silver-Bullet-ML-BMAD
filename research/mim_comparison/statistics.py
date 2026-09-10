"""Frozen prospective decision; historical results never promote."""

import numpy as np
import pandas as pd


def summary(frame):
    if not len(frame):
        return {"sessions": 0, "unavailable": "no_eligible_sessions"}
    x = frame.net.to_numpy(float)
    equity = np.r_[0, np.cumsum(x)]
    winners = np.sort(x[x > 0])[::-1]
    cut = max(1, int(np.ceil(len(x) * 0.05)))
    return {
        "sessions": len(x),
        "daily_net_expectancy": float(x.mean()),
        "total_net": float(x.sum()),
        "max_drawdown": float(np.max(np.maximum.accumulate(equity) - equity)),
        "turnover": float(frame.turnover.sum()),
        "exposure_contract_minutes": float(frame.exposure_contract_minutes.sum()),
        "net_without_largest_5pct_sessions": float(x.sum() - winners[:cut].sum()),
        "yearly": {
            str(y): {
                "sessions": len(g),
                "net": float(g.net.sum()),
                "expectancy": float(g.net.mean()),
            }
            for y, g in frame.groupby(frame.day.str[:4])
        },
    }


def stationary_means(values, block, draws=20000, seed=7):
    x = np.asarray(values, float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Finite nonempty bootstrap sample required")
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=draws)
    total = x[idx].copy()
    for _ in range(1, n):
        restart = rng.random(draws) < 1 / block
        idx = np.where(restart, rng.integers(0, n, size=draws), (idx + 1) % n)
        total += x[idx]
    return total / n


def decision(frame, complete=False):
    primary = frame[(frame.delay == 2) & (frame.cost == 2.24)] if len(frame) else frame
    paired = (
        primary.pivot(index="day", columns="arm", values="net")
        if len(primary)
        else pd.DataFrame()
    )
    if (
        not complete
        or len(paired) != 120
        or not {"A", "B"} <= set(paired)
        or paired[["A", "B"]].isna().any().any()
    ):
        return {
            "decision": "incomplete/inconclusive",
            "eligible_sessions": len(paired),
            "deployment_authorized": False,
            "reason": "Requires 120 authenticated prospective paired sessions within frozen horizon",
        }
    diff = (paired.B - paired.A).to_numpy()
    b = paired.B.to_numpy()
    intervals = {}
    for block in (5, 10, 20):
        intervals[str(block)] = {
            "incremental": np.quantile(
                stationary_means(diff, block), [0.025, 0.975]
            ).tolist(),
            "B": np.quantile(stationary_means(b, block), [0.025, 0.975]).tolist(),
        }
    high = (
        frame[(frame.delay == 2) & (frame.cost == 6.24)]
        .pivot(index="day", columns="arm", values="net")
        .reindex(paired.index)
    )
    high_ok = (
        {"A", "B"} <= set(high)
        and not high[["A", "B"]].isna().any().any()
        and high.B.mean() > 0
        and (high.B - high.A).mean() > 0
    )

    def classify(ci):
        if ci["incremental"][0] > 5 and ci["B"][0] > 0 and high_ok:
            return "supports_further_validation"
        if ci["incremental"][1] < 5 or ci["B"][1] < 0:
            return "failure"
        return "inconclusive"

    outcomes = [classify(intervals[str(block)]) for block in (5, 10, 20)]
    result = outcomes[0] if len(set(outcomes)) == 1 else "inconclusive"
    return {
        "decision": result,
        "dependence_conflict": len(set(outcomes)) > 1,
        "intervals": intervals,
        "draws": 20000,
        "seed": 7,
        "highest_cost_positive": bool(high_ok),
        "deployment_authorized": False,
        "eligible_sessions": 120,
    }


def minimum_detectable(diff):
    x = np.asarray(diff, float)
    if len(x) < 20:
        return {"unavailable": "fewer_than_20_historical_pairs"}
    # Long-run variance with Bartlett autocovariance weights; no parameter selection.
    centered = x - x.mean()
    gamma = float(centered @ centered / len(x))
    variance = gamma
    for lag in range(1, 6):
        variance += 2 * (1 - lag / 6) * float(centered[:-lag] @ centered[lag:] / len(x))
    se = np.sqrt(max(0, variance) / 120)
    detectable = float((1.96 + 0.8416212335729143) * se)
    return {
        "sessions": 120,
        "power": 0.8,
        "two_sided_alpha": 0.05,
        "assumption": "Normal approximation, historical stationary Bartlett lag5 long-run variance",
        "minimum_detectable_increment_usd": detectable,
        "required_mean_above_5_threshold": 5 + detectable,
        "underpowered_for_5_usd": detectable > 5,
    }
