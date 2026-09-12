"""Fixed-capital daily risk metrics and synchronized stationary bootstrap."""

import numpy as np
import pandas as pd
from .artifacts import CONFIG

ARMS = CONFIG["arms"]


def sharpe(x, axis=0):
    x = np.asarray(x, dtype=float) / CONFIG["capital"] - CONFIG["risk_free"]
    std = np.std(x, axis=axis, ddof=1)
    return np.divide(
        np.sqrt(CONFIG["annualization"]) * np.mean(x, axis=axis),
        std,
        out=np.full(np.shape(std), np.nan),
        where=(std > 0) & (np.ptp(x, axis=axis) > 0),
    )


def metrics(group):
    x = group.sort_values("day").net.to_numpy(float)
    equity = np.r_[0.0, np.cumsum(x)]
    drawdown = np.maximum.accumulate(equity) - equity
    longest = current = 0
    for value in drawdown[1:]:
        current = current + 1 if value > 0 else 0
        longest = max(longest, current)
    s = float(sharpe(x)) if len(x) > 1 else float("nan")
    turnover = int(group.turnover.sum())
    return dict(
        sessions=len(x),
        mean_net=float(x.mean()),
        total_net=float(x.sum()),
        sharpe=s if np.isfinite(s) else None,
        max_drawdown=float(drawdown.max()),
        longest_underwater_sessions=longest,
        unresolved_underwater_sessions=current,
        underwater_recovery_censored=bool(current),
        turnover=turnover,
        trades=turnover // 2,
        expectancy=float(x.sum() / (turnover / 2)) if turnover else None,
        exposure_contract_minutes=int(group.exposure_contract_minutes.sum()),
        exposure_fraction=float(group.exposure_contract_minutes.sum() / (390 * len(x))),
    )


def paired_matrix(daily, delay=2, cost=2.24):
    subset = daily[(daily.delay == delay) & (daily.cost == cost)]
    if subset.duplicated(["day", "arm"]).any():
        raise ValueError("Duplicate daily arm observations")
    pivot = subset.pivot(index="day", columns="arm", values="net").reindex(columns=ARMS)
    if (
        pivot.empty
        or pivot.isna().any().any()
        or not np.isfinite(pivot.to_numpy()).all()
    ):
        raise ValueError("Incomplete synchronized daily grid")
    return pivot


def stationary_bootstrap(matrix, block, draws=20000, seed=7, batch=200):
    """Each draw resamples identical indices across all five arms, with wraparound."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 5 or len(matrix) < 2:
        raise ValueError("Bootstrap needs at least two sessions and five arms")
    if block <= 0 or draws <= 0 or batch <= 0 or not np.isfinite(matrix).all():
        raise ValueError("Positive block and draws required")
    rng = np.random.default_rng(seed)
    n = len(matrix)
    result = np.empty((draws, 4))
    for start in range(0, draws, batch):
        size = min(batch, draws - start)
        indices = np.empty((size, n), dtype=np.int32)
        indices[:, 0] = rng.integers(n, size=size)
        restarts = rng.random((size, n - 1)) < 1 / block
        fresh = rng.integers(n, size=(size, n - 1))
        for j in range(1, n):
            indices[:, j] = np.where(
                restarts[:, j - 1], fresh[:, j - 1], (indices[:, j - 1] + 1) % n
            )
        estimates = sharpe(matrix[indices], axis=1)
        result[start : start + size] = estimates[:, 1:] - estimates[:, :1]
    report = {}
    for j, arm in enumerate(ARMS[1:]):
        valid = np.isfinite(result[:, j])
        undefined = int((~valid).sum())
        interval = (
            np.percentile(
                result[valid, j],
                [
                    2.5,
                    97.5,
                    50 * (1 - CONFIG["adjusted_confidence"]),
                    50 * (1 + CONFIG["adjusted_confidence"]),
                ],
            ).tolist()
            if valid.any()
            else [None] * 4
        )
        report[arm] = dict(
            draws=draws,
            undefined_draws=undefined,
            inference_available=bool(
                undefined / draws <= CONFIG["max_undefined_fraction"] and valid.any()
            ),
            ci95=interval[:2],
            ci98_75=interval[2:],
        )
    return report


def classify(primary, highcost, uncertainty):
    baseline, costly_baseline = primary["A"], highcost["A"]
    result = {}
    for arm in ARMS[1:]:
        candidate, costly = primary[arm], highcost[arm]
        delta = (
            candidate["sharpe"] - baseline["sharpe"]
            if candidate["sharpe"] is not None and baseline["sharpe"] is not None
            else None
        )
        costly_delta = (
            costly["sharpe"] - costly_baseline["sharpe"]
            if costly["sharpe"] is not None and costly_baseline["sharpe"] is not None
            else None
        )
        point = bool(
            delta is not None
            and candidate["sharpe"]
            >= baseline["sharpe"] + CONFIG["minimum_delta_sharpe"]
            and candidate["max_drawdown"]
            <= CONFIG["drawdown_ratio"] * baseline["max_drawdown"]
            and candidate["total_net"]
            >= CONFIG["profit_retention"] * baseline["total_net"]
        )
        cost = bool(
            costly["mean_net"] > 0
            and costly_delta is not None
            and costly_delta > 0
            and costly["max_drawdown"] <= costly_baseline["max_drawdown"]
        )
        adjusted = all(
            uncertainty[str(block)][arm]["inference_available"]
            and uncertainty[str(block)][arm]["ci98_75"][0] > 0
            for block in CONFIG["blocks"]
        )
        label = (
            "historical shortlist"
            if point and cost and adjusted
            else "promising but uncertain" if point else "does not meet screen"
        )
        result[arm] = dict(
            delta_sharpe=delta,
            point_screen=point,
            highcost_screen=cost,
            adjusted_intervals_positive_all_blocks=adjusted,
            classification=label,
        )
    ranked = sorted(
        [a for a in result if result[a]["classification"] == "historical shortlist"],
        key=lambda a: (-result[a]["delta_sharpe"], primary[a]["max_drawdown"], a),
    )
    return dict(
        candidates=result, ranked_shortlist=ranked, winner=ranked[0] if ranked else None
    )
