from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from .config import CONFIG


def stationary_means(
    values: np.ndarray,
    sample_size: int,
    block: int,
    draws: int,
    seed: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires finite observations")
    rng = np.random.default_rng(seed)
    result = np.empty(draws)
    probability = 1.0 / block
    for draw in range(draws):
        sampled = np.empty(sample_size, dtype=float)
        cursor = 0
        while cursor < sample_size:
            start = int(rng.integers(0, len(values)))
            length = min(int(rng.geometric(probability)), sample_size - cursor)
            sampled[cursor : cursor + length] = values[
                (start + np.arange(length)) % len(values)
            ]
            cursor += length
        result[draw] = sampled.mean()
    return result


def one_sided_ci(
    values: np.ndarray,
    block: int,
    seed: int,
    *,
    draws: int | None = None,
    alpha: float | None = None,
) -> dict[str, float]:
    draws = CONFIG["bootstrap_draws"] if draws is None else draws
    alpha = CONFIG["alpha"] if alpha is None else alpha
    means = stationary_means(
        values,
        len(values),
        block,
        draws,
        seed,
    )
    return {
        "mean": float(np.mean(values)),
        "lower95": float(np.quantile(means, alpha)),
        "upper95": float(np.quantile(means, 1 - alpha)),
    }


def power_from_null(
    null: pd.DataFrame, minimum_effect: float
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {"threshold", "shift", "session_index", "delta"}
    if required - set(null) or null.empty:
        raise ValueError("Power null is empty or incomplete")
    if not np.isfinite(null.delta.to_numpy(float)).all():
        raise ValueError("Power null contains nonfinite deltas")
    if null[["threshold", "shift", "session_index"]].duplicated().any():
        raise ValueError("Power null contains duplicate pairings")
    family_size = null.threshold.nunique()
    if family_size != len(CONFIG["threshold_quantiles"]):
        raise ValueError("Power null candidate family changed")
    if set(null["shift"].astype(int)) != set(CONFIG["null_shifts"]):
        raise ValueError("Power null shift family changed")
    counts = null.groupby(["threshold", "shift"]).size()
    session_count = int(counts.iloc[0])
    if counts.nunique() != 1 or session_count < 2:
        raise ValueError("Power null pairing grid is incomplete")
    expected_indices = set(range(session_count))
    if any(
        set(frame.session_index.astype(int)) != expected_indices
        for _, frame in null.groupby(["threshold", "shift"])
    ):
        raise ValueError("Power null session grid changed")
    alpha_adjusted = CONFIG["alpha"] / family_size
    z_alpha = NormalDist().inv_cdf(1 - alpha_adjusted)
    z_power = NormalDist().inv_cdf(CONFIG["power"])
    rows = []
    for index, (threshold, frame) in enumerate(null.groupby("threshold", sort=True)):
        draws = []
        for shift, shifted in frame.groupby("shift", sort=True):
            draws.append(
                stationary_means(
                    shifted.sort_values("session_index").delta.to_numpy(float),
                    CONFIG["endpoint_sessions"],
                    CONFIG["bootstrap_blocks"][0],
                    CONFIG["bootstrap_draws"],
                    CONFIG["seed"] + index * 1000 + int(shift),
                )
            )
        se = float(np.std(np.concatenate(draws), ddof=1))
        mde = (z_alpha + z_power) * se
        achieved = NormalDist().cdf(minimum_effect / se - z_alpha) if se > 0 else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "null_se_at_500": se,
                "mde80_usd_per_session": float(mde),
                "minimum_effect_usd_per_session": minimum_effect,
                "achieved_power": float(achieved),
                "family_adjusted_alpha": alpha_adjusted,
            }
        )
    table = pd.DataFrame(rows)
    minimum_power = float(table.achieved_power.min())
    verdict = (
        "POWERED"
        if minimum_power >= CONFIG["power"]
        else "MARGINALLY_POWERED" if minimum_power >= 0.50 else "UNDERPOWERED"
    )
    return table, {
        "verdict": verdict,
        "terminal": verdict != "POWERED",
        "minimum_family_power": minimum_power,
        "required_power": CONFIG["power"],
        "endpoint_sessions": CONFIG["endpoint_sessions"],
        "family_size": family_size,
        "identity_pairings_evaluated": 0,
        "aligned_candidate_returns_evaluated": False,
        "minimum_effect_usd_per_session": minimum_effect,
        "minimum_effect_definition": (
            "PF 1.40 with 90% baseline winning-dollar retention; prospective net "
            "retention is a separate 90%-of-baseline-net gate"
        ),
    }


def performance(
    trades: pd.DataFrame, daily: pd.DataFrame
) -> dict[str, float | int | bool | None]:
    wins = float(trades.loc[trades.net > 0, "net"].sum())
    losses = float(-trades.loc[trades.net < 0, "net"].sum())
    equity = daily.net.cumsum().to_numpy(float)
    peak = np.maximum.accumulate(np.r_[0.0, equity])[-len(equity) :]
    top_count = max(1, int(np.ceil(len(trades) * 0.05)))
    net = float(daily.net.sum())
    return {
        "sessions": len(daily),
        "trades": len(trades),
        "net": net,
        "winning_dollars": wins,
        "losing_dollars": losses,
        "pf": wins / losses if losses else None,
        "pf_infinite": bool(not losses and wins > 0),
        "max_drawdown": float(np.max(peak - equity)),
        "top_5pct_trade_fraction": (
            float(trades.nlargest(top_count, "net").net.sum() / net) if net else None
        ),
        "giveback_exits": int(trades.exit_reason.eq("GIVEBACK_EXIT").sum()),
    }
