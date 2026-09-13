"""Completed-minute features; no cross-session window or future observations."""

import numpy as np
from .artifacts import CONFIG


def features(session, sigma):
    close = session.bars.close.to_numpy(float)
    opened = float(session.bars.open.iloc[0])
    previous = session.previous_close
    upper = opened * (1 + sigma) + max(previous - opened, 0)
    lower = opened * (1 - sigma) - max(opened - previous, 0)
    n = len(close)
    slope, r2, efficiency, displacement = [np.full(n, np.nan) for _ in range(4)]
    depth = CONFIG["window"]
    x = np.arange(depth, dtype=float) - (depth - 1) / 2
    for i in range(depth - 1, n):
        window = close[i - depth + 1 : i + 1]
        centered = window - window.mean()
        slope[i] = np.dot(x, centered) / np.dot(x, x)
        variance = np.dot(centered, centered)
        r2[i] = (
            np.dot(x, centered) ** 2 / (np.dot(x, x) * variance) if variance else 0.0
        )
        displacement[i] = window[-1] - window[0]
        distance = np.abs(np.diff(window)).sum()
        efficiency[i] = abs(displacement[i]) / distance if distance else 0.0
    previous_long = np.r_[False, close[:-1] > upper[:-1]]
    previous_short = np.r_[False, close[:-1] < lower[:-1]]
    return dict(
        upper=upper,
        lower=lower,
        slope=slope,
        r2=r2,
        efficiency=efficiency,
        displacement=displacement,
        persistence_long=(close > upper) & previous_long,
        persistence_short=(close < lower) & previous_short,
    )


def gate(arm, direction, i, values, ready=True):
    if arm == "A":
        return True
    if arm == "R":
        return bool(
            values["r2"][i] >= CONFIG["threshold"]
            and direction * values["slope"][i] > 0
        )
    if arm == "E":
        return bool(
            values["efficiency"][i] >= CONFIG["threshold"]
            and direction * values["displacement"][i] > 0
        )
    if arm == "P":
        return bool(
            values["persistence_long" if direction == 1 else "persistence_short"][i]
        )
    if arm == "F":
        return ready
    raise ValueError(f"Unknown arm {arm}")
