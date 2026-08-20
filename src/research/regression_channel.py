"""
Linear regression channel — regime-detection component for the LRC strategy.

New, standalone module. Does not touch strategy_core.py or any file the live
bots import (project_yank_compressed_cascade_prereg lineage: keep new-strategy
research isolated from deployed code).

Computes, per bar of a resampled higher-timeframe series, over a rolling
`lookback`-bar OLS window ending at that bar:
    slope       -- points per bar (trend direction/strength)
    r2          -- goodness of fit (regime: trending vs choppy)
    resid_std   -- std of residuals from the fitted line (band width unit)
    predicted   -- fitted value at the window's last bar
    position_z  -- (close - predicted) / resid_std: how many "band widths"
                   above/below the regression line the close currently sits
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_X_CACHE: dict[int, tuple] = {}


def _x_stats(lookback: int) -> tuple:
    """Cache the fixed x=0..lookback-1 regression design constants."""
    if lookback not in _X_CACHE:
        x = np.arange(lookback, dtype=float)
        x_mean = x.mean()
        sxx = np.sum((x - x_mean) ** 2)
        _X_CACHE[lookback] = (x, x_mean, sxx)
    return _X_CACHE[lookback]


def compute_regression_channel(closes: pd.Series, lookback: int) -> pd.DataFrame:
    """Rolling OLS regression channel. First `lookback - 1` rows are NaN."""
    x, x_mean, sxx = _x_stats(lookback)
    n = len(closes)
    vals = closes.to_numpy(dtype=float)

    slope = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    resid_std = np.full(n, np.nan)
    predicted = np.full(n, np.nan)
    position_z = np.full(n, np.nan)

    if sxx == 0:
        return pd.DataFrame(
            {"slope": slope, "r2": r2, "resid_std": resid_std, "predicted": predicted, "position_z": position_z},
            index=closes.index,
        )

    for i in range(lookback - 1, n):
        y = vals[i - lookback + 1 : i + 1]
        y_mean = y.mean()
        b1 = np.sum((x - x_mean) * (y - y_mean)) / sxx
        b0 = y_mean - b1 * x_mean
        fitted = b0 + b1 * x
        resid = y - fitted
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((y - y_mean) ** 2)

        slope[i] = b1
        r2[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rs = np.sqrt(ss_res / lookback)
        resid_std[i] = rs
        pred_last = b0 + b1 * x[-1]
        predicted[i] = pred_last
        position_z[i] = (vals[i] - pred_last) / rs if rs > 0 else 0.0

    return pd.DataFrame(
        {"slope": slope, "r2": r2, "resid_std": resid_std, "predicted": predicted, "position_z": position_z},
        index=closes.index,
    )
