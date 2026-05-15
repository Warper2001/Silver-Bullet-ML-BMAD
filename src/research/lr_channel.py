"""Linear regression channel math — pure, deterministic, side-effect-free.

Implements OLS channels (any length) matching Pine Script's ta.linreg formula:
  slope  = (n*xy - sx*sy) / (n*sx2 - sx**2)
  intercept = (sy - slope*sx) / n
  mid_i  = slope*(n-1) + intercept   (end-of-window predicted value)
  dev    = sqrt(mean(residuals**2))  — population (biased) stddev
  width  = (upper - lower) / sqrt(1 + slope**2) / n
  momentum = slope * width
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class LRChannel(NamedTuple):
    upper: np.ndarray
    lower: np.ndarray
    mid: np.ndarray
    slope: np.ndarray
    dev: np.ndarray
    width: np.ndarray
    momentum: np.ndarray


def compute_lr_channel(closes: np.ndarray, length: int) -> LRChannel:
    """Compute a rolling linear-regression channel over *closes*.

    Returns arrays of the same length as *closes*; the first ``length - 1``
    entries of each array are ``NaN`` (warm-up period).

    Parameters
    ----------
    closes:
        1-D float64 array of bar close prices, oldest index 0.
    length:
        Regression window length (e.g. 300, 100, 30).
    """
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    out_shape = (n,)

    upper = np.full(out_shape, np.nan)
    lower = np.full(out_shape, np.nan)
    mid = np.full(out_shape, np.nan)
    slope = np.full(out_shape, np.nan)
    dev = np.full(out_shape, np.nan)
    width = np.full(out_shape, np.nan)
    momentum = np.full(out_shape, np.nan)

    if n < length:
        return LRChannel(upper, lower, mid, slope, dev, width, momentum)

    # Pre-compute OLS constants (x = 0..L-1, oldest=0, newest=L-1)
    L = float(length)
    x_vec = np.arange(length, dtype=np.float64)
    sx  = 0.5 * L * (L - 1)
    sx2 = L * (L - 1) * (2 * L - 1) / 6.0
    denom = L * sx2 - sx * sx

    # sliding_window_view: zero-copy (m, L) view; windows[k] = closes[k:k+L]
    windows = sliding_window_view(closes, length)   # shape (m, L), m = n - L + 1

    # Cheap reductions — whole-array is fine (output is 1-D, no large temps)
    sy = windows.sum(axis=1)            # (m,)
    xy = windows @ x_vec                # (m,)
    s  = (L * xy - sx * sy) / denom    # slope (m,)
    b  = (sy - s * sx) / L             # intercept (m,)
    mv = s * (L - 1) + b               # mid value at newest bar (m,)

    # Chunked dev — avoids a full (m, L) ≈ GB-scale temporary at year-scale data.
    # Each chunk allocates at most CHUNK*L*8*2 bytes (window copy + residuals).
    # At CHUNK=20_000, L=500 → ~160 MB peak; L=300 → ~96 MB.
    CHUNK = 20_000
    m_len = len(windows)
    d = np.empty(m_len)
    for ci in range(0, m_len, CHUNK):
        sl   = slice(ci, ci + CHUNK)
        wc   = np.ascontiguousarray(windows[sl])       # (chunk, L) contiguous copy
        pred = s[sl, None] * x_vec + b[sl, None]       # (chunk, L)
        d[ci : ci + CHUNK] = np.sqrt(((wc - pred) ** 2).mean(axis=1))

    w = 2.0 * d / np.sqrt(1.0 + s * s) / L            # width: (upper-lower) = 2d

    start = length - 1
    slope[start:]    = s
    mid[start:]      = mv
    upper[start:]    = mv + d
    lower[start:]    = mv - d
    dev[start:]      = d
    width[start:]    = w
    momentum[start:] = s * w

    return LRChannel(upper, lower, mid, slope, dev, width, momentum)


def detect_signals(
    closes: np.ndarray,
    timestamps,
    ch300: LRChannel,
    ch100: LRChannel,
    ch30: LRChannel,
    entry_line: str = "lower",
    mtf_slope_filter: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Detect entry and exit signals on confirmed bar closes.

    Parameters
    ----------
    closes:
        Close prices array (same length as channel arrays).
    timestamps:
        Sequence of bar timestamps (same length).
    ch300, ch100, ch30:
        LRChannel outputs for 300-bar, 100-bar, 30-bar windows.
    entry_line:
        Which 300-bar channel line triggers a long entry:
        ``'lower'``  — close crosses *below* lower rail (mean-reversion),
        ``'mid'``    — close crosses *above* midline from below,
        ``'upper'``  — close crosses *above* upper rail (breakout).
    mtf_slope_filter:
        When ``True``, only emit entries where ch100.slope > 0 AND ch30.slope > 0.

    Returns
    -------
    entries : list of dicts with keys ``bar_idx``, ``timestamp``, ``entry_line``, ``trigger``
    exits   : list of dicts with keys ``bar_idx``, ``timestamp``, ``reason``
              (bar_idx is the signal bar; the backtest fills at bar_idx+1 open)
    """
    if entry_line not in ("lower", "mid", "upper"):
        raise ValueError(f"entry_line must be 'lower', 'mid', or 'upper'; got {entry_line!r}")

    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    entries: list[dict] = []
    exits: list[dict] = []

    for i in range(1, n):
        # Skip if any required channel value is NaN at t or t-1
        if np.isnan(ch300.mid[i]) or np.isnan(ch300.mid[i - 1]):
            continue

        # --- Entry detection ---
        if entry_line == "lower":
            triggered = (
                closes[i - 1] >= ch300.lower[i - 1]
                and closes[i] < ch300.lower[i]
            )
            trigger_label = "crossunder_lower"
        elif entry_line == "mid":
            triggered = (
                closes[i - 1] <= ch300.mid[i - 1]
                and closes[i] > ch300.mid[i]
            )
            trigger_label = "crossover_mid"
        else:  # upper
            triggered = (
                closes[i - 1] <= ch300.upper[i - 1]
                and closes[i] > ch300.upper[i]
            )
            trigger_label = "crossover_upper"

        if triggered:
            passes_filter = True
            if mtf_slope_filter:
                s100 = ch100.slope[i]
                s30 = ch30.slope[i]
                passes_filter = (
                    not np.isnan(s100)
                    and not np.isnan(s30)
                    and s100 > 0
                    and s30 > 0
                )
            if passes_filter:
                entries.append({
                    "bar_idx": i,
                    "timestamp": timestamps[i],
                    "entry_line": entry_line,
                    "trigger": trigger_label,
                })

        # --- Exit detection: close crosses above 300-bar midline ---
        if (
            not np.isnan(ch300.mid[i - 1])
            and closes[i - 1] <= ch300.mid[i - 1]
            and closes[i] > ch300.mid[i]
        ):
            exits.append({
                "bar_idx": i,
                "timestamp": timestamps[i],
                "reason": "midline_return",
            })

    return entries, exits
