"""LR Channel-based regime detector — deterministic, training-free.

Replaces the HMM regime detector with a stable OLS-slope signal:
  UP       : ch_slow.slope > 0  AND  ch_fast.slope > 0
  DOWN     : ch_slow.slope < 0  AND  ch_fast.slope < 0
  SIDEWAYS : mixed or NaN (warm-up)

Default window sizes are calibrated for 15-minute bars:
  fast_len = 50   →  50 × 15 min = 12.5 hours  (intraday trend)
  slow_len = 200  → 200 × 15 min = 50 hours ≈ 2 days  (multi-day trend)

The OLS slope is far more stable than the HMM (average 10.8-bar duration
at 1-min) or a simple (close[t] - close[t-N]) / (close[t-N] × N) slope
classified by percentile thresholds.  Because it uses the full window for
fitting, a regime only flips when the overall regression trend reverses —
not on a single large bar.
"""

from __future__ import annotations

import numpy as np

import sys
from pathlib import Path

# Ensure src/ root is on the path when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.research.lr_channel import compute_lr_channel


class LRChannelRegimeDetector:
    """Deterministic regime classifier using dual-timeframe OLS slope.

    Parameters
    ----------
    fast_len:
        Short LR window (bars).  Default 50 → 12.5 h at 15-min resolution.
    slow_len:
        Long LR window (bars).  Default 200 → 50 h (~2 days) at 15-min.

    Both channels must agree on direction for an UP or DOWN label; any
    disagreement produces SIDEWAYS.  NaN entries (warm-up) are also SIDEWAYS.
    """

    LABEL_UP       = "UP"
    LABEL_DOWN     = "DOWN"
    LABEL_SIDEWAYS = "SIDEWAYS"

    def __init__(self, fast_len: int = 50, slow_len: int = 200) -> None:
        if fast_len >= slow_len:
            raise ValueError(f"fast_len ({fast_len}) must be < slow_len ({slow_len})")
        self.fast_len = fast_len
        self.slow_len = slow_len

    def fit_predict(self, closes: np.ndarray) -> np.ndarray:
        """Classify every bar into UP / DOWN / SIDEWAYS.

        Parameters
        ----------
        closes:
            1-D float64 array of bar close prices (oldest index 0).

        Returns
        -------
        np.ndarray of dtype object, same length as *closes*, containing
        ``'UP'``, ``'DOWN'``, or ``'SIDEWAYS'`` for each bar.
        """
        closes = np.asarray(closes, dtype=np.float64)
        n = len(closes)
        labels = np.full(n, self.LABEL_SIDEWAYS, dtype=object)

        ch_fast = compute_lr_channel(closes, self.fast_len)
        ch_slow = compute_lr_channel(closes, self.slow_len)

        valid = ~np.isnan(ch_fast.slope) & ~np.isnan(ch_slow.slope)

        up_mask   = valid & (ch_slow.slope > 0) & (ch_fast.slope > 0)
        down_mask = valid & (ch_slow.slope < 0) & (ch_fast.slope < 0)

        labels[up_mask]   = self.LABEL_UP
        labels[down_mask] = self.LABEL_DOWN

        return labels

    def regime_stats(self, labels: np.ndarray) -> dict[str, object]:
        """Compute basic distribution and average run-length statistics."""
        counts = {
            self.LABEL_UP:       int((labels == self.LABEL_UP).sum()),
            self.LABEL_DOWN:     int((labels == self.LABEL_DOWN).sum()),
            self.LABEL_SIDEWAYS: int((labels == self.LABEL_SIDEWAYS).sum()),
        }
        total = len(labels)

        # Average run lengths
        run_lengths: dict[str, list[int]] = {k: [] for k in counts}
        current = labels[0]
        run = 1
        for lbl in labels[1:]:
            if lbl == current:
                run += 1
            else:
                run_lengths[current].append(run)
                current = lbl
                run = 1
        run_lengths[current].append(run)

        avg_run = {
            k: float(np.mean(v)) if v else 0.0
            for k, v in run_lengths.items()
        }

        return {
            "counts": counts,
            "pct": {k: round(v / total * 100, 1) for k, v in counts.items()},
            "avg_run_bars": avg_run,
        }
