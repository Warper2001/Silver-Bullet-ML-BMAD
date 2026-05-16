"""Unit tests for src/research/lr_channel.py.

Validates:
  (a) OLS math against hand-computed values (1e-8 tolerance)
  (b) dev uses population (biased) stddev, not sample stddev
  (c) warm-up returns NaN arrays without raising
  (d) crossunder/crossover entry detection
  (e) MTF slope filter blocks entries when slopes <= 0
"""

import math

import numpy as np
import pytest

from src.research.lr_channel import LRChannel, compute_lr_channel, detect_signals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hand_compute_ols(window: list[float]) -> dict:
    """Brute-force OLS on a window (x = 0..n-1, oldest=0, newest=n-1)."""
    n = len(window)
    x = list(range(n))
    sx = sum(x)
    sy = sum(window)
    sx2 = sum(xi * xi for xi in x)
    xy = sum(xi * yi for xi, yi in zip(x, window))
    slope = (n * xy - sx * sy) / (n * sx2 - sx * sx)
    intercept = (sy - slope * sx) / n
    mid = slope * (n - 1) + intercept
    predicted = [slope * xi + intercept for xi in x]
    residuals = [yi - pi for yi, pi in zip(window, predicted)]
    dev = math.sqrt(sum(r * r for r in residuals) / n)
    upper = mid + dev
    lower = mid - dev
    return dict(slope=slope, intercept=intercept, mid=mid, dev=dev,
                upper=upper, lower=lower)


# ---------------------------------------------------------------------------
# (a) OLS math correctness
# ---------------------------------------------------------------------------

class TestOLSMath:
    def test_slope_intercept_mid_match_hand_computation(self):
        window = [1.0, 2.1, 2.9, 4.2, 5.0, 5.8, 7.1, 7.9, 9.0, 10.1]
        n = len(window)
        expected = _hand_compute_ols(window)

        closes = np.array(window)
        ch = compute_lr_channel(closes, length=n)

        assert abs(ch.slope[-1] - expected["slope"]) < 1e-8
        assert abs(ch.mid[-1] - expected["mid"]) < 1e-8
        assert abs(ch.upper[-1] - expected["upper"]) < 1e-8
        assert abs(ch.lower[-1] - expected["lower"]) < 1e-8

    def test_longer_series_last_window_matches(self):
        rng = np.random.default_rng(42)
        closes = rng.uniform(90, 110, size=50).cumsum()
        length = 10
        ch = compute_lr_channel(closes, length)

        expected = _hand_compute_ols(closes[-length:].tolist())
        assert abs(ch.slope[-1] - expected["slope"]) < 1e-8
        assert abs(ch.mid[-1] - expected["mid"]) < 1e-8
        assert abs(ch.dev[-1] - expected["dev"]) < 1e-8

    def test_warm_up_entries_are_nan(self):
        closes = np.arange(1.0, 21.0)
        length = 10
        ch = compute_lr_channel(closes, length)
        assert np.all(np.isnan(ch.slope[:length - 1]))
        assert np.all(np.isnan(ch.mid[:length - 1]))

    def test_width_and_momentum_computed(self):
        window = [float(i) + 0.5 * (i % 3) for i in range(1, 21)]
        closes = np.array(window)
        ch = compute_lr_channel(closes, length=10)
        assert not np.isnan(ch.width[-1])
        assert not np.isnan(ch.momentum[-1])


# ---------------------------------------------------------------------------
# (b) dev uses population stddev (not sample)
# ---------------------------------------------------------------------------

class TestDevFormula:
    def test_dev_equals_population_not_sample_stddev(self):
        window = [10.0, 12.0, 11.5, 13.0, 14.5, 13.5, 15.0, 14.0, 16.0, 17.0]
        n = len(window)
        expected = _hand_compute_ols(window)

        closes = np.array(window)
        ch = compute_lr_channel(closes, length=n)

        # Population dev must match our hand formula
        assert abs(ch.dev[-1] - expected["dev"]) < 1e-8

        # It must NOT equal numpy's sample stddev of residuals
        x = np.arange(n, dtype=float)
        predicted = ch.slope[-1] * x + (ch.mid[-1] - ch.slope[-1] * (n - 1))
        residuals = closes - predicted
        sample_dev = float(np.std(residuals, ddof=1))
        # They should differ (unless residuals happen to be identical, which is unlikely)
        assert abs(ch.dev[-1] - sample_dev) > 1e-10


# ---------------------------------------------------------------------------
# (c) Warm-up / insufficient bars
# ---------------------------------------------------------------------------

class TestWarmUp:
    def test_fewer_bars_than_length_returns_all_nan(self):
        closes = np.array([1.0, 2.0, 3.0])
        ch = compute_lr_channel(closes, length=10)
        assert len(ch.upper) == 3
        assert np.all(np.isnan(ch.upper))
        assert np.all(np.isnan(ch.slope))

    def test_exact_length_has_one_valid_entry(self):
        closes = np.arange(1.0, 11.0)  # length 10
        ch = compute_lr_channel(closes, length=10)
        assert np.all(np.isnan(ch.slope[:9]))
        assert not np.isnan(ch.slope[9])

    def test_empty_input_returns_empty_nan_arrays(self):
        closes = np.array([], dtype=float)
        ch = compute_lr_channel(closes, length=10)
        for arr in ch:
            assert len(arr) == 0


# ---------------------------------------------------------------------------
# (d) Entry detection (crossunder / crossover)
# ---------------------------------------------------------------------------

def _make_channels(n: int, mid: float = 100.0, half_dev: float = 2.0) -> tuple:
    """Flat synthetic channels for signal-logic tests."""
    upper = np.full(n, mid + half_dev)
    lower = np.full(n, mid - half_dev)
    m = np.full(n, mid)
    slope = np.full(n, 0.1)   # positive slope → MTF filter passes
    dev = np.full(n, half_dev)
    width = np.full(n, 0.01)
    momentum = slope * width
    return LRChannel(upper, lower, m, slope, dev, width, momentum)


class TestEntryDetection:
    def test_crossunder_lower_emits_entry(self):
        n = 10
        ch = _make_channels(n, mid=100.0, half_dev=2.0)
        # Close: starts above lower (98), dips below at bar 5
        closes = np.array([99.0, 99.0, 99.0, 99.0, 98.5, 97.5, 99.0, 99.0, 99.0, 99.0])
        timestamps = list(range(n))

        entries, _ = detect_signals(closes, timestamps, ch, ch, ch, entry_line="lower")
        assert len(entries) == 1
        assert entries[0]["bar_idx"] == 5
        assert entries[0]["trigger"] == "crossunder_lower"

    def test_crossover_mid_emits_entry(self):
        n = 10
        ch = _make_channels(n, mid=100.0, half_dev=2.0)
        closes = np.array([99.0, 99.0, 99.0, 99.0, 99.5, 100.5, 99.0, 99.0, 99.0, 99.0])
        timestamps = list(range(n))

        entries, _ = detect_signals(closes, timestamps, ch, ch, ch, entry_line="mid")
        assert len(entries) == 1
        assert entries[0]["bar_idx"] == 5

    def test_crossover_upper_emits_entry(self):
        n = 10
        ch = _make_channels(n, mid=100.0, half_dev=2.0)
        closes = np.array([101.5, 101.5, 101.5, 101.5, 101.8, 102.5, 101.5, 101.5, 101.5, 101.5])
        timestamps = list(range(n))

        entries, _ = detect_signals(closes, timestamps, ch, ch, ch, entry_line="upper")
        assert len(entries) == 1
        assert entries[0]["bar_idx"] == 5

    def test_invalid_entry_line_raises(self):
        n = 5
        ch = _make_channels(n)
        closes = np.ones(n)
        with pytest.raises(ValueError):
            detect_signals(closes, list(range(n)), ch, ch, ch, entry_line="invalid")


# ---------------------------------------------------------------------------
# (e) MTF slope filter
# ---------------------------------------------------------------------------

class TestMTFFilter:
    def test_negative_slope_blocks_entry(self):
        n = 10
        ch300 = _make_channels(n, mid=100.0, half_dev=2.0)
        # ch100 has negative slope → filter should block
        ch100_neg = LRChannel(
            ch300.upper, ch300.lower, ch300.mid,
            np.full(n, -0.1),   # negative slope
            ch300.dev, ch300.width, ch300.momentum,
        )
        closes = np.array([99.0, 99.0, 99.0, 99.0, 98.5, 97.5, 99.0, 99.0, 99.0, 99.0])
        timestamps = list(range(n))

        entries, _ = detect_signals(closes, timestamps, ch300, ch100_neg, ch300,
                                    entry_line="lower", mtf_slope_filter=True)
        assert len(entries) == 0

    def test_filter_disabled_allows_entry_despite_negative_slope(self):
        n = 10
        ch300 = _make_channels(n, mid=100.0, half_dev=2.0)
        ch100_neg = LRChannel(
            ch300.upper, ch300.lower, ch300.mid,
            np.full(n, -0.1),
            ch300.dev, ch300.width, ch300.momentum,
        )
        closes = np.array([99.0, 99.0, 99.0, 99.0, 98.5, 97.5, 99.0, 99.0, 99.0, 99.0])
        timestamps = list(range(n))

        entries, _ = detect_signals(closes, timestamps, ch300, ch100_neg, ch300,
                                    entry_line="lower", mtf_slope_filter=False)
        assert len(entries) == 1

    def test_nan_slope_blocks_entry(self):
        n = 10
        ch300 = _make_channels(n, mid=100.0, half_dev=2.0)
        ch100_nan = LRChannel(
            ch300.upper, ch300.lower, ch300.mid,
            np.full(n, np.nan),
            ch300.dev, ch300.width, ch300.momentum,
        )
        closes = np.array([99.0, 99.0, 99.0, 99.0, 98.5, 97.5, 99.0, 99.0, 99.0, 99.0])
        timestamps = list(range(n))

        entries, _ = detect_signals(closes, timestamps, ch300, ch100_nan, ch300,
                                    entry_line="lower", mtf_slope_filter=True)
        assert len(entries) == 0

    def test_ch30_negative_slope_blocks_entry_independently(self):
        """ch100 is positive but ch30 is negative — entry must be blocked."""
        n = 10
        ch300 = _make_channels(n, mid=100.0, half_dev=2.0)
        ch100_pos = ch300   # positive slope (0.1)
        ch30_neg = LRChannel(
            ch300.upper, ch300.lower, ch300.mid,
            np.full(n, -0.1),   # negative ch30 slope
            ch300.dev, ch300.width, ch300.momentum,
        )
        closes = np.array([99.0, 99.0, 99.0, 99.0, 98.5, 97.5, 99.0, 99.0, 99.0, 99.0])
        timestamps = list(range(n))

        entries, _ = detect_signals(closes, timestamps, ch300, ch100_pos, ch30_neg,
                                    entry_line="lower", mtf_slope_filter=True)
        assert len(entries) == 0
