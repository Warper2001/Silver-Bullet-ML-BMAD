"""Unit tests for calc_profit_factor, calc_sharpe, calc_max_drawdown, calc_max_drawdown_pct (Story 1.3, AC #12)."""

from __future__ import annotations

import math

import pytest

from src.research.strategy_core import (
    calc_max_drawdown,
    calc_max_drawdown_pct,
    calc_profit_factor,
    calc_sharpe,
)


class TestCalcProfitFactor:
    def test_mixed_wins_and_losses(self):
        pnls = [10.0, -5.0, 20.0, -10.0]
        pf = calc_profit_factor(pnls)
        assert pf == pytest.approx(30.0 / 15.0)

    def test_no_losses_returns_inf(self):
        pnls = [10.0, 5.0, 20.0]
        assert calc_profit_factor(pnls) == math.inf

    def test_no_wins_returns_zero(self):
        pnls = [-10.0, -5.0]
        assert calc_profit_factor(pnls) == pytest.approx(0.0)

    def test_empty_list_returns_inf(self):
        """Empty list → no losses → inf."""
        assert calc_profit_factor([]) == math.inf

    def test_single_win(self):
        assert calc_profit_factor([100.0]) == math.inf

    def test_single_loss(self):
        assert calc_profit_factor([-100.0]) == pytest.approx(0.0)

    def test_all_zeros(self):
        """All zero trades → no losses → inf."""
        assert calc_profit_factor([0.0, 0.0]) == math.inf

    def test_large_asymmetric_pf(self):
        pnls = [100.0, -1.0]
        assert calc_profit_factor(pnls) == pytest.approx(100.0)

    def test_symmetric_pf_is_one(self):
        pnls = [5.0, -5.0]
        assert calc_profit_factor(pnls) == pytest.approx(1.0)

    def test_result_is_deterministic(self):
        pnls = [10.0, -3.0, 7.0, -4.0]
        assert calc_profit_factor(pnls) == calc_profit_factor(pnls)


class TestCalcSharpe:
    def test_zero_std_returns_zero(self):
        """All identical returns → std=0 → 0.0."""
        assert calc_sharpe([1.0, 1.0, 1.0]) == pytest.approx(0.0)

    def test_single_sample_returns_zero(self):
        assert calc_sharpe([5.0]) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        assert calc_sharpe([]) == pytest.approx(0.0)

    def test_two_samples(self):
        returns = [1.0, -1.0]
        import numpy as np
        import math
        arr_std = float(np.array(returns).std())  # ddof=0
        expected = math.sqrt(252) * 0.0 / arr_std  # mean=0 → 0.0
        assert calc_sharpe(returns) == pytest.approx(expected)

    def test_positive_returns_positive_sharpe(self):
        returns = [0.01] * 5 + [0.02] * 5
        sharpe = calc_sharpe(returns)
        assert sharpe > 0.0

    def test_negative_mean_negative_sharpe(self):
        returns = [-0.02] * 5 + [-0.01] * 5
        sharpe = calc_sharpe(returns)
        assert sharpe < 0.0

    def test_annualisation_factor(self):
        """Manually verify sqrt(252) scaling."""
        import numpy as np
        import math
        returns = [0.01, 0.02, 0.03, 0.04]
        arr = np.array(returns)
        expected = math.sqrt(252) * float(arr.mean()) / float(arr.std())
        assert calc_sharpe(returns) == pytest.approx(expected)

    def test_result_is_deterministic(self):
        returns = [0.01, -0.02, 0.03]
        assert calc_sharpe(returns) == calc_sharpe(returns)


class TestCalcMaxDrawdown:
    def test_flat_equity_zero_drawdown(self):
        assert calc_max_drawdown([100.0, 100.0, 100.0]) == pytest.approx(0.0)

    def test_single_point_zero_drawdown(self):
        assert calc_max_drawdown([50.0]) == pytest.approx(0.0)

    def test_empty_list_zero(self):
        assert calc_max_drawdown([]) == pytest.approx(0.0)

    def test_monotone_rise_zero_drawdown(self):
        assert calc_max_drawdown([0.0, 10.0, 20.0, 30.0]) == pytest.approx(0.0)

    def test_single_drawdown(self):
        """Peak=100, trough=70 → drawdown=30."""
        equity = [0.0, 100.0, 70.0]
        assert calc_max_drawdown(equity) == pytest.approx(30.0)

    def test_recovery_does_not_hide_drawdown(self):
        """[0, 100, 70, 110] → max DD = 30 (100→70), even though new peak reached."""
        equity = [0.0, 100.0, 70.0, 110.0]
        assert calc_max_drawdown(equity) == pytest.approx(30.0)

    def test_multiple_drawdowns_returns_max(self):
        equity = [0.0, 50.0, 30.0, 80.0, 40.0]
        # dd1: 50→30 = 20; dd2: 80→40 = 40 → max = 40
        assert calc_max_drawdown(equity) == pytest.approx(40.0)

    def test_starts_at_nonzero(self):
        equity = [200.0, 150.0, 180.0]
        # peak=200, trough=150, dd=50
        assert calc_max_drawdown(equity) == pytest.approx(50.0)

    def test_result_is_nonnegative(self):
        equity = [1.0, 0.5, 1.5, 0.8]
        assert calc_max_drawdown(equity) >= 0.0

    def test_result_is_deterministic(self):
        equity = [0.0, 100.0, 80.0, 60.0, 90.0]
        assert calc_max_drawdown(equity) == calc_max_drawdown(equity)


class TestCalcMaxDrawdownPct:
    def test_basic_fraction(self):
        # peak=100, trough=70 → dd_pct = 30/100 = 0.30
        equity = [0.0, 100.0, 70.0]
        assert calc_max_drawdown_pct(equity) == pytest.approx(0.30)

    def test_recovery_returns_max_fraction(self):
        # peak=100, trough=70, recovery to 110 → max dd_pct stays at 0.30
        equity = [0.0, 100.0, 70.0, 110.0]
        assert calc_max_drawdown_pct(equity) == pytest.approx(0.30)

    def test_multiple_drawdowns_returns_max(self):
        # dd1: 50→30 = 40%; dd2: 80→40 = 50% → max = 0.50
        equity = [0.0, 50.0, 30.0, 80.0, 40.0]
        assert calc_max_drawdown_pct(equity) == pytest.approx(0.50)

    def test_flat_equity_returns_zero(self):
        assert calc_max_drawdown_pct([100.0, 100.0, 100.0]) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        assert calc_max_drawdown_pct([]) == pytest.approx(0.0)

    def test_single_element_returns_zero(self):
        assert calc_max_drawdown_pct([500.0]) == pytest.approx(0.0)

    def test_peak_zero_or_negative_skipped(self):
        # peak <= 0 prevents division; should not raise, returns 0.0
        assert calc_max_drawdown_pct([0.0, -50.0, -100.0]) == pytest.approx(0.0)

    def test_result_in_unit_interval(self):
        equity = [100.0, 50.0, 30.0, 80.0]
        result = calc_max_drawdown_pct(equity)
        assert 0.0 <= result <= 1.0
