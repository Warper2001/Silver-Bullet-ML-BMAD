"""Unit tests for Story 5-2: calc_consistency_ratio() pure function."""
import pytest


class TestCalcConsistencyRatio:
    def test_typical_case(self):
        """AC#1: known inputs → expected ratio. best_day=850, total=1370 → 62.04%."""
        from src.research.strategy_core import calc_consistency_ratio
        # Story example: "total accumulated = $1370, today = $520 → 37.96%"
        # means today ($520) is being checked against a history where best_day > today.
        # Direct function: best([520,850])=850, total=1370 → 62.04%
        ratio = calc_consistency_ratio([520.0, 850.0])
        assert ratio == pytest.approx(850.0 / 1370.0 * 100.0, rel=1e-4)

    def test_no_profitable_days_returns_zero(self):
        """AC#1: no profitable sessions → 0.0."""
        from src.research.strategy_core import calc_consistency_ratio
        assert calc_consistency_ratio([]) == 0.0
        assert calc_consistency_ratio([-300.0, -100.0]) == 0.0

    def test_ignores_losing_days(self):
        """AC#1: losing days not counted toward numerator or denominator."""
        from src.research.strategy_core import calc_consistency_ratio
        # Same positive days: [520, 850] → same ratio regardless of negative days
        ratio_with_losses = calc_consistency_ratio([520.0, 850.0, -200.0, -50.0])
        ratio_without = calc_consistency_ratio([520.0, 850.0])
        assert ratio_with_losses == pytest.approx(ratio_without, rel=1e-6)

    def test_single_profitable_day_returns_100(self):
        """Single profitable day: best_day == total → 100%."""
        from src.research.strategy_core import calc_consistency_ratio
        assert calc_consistency_ratio([400.0]) == pytest.approx(100.0)

    def test_equal_days_returns_50(self):
        """Two equal profitable days → 50%."""
        from src.research.strategy_core import calc_consistency_ratio
        assert calc_consistency_ratio([500.0, 500.0]) == pytest.approx(50.0)
