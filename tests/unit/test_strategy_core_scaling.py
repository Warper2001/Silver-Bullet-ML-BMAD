"""Unit tests for Story 5-3: calc_contract_limit() pure function."""
import pytest

_XFA_PLAN = [
    {"milestone_usd": 0, "max_contracts": 2},
    {"milestone_usd": 1500, "max_contracts": 3},
    {"milestone_usd": 2000, "max_contracts": 5},
]


class TestCalcContractLimit:
    def test_first_tier_below_first_milestone(self):
        """AC#3: accumulated below first paid milestone → base (tier 0) limit."""
        from src.research.strategy_core import calc_contract_limit
        assert calc_contract_limit(900.0, _XFA_PLAN) == 2

    def test_middle_tier_past_1500(self):
        """AC#1: accumulated past $1,500 → 3 contracts."""
        from src.research.strategy_core import calc_contract_limit
        assert calc_contract_limit(1600.0, _XFA_PLAN) == 3

    def test_top_tier_past_2000(self):
        """AC#2: accumulated past $2,000 → 5 contracts."""
        from src.research.strategy_core import calc_contract_limit
        assert calc_contract_limit(2100.0, _XFA_PLAN) == 5

    def test_empty_plan_returns_none(self):
        """AC#6: empty scaling_plan → None (no cap)."""
        from src.research.strategy_core import calc_contract_limit
        assert calc_contract_limit(5000.0, []) is None

    def test_exactly_at_milestone(self):
        """AC#5: accumulated exactly at milestone boundary → that tier."""
        from src.research.strategy_core import calc_contract_limit
        assert calc_contract_limit(1500.0, _XFA_PLAN) == 3
        assert calc_contract_limit(2000.0, _XFA_PLAN) == 5

    def test_zero_accumulated(self):
        """Zero accumulated profit → base tier."""
        from src.research.strategy_core import calc_contract_limit
        assert calc_contract_limit(0.0, _XFA_PLAN) == 2
