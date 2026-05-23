"""Unit tests for StrategyConfig.tuesday_exclusion field (Story 2.4).

Covers: default value, explicit False, interaction with other fields.
"""

from src.research.strategy_core import StrategyConfig


class TestTuesdayExclusion:
    def test_tuesday_exclusion_default_true(self):
        """Default StrategyConfig has tuesday_exclusion=True (preserves historic behavior)."""
        assert StrategyConfig().tuesday_exclusion is True

    def test_tuesday_exclusion_can_be_false(self):
        """tuesday_exclusion=False is accepted and stored."""
        config = StrategyConfig(tuesday_exclusion=False)
        assert config.tuesday_exclusion is False

    def test_tuesday_exclusion_independent_of_bearish_only(self):
        """tuesday_exclusion and bearish_only are independent fields."""
        config = StrategyConfig(bearish_only=True, tuesday_exclusion=False)
        assert config.bearish_only is True
        assert config.tuesday_exclusion is False

    def test_tuesday_exclusion_frozen(self):
        """StrategyConfig is frozen — tuesday_exclusion cannot be mutated."""
        import pytest

        config = StrategyConfig()
        with pytest.raises((AttributeError, TypeError)):
            config.tuesday_exclusion = False  # type: ignore[misc]
