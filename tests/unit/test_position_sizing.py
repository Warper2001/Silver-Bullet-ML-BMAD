"""Unit tests for position sizing algorithm."""

import pytest

from src.execution.entry_logic import PositionSizer


class TestPositionSizer:
    """Test PositionSizer confidence-based position sizing."""

    def test_position_size_tier_1_low_confidence(self):
        """Test Tier 1: 0.50-0.60 confidence → 1 contract."""
        sizer = PositionSizer(min_contracts=1, max_contracts=5)

        # Test boundaries
        assert sizer.calculate_position_size(0.50) == 1
        assert sizer.calculate_position_size(0.55) == 1
        assert sizer.calculate_position_size(0.599) == 1

    def test_position_size_tier_2(self):
        """Test Tier 2: 0.60-0.70 confidence → 2 contracts."""
        sizer = PositionSizer(min_contracts=1, max_contracts=5)

        # Test boundaries
        assert sizer.calculate_position_size(0.60) == 2
        assert sizer.calculate_position_size(0.65) == 2
        assert sizer.calculate_position_size(0.699) == 2

    def test_position_size_tier_3(self):
        """Test Tier 3: 0.70-0.80 confidence → 3 contracts."""
        sizer = PositionSizer(min_contracts=1, max_contracts=5)

        # Test boundaries
        assert sizer.calculate_position_size(0.70) == 3
        assert sizer.calculate_position_size(0.75) == 3
        assert sizer.calculate_position_size(0.799) == 3

    def test_position_size_tier_4(self):
        """Test Tier 4: 0.80-0.90 confidence → 4 contracts."""
        sizer = PositionSizer(min_contracts=1, max_contracts=5)

        # Test boundaries
        assert sizer.calculate_position_size(0.80) == 4
        assert sizer.calculate_position_size(0.85) == 4
        assert sizer.calculate_position_size(0.899) == 4

    def test_position_size_tier_5_high_confidence(self):
        """Test Tier 5: >0.90 confidence → 5 contracts."""
        sizer = PositionSizer(min_contracts=1, max_contracts=5)

        # Test boundaries
        assert sizer.calculate_position_size(0.90) == 5
        assert sizer.calculate_position_size(0.95) == 5
        assert sizer.calculate_position_size(1.0) == 5

    def test_confidence_below_minimum_raises_error(self):
        """Test confidence below 0 raises ValueError."""
        sizer = PositionSizer()

        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            sizer.calculate_position_size(-0.1)

    def test_confidence_above_maximum_raises_error(self):
        """Test confidence above 1 raises ValueError."""
        sizer = PositionSizer()

        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            sizer.calculate_position_size(1.1)

    def test_confidence_exactly_zero_raises_error(self):
        """Test confidence of exactly 0 raises ValueError."""
        sizer = PositionSizer()

        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            sizer.calculate_position_size(0.0)

    def test_confidence_exactly_at_tier_boundaries(self):
        """Test exact tier boundary values."""
        sizer = PositionSizer()

        # Exact boundaries should go to the higher tier
        assert sizer.calculate_position_size(0.60) == 2  # Tier 2
        assert sizer.calculate_position_size(0.70) == 3  # Tier 3
        assert sizer.calculate_position_size(0.80) == 4  # Tier 4
        assert sizer.calculate_position_size(0.90) == 5  # Tier 5

    def test_get_confidence_tier(self):
        """Test confidence tier identification."""
        sizer = PositionSizer()

        assert sizer.get_confidence_tier(0.55) == "Tier 1 (0.50-0.60)"
        assert sizer.get_confidence_tier(0.65) == "Tier 2 (0.60-0.70)"
        assert sizer.get_confidence_tier(0.75) == "Tier 3 (0.70-0.80)"
        assert sizer.get_confidence_tier(0.85) == "Tier 4 (0.80-0.90)"
        assert sizer.get_confidence_tier(0.95) == "Tier 5 (>0.90)"

    def test_custom_min_max_contracts(self):
        """Test custom min/max contract limits."""
        # Custom limits (still need at least tier-based logic)
        sizer = PositionSizer(min_contracts=2, max_contracts=4)

        # Even low confidence should respect min_contracts
        assert sizer.calculate_position_size(0.55) >= 2

        # High confidence should respect max_contracts
        assert sizer.calculate_position_size(0.95) <= 4

    def test_position_size_history_tracking(self):
        """Test position size history tracking."""
        sizer = PositionSizer()

        # Generate some position sizes
        sizer.calculate_position_size(0.55)  # 1
        sizer.calculate_position_size(0.75)  # 3
        sizer.calculate_position_size(0.85)  # 4
        sizer.calculate_position_size(0.65)  # 2
        sizer.calculate_position_size(0.95)  # 5

        # Check history
        assert len(sizer.position_size_history) == 5
        assert sizer.position_size_history == [1, 3, 4, 2, 5]

    def test_average_position_size(self):
        """Test average position size calculation."""
        sizer = PositionSizer()

        # Generate position sizes: 1, 3, 4, 2, 5
        sizer.calculate_position_size(0.55)  # 1
        sizer.calculate_position_size(0.75)  # 3
        sizer.calculate_position_size(0.85)  # 4
        sizer.calculate_position_size(0.65)  # 2
        sizer.calculate_position_size(0.95)  # 5

        # Average = (1+3+4+2+5) / 5 = 3
        assert sizer.get_average_position_size() == 3.0

    def test_position_size_distribution(self):
        """Test position size distribution."""
        sizer = PositionSizer()

        # Generate multiple position sizes
        for _ in range(3):
            sizer.calculate_position_size(0.55)  # 1
        for _ in range(2):
            sizer.calculate_position_size(0.75)  # 3

        distribution = sizer.get_position_size_distribution()

        assert distribution[1] == 3  # Three 1-contract positions
        assert distribution[3] == 2  # Two 3-contract positions
        assert distribution.get(2, 0) == 0  # No 2-contract positions
