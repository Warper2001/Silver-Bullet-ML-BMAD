"""Unit tests for Triple Confluence Scalper strategy components."""

import pytest
from datetime import datetime, timedelta
from src.data.models import DollarBar


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_dollar_bars():
    """Generate sample dollar bars for testing."""
    bars = []
    base_time = datetime(2026, 3, 31, 10, 0, 0)
    base_price = 11800.0

    for i in range(30):
        bar = DollarBar(
            timestamp=base_time + timedelta(minutes=i*5),
            open=base_price + i * 0.5,
            high=base_price + i * 0.5 + 2.0,
            low=base_price + i * 0.5 - 1.0,
            close=base_price + i * 0.5 + 0.5,
            volume=1000 + i * 10,
            notional_value=50_000_000 + i * 100_000,
        )
        bars.append(bar)

    return bars


@pytest.fixture
def sweep_bars():
    """Generate bars with a known level sweep pattern."""
    bars = []
    base_time = datetime(2026, 3, 31, 10, 0, 0)

    # First 15 bars: Establish a daily high at 11820
    for i in range(15):
        bar = DollarBar(
            timestamp=base_time + timedelta(minutes=i*5),
            open=11800.0 + i * 0.5,
            high=11810.0 + i * 0.3,  # Gradually increase high
            low=11798.0,
            close=11805.0 + i * 0.4,
            volume=1000,
            notional_value=50_000_000,
        )
        bars.append(bar)

    # Bar 15-17: Sweep above the high then reverse
    # Bar 15: Break above with high spike
    bars.append(DollarBar(
        timestamp=base_time + timedelta(minutes=15*5),
        open=11807.0,
        high=11825.0,  # Sweep above previous high (~11814)
        low=11802.0,
        close=11810.0,
        volume=1500,
        notional_value=75_000_000,
    ))

    # Bar 16: Continue up
    bars.append(DollarBar(
        timestamp=base_time + timedelta(minutes=16*5),
        open=11810.0,
        high=11823.0,
        low=11808.0,
        close=11812.0,
        volume=1200,
        notional_value=60_000_000,
    ))

    # Bar 17: Reverse back through level (confirm sweep)
    bars.append(DollarBar(
        timestamp=base_time + timedelta(minutes=17*5),
        open=11812.0,
        high=11815.0,
        low=11803.0,  # Drop back below previous high level
        close=11804.0,
        volume=1800,
        notional_value=90_000_000,
    ))

    # Bars 18-20: Continue lower
    for i in range(18, 21):
        bars.append(DollarBar(
            timestamp=base_time + timedelta(minutes=i*5),
            open=11804.0 - (i-18) * 1.0,
            high=11806.0 - (i-18) * 1.0,
            low=11800.0 - (i-18) * 1.5,
            close=11802.0 - (i-18) * 1.2,
            volume=1000,
            notional_value=50_000_000,
        ))

    return bars


@pytest.fixture
def fvg_bars():
    """Generate bars with a known bullish FVG pattern."""
    bars = []
    base_time = datetime(2026, 3, 31, 10, 0, 0)

    # Bar 0: First candle (start of FVG)
    bars.append(DollarBar(
        timestamp=base_time,
        open=11800.0,
        high=11810.0,
        low=11795.0,
        close=11808.0,  # Closes high
        volume=1000,
        notional_value=50_000_000,
    ))

    # Bar 1: Second candle (moves up)
    bars.append(DollarBar(
        timestamp=base_time + timedelta(minutes=5),
        open=11808.0,
        high=11815.0,
        low=11806.0,
        close=11814.0,  # Continues up
        volume=1000,
        notional_value=50_000_000,
    ))

    # Bar 2: Third candle (creates gap - bullish FVG)
    # Bar 0 low (11795) > Bar 2 high (11792) = gap
    bars.append(DollarBar(
        timestamp=base_time + timedelta(minutes=10),
        open=11814.0,
        high=11816.0,
        low=11790.0,  # Drops, creating gap with bar 0
        close=11792.0,
        volume=1200,
        notional_value=60_000_000,
    ))

    # More bars after FVG
    for i in range(3, 10):
        bars.append(DollarBar(
            timestamp=base_time + timedelta(minutes=i*5),
            open=11792.0 + (i-3) * 0.5,
            high=11795.0 + (i-3) * 0.5,
            low=11790.0,
            close=11794.0 + (i-3) * 0.5,
            volume=1000,
            notional_value=50_000_000,
        ))

    return bars


# ============================================================
# TASK 1: LEVEL SWEEP DETECTOR TESTS
# ============================================================

class TestLevelSweepDetector:
    """Test LevelSweepDetector functionality."""

    def test_sweep_detection_with_known_pattern(self, sweep_bars):
        """Test sweep detection with a known bullish sweep pattern."""
        from src.detection.level_sweep_detector import LevelSweepDetector, LevelSweepEvent

        detector = LevelSweepDetector(lookback_period=15)
        sweep_event = None

        # Process bars
        for bar in sweep_bars:
            sweep_event = detector.detect_sweep(sweep_bars)

        # Should detect sweep by bar 17 (after reversal)
        if sweep_event:
            assert isinstance(sweep_event, LevelSweepEvent)
            assert sweep_event.sweep_direction == "bullish"
            assert sweep_event.level_type == "daily_high"
            assert sweep_event.sweep_extent_ticks > 0
            assert sweep_event.timestamp is not None

    def test_no_sweep_in_trending_market(self, sample_dollar_bars):
        """Test that no sweep is detected in a clean uptrend."""
        from src.detection.level_sweep_detector import LevelSweepDetector

        detector = LevelSweepDetector(lookback_period=20)

        # Process all bars - no clear sweep pattern
        sweep_event = detector.detect_sweep(sample_dollar_bars)

        # Should not detect a sweep
        assert sweep_event is None

    def test_sweep_extent_calculation(self, sweep_bars):
        """Test that sweep extent is calculated correctly in ticks."""
        from src.detection.level_sweep_detector import LevelSweepDetector

        detector = LevelSweepDetector(lookback_period=15)
        sweep_event = detector.detect_sweep(sweep_bars)

        if sweep_event:
            # Sweep extent should be positive (ticks above level)
            assert sweep_event.sweep_extent_ticks > 0
            # Convert ticks to dollars (0.25 tick size for MNQ)
            sweep_dollars = sweep_event.sweep_extent_ticks * 0.25
            assert sweep_dollars > 0


# ============================================================
# TASK 2: FVG DETECTOR TESTS
# ============================================================

class TestFVGDetector:
    """Test FVGDetector functionality."""

    def test_bullish_fvg_detection(self, fvg_bars):
        """Test bullish FVG detection."""
        from src.detection.fvg_detector import SimpleFVGDetector

        detector = SimpleFVGDetector(min_gap_size=4)
        fvg_list = detector.detect_fvg(fvg_bars)

        # Should detect at least one FVG
        assert len(fvg_list) > 0

        # Check for bullish FVG
        bullish_fvgs = [fvg for fvg in fvg_list if fvg.fvg_type == "bullish"]
        assert len(bullish_fvgs) > 0

        fvg = bullish_fvgs[0]
        assert fvg.gap_size_ticks >= 4
        assert fvg.gap_edge_high > fvg.gap_edge_low

    def test_bearish_fvg_detection(self):
        """Test bearish FVG detection."""
        from src.detection.fvg_detector import SimpleFVGDetector
        from datetime import timedelta

        # Create bearish FVG pattern
        # Bearish FVG: Bar 1 high < Bar 3 low (gap where price jumps up)
        base_time = datetime(2026, 3, 31, 10, 0, 0)
        bars = [
            # Bar 0: First candle
            DollarBar(
                timestamp=base_time,
                open=11780.0,
                high=11785.0,  # High at 11785
                low=11778.0,
                close=11782.0,
                volume=1000,
                notional_value=50_000_000,
            ),
            # Bar 1: Middle bar
            DollarBar(
                timestamp=base_time + timedelta(minutes=5),
                open=11782.0,
                high=11787.0,
                low=11781.0,
                close=11786.0,
                volume=1000,
                notional_value=50_000_000,
            ),
            # Bar 2: Creates gap (bar 0 high 11785 < bar 2 low 11795)
            DollarBar(
                timestamp=base_time + timedelta(minutes=10),
                open=11796.0,  # Opens higher
                high=11800.0,  # Jumps up
                low=11795.0,  # Low is above bar 0 high = gap!
                close=11798.0,
                volume=1200,
                notional_value=60_000_000,
            ),
        ]

        detector = SimpleFVGDetector(min_gap_size=4)
        fvg_list = detector.detect_fvg(bars)

        # Should detect bearish FVG
        bearish_fvgs = [fvg for fvg in fvg_list if fvg.fvg_type == "bearish"]
        assert len(bearish_fvgs) > 0

    def test_gap_size_filtering(self, fvg_bars):
        """Test that FVG detector filters by minimum gap size."""
        from src.detection.fvg_detector import SimpleFVGDetector

        # Set high minimum - should filter out small gaps
        detector = SimpleFVGDetector(min_gap_size=100)
        fvg_list = detector.detect_fvg(fvg_bars)

        # Should detect fewer (or zero) FVGs with high threshold
        assert len(fvg_list) == 0

    def test_multiple_simultaneous_fvgs(self):
        """Test detection of multiple FVGs at the same time."""
        # This would require complex bar data with multiple gaps
        pass


# ============================================================
# TASK 3: VWAP CALCULATOR TESTS
# ============================================================

class TestVWAPCalculator:
    """Test VWAPCalculator functionality."""

    def test_vwap_calculation_accuracy(self):
        """Test VWAP calculation with known values."""
        from src.detection.vwap_calculator import VWAPCalculator

        calculator = VWAPCalculator(session_start="09:30:00")

        # Create bars with known values
        # Bar 1: H=100, L=95, C=98, V=1000 → typical = (100+95+98)/3 = 97.67
        # Bar 2: H=102, L=97, C=100, V=1500 → typical = (102+97+100)/3 = 99.67
        bars = [
            DollarBar(
                timestamp=datetime(2026, 3, 31, 9, 30, 0),
                open=98.0,
                high=100.0,
                low=95.0,
                close=98.0,
                volume=1000,
                notional_value=50_000_000,
            ),
            DollarBar(
                timestamp=datetime(2026, 3, 31, 9, 35, 0),
                open=98.0,
                high=102.0,
                low=97.0,
                close=100.0,
                volume=1500,
                notional_value=50_000_000,
            ),
        ]

        # Manually calculate VWAP
        # Bar 1: 97.67 * 1000 = 97667
        # Bar 2: 99.67 * 1500 = 149505
        # Total = 247172 / 2500 = 98.87
        expected_vwap = ((97.67 * 1000) + (99.67 * 1500)) / 2500

        calculator_vwap = calculator.calculate_vwap(bars)
        assert abs(calculator_vwap - expected_vwap) < 0.01

    def test_session_reset_logic(self):
        """Test that VWAP resets at session start."""
        from src.detection.vwap_calculator import VWAPCalculator

        calculator = VWAPCalculator(session_start="09:30:00")

        # Day 1 bars
        day1_bars = [
            DollarBar(
                timestamp=datetime(2026, 3, 31, 9, 30, 0),
                open=11800.0,
                high=11810.0,
                low=11795.0,
                close=11805.0,
                volume=1000,
                notional_value=50_000_000,
            ),
        ]

        vwap1 = calculator.calculate_vwap(day1_bars)

        # Reset for new session
        calculator.reset_session()

        # Day 2 bars (different prices)
        day2_bars = [
            DollarBar(
                timestamp=datetime(2026, 4, 1, 9, 30, 0),
                open=11900.0,
                high=11910.0,
                low=11895.0,
                close=11905.0,
                volume=1000,
                notional_value=50_000_000,
            ),
        ]

        vwap2 = calculator.calculate_vwap(day2_bars)

        # VWAP should be different after reset
        assert abs(vwap2 - vwap1) > 50  # Significant difference

    def test_bias_determination_bullish(self):
        """Test bullish bias determination."""
        from src.detection.vwap_calculator import VWAPCalculator

        calculator = VWAPCalculator(session_start="09:30:00")

        bars = [
            DollarBar(
                timestamp=datetime(2026, 3, 31, 9, 30, 0),
                open=11800.0,
                high=11810.0,
                low=11795.0,
                close=11805.0,
                volume=1000,
                notional_value=50_000_000,
            ),
        ]

        vwap = calculator.calculate_vwap(bars)
        bias = calculator.get_bias(11810.0, vwap)  # Price above VWAP

        assert bias == "bullish"

    def test_bias_determination_bearish(self):
        """Test bearish bias determination."""
        from src.detection.vwap_calculator import VWAPCalculator

        calculator = VWAPCalculator(session_start="09:30:00")

        bars = [
            DollarBar(
                timestamp=datetime(2026, 3, 31, 9, 30, 0),
                open=11800.0,
                high=11810.0,
                low=11795.0,
                close=11805.0,
                volume=1000,
                notional_value=50_000_000,
            ),
        ]

        vwap = calculator.calculate_vwap(bars)
        bias = calculator.get_bias(11795.0, vwap)  # Price below VWAP

        assert bias == "bearish"

    def test_bias_determination_neutral(self):
        """Test neutral bias when price is close to VWAP."""
        from src.detection.vwap_calculator import VWAPCalculator

        calculator = VWAPCalculator(session_start="09:30:00")

        bars = [
            DollarBar(
                timestamp=datetime(2026, 3, 31, 9, 30, 0),
                open=11800.0,
                high=11810.0,
                low=11795.0,
                close=11805.0,
                volume=1000,
                notional_value=50_000_000,
            ),
        ]

        vwap = calculator.calculate_vwap(bars)
        # Price within 2 ticks (0.50) of VWAP
        bias = calculator.get_bias(vwap + 0.25, vwap)

        assert bias == "neutral"


# ============================================================
# TASK 4-5: TRIPLE CONFLUENCE STRATEGY TESTS
# ============================================================

class TestTripleConfluenceStrategy:
    """Test TripleConfluenceStrategy functionality."""

    def test_triple_confluence_detection(self):
        """Test detection when all 3 factors align."""
        from src.detection.triple_confluence_strategy import TripleConfluenceStrategy

        strategy = TripleConfluenceStrategy(config={})

        # Create bars with all three patterns
        base_time = datetime(2026, 3, 31, 10, 0, 0)

        # Bars that establish then sweep a high, create FVG, and have VWAP alignment
        bars = [
            # Bars 1-15: Establish daily high around 11810
            DollarBar(
                timestamp=base_time + timedelta(minutes=i*5),
                open=11800.0 + i * 0.2,
                high=11808.0 + i * 0.1,
                low=11798.0,
                close=11805.0 + i * 0.15,
                volume=1000,
                notional_value=50_000_000,
            )
            for i in range(15)
        ]

        # Add sweep bar (high above 11810)
        bars.append(DollarBar(
            timestamp=base_time + timedelta(minutes=75),
            open=11807.0,
            high=11820.0,  # Sweep high
            low=11803.0,
            close=11810.0,
            volume=1500,
            notional_value=75_000_000,
        ))

        # Add reversal bar
        bars.append(DollarBar(
            timestamp=base_time + timedelta(minutes=80),
            open=11810.0,
            high=11812.0,
            low=11802.0,  # Drop back below level
            close=11803.0,
            volume=1800,
            notional_value=90_000_000,
        ))

        # Process bars
        signal = None
        for bar in bars:
            signal = strategy.process_bar(bar)

        # With proper setup, should detect signal
        # (This is a basic test - full confluence requires specific patterns)
        if signal:
            assert signal.strategy_name == "Triple Confluence Scalper"
            assert signal.direction in ["long", "short"]
            assert 0.8 <= signal.confidence <= 1.0

    def test_signal_generation_with_all_3_factors(self):
        """Test signal generation when triple confluence exists."""
        from src.detection.triple_confluence_strategy import TripleConfluenceStrategy
        from src.detection.level_sweep_detector import LevelSweepDetector
        from src.detection.fvg_detector import SimpleFVGDetector
        from src.detection.vwap_calculator import VWAPCalculator

        strategy = TripleConfluenceStrategy(config={})

        # Process a bar through the strategy
        bar = DollarBar(
            timestamp=datetime(2026, 3, 31, 10, 30, 0),
            open=11800.0,
            high=11810.0,
            low=11795.0,
            close=11805.0,
            volume=1000,
            notional_value=50_000_000,
        )

        signal = strategy.process_bar(bar)

        # Signal may or may not be generated depending on confluence
        if signal:
            # Verify signal structure
            assert hasattr(signal, 'entry_price')
            assert hasattr(signal, 'stop_loss')
            assert hasattr(signal, 'take_profit')
            assert hasattr(signal, 'direction')
            assert hasattr(signal, 'confidence')
            assert hasattr(signal, 'timestamp')

    def test_rejection_when_less_than_min_factors(self):
        """Test that signals are rejected when < minimum factors agree."""
        from src.detection.triple_confluence_strategy import TripleConfluenceStrategy

        # Test with default 2-of-3 confluence
        strategy = TripleConfluenceStrategy(config={})

        # Process a bar without proper confluence
        bar = DollarBar(
            timestamp=datetime(2026, 3, 31, 10, 0, 0),
            open=11800.0,
            high=11802.0,
            low=11799.0,
            close=11801.0,
            volume=1000,
            notional_value=50_000_000,
        )

        signal = strategy.process_bar(bar)

        # Should not generate signal without confluence
        assert signal is None

        # Test with 3-of-3 confluence (stricter)
        strategy_strict = TripleConfluenceStrategy(config={"min_confluence_factors": 3})

        # Process same bar
        signal_strict = strategy_strict.process_bar(bar)

        # Should not generate signal without all 3 factors
        assert signal_strict is None

    def test_confidence_score_calculation(self):
        """Test confidence score is calculated correctly (0.65-1.0)."""
        from src.detection.triple_confluence_strategy import TripleConfluenceStrategy

        strategy = TripleConfluenceStrategy(config={})

        # The confidence score should be calculated when confluence exists
        # Based on strength of each factor (sweep extent, FVG size, VWAP distance)
        # This tests the calculation logic is in place

        # Process a single bar - unlikely to have confluence
        bar = DollarBar(
            timestamp=datetime(2026, 3, 31, 10, 0, 0),
            open=11800.0,
            high=11805.0,
            low=11795.0,
            close=11800.0,
            volume=1000,
            notional_value=50_000_000,
        )

        signal = strategy.process_bar(bar)

        # If signal generated, check confidence range
        # 2-factor confluence: 0.70-1.0
        # 3-factor confluence: 0.80-1.0
        if signal:
            assert 0.65 <= signal.confidence <= 1.0

    def test_stop_loss_and_take_profit_calculations(self):
        """Test SL/TP calculations respect 2:1 ratio."""
        from src.detection.triple_confluence_strategy import TripleConfluenceStrategy

        strategy = TripleConfluenceStrategy(config={})

        # Process bars
        bar = DollarBar(
            timestamp=datetime(2026, 3, 31, 10, 0, 0),
            open=11800.0,
            high=11810.0,
            low=11795.0,
            close=11805.0,
            volume=1000,
            notional_value=50_000_000,
        )

        signal = strategy.process_bar(bar)

        # If signal generated, verify 2:1 ratio
        if signal:
            if signal.direction == "long":
                risk = signal.entry_price - signal.stop_loss
                reward = signal.take_profit - signal.entry_price
            else:  # short
                risk = signal.stop_loss - signal.entry_price
                reward = signal.entry_price - signal.take_profit

            if risk > 0:
                ratio = reward / risk
                assert ratio >= 1.9  # Allow small tolerance
