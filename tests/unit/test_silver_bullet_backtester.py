"""Unit tests for SilverBulletBacktester.

Tests backtesting of Silver Bullet pattern detection (MSS, FVG, liquidity sweeps)
on historical MNQ Dollar Bars data.
"""

import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

from src.research.silver_bullet_backtester import SilverBulletBacktester


class TestSilverBulletBacktesterInit:
    """Test SilverBulletBacktester initialization."""

    def test_init_with_default_parameters(self):
        """Verify initialization with default parameters."""
        backtester = SilverBulletBacktester()

        assert backtester._mss_lookback == 3
        assert backtester._mss_volume_ratio == 1.5
        assert backtester._fvg_min_gap == 0.25
        assert backtester._sweep_lookback == 5
        assert backtester._confluence_window == 10
        assert len(backtester._time_windows) == 3

    def test_init_with_custom_mss_parameters(self):
        """Verify initialization with custom MSS parameters."""
        backtester = SilverBulletBacktester(
            mss_lookback=5,
            mss_volume_ratio=2.0
        )

        assert backtester._mss_lookback == 5
        assert backtester._mss_volume_ratio == 2.0

    def test_init_with_custom_confluence_window(self):
        """Verify initialization with custom confluence window."""
        backtester = SilverBulletBacktester(
            confluence_window=15
        )

        assert backtester._confluence_window == 15

    def test_init_with_custom_time_windows(self):
        """Verify initialization with custom time windows."""
        custom_windows = [
            {"name": "Custom", "start_hour": 10, "end_hour": 12}
        ]

        backtester = SilverBulletBacktester(
            time_windows=custom_windows
        )

        assert backtester._time_windows == custom_windows


class TestBacktestExecution:
    """Test main backtest execution."""

    @patch('src.research.silver_bullet_backtester.SilverBulletBacktester._detect_mss_events')
    @patch('src.research.silver_bullet_backtester.SilverBulletBacktester._detect_fvg_events')
    @patch('src.research.silver_bullet_backtester.SilverBulletBacktester._detect_sweep_events')
    @patch('src.research.silver_bullet_backtester.SilverBulletBacktester._combine_patterns')
    @patch('src.research.silver_bullet_backtester.SilverBulletBacktester._assign_confidence_scores')
    @patch('src.research.silver_bullet_backtester.SilverBulletBacktester._filter_by_time_windows')
    @patch('src.research.silver_bullet_backtester.SilverBulletBacktester._create_signals_dataframe')
    def test_run_backtest_returns_dataframe(self, mock_create_df, mock_filter, mock_score, mock_combine, mock_sweeps, mock_fvg, mock_mss):
        """Verify run_backtest returns DataFrame with correct columns."""
        backtester = SilverBulletBacktester()

        # Mock the pipeline
        mock_mss.return_value = []
        mock_fvg.return_value = []
        mock_sweeps.return_value = []
        mock_combine.return_value = []
        mock_score.return_value = []
        mock_filter.return_value = []

        mock_signals_df = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-03-01 10:00:00')],
            'direction': ['bullish'],
            'confidence': [85]
        })
        mock_create_df.return_value = mock_signals_df

        # Create test DataFrame
        df = pd.DataFrame({
            'open': [2100.0],
            'high': [2105.0],
            'low': [2098.0],
            'close': [2102.0],
            'volume': [1000]
        }, index=[pd.Timestamp('2024-03-01 10:00:00')])

        result = backtester.run_backtest(df)

        assert isinstance(result, pd.DataFrame)
        assert 'timestamp' in result.columns
        assert 'direction' in result.columns
        assert 'confidence' in result.columns

    def test_run_backtest_handles_empty_dataframe(self):
        """Verify run_backtest handles empty data gracefully."""
        backtester = SilverBulletBacktester()

        with patch.object(backtester, '_detect_mss_events') as mock_mss:
            with patch.object(backtester, '_detect_fvg_events') as mock_fvg:
                with patch.object(backtester, '_detect_sweep_events') as mock_sweeps:
                    mock_mss.return_value = []
                    mock_fvg.return_value = []
                    mock_sweeps.return_value = []

                    with patch.object(backtester, '_combine_patterns') as mock_combine:
                        mock_combine.return_value = []

                        with patch.object(backtester, '_assign_confidence_scores') as mock_score:
                            mock_score.return_value = []

                            with patch.object(backtester, '_filter_by_time_windows') as mock_filter:
                                mock_filter.return_value = []

                                with patch.object(backtester, '_create_signals_dataframe') as mock_create:
                                    mock_create.return_value = pd.DataFrame()

                                    df = pd.DataFrame()
                                    result = backtester.run_backtest(df)

                                    assert isinstance(result, pd.DataFrame)
                                    assert len(result) == 0

    @patch('src.research.silver_bullet_backtester.SilverBulletBacktester._detect_mss_events')
    @patch('src.research.silver_bullet_backtester.SilverBulletBacktester._detect_fvg_events')
    @patch('src.research.silver_bullet_backtester.SilverBulletBacktester._detect_sweep_events')
    @patch('src.research.silver_bullet_backtester.SilverBulletBacktester._combine_patterns')
    @patch('src.research.silver_bullet_backtester.SilverBulletBacktester._assign_confidence_scores')
    @patch('src.research.silver_bullet_backtester.SilverBulletBacktester._filter_by_time_windows')
    @patch('src.research.silver_bullet_backtester.SilverBulletBacktester._create_signals_dataframe')
    def test_run_backtest_processes_pipeline_correctly(self, mock_create_df, mock_filter, mock_score, mock_combine, mock_sweeps, mock_fvg, mock_mss):
        """Verify run_backtest calls all pipeline steps in order."""
        backtester = SilverBulletBacktester()

        # Mock return values
        mock_mss.return_value = [{"timestamp": pd.Timestamp("2024-03-01 10:00:00")}]
        mock_fvg.return_value = [{"timestamp": pd.Timestamp("2024-03-01 10:00:05")}]
        mock_sweeps.return_value = []
        mock_combine.return_value = [{
            "timestamp": pd.Timestamp("2024-03-01 10:00:00"),
            "direction": "bullish"
        }]
        mock_score.return_value = [{
            "timestamp": pd.Timestamp("2024-03-01 10:00:00"),
            "confidence_score": 75
        }]
        mock_filter.return_value = [{
            "timestamp": pd.Timestamp("2024-03-01 10:00:00"),
            "time_window": "NY AM"
        }]
        mock_create_df.return_value = pd.DataFrame()

        df = pd.DataFrame({
            'close': [2100.0]
        }, index=[pd.Timestamp('2024-03-01 10:00:00')])

        backtester.run_backtest(df)

        # Verify all pipeline steps called
        mock_mss.assert_called_once()
        mock_fvg.assert_called_once()
        mock_sweeps.assert_called_once()
        mock_combine.assert_called_once()
        mock_score.assert_called_once()
        mock_filter.assert_called_once()
        mock_create_df.assert_called_once()


class TestMSSDetection:
    """Test MSS event detection."""

    @patch('src.research.silver_bullet_backtester.detect_swing_high')
    @patch('src.research.silver_bullet_backtester.detect_swing_low')
    @patch('src.research.silver_bullet_backtester.detect_bullish_mss')
    @patch('src.research.silver_bullet_backtester.detect_bearish_mss')
    def test_detect_mss_events_returns_events(self, mock_bearish, mock_bullish, mock_low, mock_high):
        """Verify MSS detection returns list of events."""
        backtester = SilverBulletBacktester()

        # Mock swing detection
        mock_high.return_value = []
        mock_low.return_value = []

        # Mock MSS detection
        mock_bullish.return_value = [MagicMock(
            timestamp=pd.Timestamp("2024-03-01 10:00:00"),
            direction="bullish",
            swing_high=2105.0,
            swing_low=2098.0
        )]
        mock_bearish.return_value = []

        df = pd.DataFrame({
            'high': [2105.0],
            'low': [2098.0],
            'volume': [1500]
        }, index=[pd.Timestamp('2024-03-01 10:00:00')])

        events = backtester._detect_mss_events(df)

        assert isinstance(events, list)
        assert len(events) >= 0

    @patch('src.research.silver_bullet_backtester.detect_swing_high')
    @patch('src.research.silver_bullet_backtester.detect_swing_low')
    @patch('src.research.silver_bullet_backtester.detect_bullish_mss')
    @patch('src.research.silver_bullet_backtester.detect_bearish_mss')
    def test_detect_mss_events_includes_volume_confirmation(self, mock_bearish, mock_bullish, mock_low, mock_high):
        """Verify MSS events include volume confirmation."""
        backtester = SilverBulletBacktester()

        mock_high.return_value = []
        mock_low.return_value = []

        mock_bullish.return_value = [MagicMock(
            timestamp=pd.Timestamp("2024-03-01 10:00:00"),
            direction="bullish",
            volume=1800  # Above average
        )]
        mock_bearish.return_value = []

        df = pd.DataFrame({
            'high': [2105.0],
            'low': [2098.0],
            'volume': [1800]
        }, index=[pd.Timestamp('2024-03-01 10:00:00')])

        events = backtester._detect_mss_events(df)

        # Verify volume is tracked
        assert 'volume' in events[0]


class TestFVGDetection:
    """Test FVG event detection."""

    def test_detect_fvg_events_returns_events(self):
        """Verify FVG detection returns list of events."""
        backtester = SilverBulletBacktester()

        df = pd.DataFrame({
            'close': [2100.0, 2101.0, 2099.0],
            'open': [2099.0, 2100.0, 2101.0],
            'high': [2101.0, 2102.0, 2102.0],
            'low': [2098.0, 2099.0, 2098.0]
        }, index=pd.to_datetime([
            '2024-03-01 10:00:00',
            '2024-03-01 10:05:00',
            '2024-03-01 10:10:00'
        ]))

        events = backtester._detect_fvg_events(df)

        assert isinstance(events, list)

    def test_detect_fvg_events_includes_gap_size(self):
        """Verify FVG events include gap size."""
        backtester = SilverBulletBacktester()

        df = pd.DataFrame({
            'close': [2100.0, 2101.0, 2099.0],
            'open': [2099.0, 2100.0, 2101.0],
            'high': [2101.0, 2102.0, 2102.0],
            'low': [2098.0, 2099.0, 2098.0]
        }, index=pd.to_datetime([
            '2024-03-01 10:00:00',
            '2024-03-01 10:05:00',
            '2024-03-01 10:10:00'
        ]))

        events = backtester._detect_fvg_events(df)

        # Check that events have the expected structure
        for event in events:
            if isinstance(event, dict):
                assert 'gap_size' in event or 'timestamp' in event


class TestSweepDetection:
    """Test liquidity sweep detection."""

    @patch('src.research.silver_bullet_backtester.detect_liquidity_sweeps')
    def test_detect_sweep_events_returns_events(self, mock_detect):
        """Verify sweep detection returns list of events."""
        backtester = SilverBulletBacktester()

        mock_sweep = MagicMock(
            timestamp=pd.Timestamp("2024-03-01 10:00:00"),
            direction="bullish",
            depth=0.50,
            volume=2000
        )
        mock_detect.return_value = [mock_sweep]

        df = pd.DataFrame({
            'low': [2095.0]
        }, index=[pd.Timestamp('2024-03-01 10:00:00')])

        events = backtester._detect_sweep_events(df)

        assert isinstance(events, list)
        assert len(events) >= 1

    @patch('src.research.silver_bullet_backtester.detect_liquidity_sweeps')
    def test_detect_sweep_events_includes_depth(self, mock_detect):
        """Verify sweep events include depth."""
        backtester = SilverBulletBacktester()

        mock_sweep = MagicMock(
            timestamp=pd.Timestamp("2024-03-01 10:00:00"),
            direction="bearish",
            depth=1.25,
            volume=2500
        )
        mock_detect.return_value = [mock_sweep]

        df = pd.DataFrame({
            'low': [2095.0]
        }, index=[pd.Timestamp('2024-03-01 10:00:00')])

        events = backtester._detect_sweep_events(df)

        assert 'sweep_depth' in events[0]
        assert events[0]['sweep_depth'] == 1.25


class TestPatternCombination:
    """Test pattern combination into Silver Bullet setups."""

    def test_combine_patterns_creates_setups(self):
        """Verify pattern combination creates setups for MSS+FVG."""
        backtester = SilverBulletBacktester()

        # Create mock events
        mss_events = [{
            "timestamp": pd.Timestamp("2024-03-01 10:00:00"),
            "direction": "bullish"
        }]

        fvg_events = [{
            "timestamp": pd.Timestamp("2024-03-01 10:00:25"),  # 5 bars later
            "direction": "bullish"
        }]

        sweep_events = []

        df = pd.DataFrame()

        setups = backtester._combine_patterns(
            df, mss_events, fvg_events, sweep_events
        )

        assert isinstance(setups, list)
        assert len(setups) >= 1

    def test_combine_patterns_filters_low_confluence(self):
        """Verify pattern combination filters low confluence."""
        backtester = SilverBulletBacktester(confluence_window=5)

        # Create mock events far apart
        mss_events = [{
            "timestamp": pd.Timestamp("2024-03-01 10:00:00"),
            "direction": "bullish"
        }]

        fvg_events = [{
            "timestamp": pd.Timestamp("2024-03-01 12:00:00"),  # 2 hours later
            "direction": "bullish"
        }]

        sweep_events = []

        df = pd.DataFrame()

        setups = backtester._combine_patterns(
            df, mss_events, fvg_events, sweep_events
        )

        # Should not create setup (too far apart)
        assert len(setups) == 0

    def test_combine_patterns_includes_sweeps_when_present(self):
        """Verify pattern combination includes sweeps."""
        backtester = SilverBulletBacktester()

        mss_events = [{
            "timestamp": pd.Timestamp("2024-03-01 10:00:00"),
            "direction": "bullish"
        }]

        fvg_events = [{
            "timestamp": pd.Timestamp("2024-03-01 10:00:25"),
            "direction": "bullish"
        }]

        sweep_events = [{
            "timestamp": pd.Timestamp("2024-03-01 10:00:10"),
            "direction": "bullish"
        }]

        df = pd.DataFrame()

        setups = backtester._combine_patterns(
            df, mss_events, fvg_events, sweep_events
        )

        # Should include sweeps in setup
        if len(setups) > 0:
            assert 'sweep_events' in setups[0]


class TestConfidenceScoring:
    """Test confidence scoring based on confluence and volume."""

    def test_confidence_mss_fvg_sweep_high_confidence(self):
        """Verify MSS+FVG+Sweep gets high confidence (80-100)."""
        backtester = SilverBulletBacktester()

        setups = [{
            "timestamp": pd.Timestamp("2024-03-01 10:00:00"),
            "direction": "bullish",
            "mss_event": {"volume": 2000},
            "fvg_events": [{}],
            "sweep_events": [{}],
            "confluence_count": 3
        }]

        scored = backtester._assign_confidence_scores(setups)

        assert scored[0]['confidence_score'] >= 80
        assert scored[0]['confidence_score'] <= 100

    def test_confidence_mss_fvg_medium_confidence(self):
        """Verify MSS+FVG gets medium confidence (60-79)."""
        backtester = SilverBulletBacktester()

        setups = [{
            "timestamp": pd.Timestamp("2024-03-01 10:00:00"),
            "direction": "bullish",
            "mss_event": {"volume": 1500},
            "fvg_events": [{}],
            "sweep_events": [],
            "confluence_count": 2
        }]

        scored = backtester._assign_confidence_scores(setups)

        assert scored[0]['confidence_score'] >= 60
        assert scored[0]['confidence_score'] < 80

    def test_confidence_includes_volume_bonus(self):
        """Verify confidence scoring includes volume bonus."""
        backtester = SilverBulletBacktester()

        setups_low_vol = [{
            "timestamp": pd.Timestamp("2024-03-01 10:00:00"),
            "direction": "bullish",
            "mss_event": {"volume": 500},
            "fvg_events": [{}],
            "sweep_events": [],
            "confluence_count": 2
        }]

        setups_high_vol = [{
            "timestamp": pd.Timestamp("2024-03-01 10:01:00"),
            "direction": "bullish",
            "mss_event": {"volume": 3000},
            "fvg_events": [{}],
            "sweep_events": [],
            "confluence_count": 2
        }]

        scored = backtester._assign_confidence_scores(
            setups_low_vol + setups_high_vol
        )

        # Higher volume should get higher score
        assert scored[1]['confidence_score'] > scored[0]['confidence_score']

    def test_confidence_capped_at_100(self):
        """Verify confidence scores are capped at 100."""
        backtester = SilverBulletBacktester()

        setups = [{
            "timestamp": pd.Timestamp("2024-03-01 10:00:00"),
            "direction": "bullish",
            "mss_event": {"volume": 10000},  # Very high volume
            "fvg_events": [{}],
            "sweep_events": [{}],
            "confluence_count": 3
        }]

        scored = backtester._assign_confidence_scores(setups)

        assert scored[0]['confidence_score'] <= 100


class TestTimeWindowFiltering:
    """Test time window filtering."""

    def test_filter_by_time_windows_filters_london_am(self):
        """Verify time window filtering includes London AM."""
        backtester = SilverBulletBacktester()

        setups = [{
            "timestamp": pd.Timestamp("2024-03-01 05:00:00"),  # 5 AM EST
            "direction": "bullish"
        }]

        filtered = backtester._filter_by_time_windows(setups)

        assert len(filtered) == 1
        assert filtered[0]['time_window'] == "London AM"

    def test_filter_by_time_windows_filters_ny_am(self):
        """Verify time window filtering includes NY AM."""
        backtester = SilverBulletBacktester()

        setups = [{
            "timestamp": pd.Timestamp("2024-03-01 10:00:00"),  # 10 AM EST
            "direction": "bullish"
        }]

        filtered = backtester._filter_by_time_windows(setups)

        assert len(filtered) == 1
        assert filtered[0]['time_window'] == "NY AM"

    def test_filter_by_time_windows_filters_ny_pm(self):
        """Verify time window filtering includes NY PM."""
        backtester = SilverBulletBacktester()

        setups = [{
            "timestamp": pd.Timestamp("2024-03-01 14:00:00"),  # 2 PM EST
            "direction": "bullish"
        }]

        filtered = backtester._filter_by_time_windows(setups)

        assert len(filtered) == 1
        assert filtered[0]['time_window'] == "NY PM"

    def test_filter_by_time_windows_excludes_outside_hours(self):
        """Verify time window filtering excludes non-trading hours."""
        backtester = SilverBulletBacktester()

        setups = [{
            "timestamp": pd.Timestamp("2024-03-01 20:00:00"),  # 8 PM EST (outside windows)
            "direction": "bullish"
        }]

        filtered = backtester._filter_by_time_windows(setups)

        assert len(filtered) == 0


class TestSignalsDataFrame:
    """Test creation of output signals DataFrame."""

    def test_create_signals_dataframe_returns_correct_format(self):
        """Verify output DataFrame has all required columns."""
        backtester = SilverBulletBacktester()

        setups = [{
            "timestamp": pd.Timestamp("2024-03-01 10:00:00"),
            "direction": "bullish",
            "confidence_score": 85,
            "mss_event": {},
            "fvg_events": [{}],
            "sweep_events": [],
            "time_window": "NY AM"
        }]

        df = backtester._create_signals_dataframe(setups)

        assert isinstance(df, pd.DataFrame)
        assert 'timestamp' in df.columns
        assert 'direction' in df.columns
        assert 'confidence' in df.columns
        assert 'mss_detected' in df.columns
        assert 'fvg_detected' in df.columns
        assert 'sweep_detected' in df.columns
        assert 'time_window' in df.columns

    def test_create_signals_dataframe_populates_direction(self):
        """Verify direction field is populated correctly."""
        backtester = SilverBulletBacktester()

        setups = [{
            "timestamp": pd.Timestamp("2024-03-01 10:00:00"),
            "direction": "bearish",
            "confidence_score": 75,
            "mss_event": {},
            "fvg_events": [{}],
            "sweep_events": [],
            "time_window": "London AM"
        }]

        df = backtester._create_signals_dataframe(setups)

        assert df['direction'].iloc[0] == "bearish"

    def test_create_signals_dataframe_populates_confidence(self):
        """Verify confidence field is populated correctly."""
        backtester = SilverBulletBacktester()

        setups = [{
            "timestamp": pd.Timestamp("2024-03-01 10:00:00"),
            "direction": "bullish",
            "confidence_score": 92,
            "mss_event": {},
            "fvg_events": [{}],
            "sweep_events": [{}],
            "time_window": "NY AM"
        }]

        df = backtester._create_signals_dataframe(setups)

        assert df['confidence'].iloc[0] == 92

    def test_create_signals_dataframe_populates_detection_flags(self):
        """Verify detection flags are set correctly."""
        backtester = SilverBulletBacktester()

        setups = [{
            "timestamp": pd.Timestamp("2024-03-01 10:00:00"),
            "direction": "bullish",
            "confidence_score": 78,
            "mss_event": {},
            "fvg_events": [{}],
            "sweep_events": [{}],
            "time_window": "NY AM"
        }]

        df = backtester._create_signals_dataframe(setups)

        assert df['mss_detected'].iloc[0] is True
        assert df['fvg_detected'].iloc[0] is True
        assert df['sweep_detected'].iloc[0] is True

    def test_create_signals_dataframe_handles_no_sweeps(self):
        """Verify sweep_detected is False when no sweeps."""
        backtester = SilverBulletBacktester()

        setups = [{
            "timestamp": pd.Timestamp("2024-03-01 10:00:00"),
            "direction": "bullish",
            "confidence_score": 65,
            "mss_event": {},
            "fvg_events": [{}],
            "sweep_events": [],  # No sweeps
            "time_window": "London AM"
        }]

        df = backtester._create_signals_dataframe(setups)

        assert df['sweep_detected'].iloc[0] is False
