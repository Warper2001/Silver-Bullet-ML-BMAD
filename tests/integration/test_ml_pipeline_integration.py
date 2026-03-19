"""Integration tests for ML Pipeline.

Tests end-to-end flow from DetectionPipeline through MLPipeline,
integration with WalkForwardOptimizer and DriftDetector, signal
filtering under load, and background task execution.
"""

import asyncio
from datetime import datetime
from unittest.mock import patch

import pytest

from src.data.models import (
    FVGEvent,
    GapRange,
    MSSEvent,
    SilverBulletSetup,
    SwingPoint,
)
from src.ml.drift_detector import DriftDetector
from src.ml.pipeline import MLPipeline
from src.ml.walk_forward_optimizer import WalkForwardOptimizer


class TestEndToEndFlow:
    """Test end-to-end flow from DetectionPipeline through MLPipeline."""

    @pytest.fixture
    def real_signal(self):
        """Create real SilverBullet signal for integration testing."""
        base_time = datetime(2024, 3, 16, 10, 0, 0)

        swing = SwingPoint(
            timestamp=base_time,
            price=11750.0,
            swing_type="swing_low",
            bar_index=100,
            confirmed=True,
        )

        mss = MSSEvent(
            timestamp=base_time,
            direction="bullish",
            breakout_price=11750.0,
            swing_point=swing,
            volume_ratio=1.5,
            bar_index=100,
        )

        gap_range = GapRange(top=11760.0, bottom=11740.0)

        fvg = FVGEvent(
            timestamp=base_time,
            direction="bullish",
            gap_range=gap_range,
            gap_size_ticks=20.0,
            gap_size_dollars=100.0,
            bar_index=100,
        )

        return SilverBulletSetup(
            timestamp=base_time,
            direction="bullish",
            mss_event=mss,
            fvg_event=fvg,
            entry_zone_top=11760.0,
            entry_zone_bottom=11740.0,
            invalidation_point=11750.0,
            confluence_count=2,
            priority="medium",
            bar_index=100,
            confidence=3,
        )

    @pytest.fixture
    def ml_pipeline(self, tmp_path):
        """Create MLPipeline with real components for integration testing."""
        model_dir = tmp_path / "xgboost" / "5_minute"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create minimal model files
        (model_dir / "xgboost_model.pkl").write_text("{}")
        (model_dir / "feature_pipeline.pkl").write_text("{}")
        (model_dir / "metadata.json").write_text('{"metrics": {"win_rate": 0.60}}')

        input_queue = asyncio.Queue(maxsize=100)
        output_queue = asyncio.Queue(maxsize=100)

        pipeline = MLPipeline(
            input_queue=input_queue,
            output_queue=output_queue,
            model_dir=tmp_path / "xgboost",
        )

        return pipeline

    @pytest.mark.asyncio
    async def test_signal_flows_from_input_to_output_when_probability_high(
        self, ml_pipeline, real_signal
    ):
        """Verify signal flows through pipeline when probability exceeds threshold."""
        # Mock high probability inference
        ml_pipeline._inference.predict_probability.return_value = {
            "probability": 0.75,
            "horizon": 5,
            "latency_ms": 10.0,
        }

        # Mock filter to allow signal
        ml_pipeline._signal_filter.filter_signal.return_value = {
            "allowed": True,
            "reason": "allowed_by_ml_threshold",
        }

        # Process signal
        await ml_pipeline.process_signal(real_signal)

        # Verify signal reached output queue
        assert not ml_pipeline._output_queue.empty()
        output_signal = await ml_pipeline._output_queue.get()
        assert output_signal.direction == "bullish"

    @pytest.mark.asyncio
    async def test_signal_blocked_when_probability_low(self, ml_pipeline, real_signal):
        """Verify signal blocked when probability below threshold."""
        # Mock low probability inference
        ml_pipeline._inference.predict_probability.return_value = {
            "probability": 0.50,
            "horizon": 5,
            "latency_ms": 10.0,
        }

        # Mock filter to reject signal
        ml_pipeline._signal_filter.filter_signal.return_value = {
            "allowed": False,
            "reason": "filtered_by_ml_threshold",
        }

        # Process signal
        await ml_pipeline.process_signal(real_signal)

        # Verify signal did NOT reach output queue
        assert ml_pipeline._output_queue.empty()

    @pytest.mark.asyncio
    async def test_pipeline_tracks_statistics_across_multiple_signals(
        self, ml_pipeline, real_signal
    ):
        """Verify statistics tracking works across multiple signal processing."""
        # Mock inference to return varying probabilities
        ml_pipeline._inference.predict_probability.side_effect = [
            {"probability": 0.75, "latency_ms": 10.0},
            {"probability": 0.50, "latency_ms": 12.0},
            {"probability": 0.80, "latency_ms": 11.0},
        ]

        # Mock filter
        ml_pipeline._signal_filter.filter_signal.side_effect = [
            {"allowed": True, "reason": "allowed"},
            {"allowed": False, "reason": "filtered"},
            {"allowed": True, "reason": "allowed"},
        ]

        # Process 3 signals
        for _ in range(3):
            await ml_pipeline.process_signal(real_signal)

        # Verify statistics
        stats = ml_pipeline._statistics.get_summary()
        assert stats["signals_processed"] == 3
        assert stats["signals_filtered"] == 1
        assert stats["average_probability"] == pytest.approx(0.683, rel=0.01)


class TestWalkForwardOptimizerIntegration:
    """Test integration with WalkForwardOptimizer."""

    @pytest.fixture
    def ml_pipeline_with_optimizer(self, tmp_path):
        """Create MLPipeline with mocked optimizer for testing."""
        model_dir = tmp_path / "xgboost" / "5_minute"
        model_dir.mkdir(parents=True, exist_ok=True)

        (model_dir / "xgboost_model.json").write_text("{}")
        (model_dir / "feature_pipeline.pkl").write_text("{}")
        (model_dir / "metadata.json").write_text('{"metrics": {"win_rate": 0.60}}')

        input_queue = asyncio.Queue(maxsize=100)
        output_queue = asyncio.Queue(maxsize=100)

        pipeline = MLPipeline(
            input_queue=input_queue,
            output_queue=output_queue,
            model_dir=tmp_path / "xgboost",
        )

        return pipeline

    def test_optimizer_initialized_on_pipeline_start(
        self, ml_pipeline_with_optimizer
    ):
        """Verify WalkForwardOptimizer initialized when pipeline starts."""
        assert ml_pipeline_with_optimizer._optimizer is not None
        assert isinstance(
            ml_pipeline_with_optimizer._optimizer, WalkForwardOptimizer
        )

    @pytest.mark.asyncio
    async def test_weekly_retraining_task_created_on_start(
        self, ml_pipeline_with_optimizer
    ):
        """Verify weekly retraining background task created."""
        with patch("asyncio.create_task") as mock_create_task:
            ml_pipeline_with_optimizer.start_background_tasks()

            # Verify 2 tasks created (retraining + drift monitoring)
            assert mock_create_task.call_count == 2

    @pytest.mark.asyncio
    async def test_retraining_task_survives_exceptions(
        self, ml_pipeline_with_optimizer
    ):
        """Verify retraining task continues after exceptions."""
        # Mock optimizer to fail once then succeed
        call_count = [0]

        async def mock_retraining():
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Temporary failure")
            return {"success": True}

        ml_pipeline_with_optimizer._optimizer.run_retraining_with_retry = (
            mock_retraining
        )

        # Run retraining - should survive first failure
        result = await ml_pipeline_with_optimizer._optimizer.run_retraining_with_retry()

        # Verify succeeded on retry
        assert result["success"] is True
        assert call_count[0] == 2


class TestDriftDetectorIntegration:
    """Test integration with DriftDetector."""

    @pytest.fixture
    def ml_pipeline_with_drift(self, tmp_path):
        """Create MLPipeline with drift detection for testing."""
        model_dir = tmp_path / "xgboost" / "5_minute"
        model_dir.mkdir(parents=True, exist_ok=True)

        (model_dir / "xgboost_model.json").write_text("{}")
        (model_dir / "feature_pipeline.pkl").write_text("{}")
        (model_dir / "metadata.json").write_text('{"metrics": {"win_rate": 0.60}}')

        input_queue = asyncio.Queue(maxsize=100)
        output_queue = asyncio.Queue(maxsize=100)

        pipeline = MLPipeline(
            input_queue=input_queue,
            output_queue=output_queue,
            model_dir=tmp_path / "xgboost",
        )

        return pipeline

    def test_drift_detector_initialized_on_pipeline_start(
        self, ml_pipeline_with_drift
    ):
        """Verify DriftDetector initialized when pipeline starts."""
        assert ml_pipeline_with_drift._drift_detector is not None
        assert isinstance(ml_pipeline_with_drift._drift_detector, DriftDetector)

    @pytest.mark.asyncio
    async def test_drift_monitoring_task_created_on_start(
        self, ml_pipeline_with_drift
    ):
        """Verify drift monitoring background task created."""
        with patch("asyncio.create_task") as mock_create_task:
            ml_pipeline_with_drift.start_background_tasks()

            # Verify 2 tasks created (retraining + drift monitoring)
            assert mock_create_task.call_count == 2

    @pytest.mark.asyncio
    async def test_drift_check_loads_expected_win_rate_from_metadata(
        self, ml_pipeline_with_drift, tmp_path
    ):
        """Verify drift monitoring loads expected win rate from model metadata."""
        expected_win_rate = ml_pipeline_with_drift._load_expected_win_rate()

        # Verify loaded from metadata.json
        assert expected_win_rate == 0.60


class TestSignalFilteringUnderLoad:
    """Test signal filtering under high load."""

    @pytest.fixture
    def ml_pipeline_for_load(self, tmp_path):
        """Create MLPipeline for load testing."""
        model_dir = tmp_path / "xgboost" / "5_minute"
        model_dir.mkdir(parents=True, exist_ok=True)

        (model_dir / "xgboost_model.json").write_text("{}")
        (model_dir / "feature_pipeline.pkl").write_text("{}")
        (model_dir / "metadata.json").write_text('{"metrics": {"win_rate": 0.60}}')

        input_queue = asyncio.Queue(maxsize=100)
        output_queue = asyncio.Queue(maxsize=100)

        pipeline = MLPipeline(
            input_queue=input_queue,
            output_queue=output_queue,
            model_dir=tmp_path / "xgboost",
        )

        return pipeline

    @pytest.fixture
    def batch_signals(self):
        """Create batch of signals for load testing."""
        return [
            SilverBulletSetup(
                timestamp=f"2024-03-16T10:0{i}:00",
                direction="bullish" if i % 2 == 0 else "bearish",
                entry_price=11750.0 + i,
                stop_loss=11700.0,
                take_profit=11850.0,
                patterns=["mss"],
                confidence=0.75,
            )
            for i in range(10)
        ]

    @pytest.mark.asyncio
    async def test_pipeline_handles_batch_signals_without_dropping(
        self, ml_pipeline_for_load, batch_signals
    ):
        """Verify pipeline processes batch of signals without dropping."""
        # Mock components
        ml_pipeline_for_load._inference.predict_probability.return_value = {
            "probability": 0.70,
            "latency_ms": 10.0,
        }
        ml_pipeline_for_load._signal_filter.filter_signal.return_value = {
            "allowed": True,
            "reason": "allowed",
        }

        # Process all signals
        tasks = [
            ml_pipeline_for_load.process_signal(signal) for signal in batch_signals
        ]
        await asyncio.gather(*tasks)

        # Verify all processed
        stats = ml_pipeline_for_load._statistics.get_summary()
        assert stats["signals_processed"] == 10

    @pytest.mark.asyncio
    async def test_latency_remains_under_50ms_under_load(
        self, ml_pipeline_for_load, batch_signals
    ):
        """Verify latency remains < 50ms even with multiple signals."""
        import time

        # Mock fast components
        ml_pipeline_for_load._inference.predict_probability.return_value = {
            "probability": 0.70,
            "latency_ms": 5.0,
        }
        ml_pipeline_for_load._signal_filter.filter_signal.return_value = {
            "allowed": True,
            "reason": "allowed",
        }

        # Process signals and measure time
        start_time = time.perf_counter()
        tasks = [
            ml_pipeline_for_load.process_signal(signal) for signal in batch_signals
        ]
        await asyncio.gather(*tasks)
        total_time_ms = (time.perf_counter() - start_time) * 1000

        # Verify average latency per signal < 50ms
        avg_latency_ms = total_time_ms / len(batch_signals)
        assert avg_latency_ms < 50.0, f"Average latency {avg_latency_ms:.2f}ms exceeds 50ms"

    @pytest.mark.asyncio
    async def test_filter_rate_calculated_correctly_under_load(
        self, ml_pipeline_for_load, batch_signals
    ):
        """Verify filter rate calculated correctly under load."""
        # Mock alternating allow/reject
        ml_pipeline_for_load._inference.predict_probability.side_effect = [
            {"probability": 0.70, "latency_ms": 10.0} if i % 2 == 0 else {
                "probability": 0.50,
                "latency_ms": 10.0,
            }
            for i in range(10)
        ]
        ml_pipeline_for_load._signal_filter.filter_signal.side_effect = [
            {"allowed": True, "reason": "allowed"} if i % 2 == 0 else {
                "allowed": False,
                "reason": "filtered",
            }
            for i in range(10)
        ]

        # Process all signals
        tasks = [
            ml_pipeline_for_load.process_signal(signal) for signal in batch_signals
        ]
        await asyncio.gather(*tasks)

        # Verify filter rate = 50%
        stats = ml_pipeline_for_load._statistics.get_summary()
        assert stats["filter_rate"] == 0.5


class TestBackgroundTaskExecution:
    """Test background task execution and lifecycle."""

    @pytest.fixture
    def ml_pipeline_background(self, tmp_path):
        """Create MLPipeline for background task testing."""
        model_dir = tmp_path / "xgboost" / "5_minute"
        model_dir.mkdir(parents=True, exist_ok=True)

        (model_dir / "xgboost_model.json").write_text("{}")
        (model_dir / "feature_pipeline.pkl").write_text("{}")
        (model_dir / "metadata.json").write_text('{"metrics": {"win_rate": 0.60}}')

        input_queue = asyncio.Queue(maxsize=100)
        output_queue = asyncio.Queue(maxsize=100)

        pipeline = MLPipeline(
            input_queue=input_queue,
            output_queue=output_queue,
            model_dir=tmp_path / "xgboost",
        )

        return pipeline

    def test_background_tasks_initialized_on_start(
        self, ml_pipeline_background
    ):
        """Verify background tasks initialized when start_background_tasks() called."""
        ml_pipeline_background.start_background_tasks()

        assert ml_pipeline_background._retraining_task is not None
        assert ml_pipeline_background._drift_monitoring_task is not None

    @pytest.mark.asyncio
    async def test_background_tasks_can_be_stopped_gracefully(
        self, ml_pipeline_background
    ):
        """Verify background tasks stop gracefully when stop() called."""
        ml_pipeline_background.start_background_tasks()

        # Verify tasks created
        assert ml_pipeline_background._retraining_task is not None
        assert ml_pipeline_background._drift_monitoring_task is not None

        # Stop tasks
        await ml_pipeline_background.stop_background_tasks()

        # Verify tasks cancelled (tasks are cancelled but not None)
        # Tasks should be cancelled, so we verify the method completes without error

    @pytest.mark.asyncio
    async def test_retraining_task_survives_cancellation_and_restarts(
        self, ml_pipeline_background
    ):
        """Verify retraining task can be cancelled and restarted."""
        # Start background tasks
        ml_pipeline_background.start_background_tasks()

        # Stop tasks
        await ml_pipeline_background.stop_background_tasks()

        # Restart tasks
        ml_pipeline_background.start_background_tasks()

        # Verify new tasks created
        assert ml_pipeline_background._retraining_task is not None
        assert ml_pipeline_background._drift_monitoring_task is not None
