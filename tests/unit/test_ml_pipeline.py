"""Unit tests for ML Pipeline.

Tests async ML prediction pipeline integration, signal processing,
background tasks, statistics tracking, and error handling.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.data.models import SilverBulletSetup
from src.ml.pipeline import MLPipeline, MLStatistics


class TestMLStatistics:
    """Test MLStatistics tracking class."""

    def test_init_with_zero_values(self):
        """Verify MLStatistics initializes with zero values."""
        stats = MLStatistics()
        assert stats.signals_processed == 0
        assert stats.signals_filtered == 0
        assert stats.total_probability == 0.0
        assert stats.latencies == []

    def test_record_signal_increments_counters(self):
        """Verify record_signal() increments counters correctly."""
        stats = MLStatistics()

        stats.record_signal(probability=0.75, filtered=False, latency_ms=25.0)

        assert stats.signals_processed == 1
        assert stats.signals_filtered == 0

    def test_record_signal_tracks_filtered_signals(self):
        """Verify record_signal() tracks filtered signals."""
        stats = MLStatistics()

        stats.record_signal(probability=0.50, filtered=True, latency_ms=20.0)

        assert stats.signals_filtered == 1

    def test_average_probability_calculation(self):
        """Verify average_probability calculated correctly."""
        stats = MLStatistics()

        stats.record_signal(probability=0.60, filtered=False, latency_ms=25.0)
        stats.record_signal(probability=0.80, filtered=False, latency_ms=30.0)

        assert stats.average_probability == 0.70

    def test_filter_rate_calculation(self):
        """Verify filter_rate calculated correctly."""
        stats = MLStatistics()

        # Process 10 signals, 3 filtered
        for i in range(10):
            filtered = i < 3  # First 3 filtered
            stats.record_signal(probability=0.70, filtered=filtered, latency_ms=25.0)

        assert stats.filter_rate == 0.3

    def test_p50_latency_calculation(self):
        """Verify p50_latency_ms calculates median."""
        stats = MLStatistics()

        # Add latencies: [20, 25, 30, 35, 40]
        for latency in [20, 25, 30, 35, 40]:
            stats.record_signal(probability=0.70, filtered=False, latency_ms=latency)

        assert stats.p50_latency_ms == 30.0


class TestMLPipelineInit:
    """Test MLPipeline initialization."""

    @pytest.fixture
    def mock_queues(self):
        """Create mock async queues."""
        input_queue = Mock(spec=set)  # Mock asyncio.Queue
        output_queue = Mock(spec=set)  # Mock asyncio.Queue
        return input_queue, output_queue

    def test_init_with_default_parameters(self, mock_queues):
        """Verify MLPipeline initializes with default parameters."""
        input_queue, output_queue = mock_queues

        with patch("src.ml.pipeline.MLInference"), \
             patch("src.ml.pipeline.SignalFilter"), \
             patch("src.ml.pipeline.DriftDetector"), \
             patch("src.ml.pipeline.WalkForwardOptimizer"):

            MLPipeline(
                input_queue=input_queue,
                output_queue=output_queue
            )
            # Test passes if initialization succeeds

            assert True  # Verifies initialization completed

    def test_init_with_queue_size_enforcement(self, mock_queues):
        """Verify MLPipeline enforces queue size limits when queues not provided."""
        input_queue, output_queue = mock_queues

        with patch("src.ml.pipeline.MLInference"), \
             patch("src.ml.pipeline.SignalFilter"), \
             patch("src.ml.pipeline.DriftDetector"), \
             patch("src.ml.pipeline.WalkForwardOptimizer"):

            pipeline = MLPipeline(
                input_queue=None,  # No queue provided
                output_queue=None
            )

            # Verify queues created with MAX_QUEUE_SIZE
            assert pipeline._input_queue.maxsize == MLPipeline.MAX_QUEUE_SIZE
            assert pipeline._output_queue.maxsize == MLPipeline.MAX_QUEUE_SIZE

    def test_init_initializes_ml_components(self, mock_queues):
        """Verify MLPipeline initializes all ML components."""
        input_queue, output_queue = mock_queues

        with patch("src.ml.pipeline.MLInference") as mock_inference, \
             patch("src.ml.pipeline.SignalFilter") as mock_filter, \
             patch("src.ml.pipeline.DriftDetector") as mock_drift, \
             patch("src.ml.pipeline.WalkForwardOptimizer") as mock_optimizer:

            pipeline = MLPipeline(  # noqa: F841
                input_queue=input_queue,
                output_queue=output_queue,
                model_dir="models/xgboost"
            )

            mock_inference.assert_called_once_with(model_dir="models/xgboost")
            mock_filter.assert_called_once_with(model_dir="models/xgboost")
            mock_drift.assert_called_once_with(model_dir="models/xgboost")
            mock_optimizer.assert_called_once_with(model_dir="models/xgboost")

    def test_init_creates_statistics_tracker(self, mock_queues):
        """Verify MLPipeline creates MLStatistics instance."""
        input_queue, output_queue = mock_queues

        with patch("src.ml.pipeline.MLInference"), \
             patch("src.ml.pipeline.SignalFilter"), \
             patch("src.ml.pipeline.DriftDetector"), \
             patch("src.ml.pipeline.WalkForwardOptimizer"):

            pipeline = MLPipeline(
                input_queue=input_queue,
                output_queue=output_queue
            )

            assert isinstance(pipeline._statistics, MLStatistics)


class TestSignalProcessing:
    """Test signal processing through ML pipeline."""

    @pytest.fixture
    def mock_signal(self):
        """Create mock SilverBullet signal."""
        signal = Mock(spec=SilverBulletSetup)
        signal.timestamp = "2024-03-16T10:00:00"
        signal.direction = "bullish"
        return signal

    @pytest.fixture
    def mock_pipeline(self):
        """Create mock MLPipeline with mocked components."""
        with patch("src.ml.pipeline.MLInference") as MockInference, \
             patch("src.ml.pipeline.SignalFilter") as MockFilter, \
             patch("src.ml.pipeline.DriftDetector"), \
             patch("src.ml.pipeline.WalkForwardOptimizer"):

            input_queue = Mock()
            output_queue = Mock()

            pipeline = MLPipeline(
                input_queue=input_queue,
                output_queue=output_queue
            )

            # Mock output queue as async queue
            pipeline._output_queue = AsyncMock()

            return pipeline

    @pytest.mark.asyncio
    async def test_process_signal_calls_inference(self, mock_pipeline, mock_signal):
        """Verify process_signal() calls MLInference.predict_probability()."""
        mock_pipeline._inference.predict_probability.return_value = {
            "probability": 0.75,
            "horizon": 5,
            "latency_ms": 10.0
        }

        await mock_pipeline.process_signal(mock_signal)

        mock_pipeline._inference.predict_probability.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_signal_calls_filter(self, mock_pipeline, mock_signal):
        """Verify process_signal() calls SignalFilter.filter_signal()."""
        mock_pipeline._inference.predict_probability.return_value = {
            "probability": 0.75,
            "latency_ms": 10.0
        }

        await mock_pipeline.process_signal(mock_signal)

        mock_pipeline._signal_filter.filter_signal.assert_called_once()
        # Verify call was made with signal and probability
        call_args = mock_pipeline._signal_filter.filter_signal.call_args
        assert call_args[1]["signal"] == mock_signal
        assert call_args[1]["probability"] == 0.75

    @pytest.mark.asyncio
    async def test_process_signal_publishes_allowed_signals(self, mock_pipeline, mock_signal):
        """Verify process_signal() publishes allowed signals to output queue."""
        # Mock inference result
        mock_pipeline._inference.predict_probability.return_value = {
            "probability": 0.75,
            "latency_ms": 10.0
        }

        # Mock filter to allow signal
        mock_pipeline._signal_filter.filter_signal.return_value = {
            "allowed": True,
            "reason": "allowed_by_ml_threshold"
        }

        await mock_pipeline.process_signal(mock_signal)

        # Verify signal was published
        mock_pipeline._output_queue.put.assert_called_once_with(mock_signal)

    @pytest.mark.asyncio
    async def test_process_signal_skips_filtered_signals(self, mock_pipeline, mock_signal):
        """Verify process_signal() does NOT publish filtered signals."""
        # Mock inference result
        mock_pipeline._inference.predict_probability.return_value = {
            "probability": 0.50,
            "latency_ms": 10.0
        }

        # Mock filter to reject signal
        mock_pipeline._signal_filter.filter_signal.return_value = {
            "allowed": False,
            "reason": "filtered_by_ml_threshold"
        }

        await mock_pipeline.process_signal(mock_signal)

        # Verify signal was NOT published
        mock_pipeline._output_queue.put.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_signal_updates_statistics(self, mock_pipeline, mock_signal):
        """Verify process_signal() updates statistics tracking."""
        mock_pipeline._inference.predict_probability.return_value = {
            "probability": 0.75,
            "latency_ms": 10.0
        }

        mock_pipeline._signal_filter.filter_signal.return_value = {
            "allowed": True,
            "reason": "allowed"
        }

        await mock_pipeline.process_signal(mock_signal)

        assert mock_pipeline._statistics.signals_processed == 1


class TestBackgroundTasks:
    """Test background task management."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create mock MLPipeline for background task testing."""
        with patch("src.ml.pipeline.MLInference"), \
             patch("src.ml.pipeline.SignalFilter"), \
             patch("src.ml.pipeline.DriftDetector"), \
             patch("src.ml.pipeline.WalkForwardOptimizer"):

            input_queue = Mock()
            output_queue = AsyncMock()

            pipeline = MLPipeline(
                input_queue=input_queue,
                output_queue=output_queue
            )

            return pipeline

    def test_start_background_tasks_creates_tasks(self, mock_pipeline):
        """Verify start_background_tasks() creates background tasks."""
        with patch("asyncio.create_task") as mock_create_task:
            mock_pipeline.start_background_tasks()

            # Should create 2 tasks (retraining + drift monitoring)
            assert mock_create_task.call_count == 2

    @pytest.mark.asyncio
    async def test_stop_background_tasks_cancels_tasks(self, mock_pipeline):
        """Verify stop_background_tasks() cancels all tasks."""
        # Create mock tasks that are async-awaitable
        async def dummy_task():
            pass

        mock_pipeline._retraining_task = asyncio.create_task(dummy_task())
        mock_pipeline._drift_monitoring_task = asyncio.create_task(dummy_task())

        # Mock the cancel method
        with patch.object(mock_pipeline._retraining_task, "cancel") as mock_cancel1, \
             patch.object(mock_pipeline._drift_monitoring_task, "cancel") as mock_cancel2:
            await mock_pipeline.stop_background_tasks()

            # Verify tasks were cancelled
            mock_cancel1.assert_called_once()
            mock_cancel2.assert_called_once()


class TestErrorHandling:
    """Test error handling and resilience."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create mock MLPipeline for error testing."""
        with patch("src.ml.pipeline.MLInference"), \
             patch("src.ml.pipeline.SignalFilter"), \
             patch("src.ml.pipeline.DriftDetector"), \
             patch("src.ml.pipeline.WalkForwardOptimizer"):

            input_queue = Mock()
            output_queue = AsyncMock()

            pipeline = MLPipeline(
                input_queue=input_queue,
                output_queue=output_queue
            )

            return pipeline

    @pytest.mark.asyncio
    async def test_process_signal_handles_inference_errors(self, mock_pipeline):
        """Verify process_signal() handles inference errors gracefully."""
        mock_signal = Mock()

        # Mock inference to raise error
        mock_pipeline._inference.predict_probability.side_effect = Exception("Model error")

        # Should not raise exception
        await mock_pipeline.process_signal(mock_signal)

        # Signal should not be published
        mock_pipeline._output_queue.put.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_signal_handles_filter_errors(self, mock_pipeline):
        """Verify process_signal() handles filter errors gracefully."""
        mock_signal = Mock()

        # Mock inference success
        mock_pipeline._inference.predict_probability.return_value = {
            "probability": 0.75,
            "latency_ms": 10.0
        }

        # Mock filter to raise error
        mock_pipeline._signal_filter.filter_signal.side_effect = Exception("Filter error")

        # Should not raise exception
        await mock_pipeline.process_signal(mock_signal)

        # Signal should not be published
        mock_pipeline._output_queue.put.assert_not_called()


class TestPerformanceRequirements:
    """Test performance requirements."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create mock MLPipeline with fast mocked components."""
        with patch("src.ml.pipeline.MLInference"), \
             patch("src.ml.pipeline.SignalFilter"), \
             patch("src.ml.pipeline.DriftDetector"), \
             patch("src.ml.pipeline.WalkForwardOptimizer"):

            input_queue = Mock()
            output_queue = AsyncMock()

            pipeline = MLPipeline(
                input_queue=input_queue,
                output_queue=output_queue
            )

            # Mock fast operations
            pipeline._inference.predict_probability.return_value = {
                "probability": 0.75,
                "latency_ms": 10.0
            }
            pipeline._signal_filter.filter_signal.return_value = {
                "allowed": True
            }

            return pipeline

    @pytest.mark.asyncio
    async def test_process_signal_under_50ms(self, mock_pipeline):
        """Verify process_signal() completes in < 50ms."""
        import time

        mock_signal = Mock()

        start_time = time.perf_counter()
        await mock_pipeline.process_signal(mock_signal)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Should be much faster than 50ms with mocked components
        assert elapsed_ms < 50.0, f"Processing took {elapsed_ms:.2f}ms, exceeds 50ms limit"


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create mock MLPipeline for health check testing."""
        with patch("src.ml.pipeline.MLInference") as MockInference, \
             patch("src.ml.pipeline.SignalFilter"), \
             patch("src.ml.pipeline.DriftDetector") as MockDrift, \
             patch("src.ml.pipeline.WalkForwardOptimizer") as MockOptimizer:

            input_queue = Mock()
            input_queue.qsize.return_value = 5
            input_queue.full.return_value = False

            output_queue = Mock()
            output_queue.qsize.return_value = 2
            output_queue.full.return_value = False

            pipeline = MLPipeline(
                input_queue=input_queue,
                output_queue=output_queue
            )

            # Mock inference as loaded
            pipeline._inference._model = Mock()
            pipeline._inference._pipeline = Mock()

            # Mock optimizer scheduler
            mock_scheduler = Mock()
            mock_scheduler.running = True
            pipeline._optimizer._scheduler = mock_scheduler

            return pipeline

    def test_health_check_returns_all_required_fields(self, mock_pipeline):
        """Verify health_check() returns all required fields."""
        result = mock_pipeline.health_check()

        # Verify all required keys present
        assert "healthy" in result
        assert "inference_loaded" in result
        assert "optimizer_scheduler_running" in result
        assert "queue_depth" in result
        assert "statistics" in result

    def test_health_check_reports_healthy_when_all_systems_go(self, mock_pipeline):
        """Verify health_check() reports healthy when all systems operational."""
        result = mock_pipeline.health_check()

        assert result["healthy"] is True
        assert result["inference_loaded"] is True
        assert result["optimizer_scheduler_running"] is True

    def test_health_check_reports_unhealthy_when_inference_not_loaded(self, mock_pipeline):
        """Verify health_check() reports unhealthy when inference not loaded."""
        mock_pipeline._inference._model = None

        result = mock_pipeline.health_check()

        assert result["healthy"] is False
        assert result["inference_loaded"] is False

    def test_health_check_reports_queue_depths(self, mock_pipeline):
        """Verify health_check() reports current queue depths."""
        result = mock_pipeline.health_check()

        assert result["queue_depth"]["input_queue_size"] == 5
        assert result["queue_depth"]["output_queue_size"] == 2
        assert result["queue_depth"]["input_queue_full"] is False
        assert result["queue_depth"]["output_queue_full"] is False

    def test_health_check_reports_unhealthy_when_queues_full(self, mock_pipeline):
        """Verify health_check() reports unhealthy when queues full."""
        mock_pipeline._input_queue.full.return_value = True

        result = mock_pipeline.health_check()

        assert result["healthy"] is False
        assert result["queue_depth"]["input_queue_full"] is True

    @pytest.mark.asyncio
    async def test_process_signal_under_50ms(self, mock_pipeline):
        """Verify process_signal() completes in < 50ms."""
        import time

        mock_signal = Mock()

        start_time = time.perf_counter()
        await mock_pipeline.process_signal(mock_signal)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Should be much faster than 50ms with mocked components
        assert elapsed_ms < 50.0, f"Processing took {elapsed_ms:.2f}ms, exceeds 50ms limit"
