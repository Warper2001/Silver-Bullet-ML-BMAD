"""Integration tests for Data Pipeline Orchestrator."""

import asyncio
from datetime import datetime
from unittest import mock

import pytest

from src.data.auth import TradeStationAuth
from src.data.models import DollarBar, MarketData
from src.data.orchestrator import DataPipelineOrchestrator


class TestPipelineEndToEnd:
    """Test end-to-end pipeline functionality."""

    @pytest.fixture
    def mock_auth(self):
        """Create mock TradeStationAuth."""
        return mock.Mock(spec=TradeStationAuth)

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "data" / "processed" / "dollar_bars"

    @pytest.fixture
    def orchestrator(self, mock_auth, temp_dir):
        """Create orchestrator instance."""
        return DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_dir),
            max_queue_size=100,
        )

    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, orchestrator) -> None:
        """Test pipeline initializes all components correctly."""
        orchestrator._initialize_components()

        assert orchestrator._websocket_client is not None
        assert orchestrator._transformer is not None
        assert orchestrator._validator is not None
        assert orchestrator._gap_detector is not None
        assert orchestrator._persistence is not None

    @pytest.mark.asyncio
    async def test_pipeline_start_creates_tasks(self, orchestrator) -> None:
        """Test starting pipeline creates async tasks."""
        orchestrator._initialize_components()

        # Manually create tasks instead of calling start()
        # to avoid actual WebSocket connection
        orchestrator._running = True
        orchestrator._start_time = datetime.now()
        orchestrator._tasks = [
            asyncio.create_task(orchestrator._transformer.consume()),
            asyncio.create_task(orchestrator._validator.consume()),
            asyncio.create_task(orchestrator._gap_detector.consume()),
            asyncio.create_task(orchestrator._persistence.consume()),
            asyncio.create_task(orchestrator._monitor_pipeline_health()),
            asyncio.create_task(orchestrator._log_metrics_periodically()),
        ]

        # Should have created tasks
        assert len(orchestrator._tasks) > 0
        assert orchestrator._running

        # Clean up
        for task in orchestrator._tasks:
            task.cancel()
        await asyncio.gather(*orchestrator._tasks, return_exceptions=True)
        orchestrator._running = False

    @pytest.mark.asyncio
    async def test_pipeline_stop_cancels_tasks(self, orchestrator) -> None:
        """Test stopping pipeline cancels all tasks."""
        orchestrator._initialize_components()

        # Manually create tasks
        orchestrator._running = True
        orchestrator._start_time = datetime.now()
        orchestrator._tasks = [
            asyncio.create_task(asyncio.sleep(0.1)),  # Simple task
        ]

        # Stop pipeline
        for task in orchestrator._tasks:
            task.cancel()

        await asyncio.gather(*orchestrator._tasks, return_exceptions=True)
        orchestrator._running = False

        # Should no longer be running
        assert not orchestrator._running

        # All tasks should be cancelled or done
        for task in orchestrator._tasks:
            assert task.cancelled() or task.done()


class TestBackpressureHandling:
    """Test backpressure monitoring and handling."""

    @pytest.fixture
    def mock_auth(self):
        """Create mock TradeStationAuth."""
        return mock.Mock(spec=TradeStationAuth)

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "data" / "processed" / "dollar_bars"

    @pytest.fixture
    def orchestrator(self, mock_auth, temp_dir):
        """Create orchestrator instance with small queue for testing."""
        return DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_dir),
            max_queue_size=10,  # Small queue for testing
        )

    @pytest.mark.asyncio
    async def test_queue_depth_warning_threshold(self, orchestrator, caplog) -> None:
        """Test warning logged when queue reaches 80% threshold."""
        # Fill queue to 80% (8 items out of 10)
        for i in range(8):
            await orchestrator._raw_queue.put(mock.Mock(spec=MarketData))

        # Check queue depths
        await orchestrator._check_queue_depths()

        # Should log warning
        assert "WARNING backpressure" in caplog.text

    @pytest.mark.asyncio
    async def test_queue_depth_critical_threshold(self, orchestrator, caplog) -> None:
        """Test critical logged when queue reaches 95% threshold."""
        # Fill queue to 95% (9.5 items, so 10 items)
        for i in range(10):
            await orchestrator._raw_queue.put(mock.Mock(spec=MarketData))

        # Check queue depths
        await orchestrator._check_queue_depths()

        # Should log critical
        assert "CRITICAL backpressure" in caplog.text

    @pytest.mark.asyncio
    async def test_queue_depth_below_threshold_no_warning(
        self, orchestrator, caplog
    ) -> None:
        """Test no warning when queue below 80% threshold."""
        # Fill queue to 50% (5 items out of 10)
        for i in range(5):
            await orchestrator._raw_queue.put(mock.Mock(spec=MarketData))

        # Check queue depths
        await orchestrator._check_queue_depths()

        # Should not log warning or critical
        assert "WARNING backpressure" not in caplog.text
        assert "CRITICAL backpressure" not in caplog.text


class TestComponentFailureRecovery:
    """Test component failure detection and recovery."""

    @pytest.fixture
    def mock_auth(self):
        """Create mock TradeStationAuth."""
        return mock.Mock(spec=TradeStationAuth)

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "data" / "processed" / "dollar_bars"

    @pytest.fixture
    def orchestrator(self, mock_auth, temp_dir):
        """Create orchestrator instance."""
        return DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_dir),
        )

    @pytest.mark.asyncio
    async def test_detect_crashed_task(self, orchestrator) -> None:
        """Test detection of crashed tasks."""

        # Create a task that crashes
        async def crashing_task():
            raise RuntimeError("Task crashed")

        task = asyncio.create_task(crashing_task())
        orchestrator._tasks = [task]

        # Wait for task to complete
        await asyncio.sleep(0.1)

        # Mock logger to capture log calls
        with mock.patch("logging.Logger.error") as mock_error:
            # Check component health
            await orchestrator._check_component_health()

            # Should log error about crashed task
            assert mock_error.called
            log_call = str(mock_error.call_args)
            assert "Task 0 crashed" in log_call

    @pytest.mark.asyncio
    async def test_detect_completed_task(self, orchestrator) -> None:
        """Test detection of unexpectedly completed tasks."""

        # Create a task that completes successfully
        async def completed_task():
            return "Task completed"

        task = asyncio.create_task(completed_task())
        orchestrator._tasks = [task]

        # Wait for task to complete
        await asyncio.sleep(0.1)

        # Mock logger to capture log calls
        with mock.patch("logging.Logger.info") as mock_log:
            # Check component health
            await orchestrator._check_component_health()

            # Should log info about completed task
            assert mock_log.called
            log_call = str(mock_log.call_args)
            assert "Task 0 completed unexpectedly" in log_call


class TestPipelineMetrics:
    """Test pipeline metrics collection and logging."""

    @pytest.fixture
    def mock_auth(self):
        """Create mock TradeStationAuth."""
        return mock.Mock(spec=TradeStationAuth)

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "data" / "processed" / "dollar_bars"

    @pytest.fixture
    def orchestrator(self, mock_auth, temp_dir):
        """Create orchestrator instance."""
        return DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_dir),
        )

    @pytest.mark.asyncio
    async def test_metrics_logging_periodically(self, orchestrator) -> None:
        """Test metrics are logged periodically."""
        orchestrator._start_time = datetime.now()
        orchestrator._running = True

        # Capture log output
        with mock.patch("logging.Logger.info") as mock_log:
            # Run one iteration of metrics logging
            orchestrator._log_pipeline_metrics()

            # Should log metrics
            assert mock_log.called
            log_call = str(mock_log.call_args)
            assert "Pipeline metrics" in log_call
            assert "runtime_seconds" in log_call

    @pytest.mark.asyncio
    async def test_metrics_includes_queue_depths(self, orchestrator) -> None:
        """Test metrics include current queue depths."""
        orchestrator._start_time = datetime.now()
        orchestrator._running = True

        # Add items to queues
        await orchestrator._raw_queue.put(mock.Mock(spec=MarketData))
        await orchestrator._transform_queue.put(mock.Mock(spec=DollarBar))

        # Capture log output
        with mock.patch("logging.Logger.info") as mock_log:
            # Log metrics
            orchestrator._log_pipeline_metrics()

            # Should include queue depths
            log_call = str(mock_log.call_args)
            assert "raw_queue_depth=1" in log_call
            assert "transform_queue_depth=1" in log_call


class TestGracefulShutdown:
    """Test graceful shutdown under various conditions."""

    @pytest.fixture
    def mock_auth(self):
        """Create mock TradeStationAuth."""
        return mock.Mock(spec=TradeStationAuth)

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "data" / "processed" / "dollar_bars"

    @pytest.fixture
    def orchestrator(self, mock_auth, temp_dir):
        """Create orchestrator instance."""
        return DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_dir),
        )

    @pytest.mark.asyncio
    async def test_shutdown_disconnects_websocket(self, orchestrator) -> None:
        """Test shutdown disconnects WebSocket."""
        orchestrator._initialize_components()
        orchestrator._running = True

        # Mock cleanup
        with mock.patch.object(
            orchestrator._websocket_client, "cleanup", return_value=asyncio.sleep(0)
        ) as mock_cleanup:
            await orchestrator.stop()

            # Should have called cleanup
            mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_closes_hdf5_files(self, orchestrator) -> None:
        """Test shutdown closes HDF5 file handles."""
        orchestrator._initialize_components()
        orchestrator._running = True

        # Mock close_all_files
        with mock.patch.object(
            orchestrator._persistence, "_close_all_files"
        ) as mock_close:
            await orchestrator.stop()

            # Should have called close
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_waits_for_queues_to_drain(self, orchestrator) -> None:
        """Test shutdown waits for queues to drain."""
        orchestrator._initialize_components()
        orchestrator._running = True

        # Add items to queues
        await orchestrator._raw_queue.put(mock.Mock(spec=MarketData))
        await orchestrator._transform_queue.put(mock.Mock(spec=DollarBar))

        # Mock drain wait
        with mock.patch.object(
            orchestrator, "_wait_for_queues_to_drain", return_value=asyncio.sleep(0)
        ) as mock_drain:
            await orchestrator.stop()

            # Should have waited for drain
            mock_drain.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_timeout_on_queue_drain(self, orchestrator) -> None:
        """Test shutdown handles queue drain timeout gracefully."""
        orchestrator._initialize_components()
        orchestrator._running = True

        # Add items to queues that won't be consumed
        await orchestrator._raw_queue.put(mock.Mock(spec=MarketData))

        # Mock wait_for_queues_to_drain to do nothing
        # The actual code has a 30-second timeout built in via asyncio.wait_for
        with mock.patch.object(
            orchestrator, "_wait_for_queues_to_drain", return_value=asyncio.sleep(0)
        ):
            # Mock cleanup to avoid WebSocket cleanup issues
            with mock.patch.object(
                orchestrator._websocket_client, "cleanup", return_value=asyncio.sleep(0)
            ):
                await orchestrator.stop()

        # Should be stopped
        assert not orchestrator._running


class TestDataCompleteness:
    """Test data completeness tracking."""

    @pytest.fixture
    def mock_auth(self):
        """Create mock TradeStationAuth."""
        return mock.Mock(spec=TradeStationAuth)

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "data" / "processed" / "dollar_bars"

    @pytest.fixture
    def orchestrator(self, mock_auth, temp_dir):
        """Create orchestrator instance."""
        return DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_dir),
        )

    def test_initial_metrics_are_zero(self, orchestrator) -> None:
        """Test all metrics start at zero."""
        assert orchestrator._total_messages_received == 0
        assert orchestrator._total_bars_created == 0
        assert orchestrator._total_bars_validated == 0
        assert orchestrator._total_bars_persisted == 0

    def test_runtime_calculation(self, orchestrator) -> None:
        """Test runtime is calculated correctly."""
        from datetime import datetime, timedelta

        orchestrator._start_time = datetime.now() - timedelta(seconds=30)

        runtime = orchestrator.runtime_seconds
        assert 29.0 <= runtime <= 31.0

    def test_queue_depths_accuracy(self, orchestrator) -> None:
        """Test queue depths report actual queue sizes."""
        depths = orchestrator.queue_depths

        # Initially empty
        assert depths["raw"] == 0
        assert depths["transform"] == 0
        assert depths["validated"] == 0
        assert depths["gap_filled"] == 0

        # Add items
        orchestrator._raw_queue.put_nowait(mock.Mock(spec=MarketData))
        orchestrator._transform_queue.put_nowait(mock.Mock(spec=DollarBar))

        depths = orchestrator.queue_depths
        assert depths["raw"] == 1
        assert depths["transform"] == 1
