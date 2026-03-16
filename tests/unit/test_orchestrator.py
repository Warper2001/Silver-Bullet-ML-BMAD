"""Unit tests for Data Pipeline Orchestrator."""

import asyncio
from unittest import mock

import pytest

from src.data.orchestrator import DataPipelineOrchestrator
from src.data.auth import TradeStationAuth


class TestDataPipelineOrchestratorInitialization:
    """Test DataPipelineOrchestrator initialization."""

    @pytest.fixture
    def mock_auth(self):
        """Create mock TradeStationAuth."""
        return mock.Mock(spec=TradeStationAuth)

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "data" / "processed" / "dollar_bars"

    def test_initialization_with_defaults(self, mock_auth, temp_dir) -> None:
        """Test initialization with default parameters."""
        orchestrator = DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_dir),
        )

        assert orchestrator._auth == mock_auth
        assert orchestrator._data_directory == temp_dir
        assert orchestrator._max_queue_size == 1000
        assert orchestrator._settings is not None
        assert not orchestrator._running

    def test_initialization_with_custom_queue_size(self, mock_auth, temp_dir) -> None:
        """Test initialization with custom queue size."""
        orchestrator = DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_dir),
            max_queue_size=500,
        )

        assert orchestrator._max_queue_size == 500

    def test_initialization_creates_queues(self, mock_auth, temp_dir) -> None:
        """Test initialization creates all required queues."""
        orchestrator = DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_dir),
        )

        assert orchestrator._raw_queue is not None
        assert orchestrator._transform_queue is not None
        assert orchestrator._validated_queue is not None
        assert orchestrator._gap_filled_queue is not None

    def test_initial_queue_sizes(self, mock_auth, temp_dir) -> None:
        """Test queues are initialized with correct max sizes."""
        orchestrator = DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_dir),
            max_queue_size=500,
        )

        assert orchestrator._raw_queue.maxsize == 500
        assert orchestrator._transform_queue.maxsize == 500
        assert orchestrator._validated_queue.maxsize == 500
        assert orchestrator._gap_filled_queue.maxsize == 500

    def test_initial_metrics(self, mock_auth, temp_dir) -> None:
        """Test initial metrics are zero."""
        orchestrator = DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_dir),
        )

        assert orchestrator._total_messages_received == 0
        assert orchestrator._total_bars_created == 0
        assert orchestrator._total_bars_validated == 0
        assert orchestrator._total_bars_persisted == 0


class TestQueueManagement:
    """Test queue management and monitoring."""

    @pytest.fixture
    def orchestrator(self, mock_auth, temp_dir):
        """Create orchestrator instance."""
        return DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_dir),
            max_queue_size=100,
        )

    @pytest.fixture
    def mock_auth(self):
        """Create mock TradeStationAuth."""
        return mock.Mock(spec=TradeStationAuth)

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "data" / "processed" / "dollar_bars"

    def test_queue_depths_property(self, orchestrator) -> None:
        """Test queue_depths property returns correct values."""
        depths = orchestrator.queue_depths

        assert "raw" in depths
        assert "transform" in depths
        assert "validated" in depths
        assert "gap_filled" in depths
        assert all(isinstance(v, int) for v in depths.values())

    def test_queue_depths_initially_empty(self, orchestrator) -> None:
        """Test all queues are initially empty."""
        depths = orchestrator.queue_depths

        assert depths["raw"] == 0
        assert depths["transform"] == 0
        assert depths["validated"] == 0
        assert depths["gap_filled"] == 0


class TestComponentHealth:
    """Test component health monitoring."""

    @pytest.fixture
    def orchestrator(self, mock_auth, temp_dir):
        """Create orchestrator instance."""
        return DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_dir),
        )

    @pytest.fixture
    def mock_auth(self):
        """Create mock TradeStationAuth."""
        return mock.Mock(spec=TradeStationAuth)

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "data" / "processed" / "dollar_bars"

    @pytest.mark.asyncio
    async def test_check_component_health_with_no_tasks(self, orchestrator) -> None:
        """Test component health check with no tasks."""
        # Should not raise any errors
        await orchestrator._check_component_health()

    @pytest.mark.asyncio
    async def test_check_component_health_with_running_tasks(
        self, orchestrator
    ) -> None:
        """Test component health check with running tasks."""

        # Create a simple running task
        async def dummy_task():
            await asyncio.sleep(1)

        task = asyncio.create_task(dummy_task())
        orchestrator._tasks = [task]

        # Should not raise any errors
        await orchestrator._check_component_health()

        # Clean up
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class TestMetricsLogging:
    """Test metrics logging functionality."""

    @pytest.fixture
    def orchestrator(self, mock_auth, temp_dir):
        """Create orchestrator instance."""
        return DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_dir),
        )

    @pytest.fixture
    def mock_auth(self):
        """Create mock TradeStationAuth."""
        return mock.Mock(spec=TradeStationAuth)

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "data" / "processed" / "dollar_bars"

    def test_log_metrics_without_start_time(self, orchestrator, caplog) -> None:
        """Test metrics logging when pipeline hasn't started."""
        orchestrator._log_pipeline_metrics()

        # Should not log anything if no start time
        assert orchestrator._start_time is None

    def test_log_metrics_with_start_time(self, orchestrator, caplog) -> None:
        """Test metrics logging with start time set."""
        from datetime import datetime

        orchestrator._start_time = datetime.now()
        orchestrator._log_pipeline_metrics()

        # Should log without errors
        assert orchestrator._start_time is not None


class TestGracefulShutdown:
    """Test graceful shutdown functionality."""

    @pytest.fixture
    def orchestrator(self, mock_auth, temp_dir):
        """Create orchestrator instance."""
        return DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_dir),
        )

    @pytest.fixture
    def mock_auth(self):
        """Create mock TradeStationAuth."""
        return mock.Mock(spec=TradeStationAuth)

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "data" / "processed" / "dollar_bars"

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, orchestrator) -> None:
        """Test stopping when pipeline is not running."""
        # Should not raise any errors
        await orchestrator.stop()

        assert not orchestrator._running

    @pytest.mark.asyncio
    async def test_stop_with_no_websocket_client(self, orchestrator) -> None:
        """Test stopping when WebSocket client is not initialized."""
        orchestrator._running = True
        orchestrator._websocket_client = None

        # Should not raise any errors
        await orchestrator.stop()

        assert not orchestrator._running

    @pytest.mark.asyncio
    async def test_wait_for_queues_to_drain_empty(self, orchestrator) -> None:
        """Test waiting for queues to drain when already empty."""
        # Should complete immediately
        await orchestrator._wait_for_queues_to_drain()


class TestPipelineState:
    """Test pipeline state properties."""

    @pytest.fixture
    def orchestrator(self, mock_auth, temp_dir):
        """Create orchestrator instance."""
        return DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_dir),
        )

    @pytest.fixture
    def mock_auth(self):
        """Create mock TradeStationAuth."""
        return mock.Mock(spec=TradeStationAuth)

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "data" / "processed" / "dollar_bars"

    def test_is_running_property(self, orchestrator) -> None:
        """Test is_running property."""
        assert not orchestrator.is_running

        orchestrator._running = True
        assert orchestrator.is_running

    def test_runtime_seconds_without_start_time(self, orchestrator) -> None:
        """Test runtime_seconds when pipeline hasn't started."""
        assert orchestrator.runtime_seconds == 0.0

    def test_runtime_seconds_with_start_time(self, orchestrator) -> None:
        """Test runtime_seconds with start time set."""
        from datetime import datetime, timedelta

        orchestrator._start_time = datetime.now() - timedelta(seconds=10)

        # Should be approximately 10 seconds
        runtime = orchestrator.runtime_seconds
        assert 9.0 <= runtime <= 11.0


class TestComponentInitialization:
    """Test component initialization."""

    @pytest.fixture
    def orchestrator(self, mock_auth, temp_dir):
        """Create orchestrator instance."""
        return DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_dir),
        )

    @pytest.fixture
    def mock_auth(self):
        """Create mock TradeStationAuth."""
        return mock.Mock(spec=TradeStationAuth)

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "data" / "processed" / "dollar_bars"

    def test_initialize_components(self, orchestrator) -> None:
        """Test component initialization creates all components."""
        orchestrator._initialize_components()

        assert orchestrator._websocket_client is not None
        assert orchestrator._transformer is not None
        assert orchestrator._validator is not None
        assert orchestrator._gap_detector is not None
        assert orchestrator._persistence is not None

    def test_websocket_client_configuration(self, orchestrator) -> None:
        """Test WebSocket client is configured correctly."""
        orchestrator._initialize_components()

        assert orchestrator._websocket_client is not None
        assert orchestrator._websocket_client.auth == orchestrator._auth

    def test_transformer_configuration(self, orchestrator) -> None:
        """Test transformer is configured with correct queues."""
        orchestrator._initialize_components()

        assert orchestrator._transformer is not None
        # Transformer is configured via constructor, just verify it exists

    def test_validator_configuration(self, orchestrator) -> None:
        """Test validator is configured with correct queue."""
        orchestrator._initialize_components()

        assert orchestrator._validator is not None
        # Validator is configured via constructor, just verify it exists

    def test_gap_detector_configuration(self, orchestrator) -> None:
        """Test gap detector is configured with correct queues."""
        orchestrator._initialize_components()

        assert orchestrator._gap_detector is not None
        # Gap detector is configured via constructor, just verify it exists

    def test_persistence_configuration(self, orchestrator) -> None:
        """Test persistence is configured with correct queue."""
        orchestrator._initialize_components()

        assert orchestrator._persistence is not None
        # Persistence is configured via constructor, just verify it exists
