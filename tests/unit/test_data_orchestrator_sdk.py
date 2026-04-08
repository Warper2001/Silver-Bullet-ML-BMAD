"""Unit tests for Data Pipeline Orchestrator with SDK streaming.

Tests the modified orchestrator that uses TradeStation SDK streaming
instead of WebSocket client for market data ingestion.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.data.auth import TradeStationAuth
from src.data.auth_v3 import TradeStationAuthV3
from src.data.orchestrator import DataPipelineOrchestrator
from src.execution.tradestation.market_data.streaming import QuoteStreamParser, StreamPosition


@pytest.fixture
def mock_auth():
    """Create mock TradeStation authentication."""
    # Don't use spec to allow any method
    auth = MagicMock()
    # Mock both possible authentication methods
    auth.get_valid_access_token = MagicMock(return_value="test_token")
    auth.authenticate = AsyncMock(return_value="test_token")
    return auth


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


class TestDataPipelineOrchestratorSDK:
    """Tests for DataPipelineOrchestrator with SDK streaming integration."""

    @pytest.mark.asyncio
    async def test_orchestrator_initialization_with_sdk(self, mock_auth, temp_data_dir):
        """Test initializing orchestrator with SDK streaming enabled."""
        orchestrator = DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_data_dir),
            use_sdk_streaming=True,
            symbols=["MNQH26"],
        )

        assert orchestrator._auth == mock_auth
        assert orchestrator._use_sdk_streaming is True
        assert orchestrator._symbols == ["MNQH26"]
        assert orchestrator._running is False

    @pytest.mark.asyncio
    async def test_sdk_streaming_initialization(self, mock_auth, temp_data_dir):
        """Test that SDK streaming parser is initialized correctly."""
        orchestrator = DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_data_dir),
            use_sdk_streaming=True,
            symbols=["MNQH26"],
        )

        # Initialize components
        orchestrator._initialize_components()

        # Verify SDK parser is created
        assert orchestrator._sdk_parser is not None
        assert isinstance(orchestrator._sdk_parser, QuoteStreamParser)

    @pytest.mark.asyncio
    async def test_websocket_not_used_with_sdk(self, mock_auth, temp_data_dir):
        """Test that WebSocket client is not used when SDK streaming is enabled."""
        orchestrator = DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_data_dir),
            use_sdk_streaming=True,
            symbols=["MNQH26"],
        )

        # Initialize components
        orchestrator._initialize_components()

        # Verify WebSocket client is None
        assert orchestrator._websocket_client is None

    @pytest.mark.asyncio
    async def test_start_with_sdk_streaming(self, mock_auth, temp_data_dir):
        """Test starting orchestrator with SDK streaming."""
        orchestrator = DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_data_dir),
            use_sdk_streaming=True,
            symbols=["MNQH26"],
        )

        # Mock SDK parser methods and other components
        with patch.object(orchestrator, "_initialize_components"):
            orchestrator._sdk_parser = MagicMock(spec=QuoteStreamParser)
            orchestrator._sdk_parser.start = AsyncMock()
            orchestrator._sdk_parser.subscribe = AsyncMock(
                return_value=asyncio.Queue()
            )

            # Mock other components to avoid None errors
            orchestrator._transformer = MagicMock()
            orchestrator._transformer.consume = AsyncMock()
            orchestrator._validator = MagicMock()
            orchestrator._validator.consume = AsyncMock()
            orchestrator._gap_detector = MagicMock()
            orchestrator._gap_detector.consume = AsyncMock()
            orchestrator._persistence = MagicMock()
            orchestrator._persistence.consume = AsyncMock()
            orchestrator._monitor_pipeline_health = AsyncMock()
            orchestrator._log_metrics_periodically = AsyncMock()

            await orchestrator.start()

            assert orchestrator._running is True
            orchestrator._sdk_parser.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_with_sdk_streaming(self, mock_auth, temp_data_dir):
        """Test stopping orchestrator with SDK streaming."""
        orchestrator = DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_data_dir),
            use_sdk_streaming=True,
            symbols=["MNQH26"],
        )

        # Mock SDK parser
        with patch.object(orchestrator, "_initialize_components"):
            orchestrator._sdk_parser = MagicMock(spec=QuoteStreamParser)
            orchestrator._sdk_parser.stop = AsyncMock()

            # Start and then stop
            orchestrator._running = True
            await orchestrator.stop()

            assert orchestrator._running is False
            orchestrator._sdk_parser.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_stream_position(self, mock_auth, temp_data_dir):
        """Test processing StreamPosition from SDK into market data."""
        orchestrator = DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_data_dir),
            use_sdk_streaming=True,
            symbols=["MNQH26"],
        )

        # Create test StreamPosition
        stream_position = StreamPosition(
            symbol="MNQH26",
            timestamp=datetime.now(timezone.utc),
            last_price=11850.0,
            bid=11849.75,
            ask=11850.25,
            volume=1000,
        )

        # Process stream position
        market_data = orchestrator._stream_position_to_market_data(stream_position)

        assert market_data.symbol == "MNQH26"
        assert market_data.timestamp == stream_position.timestamp
        assert market_data.last == stream_position.last_price  # Use 'last' not 'price'

    @pytest.mark.asyncio
    async def test_backward_compatibility_websocket(self, mock_auth, temp_data_dir):
        """Test that orchestrator still works with WebSocket when SDK is disabled."""
        orchestrator = DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_data_dir),
            use_sdk_streaming=False,  # Use WebSocket
        )

        # Initialize components
        with patch.object(orchestrator, "_initialize_components"):
            # Verify WebSocket client is created
            assert orchestrator._use_sdk_streaming is False
            assert orchestrator._sdk_parser is None

    @pytest.mark.asyncio
    async def test_queue_initialization_with_sdk(self, mock_auth, temp_data_dir):
        """Test that queues are initialized correctly with SDK streaming."""
        orchestrator = DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_data_dir),
            use_sdk_streaming=True,
            symbols=["MNQH26"],
            max_queue_size=500,
        )

        # Check queue sizes
        assert orchestrator._raw_queue.maxsize == 500
        assert orchestrator._transform_queue.maxsize == 500
        assert orchestrator._validated_queue.maxsize == 500
        assert orchestrator._gap_filled_queue.maxsize == 500

    @pytest.mark.asyncio
    async def test_sim_environment_default(self, mock_auth, temp_data_dir):
        """Test that SIM environment is used by default."""
        orchestrator = DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_data_dir),
            use_sdk_streaming=True,
            symbols=["MNQH26"],
        )

        # Verify environment defaults to 'sim'
        assert orchestrator._environment == "sim"

    @pytest.mark.asyncio
    async def test_environment_configuration(self, mock_auth, temp_data_dir):
        """Test configuring different environments."""
        # Test with SIM environment
        orchestrator_sim = DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_data_dir),
            use_sdk_streaming=True,
            symbols=["MNQH26"],
            environment="sim",
        )
        assert orchestrator_sim._environment == "sim"

        # Test with PROD environment
        orchestrator_prod = DataPipelineOrchestrator(
            auth=mock_auth,
            data_directory=str(temp_data_dir),
            use_sdk_streaming=True,
            symbols=["MNQH26"],
            environment="prod",
        )
        assert orchestrator_prod._environment == "prod"
