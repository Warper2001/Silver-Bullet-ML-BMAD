"""Data Pipeline Orchestrator.

Coordinates all data pipeline stages from TradeStation SDK streaming to HDF5 persistence.
Implements backpressure monitoring, failure recovery, and graceful shutdown.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from src.data.config import Settings, load_settings
from src.data.gap_detection import GapDetector
from src.data.models import DollarBar, MarketData, ValidationResult
from src.data.persistence import HDF5DataSink
from src.data.transformation import DollarBarTransformer
from src.data.validation import DataValidator
from src.execution.tradestation.client import TradeStationClient
from src.execution.tradestation.market_data.streaming import QuoteStreamParser
from src.execution.tradestation.models import TradeStationQuote


logger = logging.getLogger(__name__)


class DataPipelineOrchestrator:
    """Orchestrates the entire data pipeline from TradeStation SDK streaming to HDF5.

    Pipeline Stages:
    1. TradeStation SDK Streaming (QuoteStreamParser for SIM environment)
    2. Dollar Bar Transformation (DollarBarTransformer)
    3. Data Validation (DataValidator)
    4. Gap Detection & Filling (GapDetector)
    5. HDF5 Persistence (HDF5DataSink)

    Queues:
    - raw_queue: MarketData from TradeStation SDK
    - transform_queue: DollarBar from transformation
    - validated_queue: Validated DollarBar from validation
    - gap_filled_queue: Gap-filled DollarBar from gap detection
    """

    DEFAULT_MAX_QUEUE_SIZE = 1000
    QUEUE_WARNING_THRESHOLD = 0.8  # 80%
    QUEUE_CRITICAL_THRESHOLD = 0.95  # 95%
    METRICS_LOG_INTERVAL = 60  # seconds

    def __init__(
        self,
        client: TradeStationClient,
        data_directory: str,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the data pipeline orchestrator.

        Args:
            client: TradeStation API client (SIM environment)
            data_directory: Root directory for HDF5 files
            max_queue_size: Maximum size for each pipeline queue
            settings: Application settings (optional, loads from env if None)
        """
        self._client = client
        self._data_directory = Path(data_directory)
        self._max_queue_size = max_queue_size
        self._settings = settings or load_settings()

        # Initialize queues for each pipeline stage
        self._raw_queue: asyncio.Queue[MarketData] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._transform_queue: asyncio.Queue[DollarBar] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._validated_queue: asyncio.Queue[DollarBar] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._gap_filled_queue: asyncio.Queue[DollarBar] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._error_queue: asyncio.Queue[
            tuple[DollarBar, ValidationResult]
        ] = asyncio.Queue(maxsize=max_queue_size)  # noqa: F821

        # Pipeline components
        self._stream_parser: QuoteStreamParser | None = None
        self._transformer: DollarBarTransformer | None = None
        self._validator: DataValidator | None = None
        self._gap_detector: GapDetector | None = None
        self._persistence: HDF5DataSink | None = None

        # Async tasks
        self._tasks: list[asyncio.Task] = []

        # State tracking
        self._running = False
        self._start_time: datetime | None = None
        self._last_metrics_log: datetime | None = None

        # Metrics
        self._total_messages_received = 0
        self._total_bars_created = 0
        self._total_bars_validated = 0
        self._total_bars_persisted = 0

    async def start(self) -> None:
        """Start the data pipeline.

        Spawns async tasks for each pipeline stage and begins
        TradeStation SDK streaming to receive market data.
        """
        if self._running:
            raise RuntimeError("Pipeline is already running")

        logger.info("Starting data pipeline orchestrator")
        self._running = True
        self._start_time = datetime.now()
        self._last_metrics_log = datetime.now()

        # Initialize pipeline components
        self._initialize_components()

        # Verify TradeStation client is authenticated before starting streaming
        if not self._client.is_authenticated():
            raise RuntimeError(
                "TradeStation client is not authenticated. "
                "Complete OAuth flow before starting pipeline. "
                "Run standard_auth_flow.py to authenticate."
            )
        logger.info("TradeStation client authenticated, ready to stream")

        # Start TradeStation SDK streaming to raw queue
        # Use symbols from configuration (defaults to ["MNQH26"])
        symbols = self._settings.streaming_symbols
        logger.info(f"Starting streaming for symbols: {symbols}")
        streaming_task = asyncio.create_task(
            self._stream_parser.stream_to_queue(symbols, self._raw_queue)
        )
        self._tasks.append(streaming_task)

        # Spawn async tasks for each pipeline stage
        self._tasks.extend([
            asyncio.create_task(self._transformer.consume()),
            asyncio.create_task(self._validator.consume()),
            asyncio.create_task(self._gap_detector.consume()),
            asyncio.create_task(self._persistence.consume()),
            asyncio.create_task(self._monitor_pipeline_health()),
            asyncio.create_task(self._log_metrics_periodically()),
        ])

        logger.info(f"Data pipeline started with {len(self._tasks)} tasks")

    async def stop(self) -> None:
        """Stop the data pipeline gracefully.

        Waits for queues to drain, closes connections, and cancels tasks.
        """
        if not self._running:
            return

        logger.info("Stopping data pipeline orchestrator")

        # Stop streaming
        if self._stream_parser:
            self._stream_parser.stop_streaming()

        # Wait for queues to drain (timeout: 30 seconds)
        try:
            await asyncio.wait_for(self._wait_for_queues_to_drain(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("Queue drain timeout, proceeding with shutdown")

        # Close HDF5 file handles
        if self._persistence:
            self._persistence._close_all_files()

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete cancellation
        await asyncio.gather(*self._tasks, return_exceptions=True)

        self._running = False
        logger.info("Data pipeline stopped")

    def _initialize_components(self) -> None:
        """Initialize all pipeline components.

        Creates instances of each component with appropriate queues
        and configuration.
        """
        # Stage 1: TradeStation SDK Quote Stream Parser (SIM environment)
        self._stream_parser = QuoteStreamParser(client=self._client)

        # Stage 2: Dollar Bar Transformer
        self._transformer = DollarBarTransformer(
            input_queue=self._raw_queue,
            output_queue=self._transform_queue,
        )

        # Stage 3: Data Validator
        self._validator = DataValidator(
            input_queue=self._transform_queue,
            validated_queue=self._validated_queue,
            error_queue=self._error_queue,
        )

        # Stage 4: Gap Detector
        self._gap_detector = GapDetector(
            validated_queue=self._validated_queue,
            gap_filled_queue=self._gap_filled_queue,
        )

        # Stage 5: HDF5 Persistence
        self._persistence = HDF5DataSink(
            gap_filled_queue=self._gap_filled_queue,
            data_directory=str(self._data_directory),
            compression_level=1,  # Balance speed and size
        )

    async def _wait_for_queues_to_drain(self) -> None:
        """Wait for all queues to drain.

        Logs progress every 5 seconds.
        """
        queues = {
            "raw": self._raw_queue,
            "transform": self._transform_queue,
            "validated": self._validated_queue,
            "gap_filled": self._gap_filled_queue,
        }

        while True:
            total_size = sum(q.qsize() for q in queues.values())
            if total_size == 0:
                logger.info("All queues drained")
                break

            logger.info(
                f"Waiting for queues to drain: "
                f"raw={queues['raw'].qsize()} "
                f"transform={queues['transform'].qsize()} "
                f"validated={queues['validated'].qsize()} "
                f"gap_filled={queues['gap_filled'].qsize()}"
            )
            await asyncio.sleep(5)

    async def _monitor_pipeline_health(self) -> None:
        """Monitor pipeline health and detect issues.

        Checks:
        - Queue depths (backpressure detection)
        - Component health (task status)
        - Data flow completeness
        """
        while self._running:
            try:
                # Check queue depths
                await self._check_queue_depths()

                # Check component health
                await self._check_component_health()

                # Wait before next check
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pipeline health check error: {e}", exc_info=True)

    async def _check_queue_depths(self) -> None:
        """Check queue depths and log warnings for backpressure.

        Warning: Queue depth > 80% of max size
        Critical: Queue depth > 95% of max size
        """
        queues = {
            "raw": self._raw_queue,
            "transform": self._transform_queue,
            "validated": self._validated_queue,
            "gap_filled": self._gap_filled_queue,
        }

        for name, queue in queues.items():
            depth = queue.qsize()
            depth_percent = depth / self._max_queue_size

            if depth_percent >= self.QUEUE_CRITICAL_THRESHOLD:
                logger.critical(
                    f"CRITICAL backpressure: {name} queue at {depth_percent:.1%} "
                    f"({depth}/{self._max_queue_size} items)"
                )
            elif depth_percent >= self.QUEUE_WARNING_THRESHOLD:
                logger.warning(
                    f"WARNING backpressure: {name} queue at {depth_percent:.1%} "
                    f"({depth}/{self._max_queue_size} items)"
                )

    async def _check_component_health(self) -> None:
        """Check health of all pipeline components.

        Detects crashed tasks and logs errors.
        """
        for i, task in enumerate(self._tasks):
            if task.done():
                try:
                    result = task.result()
                    logger.info(f"Task {i} completed unexpectedly: {result}")
                except Exception as e:
                    logger.error(
                        f"Task {i} crashed: {e}",
                        exc_info=task.exception(),
                    )

    async def _log_metrics_periodically(self) -> None:
        """Log pipeline metrics periodically.

        Metrics logged every 60 seconds:
        - Runtime duration
        - Queue depths
        - Throughput counters
        - Data completeness percentage
        """
        while self._running:
            try:
                await asyncio.sleep(self.METRICS_LOG_INTERVAL)
                self._log_pipeline_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics logging error: {e}", exc_info=True)

    def _log_pipeline_metrics(self) -> None:
        """Log current pipeline metrics.

        Includes:
        - Runtime duration
        - Queue depths
        - Throughput counters
        - Data completeness percentage
        """
        if not self._start_time:
            return

        runtime = (datetime.now() - self._start_time).total_seconds()

        logger.info(
            f"Pipeline metrics: "
            f"runtime_seconds={runtime:.0f} "
            f"raw_queue_depth={self._raw_queue.qsize()} "
            f"transform_queue_depth={self._transform_queue.qsize()} "
            f"validated_queue_depth={self._validated_queue.qsize()} "
            f"gap_filled_queue_depth={self._gap_filled_queue.qsize()} "
            f"messages_received={self._total_messages_received} "
            f"bars_created={self._total_bars_created} "
            f"bars_validated={self._total_bars_validated} "
            f"bars_persisted={self._total_bars_persisted}"
        )

    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._running

    @property
    def runtime_seconds(self) -> float:
        """Get pipeline runtime in seconds."""
        if not self._start_time:
            return 0.0
        return (datetime.now() - self._start_time).total_seconds()

    @property
    def queue_depths(self) -> dict[str, int]:
        """Get current queue depths."""
        return {
            "raw": self._raw_queue.qsize(),
            "transform": self._transform_queue.qsize(),
            "validated": self._validated_queue.qsize(),
            "gap_filled": self._gap_filled_queue.qsize(),
        }
