"""ML Prediction Pipeline for Silver Bullet signals.

This module integrates all ML components into the asyncio data pipeline.
Consumes Silver Bullet signals from DetectionPipeline, applies feature
engineering, ML inference, and probability filtering, then publishes
high-probability signals to the execution pipeline.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.data.models import SilverBulletSetup
from src.ml.drift_detector import DriftDetector
from src.ml.inference import MLInference
from src.ml.signal_filter import SignalFilter
from src.ml.walk_forward_optimizer import WalkForwardOptimizer

logger = logging.getLogger(__name__)


class MLStatistics:
    """Tracks ML pipeline statistics for monitoring and logging.

    Attributes:
        signals_processed: Total number of signals processed
        signals_filtered: Number of signals filtered out
        total_probability: Sum of all probability scores (for calculating average)
        latencies: List of recent processing latencies (last 100)
    """

    def __init__(self) -> None:
        """Initialize ML statistics with zero values."""
        self.signals_processed: int = 0
        self.signals_filtered: int = 0
        self.total_probability: float = 0.0
        self.latencies: list[float] = []

    def record_signal(
        self, probability: float, filtered: bool, latency_ms: float
    ) -> None:
        """Record a processed signal with its metrics.

        Args:
            probability: Success probability score from ML inference
            filtered: Whether signal was filtered out
            latency_ms: Total processing latency in milliseconds
        """
        self.signals_processed += 1
        if filtered:
            self.signals_filtered += 1
        self.total_probability += probability
        self.latencies.append(latency_ms)

        # Keep only last 100 latencies
        if len(self.latencies) > 100:
            self.latencies = self.latencies[-100:]

    @property
    def average_probability(self) -> float:
        """Calculate average probability score across all signals.

        Returns:
            Average probability score, or 0.0 if no signals recorded
        """
        if self.signals_processed == 0:
            return 0.0
        return self.total_probability / self.signals_processed

    @property
    def filter_rate(self) -> float:
        """Calculate signal filter rate.

        Returns:
            Proportion of signals filtered (0.0 to 1.0), or 0.0 if no signals
        """
        if self.signals_processed == 0:
            return 0.0
        return self.signals_filtered / self.signals_processed

    @property
    def p50_latency_ms(self) -> float:
        """Calculate median (p50) processing latency.

        Returns:
            Median latency in milliseconds, or 0.0 if no signals
        """
        if not self.latencies:
            return 0.0
        return float(np.median(self.latencies))

    def get_summary(self) -> dict[str, Any]:
        """Get statistics summary as a dictionary.

        Returns:
            Dictionary containing all statistics metrics
        """
        return {
            "signals_processed": self.signals_processed,
            "signals_filtered": self.signals_filtered,
            "average_probability": self.average_probability,
            "filter_rate": self.filter_rate,
            "p50_latency_ms": self.p50_latency_ms,
        }


class MLPipeline:
    """Integrates ML prediction into the data pipeline.

    Consumes Silver Bullet signals from DetectionPipeline, applies feature
    engineering, ML inference, and probability filtering, then publishes
    high-probability signals to the execution pipeline.

    Pipeline Flow:
    1. Consume Silver Bullet signal from input queue
    2. Generate features using FeatureEngineer (internal to MLInference)
    3. Predict success probability using MLInference
    4. Filter by probability threshold using SignalFilter
    5. Publish high-probability signals to output queue
    6. Run background tasks (retraining, drift monitoring)

    Attributes:
        inference: ML inference engine
        signal_filter: Probability-based signal filter
        drift_detector: Model drift monitoring
        optimizer: Walk-forward optimization
        statistics: ML pipeline statistics tracker

    Performance:
        Total latency < 50ms from signal input to filtered output
    """

    MAX_QUEUE_SIZE = 100  # Prevent memory buildup

    def __init__(
        self,
        input_queue: asyncio.Queue[SilverBulletSetup] | None = None,
        output_queue: asyncio.Queue[SilverBulletSetup] | None = None,
        model_dir: str | Path = "models/xgboost",
    ) -> None:
        """Initialize the ML pipeline.

        Args:
            input_queue: Queue consuming signals from DetectionPipeline.
                If None, creates new queue with MAX_QUEUE_SIZE.
            output_queue: Queue publishing filtered signals to execution.
                If None, creates new queue with MAX_QUEUE_SIZE.
            model_dir: Directory containing ML models and artifacts
        """
        # Create queues if not provided (enforces size limits for overflow protection)
        self._input_queue = (
            input_queue
            if input_queue is not None
            else asyncio.Queue(maxsize=self.MAX_QUEUE_SIZE)
        )
        self._output_queue = (
            output_queue if output_queue is not None else asyncio.Queue(maxsize=self.MAX_QUEUE_SIZE)
        )
        self._model_dir = Path(model_dir)

        # Initialize ML components
        self._inference = MLInference(model_dir=model_dir)
        self._signal_filter = SignalFilter(model_dir=model_dir)
        self._drift_detector = DriftDetector(model_dir=model_dir)
        self._optimizer = WalkForwardOptimizer(model_dir=model_dir)

        # Statistics tracking
        self._statistics = MLStatistics()

        # Background tasks
        self._retraining_task: asyncio.Task | None = None
        self._drift_monitoring_task: asyncio.Task | None = None

        logger.info("MLPipeline initialized with ML components")

    async def process_signal(self, signal: SilverBulletSetup) -> None:
        """Process a Silver Bullet signal through the ML pipeline.

        Args:
            signal: Silver Bullet setup from DetectionPipeline

        Performance:
            Total latency < 50ms (feature engineering + inference + filtering)
        """
        start_time = time.perf_counter()

        try:
            # Step 1: ML Inference (includes feature engineering internally)
            inference_result = self._inference.predict_probability(
                signal=signal, horizon=5  # 5-minute horizon
            )

            probability = inference_result["probability"]
            inference_latency = inference_result["latency_ms"]

            # Step 2: Signal Filtering
            filter_result = self._signal_filter.filter_signal(
                signal=signal, probability=probability
            )

            # Step 3: Publish if allowed
            if filter_result["allowed"]:
                await self._output_queue.put(signal)
                logger.debug(
                    f"Signal allowed: P(Success)={probability:.2%}, "
                    f"latency={inference_latency:.2f}ms"
                )
            else:
                logger.debug(
                    f"Signal filtered: P(Success)={probability:.2%} < threshold"
                )

            # Step 4: Track statistics
            total_latency_ms = (time.perf_counter() - start_time) * 1000
            self._statistics.record_signal(
                probability=probability,
                filtered=not filter_result["allowed"],
                latency_ms=total_latency_ms,
            )

            # Log statistics every 100 signals
            if self._statistics.signals_processed % 100 == 0:
                logger.info(self._statistics.get_summary())

        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            # Continue processing next signal (don't crash pipeline)

    def start_background_tasks(self) -> None:
        """Start background tasks for retraining and drift monitoring."""
        logger.info("Starting ML pipeline background tasks...")

        # Create background tasks
        self._retraining_task = asyncio.create_task(self._run_weekly_retraining())
        self._drift_monitoring_task = asyncio.create_task(
            self._check_drift_periodically()
        )

        logger.info("Background tasks started: weekly_retraining, drift_monitoring")

    async def stop_background_tasks(self) -> None:
        """Stop background tasks and cleanup."""
        logger.info("Stopping ML pipeline background tasks...")

        # Cancel background tasks
        if self._retraining_task:
            self._retraining_task.cancel()
        if self._drift_monitoring_task:
            self._drift_monitoring_task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(
            self._retraining_task,
            self._drift_monitoring_task,
            return_exceptions=True,
        )

        logger.info("Background tasks stopped")

    async def _run_weekly_retraining(self) -> None:
        """Run weekly retraining as background task."""
        while True:
            try:
                # Sleep for 1 week (7 days * 24 hours * 3600 seconds)
                await asyncio.sleep(7 * 24 * 3600)

                logger.info("Starting weekly walk-forward optimization...")
                result = await self._optimizer.run_retraining_with_retry()

                if result.get("success"):
                    logger.info("Weekly retraining completed successfully")
                else:
                    logger.error(
                        f"Weekly retraining failed: {result.get('error')}"
                    )

            except asyncio.CancelledError:
                logger.info("Weekly retraining task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in weekly retraining: {e}")
                # Continue and retry next week

    async def _check_drift_periodically(self) -> None:
        """Check for model drift periodically."""
        while True:
            try:
                # Check every hour
                await asyncio.sleep(3600)

                # Get actual win rate from signal filter
                actual_win_rate = self._signal_filter.calculate_win_rate()
                expected_win_rate = self._load_expected_win_rate()

                if actual_win_rate is not None and expected_win_rate is not None:
                    drift_result = self._drift_detector.check_drift(
                        actual_win_rate, expected_win_rate
                    )

                    if drift_result["drift_detected"]:
                        drift_id = self._drift_detector.track_drift_event(
                            actual_win_rate=actual_win_rate,
                            expected_win_rate=expected_win_rate,
                            severity=drift_result["severity"],
                        )
                        logger.warning(
                            "Model drift detected: {:.2%} "
                            "(actual={:.2%}, expected={:.2%}), drift_id={}".format(
                                drift_result["drift_magnitude"],
                                actual_win_rate,
                                expected_win_rate,
                                drift_id,
                            ),
                        )

                        # Check if trading should halt
                        should_halt, reason = self._drift_detector.should_halt_trading()
                        if should_halt:
                            logger.critical(f"TRADING HALT RECOMMENDED: {reason}")

            except asyncio.CancelledError:
                logger.info("Drift monitoring task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in drift monitoring: {e}")
                # Continue and check again next hour

    def _load_expected_win_rate(self) -> float | None:
        """Load expected win rate from model metadata.

        Returns:
            Expected win rate from model metadata, or None if not available
        """
        try:
            import json

            metadata_file = self._model_dir / "5_minute" / "metadata.json"
            if not metadata_file.exists():
                return None

            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            # Extract win rate from metrics
            metrics = metadata.get("metrics", {})
            return metrics.get("win_rate")

        except Exception as e:
            logger.error(f"Error loading expected win rate: {e}")
            return None

    async def start(self) -> None:
        """Start the ML pipeline and background tasks."""
        logger.info("Starting MLPipeline...")

        # Start background tasks
        self.start_background_tasks()

        # Start signal processing loop
        try:
            await self._processing_loop()
        except asyncio.CancelledError:
            logger.info("MLPipeline processing cancelled")
        finally:
            await self.stop_background_tasks()

    async def stop(self) -> None:
        """Stop the ML pipeline and cleanup."""
        logger.info("Stopping MLPipeline...")
        await self.stop_background_tasks()
        logger.info("MLPipeline stopped")

    async def _processing_loop(self) -> None:
        """Main processing loop for signals."""
        while True:
            signal = await self._input_queue.get()
            await self.process_signal(signal)

    def health_check(self) -> dict[str, Any]:
        """Check health status of ML pipeline.

        Returns:
            Dictionary containing health status with keys:
            - healthy: bool - Overall health status
            - inference_loaded: bool - Whether ML inference model is loaded
            - optimizer_scheduler_running: bool - Whether optimizer scheduler is running
            - queue_depth: dict - Current queue depths
            - statistics: dict - Current statistics summary
        """
        # Check if inference model is loaded
        inference_loaded = (
            self._inference._model is not None
            and self._inference._pipeline is not None
        )

        # Check if optimizer scheduler is running
        optimizer_scheduler_running = (
            self._optimizer._scheduler.running if self._optimizer._scheduler else False
        )

        # Get queue depths
        queue_depth = {
            "input_queue_size": self._input_queue.qsize(),
            "output_queue_size": self._output_queue.qsize(),
            "input_queue_full": self._input_queue.full(),
            "output_queue_full": self._output_queue.full(),
        }

        # Determine overall health
        healthy = (
            inference_loaded
            and not queue_depth["input_queue_full"]
            and not queue_depth["output_queue_full"]
        )

        return {
            "healthy": healthy,
            "inference_loaded": inference_loaded,
            "optimizer_scheduler_running": optimizer_scheduler_running,
            "queue_depth": queue_depth,
            "statistics": self._statistics.get_summary(),
        }
