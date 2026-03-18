"""Monitoring Integration for asyncio data pipeline.

Integrates all monitoring components into data pipeline with background
tasks for health checks, resource monitoring, state persistence, and
daily report generation.
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List


class MonitoringIntegration:
    """Integrate all monitoring components into data pipeline.

    Manages background tasks for health checks, resource monitoring,
    state persistence, and coordinates all monitoring components.
    """

    def __init__(
        self,
        health_check_manager=None,
        resource_monitor=None,
        crash_recovery=None,
        audit_trail=None,
        staleness_detector=None,
        notification_integration=None,
        daily_report_generator=None,
        error_logger=None,
        graceful_shutdown_manager=None,
        state_file_path: str = "data/state/system_state.json"
    ):
        """Initialize monitoring integration.

        Args:
            health_check_manager: Health check manager
            resource_monitor: Resource monitor (CPU, memory, disk)
            crash_recovery: Crash recovery manager
            audit_trail: Immutable audit trail
            staleness_detector: Data staleness detector
            notification_integration: Notification integration
            daily_report_generator: Daily report generator
            error_logger: Error logger with stack traces
            graceful_shutdown_manager: Graceful shutdown manager
            state_file_path: Path to state file for persistence
        """
        self._health_check_manager = health_check_manager
        self._resource_monitor = resource_monitor
        self._crash_recovery = crash_recovery
        self._audit_trail = audit_trail
        self._staleness_detector = staleness_detector
        self._notification_integration = notification_integration
        self._daily_report_generator = daily_report_generator
        self._error_logger = error_logger
        self._graceful_shutdown = graceful_shutdown_manager
        self._state_file_path = state_file_path

        # Background tasks
        self._health_check_task = None
        self._resource_monitor_task = None
        self._state_persistence_task = None
        self._daily_report_task = None

        # Monitoring statistics
        self._health_check_count = 0
        self._health_check_pass_count = 0
        self._error_count = 0
        self._recovery_count = 0
        self._total_uptime_seconds = 0
        self._start_time = None

    async def start(self) -> None:
        """Start all background monitoring tasks.

        Creates async tasks for:
        - Health checks every 5 seconds
        - Resource monitoring every 10 seconds
        - State persistence every 10 seconds
        - Daily report generation at 4 PM EST

        Example:
            >>> monitoring = MonitoringIntegration(...)
            >>> await monitoring.start()
        """
        self._start_time = self._get_current_time()

        # Start health check task (every 5 seconds)
        self._health_check_task = asyncio.create_task(
            self._run_health_checks_loop()
        )

        # Start resource monitoring task (every 10 seconds)
        self._resource_monitor_task = asyncio.create_task(
            self._run_resource_monitoring_loop()
        )

        # Start state persistence task (every 10 seconds)
        self._state_persistence_task = asyncio.create_task(
            self._run_state_persistence_loop()
        )

        # Start daily report task (runs at 4 PM EST)
        self._daily_report_task = asyncio.create_task(
            self._run_daily_report_loop()
        )

        # Setup signal handlers for graceful shutdown
        if self._graceful_shutdown:
            from src.monitoring.graceful_shutdown_manager import (
                setup_signal_handlers
            )
            setup_signal_handlers(self._graceful_shutdown)

    async def stop(self) -> None:
        """Stop all background monitoring tasks.

        Cancels all background tasks and waits for completion.

        Example:
            >>> await monitoring.stop()
        """
        # Cancel all tasks
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._resource_monitor_task:
            self._resource_monitor_task.cancel()
        if self._state_persistence_task:
            self._state_persistence_task.cancel()
        if self._daily_report_task:
            self._daily_report_task.cancel()

        # Wait for tasks to complete (filter None tasks)
        tasks = [
            task for task in [
                self._health_check_task,
                self._resource_monitor_task,
                self._state_persistence_task,
                self._daily_report_task
            ] if task is not None
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def log_action(
        self,
        action: str,
        component: str,
        target: str,
        metadata: Dict = None
    ) -> None:
        """Log action to audit trail.

        Args:
            action: Action performed (e.g., "ORDER_SUBMIT")
            component: Component performing action
            target: Target of action
            metadata: Additional metadata
        """
        if self._audit_trail:
            self._audit_trail.log_action(
                action, component, target, metadata or {}
            )

    async def log_error(
        self,
        error: Exception,
        context: Dict = None,
        severity: str = "ERROR"
    ) -> str:
        """Log error with full context.

        Args:
            error: Exception object
            context: Additional context
            severity: ERROR, WARNING, CRITICAL

        Returns:
            Error ID
        """
        self._error_count += 1

        if self._error_logger:
            error_id = self._error_logger.log_error(
                error, context=context, severity=severity
            )

            # Send notification for CRITICAL errors
            if severity == "CRITICAL" and self._notification_integration:
                await self._send_critical_notification_async(error)

            return error_id

        return "UNKNOWN"

    async def check_data_staleness(self, last_timestamp) -> bool:
        """Check if data is stale.

        Args:
            last_timestamp: Last data packet timestamp

        Returns:
            True if stale, False otherwise
        """
        if self._staleness_detector:
            is_stale = self._staleness_detector.is_data_stale(last_timestamp)

            if is_stale:
                await self.log_action(
                    "DATA_STALENESS_DETECTED",
                    "staleness_detector",
                    "data_pipeline",
                    {"last_timestamp": last_timestamp.isoformat()}
                )

            return is_stale

        return False

    async def persist_state(self) -> None:
        """Persist current system state to disk.

        Saves positions, orders, model status, configuration.
        """
        # Create state directory
        state_path = Path(self._state_file_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)

        # Gather state from all components
        state = {
            "timestamp": self._get_current_time().isoformat(),
            "positions": await self._get_positions(),
            "orders": await self._get_orders(),
            "model_status": await self._get_model_status(),
            "configuration": await self._get_configuration(),
            "health_checks": self._get_health_check_stats(),
            "resource_usage": self._get_resource_stats(),
            "uptime_seconds": self._total_uptime_seconds
        }

        # Write to file
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

    def get_monitoring_statistics(self) -> Dict:
        """Get monitoring statistics.

        Returns:
            Dictionary with monitoring statistics
        """
        uptime = (
            (self._get_current_time() - self._start_time).total_seconds()
            if self._start_time
            else 0
        )

        return {
            "uptime_seconds": uptime,
            "health_check_count": self._health_check_count,
            "health_check_pass_count": self._health_check_pass_count,
            "health_check_pass_rate": (
                self._health_check_pass_count / self._health_check_count
                if self._health_check_count > 0
                else 0
            ),
            "error_count": self._error_count,
            "recovery_count": self._recovery_count,
            "avg_cpu_usage": self._get_avg_cpu_usage(),
            "avg_memory_usage": self._get_avg_memory_usage()
        }

    async def _run_health_checks_loop(self) -> None:
        """Run health checks every 5 seconds."""
        while True:
            try:
                # Check if shutdown requested
                if self._graceful_shutdown and \
                   self._graceful_shutdown.is_shutdown_requested():
                    break

                # Run health checks
                if self._health_check_manager:
                    result = self._health_check_manager.run_health_checks()
                    self._health_check_count += 1

                    if result["overall_status"] == "healthy":
                        self._health_check_pass_count += 1

                    # Log unhealthy status
                    if result["overall_status"] != "healthy":
                        await self.log_action(
                            "HEALTH_CHECK_FAILED",
                            "health_check_manager",
                            "system",
                            {"result": result}
                        )

                        # Attempt recovery if unhealthy
                        if self._crash_recovery:
                            recovered = self._crash_recovery.recover()
                            if recovered:
                                self._recovery_count += 1

                # Wait 5 seconds
                await asyncio.sleep(5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.log_error(e, severity="ERROR")
                await asyncio.sleep(5)

    async def _run_resource_monitoring_loop(self) -> None:
        """Run resource monitoring every 10 seconds."""
        while True:
            try:
                # Check if shutdown requested
                if self._graceful_shutdown and \
                   self._graceful_shutdown.is_shutdown_requested():
                    break

                # Monitor resources
                if self._resource_monitor:
                    stats = self._resource_monitor.get_current_stats()

                    # Check for critical resource usage
                    if stats.get("memory_percent", 0) > 90 or \
                       stats.get("cpu_percent", 0) > 95:
                        await self.log_action(
                            "RESOURCE_CRITICAL",
                            "resource_monitor",
                            "system",
                            {"stats": stats}
                        )

                        # Send notification
                        if self._notification_integration:
                            self._notification_integration.send_notification(
                                "RESOURCE_CRITICAL",
                                "CRITICAL",
                                "Critical Resource Usage",
                                "CPU: {}%, Memory: {}%".format(
                                    stats.get("cpu_percent", 0),
                                    stats.get("memory_percent", 0)
                                )
                            )

                # Wait 10 seconds
                await asyncio.sleep(10)

            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.log_error(e, severity="ERROR")
                await asyncio.sleep(10)

    async def _run_state_persistence_loop(self) -> None:
        """Persist state every 10 seconds."""
        while True:
            try:
                # Check if still accepting new data
                if self._graceful_shutdown and \
                   self._graceful_shutdown.is_accepting_new_data():
                    await self.persist_state()

                # Wait 10 seconds
                await asyncio.sleep(10)

            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.log_error(e, severity="ERROR")
                await asyncio.sleep(10)

    async def _run_daily_report_loop(self) -> None:
        """Generate daily report at 4 PM EST."""
        while True:
            try:
                # Check if shutdown requested
                if self._graceful_shutdown and \
                   self._graceful_shutdown.is_shutdown_requested():
                    break

                # Get current time in EST
                utc_now = self._get_current_time()
                est_now = utc_now - timedelta(hours=5)

                # Check if it's 4 PM EST (within 1 minute window)
                if est_now.hour == 16 and est_now.minute == 0:
                    if self._daily_report_generator:
                        trade_date = est_now.strftime("%Y-%m-%d")
                        self._daily_report_generator.generate_report(
                            trade_date
                        )

                    # Wait until next day (4 PM EST tomorrow)
                    seconds_until_next = (
                        (24 - 16) * 3600 + 16 * 3600
                    )
                    await asyncio.sleep(seconds_until_next)
                else:
                    # Calculate seconds until 4 PM EST today
                    target_hour = 16
                    target_minute = 0
                    seconds_until_target = (
                        (target_hour - est_now.hour) * 3600 +
                        (target_minute - est_now.minute) * 60 -
                        est_now.second
                    )

                    # If already past 4 PM, wait until tomorrow
                    if seconds_until_target < 0:
                        seconds_until_target += 24 * 3600

                    await asyncio.sleep(seconds_until_target)

            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.log_error(e, severity="ERROR")
                await asyncio.sleep(60)

    async def _send_critical_notification_async(self, error: Exception) -> None:
        """Send critical error notification asynchronously.

        Args:
            error: Exception object
        """
        if self._notification_integration:
            self._notification_integration.send_notification(
                "CRITICAL_ERROR",
                "CRITICAL",
                "Critical Error: {}".format(type(error).__name__),
                str(error)
            )

    async def _get_positions(self) -> List[Dict]:
        """Get current positions.

        Returns:
            List of position dictionaries
        """
        # In production, query position manager
        return []

    async def _get_orders(self) -> List[Dict]:
        """Get pending orders.

        Returns:
            List of order dictionaries
        """
        # In production, query order manager
        return []

    async def _get_model_status(self) -> Dict:
        """Get model status.

        Returns:
            Model status dictionary
        """
        # In production, query ML components
        return {}

    async def _get_configuration(self) -> Dict:
        """Get system configuration.

        Returns:
            Configuration dictionary
        """
        # In production, read config
        return {}

    def _get_health_check_stats(self) -> Dict:
        """Get health check statistics.

        Returns:
            Health check statistics
        """
        return {
            "total_checks": self._health_check_count,
            "passed_checks": self._health_check_pass_count,
            "pass_rate": (
                self._health_check_pass_count / self._health_check_count
                if self._health_check_count > 0
                else 0
            )
        }

    def _get_resource_stats(self) -> Dict:
        """Get resource statistics.

        Returns:
            Resource statistics
        """
        if self._resource_monitor:
            return self._resource_monitor.get_current_stats()
        return {}

    def _get_avg_cpu_usage(self) -> float:
        """Get average CPU usage.

        Returns:
            Average CPU percentage
        """
        if self._resource_monitor:
            return self._resource_monitor.get_average_cpu_usage()
        return 0.0

    def _get_avg_memory_usage(self) -> float:
        """Get average memory usage.

        Returns:
            Average memory percentage
        """
        if self._resource_monitor:
            return self._resource_monitor.get_average_memory_usage()
        return 0.0

    def _get_current_time(self) -> datetime:
        """Get current time in UTC.

        Returns:
            Current datetime in UTC
        """
        return datetime.now(timezone.utc)
