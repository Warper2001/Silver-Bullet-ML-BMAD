"""Resource Monitor for CPU, memory, and disk usage.

Monitors system resources with alert thresholds, consecutive check
tracking, history tracking, and trend analysis.
"""

import logging
from datetime import datetime, timezone
from collections import deque
from typing import Dict, Optional, Callable

import psutil


class ResourceMonitor:
    """Monitors CPU, memory, and disk usage with alerting.

    Tracks resource usage over time and triggers alerts when
    thresholds are exceeded. Logs metrics to audit trail for analysis.
    """

    CPU_THRESHOLD = 80.0
    MEMORY_THRESHOLD = 80.0
    DISK_THRESHOLD = 80.0
    CONSECUTIVE_CHECKS_THRESHOLD = 3

    def __init__(
        self,
        check_interval_seconds: int = 10,
        alert_callback: Optional[Callable] = None,
        audit_trail=None
    ):
        """Initialize resource monitor.

        Args:
            check_interval_seconds: Seconds between checks (default 10)
            alert_callback: Optional callback for alerts
            audit_trail: Optional audit trail for logging
        """
        self._check_interval = check_interval_seconds
        self._alert_callback = alert_callback
        self._audit_trail = audit_trail
        self._logger = logging.getLogger(__name__)

        # Alert tracking (consecutive checks)
        self._cpu_consecutive_high = 0
        self._memory_consecutive_high = 0

        # History tracking (24 hours at 10s intervals = 8640 data points)
        self._history_length = 8640
        self._cpu_history = deque(maxlen=self._history_length)
        self._memory_history = deque(maxlen=self._history_length)
        self._disk_history = deque(maxlen=self._history_length)

    def check_resources(self) -> Dict:
        """Check current resource usage and trigger alerts if needed.

        Returns:
            Resource metrics with alert status
        """
        # Measure resources using psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        metrics = {
            "timestamp": self._get_current_time(),
            "cpu_percent": cpu_percent,
            "memory_mb": memory.rss / (1024 * 1024),
            "memory_percent": memory.percent,
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3),
            "disk_total_gb": disk.total / (1024**3),
            "cpu_alert": False,
            "memory_alert": False,
            "disk_alert": False
        }

        # Check CPU alert (80% threshold, 3 consecutive)
        if cpu_percent > self.CPU_THRESHOLD:
            self._cpu_consecutive_high += 1
            if self._cpu_consecutive_high >= self.CONSECUTIVE_CHECKS_THRESHOLD:
                metrics["cpu_alert"] = True
                self._trigger_alert("cpu", cpu_percent, metrics)
        else:
            self._cpu_consecutive_high = 0

        # Check memory alert (80% threshold, 3 consecutive)
        if memory.percent > self.MEMORY_THRESHOLD:
            self._memory_consecutive_high += 1
            if self._memory_consecutive_high >= self.CONSECUTIVE_CHECKS_THRESHOLD:
                metrics["memory_alert"] = True
                self._trigger_alert("memory", memory.percent, metrics)
        else:
            self._memory_consecutive_high = 0

        # Check disk alert (80% threshold, immediate)
        if disk.percent > self.DISK_THRESHOLD:
            metrics["disk_alert"] = True
            self._trigger_alert("disk", disk.percent, metrics)

        # Update history
        self._cpu_history.append(cpu_percent)
        self._memory_history.append(memory.percent)
        self._disk_history.append(disk.percent)

        # Log metrics
        self._log_metrics(metrics)

        return metrics

    def get_resource_history(self) -> Dict:
        """Get resource usage history and statistics.

        Returns:
            Dictionary with historical data and statistics
        """
        if len(self._cpu_history) == 0:
            return {"error": "No history available"}

        return {
            "cpu": {
                "current": self._cpu_history[-1],
                "average": sum(self._cpu_history) / len(self._cpu_history),
                "peak": max(self._cpu_history),
                "trend": self._calculate_trend(list(self._cpu_history))
            },
            "memory": {
                "current": self._memory_history[-1],
                "average": sum(self._memory_history) / len(self._memory_history),
                "peak": max(self._memory_history),
                "trend": self._calculate_trend(list(self._memory_history))
            },
            "disk": {
                "current": self._disk_history[-1],
                "average": sum(self._disk_history) / len(self._disk_history),
                "peak": max(self._disk_history),
                "trend": self._calculate_trend(list(self._disk_history))
            }
        }

    def _trigger_alert(
        self,
        resource_type: str,
        value: float,
        metrics: Dict
    ) -> None:
        """Trigger resource alert.

        Args:
            resource_type: Type of resource (cpu, memory, disk)
            value: Current resource value
            metrics: Full resource metrics
        """
        if resource_type == "cpu":
            message = "High CPU usage - {:.1f}%".format(value)
        elif resource_type == "memory":
            message = "High memory usage - {:.1f}%".format(value)
        elif resource_type == "disk":
            message = "Low disk space - {:.1f}% used, {:.1f} GB free".format(
                metrics["disk_percent"],
                metrics["disk_free_gb"]
            )
        else:
            message = "Resource alert: {} = {:.1f}".format(
                resource_type, value
            )

        # Call alert callback if provided
        if self._alert_callback:
            self._alert_callback(resource_type, message, value)

        self._logger.warning(message)

    def _log_metrics(self, metrics: Dict) -> None:
        """Log resource metrics to audit trail.

        Args:
            metrics: Resource metrics dictionary
        """
        if not self._audit_trail:
            return

        # Log to audit trail
        self._audit_trail.log_action(
            "RESOURCE_CHECK",
            "resource_monitor",
            "system",
            {
                "cpu_percent": metrics["cpu_percent"],
                "memory_mb": metrics["memory_mb"],
                "memory_percent": metrics["memory_percent"],
                "disk_percent": metrics["disk_percent"],
                "disk_free_gb": metrics["disk_free_gb"],
                "cpu_alert": metrics["cpu_alert"],
                "memory_alert": metrics["memory_alert"],
                "disk_alert": metrics["disk_alert"]
            }
        )

    def _calculate_trend(self, history: list) -> str:
        """Calculate resource usage trend.

        Args:
            history: List of historical values

        Returns:
            Trend: "increasing", "decreasing", or "stable"
        """
        if len(history) < 10:
            return "stable"

        # Compare recent vs older values
        recent_avg = sum(history[-10:]) / 10
        older_avg = (
            sum(history[-20:-10]) / 10
            if len(history) >= 20 else recent_avg
        )

        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing"
        else:
            return "stable"

    def _get_current_time(self) -> datetime:
        """Get current time in UTC.

        Returns:
            Current datetime in UTC
        """
        return datetime.now(timezone.utc)
