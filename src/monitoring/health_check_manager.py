"""Health check system for monitoring component status.

This module implements a health check system that continuously monitors
the operational status of all trading system components. This provides
real-time visibility into system health and enables early detection
of potential issues.

Features:
- Component health check registration
- Overall system health status
- Health status determination (HEALTHY, DEGRADED, UNHEALTHY)
- Critical vs non-critical component distinction
- Response time tracking
- CSV audit trail logging
"""

import csv
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger(__name__)


class HealthCheckManager:
    """Manage health checks for all system components.

    Attributes:
        _health_checks: Dictionary of registered health checks
        _last_check_times: Timestamp of last health check per component
        _check_interval_seconds: Minimum seconds between checks
        _audit_trail_path: Path to CSV audit trail file

    Example:
        >>> manager = HealthCheckManager(check_interval_seconds=60)
        >>> manager.register_check("data_feed", check_data_feed_health)
        >>> status = manager.get_system_health()
        >>> print(status['overall_status'])
        'HEALTHY'
    """

    # Health status constants
    STATUS_HEALTHY = "HEALTHY"
    STATUS_DEGRADED = "DEGRADED"
    STATUS_UNHEALTHY = "UNHEALTHY"

    def __init__(
        self,
        check_interval_seconds: int = 60,
        audit_trail_path: Optional[str] = None
    ) -> None:
        """Initialize health check manager.

        Args:
            check_interval_seconds: Minimum seconds between checks
            audit_trail_path: Path to CSV audit trail file (optional)

        Example:
            >>> manager = HealthCheckManager(check_interval_seconds=60)
        """
        self._check_interval_seconds = check_interval_seconds
        self._audit_trail_path = audit_trail_path

        # Registered health checks
        self._health_checks: Dict[str, Dict[str, Any]] = {}

        # Last check times per component
        self._last_check_times: Dict[str, datetime] = {}

        logger.info(
            "HealthCheckManager initialized: interval={}s".format(
                check_interval_seconds
            )
        )

    def register_check(
        self,
        component_name: str,
        check_function: Callable,
        critical: bool = True
    ) -> None:
        """Register a health check for a component.

        Args:
            component_name: Name of the component (e.g., "data_feed")
            check_function: Function that performs health check
            critical: Whether component is critical for trading

        Example:
            >>> def check_data_feed():
            ...     # Check data feed health
            ...     return {"status": "HEALTHY", "message": "OK"}
            >>> manager.register_check("data_feed", check_data_feed)
        """
        self._health_checks[component_name] = {
            'function': check_function,
            'critical': critical
        }

        logger.info(
            "Registered health check: {} (critical={})".format(
                component_name,
                critical
            )
        )

    def get_system_health(self) -> dict:
        """Get overall system health status.

        Returns:
            Dictionary with:
            - overall_status: "HEALTHY", "DEGRADED", or "UNHEALTHY"
            - timestamp: When health check was performed
            - component_count: Total number of components
            - healthy_count: Number of healthy components
            - degraded_count: Number of degraded components
            - unhealthy_count: Number of unhealthy components
            - components: Dict of component statuses

        Example:
            >>> status = manager.get_system_health()
            >>> if status['overall_status'] != "HEALTHY":
            ...     print("System unhealthy!")
        """
        timestamp = self._get_current_time()

        # Check all components
        component_statuses = {}
        for component_name in self._health_checks:
            status_result = self.check_component(component_name)
            component_statuses[component_name] = status_result

        # Count statuses
        healthy_count = sum(
            1 for status in component_statuses.values()
            if status['status'] == self.STATUS_HEALTHY
        )
        degraded_count = sum(
            1 for status in component_statuses.values()
            if status['status'] == self.STATUS_DEGRADED
        )
        unhealthy_count = sum(
            1 for status in component_statuses.values()
            if status['status'] == self.STATUS_UNHEALTHY
        )

        # Determine overall status
        # If any critical component is unhealthy → UNHEALTHY
        # If any critical component is degraded → DEGRADED
        # If non-critical components have issues → DEGRADED
        # Otherwise → HEALTHY

        critical_unhealthy = any(
            component_statuses[name]['status'] == self.STATUS_UNHEALTHY
            for name, config in self._health_checks.items()
            if config['critical'] and name in component_statuses
        )

        critical_degraded = any(
            component_statuses[name]['status'] == self.STATUS_DEGRADED
            for name, config in self._health_checks.items()
            if config['critical'] and name in component_statuses
        )

        any_unhealthy = unhealthy_count > 0
        any_degraded = degraded_count > 0

        if critical_unhealthy:
            overall_status = self.STATUS_UNHEALTHY
        elif critical_degraded:
            overall_status = self.STATUS_DEGRADED
        elif any_unhealthy:
            overall_status = self.STATUS_DEGRADED
        elif any_degraded:
            overall_status = self.STATUS_DEGRADED
        else:
            overall_status = self.STATUS_HEALTHY

        return {
            'overall_status': overall_status,
            'timestamp': timestamp.isoformat(),
            'component_count': len(self._health_checks),
            'healthy_count': healthy_count,
            'degraded_count': degraded_count,
            'unhealthy_count': unhealthy_count,
            'components': component_statuses
        }

    def check_component(self, component_name: str) -> dict:
        """Check health of a specific component.

        Args:
            component_name: Name of component to check

        Returns:
            Component health status:
            - status: "HEALTHY", "DEGRADED", or "UNHEALTHY"
            - message: Status message
            - timestamp: When check was performed
            - response_time_ms: Check execution time

        Example:
            >>> status = manager.check_component("data_feed")
            >>> print(status['status'])
            'HEALTHY'
        """
        if component_name not in self._health_checks:
            return {
                'status': self.STATUS_UNHEALTHY,
                'message': 'Component not registered',
                'timestamp': self._get_current_time().isoformat(),
                'response_time_ms': 0
            }

        # Check if interval has passed
        last_check = self._last_check_times.get(component_name)
        current_time = self._get_current_time()

        if last_check is not None:
            elapsed = (current_time - last_check).total_seconds()
            if elapsed < self._check_interval_seconds:
                # Return cached result
                logger.debug(
                    "Using cached health check for {} ({}s ago)".format(
                        component_name,
                        int(elapsed)
                    )
                )
                # Return empty result with cached flag
                return {
                    'status': self.STATUS_UNHEALTHY,
                    'message': 'Check too recent',
                    'timestamp': current_time.isoformat(),
                    'response_time_ms': 0,
                    'cached': True
                }

        # Perform health check
        check_config = self._health_checks[component_name]
        check_function = check_config['function']

        start_time = time.time()
        try:
            result = check_function()
            response_time_ms = int((time.time() - start_time) * 1000)

            # Validate result structure
            if not isinstance(result, dict):
                raise ValueError("Health check must return dict")

            if 'status' not in result:
                raise ValueError("Health check result must have 'status' key")

            # Ensure status is valid
            if result['status'] not in [
                self.STATUS_HEALTHY,
                self.STATUS_DEGRADED,
                self.STATUS_UNHEALTHY
            ]:
                raise ValueError("Invalid health status: {}".format(
                    result['status']
                ))

            # Add metadata
            result['timestamp'] = current_time.isoformat()
            result['response_time_ms'] = response_time_ms

            # Update last check time
            self._last_check_times[component_name] = current_time

            # Log result
            self._log_health_check_result(component_name, result)

            return result

        except Exception as e:
            # Check function raised exception
            logger.error(
                "Health check failed for {}: {}".format(
                    component_name,
                    str(e)
                )
            )

            error_result = {
                'status': self.STATUS_UNHEALTHY,
                'message': 'Health check exception: {}'.format(str(e)),
                'timestamp': current_time.isoformat(),
                'response_time_ms': int((time.time() - start_time) * 1000)
            }

            # Log error result
            self._log_health_check_result(component_name, error_result)

            return error_result

    def is_system_healthy(self) -> bool:
        """Check if overall system is healthy.

        Returns:
            True if all critical components are healthy

        Example:
            >>> if manager.is_system_healthy():
            ...     submit_order()
        """
        system_health = self.get_system_health()
        return system_health['overall_status'] == self.STATUS_HEALTHY

    def get_unhealthy_components(self) -> list:
        """Get list of unhealthy components.

        Returns:
            List of component names that are unhealthy

        Example:
            >>> unhealthy = manager.get_unhealthy_components()
            >>> for component in unhealthy:
            ...     print("{} is unhealthy".format(component))
        """
        system_health = self.get_system_health()

        unhealthy = []
        for name, status in system_health['components'].items():
            if status['status'] in [
                self.STATUS_DEGRADED,
                self.STATUS_UNHEALTHY
            ]:
                unhealthy.append(name)

        return unhealthy

    def _log_health_check_result(
        self,
        component_name: str,
        result: dict
    ) -> None:
        """Log health check result to CSV audit trail.

        Args:
            component_name: Name of component checked
            result: Health check result
        """
        if self._audit_trail_path is None:
            return

        # Ensure audit trail directory exists
        audit_path = Path(self._audit_trail_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists
        file_exists = (
            audit_path.exists() and
            audit_path.stat().st_size > 0
        )

        # Append to CSV
        with open(audit_path, "a", newline="") as f:
            writer = csv.writer(f)

            # Write header if new file
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "component_name",
                    "status",
                    "message",
                    "response_time_ms",
                    "critical"
                ])

            # Get critical flag
            critical = self._health_checks.get(component_name, {}).get(
                'critical', True
            )

            # Write event
            writer.writerow([
                result.get('timestamp', ''),
                component_name,
                result.get('status', 'UNKNOWN'),
                result.get('message', ''),
                result.get('response_time_ms', 0),
                critical
            ])

        logger.debug("Health check logged: {}".format(component_name))

    def _get_current_time(self) -> datetime:
        """Get current time (UTC).

        Returns:
            Current datetime in UTC
        """
        return datetime.now(timezone.utc)


# Built-in health check functions

def check_system_resources() -> dict:
    """Check CPU, memory, and disk usage.

    Returns:
        System resource health status
    """
    try:
        import psutil

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent

        # Determine overall status
        issues = []
        if cpu_percent > 90:
            issues.append("CPU critical ({}%)".format(cpu_percent))
        elif cpu_percent > 70:
            issues.append("CPU elevated ({}%)".format(cpu_percent))

        if memory_percent > 90:
            issues.append("Memory critical ({}%)".format(memory_percent))
        elif memory_percent > 75:
            issues.append("Memory elevated ({}%)".format(memory_percent))

        if disk_percent > 90:
            issues.append("Disk critical ({}%)".format(disk_percent))
        elif disk_percent > 80:
            issues.append("Disk elevated ({}%)".format(disk_percent))

        if len(issues) == 0:
            status = HealthCheckManager.STATUS_HEALTHY
            message = "System resources normal"
        elif cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
            status = HealthCheckManager.STATUS_UNHEALTHY
            message = "; ".join(issues)
        else:
            status = HealthCheckManager.STATUS_DEGRADED
            message = "; ".join(issues)

        return {
            "status": status,
            "message": message,
            "metrics": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent
            }
        }

    except ImportError:
        # psutil not available
        return {
            "status": HealthCheckManager.STATUS_DEGRADED,
            "message": "psutil not available - cannot monitor resources",
            "metrics": {}
        }
    except Exception as e:
        return {
            "status": HealthCheckManager.STATUS_UNHEALTHY,
            "message": "Error checking resources: {}".format(str(e)),
            "metrics": {}
        }
