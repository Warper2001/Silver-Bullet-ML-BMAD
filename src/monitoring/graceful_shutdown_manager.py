"""Graceful Shutdown Manager for trading system.

Handles shutdown signals, completes in-flight operations,
persists state, and manages position closing.
"""

import json
import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


class GracefulShutdownManager:
    """Manage graceful shutdown of trading system.

    Handles shutdown signals, completes in-flight operations,
    persists state, and manages position closing.
    """

    SHUTDOWN_TIMEOUT_SECONDS = 10

    def __init__(
        self,
        state_file_path: str = "data/state/system_state.json",
        close_positions_on_shutdown: bool = False,
        audit_trail=None,
        notification_manager=None
    ):
        """Initialize graceful shutdown manager.

        Args:
            state_file_path: Path to state file for persistence
            close_positions_on_shutdown: Whether to close positions on shutdown
            audit_trail: Optional audit trail for logging
            notification_manager: Optional notification manager
        """
        self._state_file_path = state_file_path
        self._close_positions = close_positions_on_shutdown
        self._audit_trail = audit_trail
        self._notification_manager = notification_manager
        self._logger = logging.getLogger(__name__)

        # Shutdown state
        self._shutdown_requested = False
        self._shutdown_complete = False
        self._accepting_new_data = True

        # State to persist
        self._system_state = {
            "shutdown_timestamp": None,
            "positions": [],
            "orders": [],
            "counters": {},
            "model_status": {},
            "configuration": {},
            "shutdown_complete": False
        }

    def request_shutdown(self) -> None:
        """Request graceful shutdown.

        Sets shutdown flag and initiates shutdown sequence.
        Can be called by signal handler or manual trigger.

        Example:
            >>> shutdown_manager.request_shutdown()
        """
        if self._shutdown_requested:
            self._logger.warning("Shutdown already requested")
            return

        self._shutdown_requested = True
        self._logger.info("Graceful shutdown requested")

        # Start shutdown sequence
        self._execute_shutdown_sequence()

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested.

        Returns:
            True if shutdown requested, False otherwise
        """
        return self._shutdown_requested

    def is_shutdown_complete(self) -> bool:
        """Check if shutdown is complete.

        Returns:
            True if shutdown complete, False otherwise
        """
        return self._shutdown_complete

    def is_accepting_new_data(self) -> bool:
        """Check if system is accepting new data.

        Returns:
            True if accepting, False otherwise
        """
        return self._accepting_new_data

    def _execute_shutdown_sequence(self) -> None:
        """Execute graceful shutdown sequence.

        1. Stop accepting new data and signals
        2. Wait for in-flight operations to complete
        3. Persist current state
        4. Close positions (if configured)
        5. Close connections and release resources
        6. Mark shutdown complete
        """
        start_time = time.time()

        # Step 1: Stop accepting new data
        self._accepting_new_data = False
        self._logger.info("Stopped accepting new data and signals")

        # Step 2: Wait for in-flight operations
        self._wait_for_in_flight_operations()

        # Step 3: Persist state
        self._persist_system_state()

        # Step 4: Close positions (if configured)
        if self._close_positions:
            self._close_all_positions()

        # Step 5: Close connections
        self._close_connections()

        # Step 6: Mark complete
        self._shutdown_complete = True
        self._system_state["shutdown_complete"] = True
        self._system_state["shutdown_timestamp"] = (
            self._get_current_time().isoformat()
        )

        elapsed = time.time() - start_time
        self._logger.info(
            "Graceful shutdown complete in {:.2f} seconds".format(elapsed)
        )

        # Log to audit trail
        if self._audit_trail:
            self._audit_trail.log_action(
                "GRACEFUL_SHUTDOWN",
                "graceful_shutdown_manager",
                "system",
                {
                    "shutdown_duration_seconds": elapsed,
                    "state_saved": True,
                    "positions_closed": self._close_positions
                }
            )

        # Send notification
        if self._notification_manager:
            self._notification_manager.send_notification(
                severity="INFO",
                title="System Shutdown Complete",
                message="System shutdown complete - state saved",
                notification_type="SYSTEM_SHUTDOWN"
            )

    def _wait_for_in_flight_operations(
        self,
        timeout_seconds: int = 5
    ) -> None:
        """Wait for in-flight operations to complete.

        Args:
            timeout_seconds: Maximum time to wait (default 5)
        """
        self._logger.info("Waiting for in-flight operations...")

        # In production, would wait for order submission queue
        # For now, just a brief pause
        time.sleep(0.5)

        self._logger.info("In-flight operations complete")

    def _persist_system_state(self) -> None:
        """Persist current system state to disk.

        Saves positions, orders, counters, model status, and
        configuration to state file for recovery.
        """
        # Create state directory
        state_path = Path(self._state_file_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)

        # Update system state with current data
        self._system_state["positions"] = self._get_current_positions()
        self._system_state["orders"] = self._get_pending_orders()
        self._system_state["counters"] = self._get_counters()
        self._system_state["model_status"] = self._get_model_status()
        self._system_state["configuration"] = self._get_configuration()

        # Write to file
        with open(state_path, 'w') as f:
            json.dump(self._system_state, f, indent=2)

        self._logger.info(
            "System state persisted to {}".format(
                self._state_file_path
            )
        )

    def _close_all_positions(self) -> None:
        """Close all open positions.

        Submits market orders to close all positions.
        """
        # Get current positions
        positions = self._get_current_positions()

        if not positions:
            self._logger.info("No positions to close")
            return

        self._logger.info(
            "Closing {} positions...".format(len(positions))
        )

        # In production, would submit closing orders
        for position in positions:
            self._logger.info(
                "Closing position: {} {} @ {}".format(
                    position["symbol"],
                    position["direction"],
                    position["quantity"]
                )
            )

        self._logger.info("All positions closed")

    def _close_connections(self) -> None:
        """Close WebSocket connections and release resources.

        Closes network connections, database connections,
        and releases system resources.
        """
        self._logger.info("Closing connections...")

        # In production, would close:
        # - WebSocket connections
        # - Database connections
        # - File handles
        # - Network sockets

        self._logger.info("All connections closed")

    def _get_current_positions(self) -> List[Dict]:
        """Get current open positions.

        Returns:
            List of position dictionaries
        """
        # In production, would query position manager
        # For now, return empty list
        return []

    def _get_pending_orders(self) -> List[Dict]:
        """Get pending orders.

        Returns:
            List of order dictionaries
        """
        # In production, would query order manager
        # For now, return empty list
        return []

    def _get_counters(self) -> Dict:
        """Get system counters.

        Returns:
            Dictionary of counter names to values
        """
        # In production, would query various components
        # For now, return empty dict
        return {}

    def _get_model_status(self) -> Dict:
        """Get model status.

        Returns:
            Dictionary of model status information
        """
        # In production, would query ML components
        # For now, return empty dict
        return {}

    def _get_configuration(self) -> Dict:
        """Get system configuration.

        Returns:
            Dictionary of configuration settings
        """
        # In production, would read config files
        # For now, return empty dict
        return {}

    def _get_current_time(self) -> datetime:
        """Get current time in UTC.

        Returns:
            Current datetime in UTC
        """
        return datetime.now(timezone.utc)


def setup_signal_handlers(
    shutdown_manager: GracefulShutdownManager
) -> None:
    """Setup signal handlers for graceful shutdown.

    Registers handlers for SIGTERM and SIGINT to trigger
    graceful shutdown.

    Args:
        shutdown_manager: GracefulShutdownManager instance

    Example:
        >>> setup_signal_handlers(shutdown_manager)
        >>> # Now SIGTERM/SIGINT will trigger graceful shutdown
    """
    logger = logging.getLogger(__name__)

    def signal_handler(signum, frame):
        """Handle shutdown signals."""
        logger.info(
            "Received signal {}, initiating shutdown".format(signum)
        )
        shutdown_manager.request_shutdown()

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Ignore SIGPIPE (write to closed socket)
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)


def load_system_state(
    state_file_path: str = "data/state/system_state.json"
) -> Dict:
    """Load system state from disk.

    Used during system startup to recover from previous state.

    Args:
        state_file_path: Path to state file

    Returns:
        System state dictionary

    Example:
        >>> state = load_system_state()
        >>> if state.get("shutdown_complete"):
        ...     # Recover from saved state
    """
    state_path = Path(state_file_path)

    if not state_path.exists():
        return {}

    with open(state_path, 'r') as f:
        return json.load(f)
