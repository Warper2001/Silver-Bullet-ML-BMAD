"""Command-line interface for emergency stop control.

This module provides CLI commands for manually activating and
deactivating the emergency stop button.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports (must come before module imports)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Disable E402 for this conditional import
# flake8: noqa
from src.risk.emergency_stop import EmergencyStop  # noqa: E402


def activate_emergency_stop(reason: str) -> None:
    """Activate emergency stop from command line.

    Args:
        reason: Reason for activating emergency stop

    Example:
        $ python -m src.cli.emergency_stop activate "Manual intervention"
        Emergency stop activated at 2026-03-17 14:30:00 UTC
    """
    stop = EmergencyStop(
        audit_trail_path="data/audit/emergency_stop.csv",
        state_path="data/state/emergency_stop.json"
    )
    stop.activate(reason)

    print("Emergency stop activated")
    print("Reason: {}".format(reason))


def deactivate_emergency_stop() -> None:
    """Deactivate emergency stop from command line.

    Example:
        $ python -m src.cli.emergency_stop deactivate
        Emergency stop deactivated at 2026-03-17 14:35:00 UTC
    """
    stop = EmergencyStop(
        audit_trail_path="data/audit/emergency_stop.csv",
        state_path="data/state/emergency_stop.json"
    )
    stop.deactivate()

    print("Emergency stop deactivated")


def check_emergency_stop_status() -> None:
    """Check and display emergency stop status.

    Example:
        $ python -m src.cli.emergency_stop status
        Emergency stop status: ACTIVE
        Activated: 2026-03-17 14:30:00 UTC
        Reason: Manual intervention
        Time stopped: 300 seconds
    """
    stop = EmergencyStop(
        audit_trail_path="data/audit/emergency_stop.csv",
        state_path="data/state/emergency_stop.json"
    )
    status = stop.get_status()

    if status['is_stopped']:
        print("Emergency stop status: ACTIVE")
        print("Activated: {}".format(status['stop_time']))
        print("Reason: {}".format(status['stop_reason']))
        print("Time stopped: {} seconds".format(
            status['time_stopped_seconds']
        ))
    else:
        print("Emergency stop status: INACTIVE")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Emergency stop control for trading system"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Activate command
    activate_parser = subparsers.add_parser(
        'activate',
        help='Activate emergency stop'
    )
    activate_parser.add_argument(
        'reason',
        help='Reason for activating emergency stop'
    )

    # Deactivate command
    subparsers.add_parser(
        'deactivate',
        help='Deactivate emergency stop'
    )

    # Status command
    subparsers.add_parser(
        'status',
        help='Check emergency stop status'
    )

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if args.command == 'activate':
        activate_emergency_stop(args.reason)
    elif args.command == 'deactivate':
        deactivate_emergency_stop()
    elif args.command == 'status':
        check_emergency_stop_status()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
