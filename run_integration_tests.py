#!/usr/bin/env python3
"""
TradeStation Integration Test Runner

This script validates configuration and runs integration tests against the
TradeStation SIM API. It performs pre-flight checks, runs tests sequentially,
and generates a detailed report.

Prerequisites:
- TRADESTATION_SIM_CLIENT_ID environment variable
- TRADESTATION_SIM_CLIENT_SECRET environment variable
- Active internet connection
- TradeStation SIM account

Usage:
    # Run all integration tests
    python run_integration_tests.py

    # Run specific test module
    python run_integration_tests.py --module test_auth_flow

    # Run with verbose output
    python run_integration_tests.py --verbose

    # Skip order tests (safer)
    python run_integration_tests.py --skip-orders
"""

import argparse
import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path


def check_credentials():
    """Check if TradeStation SIM credentials are configured."""
    client_id = os.getenv("TRADESTATION_SIM_CLIENT_ID")
    client_secret = os.getenv("TRADESTATION_SIM_CLIENT_SECRET")

    if not client_id:
        print("❌ TRADESTATION_SIM_CLIENT_ID environment variable not set")
        print("   Please set: export TRADESTATION_SIM_CLIENT_ID='your_client_id'")
        return False

    if not client_secret:
        print("❌ TRADESTATION_SIM_CLIENT_SECRET environment variable not set")
        print("   Please set: export TRADESTATION_SIM_CLIENT_SECRET='your_client_secret'")
        return False

    print("✅ Credentials configured")
    return True


def check_python_environment():
    """Check if Python environment is properly set up."""
    # Check Python version
    if sys.version_info < (3, 11):
        print(f"❌ Python 3.11+ required, found {sys.version_info.major}.{sys.version_info.minor}")
        return False

    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    # Check pytest
    try:
        result = subprocess.run(
            ["pytest", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"✅ {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ pytest not found. Install with: pip install pytest pytest-asyncio")
        return False

    return True


def check_project_structure():
    """Check if project structure is correct."""
    required_paths = [
        "src/execution/tradestation",
        "tests/integration/test_tradestation_api",
        "tests/integration/config",
    ]

    for path in required_paths:
        if not Path(path).exists():
            print(f"❌ Required path not found: {path}")
            return False

    print("✅ Project structure validated")
    return True


def pre_flight_checks():
    """Run all pre-flight checks before running tests."""
    print("=" * 70)
    print("TradeStation Integration Test Runner - Pre-Flight Checks")
    print("=" * 70)
    print()

    checks = [
        ("Credentials", check_credentials),
        ("Python Environment", check_python_environment),
        ("Project Structure", check_project_structure),
    ]

    all_passed = True
    for name, check_func in checks:
        print(f"Checking {name}...")
        if not check_func():
            all_passed = False
        print()

    return all_passed


def run_tests(module=None, verbose=False, skip_orders=False):
    """Run integration tests."""
    print("=" * 70)
    print("Running Integration Tests")
    print("=" * 70)
    print()

    # Build pytest command
    pytest_args = [
        "pytest",
        "tests/integration/test_tradestation_api/",
        "-v" if verbose else "",
        "-s" if verbose else "",
        "--tb=short",
        "--durations=10",
    ]

    # Filter out empty strings
    pytest_args = [arg for arg in pytest_args if arg]

    # Add module filter if specified
    if module:
        pytest_args.append(f"tests/integration/test_tradestation_api/{module}.py")

    # Skip order tests if requested
    if skip_orders:
        pytest_args.extend(["-k", "not (Order or order)"])

    print(f"Command: {' '.join(pytest_args)}")
    print()

    # Run tests and capture output
    start_time = time.time()

    try:
        result = subprocess.run(
            pytest_args,
            check=False,  # Don't raise exception on test failures
        )

        test_duration = time.time() - start_time

        print()
        print("=" * 70)
        if result.returncode == 0:
            print("✅ ALL TESTS PASSED")
        else:
            print(f"⚠️  SOME TESTS FAILED (exit code: {result.returncode})")
        print(f"Duration: {test_duration:.1f} seconds")
        print("=" * 70)

        return result.returncode == 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        return False
    except Exception as e:
        print(f"\n\n❌ Error running tests: {e}")
        return False


def generate_report():
    """Generate test report."""
    print()
    print("=" * 70)
    print("Test Report")
    print("=" * 70)
    print()

    # Try to read pytest cache for test results
    # This is a simple version - could be enhanced with JSON/XML reports
    print("Test completed at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"))
    print()
    print("Next steps:")
    print("1. Review test output above for any failures")
    print("2. Check logs for detailed error messages")
    print("3. Verify all orders were cancelled in TradeStation platform")
    print("4. Update documentation if any API changes detected")


def print_safety_warnings():
    """Print safety warnings before running tests."""
    print()
    print("⚠️  SAFETY WARNINGS:")
    print("=" * 70)
    print()
    print("• These tests will place REAL orders in TradeStation SIM environment")
    print("• SIM environment uses simulated money (no real financial risk)")
    print("• Orders will be cancelled automatically after testing")
    print("• Verify your SIM account has sufficient buying power")
    print("• Monitor order status in TradeStation platform during tests")
    print()
    print("Recommended:")
    print("• Start with test_auth_flow.py (no orders placed)")
    print("• Then test_market_data_flow.py (read-only)")
    print("• Finally test_order_flow.py (places SIM orders)")
    print()
    print("Press Enter to continue or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n\nAborted by user")
        sys.exit(1)


def print_quick_start_guide():
    """Print quick start guide."""
    print()
    print("=" * 70)
    print("TradeStation Integration Test Quick Start")
    print("=" * 70)
    print()
    print("1. Set your credentials:")
    print("   export TRADESTATION_SIM_CLIENT_ID='your_client_id'")
    print("   export TRADESTATION_SIM_CLIENT_SECRET='your_client_secret'")
    print()
    print("2. Run authentication tests (no orders):")
    print("   python run_integration_tests.py --module test_auth_flow")
    print()
    print("3. Run market data tests (read-only):")
    print("   python run_integration_tests.py --module test_market_data_flow")
    print()
    print("4. Run order lifecycle tests (places SIM orders):")
    print("   python run_integration_tests.py --module test_order_flow")
    print()
    print("5. Run all tests:")
    print("   python run_integration_tests.py")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run TradeStation integration tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_integration_tests.py                 # Run all tests
  python run_integration_tests.py --module test_auth_flow  # Run specific module
  python run_integration_tests.py --verbose       # Verbose output
  python run_integration_tests.py --skip-orders   # Skip order placement tests
  python run_integration_tests.py --guide         # Show quick start guide
        """,
    )

    parser.add_argument(
        "--module", "-m",
        help="Specific test module to run (e.g., test_auth_flow)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--skip-orders",
        action="store_true",
        help="Skip order placement tests (safer, no orders placed)",
    )

    parser.add_argument(
        "--guide",
        action="store_true",
        help="Show quick start guide and exit",
    )

    parser.add_argument(
        "--no-warnings",
        action="store_true",
        help="Don't show safety warnings",
    )

    args = parser.parse_args()

    # Show quick start guide if requested
    if args.guide:
        print_quick_start_guide()
        return 0

    # Print header
    print()
    print("=" * 70)
    print("TradeStation API Integration Test Suite")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()

    # Run pre-flight checks
    if not pre_flight_checks():
        print()
        print("❌ Pre-flight checks failed. Please fix issues above and retry.")
        return 1

    # Show safety warnings (unless running auth tests only or --no-warnings)
    if not args.skip_orders and not args.no_warnings and args.module not in ["test_auth_flow", None]:
        print_safety_warnings()

    # Run tests
    success = run_tests(
        module=args.module,
        verbose=args.verbose,
        skip_orders=args.skip_orders,
    )

    # Generate report
    generate_report()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
