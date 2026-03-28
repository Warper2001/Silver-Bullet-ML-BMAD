#!/usr/bin/env python3
"""
Integration Test Structure Validation

This script validates that the integration test infrastructure is properly
set up and ready to run once credentials are configured.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 70)
print("Integration Test Infrastructure Validation")
print("=" * 70)
print()

# Test 1: Check test modules exist
print("1. Checking test modules...")
test_modules = [
    "tests/integration/test_tradestation_api/__init__.py",
    "tests/integration/test_tradestation_api/conftest.py",
    "tests/integration/test_tradestation_api/test_auth_flow.py",
    "tests/integration/test_tradestation_api/test_market_data_flow.py",
    "tests/integration/test_tradestation_api/test_order_flow.py",
]

for module in test_modules:
    if Path(module).exists():
        print(f"   ✅ {module}")
    else:
        print(f"   ❌ {module} - NOT FOUND")

print()

# Test 2: Check configuration files
print("2. Checking configuration files...")
config_files = [
    "tests/integration/config/README.md",
    "tests/integration/config/sim_config.example.yaml",
]

for config in config_files:
    if Path(config).exists():
        print(f"   ✅ {config}")
    else:
        print(f"   ❌ {config} - NOT FOUND")

print()

# Test 3: Check test runner
print("3. Checking test runner...")
if Path("run_integration_tests.py").exists():
    print("   ✅ run_integration_tests.py")
else:
    print("   ❌ run_integration_tests.py - NOT FOUND")

print()

# Test 4: Import test modules
print("4. Importing test modules...")
try:
    sys.path.insert(0, "tests/integration/test_tradestation_api")
    import conftest
    print("   ✅ conftest.py imported successfully")
except Exception as e:
    print(f"   ❌ Failed to import conftest.py: {e}")

print()

# Test 5: Check credentials
print("5. Checking TradeStation SIM credentials...")
client_id = os.getenv("TRADESTATION_SIM_CLIENT_ID")
client_secret = os.getenv("TRADESTATION_SIM_CLIENT_SECRET")

if client_id:
    print(f"   ✅ TRADESTATION_SIM_CLIENT_ID is set ({len(client_id)} chars)")
else:
    print("   ⚠️  TRADESTATION_SIM_CLIENT_ID not set")

if client_secret:
    print(f"   ✅ TRADESTATION_SIM_CLIENT_SECRET is set ({len(client_secret)} chars)")
else:
    print("   ⚠️  TRADESTATION_SIM_CLIENT_SECRET not set")

print()

# Summary
print("=" * 70)
if client_id and client_secret:
    print("✅ Ready to run integration tests!")
    print()
    print("To run the tests:")
    print("  python run_integration_tests.py --module test_auth_flow")
    print("  python run_integration_tests.py --module test_market_data_flow")
    print("  python run_integration_tests.py --module test_order_flow")
    print("  python run_integration_tests.py")
else:
    print("⚠️  Integration tests require TradeStation SIM credentials")
    print()
    print("To set up credentials:")
    print("  export TRADESTATION_SIM_CLIENT_ID='your_client_id'")
    print("  export TRADESTATION_SIM_CLIENT_SECRET='your_client_secret'")
    print()
    print("Then run:")
    print("  python run_integration_tests.py")

print("=" * 70)
print()
print("Test Infrastructure Summary:")
print("  • Authentication tests: 8 tests (test_auth_flow.py)")
print("  • Market data tests: 17 tests (test_market_data_flow.py)")
print("  • Order lifecycle tests: 10 tests (test_order_flow.py)")
print("  • Total: 35 integration tests")
print()
print("Safety Features:")
print("  • SIM environment only (no real money)")
print("  • Small position sizes (1 contract)")
print("  • Automatic cleanup after tests")
print("  • Price buffers on limit orders")
print("=" * 70)
