"""Integration tests for TradeStation API.

These tests require actual TradeStation SIM environment credentials and
will make real API calls. They should NOT be run in CI/CD pipelines
without proper SIM environment setup.

Prerequisites:
- TradeStation SIM account
- SIM API credentials (client_id, client_secret)
- Active SIM environment access

Environment Variables:
- TRADESTATION_SIM_CLIENT_ID: SIM API client ID
- TRADESTATION_SIM_CLIENT_SECRET: SIM API client secret

Usage:
    # Run all integration tests
    pytest tests/integration/test_tradestation_api/

    # Run specific test module
    pytest tests/integration/test_tradestation_api/test_auth_flow.py

    # Run with verbose output
    pytest tests/integration/test_tradestation_api/ -v -s
"""
