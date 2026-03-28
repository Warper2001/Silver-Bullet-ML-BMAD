# Integration Test Setup Guide

This directory contains configuration files for TradeStation API integration tests.

## Prerequisites

Before running integration tests, you need:

1. **TradeStation SIM Account**
   - Sign up at https://tradestation.com
   - Request SIM environment access
   - Get API credentials from TradeStation Developer Portal

2. **Python Environment**
   - Poetry installed
   - Dependencies installed: `poetry install`

## Configuration

### Step 1: Set Environment Variables (Recommended)

Export your TradeStation SIM credentials as environment variables:

```bash
export TRADESTATION_SIM_CLIENT_ID="your_client_id_here"
export TRADESTATION_SIM_CLIENT_SECRET="your_client_secret_here"
```

Add these to your `~/.bashrc` or `~/.zshrc` for persistence:

```bash
echo 'export TRADESTATION_SIM_CLIENT_ID="your_client_id_here"' >> ~/.bashrc
echo 'export TRADESTATION_SIM_CLIENT_SECRET="your_client_secret_here"' >> ~/.bashrc
source ~/.bashrc
```

### Step 2: Verify Configuration

Run the configuration check script:

```bash
python tests/integration/check_config.py
```

This will verify:
- Environment variables are set
- Credentials are valid
- SIM environment is accessible

## Configuration Files

### sim_config.example.yaml

Example configuration file showing all available options. To use:

1. Copy to `sim_config.yaml`:
   ```bash
   cp sim_config.example.yaml sim_config.yaml
   ```

2. Edit `sim_config.yaml` with your settings (optional)
   - Most settings work with defaults
   - Environment variables take precedence

**WARNING:** Never commit `sim_config.yaml` with real credentials!

## Running Tests

### Run All Integration Tests

```bash
# From project root
pytest tests/integration/test_tradestation_api/ -v

# With detailed output
pytest tests/integration/test_tradestation_api/ -v -s

# With coverage
pytest tests/integration/test_tradestation_api/ --cov=src/execution/tradestation
```

### Run Specific Test Modules

```bash
# Authentication tests only
pytest tests/integration/test_tradestation_api/test_auth_flow.py -v

# Market data tests only
pytest tests/integration/test_tradestation_api/test_market_data_flow.py -v

# Order lifecycle tests only (CAUTION: places real SIM orders)
pytest tests/integration/test_tradestation_api/test_order_flow.py -v
```

### Run Specific Tests

```bash
# Run a single test
pytest tests/integration/test_tradestation_api/test_auth_flow.py::test_client_credentials_flow -v

# Run tests matching a pattern
pytest tests/integration/test_tradestation_api/ -k "test_stream" -v
```

## Test Categories

### 1. Authentication Tests (`test_auth_flow.py`)
- Test OAuth 2.0 client credentials flow
- Test token refresh mechanism
- Test authenticated API calls
- **Risk:** Low (no orders placed)

### 2. Market Data Tests (`test_market_data_flow.py`)
- Test real-time quotes
- Test historical data download
- Test quote streaming
- **Risk:** Low (read-only operations)

### 3. Order Lifecycle Tests (`test_order_flow.py`)
- Test order placement
- Test order modification
- Test order cancellation
- Test order status streaming
- **Risk:** Medium (places real SIM orders, but no real money)

## Safety Features

### Order Test Safeguards

1. **Small Position Sizes**
   - Default: 1 contract
   - Configurable via `test_order_quantity`

2. **Price Buffers**
   - Limit orders placed away from market
   - Prevents accidental fills

3. **Automatic Cleanup**
   - All test orders cancelled after test
   - Cancel all orders option available

4. **SIM Environment Only**
   - Tests refuse to run in LIVE environment
   - Hardcoded to use SIM API endpoints

### Test Isolation

- Each test cleans up after itself
- No state shared between tests
- Independent test execution

## Troubleshooting

### Credentials Not Found

```
pytest: error: TradeStation SIM credentials not configured
```

**Solution:** Set environment variables:
```bash
export TRADESTATION_SIM_CLIENT_ID="your_client_id"
export TRADESTATION_SIM_CLIENT_SECRET="your_client_secret"
```

### Authentication Failed

```
AuthenticationError: Invalid credentials
```

**Solution:**
- Verify credentials are correct
- Check SIM environment access is enabled
- Ensure API access is enabled in TradeStation portal

### Network Timeout

```
TimeoutError: Request timed out after 30.0 seconds
```

**Solution:**
- Check internet connection
- Verify TradeStation SIM API is accessible
- Increase timeout in `sim_config.yaml`

### Order Tests Failing

```
OrderRejectedError: Order rejected
```

**Solution:**
- Verify account has sufficient buying power
- Check symbol is valid and actively trading
- Ensure market is open (for market orders)

## Best Practices

1. **Run in Off-Hours**
   - Market data tests work best during trading hours
   - Order tests: avoid market open/close volatility

2. **Start with Read-Only Tests**
   - Begin with `test_auth_flow.py`
   - Then `test_market_data_flow.py`
   - Finally `test_order_flow.py` (after understanding risks)

3. **Monitor SIM Account**
   - Check order status in TradeStation platform
   - Verify orders are cancelled after tests
   - Monitor account balance

4. **Keep Credentials Secure**
   - Never commit credentials to git
   - Use environment variables
   - Rotate credentials regularly

5. **Review Test Results**
   - Check logs for any errors
   - Verify order cleanup
   - Report any issues

## Continuous Integration

Integration tests should **NOT** run automatically in CI/CD pipelines because:

1. Require real credentials (security risk)
2. Make actual API calls (rate limits)
3. Place real orders (even in SIM, requires cleanup)

### Recommended CI Strategy

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run unit tests
        run: pytest tests/unit/ -v

  integration-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'workflow_dispatch'  # Manual trigger only
    steps:
      - uses: actions/checkout@v2
      - name: Run integration tests
        env:
          TRADESTATION_SIM_CLIENT_ID: ${{ secrets.TRADESTATION_SIM_CLIENT_ID }}
          TRADESTATION_SIM_CLIENT_SECRET: ${{ secrets.TRADESTATION_SIM_CLIENT_SECRET }}
        run: pytest tests/integration/ -v
```

## Support

For issues with:
- **TradeStation API:** Contact TradeStation support
- **Integration Tests:** Open GitHub issue
- **Credentials:** Reset in TradeStation Developer Portal

## Additional Resources

- [TradeStation API Documentation](https://tradestation.com/api)
- [OAuth 2.0 Documentation](https://oauth.net/2/)
- [Pytest Documentation](https://docs.pytest.org/)
