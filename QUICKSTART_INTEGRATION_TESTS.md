# 🚀 Integration Tests Quick Start Guide

## Current Status

✅ **All 35 integration tests are implemented and ready to run**
⚠️ **Waiting for your TradeStation SIM credentials to execute**

---

## Why Credentials Are Needed

The integration tests connect to the **actual TradeStation SIM API** to validate:
- OAuth 2.0 authentication flow
- Real-time market data retrieval
- Order placement, modification, and cancellation
- API response times and performance

**This is a feature, not a bug!** Real API testing ensures the SDK works correctly.

---

## Getting Your Credentials (5 Minutes)

### Step 1: Access TradeStation Developer Portal
1. Go to https://tradestation.com
2. Log into your account
3. Navigate to "Developer Portal" or "API Access"

### Step 2: Create a SIM Application
1. Click "Create New Application"
2. Select **SIM environment** (NOT LIVE!)
3. Choose "Client Credentials" flow
4. Name it (e.g., "Silver Bullet Testing")

### Step 3: Copy Your Credentials
You'll receive:
- **Client ID** (like "abc123def456")
- **Client Secret** (like "xyz789ghi012")

⚠️ **Keep these secret!** Never commit to git or share publicly.

---

## Running the Tests

### Option A: Quick Setup Script (Recommended)

```bash
./setup_and_run_tests.sh
```

This interactive script will:
1. Check if credentials are set
2. Help you configure them
3. Let you choose which tests to run

### Option B: Manual Setup

**1. Set credentials (temporary):**
```bash
export TRADESTATION_SIM_CLIENT_ID='your_client_id'
export TRADESTATION_SIM_CLIENT_SECRET='your_client_secret'
```

**2. Run tests:**
```bash
# Start with safe tests (no orders)
python run_integration_tests.py --module test_auth_flow

# Then market data (read-only)
python run_integration_tests.py --module test_market_data_flow

# Finally order tests (if confident)
python run_integration_tests.py --module test_order_flow
```

### Option C: Permanent Setup

Add to `~/.bashrc`:
```bash
echo 'export TRADESTATION_SIM_CLIENT_ID="your_client_id"' >> ~/.bashrc
echo 'export TRADESTATION_SIM_CLIENT_SECRET="your_client_secret"' >> ~/.bashrc
source ~/.bashrc
```

---

## What the Tests Will Do

### 📋 Authentication Tests (8 tests, ~1 minute)
- ✅ Connect to TradeStation SIM API
- ✅ Exchange credentials for access token
- ✅ Validate token is working
- ✅ Test token refresh mechanism
- ✅ Measure authentication speed (< 5s target)

**Safety:** No orders placed, no risk

### 📊 Market Data Tests (17 tests, ~3 minutes)
- ✅ Fetch real-time quotes for MNQH26
- ✅ Download historical bars (daily & minute)
- ✅ Stream quotes in real-time
- ✅ Validate data completeness (> 95%)
- ✅ Measure data latency (< 500ms target)

**Safety:** Read-only operations, no risk

### 📝 Order Lifecycle Tests (10 tests, ~5 minutes)
- ✅ Place limit orders (5% away from market)
- ✅ Modify order price and quantity
- ✅ Cancel individual orders
- ✅ Cancel all orders
- ✅ Stream order status updates
- ✅ Measure order placement speed (< 200ms target)

**Safety:** Real SIM orders, but no real money at risk

---

## Safety Guarantees

All order tests include these protections:

✅ **SIM Environment Only**
- Hardcoded SIM API endpoints
- Cannot access LIVE environment

✅ **Small Positions**
- Maximum 1 contract per order
- No large positions possible

✅ **Price Buffers**
- Limit orders 5% away from market
- Won't accidentally fill

✅ **Auto Cleanup**
- Every test cleans up after itself
- Cancel all orders on completion
- No orphaned orders

✅ **Order Type Safety**
- Market orders use IOC (Immediate or Cancel)
- Stop-limit orders tested safely

---

## Expected Output

When tests run successfully:

```
======================================================================
TradeStation API Integration Test Suite
======================================================================

✅ Authentication Tests: 8/8 PASSED
✅ Market Data Tests: 17/17 PASSED
✅ Order Lifecycle Tests: 10/10 PASSED

Total: 35/35 PASSED (100%)

Performance:
  • Authentication: 1.2s ✓ (target: < 5s)
  • Order placement: 45ms ✓ (target: < 200ms)
  • Market data: 120ms ✓ (target: < 500ms)

Duration: 8.5 minutes
======================================================================
```

---

## Troubleshooting

### "Credentials not configured"
**Solution:** Set environment variables (see above)

### "Authentication failed"
**Solution:**
- Verify credentials are correct
- Check SIM environment is enabled
- Ensure API access is active in TradeStation portal

### "Tests timeout"
**Solution:**
- Check internet connection
- Verify TradeStation SIM API is accessible
- Try running individual modules

### "Order rejected"
**Solution:**
- Ensure account has sufficient buying power
- Check symbol is valid (MNQH26, MESM26)
- Verify market is open (for market orders)

---

## Ready to Start?

### Choose Your Path:

**Path 1: I have credentials ready**
```bash
export TRADESTATION_SIM_CLIENT_ID='your_id'
export TRADESTATION_SIM_CLIENT_SECRET='your_secret'
python run_integration_tests.py
```

**Path 2: I need to get credentials**
1. Go to TradeStation Developer Portal
2. Create SIM application
3. Copy Client ID and Secret
4. Return here and run tests

**Path 3: Show me the demo first**
```bash
python demo_integration_tests.py
```

---

## Next Steps After Testing

Once tests pass successfully:

✅ **SDK validated** against real API
✅ **Performance benchmarks** established
✅ **Ready for production** use

**Then you can:**
- Implement triple barrier exits
- Set up production monitoring
- Deploy to LIVE environment (with caution)
- Or continue development

---

## Questions?

**Q: Do I need a funded account?**
A: No, SIM environment uses simulated money.

**Q: Will this place real trades?**
A: Only in SIM environment. No real money at risk.

**Q: How long does it take?**
A: ~10 minutes for all 35 tests.

**Q: Can I run just the safe tests?**
A: Yes! Use `--skip-orders` flag or run specific modules.

---

**Your integration test infrastructure is ready and waiting!** 🎉

Get your TradeStation SIM credentials and you're ready to go.
