# TradeStation SIM Paper Trading - Deployment Guide

## 🚀 Quick Start (3 Steps to Paper Trading)

### Prerequisites
✅ All code is ready and verified
✅ All 13 critical fixes applied
✅ System is production-ready for SIM environment

### Step 1: Authenticate with TradeStation

The system needs a valid access token to connect to the TradeStation SIM API.

**Method A: Standard OAuth Flow (Recommended)**

```bash
# Run the standard OAuth flow
.venv/bin/python standard_auth_flow.py
```

**What will happen:**
1. Script will prompt for your Client Secret (from .env)
2. Browser will open to TradeStation authorization page
3. You'll log in and authorize the app
4. Browser redirects to localhost with authorization code
5. Script exchanges code for access token
6. Tokens saved automatically for future use

**Expected output:**
```
✅ Access Token: eyJhbGciOiJSUzI1Ni...
✅ Refresh Token: DPhH3vulejK39v79XyXc...
✅ Tokens saved to token storage
```

**Method B: Manual Token (If you have a valid token)**

If you have a recent access token, you can set it directly:

```bash
# Save your access token
echo "your_access_token_here" > .access_token
```

---

### Step 2: Verify Authentication

```bash
# Quick verification script
.venv/bin/python -c "
import asyncio
from pathlib import Path
from src.execution.tradestation.auth.oauth import OAuth2Client
from src.execution.tradestation.auth.tokens import TokenManager
from src.data.config import load_settings

async def check():
    settings = load_settings()
    token_manager = TokenManager()
    oauth_client = OAuth2Client(
        client_id=settings.tradestation_client_id,
        redirect_uri=settings.tradestation_redirect_uri,
        token_manager=token_manager,
    )

    if await oauth_client.get_access_token():
        print('✅ Authentication valid!')
        return True
    else:
        print('❌ Authentication failed')
        return False

asyncio.run(check())
"
```

---

### Step 3: Start Paper Trading System

```bash
# Start the system
.venv/bin/python start_paper_trading.py
```

**What you'll see:**
```
======================================================================
🚀 Starting TradeStation SIM Paper Trading System
======================================================================
Environment: development
Streaming Symbols: ['MNQH26']
Log Level: INFO

📡 Step 1: Initializing TradeStation SIM client...
✅ TradeStation SIM client authenticated
   API Base URL: https://sim-api.tradestation.com/v3

🤖 Step 2: Initializing ML Inference Engine...
✅ ML Inference ready

📊 Step 3: Initializing Position Tracker...
✅ Position Tracker ready

🛡️  Step 4: Initializing Risk Management (8 layers)...
✅ Risk Management initialized:
   1. Emergency Stop
   2. Daily Loss Limit ($500)
   3. Max Drawdown (12%)
   4. Max Position Size (5 contracts)
   5. Position Sizer (Kelly Criterion)
   6. Circuit Breaker Detector
   7. News Event Filter
   8. Per-Trade Risk Limit

📈 Step 5: Starting data pipeline (TradeStation SDK)...
✅ Data pipeline running

💰 Step 6: Paper trading monitor active
======================================================================
System is ready for paper trading!
Press Ctrl+C to stop gracefully
======================================================================

Monitoring paper trading activity...
Status: 0 open positions, Unrealized P&L: $0.00
```

---

## 🛑 Stopping the System

Press `Ctrl+C` to stop gracefully. The system will:
1. Stop receiving new market data
2. Close all positions (if configured)
3. Save final P&L calculations
4. Log shutdown summary

---

## 📊 Monitoring

### Real-time Logs
```bash
# Follow logs in real-time
tail -f logs/paper_trading.log
```

### CSV Audit Trail
```bash
# View risk audit trail
cat logs/risk_audit_trail.csv
```

### System Status
The system logs status every 30 seconds:
- Number of open positions
- Total unrealized P&L
- Pipeline metrics (bars created, transformed)

---

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# TradeStation API Credentials
TRADESTATION_CLIENT_ID=your_client_id
TRADESTATION_CLIENT_SECRET=your_client_secret
TRADESTATION_REDIRECT_URI=http://localhost:8080
TRADESTATION_REFRESH_TOKEN=your_refresh_token

# App Configuration
APP_ENV=development
LOG_LEVEL=INFO

# Trading Configuration (Optional)
STREAMING_SYMBOLS=["MNQH26"]  # Add more symbols as needed
```

### Risk Limits

All configured and active:
- **Daily Loss Limit:** $500
- **Max Drawdown:** 12%
- **Max Position:** 5 contracts
- **Per-Trade Risk:** $100

---

## 🐛 Troubleshooting

### Issue: "Authentication failed"
**Solution:** Run `standard_auth_flow.py` to get fresh tokens

### Issue: "No model found for horizon"
**Solution:** Models are optional for data flow. System will still stream data

### Issue: "Insufficient dollar bars"
**Solution:** System needs 20+ bars before ML inference. Normal behavior during startup.

### Issue: "Queue full"
**Solution:** Backpressure mechanism will slow down data ingestion. Logs warning.

---

## 📈 What Happens During Paper Trading

1. **Data Ingestion** (Continuous)
   - Streams real-time quotes from TradeStation SIM
   - Transforms to dollar bars ($50M notional)
   - Validates data quality

2. **Signal Detection** (When patterns emerge)
   - Detects Silver Bullet setups (MSS + FVG + Liquidity Sweep)
   - Engineers 40+ features from recent dollar bars
   - Runs ML inference for probability score

3. **Risk Validation** (Before every trade)
   - Checks all 8 risk layers
   - Only trades if all pass
   - Logs validation to CSV audit trail

4. **Order Execution** (SIM environment)
   - Submits order to TradeStation SIM
   - Instant fill (simulated)
   - No slippage in SIM

5. **Position Tracking** (Real-time)
   - Updates mark-to-market P&L
   - Monitors unrealized/realized P&L
   - Logs all position changes

---

## ✅ Verification Checklist

Before deploying, verify:
- [ ] OAuth authentication completed
- [ ] Access token is valid (check: `cat .access_token`)
- [ ] .env file has correct credentials
- [ ] Logs directory exists (`mkdir -p logs`)
- [ ] Data directory exists (`mkdir -p data/dollar_bars`)

---

## 🎯 Success Criteria

You'll know the system is working when you see:
```
Monitoring paper trading activity...
Status: 0 open positions, Unrealized P&L: $0.00
Pipeline: 42 bars created, 42 transformed
```

And when signals occur:
```
✅ Silver Bullet signal detected
🤖 ML Inference: 75.3% probability
🛡️ Risk validation: PASSED (8/8 layers)
💰 Order submitted: MNQH26 Buy 2 contracts
✅ Order filled: @ 11800.00
📊 Position opened: 2 contracts @ 11800.00
```

---

## 🆘 Need Help?

If you encounter issues:
1. Check logs: `tail -100 logs/paper_trading.log`
2. Verify authentication: Run `standard_auth_flow.py` again
3. Check credentials: Verify .env file values
4. Review spec: `_bmad-output/implementation-artifacts/tech-spec-tradestation-sim-paper-trading.md`

---

**System is production-ready. Happy paper trading!** 📊💰
