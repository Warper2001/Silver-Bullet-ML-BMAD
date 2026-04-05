# Paper Trading Deployment Guide

**Last Updated:** 2026-04-04
**Epic:** Epic 4 - Paper Trading Deployment
**Environment:** TradeStation SIM (Paper Trading)
**System:** Ensemble-Weighted Confidence System

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [TradeStation SIM Configuration](#tradestation-sim-configuration)
4. [Deployment Steps](#deployment-steps)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)
7. [Shutdown Procedures](#shutdown-procedures)

---

## Prerequisites

### System Requirements

- **OS:** Linux (Ubuntu 20.04+ recommended) or macOS 10.15+
- **Python:** 3.11 or higher
- **Memory:** 4GB RAM minimum, 8GB recommended
- **Disk:** 10GB free space for logs and data
- **Network:** Stable internet connection for TradeStation WebSocket

### Software Requirements

```bash
# Check Python version
python --version  # Should be 3.11+

# Check if Poetry is installed (for dependency management)
poetry --version

# Check if Git is installed
git --version
```

### TradeStation SIM Account

Before starting, you need:
- ✅ TradeStation account with SIM (paper trading) enabled
- ✅ TradeStation API credentials (App ID, App Secret, Refresh Token)
- ✅ MNQ (Micro E-mini Nasdaq-100) trading enabled in SIM
- ✅ Sufficient simulated buying power (minimum $50,000 recommended)

---

## Environment Setup

### 1. Clone Repository (if not already done)

```bash
git clone <repository-url>
cd Silver-Bullet-ML-BMAD
```

### 2. Install Dependencies

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Activate virtual environment
poetry shell
# OR use .venv directly:
source .venv/bin/activate
```

### 3. Verify Installation

```bash
# Check Python packages
.venv/bin/python -m pip list | grep -E "(pandas|numpy|pydantic|streamlit|xgboost)"

# Run tests to verify installation
.venv/bin/python -m pytest tests/ -v
```

---

## TradeStation SIM Configuration

### Step 1: Obtain TradeStation API Credentials

1. Log in to TradeStation (https://tradingstation.com)
2. Navigate to: **Account → API Management**
3. Create a new API application:
   - **Application Name:** Silver-Bullet-ML-BMAD-PROD
   - **Redirect URI:** http://localhost:8080/callback
   - **Scope:** Read, Trade
4. Save your credentials:
   - **App ID (Client ID)**
   - **App Secret (Client Secret)**
   - **Refresh Token**

### Step 2: Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Copy example file
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
# TradeStation API Credentials
TRADESTATION_APP_ID=your_app_id_here
TRADESTATION_APP_SECRET=your_app_secret_here
TRADESTATION_REFRESH_TOKEN=your_refresh_token_here

# Environment (SIM for paper trading, PROD for live)
TRADESTATION_ENVIRONMENT=SIM

# Account ID (found in TradeStation dashboard)
TRADESTATION_ACCOUNT_ID=your_sim_account_id

# Symbol to trade
TRADING_SYMBOL=MNQ  # Micro E-mini Nasdaq-100

# Trading Mode (PAPER_TRADING or LIVE_TRADING)
TRADING_MODE=PAPER_TRADING
```

**⚠️ SECURITY WARNING:** Never commit `.env` to Git! It's already in `.gitignore`.

### Step 3: Verify Credentials

Test your credentials:

```bash
.venv/bin/python -c "
from src.data.tradestation_auth import TradeStationAuth
import os
from dotenv import load_dotenv

load_dotenv()
auth = TradeStationAuth(
    app_id=os.getenv('TRADESTATION_APP_ID'),
    app_secret=os.getenv('TRADESTATION_APP_SECRET'),
    refresh_token=os.getenv('TRADESTATION_REFRESH_TOKEN'),
    environment=os.getenv('TRADESTATION_ENVIRONMENT', 'SIM')
)

# Test authentication
token = auth.get_access_token()
print(f'✅ Authentication successful! Token: {token[:20]}...')
print(f'Environment: {auth.environment}')
"
```

Expected output:
```
✅ Authentication successful! Token: eyJhbGciOiJIUzI1NiI...
Environment: SIM
```

If authentication fails, verify:
- Credentials are correct (no extra spaces)
- API app has "Read" and "Trade" scopes
- Refresh token is valid (not expired)

---

## Deployment Steps

### Step 1: Review System Configuration

Review `config.yaml`:

```bash
# View current configuration
cat config.yaml
```

**Key settings to verify:**

```yaml
# Risk Limits
risk:
  daily_loss_limit: 500  # USD - maximum daily loss
  max_drawdown_percent: 12  # % - maximum drawdown from peak
  max_position_size: 5  # contracts - maximum position
  per_trade_risk_percent: 2  # % - risk per trade

# ML Threshold
ml:
  probability_threshold: 0.65  # Minimum success probability

# Data Pipeline
data:
  dollar_bar_threshold: 50000000  # $50M notional value

# Trading Hours
trading:
  start_time: "09:30:00"  # Market open (Eastern)
  end_time: "16:00:00"    # Market close (Eastern)
  timezone: "America/New_York"
```

**⚠️ IMPORTANT:** Do not change risk limits without understanding the implications!

### Step 2: Create Required Directories

```bash
# Create data and log directories
mkdir -p data/processed/dollar_bars
mkdir -p data/state
mkdir -p logs
mkdir -p models/xgboost/5_minute

# Verify directories exist
ls -la data/ logs/
```

### Step 3: Validate Models

Ensure XGBoost models are present:

```bash
# Check for models
ls -lh models/xgboost/5_minute/

# Expected files:
# - model.joblib (trained XGBoost model)
# - preprocessor.pkl (feature preprocessing pipeline)
# - metadata.json (model performance metrics)
# - threshold.json (optimal probability threshold)
```

If models are missing, train them:

```bash
# Generate training data and train model
.venv/bin/python generate_ml_training_data.py
.venv/bin/python train_meta_model.py
```

### Step 4: Pre-Deployment Validation

Run the deployment validator:

```bash
# This checks prerequisites, authentication, and system readiness
./deploy_paper_trading.sh validate
```

Expected output:
```
✅ Python version: 3.11.5
✅ Virtual environment: .venv
✅ Required packages installed
✅ TradeStation authentication: SUCCESS
✅ Models present: model.joblib, preprocessor.pkl
✅ Directories created: data, logs, models
✅ Configuration valid: config.yaml

✅ DEPLOYMENT VALIDATION PASSED
Ready to start paper trading system.
```

### Step 5: Start Paper Trading System

Start the system in detached mode (runs in background):

```bash
./deploy_paper_trading.sh start
```

Expected output:
```
Starting paper trading system...

[2026-04-04 09:30:00] INFO: Starting Silver Bullet Paper Trading System
[2026-04-04 09:30:00] INFO: Environment: SIM (Paper Trading)
[2026-04-04 09:30:01] INFO: TradeStation authentication: SUCCESS
[2026-04-04 09:30:01] INFO: Connecting to WebSocket...
[2026-04-04 09:30:02] INFO: WebSocket connected
[2026-04-04 09:30:02] INFO: Subscribing to MNQ market data...
[2026-04-04 09:30:03] INFO: Subscription successful
[2026-04-04 09:30:03] INFO: Starting dashboard on http://localhost:8501
[2026-04-04 09:30:04] INFO: System started successfully

✅ Paper trading system started
Dashboard: http://localhost:8501
PID: 12345
Logs: logs/paper_trading.log
```

**🎉 Success!** The system is now running.

### Step 6: Verify Deployment

1. **Check Dashboard**
   - Open: http://localhost:8501
   - Verify "Overview" page shows account equity
   - Check "Signals" page for any signals
   - Verify "Settings" page shows configuration

2. **Check Logs**
   ```bash
   tail -f logs/paper_trading.log
   ```
   Look for:
   - `INFO: WebSocket connected` ✅
   - `INFO: Subscribing to MNQ market data` ✅
   - `INFO: Dollar bar created` (appears when trading)
   - No `ERROR` messages

3. **Check Process Status**
   ```bash
   ./deploy_paper_trading.sh status
   ```

   Expected output:
   ```
   Paper Trading System Status: RUNNING
   PID: 12345
   Uptime: 5 minutes
   Memory: 250MB
   Log Size: 2.3MB
   ```

4. **Verify WebSocket Connection**
   ```bash
   # Check for WebSocket messages in logs
   grep "messages_received" logs/paper_trading.log | tail -5
   ```

   Should show incrementing count:
   ```
   [2026-04-04 09:35:00] DEBUG: messages_received=125, bars_created=3
   [2026-04-04 09:40:00] DEBUG: messages_received=287, bars_created=8
   ```

   If `messages_received=0`, WebSocket is not connected (see [Troubleshooting](#troubleshooting))

---

## Verification

### Health Check Script

Run automated health checks:

```bash
.venv/bin/python scripts/health_check.py
```

This verifies:
- ✅ Process is running
- ✅ WebSocket is connected
- ✅ Market data is flowing
- ✅ Dashboard is accessible
- ✅ Risk management is active
- ✅ CSV audit trails are being written

### Manual Verification Checklist

- [ ] Dashboard loads at http://localhost:8501
- [ ] Account equity displays correctly on Overview page
- [ ] No open positions (fresh start)
- [ ] WebSocket messages received count > 0
- [ ] Dollar bars are being created (during market hours)
- [ ] CSV audit trails exist: `logs/daily_loss.csv`, `logs/drawdown.csv`
- [ ] No ERROR messages in `logs/paper_trading.log`
- [ ] Risk limits display correctly in Settings page

---

## Troubleshooting

### Issue 1: "Authentication Failed"

**Symptoms:**
```
ERROR: TradeStation authentication failed
ERROR: Invalid credentials
```

**Solutions:**
1. Verify credentials in `.env` (no extra spaces)
2. Check API app is enabled in TradeStation dashboard
3. Regenerate refresh token in TradeStation dashboard
4. Verify environment is `SIM` (not `PROD`)

**Test credentials:**
```bash
.venv/bin/python -c "
from src.data.tradestation_auth import TradeStationAuth
from dotenv import load_dotenv
import os
load_dotenv()
auth = TradeStationAuth(
    app_id=os.getenv('TRADESTATION_APP_ID'),
    app_secret=os.getenv('TRADESTATION_APP_SECRET'),
    refresh_token=os.getenv('TRADESTATION_REFRESH_TOKEN'),
    environment='SIM'
)
print(auth.get_access_token())
"
```

### Issue 2: "WebSocket Connection Failed"

**Symptoms:**
```
ERROR: WebSocket connection failed
ERROR: Unable to establish connection
```
`messages_received=0` in logs

**Solutions:**
1. Check internet connection
2. Verify TradeStation services are operational (https://status.tradestation.com)
3. Check firewall rules (WebSocket uses wss:// on port 443)
4. Verify API app has "Read" scope

**Test WebSocket connection:**
```bash
.venv/bin/python scripts/test_websocket.py
```

### Issue 3: "No Market Data Flowing"

**Symptoms:**
```
messages_received=0, bars_created=0
```
WebSocket connected but no data

**Solutions:**
1. Verify market is open (Mon-Fri, 9:30 AM - 4:00 PM ET)
2. Check symbol subscription (`MNQ` for Micro Nasdaq-100)
3. Verify account has market data permissions for MNQ
4. Check TradeStation dashboard for real-time quotes

**Market hours check:**
```bash
.venv/bin/python -c "
from datetime import datetime
import pytz
et = pytz.timezone('America/New_York')
now = datetime.now(et)
print(f'Current time (ET): {now.strftime(\"%Y-%m-%d %H:%M:%S %Z\")}')
print(f'Market open: 09:30:00 - 16:00:00 ET')
print(f'Weekend: {now.weekday() >= 5}')
"
```

### Issue 4: "Dashboard Not Loading"

**Symptoms:**
- Browser shows "Connection refused"
- http://localhost:8501 doesn't load

**Solutions:**
1. Check if Streamlit process is running:
   ```bash
   ps aux | grep streamlit
   ```
2. Kill existing process and restart:
   ```bash
   pkill -f streamlit
   ./deploy_paper_trading.sh start
   ```
3. Check if port 8501 is available:
   ```bash
   lsof -i :8501
   ```

### Issue 5: "Risk Limit Breached - System Halted"

**Symptoms:**
```
WARNING: Daily loss limit breached (-$600 vs $500 limit)
INFO: Trading halted - Risk management triggered
```

**Solutions:**
1. This is **expected behavior** - system is protecting itself
2. Review what caused the loss in `logs/paper_trading.log`
3. Check CSV audit trails: `logs/daily_loss.csv`, `logs/drawdown.csv`
4. Reset daily loss limit (new day):
   - Edit `logs/daily_loss.csv`
   - Set current loss to 0
   - Restart system: `./deploy_paper_trading.sh restart`

**⚠️ WARNING:** Only reset limits if you understand the root cause!

### Issue 6: "Memory/CPU Too High"

**Symptoms:**
- System using > 2GB RAM
- CPU consistently > 50%

**Solutions:**
1. Check log file size (logs can grow large):
   ```bash
   du -sh logs/paper_trading.log
   ```
2. Rotate logs if > 100MB:
   ```bash
   mv logs/paper_trading.log logs/paper_trading.log.old
   ./deploy_paper_trading.sh restart
   ```
3. Reduce data retention in `config.yaml`

### Issue 7: "Model Not Found"

**Symptoms:**
```
ERROR: Model file not found: models/xgboost/5_minute/model.joblib
```

**Solutions:**
1. Train model:
   ```bash
   .venv/bin/python generate_ml_training_data.py
   .venv/bin/python train_meta_model.py
   ```
2. Verify model files created:
   ```bash
   ls -lh models/xgboost/5_minute/
   ```

---

## Shutdown Procedures

### Graceful Shutdown

To stop the paper trading system gracefully:

```bash
./deploy_paper_trading.sh stop
```

Expected output:
```
Stopping paper trading system...

[2026-04-04 15:55:00] INFO: Shutdown signal received
[2026-04-04 15:55:00] INFO: Closing WebSocket connection...
[2026-04-04 15:55:01] INFO: Flushing buffers...
[2026-04-04 15:55:01] INFO: Saving state...
[2026-04-04 15:55:02] INFO: Shutdown complete

✅ Paper trading system stopped
```

### Emergency Stop

For immediate shutdown (use only in emergencies):

```bash
# Option 1: Use emergency stop script
.venv/bin/python src/cli/emergency_stop.py --reason "Manual emergency stop"

# Option 2: Kill process directly
./deploy_paper_trading.sh stop --force
```

**⚠️ WARNING:** Emergency stop may interrupt active trades or data processing!

### Restart System

To restart the system:

```bash
./deploy_paper_trading.sh restart
```

This is equivalent to:
```bash
./deploy_paper_trading.sh stop
# Wait for shutdown
./deploy_paper_trading.sh start
```

---

## Post-Deployment

### First Day Monitoring

On the first day of deployment:

1. **Hour 0-1 (Initial Startup):**
   - Monitor logs: `tail -f logs/paper_trading.log`
   - Check dashboard every 10 minutes
   - Verify WebSocket connection stable
   - Ensure no ERROR messages

2. **Hour 1-4 (Monitoring):**
   - Check for signals generated
   - Verify risk management working
   - Monitor CSV audit trails
   - Check memory/CPU usage

3. **End of Day:**
   - Generate daily report: `.venv/bin/python src/monitoring/daily_report_generator.py`
   - Review all trades executed
   - Verify P&L calculations
   - Document any anomalies

### Ongoing Maintenance

**Daily:**
- Check dashboard for account equity and drawdown
- Review logs for errors or warnings
- Verify risk limits not breached

**Weekly:**
- Generate weekly performance report
- Analyze signal frequency and win rate
- Review weight optimization history
- Backup log files

**Monthly:**
- Review overall performance vs expectations
- Validate system stability
- Check for model drift
- Update documentation if needed

---

## Support & Resources

### Documentation

- **Project README:** `README.md`
- **Operations Guide:** `OPERATIONS.md`
- **Epic 4 Completion Report:** `EPIC4_COMPLETION_REPORT.md`
- **Epic 4 Retrospective:** `_bmad-output/epic-retrospectives/epic-4-retrospective.md`

### Configuration Files

- **System Config:** `config.yaml`
- **Environment Variables:** `.env` (do not commit!)
- **Poetry Dependencies:** `pyproject.toml`

### Log Files

- **Main Log:** `logs/paper_trading.log`
- **Daily Loss:** `logs/daily_loss.csv`
- **Drawdown:** `logs/drawdown.csv`
- **Per-Trade Risk:** `logs/per_trade_risk.csv`
- **Emergency Stop:** `logs/emergency_stop.csv`

### State Files

- **Weight History:** `data/state/weight_history.csv`
- **Models:** `models/xgboost/5_minute/`

### Getting Help

If you encounter issues not covered in this guide:

1. Check troubleshooting section above
2. Review logs: `grep ERROR logs/paper_trading.log`
3. Run health check: `.venv/bin/python scripts/health_check.py`
4. Consult Epic 4 retrospective for known issues

---

**Deployment Guide Version:** 1.0
**Last Updated:** 2026-04-04
**Maintained By:** Development Team
