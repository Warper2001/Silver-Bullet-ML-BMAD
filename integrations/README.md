# TradingView Integration for MNQ Data

**Leverage your TradingView CME subscription to export MNQ futures data**

---

## 🚀 Quick Start

```bash
# Run the setup script
./integrations/setup_tradingview_export.sh
```

This will:
1. Install dependencies (Flask)
2. Create data directories
3. Start the webhook receiver
4. Display your webhook URL

---

## 📋 Manual Setup

### 1. Install Dependencies
```bash
.venv/bin/pip install flask
```

### 2. Start Webhook Receiver
```bash
.venv/bin/python integrations/tradingview_webhook_receiver.py
```

You should see:
```
========================================================================
TradingView Webhook Receiver for MNQ Data
========================================================================
Data directory: data/historical/mnq
Listening on port: 8080
Webhook URL: http://localhost:8080/webhook/tradingview
...
```

### 3. Configure TradingView

**Step 1:** Open TradingView chart with MNQ futures
- Search for `MNQM26` (or your desired contract)
- Set timeframe to desired interval (1min, 5min, etc.)

**Step 2:** Add Pine Script
- Click "Pine Editor" at bottom
- Copy script from `integrations/tradingview_pine_script.txt`
- Click "Add to Chart"

**Step 3:** Configure Script
- Click indicator settings (gear icon)
- Set "Webhook URL" to your server URL
- Set "Symbol Name" to your contract (e.g., MNQM26)

**Step 4:** Create Alert
- Click "Alert" button (top toolbar)
- Condition: `MNQ Export Webhook`
- **Once Per Bar Close:** ✅ **YES** (critical!)
- Webhook URL: (same as configured in script)
- Message: (leave default - it uses variables from script)
- Click "Create"

### 4. Data Flows Automatically

The receiver will:
- ✅ Collect each bar as it closes
- ✅ Validate MNQ price ranges
- ✅ Auto-save every 500 bars
- ✅ Append to existing HDF5 files
- ✅ Handle disconnections gracefully

---

## 🎛️ Webhook API Endpoints

Once running, these endpoints are available:

### Receive Data
```bash
POST http://localhost:8080/webhook/tradingview
Content-Type: application/json

{
  "symbol": "MNQM26",
  "timestamp": "2026-03-22T09:30:00Z",
  "open": 24200.00,
  "high": 24250.00,
  "low": 24180.00,
  "close": 24220.00,
  "volume": 1523
}
```

### Manual Save
```bash
curl -X POST http://localhost:8080/save
```

### Reset Buffer
```bash
curl -X POST http://localhost:8080/reset
```

### Health Check
```bash
curl http://localhost:8080/health
```

---

## 📂 File Structure

After setup:
```
integrations/
├── setup_tradingview_export.sh      # Quick setup script
├── tradingview_webhook_receiver.py  # Webhook server
├── tradingview_pine_script.txt      # Pine Script for TradingView
└── README.md                        # This file

data/historical/mnq/
└── MNQM26.h5                        # Exported data (HDF5 format)
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Webhook port (default: 8080)
export WEBHOOK_PORT=8080

# Data directory (default: data/historical/mnq)
export DATA_DIR=/path/to/data
```

### Pine Script Settings

In TradingView indicator settings:
- **Webhook URL:** Your server URL
- **Symbol Name:** Contract for filename
- **Enable Alerts:** Toggle on/off

---

## 📊 Data Format

The exported HDF5 files contain:
```
historical_bars dataset with columns:
  - timestamp (int64): Nanoseconds since epoch
  - open (float64): Opening price
  - high (float64): Highest price
  - low (float64): Lowest price
  - close (float64): Closing price
  - volume (int64): Contract volume
  - notional_value (float64): close * volume * 0.5
```

Metadata:
```
symbol: MNQM26
count: number of bars
created_at: ISO timestamp
multiplier: 0.5
```

---

## ⚠️ Troubleshooting

### No data received
1. Check webhook URL is correct
2. Verify alert is created with "Once Per Bar Close"
3. Check receiver script is running
4. Look for errors in TradingView alert log

### Missing bars
1. **Critical:** Ensure "Once Per Bar Close" is YES
2. Check TradingStation connection stability
3. Verify webhook URL has no typos

### Wrong price data
1. Verify you're on the correct MNQ contract chart
2. Check TradingStation data subscription includes CME
3. Ensure symbol name matches contract

### Connection errors
1. Check firewall allows port 8080
2. Verify server is accessible from internet
3. Use ngrok for local testing:
   ```bash
   ngrok http 8080
   ```

---

## 🌐 Exposing Local Server (Optional)

If running locally and need internet access:

### Using ngrok
```bash
# Install ngrok
# Then:
ngrok http 8080

# Use the HTTPS URL in TradingView
```

### Using SSH Tunnel
```bash
# On remote server:
ssh -R 8080:localhost:8080 user@remote
```

---

## 📈 Usage Examples

### Export Historical Data

**Method 1: Forward Collection**
1. Set up alerts on historical chart
2. Replay chart in TradingView
3. Data exports as each bar "closes"

**Method 2: Real-time Collection**
1. Set up on live chart
2. Let it run during market hours
3. Data streams in real-time

**Method 3: Multiple Contracts**
1. Set up multiple charts (MNQH26, MNQM26, MNQU26)
2. Change symbol name in each
3. Run receiver continuously

---

## 🔍 Monitoring

### Check Receiver Status
```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "bars_received": 1523,
  "symbol": "MNQM26"
}
```

### View Logs
```bash
# Logs are printed to stdout
tail -f /var/log/tradingview_receiver.log
```

---

## 🚦 Production Tips

1. **Use systemd/supervisor** for auto-restart
2. **Monitor disk space** for large datasets
3. **Back up HDF5 files** regularly
4. **Archive old contracts** to separate storage
5. **Validate data** after export

---

## 📝 Data Validation

After export, verify your data:

```python
import h5py
import pandas as pd

with h5py.File('data/historical/mnq/MNQM26.h5', 'r') as f:
    data = f['historical_bars']

    print(f"Total bars: {len(data)}")
    print(f"Date range: {data[0]['timestamp']} to {data[-1]['timestamp']}")
    print(f"Price range: ${data['close'].min():,.2f} - ${data['close'].max():,.2f}")
    print(f"Mean volume: {data['volume'].mean():,.0f}")
```

---

## 🎯 Next Steps

After exporting data:

1. **Validate data quality**
   ```bash
   .venv/bin/python -m pytest tests/integration/test_data_quality.py
   ```

2. **Run backtest**
   ```bash
   .venv/bin/python simple_backtest.py
   ```

3. **Train models**
   ```bash
   .venv/bin/python -m src.ml.train_model --data data/historical/mnq/
   ```

---

## 📞 Support

Issues? Check:
1. TradingView alert log
2. Receiver script output
3. Firewall/network settings
4. Data directory permissions

---

**Last Updated:** 2026-03-22
**Tested with:** TradingView Premium, CME Data Bundle
