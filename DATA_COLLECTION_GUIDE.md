# Data Collection Guide for MNQ Futures Trading System

This guide explains how to collect historical and real-time market data for your MNQ futures trading system.

## 🎯 Overview

Your trading system needs historical MNQ (Micro E-mini Nasdaq-100) futures data for:
- **Backtesting** - Testing strategies on historical data
- **Training ML models** - Teaching your AI to recognize patterns
- **Validation** - Ensuring strategies work before risking real money

## 📊 Data Requirements

### What You Need:
- **Symbol:** MNQ (Micro E-mini Nasdaq-100 Futures)
- **Frequency:** 5-minute bars (minimum), 1-minute preferred
- **History:** 6-12 months minimum, 2+ years preferred
- **Fields:** Open, High, Low, Close, Volume
- **Format:** HDF5 files (`.h5`)

### Data Storage Structure:
```
data/processed/dollar_bars/
├── MNQ_dollar_bars_202409.h5
├── MNQ_dollar_bars_202410.h5
├── MNQ_dollar_bars_202411.h5
├── MNQ_dollar_bars_202412.h5
├── MNQ_dollar_bars_202501.h5
└── ...
```

## 🔧 Data Collection Options

### Option 1: Test Data (Immediate, Free)
**Generate realistic synthetic data for testing**

```bash
# Generate 6 months of test data
venv/bin/python generate_test_data.py
```

✅ **Pros:** Free, immediate, realistic patterns
❌ **Cons:** Not real market data, synthetic patterns

### Option 2: TradeStation API (Real Data, Requires Credentials)
**Collect real historical and live data from TradeStation**

#### Step 1: Get TradeStation API Credentials

1. **Create Developer Account:**
   - Go to https://developers.tradestation.com/
   - Sign up for a free developer account
   - Verify your email address

2. **Create Application:**
   - Go to "My Apps" → "Create New App"
   - Give it a name: "MNQ Trading System"
   - Set Redirect URI: `http://localhost:8080/callback`
   - Note your **Client ID** and **Client Secret**

3. **Update Credentials:**
   ```bash
   # Edit your .env file
   nano .env

   # Replace with your real credentials:
   TRADESTATION_CLIENT_ID=your_real_client_id_here
   TRADESTATION_CLIENT_SECRET=your_real_client_secret_here
   ```

#### Step 2: Collect Historical Data

```bash
# Collect 3 months of historical data (5-min bars)
venv/bin/python collect_historical_data.py
```

**What it does:**
- Fetches historical MNQ data from TradeStation
- Saves to HDF5 format automatically
- Organizes by month for efficient loading

#### Step 3: Collect Real-Time Data (Optional)

```bash
# Collect 1 hour of live data
venv/bin/python collect_realtime_data.py
```

**What it does:**
- Connects to TradeStation WebSocket
- Streams live market data
- Creates 5-minute bars
- Auto-saves to HDF5 files

### Option 3: Free Data Sources (Alternative)
**Download free historical futures data**

#### Sources to Try:
1. **Interactive Brokers** - Has free historical data for account holders
2. **TD Ameritrade** - ThinkOrSwim has historical futures data
3. **NinjaTrader** - Free historical futures data
4. **Quandl** - Some free futures data available

#### Manual Import:
If you get data from another source, convert it to our format:

```python
import pandas as pd
import h5py

# Your data should have these columns:
# timestamp, open, high, low, close, volume

# Read your data (CSV example)
df = pd.read_csv('your_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Calculate dollar volume
df['dollar_volume'] = df['volume'] * df['close'] * 2  # MNQ: $2 per point

# Save to HDF5
data_array = []
for _, row in df.iterrows():
    ts_ms = int(row['timestamp'].timestamp() * 1000)
    data_array.append([
        ts_ms,
        row['open'],
        row['high'],
        row['low'],
        row['close'],
        row['volume'],
        row['dollar_volume']
    ])

with h5py.File('MNQ_dollar_bars_202409.h5', 'w') as f:
    f.create_dataset('dollar_bars', data=data_array)
```

## 🚀 Quick Start Guide

### For Testing (No Credentials Needed):
```bash
# 1. Generate test data
venv/bin/python generate_test_data.py

# 2. Run backtest
venv/bin/python simple_backtest.py
```

### For Real Data (With TradeStation Credentials):
```bash
# 1. Get credentials from https://developers.tradestation.com/

# 2. Update .env file with your credentials
nano .env

# 3. Collect historical data
venv/bin/python collect_historical_data.py

# 4. Run real backtest
venv/bin/python -m src.cli.backtest \
  --start 2024-12-19 \
  --end 2025-03-19 \
  --threshold 0.65 \
  --verbose
```

## 📋 Data Quality Checklist

Before using data for backtesting, ensure:

- ✅ **Completeness:** ≥ 99% of expected bars present
- ✅ **Accuracy:** Prices realistic for MNQ (around $21,000)
- ✅ **Volume:** Non-zero volume during market hours
- ✅ **Continuity:** No large unexplained gaps
- ✅ **Format:** Correct HDF5 structure

## 🔍 Data Validation

Check your collected data:

```python
import pandas as pd
from src.research.historical_data_loader import HistoricalDataLoader

# Load and validate data
loader = HistoricalDataLoader(
    data_directory="data/processed/dollar_bars/",
    min_completeness=99.99
)

try:
    data = loader.load_data("2024-12-19", "2025-03-19")
    print(f"✅ Loaded {len(data)} bars")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"Average volume: {data['volume'].mean():.0f}")
except Exception as e:
    print(f"❌ Error: {e}")
```

## 💡 Tips for Better Data

1. **Start with test data** - Verify your system works
2. **Get 6+ months** - Minimum for meaningful backtests
3. **Include different market conditions** - Bull, bear, sideways
4. **Check data quality** - Look for gaps or errors
5. **Keep it updated** - Collect new data regularly

## 🆘 Troubleshooting

### "Invalid API Key" Error:
- Your TradeStation credentials are expired or invalid
- Get new credentials from https://developers.tradestation.com/
- Update your .env file

### "No HDF5 files found" Error:
- Check data directory exists: `ls -la data/processed/dollar_bars/`
- Verify files have correct naming: `MNQ_dollar_bars_YYYYMM.h5`
- Ensure date range matches your data

### Authentication Failed:
- Verify Client ID and Secret are correct
- Check Redirect URI matches in TradeStation
- Ensure your account has API access enabled

## 📞 Next Steps

1. **Generate test data** (immediate)
   ```bash
   venv/bin/python generate_test_data.py
   ```

2. **Test backtesting framework**
   ```bash
   venv/bin/python simple_backtest.py
   ```

3. **Get TradeStation credentials** (for real data)
   - Visit https://developers.tradestation.com/
   - Create developer account
   - Generate API credentials

4. **Collect real historical data**
   ```bash
   venv/bin/python collect_historical_data.py
   ```

5. **Run realistic backtests**
   ```bash
   venv/bin/python -m src.cli.backtest --start 2024-09-01 --end 2025-03-19 --verbose
   ```

---

**Remember:** Test data is great for development, but real market data is essential for reliable backtesting results! 🎯