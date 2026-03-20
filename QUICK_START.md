# 🚀 Quick Start - Data Collection & Backtesting

## ⚡ Fast Path (Works Right Now)

```bash
# 1. Generate test data (30 seconds)
venv/bin/python generate_test_data.py

# 2. Run backtest (1 minute)
venv/bin/python simple_backtest.py

# ✅ Done! You'll see backtest results.
```

## 📊 What You Just Did

- ✅ Generated 6+ months of realistic MNQ test data
- ✅ Ran a complete backtest with 100 simulated trades
- ✅ Got performance metrics (Sharpe ratio, win rate, returns)
- ✅ Created CSV reports in `data/reports/`

## 🎯 Next Steps

### **Option A: Continue Testing (No credentials needed)**
```bash
# Generate more test data
venv/bin/python generate_test_data.py

# Run more backtests
venv/bin/python simple_backtest.py
```

### **Option B: Get Real Data (Requires TradeStation account)**
```bash
# 1. Get credentials from https://developers.tradestation.com/

# 2. Update .env file with your Client ID and Secret

# 3. Test authentication
venv/bin/python test_auth.py

# 4. Collect real historical data
venv/bin/python collect_historical_data.py
```

## 📈 Expected Results

When you run the backtest, you should see:
```
🎉 BACKTEST COMPLETED!
📅 Summary: 100 trades executed
💰 Total Return: 3.53%
📊 Sharpe Ratio: 20.08
🎯 Win Rate: 98.0%
```

## 🆘 Quick Help

**Problem? Try this:**
```bash
# Check if data exists
ls -la data/processed/dollar_bars/

# Generate fresh data
venv/bin/python generate_test_data.py

# Run backtest again
venv/bin/python simple_backtest.py
```

## 📚 Full Documentation

- **Complete Guide:** `DATA_COLLECTION_GUIDE.md`
- **Status Summary:** `DATA_COLLECTION_COMPLETE.md`
- **Project README:** `README.md`

---

**✅ You're ready to go!** Run the quick start commands above to see your trading system in action! 🚀