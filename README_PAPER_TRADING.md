# TIER 1 Paper Trading - Quick Start Guide

## 🚀 Market Open Startup (6:00 PM CDT)

### Step 1: Start Paper Trading
```bash
./start_paper_trading_at_open.sh
```

### Step 2: Verify Connection
```bash
# Check system status
./deploy_tier1_realtime.sh status

# Monitor live logs
./deploy_tier1_realtime.sh logs

# Verify OAuth token refresh
./monitor_oauth_refresh.sh
```

## ⚠️ CRITICAL: OAuth Token Refresh

**The system MUST refresh OAuth tokens every 10 minutes or API access will be BLOCKED!**

### What This Means:
- ✅ Token refresh runs automatically in background
- ✅ Log entries: "Running scheduled token refresh..."
- ✅ Success: "Token refreshed successfully (expires at HH:MM:SS)"
- ❌ Failure: If refresh stops, you'll need new authorization code

### How to Monitor:
```bash
# Check recent refresh activity
./monitor_oauth_refresh.sh

# Manual log check
tail -f logs/tier1_paper_rest.log | grep -i "token refresh"
```

### If Refresh Stops:
1. **IMMEDIATELY** stop the system: `./deploy_tier1_realtime.sh stop`
2. Contact user to generate new authorization code
3. Restart with fresh token

## 📊 System Status Commands

```bash
# Full status report
./deploy_tier1_realtime.sh status

# Live log monitoring
./deploy_tier1_realtime.sh logs

# OAuth refresh monitoring
./monitor_oauth_refresh.sh

# Stop system
./deploy_tier1_realtime.sh stop

# Restart system
./deploy_tier1_realtime.sh restart
```

## 🎯 Performance Monitoring

### Expected Metrics:
- Win Rate: 73-89% (target: ≥60%)
- Profit Factor: 1.7-4.6 (target: ≥1.7)
- Trade Frequency: 10-15/day (target: 8-15/day)
- Expectancy: $11-27/trade

### Log Patterns to Watch:
```
✅ "PAPER TRADE: LONG entry $XXXXX"     # New trade generated
✅ "PAPER TRADE: TP hit +$XXX"          # Take profit
✅ "PAPER TRADE: SL hit -$XXX"          # Stop loss
✅ "Token refreshed successfully"       # OAuth working
❌ "401 Unauthorized"                   # OAuth problem
❌ "Failed to fetch new bars"           # API problem
```

## 🔧 Troubleshooting

### No Trades Generated:
- Check if markets are open
- Verify API connection: `./deploy_tier1_realtime.sh status`
- Check for error messages in logs

### API Connection Issues:
- Verify OAuth token refresh working: `./monitor_oauth_refresh.sh`
- Check TradeStation API status
- Look for 404/401 errors in logs

### System Crashes:
- Check `logs/tier1_paper_rest.log` for error messages
- Restart: `./deploy_tier1_realtime.sh restart`
- If OAuth errors, get new authorization code

## 📈 Configuration

**Strategy:** TIER 1 FVG with optimal filters
- Stop Loss: 2.5× gap size
- ATR Threshold: 0.7×
- Volume Ratio: 2.25×
- Max Gap Size: $50
- Max Hold Time: 10 bars

**Transaction Costs:** $1.40/round-trip (realistic)
**Contracts:** 1 contract per trade
**Symbol:** MNQ/M26 (Micro E-mini Nasdaq-100 June 2026)

---

## ✅ Pre-Market Checklist (6:00 PM CDT)

- [ ] Start system: `./start_paper_trading_at_open.sh`
- [ ] Verify status: `./deploy_tier1_realtime.sh status`
- [ ] Check OAuth refresh: `./monitor_oauth_refresh.sh`
- [ ] Monitor first 5 minutes for trade generation
- [ ] Verify API data flow: Look for "Dollar Bars: [count]"

## 🔄 Continuous Monitoring

**Every 30 minutes:**
```bash
./monitor_oauth_refresh.sh  # Critical - don't skip!
./deploy_tier1_realtime.sh status
```

**Every 2 hours:**
- Review trade count and performance
- Check for any error patterns
- Verify system stability

