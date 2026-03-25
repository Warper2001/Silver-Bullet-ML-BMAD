# Alternative Historical Data Sources for MNQ Futures

**Research Date:** 2026-03-22
**Status:** TradeStation `/bars` endpoint requires upgraded subscription

---

## 🎯 Quick Summary

Your TradeStation credentials support real-time quotes and WebSocket streams but **NOT** historical bar data. Here are alternative sources for obtaining MNQ historical data.

---

## ✅ Free Data Sources

### 1. **CME Group (Official Exchange)**
- **URL:** https://www.cmegroup.com/market-data/historical-data.html
- **Data:** Official MNQ futures data
- **Cost:** Free for daily data
- **Format:** CSV download
- **Pros:** Official source, reliable
- **Cons:** Daily resolution only (no intraday)

### 2. **Interactive Brokers**
- **URL:** https://www.interactivebrokers.com/
- **Data:** Historical futures data
- **Cost:** Free with IB account
- **Format:** API or CSV export
- **Pros:** High-quality data, intraday available
- **Cons:** Requires account opening

### 3. **TradingView**
- **URL:** https://www.tradingview.com/
- **Data:** MNQ historical charts
- **Cost:** Free tier available
- **Format:** Export via scripts (Pro/Pro+)
- **Pros:** Easy to view data
- **Cons:** Limited export on free tier

### 4. **HistData.com**
- **URL:** https://www.histdata.com/
- **Data:** Futures historical data
- **Cost:** Free for some datasets
- **Format:** CSV
- **Pros:** Clean data format
- **Cons:** Limited futures coverage

### 5. **QuantConnect**
- **URL:** https://www.quantconnect.com/
- **Data:** Historical futures data
- **Cost:** Free tier available
- **Format:** API access
- **Pros:** Good for backtesting
- **Cons:** Requires using their platform

---

## 💰 Paid Data Sources

### 1. **Interactive Brokers (Premium)**
- **Cost:** Included with account
- **Resolution:** Tick to daily
- **History:** Several years
- **Access:** API + download

### 2. **Kibot**
- **URL:** https://www.kibot.com/
- **Cost:** ~$50-200/month
- **Resolution:** Tick, minute, daily
- **History:** 10+ years
- **Quality:** Professional grade

### 3. **TickDataSuite**
- **URL:** https://tickdata-suite.com/
- **Cost:** ~$100-300/month
- **Resolution:** Tick, minute
- **History:** 20+ years
- **Quality:** Institutional quality

### 4. **Norgate Data**
- **URL:** https://www.norgatedata.com/
- **Cost:** ~$50-100/month
- **Resolution:** Daily, weekly
- **History:** 40+ years
- **Focus:** Futures & stocks

### 5. **Pristine Data**
- **URL:** https://pristinedata.com/
- **Cost:** ~$100/month
- **Resolution:** Minute, daily
- **History:** 10+ years
- **Quality:** Cleaned and adjusted

---

## 🔧 API-Based Solutions

### 1. **Polygon.io**
- **URL:** https://polygon.io/
- **Cost:** Free tier available
- **Data:** Stocks and some futures
- **API:** REST + WebSocket
- **Free:** Limited calls/month

### 2. **Alpha Vantage**
- **URL:** https://www.alphavantage.co/
- **Cost:** Free tier available
- **Data:** Stocks and forex
- **API:** REST
- **Note:** Limited futures coverage

### 3. **IEX Cloud**
- **URL:** https://iexcloud.io/
- **Cost:** Free tier available
- **Data:** Primarily US stocks
- **API:** REST
- **Note:** No futures data

---

## 📊 Data Format Requirements

Your system requires:
```
Format: HDF5 (.h5)
Structure:
  - timestamp (int64 nanoseconds)
  - open (float64)
  - high (float64)
  - low (float64)
  - close (float64)
  - volume (int64)
  - notional_value (float64) = close * volume * 0.5
```

**Conversion script needed** for CSV → HDF5

---

## 🚀 Recommended Next Steps

### Option A: **Quick Start (Free)**
1. Download daily MNQ data from CME Group
2. Convert to HDF5 format
3. Use for initial backtesting
4. Upgrade to minute data later

### Option B: **Investment ($50-100/month)**
1. Sign up for Kibot or Norgate Data
2. Download 2+ years of minute data
3. Format for your system
4. Have production-ready dataset

### Option C: **Build Over Time**
1. Use existing `collect_realtime_data.py`
2. Stream data during market hours
3. Build historical dataset over weeks/months
4. No upfront cost, takes time

---

## 📝 Implementation Checklist

- [ ] Choose data source
- [ ] Sign up for account (if needed)
- [ ] Download historical MNQ data
- [ ] Convert CSV to HDF5 format
- [ ] Validate data quality
- [ ] Test with backtesting pipeline
- [ ] Document data provenance

---

## 🤖 TradeStation Alternative

If you want to stay with TradeStation, contact them about:
- **TradeStation API Premium Tier**
- **Historical Data Package**
- **MarketData Access Levels**

The `/bars` endpoint may require:
- Professional account tier
- Historical data add-on
- Institutional API access

---

**Last Updated:** 2026-03-22
**Next Review:** After testing data source
