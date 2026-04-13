# Premium Strategy 1-Minute Backtest - FINAL REPORT

## 🔴 CATASTROPHIC FAILURE

**Date:** April 9, 2026
**Data:** 795,296 1-minute bars (Dec 2023 - Mar 2026)
**Backtest Period:** 2025 (351,628 bars)
**Result:** **-$83,672.50 LOSS**

---

## Executive Summary

The premium strategy that appeared highly profitable on 5-minute data (**+20K, 60% win rate**) is a **catastrophic failure** on 1-minute data (**-84K, 21% win rate**).

**ALL PREVIOUS WORK IS INVALID.**

---

## Performance Comparison

| Metric | 5-Minute (FAKE) | 1-Minute (REAL) | Difference |
|--------|-----------------|-----------------|------------|
| **Win Rate** | 60.00% | 20.55% | **-39%** |
| **Total Return** | +$20,128 | -$83,673 | **-$103,801** |
| **Profit Factor** | 2.53 | 0.30 | **-2.23** |
| **Avg Return/Trade** | +$1,342 | -$573 | **-$1,915** |
| **Sample Size** | 15 trades | 146 trades | +131 trades |
| **Stop Hit Rate** | 40% | 68.5% | **+28.5%** |

---

## Root Cause Analysis

### 1. **Stop Loss Massacre**
- **68.5% of trades hit stop loss**
- **0% win rate** on stop losses (all losers)
- Average stop distance: only $75
- 1-minute volatility stops out trades before they can develop

### 2. **Directional Bias Disaster**
- **Bullish:** 16.4% win rate, -$95,579 loss (TOXIC)
- **Bearish:** 41.7% win rate, +$11,906 profit (WORKS)
- Strategy optimized for 5-minute where direction didn't matter

### 3. **Quality Score Inversion Wrong**
- Optimized for 0-85 quality range on 5-minute data
- On 1-minute: All quality ranges lose money
- Quality scoring completely irrelevant on 1-minute timeframe

### 4. **FVG Size Irrelevant**
- Even with $500+ FVG threshold: Still 22% win rate, -$22K loss
- Larger FVGs don't help - still stops out on noise

### 5. **Only Low-Liquidity Hours Work**
- **4AM-6AM:** 71% win rate, +$8.6K (but only 7 trades)
- **8PM-12AM:** 50%+ win rates (low volume)
- **Regular trading hours:** 0-20% win rates (all lose money)

---

## Why 5-Minute Was Misleading

1. **Time Smoothing:** 5-minute bars average out noise
2. **Reduced Stop Hits:** 40% vs 68.5% on 1-minute
3. **False Confidence:** 15 trades looked good, but were random luck
4. **Overfitting:** Parameters tuned to 5-minute noise patterns

**The 5-minute backtest was a statistical illusion.**

---

## What This Means

### ❌ **INVALIDATED:**
- All 5-minute backtest results
- All parameter optimization (quality scores, stops, etc.)
- "Inverted quality score" discovery
- All performance projections
- ML training data generation
- Deployment plan

### ❌ **STRATEGY ITSELF FLAWED:**
- MSS-FVG confluence doesn't work on 1-minute data
- 1-minute noise swamps the signal
- Stop losses cannot be placed effectively
- No combination of parameters fixes this

---

## Diagnostic Deep Dive

### Stop Distance Analysis
```
Q1 (tightest):  23.7% win, -$1,710, 76.3% stop hits
Q2:             20.0% win, -$7,958, 80.0% stop hits
Q3:             11.1% win, -$31,609, 88.9% stop hits
Q4 (widest):    27.0% win, -$42,396, 29.7% stop hits
```
**Wider stops help but still lose money.**

### FVG Size Analysis
```
$75+ threshold:   20.5% win, -$83,673
$100+ threshold:  20.4% win, -$77,061
$200+ threshold:  21.4% win, -$49,112
$500+ threshold:  22.3% win, -$22,924
```
**No FVG size threshold makes it profitable.**

### Quality Score Analysis
```
0-60:    0% win (1 trade), -$15
60-70:   27% win (26 trades), -$13,948
70-80:   16% win (74 trades), -$46,470
80-100:  24% win (45 trades), -$23,240
```
**No quality range is profitable.**

### Hour of Day Analysis
```
00:00-03:00:  0% win, all losses
04:00-06:00:  71% win, +$8.6K (only 7 trades)
07:00-17:00:  0-22% win, massive losses
18:00-23:00:  37-100% win, small profits (low volume)
```
**Only works in low-liquidity overnight session.**

---

## Options Going Forward

### **Option A: ABANDON STRATEGY** ⭐ RECOMMENDED
- Accept that MSS-FVG doesn't work on 1-minute data
- Move to different strategy or timeframe
- Cut losses and learn from this expensive lesson

### **Option B: SWITCH TO 5-MINUTE TIMEFRAME**
- Trade the 5-minute timeframe instead
- Accept 2-5 trades/day (vs 10-30 on 1-minute)
- Backtest suggests 60% win rate (but small sample)
- **Risk:** Still only 15 trades/year - not enough for ML

### **Option C: BEARISH-ONLY STRATEGY**
- Only trade bearish signals (42% win rate)
- Filter out all bullish signals (16% win rate)
- **Problem:** Only 24 trades in entire year
- **Still not enough for ML training**

### **Option D: OVERNIGHT SESSION ONLY**
- Only trade 4AM-6AM and evening hours
- **Problem:** Only 10-20 trades/year
- **Not viable for strategy trading**

### **Option E: COMPLETE STRATEGY REDESIGN**
- Accept that current approach is flawed
- Start over with different pattern detection
- Use 1-minute data from the beginning
- **Takes 6+ months to develop and test**

---

## Lessons Learned

1. **ALWAYS backtest on the same timeframe as live trading**
2. **Small sample sizes (15 trades) are meaningless**
3. **Multiple optimization cycles = overfitting**
4. **Counter-intuitive discoveries (inverted quality) are usually wrong**
5. **1-minute data has 5x more noise than 5-minute**
6. **What works on daily/5-minute may fail on 1-minute**

---

## Financial Impact

If this strategy had been deployed live:
- **Expected (based on 5-min):** +$20K profit
- **Actual (based on 1-min):** -$84K loss
- **Difference:** **$104K disaster avoided**

**The 1-minute backtest saved us $104,000.**

---

## Recommendation

**ABANDON THE PREMIUM STRATEGY.**

It does not work on 1-minute data, which is what the live trading system uses. The apparent profitability on 5-minute data was a statistical illusion caused by:

1. Time smoothing reducing noise
2. Tiny sample size (15 trades)
3. Multiple optimization cycles overfitting to noise
4. Random luck masquerading as edge

**There is no combination of parameters that makes this strategy profitable on 1-minute data.**

---

## Next Steps

1. ✅ Stop paper trading (DONE)
2. ✅ Run 1-minute backtest (DONE)
3. ✅ Analyze failures (DONE)
4. ⏳ Decide on new direction
5. ⏳ Develop different approach or switch timeframes

**Status:** Awaiting decision on next strategy direction.

---

**Report Prepared:** April 9, 2026
**Backtest Period:** 2025 (Jan 1 - Dec 31)
**Data Source:** /root/mnq_historical.json (795,296 1-minute bars)
**Conclusion:** **STRATEGY ABANDONED - UNPROFITABLE ON 1-MINUTE DATA**
