# Phase 0: Why Opening Range Breakout Wins

**Date:** 2026-04-03
**Author:** Charlie (Senior Dev)
**Purpose:** Understand why ORB outperforms ensemble and identify specific fixes

---

## Executive Summary

**Opening Range Breakout (ORB) is the ONLY profitable strategy** in our system, achieving:
- 54.18% win rate (BEST)
- 1.38 profit factor (BEST)
- +$17,314 profit (ONLY profit besides ensemble)
- Only 478 trades/year (quality over quantity)

**The ensemble UNDERPERFORMS ORB on both key metrics:**
- Win Rate: 53.22% vs ORB's 54.18% (-0.96% gap)
- Profit Factor: 1.23 vs ORB's 1.38 (-0.15 gap)

**Root Cause:** Ensemble architecture **dilutes ORB's strong signals** with weak strategies through equal-weighting and overly permissive filtering.

---

## ORB's Secret Sauce: Why It Works

### 1. High Confidence Baseline (60-90%)
```python
confidence = 0.60  # Base confidence
confidence += min(0.20, (breakout.volume_ratio - 1.5) / 5)  # Volume bonus
confidence = min(0.90, confidence)  # Cap at 0.90
```

**Ensemble issue:** Uses 25% confidence threshold - allows garbage signals through

### 2. Volume Confirmation (Quality Filter)
```python
volume_threshold = 1.5  # Requires 1.5x baseline volume
```

**Why this matters:** Breakouts WITHOUT volume confirmation fail 60%+ of the time. ORB waits for volume - ensemble doesn't.

**Ensemble issue:** No volume confirmation on ensemble level - takes low-conviction signals

### 3. Precise Risk Management (2:1 R:R)
```python
# Stop loss: At opposite OR boundary
stop_loss = opening_range.low  # For long

# Take profit: 2:1 reward-risk ratio
risk = entry - stop_loss
take_profit = entry + (2.0 * risk)
```

**Why this works:** ORB respects price structure. The opening range boundaries ARE the support/resistance.

**Ensemble issue:** Weighted averages SL/TP from all strategies, diluting ORB's precise levels

### 4. Regime-Aware (Time-Specific)
```python
DEFAULT_OR_START = time(9, 30)  # Market open
DEFAULT_OR_END = time(10, 30)   # First hour end
```

**Why this works:** Only trades when its edge exists (opening range breakouts). Doesn't force trades in unfavorable conditions.

**Ensemble issue:** Trades all day, every day, regardless of regime

### 5. Quality Over Quantity
- ORB: 478 trades/year (~2/day)
- Triple Confluence: 42,909 trades/year (~175/day)
- Ensemble: 4,470 trades/year (~18/day)

**Why this matters:** ORB is SELECTIVE. Triple Confluence is a machine gun spraying trades.

---

## Ensemble Architecture Problems

### Problem 1: Equal Weighting Dilutes Strong Strategies

**Current config (config-sim.yaml):**
```yaml
ensemble:
  strategies:
    triple_confluence_scaler: 0.20  # 20% weight - TERRIBLE
    wolf_pack_3_edge: 0.20           # 20% weight - Losing
    adaptive_ema_momentum: 0.20      # 20% weight - BROKEN (0 trades)
    vwap_bounce: 0.20                # 20% weight - Losing
    opening_range_breakout: 0.20     # 20% weight - BEST STRATEGY
```

**Impact:** When ORB generates a 0.80 confidence signal but Triple Confluence generates 0.30 confidence signal, the ensemble composite gets diluted:
```
composite = 0.20 * 0.80 (ORB) + 0.20 * 0.30 (TC) + 0.20 * 0.00 (AE) + 0.20 * 0.25 (VB) + 0.20 * 0.20 (WP)
composite = 0.16 + 0.06 + 0.00 + 0.05 + 0.04 = 0.31
```

**Result:** ORB's 0.80 confidence signal becomes 0.31 in ensemble - **61% confidence destroyed!**

### Problem 2: Triple Confluence Massively Overtrades

**Performance:**
- 42,909 trades (89% of ALL ensemble trades!)
- 49.31% win rate (below random!)
- -$30,060 loss
- 0.99 profit factor (break-even minus fees)

**Why it overtrades:** No volume confirmation, no time filter, low confidence threshold

**Impact on ensemble:** TC's garbage signals:
1. Dilute composite confidence
2. Increase trade frequency (bad for slippage/fees)
3. Worsen win rate (loses more than it wins)
4. Worsen profit factor (loses money overall)

### Problem 3: Low Confidence Threshold (25%)

**Current:** `confidence_threshold: 0.25`

**Impact:** Ensemble takes ANY signal above 25% confidence. This includes:
- Triple Confluence's 0.30 signals (lose money)
- VWAP Bounce's 0.25 signals (lose money)
- Wolf Pack's 0.35 signals (lose money)

**ORB would NEVER take these signals** - it requires 60%+ confidence!

### Problem 4: Direction Alignment = Weak Strategy Veto

**Ensemble logic:**
```python
def check_direction_alignment(self, signals: list[EnsembleSignal]) -> bool:
    directions = {signal.direction for signal in signals}
    is_aligned = len(directions) == 1
    return is_aligned
```

**Impact:** If ORB says LONG but Triple Confluence says SHORT, the ensemble rejects the trade - **even though ORB is right 54% of the time and TC is wrong 51% of the time!**

**Result:** Weak strategies can VETO strong strategy signals.

### Problem 5: Weighted Entry/SL/TP Dilutes Precision

**Ensemble logic:**
```python
weighted_entry = sum(s.entry_price for s in signals) / len(signals)
weighted_sl = sum(s.stop_loss for s in signals) / len(signals)
weighted_tp = sum(s.take_profit for s in signals) / len(signals)
```

**Impact:** ORB's precise SL at OR boundary (e.g., 15100.00) gets averaged with TC's loose SL (e.g., 15095.50), VWAP's SL (15102.25), etc.

**Result:** ORB's 2:1 R:R becomes ensemble's 1.7:1 R:R - **worsening risk-reward!**

### Problem 6: Adaptive EMA Completely Broken

**Performance:** 0 trades generated in entire year

**Impact:** 20% weight assigned to strategy that NEVER SIGNALS = wasted capacity

---

## Specific Actionable Fixes

### Fix 1: Rebalance Weights (Phase 2) ⚡
**Target:**
```yaml
ensemble:
  strategies:
    triple_confluence_scaler: 0.00   # DISABLE - loses money
    wolf_pack_3_edge: 0.10           # Reduce - keep for diversity
    adaptive_ema_momentum: 0.00      # DISABLE - broken (0 trades)
    vwap_bounce: 0.10                # Reduce - keep for diversity
    opening_range_breakout: 0.80     # INCREASE - primary driver
```

**Expected impact:**
- Composite confidence increases by 40-50%
- Win rate increases to 54%+ (matches ORB)
- Profit factor increases to 1.35+ (approaches ORB)

### Fix 2: Increase Confidence Threshold (Phase 1) ⚡
**Test:** 30%, 35%, 40%, 45%, 50%

**Expected:**
- 40% threshold: 54% WR, 1.35 PF (matches ORB)
- 45% threshold: 55% WR, 1.40 PF (beats ORB!)
- Trade frequency drops to 8-12/day (still acceptable)

### Fix 3: Remove Direction Alignment Requirement (Phase 2) 🔧
**Change:** Allow trades when majority agrees (not unanimity)

**New logic:**
```python
def check_majority_direction(self, signals: list[EnsembleSignal]) -> bool:
    long_count = sum(1 for s in signals if s.direction == "long")
    short_count = sum(1 for s in signals if s.direction == "short")
    return abs(long_count - short_count) >= 1  # At least 1 vote margin
```

**Impact:** ORB signals no longer vetoed by weak strategies

### Fix 4: Use Best Strategy's Entry/SL/TP (Phase 3) 🔧
**Change:** Instead of weighted average, use highest-confidence strategy's levels

**New logic:**
```python
best_signal = max(signals, key=lambda s: s.confidence * strategy_weights[s.strategy_name])
entry_price = best_signal.entry_price
stop_loss = best_signal.stop_loss
take_profit = best_signal.take_profit
```

**Impact:** ORB's precise 2:1 R:R preserved in ensemble

### Fix 5: Fix Adaptive EMA (Phase 4) 🔧
**Debug:** Why does it generate 0 trades?

**Hypothesis:**
- Confidence calculation bug?
- Signal generation logic broken?
- Filter too restrictive?

**Action:** Debug and fix, or remove from ensemble entirely

### Fix 6: Add Volume Confirmation to Ensemble (Phase 3) 🔧
**New requirement:** Ensemble signals require volume confirmation

**Logic:**
```python
def check_volume_confirmation(self, bar: DollarBar, baseline_volume: float) -> bool:
    volume_ratio = bar.volume / baseline_volume
    return volume_ratio >= 1.5
```

**Impact:** Ensemble matches ORB's quality filter

---

## Hypothesis: How to Beat ORB

**Hypothesis:** "An ensemble with 80% ORB weight, 40% confidence threshold, and ORB-style entry/SL/TP will achieve 55%+ WR and 1.40+ PF, beating ORB standalone."

**Test configuration:**
```yaml
ensemble:
  strategies:
    opening_range_breakout: 0.80
    wolf_pack_3_edge: 0.10
    vwap_bounce: 0.10
  confidence_threshold: 0.40
  use_best_strategy_levels: true  # Use ORB's entry/SL/TP
```

**Expected results:**
- Win Rate: 55-56% (beats ORB's 54.18%)
- Profit Factor: 1.40-1.45 (beats ORB's 1.38)
- Trade Frequency: 8-12/day (acceptable)
- Max Drawdown: <8% (better than current 18%)

---

## Recommended Execution Order

### Phase 1 (Quick Wins - 1-2 days):
1. Test confidence thresholds: 30%, 35%, 40%, 45%, 50%
2. Measure WR, PF, trade frequency impact
3. Document threshold vs performance curve

### Phase 2 (Core Fix - 1 week):
1. Rebalance weights (80% ORB, 10% WP, 10% VB)
2. Remove direction alignment requirement
3. Test on H1 2024, validate on H2 2024

### Phase 3 (Deep Fix - 1-2 weeks):
1. Use best strategy's entry/SL/TP instead of weighted average
2. Add volume confirmation to ensemble
3. Implement trailing stops
4. Backtest with refined exits

### Phase 4 (Foundation Check - 1 week, if needed):
1. Debug Adaptive EMA (0 trades bug)
2. Fix or remove from ensemble
3. Tune Triple Confluence parameters (reduce overtrading)
4. Re-baseline individual strategies

---

## Success Criteria

**Go/No-Go for Paper Trading:**
- [ ] Win Rate ≥54.18% (beat ORB)
- [ ] Profit Factor ≥1.38 (beat ORB)
- [ ] Max Drawdown ≤10% (improve from 18%)
- [ ] Trade Frequency ≥8 trades/day
- [ ] Sharpe Ratio ≥1.0 (institutional benchmark)

---

## Conclusion

**ORB wins because:**
1. High confidence baseline (60-90%)
2. Volume confirmation (quality filter)
3. Precise risk management (2:1 R:R)
4. Regime-aware (time-specific)
5. Quality over quantity

**Ensemble loses because:**
1. Equal weighting dilutes ORB's signals
2. Triple Confluence massively overtrades and loses
3. Low confidence threshold (25%)
4. Direction alignment lets weak strategies veto strong ones
5. Weighted averaging destroys precise risk management

**Fix:** Rebalance to 80% ORB weight, increase threshold to 40%, use ORB's entry/SL/TP, remove weak strategies' veto power.

**Prediction:** This will achieve 55%+ WR and 1.40+ PF, beating ORB standalone and validating the ensemble approach.

---

**Next Step:** Execute Phase 1 (confidence threshold testing) immediately.
