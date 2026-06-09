# Pre-Registration: HCVWAP v2 — 5-min False-Breakout Rejection Candle

**Generated:** 2026-06-09
**Experiment ID:** hcvwap-v2
**Pre-registration commit:** (populate after `git commit`)
**Supersedes:** hcvwap-v1 (commit 4531a3d, failed Gate 0 MNQ PF=0.813, MES PF=0.704)
**Status:** SEALED — study_hcvwap_v2.py does not yet exist at time of this document

---

## 1. Why v1 Failed and What v2 Changes

### v1 diagnosis (from Gate 0 results + web reconciliation)

HCVWAP v1 tested the correct thesis (confirmation stack rescues VWAP fade edge) but used
the wrong implementation:

| Element | v1 | Research describes | Impact |
|---|---|---|---|
| Bar resolution | 1-min | **5-min** | 1-min bars = false positives; brief extensions that continue |
| Entry trigger | Bar close ≥ 2σ (first touch) | **Rejection candle**: price pierces 2σ AND closes back inside | Wrong population — captures continuation bars, not exhaustion |
| Stop size | 6 pts ($12 MNQ) | **10–20 pts on NQ** | 6pts = 14% of ATR(44pts); pure noise, stops out before reversion |
| Target | Fixed 12 pts | **VWAP centerline** (20–50+ pts = 3–6× R/R) | Fixed 12pts needs 47% WR; 5× R/R only needs 17% WR |
| Result | PF=0.813, WR=41.6% | Expected ~55% WR | Gate 0 FAIL |

The exit count in v1 confirms the stop diagnosis: 281 stop-outs vs 200 TP hits on MNQ.
The trade population was dominated by continuation bars that briefly touched 2σ before
extending further — exactly what the fixed 6-pt stop captured as losses.

### What v2 changes

1. **5-min bars as the signal timeframe.** Each bar represents 5 minutes of accumulated order
   flow. A 5-min bar that wicks to 2σ but closes back inside represents a genuine test-and-
   reject of the level, backed by 5 minutes of participant behavior. A 1-min bar at 2σ could
   be a single institutional order passing through.

2. **False-breakout rejection entry.** Signal condition (both required):
   - `bar.high ≥ vwap + SD × σ_vwap`  AND  `bar.close < vwap + SD × σ_vwap` (short)
   - `bar.low  ≤ vwap − SD × σ_vwap`  AND  `bar.close > vwap − SD × σ_vwap` (long)
   This is the "rejection candle" / "shooting star" / "hammer" at the VWAP band that
   professional traders cite. The wick proves the level acted as resistance/support.

3. **Wider stop: 15 pts (primary).** Just outside the typical 5-min bar's intrabar range.
   Avoids the "stopped by noise" failure mode that dominated v1 exits.

4. **Dynamic target: VWAP centerline.** Target is the actual session VWAP price at trade
   entry. This naturally gives 2:1 to 6:1+ R/R depending on how extended price is. The
   breakeven WR drops to ~15–25% instead of ~47%. Minimum R/R filter: skip if VWAP is closer
   than 1.5× the stop (degenerate cases where the trade barely has room).

---

## 2. Hypothesis

### H₁ (alternative)

A VWAP mean-reversion trade filtered by (a) the false-breakout rejection candle pattern on
5-min bars AND (b) the same 4-condition confirmation stack (SD extension + time window +
volume spike + HTF ranging) produces positive expectancy on both MNQ and MES in-sample
(2025-01-01 → 2026-02-28) — measured by:

- **EV > $0** per trade net of commission
- **PF ≥ 1.20**
- **WR ≥ weighted-average breakeven + 5pp**  
  (breakeven = 1/(avg_R/R + 1); at 3:1 R/R = 25%; gate ≈ 30%)
- **N ≥ 20** on the filtered population
- **Median stop ≤ $150/contract** (15 pts × $2 MNQ = $30 ✓; 15 pts × $5 MES = $75 ✓)

### H₀ (null)

Even with the correct architectural implementation, VWAP fades on MNQ/ES fail. The MNQ
momentum regime (confirmed across 11 prior families) prevents mean reversion at any
selectivity level, any bar resolution, or any stop/target geometry. The Sep-2025
WR=68.2% anomaly is a rare regime outlier, not a repeatable edge.

---

## 3. Signal Definition (Frozen)

### 3a. Session VWAP (5-min bars)

- Computed on 5-min RTH bars (resampled from 1-min)
- `typical_price = (high + low + close) / 3`
- `vwap = cumsum(tp × vol) / cumsum(vol)` — reset at 09:30 ET each day
- `σ_vwap` = running intra-session `std(close − vwap)` — reset daily; minimum 3 bars before valid

### 3b. False-Breakout Rejection (the entry condition)

```
SHORT signal:
  bar.high >= vwap + SD_THRESH × σ_vwap   (price PIERCED the upper band)
  AND
  bar.close < vwap + SD_THRESH × σ_vwap   (close RETURNED inside the band)

LONG signal:
  bar.low <= vwap - SD_THRESH × σ_vwap    (price PIERCED the lower band)
  AND
  bar.close > vwap - SD_THRESH × σ_vwap   (close RETURNED inside the band)
```

**Entry:** close of the rejection bar.

### 3c. Additional Confirmation Conditions (same 4 as v1, applied to 5-min bars)

1. **SD extension**: covered by signal definition (condition 1 above)
2. **Time window**: 09:45–11:30 ET OR 14:00–15:00 ET
3. **Volume spike**: 5-min bar volume > 1.5 × 20-bar rolling mean (5-min volumes)
4. **HTF ranging**: 15-min |EMA(9) − EMA(21)| < 0.5 × ATR(14) on 15-min bars

All 4 conditions required simultaneously.

### 3d. Trade Rules

| Element | Rule |
|---|---|
| Stop | Fixed 15 pts from entry (primary); below/above the signal candle extreme |
| Target | Session VWAP price at entry time (dynamic, variable R/R) |
| Min R/R | Skip trade if VWAP distance < 1.5 × 15 = 22.5 pts (degenerate) |
| Hold max | 12 × 5-min bars = 60 min (session flat) |
| Session close | 15:55 ET force-close |
| One trade at a time | No concurrent positions |

---

## 4. Gate 0 Thresholds

| Criterion | Gate | Notes |
|---|---|---|
| EV per trade | > $0 | Primary edge gate |
| Profit factor | ≥ 1.20 | |
| WR vs breakeven | ≥ avg_be_wr + 5pp | avg_be_wr computed from realized per-trade R/R |
| Median stop | ≤ $150/contract | 15 pts × $2 = $30 MNQ; × $5 = $75 MES — both well inside |
| N (filtered) | ≥ 20 | Lowered from 30: fewer signals due to rejection-wick filter |
| Worst-month avg | ≥ −$50/trade | Variance guard |
| Frequency | Informational only | ~0.1–0.3/day expected |

**Combine-math path simulation** (if edge gates pass): P(target $3k in 30 days, 3 contracts)
without hitting −$2k trailing DD. Advance if P(ruin) < 20% AND E[P&L] > $0.

---

## 5. Data Observation Disclosure

**Clean pre-registration.** No backtest of v2 architecture has been run.

Prior observation from v1 (disclosed):
- MNQ long side WR=45.3%, Avg=−$0.50 (marginal, nearly breakeven)
- MNQ short side WR=35.7%, Avg=−$3.96 (clearly losing)
- This asymmetry is NOT used to pre-select direction for v2 — both directions are tested
- The architectural changes (5-min rejection candle + wider stop + VWAP target) were derived
  from web reconciliation of external research, not from optimization on MNQ data

---

## 6. Integrity Seal

| Item | Value |
|---|---|
| hcvwap_v2_config.yaml | (SHA-256 computed by git at commit time) |
| Git HEAD at pre-registration | (populate after `git commit`) |
| study_hcvwap_v2.py | NOT YET WRITTEN at time of this document |

`study_hcvwap_v2.py` is written AFTER this pre-registration commit — tamper-evident.

---

## 7. Scope Constraint

- This is the **final attempt on the VWAP hypothesis** before the search for MNQ-based
  combine strategies is declared exhausted.
- If v2 fails Gate 0, the conclusion is: VWAP mean reversion is not viable for the Topstep
  $50K combine on MNQ/ES at any bar resolution or entry architecture tested.
- S25 (`tier2_streaming_working.py`) continues unchanged on account 23884932.
- Sealed holdout ≥2026-03-01 stays sealed.
- GC CPI prospective test (event 1 June 11) continues independently.
