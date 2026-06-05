# Pre-Registration: M2/DXY Macro Gate Overlay

**Sealed:** 2026-06-05
**Researcher:** Alex
**Status:** PRE-REGISTRATION (sealed before any backtest data has been examined)

---

## 1. Strategy Description

**Name:** MACRO-GATE — M2/DXY Binary Macro Filter for BTC Swing Strategies

**Basis:**
- BTC vs. Global M2 liquidity correlation: 0.94 (MacroMicro/Newhedge 2024–2025);
  M2 expansion explains up to 90% of BTC price direction over weekly timeframes.
- BTC vs. DXY: −0.72 correlation in 2024 (Altrady research); dollar weakness
  is the strongest free macro predictor of BTC upside.
- Post-2024 ETF approval, BTC S&P500 correlation = 0.77; macro cycle now dominates
  directional BTC bias at swing frequency.
- Hypothesis: filtering S26 and carry trades through a macro bullish gate will
  improve profit factor by eliminating trades taken against the macro tide.

**Application targets:**
1. S26 Kraken crypto swing trades (from `logs/s26_crypto_filter_log.csv`)
2. BTC Carry executor re-entry decisions (from `data/kraken/PF_XBTUSD_funding_rate.csv`)

**Data sources:**
- DXY proxy: FRED Trade Weighted USD Index Broad (`DTWEXBGS`), weekly
- US M2: FRED M2 Money Stock (`M2SL`), monthly, weekly-interpolated

---

## 2. Frozen Parameters

All parameters set before any backtest output is examined.

### Gate Definition

| Parameter | Value | Rationale |
|---|---|---|
| `dxy_lookback_weeks` | **4** | 4-week rate of change for DXY direction |
| `m2_lookback_weeks` | **8** | 2-month rate of change for M2 direction |
| `dxy_threshold` | **0.0** | DXY 4w ROC < 0 = dollar weakening = bullish BTC |
| `m2_threshold` | **0.0** | M2 8w ROC > 0 = liquidity expanding = bullish BTC |
| `data_lag_days` | **7** | Apply data with 7-day lag (publication delay for FRED series) |

Gate signal:
```
dxy_4w_roc_t = (dxy_t - dxy_{t-4w}) / dxy_{t-4w}
m2_8w_roc_t  = (m2_t  - m2_{t-8w})  / m2_{t-8w}

macro_bull_t = (dxy_4w_roc_t < dxy_threshold) AND (m2_8w_roc_t > m2_threshold)
macro_bear_t = NOT macro_bull_t
```

### Application Rule

For S26 trade log: mark each trade entry date as macro_bull or macro_bear.
Compare profit factor of macro_bull trades vs. all trades.

For carry executor: only enter carry position if macro_bull = True.

---

## 3. Decision Rule

| Outcome | Condition | Verdict |
|---|---|---|
| **PASS** | PF(macro_bull_trades) ≥ PF(all_trades) + 0.10 AND macro_bull_trades ≥ 40% of all trades | Integrate gate into live strategies |
| **FAIL** | PF(macro_bull_trades) < PF(all_trades) − 0.05 OR macro_bull_trades < 20% of all trades | Gate adds no value; discard |
| **AMBIGUOUS** | All other cases | Try single-factor gates (DXY-only or M2-only) |

The 40% minimum trade retention threshold prevents a vacuous result where the gate
keeps only 2 trades that happened to win.

---

## 4. Sample Size Note

S26 Kraken trade log covers ~6 months (Dec 2025 → June 2026). Expected N ≈ 50–100
trades. This is borderline for a definitive verdict but sufficient for a diagnostic
signal-validity check.

---

## 5. Integrity

This document was written on **2026-06-05** and committed to git **before**
`backtest_m2_dxy_gate.py` was created or executed, and before any FRED data was
downloaded or inspected.

**Parameters NOT tuned by data.** The 4-week DXY lookback and 8-week M2 lookback
are drawn from practitioner macro models (Global M2 macro cycle literature) and the
7-day lag matches standard FRED data publication delays. No BTC trade data was
examined before setting these parameters.

**Git commit SHA:** [populated by git on commit]
**Required new data:** FRED `DTWEXBGS` + `M2SL` (to be downloaded via public FRED CSV API)
**Existing data:** `logs/s26_crypto_filter_log.csv`, `data/kraken/PF_XBTUSD_funding_rate.csv`
