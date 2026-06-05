# Pre-Registration: MVRV On-Chain Position Scaler for Carry Executor

**Sealed:** 2026-06-05
**Researcher:** Alex
**Status:** PRE-REGISTRATION (sealed before any backtest data has been examined)

---

## 1. Strategy Description

**Name:** MVRV-SCALER — On-Chain MVRV Ratio Position Sizing for BTC Carry

**Basis:**
- MVRV (Market Value to Realized Value) has correctly identified BTC macro tops
  and bottoms across all four major cycles (2018, 2020, 2022 bottom, 2021 top).
- Strategies combining on-chain indicators with stop-loss rules outperformed
  buy-and-hold in volatile markets (multiple academic studies, checkonchain.com).
- MVRV > 3.5 has historically preceded 50%+ drawdowns; MVRV < 1.0 has preceded
  major bull runs. Using MVRV as a position scaler on a carry strategy reduces
  drawdown risk during distribution zones without eliminating carry income entirely.
- QuantumResearch On-chain Z-Score composite (SOPR + NUPL + MVRV) used by
  institutions for overbought/oversold detection with adaptive thresholds.

**Application target:** `btc_carry_executor.py` position sizing; position is
currently binary (in/out) — this adds a scalar to the "in" state.

**Data source:** CoinMetrics Community API (`CapMVRVCur` metric, daily cadence).
Fallback if API unavailable: 2-year simple moving average price ratio as MVRV proxy
(`mvrv_proxy_t = btc_close_t / sma(btc_close, 730_days)`).

---

## 2. Frozen Parameters

All parameters set before any backtest output is examined.

### MVRV Regime Thresholds

| Regime | MVRV Range | Position Scale | Label |
|---|---|---|---|
| Deep Value / Accumulation | MVRV < 1.0 | **2.0×** | Strong buy zone |
| Fair Value | 1.0 ≤ MVRV < 2.0 | **1.0×** | Normal carry size |
| Mild Overvaluation | 2.0 ≤ MVRV < 3.0 | **0.5×** | Reduce carry exposure |
| Distribution Zone | MVRV ≥ 3.0 | **0.0×** | No new carry entries; exit existing |

### Application Rule

```
carry_notional_t = base_notional × mvrv_scale(mvrv_t)

where mvrv_t is the most recent daily MVRV with 1-day publication lag applied.

The existing entry/exit carry logic (hurdle_annual_pct=10%, neg_stop_threshold,
neg_stop_periods) is unchanged — MVRV only modulates SIZE, not the entry/exit signal.
```

### MVRV Data Parameters

| Parameter | Value |
|---|---|
| `mvrv_metric` | `CapMVRVCur` from CoinMetrics Community API |
| `fallback_metric` | `price / sma(price, 730)` from `data/kraken/PF_XBTUSD_1min.csv` |
| `data_lag_days` | 1 (daily MVRV published with 1-day lag) |
| `backtest_start` | 2024-11-01 (matches carry executor data start) |

---

## 3. Decision Rule

Applied to the carry backtest (`backtest_btc_carry.py` results as baseline):

| Outcome | Condition | Verdict |
|---|---|---|
| **PASS** | MVRV-scaled Sharpe > baseline Sharpe + 0.10 OR MVRV-scaled MaxDD < baseline MaxDD × 0.85 | Integrate scaler into live carry executor |
| **FAIL** | MVRV-scaled return < baseline return × 0.80 (scaler hurts too much) | Discard; keep binary on/off |
| **AMBIGUOUS** | All other cases | Test alternative thresholds (e.g., 2.5/3.5 vs 2.0/3.0) |

The test must improve EITHER Sharpe (risk-adjusted) OR max drawdown (risk control),
not necessarily both. The carry strategy's primary benefit is drawdown control, so
drawdown improvement is weighted equally to Sharpe improvement.

---

## 4. Sample Size Note

With ~18 months of 8h funding rate data (1,732 periods), MVRV regime changes are
infrequent (typically 2–4 regime transitions per year). The decision rule therefore
focuses on regime-level yield comparison rather than statistical significance of
individual trade improvements.

Current MVRV level (June 2026): approximately 2.0–2.5 (mid-cycle; based on BTC
price ~$93k vs. estimated realized price ~$45–55k). This means the MVRV scaler
would currently apply 0.5× sizing — a meaningful live impact.

---

## 5. Integrity

This document was written on **2026-06-05** and committed to git **before**
`backtest_mvrv_carry_scaler.py` was created or executed, and before any CoinMetrics
MVRV data was downloaded or inspected.

**Parameters NOT tuned by data.** The MVRV thresholds (1.0 / 2.0 / 3.0) are drawn
directly from the academic literature and widely-used practitioner levels (Glassnode
dashboard standard ranges). The 2× accumulation multiplier is capped to avoid
excessive leverage in the carry leg.

**Git commit SHA:** [populated by git on commit]
**Required new data:** CoinMetrics `CapMVRVCur` daily series (to be downloaded)
**Existing data:** `data/kraken/PF_XBTUSD_funding_rate.csv`, `backtest_btc_carry.py` baseline
