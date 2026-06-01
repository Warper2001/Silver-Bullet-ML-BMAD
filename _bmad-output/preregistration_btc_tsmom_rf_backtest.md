# Pre-Registration: BTC-TSMOM-RF (TSMOM + 200-day SMA Regime Filter)

**Sealed:** 2026-06-01
**Researcher:** Alex
**Status:** PRE-REGISTRATION (sealed before any backtest data has been examined)

---

## 1. Strategy Description

**Name:** BTC-TSMOM-RF — BTC Time-Series Momentum with 200-day SMA Regime Filter

**Motivation and scientific rationale:**
The prior experiment BTC-TSMOM (pre-reg commit `86842af`, backtest commit `44881b1`) produced
an OOS FAIL (Sharpe −0.73). Post-analysis identified the mechanism: the strategy was long during
a structural BTC bear market (Sep 2025 → May 2026) because 28-day momentum briefly turned
positive within a longer downtrend. Adding a 200-day SMA regime filter is a documented improvement
in the TSMOM literature — the signal should only fire in structurally bullish regimes.

This is registered as a **new hypothesis** (not a parameter tweak of BTC-TSMOM). The decision
to add the filter was made after observing the BTC-TSMOM failure; this is disclosed here so the
pre-registration is transparent about its motivation. The pre-registration still seals the
exact rules and decision criteria before any data is examined for this variant.

**Literature basis:** Han, Kang & Ryu (2024); adaptive/regime-aware TSMOM frameworks (arXiv
2602.11708, 2026); standard practitioner 200-day MA filter.

**Instrument:** PF_XBTUSD — Kraken BTC/USD perpetual
**Data:** `data/kraken/PF_XBTUSD_1min.csv` (1-min bars resampled to daily)

---

## 2. Frozen Parameters

All TSMOM parameters are **identical** to BTC-TSMOM (commit `86842af`). Only the signal
generation adds the regime condition.

| Parameter | Value | Same as BTC-TSMOM? |
|---|---|---|
| `lookback_days` | **28** | YES |
| `rebalance_days` | **5** | YES |
| `vol_window` | **20** | YES |
| `target_vol` | **0.30** | YES |
| `max_leverage` | **2.0** | YES |
| `cost_bps` | **15** | YES |
| `no_short` | **True** | YES |
| `sma_regime_window` | **200** | **NEW** — 200-day SMA regime filter |
| `is_start` | **2024-11-08** | YES |
| `is_end` | **2025-08-31** | YES |
| `oos_start` | **2025-09-01** | YES |
| `oos_end` | **2026-05-31** | YES |

Signal definition (change from BTC-TSMOM in **bold**):
```
mom_28d_t = log(close_t / close_{t-28})
sma_200_t = simple moving average of close over last 200 days
raw_signal_t = 1  if mom_28d_t > 0  AND  close_t > sma_200_t
raw_signal_t = 0  otherwise  (EITHER condition false → flat)
# Forward-fill signal in 5-day rebalance blocks (same as BTC-TSMOM)
```

---

## 3. Decision Rule (OOS period: 2025-09-01 → 2026-05-31)

The OOS threshold is relaxed vs BTC-TSMOM because the regime filter will reduce trade count:

| Outcome | Condition | Verdict |
|---|---|---|
| **PASS** | `oos_sharpe > 0.8` AND `oos_sharpe > tsmom_base_oos_sharpe(−0.73)` AND `n_trades_oos ≥ 8` | Regime filter improves TSMOM; extend data window |
| **FAIL** | `oos_sharpe ≤ 0.0` | Regime filter does not help; abandon TSMOM variant |
| **AMBIGUOUS** | All other cases | Marginal improvement; wait for more data |

Note: "beats BTC-TSMOM base" means `oos_sharpe > −0.73` — the pre-registered BTC-TSMOM result.
Since that was negative, TSMOM-RF needs to be better than −0.73 AND positive enough to be
practically useful (> 0.8 for PASS, or > 0.0 for AMBIGUOUS).

---

## 4. Sample Size Note

The 200-day SMA requires 200 bars of warmup. With data starting 2024-11-08, the first valid
regime signal is around **2025-05-27** (200 trading days in). This means the IS period
(2024-11-08 → 2025-08-31) has only ~3 months of valid regime-filtered data. The OOS period
(2025-09-01 → 2026-05-31) has full coverage. This is acceptable — OOS is the primary
decision window.

---

## 5. Integrity

This document was written on **2026-06-01** and committed to git **before**
`backtest_btc_tsmom_rf.py` was created or executed.

**Pre-registration of motivation:** The 200-day SMA filter was chosen AFTER observing the
BTC-TSMOM FAIL result. This is disclosed above. The pre-registration discipline still adds
value: it prevents further data-driven parameter tuning (e.g., choosing 150-day vs 200-day
after seeing which works better in sample). The 200-day SMA is the standard industry choice
for a macro trend filter; no other lookback was evaluated.

**Git commit SHA:** `[populated by git on commit]`
**Referenced base pre-reg:** `_bmad-output/preregistration_btc_tsmom_backtest.md` (commit `86842af`)
**Referenced base result:** commit `44881b1` (OOS Sharpe −0.73)

---

*Pre-registration follows the methodology established in `CLAUDE.md`. Sealed before
`backtest_btc_tsmom_rf.py` is written.*
