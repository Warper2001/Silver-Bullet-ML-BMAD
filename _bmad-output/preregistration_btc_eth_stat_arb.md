# Pre-Registration: BTC-ETH Statistical Arbitrage Backtest

**Sealed:** 2026-06-05
**Researcher:** Alex
**Status:** PRE-REGISTRATION (sealed before any backtest data has been examined)

---

## 1. Strategy Description

**Name:** BTC-ETH-STATARB — Statistical Arbitrage on BTC/ETH Perpetual Pair

**Basis:**
- SpringerLink 2024 (*"Profiting Off the High Correlation of Cryptocurrency Pairs"*):
  Sharpe 2.45, 16.34% annualized return on BTC-ETH cointegration strategy.
- MDPI 2019 (*"Statistical Arbitrage in Cryptocurrency Markets"*): established
  structural cointegration between BTC and ETH.
- IJSRA 2026: cointegration strategies achieve 79–100% win rates live.
- Rationale: BTC and ETH are cointegrated in log-price space; temporary divergences
  revert as capital flows between the two dominant crypto assets.

**Instruments:** PF_XBTUSD (Kraken BTC perp) + PF_ETHUSD (Kraken ETH perp)
**Data:** 1-minute OHLCV from Kraken Futures, resampled to 1-hour bars
**Bar type:** 1-hour close prices for spread construction; 1-min for execution timing

---

## 2. Frozen Parameters

All parameters set before any backtest output is examined.

### Spread Construction

| Parameter | Value | Rationale |
|---|---|---|
| `spread_type` | log-ratio | `spread_t = log(btc_close_t) - log(eth_close_t)` |
| `zscore_window` | **60** bars (60h rolling) | ~2.5 day window; captures short swing cycles |
| `entry_z` | **2.0** | Standard 2σ threshold for stat arb entries |
| `exit_z` | **0.25** | Exit near mean-reversion; not zero to avoid slippage |
| `stop_z` | **3.5** | Hard stop if spread widens further (regime break signal) |

### Trade Management

| Parameter | Value | Rationale |
|---|---|---|
| `max_hold_bars` | **120** bars (5 days) | Swing-frequency time stop |
| `position_notional` | $10,000 per leg | Research size (equal notional BTC leg + ETH leg) |
| `transaction_cost_bps` | **10** bps per leg | Kraken taker rate + slippage allowance |
| `min_spread_vol` | Spread rolling std > 0.005 | Block entries in ultra-low-vol periods (stale data) |

### Signal Direction

```
z_t = (spread_t - mean(spread_{t-60:t})) / std(spread_{t-60:t})

if z_t < -entry_z:  # BTC cheap vs ETH
    → LONG BTC perp, SHORT ETH perp (equal $ notional)
if z_t > +entry_z:  # BTC expensive vs ETH
    → SHORT BTC perp, LONG ETH perp (equal $ notional)

Exit when z_t crosses ±exit_z toward 0, OR z_t crosses ±stop_z, OR max_hold_bars reached.
```

### Sample Period

| Parameter | Value |
|---|---|
| `backtest_start` | 2024-11-01 (first date both BTC and ETH Kraken data available) |
| `backtest_end` | latest available (no holdout split — insufficient independent spread regimes) |
| `min_trades_required` | 30 |

---

## 3. Decision Rule

| Outcome | Condition | Verdict |
|---|---|---|
| **PASS** | Sharpe ≥ 1.5 AND MaxDD < 15% AND N ≥ 30 trades | Viable; build live executor |
| **FAIL** | Sharpe < 0.8 OR MaxDD > 25% | No edge; close |
| **AMBIGUOUS** | All other cases | Investigate z-score window or entry threshold |

Sharpe computed on hourly P&L series, annualized (× √8760).

---

## 4. Sample Size Note

With ~13 months of 1-hour data (~9,500 bars), a 60-bar z-score window produces
roughly 100–200 independent spread extremes. This is sufficient for a first-pass
signal validity check at N ≥ 30 trades minimum.

---

## 5. Integrity

This document was written on **2026-06-05** and committed to git **before**
`backtest_btc_eth_stat_arb.py` was created or run, and before
`data/kraken/PF_ETHUSD_1min.csv` was downloaded.

**Parameters NOT tuned by data.** The 2σ entry, 60-bar window, 5-day time stop,
and $10k notional are all drawn from the academic literature and standard stat-arb
practice, not from iterating on backtest results.

**Git commit SHA:** [populated by git on commit]
**Required new data:** `data/kraken/PF_ETHUSD_1min.csv` (to be downloaded)
**Existing data:** `data/kraken/PF_XBTUSD_1min.csv`
