# Pre-Registration: BTC Time-Series Momentum Backtest (BTC-TSMOM)

**Sealed:** 2026-05-31
**Researcher:** Alex
**Status:** PRE-REGISTRATION (sealed before any backtest data has been examined)

---

## 1. Strategy Description

**Name:** BTC-TSMOM — BTC Time-Series Momentum, Long/Flat, Volatility-Targeted

**Basis:** Han, Kang & Ryu (2024), *"Time-Series and Cross-Sectional Momentum in the
Cryptocurrency Market: A Comprehensive Analysis under Realistic Assumptions."* Key finding:
TSMOM with 28-day lookback / 5-day hold survives 15 bps/trade transaction costs; cross-sectional
momentum does not. Short positions inflict significant losses due to large jumps; long/flat
outperforms long/short.

**Instrument:** PF_XBTUSD — Kraken BTC/USD perpetual futures

**Data source:** `data/kraken/PF_XBTUSD_1min.csv` (1-minute OHLCV bars, UTC timestamps)

---

## 2. Frozen Parameters

All parameters below are set from the literature **before** any data exploration. No parameter
was chosen by looking at backtest results.

| Parameter | Value | Source |
|---|---|---|
| `lookback_days` | **28** | Han et al. 2024 optimal (cost-adjusted) |
| `rebalance_days` | **5** | Han et al. 2024 optimal holding period |
| `vol_window` | **20** | Standard realized-vol estimate window |
| `target_vol` | **0.30** | 30% annualized — moderate risk budget |
| `max_leverage` | **2.0** | Hard cap on vol-scaling multiplier |
| `cost_bps` | **15** | Round-trip baseline (Han et al. assumption, conservative) |
| `no_short` | **True** | Long/flat ONLY — shorting forbidden per literature |
| `is_start` | **2024-11-08** | First bar of available Kraken data |
| `is_end` | **2025-08-31** | Last bar of in-sample period |
| `oos_start` | **2025-09-01** | First OOS bar (holdout — do not examine before decision rule) |
| `oos_end` | **2026-05-31** | Last available bar |

Signal definition:

```
daily_log_ret_28 = log(close_t / close_{t-28})
signal_t = 1 if daily_log_ret_28 > 0 else 0   # 1=long, 0=flat
rebalance: signal evaluated once every 5 days; held constant between evaluations
```

Position sizing:

```
rvol_t  = rolling 20-day stdev(daily_log_ret) * sqrt(365)
size_t  = clip(target_vol / rvol_t, 0, max_leverage) * signal_t
strat_ret_t = size_{t-1} * daily_log_ret_t - |Δsignal_t| * cost_bps
```

---

## 3. Robustness Sweep (post-primary, full sample)

The following sweep will be run **after** the primary OOS decision:

- **Lookbacks:** 20, 28, 40, 60 days
- **Cost scenarios:** 5 bps (perp maker limit fill), 15 bps (baseline), 26 bps (spot taker)
- **Target vols:** 0.20, 0.30, 0.40
- Total: 4 × 3 × 3 = **36 combinations**

A strategy that passes the primary decision rule but fails the sweep (e.g., result is sensitive
to a single lookback value) should be treated as AMBIGUOUS.

---

## 4. Decision Rule (OOS only — do not evaluate until OOS has been observed)

The OOS period is **2025-09-01 to 2026-05-31** (~9 months). Do NOT look at OOS results until
the pre-registration commit is in git history.

### Primary decision (OOS Sharpe, OOS Profit Factor, HODL comparison)

| Outcome | Condition | Verdict |
|---|---|---|
| **PASS** | `oos_sharpe > 1.0` AND `oos_sharpe > hodl_sharpe_oos` AND `n_trades_oos ≥ 10` | Promote to live research (Phase 2: parameter sweep + funding-carry sleeve) |
| **FAIL** | `oos_sharpe ≤ 0.5` OR `max_drawdown_oos_pct > 0.40` | Edge not confirmed; do not proceed |
| **AMBIGUOUS** | All other cases | Collect more data or widen OOS window |

### HODL baseline definition
Long BTC every day, no sizing, no costs. The HODL Sharpe is `calc_sharpe(hodl_daily_log_returns)`.

---

## 5. Stopping Rule and Sample-Size Note

With 18 months of data (2024-11-08 to 2026-05-31) and 28-day lookback, ~15 months are
tradeable (~450 daily bars). A TSMOM strategy with a 5-day rebalance typically produces
**O(10–30) discrete long trades** over the OOS period, depending on market regime. N ≥ 10
trades is the minimum for the primary decision rule; the Sharpe is computed on daily returns
(not per-trade), which provides more statistical power.

This pre-registration does **not** lock in a minimum N for an "inconclusive" override — the
verdict table above is the authoritative decision mechanism.

---

## 6. Integrity

This document was written on **2026-05-31** and committed to git **before**
`backtest_btc_tsmom.py` was created or executed.

The SHA-256 hash of this file's content and the git commit SHA are the tamper-evidence record.

**File hash (computed at commit time):** `[git-managed]`
**Git commit SHA:** `[populated by git on commit]`
**Referenced code files:**
- `src/research/strategy_core.py` — helpers `calc_sharpe`, `calc_profit_factor`, `calc_max_drawdown_pct`
- `data/kraken/PF_XBTUSD_1min.csv` — source data

**Parameters NOT tuned by data:** all parameters in Section 2 were set from published literature
(Han et al. 2024) or from standard industry practice (20-day vol window, 30% target vol).
No parameter was selected after observing backtest output.

---

*This document follows the pre-registration methodology established in `CLAUDE.md` §"Weekly
Config Change Workflow (Epic 8)" and the `prereg_seal.py` pattern. It serves the same
integrity function as prior pre-registrations in `_bmad-output/`.*
