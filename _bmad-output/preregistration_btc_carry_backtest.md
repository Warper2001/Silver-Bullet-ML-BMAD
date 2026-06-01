# Pre-Registration: BTC Funding-Rate Cash-and-Carry Backtest (BTC-CARRY)

**Sealed:** 2026-06-01
**Researcher:** Alex
**Status:** PRE-REGISTRATION (sealed before any backtest data has been examined)

---

## 1. Strategy Description

**Name:** BTC-CARRY — BTC Funding-Rate Cash-and-Carry, Market-Neutral

**Basis:**
- Christin, Routledge, Soska & Zetlin-Jones (CMU, 2022), *"The Crypto Carry Trade"* — in-sample
  Sharpe 7–11 on short-perp / long-spot; structural cash flow from funding mechanism.
- Practical 2024–2025 data: ~14–19% net annual return, MaxDD < 2% (Zipmex/ScienceDirect).
- Current environment (June 2026): funding compressed to ~0.65% APY on Binance/Bybit — this
  backtest will quantify Kraken's historical yield and determine whether the hurdle is met.

**Instrument:** PF_XBTUSD — long BTC spot + short BTC perpetual, notional-neutral
**Data:** `data/kraken/PF_XBTUSD_funding_rate.csv` (8h cadence, Nov 2024 → present)

---

## 2. Frozen Parameters

All parameters set before examining any backtest output.

| Parameter | Value | Rationale |
|---|---|---|
| `hurdle_annual_pct` | **10.0%** | Minimum net annualized yield to enter; 10% covers costs + risk premium |
| `cost_bps_entry` | **15** | Round-trip perp entry cost (Kraken taker ~0.05%; spot leg adds ~0.10%) |
| `cost_bps_exit` | **15** | Round-trip perp exit cost |
| `neg_stop_threshold` | **-0.01%** | Exit if funding < −0.01% per 8h |
| `neg_stop_periods` | **3** | Require 3 consecutive negative-threshold periods before exit |
| `full_period_start` | **2024-11-01** | First available Kraken funding row |
| `no_oos_split` | **True** | No IS/OOS split — too few independent funding regimes for split to be meaningful |

Signal definition:
```
annualized_funding_t = funding_rate_8h * 3 * 365
in_carry_t = 1  if annualized_funding_t > hurdle_annual_pct/100
in_carry_t = 0  if funding_rate_8h < neg_stop_threshold for >= neg_stop_periods consecutive periods
# Once out, re-enter only when annualized > hurdle again
```

P&L per 8h period:
```
carry_pnl_t = in_carry_{t-1} * funding_rate_8h * notional
cost incurred on entry/exit: -cost_bps/10_000 * notional per leg transition
annualized_return = total_carry_pnl / notional * (8760 / n_8h_periods)  # 8760h/yr
```

---

## 3. Decision Rule (full sample)

| Outcome | Condition | Verdict |
|---|---|---|
| **PASS** | `net_annual_return_pct > 10.0` AND `max_drawdown_pct < 0.05` | Viable; build live executor |
| **FAIL** | `net_annual_return_pct < 5.0` OR `max_drawdown_pct > 0.10` | Compressed; do not pursue |
| **AMBIGUOUS** | All other cases | Monitor; revisit when funding normalises |

Drawdown is measured in units of notional (percentage of initial capital at risk), not P&L swings
from directional exposure (since the strategy is market-neutral, P&L comes only from funding accrual).

---

## 4. Sample Size Note

With 1,732 8h periods (~18 months), the carry return is dominated by the current funding regime
rather than statistical sampling. The decision rule is therefore based on the **regime-level
annualized yield** rather than a Sharpe ratio. If the yield is below 5% it is not worth the
operational complexity and counterparty risk of a two-leg position.

---

## 5. Integrity

This document was written on **2026-06-01** and committed to git **before**
`backtest_btc_carry.py` was created or executed.

**Parameters NOT tuned by data.** The 10% hurdle is a standard carry-trade entry threshold
from the academic literature (CMU carry-trade paper; Sharpe.ai practitioner guidance). The
negative-funding stop (−0.01%/8h for 3 periods) is standard risk management for this trade.

**Git commit SHA:** `[populated by git on commit]`
**Referenced data:** `data/kraken/PF_XBTUSD_funding_rate.csv`

---

*Pre-registration follows the methodology established in `CLAUDE.md` and the prior pre-regs
in `_bmad-output/`. Sealed before `backtest_btc_carry.py` is written.*
