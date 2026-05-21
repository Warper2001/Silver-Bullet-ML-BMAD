# Pre-Registration: S25 Live Paper Trading — H1·M15·M1·g0.25
**Registered:** 2026-05-21
**Authored by:** Alex (warper2001@gmail.com)
**Status:** ACTIVE — frozen at commit time.

---

## Purpose

The sealed holdout (2026-03-01 to 2026-05-19) was consumed by S22, which returned
`edge_exceeds_insample` (PF=1.2742, N=15). Phase 2 ML meta-labeling (S23/S24)
found no reliable improvement from XGBoost filtering at N=109 in-sample trades.

S25 is the **prospective live paper trading phase**: deploying H1·M15·M1·g0.25 into
`Tier2StreamingTrader` and measuring actual paper trade P&L against a pre-committed
decision rule.

This is the natural successor to the sealed holdout — the only remaining truly
out-of-sample data is the future.

---

## Architecture (identical to S22 — no changes permitted after this commit)

| Parameter | Value | Source |
|---|---|---|
| Sweep TF | H1 (1-hour) | S22 frozen |
| Confirm TF | M15 CHoCH | S22 frozen |
| Entry TF | M1 FVG | S22 frozen |
| `MIN_GAP_ATR_RATIO` | **0.25** | S22 frozen |
| `SL_MULTIPLIER` | 5.0 | Phase 1 frozen |
| `TP_MULTIPLIER` | 6.0 | Phase 1 frozen |
| `ENTRY_PCT` | 0.5 (FVG midpoint) | Phase 1 frozen |
| `MAX_HOLD_BARS` | 60 M1 bars | Phase 1 frozen |
| `MAX_PENDING_BARS` | 240 M1 bars | Phase 1 frozen |
| `VOL_REGIME_LOOKBACK` | 120 H1 bars | Phase 1 frozen |
| `VOL_REGIME_THRESHOLD` | 0.75 | Phase 1 frozen |
| `MAX_GAP_DOLLARS` | $60.00 | Phase 1 frozen |
| `ATR_THRESHOLD` | 0.5 × M1 ATR | Phase 1 frozen |
| Direction | Bearish only | Phase 1 frozen |
| Tuesday | Blocked | Phase 1 frozen |
| ML filter | **Disabled** | S24 verdict: marginal OOS gain at N=109 |

### M15 CHoCH Definition (new vs current Tier2)
- First M15 bar whose close is below the most recent M15 swing low by ≥ 0.3 × M15 ATR
- Swing detection: 2-bar symmetric radius, must be ≥ 2 bars old
- CHoCH window expires when the H1 sweep expires (6 hours after sweep bar)

---

## Changes to Tier2StreamingTrader

The current deployed system (`src/research/tier2_streaming_working.py`) uses:
- `MIN_GAP_ATR_RATIO = 0.15` → change to **0.25**
- No M15 CHoCH confirm layer → **add M15 CHoCH state machine**

All other parameters remain unchanged.

The M15 CHoCH layer inserts between the existing H1 sweep detection and the M1 FVG scan:
```
H1 sweep active?
  └─ M15 CHoCH fired?
       └─ M1 FVG detected?
            └─ Enter limit order
```

---

## Hypothesis

> H1·M15·M1·g0.25 produces profit factor > 1.1350 (S12 p90 random baseline)
> over a minimum of 20 live paper trades on MNQ.

---

## Evaluation Window

- **Start:** date of first paper trade after Tier2StreamingTrader is updated
- **End:** when N_live ≥ 20 AND 60 calendar days have elapsed (whichever is later)
- **Maximum:** 90 calendar days from deployment start date

The 20-trade minimum is set conservatively (vs S22's N=10) because live data includes
slippage, missed fills, and connectivity gaps not present in historical backtest.

---

## S25 Decision Rule (pre-committed, no exceptions)

Measured after evaluation window closes:

| Condition | Verdict | Action |
|---|---|---|
| N_live < 20 after 90 days | `insufficient_live_sample` | Strategy fires too rarely in live market. Re-evaluate architecture frequency. |
| N_live ≥ 20 AND PF ≤ 1.0 | `live_no_edge` | Historical edge did not transfer to live trading. **STOP paper trading. PIVOT.** |
| N_live ≥ 20 AND PF > 1.0 AND PF ≤ 1.1350 | `live_marginal` | Edge present but below S12 random baseline. Extend evaluation window by 30 days before deciding. |
| N_live ≥ 20 AND PF > 1.1350 AND PF ≤ 1.2742 | `live_edge_confirmed` | Live PF beats S12 random baseline. **Promote to full position size.** |
| N_live ≥ 20 AND PF > 1.2742 | `live_exceeds_holdout` | Live PF exceeds S22 holdout PF. **Promote to full position size with high confidence.** |

---

## What Is Not Pre-Committed

- Adjusting `MIN_GAP_ATR_RATIO` after observing live results
- Removing the M15 CHoCH confirm after observing live results
- Changing evaluation window length after observing results
- Using live results to select a new gap threshold and re-deploying without pre-registration
- Claiming "paper trading proved the edge" after fewer than 20 trades

---

## Monitoring

Live paper trades are logged to `logs/tier2_filter_log.csv`. P&L is computed from
TradeStation SIM execution prices (not theoretical backtest prices).

The live P&L should be compared against the historical P&L distribution:
- S22 holdout: PF=1.2742 (N=15, 2026-03-01 to 2026-05-19)
- S21 in-sample: PF=1.1656 (N=109, 2025 full year)

Any single large loss or win that moves PF dramatically is expected at N<20 —
do not adjust parameters based on individual trade outcomes.

---

## Implementation Reference

Script: `src/research/tier2_streaming_working.py`

Changes required:
1. `MIN_GAP_ATR_RATIO = 0.15` → `MIN_GAP_ATR_RATIO = 0.25`
2. Add M15 resampling to `_update_h1_structure()` or a new `_update_m15_structure()` method
3. Add CHoCH state: `self._m15_choch_active: bool = False`
4. In `_detect_and_enter()`: gate M1 FVG scan on `self._m15_choch_active`

---

## Acknowledgement

By committing this document, the author pre-commits to all decision rules above.
Any deviation must be disclosed in `data/sealed_holdout/ACCESS_LOG.md`.

*This document is intentionally difficult to amend — that is its purpose.*
