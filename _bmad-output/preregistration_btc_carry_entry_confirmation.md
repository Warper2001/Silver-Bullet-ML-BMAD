# Pre-Registration: BTC Carry — Entry-Confirmation Window

**Status:** PRE-REGISTRATION — sealed in git BEFORE `backtest_btc_carry_v3_entry.py` is run.
**Date:** 2026-06-13
**Author:** Alex (via BMAD roundtable: Mary, Victor, John, Dr. Quinn)

## Lineage
- Original carry backtest: `preregistration_btc_carry_backtest.md` (commit 35d9e4d) — v1 PASS, 23.6% ann, Sharpe 12.64, MaxDD 1.93%.
- Exit hardening: `preregistration_btc_carry_exit_rules_v2.md` (79612bc) → v3 (12-period below-hurdle). The v3 exit is LIVE in `btc_carry_executor.py`.
- This doc addresses the **entry** side, untouched by all prior preregs.

## Seal
- Harness: `backtest_btc_carry_v3_entry.py`
  SHA-256 `42b19f4742a6aa95fe5fb9f1f8b99ff66660e5e9994544f429a2d463695ad751`
- Funding data: `data/kraken/PF_XBTUSD_funding_rate.csv` (8h cadence, 2024-11-01 → 2026-05-31, 1731 rows)
  SHA-256 `c5890d2f188b9dbbb96edbcb1a36d1f75b324b38dfd0ef6d5d4c9979f8ad3515`
- Git HEAD at seal: `8f5f997bb5ea9399422a80bfec84fdbd4bdd132d`

## Motivation (mechanism, stated before results)

The live carry executor enters carry on a **single** funding reading above the 10%
annualized hurdle, but **exits** only after a hardened 4-day confirmation (v3:
`BELOW_HURDLE_EXIT_PERIODS = 12`, plus a 3-of-5 negative-window rule). This is a
**control-loop asymmetry**: the exit is low-pass filtered against whipsaw, the entry is
a raw sample. Because perpetual funding is spiky and mean-reverting, a single-reading
entry adversely selects transient spikes that revert before the position recoups its
round-trip cost.

**Corrected cost basis (verified in code, 2026-06-13):** round-trip cost = **30 bps**
(`COST_BPS = 15` charged on each of enter and exit), NOT the 105 bps initially misread
from cumulative `total_cost`. Breakeven hold to recoup 30 bps:
`d_BE = 0.003 × 365 / hurdle` → **~11.0 days at 10%**, ~7.3 days at 15%, ~4.0 days at the
27.2% level the live position actually entered at on 2026-06-12.

This correction **downgrades the concern from "bug" to "robustness refinement."** The
v1 single-reading-10% entry already PASSED at 23.6%. The hypothesis is that a modest
entry confirmation trims cost-losing whipsaw trips **without** materially sacrificing
return — the entry-side analogue of the v3 exit hardening.

## Frozen design

The **v3 live exit is held constant** across all arms (3-of-5 negative window OR
12 consecutive below-hurdle periods; exit hurdle stays 10%). Only the **entry** varies.
Single backtest on the sealed 18-month funding series. One variable per arm.

| Arm | Entry rule | Exit | Breakeven d_BE |
|---|---|---|---|
| **H0** (baseline) | single reading: ann funding > **10%** | v3 live | ~11.0 d |
| **H1** (primary) | ann funding > **10%** for **≥3 consecutive 8h periods** (24h confirm) | v3 live | ~11.0 d |
| **H2** (control) | single reading: ann funding > **15%** | v3 live | ~7.3 d |

H1 isolates the *confirmation* lever (same 10% level as H0, only the window differs).
H2 is a *level* control to show — per the corrected math — that the level lever is
largely dominated by confirmation. No co-tuning of level and window.

## Metrics (per arm)
Annualized net return, Sharpe, max drawdown, round-trip count, % time in carry, average
annualized funding while in carry, and the **key differentiator: cost-loss round-trips**
= count of round-trips whose total net P&L (carry collected − 2×cost) is < 0.

## Decision rule (frozen)

1. **Gate (per arm):** PASS if `net_annual_return > 10%` AND `max_drawdown < 5%`;
   FAIL if `net_annual_return < 5%` OR `max_drawdown > 10%`.
2. **Adopt H1 as the new live entry rule IFF BOTH:**
   (a) H1 annualized return ≥ 90% of H0's (no material return sacrifice), AND
   (b) H1 cost-loss round-trips **strictly fewer** than H0's.
3. If H1 fails (a) — confirmation starves genuine regimes — the asymmetry was benign;
   **keep the live H0 single-reading entry unchanged.** A null result is a valid,
   publishable outcome and the default action is no live change.
4. H2 is reported for understanding only; it does not itself authorize a live change.

## Discipline notes
- This is sealed BEFORE the harness runs. The live whipsaw of 2026-06-12 may **motivate**
  the hypothesis but must **never select** the parameters — the confirmation window (3)
  and the H2 level (15%) are fixed here, not after seeing results. Tuning them on the one
  live path would be the documented restrict-to-favorable-subset failure.
- Changing a live trigger requires this prereg PASS first; the executor is **not** edited
  until the decision rule says "adopt."
- The current live carry position is held per the existing v3 rules regardless of this
  backtest; no hand-trading on one day of data.
