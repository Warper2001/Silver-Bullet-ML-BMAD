# TSC-1 Backtest Results

Pre-registration: `_bmad-output/preregistration_term_structure_carry.md`

Run date: 2026-09-03. Data: TradeStation daily settle, 74065 raw bar-rows across 188 contract-months, 5 instruments.

## Deadzone sweep (dev window, pre-declared grid, locked on max Sharpe)

| d (%) | Sharpe | PF | Total P&L | Days |
|---|---|---|---|---|
| 0.0 | -0.706 | 0.872 | $-116,824.69 | 1258 |
| 3.0 | -1.011 | 0.764 | $-144,682.59 | 1258 |
| 5.0 | -1.277 | 0.683 | $-87,073.23 | 1258 |
| 8.0 | +0.169 | 1.351 | $+936.25 | 1258 |
| 12.0 | +0.000 | nan | $+0.00 | 1258 |

**Locked d = 8.0%**

## Gate 0 (dev window 2021-01-01..2025-12-31): FAIL

Sharpe=+0.169, PF=1.351, total P&L=$+936.25, trading days=1258

### Per-instrument (dev window, locked d)

| Instrument | Sharpe | PF | Total P&L | Nonzero-position days |
|---|---|---|---|---|
| MGC | +0.000 | nan | $+0.00 | 0 |
| MHG | +0.477 | 1.617 | $+558.25 | 21 |
| MNQ | +0.000 | nan | $+0.00 | 0 |
| PL | +0.071 | 1.215 | $+378.00 | 6 |
| SIL | +0.000 | nan | $+0.00 | 0 |

## Gate 1: NOT EVALUATED (Gate 0 failed -- holdout not spent per pre-registration)

For transparency only (does not count toward PASS/FAIL): holdout at locked d would have shown Sharpe=+0.000, PF=nan, total P&L=$+0.00.


## Verdict

**FAIL at Gate 0.** The time-series carry-timing mechanism, as specified, does not clear this shop's own cost-adjusted bar on this universe. Per the pre-registration this is terminal for TSC-1 as specified -- no re-sweep, no new deadzone grid, no added instruments under this seal.

## Diagnostic note (observational only -- does not reopen the gate)

Checked after the fact to make sure the FAIL is a real finding and not an implementation bug: annualized roll yield for MGC, MNQ, and SIL sits in a narrow, almost always-negative band for the entire 2021-2025 dev window (means -3.25%, -2.49%, -3.42%; 90th percentiles still negative or barely positive for MGC/SIL). This is economically coherent, not a defect -- stock-index and precious-metal futures carry a persistent, roughly constant cost-of-carry contango (financing-rate-driven for MNQ; storage+financing for MGC/SIL), rather than oscillating between backwardation and contango the way the classic Erb-Harvey commodity evidence (built mostly on agricultural/energy futures) does. At the locked d=8%, MGC/MNQ/SIL never once cleared the deadzone in either direction across 5 years -- they sat flat the entire window (0 nonzero-position days). Every dollar of Gate-0 P&L above came from MHG (21 days) and PL (6 days) alone.

This means the pre-registered rule -- a fixed absolute-percent deadzone on raw front/next roll yield -- is close to untestable on this specific universe: it is either too tight (d<=5%, which just re-creates a near-permanent short position that loses money outright on the ever-present contango, not a timing signal) or too wide (d>=8-12%, which fires on 2-6 instruments' worth of rare deviations and starves the test of trades). This is a genuine finding about the *signal construction*, not a reason to re-sweep under this seal. A future pre-registration for this same mechanism should very likely measure carry *relative to the instrument's own trailing history* (e.g. a rolling z-score of roll yield, closer to KMPV's actual equity-carry construction of dividend-yield-minus-risk-free-rate) rather than an absolute cutoff -- logged here as a lead for a fresh TSC-2 seal, not executed under TSC-1.
