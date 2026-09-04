# TSMOM-1 Backtest Results

Pre-registration: `_bmad-output/preregistration_tsmom1_time_series_momentum.md`

Data: TSC-1's fetched TradeStation daily settle data, 8914 front-contract daily rows, 5 instruments.

## Lookback sweep (dev window, pre-declared grid {1,3,6,12} months, locked on max Sharpe)

| k (months) | Sharpe | PF | Total P&L | Days |
|---|---|---|---|---|
| 1 | +0.359 | 1.075 | $+54,444.22 | 1259 |
| 3 | +0.331 | 1.069 | $+51,376.30 | 1259 |
| 6 | +0.347 | 1.072 | $+53,491.24 | 1259 |
| 12 | +0.399 | 1.085 | $+62,677.22 | 1259 |

**Locked k = 12 months**

## Gate 0 (dev window 2021-01-01..2025-12-31): FAIL

Sharpe=+0.399, PF=1.085, total P&L=$+62,677.22, trading days=1259

### Per-instrument (dev window, locked k)

| Instrument | Sharpe | PF | Total P&L | Nonzero-position days |
|---|---|---|---|---|
| MGC | +0.791 | 1.164 | $+16,454.00 | 1252 |
| MHG | +0.081 | 1.017 | $+934.50 | 922 |
| MNQ | +0.566 | 1.104 | $+20,219.72 | 1255 |
| PL | -0.071 | 0.986 | $-7,251.00 | 1254 |
| SIL | +0.596 | 1.135 | $+32,320.00 | 1254 |

## Gate 1: NOT EVALUATED (Gate 0 failed -- holdout not spent per pre-registration)

For transparency only: holdout at locked k would have shown Sharpe=-0.111, PF=0.979, total P&L=$-10,508.73.


## Section 5 mechanism-check controls (dev window, non-tradeable, informational only)

| Strategy | Sharpe | PF | Total P&L |
|---|---|---|
| Locked-k TSMOM (k=12mo, primary) | +0.399 | 1.085 | $+62,677.22 |
| Always-long baseline | +0.772 | 1.156 | $+131,849.70 |
| Full-sample-mean-sign baseline (look-ahead, non-tradeable) | +0.772 | 1.156 | $+131,849.70 |

## Verdict

**FAIL at Gate 0.** Time-series momentum, as specified, does not clear this shop's own cost-adjusted bar on this universe. Per the pre-registration this is terminal for TSMOM-1 as specified -- no re-sweep, no vol-scaling added, no new instruments under this seal.

## Diagnostic note (observational only -- does not reopen the gate)

The Section 5 mechanism-check control resolves this run's central open question decisively, and in the more damning direction. The always-long baseline and the full-sample-mean-sign baseline are numerically identical (Sharpe +0.772 both) because every one of the 5 instruments had a positive average daily return over the 2021-2025 dev window -- this was a broad, one-directional bull run across precious metals and MNQ, not a market with genuine two-sided trend episodes to detect.

Against that backdrop, locked-k TSMOM (dev Sharpe +0.399) **underperforms both baselines by roughly half** -- the real, tradeable, time-varying momentum signal did *worse* than either a static always-long position or the look-ahead "cheat" of already knowing each instrument's whole-window average sign. This is not merely "no evidence for genuine predictability" (Huang-Li-Wang-Zhou's finding of indistinguishability from the baseline) -- on this specific universe and era, TSMOM's own sign-flipping actively destroyed value relative to doing nothing sophisticated at all. Per-instrument, only PL (the one instrument with a full-window mean-sign flip somewhere in the window, or a run of costly whipsaws) shows this pattern most starkly (Sharpe -0.071 vs. the other four all positive), consistent with the signal reacting to noise around trend reversals in an otherwise-persistent uptrend rather than detecting anything real.

This closes the entire candidate list from the 2026-09-03 low-frequency deep-recon: term-structure carry (TSC-1) failed Gate 0, this shop's follow-on prop-firm-strategy work (VPOC-1) failed Gate 0, and time-series momentum (TSMOM-1) now fails Gate 0 more decisively than either, with its own built-in control ruling out "just needs a longer sample" as an excuse. A future revisit of this general family (cross-market trend-following) would need either a genuinely different, larger, more diverse universe (the original academic papers used 58-67 markets spanning currencies/rates/equities/commodities, not 5 instruments dominated by precious metals) or a materially different construction (vol-scaled sizing, cross-sectional rather than time-series) -- both out of scope for a re-sweep of this seal.
