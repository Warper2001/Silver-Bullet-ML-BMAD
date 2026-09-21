# MIM-NB ledger vs the "edge compressed since 2025" report (2026-09-21)

Read-only. `compare.py` reads `data/mim_nb/trades.csv` (never writes to `data/mim_nb/`); output `compare.json`.
Report compared: github.com/giovannibrusco/zarattini-2024-momentum-spy — single author, unreviewed. Its own numbers:
full-period Sharpe 1.11, +2.6 bp/trade, 41% win rate, payoff 1.69; 2020-24 "Sharpe 1.4-2.0 every year"; 2025-26 "below zero" /
"recent Sharpe ~0" with **no magnitudes and no trade counts**. Spec differs from MIM-NB (SPY/ES, 2% vol targeting, VWAP trailing
stop vs MNQ x1, band stop + 250-pt cat-stop), so only trade statistics and the sign of the recent period are comparable.

## Verdict: the ledger cannot confirm or refute the report
The 29 live trades are consistent with a Sharpe of about 1 and with a Sharpe of about 0. Nothing here supports raising or lowering the case.

| | report (full period) | MIM-NB ledger, N=29, 2026-06-11 to 09-21 | 95% CI (bootstrap) |
|---|---|---|---|
| mean per trade | +2.6 bp | **+0.31 bp** (median -2.5 bp, SD 75.9 bp) | -26.7 to +27.6 bp |
| win rate | 41% | 44.8% | – |
| payoff (avg win / avg loss) | 1.69 | 1.23 | – |
| net / PF | – | **-$20.00 / 0.996** | – |
| daily Sharpe (70 sessions, 26 traded) | 1.11 (recent ~0) | **-0.02** | **-3.9 to +3.7** |

- **Ledger state reconciles with the 09-20 memo:** N=28, -$753, PF 0.848, plus the 2026-09-21 trade (+$733) gives N=29, -$20, PF 0.996. The next completed trade is #30; the N=30 pre-commitment is already committed.
- **Resolution:** the report's effect is d = 0.034 per trade against the ledger's SD. Detecting it needs about **5,270 trades**. At N=29 the smallest detectable mean is about 35 bp, **13x** the report's edge. Any live figure here is noise-dominated.
- **Where the ledger sits:** it lies entirely in the report's "compressed" window (2026), and its point estimates are flat. That is compatible with "Sharpe ~0", but equally with +2.6 bp.

## Sensitivities (descriptive, not tests)
- **Excluding the 09-15 trade** (the roll-contaminated spurious LONG, -$355): N=28, +$335, PF 1.07, +2.5 bp/trade. That is almost exactly the report's +2.6 bp, by coincidence at this noise level.
- **Excluding the 07-29..08-04 run** (4 winning days, +$2,279.50): N=25, -$2,299.50, PF 0.536, -15.6 bp/trade (CI -41 to +11). The book's result depends on that run, as the 09-20 memo said.
- **Monthly:** Jun -$157 (6 trades), Jul -$534.50 (9), Aug +$887.50 (8), Sep -$216 (6).
- **Long vs short (post-hoc slice, not a finding):** longs N=19, PF 1.28; shorts N=10, PF 0.44, one winner in ten trades. Roughly 5% under a 40% win-rate null, but it is one of several slices, so it is not evidence and not a filter (policy: no restriction chosen on past data without retesting on unseen data).

## Limits
- The report gives no 2025-26 magnitudes, so nothing quantitative can be checked against the ledger, only the sign.
- The ledger is gross of commissions and fees (about 0.2-0.4 bp of notional per trade, negligible against SD 76 bp).
- Some ledger exits are not strategy exits (EXTERNAL_FLATTEN 07-06, EXTERNAL_CLOSE 08-13, CAT_STOP_OFFLINE 06-25). I kept them: they are what the account earned.
- Daily Sharpe is an iid bootstrap over sessions; zero days include any bot downtime (unknown). Hash chain not re-verified in this pass; the file grew by exactly one row, consistent with the memo.
- Different strategy variant and instruments: a decay in SPY/ES noise-area momentum says nothing certain about MIM-NB's V2 spec.

## What would inform the question
Only the report's own out-of-sample years on MNQ-like bars: MIM-NB's sealed backtest re-run on 2025-01 to 2026-05 bars, front-month rebuilt (the 2025/2026 CSVs have roll defects; see AGENTS.md). That is a separate, pre-registered read of data the strategy never saw, not something the live ledger can answer. Not run.
