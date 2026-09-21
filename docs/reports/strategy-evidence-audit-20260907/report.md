# Strategy evidence audit — 2026-09-07

**Decision: prioritize YANK run provenance and a faithful research harness. Status: HOLD_VALIDATION.**

**Follow-up:** [Historical dynamic sizing now reconciles all 189 distinct saved trades exactly](sizing-findings.md). The initial fixed-size discrepancy below is retained as the diagnostic that prompted the investigation; it is no longer an unexplained arithmetic issue. None of the reviewed evidence establishes a strategy as validated profitable. YANK has traceable positive saved trade exports and an immediately actionable reconciliation problem. This selects an engineering investigation, not a trading recommendation or permission to retune a failed study.

Scope: the named strategy families and saved runs below, not an exhaustive ranking of every experiment. This audit reads existing reports, three exported trade lists, and historical source code. It does not run strategies, acquire data, open raw sealed holdouts, inspect prospective strategy returns, or change live configuration. Commodity research remains parked at HOLD-DATA.

## Evidence comparison

| Candidate | Existing evidence | Disposition |
|---|---|---|
| YANK MNQ ML 0.50 | Saved Jan–May 2026 export: 54 trades, $6,917.50, PF 1.6015. Mixed implied sizing and previously explored 2026 window undermine interpretation. | First priority: accounting/provenance repair, then a separately specified validation. |
| BTC carry | v3 H1 reports approximately 24% annual return, but simulator models funding less transition costs; preregistration explicitly has no OOS split. | Second priority: independently account for both hedge legs, basis, collateral and costs before treating headline returns as usable. |
| Older Tier-1/ML claims | Audit admits training/test contamination and simulated probabilities. Corrected “production-ready” claims use estimated Sharpe from win rate. | Exclude unsupported profitability claims. |
| Copper HG | Sealed test N=26, net PF 0.463, −$399 at $4 roundtrip. | Closed under its preregistration. |
| Platinum PL | Tested account fit: worst loss $1,914, drawdown $4,890 against $2,000 trailing limit. | Closed for the tested account/configuration; leave untouched holdout alone. |
| BTC TSMOM | Saved risk-filter test fails its rule; baseline annual return −21.8%, filtered −2.5%. | Do not rescue through further tuning on the same test. |
| MIM-NB | Post-repair proposal separates operational parity from slow edge confirmation; its power calculation estimates about 867 trades. | Useful parity work, not a quick profitability verdict. Current activation not verified. |
| Thursday short | Prospective protocol with precommitted sample gate. | Preserve evaluation; no interim return inspection in this audit. |
| Tick simulator | Two execution-parity cycles failed; R3 explicitly closed. | Retain engineering assets, not evidence of strategy profitability. |

## YANK ledger reconciliation

Reproduce with `.venv/bin/python docs/reports/strategy-evidence-audit-20260907/reconcile.py` from the repository root. Detailed per-trade discrepancies and input hashes are in [ledger-audit.json](ledger-audit.json). The script uses only its three explicit saved CSV inputs and standard-library arithmetic. Passing its checks means the audit reproduced saved totals; it does not validate the strategy.

The June 15 exports associated with ML and no-ML reproduce the published rounded totals. Association comes from matching totals and the results note, not an embedded configuration manifest.

| Saved export | Full-period N | Full-period net | 2026 N | 2026 net | 2026 PF |
|---|---:|---:|---:|---:|---:|
| 181838, associated ML | 82 | $7,804.00 | 54 | $6,917.50 | 1.601469 |
| 185354, associated no-ML | 107 | $1,748.00 | 70 | $3,064.50 | 1.322596 |

**Initial fixed-size diagnostic (resolved by the follow-up).** The documented setting is five contracts, point value $2, and $4 roundtrip commission. Under `signed price change × 2 × 5 − 4`, 23 of 82 ML rows and 64 of 107 baseline rows do not reconcile. Every discrepant row instead algebraically matches one contract with the same $4 fee. These are implied quantities, not independently verified fills. The CSV omits quantity, multiplier, fees and configuration identifiers, so the discrepancy could reflect sizing behavior, mixed provenance, or a reporting defect. It must not be silently repaired by rescaling P&L. Within 2026, the ML discrepancy includes the January 2 trade: reported $352 versus $1,776 at documented size.

The referenced historical commit `138cab1b31d064555ede4c9c07503399a743893f` exists. The first audit inspected the similarly named `yank_streaming_working.py`; the runner actually imports `tier2_streaming_working.py`. Its dynamic per-trade sizing reconciles the exported P&L, as documented in the follow-up. No strategy replay was attempted, and original run identity remains incompletely recorded.

**The “removed 16 losing trades” interpretation is unsupported.** Matching 2026 entries by timestamp and direction yields 39 shared entries, 15 ML-only entries and 31 baseline-only entries. Of the 39 shared entries, 19 have different exported records. The net count difference is 16, but the trade sets are not a subset. Filtering can change subsequent opportunities and account state; the saved lists alone do not establish which mechanism caused each difference.

**Concentration is high.** January contributes $6,387 of the $6,917.50 reported 2026 net, approximately 92%. February through May sum to $530.50. These are descriptive diagnostics, not instructions to select months or retune.

**Model OOS is not an untouched strategy test.** The preregistration itself says an SL×TP×ML grid had already encountered 2026 holdout results. Its listed holdout start and end are both March 1, while the verdict evaluates January–May. Training the model only on 2025 does not remove strategy-selection reuse of 2026.

**Costs and generalization remain unproven.** The historical closing calculation subtracts commission; the exports do not demonstrate realistic slippage or fill availability. The reported Sharpe annualizes per-trade returns with √252, which is not a daily-calendar Sharpe for 82 annual trades. A later 2021–2024 floor study reports baseline PF 1.064, mean $9.62 and t=0.514, while the proposed loosening fails. That study is a separate configuration comparison, not a clean replication of the June ML arm. MES transfer also fails its own test.

The 214013 ML export has exactly the same parsed rows as 181838. This verifies duplicate saved output consistency, not reproducibility of the underlying backtest.

## Concrete next validation boundary

1. Dynamic sizing is now reconciled. Recover the exact June run configuration and loaded model identity; treat the exports as internally reconciled historical output, not fresh independent validation, while run provenance is incomplete.
2. Make any subsequent replay export explicit quantities, fees, multiplier, fill assumptions, configuration/model/source hashes, and clean initial state. Reconcile cash and marks independently before interpreting performance.
3. Reproduce only an already-consumed development window to check accounting and arm comparability. Fix execution assumptions and sample rules before requesting any fresh validation; do not optimize on the existing 2026 or failed transfer results.
4. Specify a genuinely unconsumed or prospective evaluation only after the mechanics pass. Keep recurring strategy automation disabled pending sufficient evidence.

The ledger audit, sizing reconstruction and candidate triage are complete. Original run provenance and validation quality remain unresolved; no repaired strategy or profitability result is claimed.

## Source record

Paths below are relative to the repository root. Exact audited CSV hashes are recorded in the JSON.

- `_bmad-output/preregistration_yank_sl2tp8_ml050.md`; `_bmad-output/results_yank_sl2tp8_ml050.md`.
- `data/reports/backtest_1year_20260615_{181838,185354,214013}.{csv,txt}`.
- `data/reports/yank_gap_floor_oos_20260828.txt`; `data/reports/yank_viability_mes_20260828.txt`.
- `_bmad-output/validation_audit_report.md`; `_bmad-output/validation_corrections_summary.md`.
- `_bmad-output/preregistration_btc_carry_backtest.md`; `data/reports/backtest_btc_carry_v3_entry_20260613_184753.txt`; `backtest_btc_carry_v3_entry.py`.
- `_bmad-output/preregistration_btc_tsmom_backtest.md`; `data/reports/backtest_btc_tsmom_rf_20260601_005453.txt`.
- `_bmad-output/hg_gate1_holdout_verdict_20260704.md`; `_bmad-output/pl_combine_fit_verdict_20260705.md`.
- `_bmad-output/preregistration_mim_nb_post_repair_evaluation.md`.
- `_bmad-output/preregistration_kraken_thursday_short_amendment3_eval_series.md`.
- `_bmad-output/preregistration_tick_data_infrastructure.md`, amendments 10–12.
