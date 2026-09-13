# Portfolio PF improvement shortlist execution

## Stage 1 — execution reconciliation

**Gate: `INSUFFICIENT_CAUSAL_EVIDENCE`.** Later feasibility stages were allowed because no current repair candidate was proven. Exact order IDs were used before any descriptive matching. Broker net appears only for complete two-leg MIM saved-export round trips with explicit fees and commissions. GAP remains gross-only because timestamps, executable quotes and complete costs are absent.

- MIM saved export contains 7 complete shared-account round trips; 5 have both order IDs and one unique local MIM trade, while 2 remain unattributed.
- No executable quote series binds decision-to-arrival and arrival-to-fill movement for MIM.
- GAP has exact entry/exit order IDs and gross differences, but no fill timestamps, executable quotes, or complete costs.
- 2 MIM source chains have unregistered historical breaks; chain validity does not establish record completeness.
- Local trade rows do not carry broker order IDs, so missing broker observations remain explicit rather than being repaired by inferred price matching.

Observed signed fill-minus-model differences are **-$6.00 for five exact MIM pairs** and **-$113.00 for 21 GAP gross-only rows**. Neither is a recoverable-loss estimate; MIM coverage is partial and GAP lacks complete costs and causal timestamps.

## Stage 2 — MIM profit-giveback feasibility

**Verdict: `DISTINCT_HYPOTHESIS_REMAINS`.** The hypothesis is distinct from the prior exit rules and has complete descriptive coverage, but this does not establish predictive separation or PF improvement. At observed marks, eventual losers had median giveback **$155.50** versus **$54.50** for eventual winners. Their fixed 10th--90th percentile ranges overlap: **$39.50--$384.30** for losers and **$10.00--$199.65** for winners. These groups use retrospective labels.

## Unchanged MIM baseline

| Metric | Value |
| --- | --- |
| Closed-trade net PF | 1.28566880 |
| Net profit | $21,889.76 |
| Trades / sessions | 801 / 1323 |
| Exposure contract-minutes | 222,077--222,148 |
| Daily max drawdown | $2,437.26 |
| Top 5% trade net / total net | 1.35x |
| Top 67 day net / total net | 1.81x |


The 6,673-row scheduled-mark ledger contains only completed information available at the existing 10:00--15:30 ET marks. It includes current causal P&L, running MFE/MAE, giveback, prior-mark change and original retrospective outcome labels. The seven reversal marks attach to the exiting leg. It calculates no candidate exit return, threshold, window or classifier.

Exit labels remain 71 `CAT_STOP`, 723 `EOD_CLOSE_PROXY`, and seven `REVERSAL`.

## Stage 3 — commodity carry feasibility

The 12-root by 3-venue matrix contains 36 rows. Overall verdicts are `PARK_ACCOUNT, PARK_DATA, PARK_POWER, SOURCE_CLARIFICATION_REQUIRED`. Topstep is unavailable for the proposed monthly overnight book; TradeStation SIM is validation-only; a future self-funded path remains requirements-only. Account, data and power reasons coexist. Prepared source/account questions remain unsent, and no alternative-strategy return was calculated.

Official-source observations are dated in `carry_evidence.csv`: [Topstep trading hours](https://help.topstep.com/en/articles/8284206-when-and-what-products-can-i-trade), [TradeStation futures margins](https://www.tradestation.com/pricing/futures-margin-requirements/), and [TradeStation physical-delivery policy](https://uploads.tradestation.com/uploads/Futures-Physical-Delivery.pdf).

Prior verdicts remain TSC-1/TSMOM-1/COT `FAIL` and XSMOM-1/VRP-1 `UNDERPOWERED`.

## Portfolio decay monitor boundary

The monitor file's observation timestamp and row coverage are inventoried in `decay_monitor_coverage.json`. Its efficacy is not interpreted; the monitor script and log were not called or edited.

## Artifacts

`execution_events.csv`, `execution_round_trips.csv`, `execution_coverage.csv`, `mim_trades.csv`, `mim_decision_marks.csv`, `mim_lineage_audit.csv`, `mim_path_distributions.csv`, `mim_path_verdict.json`, `mim_hypothesis_specification.md`, `carry_matrix.csv`, `carry_evidence.csv`, `carry_questions.md`, frozen source snapshots and completion hashes are included in this sealed run.
