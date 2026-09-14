---
title: 'MNQ Globex value-area reclaim: evaluation and gated implementation'
type: academic-lit
topic: Valentini-inspired Globex value-area reclaim
decision: Whether to backtest or prospectively study the specified long-only setup
source: native web research plus explicitly separated local evidence imports
status: complete
preset: standard
validation: normal
verified_claims: 3
unverified_claims: 9
created: 2026-09-13
updated: 2026-09-14
---

# MNQ Globex value-area reclaim

## Decision

**Recommendation: retain this as an unvalidated research hypothesis; do not deploy it or treat a backtest using open, high, low, close and volume (OHLCV) bars as validation of absorption.** The supplied setup is precise enough to implement after resolving its mechanics, but this search found neither an independently measured net edge for the exact rules nor an effect-size estimate applicable to calculating this setup’s statistical power. That is an evidence gap, not proof that the strategy loses. Adjacent studies use different markets, horizons, features, or outcomes. [1] [2] [11]

**The inspected local data are not admitted for a performance test.** The main 2025 candidate matches a dollar-aggregated file, despite its minute-bar filename. Other CSVs lack established timestamp, contract, and session provenance. The identified tick manifest contains windows selected around earlier fills. These are local audit findings, documented in [the imported evidence](imports/local-evidence.md) and [hashed inventory](imports/local-data-inventory.json).

The appropriate current disposition is **DATA_UNSUITABLE**, with statistical power **UNDETERMINED**. The delivered simulator is synthetic-tested; a calibrated market-data evaluator remains unimplemented. No numerical power, profit factor, win rate, expectancy, or drawdown is claimed for this strategy. Power requires a defensible target effect and assumptions about variability and dependence; varying the assumed sample size cannot supply those missing inputs. This methodological claim is independently corroborated, not a measured power result for MNQ. [7] [13]

## Setup under evaluation

The user's choices define a long-only developing Globex profile with a 70% contiguous value area and full-session entries. After a preceding close inside value, require a close below the value-area low (VAL) on lower volume than the previous minute. Freeze both boundaries. The first later close back inside must have higher volume than its preceding minute; otherwise cancel. Enter at the next contiguous minute's open, stop one tick below the excursion low including the reclaim, and target the frozen value-area high (VAH). Profile tie rules are our deterministic convention, not a claim of exact equivalence to Sierra or Fabio. The complete mechanics and implementation limitations are in the [harness documentation](../../../../docs/valentini-reclaim.md). [3]

## 1. What the evidence supports

**Attribution — single-source, unverified.** The Fabio-associated ChartFanatics playbook describes a related mean-reversion model using a prior balance area, reclaim/pullback and order-flow aggression, usually targeting the point of control (POC). It does not establish the user's developing-Globex, adjacent-minute-volume, opposite-VA-edge model. Its examples and commercial educational claims are not audited performance evidence. The exact short video behind the supplied timestamps was not verified. [1]

**Mechanism — supported only by adjacent research.** Cont, Kukanov and Stoikov study short-interval price changes and order-flow imbalance in US stocks; their raw trade-volume relationship is noisier and less robust. This supports distinguishing buying/selling pressure from total activity, not transferring a stock-market association into MNQ net expectancy. [2]

**Positive findings from a different strategy — single study; applicability to this setup unverified.** Chen and colleagues report benefits from market-profile features in a Taiwan futures neural-network model. The retrieved full text uses time-price opportunity (TPO) features, five-minute data, a train/test split and a transaction-cost assumption. This demonstrates that profile-related models have been studied; it does not validate the specified volume-profile reclaim or supply its effect size. [11]

The 70% value area is a construction convention for accumulated volume. It is not a 70% probability that a future trade reaches VAH. This is an inference from the documented histogram construction; no such target-hit probability is established. [3]

## 2. Measurement and the chosen rules

**Measurement limitation — verified: in general, OHLCV cannot identify volume-at-price or absorption.** Uniformly distributing a minute's volume across its high–low range conserves volume but invents its allocation. Multiple transaction histories can have identical OHLCV and different profile concentrations; the explicit counterexample is retained in the measurement digest. Sierra documents approximation limitations, while the trade schema exposes price/size and aggressor information that OHLCV lacks. [3] [4]

Use complete transaction records for the eventual reference profile; full order-book data are unnecessary for volume-at-price alone. An absorption study would require a separate operational definition and richer data, and could still only infer behavior rather than trader intent. [2] [4]

**Calendar — authoritative single-source, historical scope unverified.** CME's retrieved regular schedule is 18:00–17:00 New York time with a 16:15–16:30 halt. Historical holidays, early closes, expiry and exceptional halts require dated schedules; the current general page is insufficient to certify the entire historical sample. The harness therefore accepts explicit scheduled minutes rather than inventing a calendar from observed timestamps. [5] [6]

## 3. Why the backtest must wait

Local inventory, measured this run; see the import for hashes and source qualifications:

| Candidate | Stored rows | Observed timestamp span | Admission issue |
|---|---:|---|---|
| 2025 CSV | 289,230 | 2025-01-01 to 2025-12-31 UTC | Matches audited dollar aggregation; interval and contract provenance unresolved |
| 2026 YTD CSV | 127,550 | 2026-01-01 to 2026-06-11 UTC | Timestamp semantics, contract mapping and full coverage unresolved |
| 2024 Sep–Nov CSV | 87,723 | 2024-09-01 to 2024-11-29 UTC | Timestamp semantics, contract mapping and full coverage unresolved |
| 2023 Sep–Nov CSV | 86,927 | 2023-09-01 to 2023-12-01 UTC | Timestamp semantics, contract mapping and full coverage unresolved |
| Parity tick manifest | 28 windows / 5,154 minutes | June–August 2026 | Fill-centered selection; insufficient full-session history |
| Existing MNQM5 native pilot | MBO/status/definition metadata | Request: 2025-05-19 00:00 to 2025-05-31 00:00 UTC, end exclusive | Candidate for a measurement audit; full-session/event completeness and sampling suitability not certified here |

The additional native pilot was found in a broader metadata-only search; [its request metadata](imports/additional-native-metadata.json) establish a candidate for future measurement work, not a certified power sample. Its first UTC-midnight boundary misses the preceding Globex opening; interior-session eligibility needs checking. No native tick prices were decoded.

These counts are not eligible sessions or independent observations. Larger gaps between timestamps include scheduled closures and must not be relabeled as missing data without coverage evidence. Permission to use a minute-bar proxy does not make dollar bars equivalent to minutes.

Prior related local experiments also prevent simply declaring a familiar date range untouched. Repeated specification searches on reused data can produce chance findings; a new name or exit target does not refresh the information in that data. This methodological risk is independently corroborated; it does not imply every reused-data result is false. [8] [14]

## 4. Power and falsification design

Use a separate representative pilot to identify signal incidence, session dependence and execution uncertainty; do not use final-test profitability to choose a favorable effect size. With a fixed dataset, report minimum detectable effects conditional on justified assumptions. If the inputs remain unidentified, report undetermined power rather than inventing a numerical UNDERPOWERED verdict. [7]

Consecutive-session resampling can preserve some dependence under stationarity assumptions. It cannot repair selective acquisition, missing regimes, or a structurally different sample. These are methodological applications of bootstrap and missing-data results, not claims that their assumptions already hold here. [9] [12]

The synthetic null utility deliberately excludes matched signal–outcome pairings to preserve the pre-test firewall. **Its mismatched-pair sensitivity is not an exact significance test or minimum detectable effect on strategy returns.** Exact permutation inference needs appropriate invariance and transformation rules; an attractive surrogate ranking alone is not statistical validation. A later market-data gate must calibrate and validate its null independently before permitting performance evaluation. [10]

Tests of positive net expectancy and tests of incremental volume/timing information answer different questions. A volume-ablation experiment, if later pursued, changes one parameter at a time with its own preregistration and untouched evaluation; it is not an invitation to optimize this run. [8] [10]

## 5. Recommendations and next evidence

1. **Audit the existing native pilot first, then recover or obtain representative full-session MNQ history with explicit contracts and interval semantics.** The pilot can support a future measurement feasibility check if interior sessions pass coverage checks; its short, previously selected span is not certified as a power sample. Prefer trades for reference profiles and independently generated one-minute bars for the proxy. Audit calendar coverage and compare profile boundaries and signal disagreement without selecting on P&L. Confidence: high in the measurement requirement; actual data availability remains unverified. [3] [4] [6]
2. **Keep the requested VAH target as the hypothesis.** Do not replace it with POC based on an educator's example or a favorable retrospective result. POC is a distinct future experiment. Confidence: methodological recommendation, not evidence that VAH is superior. [1] [8]
3. **Use an independently designated pilot to calibrate the gate, then reserve later data for confirmation.** Record cost sensitivity, dependence and all attempted variants. Confidence: supported statistical design; no numerical required sample size is justified yet. [7] [8] [9]
4. **Do not deploy this implementation.** It is a research harness with synthetic verification and conservative market-data admission. Lack of admission is a limitation of the available evidence, not a negative measured trading result.

The combination matters: inaccurate profile allocation changes the signal, varying bar duration changes the volume comparison, and selected tick windows distort event frequency. More resampling cannot resolve these differences between the available inputs and the data needed to test the requested strategy.

## Open questions

- Exact original video and intended profile cutoff/tie convention: resolve with an original transcript or reproducible specification.
- Complete authenticated session coverage and contract lineage: resolve with provider receipts, explicit expiries, timestamp semantics and dated exchange schedules.
- Proxy accuracy on MNQ: measure on predeclared complete sessions; no accepted numerical tolerance is asserted here.
- Net effect target, nuisance variability and costs: resolve through independent evidence or an economically justified decision target plus a separate pilot. No fee or slippage constant is borrowed from unrelated historical studies.
- Prospective observation length: derive after the above calibration. Do not turn an arbitrary number of trades into a sealed gate.

## Source appendix

All sources accessed 2026-09-13. Peer-reviewed status refers to the research work, not to replication of this strategy. “Unverified” means independent corroboration of that claim has not been established in this run.

| Ref | Supports | Publisher and source | Publication date | Accessed | Confidence |
|---|---|---|---|---|---|
| [1] | Related Fabio playbook, attribution limits | [ChartFanatics: Auction Market Strategy](https://www.chartfanatics.com/strategies/auction-market-strategy) | Undated | 2026-09-13 | Medium; unverified efficacy |
| [2] | Order-flow imbalance versus raw volume | [Cont et al.: The Price Impact of Order Book Events](https://arxiv.org/abs/1011.6402) | 2014 journal; 2010 preprint | 2026-09-13 | Medium; peer-reviewed, adjacent evidence |
| [3] | Profile construction and approximation | [Sierra Chart: Volume by Price](https://www.sierrachart.com/index.php?ID=141&Name=Volume_by_Price&page=doc/StudiesReference.php) | Undated | 2026-09-13 | High for information-loss claim with [4]; other platform facts medium |
| [4] | Transaction fields needed for profile | [Databento: Trades schema](https://databento.com/docs/schemas-and-data-formats/trades) | Undated | 2026-09-13 | Medium; primary documentation |
| [5] | Regular Micro E-mini schedule | [CME: Micro E-mini FAQ](https://www.cmegroup.com/articles/faqs/micro-e-mini-equity-index-futures-frequently-asked-questions.html) | Undated | 2026-09-13 | Medium; historical applicability unverified |
| [6] | Dated exchange calendar requirement | [CME: Holiday and Trading Hours](https://www.cmegroup.com/trading-hours.html) | Undated | 2026-09-13 | Medium; primary documentation |
| [7] | Conditional power and sample-size justification | [Lakens: Sample Size Justification](https://doi.org/10.1525/collabra.33267) | 2022-03-22 | 2026-09-13 | High for conditional-power claim with [13]; general method |
| [8] | Data reuse and specification searches | [White: A Reality Check for Data Snooping](https://doi.org/10.1111/1468-0262.00152) | 2000-09 | 2026-09-13 | High for reuse-risk claim with [14]; general method |
| [9] | Weakly dependent stationary resampling | [Politis and Romano: The Stationary Bootstrap](https://doi.org/10.1080/01621459.1994.10476870) | 1994-12 | 2026-09-13 | Medium; peer-reviewed, abstract inspected |
| [10] | Conditions for permutation inference | [Hemerik and Goeman: Exact testing with random permutations](https://arxiv.org/html/1411.7565) | Online 2017-11-30; issue 2018-12 | 2026-09-13 | Medium; peer-reviewed methodology |
| [11] | Different futures profile model | [Chen et al.: Applying market profile theory to forecast Taiwan Index Futures market](https://ir.lib.nycu.edu.tw/server/api/core/bitstreams/b840e832-cf38-4cda-96c0-3626e6ae5a03/content) | 2014-08 | 2026-09-13 | Medium; peer-reviewed, strategy transfer unverified |
| [12] | Assumptions needed to ignore missingness | [Rubin: Inference and missing data](https://doi.org/10.1093/biomet/63.3.581) | 1976-12-01 | 2026-09-13 | Medium; peer-reviewed methodology |
| [13] | Independent corroboration of conditional power | [NIST: Sample sizes required](https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm) | Undated | 2026-09-13 | High for conditionality with [7]; no trading effect |
| [14] | Independent corroboration of backtest selection risk | [Bailey et al.: The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf) | Manuscript 2015-02-27; journal 2017-04 | 2026-09-13 | High for selection risk with [8]; no strategy replication |

## Implementation and verification

The research-only implementation was committed as `530aedd5d835cbce9da7525871e14c7b5b3f2b9c` and merged into `main`. It provides a [pure simulator and admission CLI](../../../../tools/valentini_reclaim.py) plus [usage and limitations](../../../../docs/valentini-reclaim.md).

**Delivered:** causal proxy profile and reclaim state machine; one-contract execution simulation with costs, scheduled halts, gap chronology and explicit unknown intrabar execution times; metadata audit with input hashes; synthetic null/MDE utility; and terminal admission reporting.

**Not implemented:** a provenance validator and independently calibrated gate capable of issuing POWERED, or a CLI path that executes a market-data performance backtest. The delivered `evaluate` command refuses every promotion, including fabricated POWERED artifacts. The data audit and research independently explain why this run cannot supply those missing calibrations; the terminal gate is not a measured statistical-power experiment.

**Verification:** 63 synthetic tests passed both in the isolated worktree and after merge, with zero failures, errors or skips. Targeted Black, strict mypy and flake8 checks passed. Three review layers plus the lead identified 13 recorded findings, including an overlapping hard-link finding; all were fixed and regression-tested, with none deferred. Tests cover causal prefix invariance, profile ties/conservation, invalid minutes, scheduled halts, volume equality, execution bounds, stop/target ordering, costs, timestamps, numeric MDE magnitude, degenerate nulls, malformed files and input-overwrite protection. These are implementation tests, not historical strategy trades. See [verification manifest](verification.json) and [test results](tests-main.xml).

**Actual commands run:** metadata audit on the four CSVs, terminal power reporting, and an evaluation refusal check. The audit processed 591,430 rows and found zero structurally invalid OHLCV rows; this does not establish interval provenance or complete sessions. Its 2025 hash flagged known dollar aggregation, and every input retained unknown timestamp/calendar/contract provenance. The gate wrote **DATA_UNSUITABLE / POWER_UNDETERMINED**, `null_computed: false`, and `performance_computed: false`. Evaluation exited 2 before any performance calculation. All input, audit, evidence and code hashes matched after merge. See [audit](data-audit.json), [gate](power-verdict.json), and [command transcript](cli-verification.json).

No live configuration, trader parameter, credential, sealed holdout or live ledger was modified or used. No service restart or remote push was performed.

The next evidence step is to audit complete interior sessions from the existing native pilot, then design a separate representative calibration sample. Optional BMAD continuation: `[RS] bmad-deep-recon`, deepen this run with a technical data-admission question in a fresh context. This is a future task, not a performance result of this run.

## Staleness map

Computed by `recon_kit.py staleness`; see [machine-readable results](staleness.json). Earliest scheduled recheck: **2026-10-13**, for vendor schema and exchange schedules. Measurement/attribution sources are due 2027-09-13; recheck them sooner if the original video or a new data source arrives. Seminal-method claims have no automatic age limit under the academic-literature pack, but their applicability remains conditional. Undated living documentation uses this run’s access date solely as the scheduling anchor, explicitly marked `last_access_not_publication` in the input; publication dates remain unknown. Recheck applicable historical calendars before admission regardless of this schedule.
