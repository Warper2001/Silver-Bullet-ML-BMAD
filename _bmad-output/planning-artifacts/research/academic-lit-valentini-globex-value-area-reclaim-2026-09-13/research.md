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
unverified_claims: 12
overturned_claims: 1
created: 2026-09-13
updated: 2026-09-14
---

# MNQ Globex value-area reclaim

## Current decision

**September 14 follow-up:** the native pilot audit and conditional power calculations are complete. The original CSV admission objections below still apply. Native measurement admitted six sessions; the subsequent gate remains POWER_UNDETERMINED. The current next action is the [fixed-quarter acquisition/calibration proposal](../../../../docs/valentini-calibration-plan.md), estimated at $117.35 and awaiting paid-purchase approval. These are explicitly imported local/provider results, bound in [the calibration evidence import](imports/calibration-evidence-20260914.json).


## September 14 acquisition and calibration deepening

The [imported authenticated metadata](imports/calibration-evidence-20260914.json) record 19 free schema/symbology calls and two matching account-inventory calls. No time-series request or purchase was made. The fixed candidate is MNQU5, 2025-06-22 00:00 UTC through 2025-09-13 00:00 UTC exclusive, resolving to instrument 42003472. All 60 enclosed weekdays are merely candidate trade dates. June expiry and September customary roll boundaries come from the primary CME table, not from favorable strategy outcomes. [18]

| Package, including status and definitions | Estimated USD |
|---|---:|
| Trades |46.43|
| Trade-sampled BBO |77.38|
| Best-quote updates, MBP-1 |102.97|
| Full order book, MBO |117.35|

These account-specific estimates are single-provider observations, not guaranteed charges, entitlement proof or independent-source verification. Metadata access is free; actual data delivery is usage-billed. Existing account jobs contain the previously inspected May pilot, short June 2026 windows and unrelated MSFT examples, with no copy of this proposed quarter. [16]

Recommendation: prefer MBO here for its extra order/depth evidence and reuse of the audited record format, at an incremental estimate of $14.38 over MBP-1. This is an engineering judgment combining the imported quotes and implementation evidence; new instrument/source pins and execution assumptions still require implementation and review. The acquisition is a historical calibration block, not untouched validation or proof of representativeness. A future confirmation sample starts only after its rule and analysis freeze.

The current TopstepX product-specific table supplies a candidate MNQ cost of $1.22 per contract round trip. This is a prospective fee scenario, not a claim about 2025 historical charges or actual account receipts; spread, slippage and fixed costs remain separate. Actual account receipts have not been inspected, so account confirmation remains required. No fee is adopted in live code. [17]

The [calibration plan](../../../../docs/valentini-calibration-plan.md) specifies outcome-independent coverage audit, fee/quote evidence, an explicitly preregistered descriptive nuisance study and later confirmatory power design. This resolves the next acquisition decision, but does not change the unvalidated-strategy conclusion. The [exact requests](../../../../_bmad-output/valentini-calibration-20260914/proposed-purchase.json) are reviewable; purchase permission is outstanding.

New sources: accessed 2026-09-14; credible single-primary-source claims are medium confidence and remain independently unverified. Provider quotations are point-in-time observations and must be refreshed immediately before order submission. Fee evidence should be rechecked before registering any cost model; holiday evidence is still unresolved for the proposed quarter.


## Completed follow-ups

The [native measurement](../../../../docs/valentini-native/results.md) admitted six complete regular sessions and found material differences between native and uniform-OHLCV profiles. The [conditional power gate](../../../../docs/valentini-power-results.md) calculated hypothetical sample requirements but retained POWER_UNDETERMINED and evaluation_allowed=false. Its 214 focused tests passed. These follow-ups supersede the initial to-do recommendations preserved below; no strategy profitability was tested. Their exact report hashes are in the [local evidence import](imports/calibration-evidence-20260914.json).

## Setup under evaluation

The user's choices define a long-only developing Globex profile with a 70% contiguous value area and full-session entries. After a preceding close inside value, require a close below the value-area low (VAL) on lower volume than the previous minute. Freeze both boundaries. The first later close back inside must have higher volume than its preceding minute; otherwise cancel. Enter at the next contiguous minute's open, stop one tick below the excursion low including the reclaim, and target the frozen value-area high (VAH). Profile tie rules are our deterministic convention, not a claim of exact equivalence to Sierra or Fabio. The complete mechanics and implementation limitations are in the [harness documentation](../../../../docs/valentini-reclaim.md). [3]


## September 13 initial assessment

The following findings, recommendations and implementation results describe the initial stage. Completed follow-ups and the current action are above.

### Initial decision

**Recommendation: retain this as an unvalidated research hypothesis; do not deploy it or treat a backtest using open, high, low, close and volume (OHLCV) bars as validation of absorption.** The supplied setup is precise enough to implement after resolving its mechanics, but this search found neither an independently measured net edge for the exact rules nor an effect-size estimate applicable to calculating this setup’s statistical power. That is an evidence gap, not proof that the strategy loses. Adjacent studies use different markets, horizons, features, or outcomes. [1] [2] [11]

**The inspected local data are not admitted for a performance test.** The main 2025 candidate matches a dollar-aggregated file, despite its minute-bar filename. Other CSVs lack established timestamp, contract, and session provenance. The identified tick manifest contains windows selected around earlier fills. These are local audit findings, documented in [the imported evidence](imports/local-evidence.md) and [hashed inventory](imports/local-data-inventory.json).

The appropriate current disposition is **DATA_UNSUITABLE**, with statistical power **UNDETERMINED**. The delivered simulator is synthetic-tested; a calibrated market-data evaluator remains unimplemented. At that original stage no numerical power, profit factor, win rate, expectancy, or drawdown was claimed. The later conditional-power study supplies hypothetical planning calculations, not measured strategy power or profitability. Power requires a defensible target effect and assumptions about variability and dependence; varying the assumed sample size cannot supply those missing inputs. This methodological claim is independently corroborated, not a measured power result for MNQ. [7] [13]


### 1. What the evidence supports

**Attribution — single-source, unverified.** The Fabio-associated ChartFanatics playbook describes a related mean-reversion model using a prior balance area, reclaim/pullback and order-flow aggression, usually targeting the point of control (POC). It does not establish the user's developing-Globex, adjacent-minute-volume, opposite-VA-edge model. Its examples and commercial educational claims are not audited performance evidence. The exact short video behind the supplied timestamps was not verified. [1]

**Mechanism — supported only by adjacent research.** Cont, Kukanov and Stoikov study short-interval price changes and order-flow imbalance in US stocks; their raw trade-volume relationship is noisier and less robust. This supports distinguishing buying/selling pressure from total activity, not transferring a stock-market association into MNQ net expectancy. [2]

**Positive findings from a different strategy — single study; applicability to this setup unverified.** Chen and colleagues report benefits from market-profile features in a Taiwan futures neural-network model. The retrieved full text uses time-price opportunity (TPO) features, five-minute data, a train/test split and a transaction-cost assumption. This demonstrates that profile-related models have been studied; it does not validate the specified volume-profile reclaim or supply its effect size. [11]

The 70% value area is a construction convention for accumulated volume. It is not a 70% probability that a future trade reaches VAH. This is an inference from the documented histogram construction; no such target-hit probability is established. [3]


### 2. Measurement and the chosen rules

**Measurement limitation — verified: in general, OHLCV cannot identify volume-at-price or absorption.** Uniformly distributing a minute's volume across its high–low range conserves volume but invents its allocation. Multiple transaction histories can have identical OHLCV and different profile concentrations; the explicit counterexample is retained in the measurement digest. Sierra documents approximation limitations, while the trade schema exposes price/size and aggressor information that OHLCV lacks. [3] [4]

Use complete transaction records for the eventual reference profile; full order-book data are unnecessary for volume-at-price alone. An absorption study would require a separate operational definition and richer data, and could still only infer behavior rather than trader intent. [2] [4]

**Calendar — earlier pause assumption overturned.** A dated CME notice eliminated the 15:15–15:30 CT equity-index pause effective June 27, 2021; the living FAQ used earlier [5] is not a valid basis for inserting that pause into regular 2025 sessions. Historical holidays, early closes, expiry and exceptional halts still require dated schedules. The native audit applies this correction; the harness accepts explicit scheduled minutes. [6] [15]


### 3. Why the backtest must wait

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


### 4. Power and falsification design

Use a separate representative pilot to identify signal incidence, session dependence and execution uncertainty; do not use final-test profitability to choose a favorable effect size. With a fixed dataset, report minimum detectable effects conditional on justified assumptions. If the inputs remain unidentified, report undetermined power rather than inventing a numerical UNDERPOWERED verdict. [7]

Consecutive-session resampling can preserve some dependence under stationarity assumptions. It cannot repair selective acquisition, missing regimes, or a structurally different sample. These are methodological applications of bootstrap and missing-data results, not claims that their assumptions already hold here. [9] [12]

The synthetic null utility deliberately excludes matched signal–outcome pairings to preserve the pre-test firewall. **Its mismatched-pair sensitivity is not an exact significance test or minimum detectable effect on strategy returns.** Exact permutation inference needs appropriate invariance and transformation rules; an attractive surrogate ranking alone is not statistical validation. A later market-data gate must calibrate and validate its null independently before permitting performance evaluation. [10]

Tests of positive net expectancy and tests of incremental volume/timing information answer different questions. A volume-ablation experiment, if later pursued, changes one parameter at a time with its own preregistration and untouched evaluation; it is not an invitation to optimize this run. [8] [10]


### 5. Recommendations and next evidence

1. **Audit the existing native pilot first, then recover or obtain representative full-session MNQ history with explicit contracts and interval semantics.** The pilot can support a future measurement feasibility check if interior sessions pass coverage checks; its short, previously selected span is not certified as a power sample. Prefer trades for reference profiles and independently generated one-minute bars for the proxy. Audit calendar coverage and compare profile boundaries and signal disagreement without selecting on P&L. Confidence: high in the measurement requirement; actual data availability remains unverified. [3] [4] [6]
2. **Keep the requested VAH target as the hypothesis.** Do not replace it with POC based on an educator's example or a favorable retrospective result. POC is a distinct future experiment. Confidence: methodological recommendation, not evidence that VAH is superior. [1] [8]
3. **Use an independently designated pilot to calibrate the gate, then reserve later data for confirmation.** Record cost sensitivity, dependence and all attempted variants. Confidence: supported statistical design; no numerical required sample size is justified yet. [7] [8] [9]
4. **Do not deploy this implementation.** It is a research harness with synthetic verification and conservative market-data admission. Lack of admission is a limitation of the available evidence, not a negative measured trading result.

The combination matters: inaccurate profile allocation changes the signal, varying bar duration changes the volume comparison, and selected tick windows distort event frequency. More resampling cannot resolve these differences between the available inputs and the data needed to test the requested strategy.


### Open questions

- Exact original video and intended profile cutoff/tie convention: resolve with an original transcript or reproducible specification.
- Complete authenticated session coverage and contract lineage: resolve with provider receipts, explicit expiries, timestamp semantics and dated exchange schedules.
- Proxy accuracy on MNQ: measure on predeclared complete sessions; no accepted numerical tolerance is asserted here.
- Net effect target, nuisance variability and costs: resolve through independent evidence or an economically justified decision target plus a separate pilot. No fee or slippage constant is borrowed from unrelated historical studies.
- Prospective observation length: derive after the above calibration. Do not turn an arbitrary number of trades into a sealed gate.


### Initial implementation and verification

The research-only implementation was committed as `530aedd5d835cbce9da7525871e14c7b5b3f2b9c` and merged into `main`. It provides a [pure simulator and admission CLI](../../../../tools/valentini_reclaim.py) plus [usage and limitations](../../../../docs/valentini-reclaim.md).

**Delivered:** causal proxy profile and reclaim state machine; one-contract execution simulation with costs, scheduled halts, gap chronology and explicit unknown intrabar execution times; metadata audit with input hashes; synthetic null/MDE utility; and terminal admission reporting.

**Not implemented:** a provenance validator and independently calibrated gate capable of issuing POWERED, or a CLI path that executes a market-data performance backtest. The delivered `evaluate` command refuses every promotion, including fabricated POWERED artifacts. The data audit and research independently explain why this run cannot supply those missing calibrations; the terminal gate is not a measured statistical-power experiment.

**Verification:** 63 synthetic tests passed both in the isolated worktree and after merge, with zero failures, errors or skips. Targeted Black, strict mypy and flake8 checks passed. Three review layers plus the lead identified 13 recorded findings, including an overlapping hard-link finding; all were fixed and regression-tested, with none deferred. Tests cover causal prefix invariance, profile ties/conservation, invalid minutes, scheduled halts, volume equality, execution bounds, stop/target ordering, costs, timestamps, numeric MDE magnitude, degenerate nulls, malformed files and input-overwrite protection. These are implementation tests, not historical strategy trades. See [verification manifest](verification.json) and [test results](tests-main.xml).

**Actual commands run:** metadata audit on the four CSVs, terminal power reporting, and an evaluation refusal check. The audit processed 591,430 rows and found zero structurally invalid OHLCV rows; this does not establish interval provenance or complete sessions. Its 2025 hash flagged known dollar aggregation, and every input retained unknown timestamp/calendar/contract provenance. The gate wrote **DATA_UNSUITABLE / POWER_UNDETERMINED**, `null_computed: false`, and `performance_computed: false`. Evaluation exited 2 before any performance calculation. All input, audit, evidence and code hashes matched after merge. See [audit](data-audit.json), [gate](power-verdict.json), and [command transcript](cli-verification.json).

No live configuration, trader parameter, credential, sealed holdout or live ledger was modified or used. No service restart or remote push was performed.

The native measurement and conditional power follow-ups are now complete. The next action is the exact three-request calibration acquisition described in the linked calibration plan; it is awaiting paid-purchase approval and remains separate from performance testing.



## Source appendix

Sources [1]–[14] were accessed September 13, 2026; sources [15]–[18] were accessed September 14, 2026. Peer-reviewed status refers to the research work, not to replication of this strategy. “Unverified” means independent corroboration of that claim has not been established in this run.

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


Additional acquisition/calibration sources accessed September 14, 2026:

| Ref | Supports | Publisher and source | Publication date | Accessed | Confidence |
|---|---|---|---|---|---|
| [15] | Correction of prior afternoon pause | [CME dated Globex notice](https://www.cmegroup.com/notices/electronic-trading/2021/06/20210621.html) |Notice June 24, 2021; title June 21|2026-09-14|Medium; authoritative single source|
| [16] | Free metadata, usage billing and estimates | [Databento historical API](https://databento.com/docs/api-reference-historical?historical=http) |Undated|2026-09-14|Medium; primary vendor|
| [17] | Current MNQ product-specific fees | [TopstepX commissions and fees](https://help.topstep.com/en/articles/8284213-topstepx-commissions-and-fees) |2026-07-28|2026-09-14|Medium; primary provider; actual receipt unverified|
| [18] |2025 expiry and customary roll boundaries | [CME equity-index roll dates](https://www.cmegroup.com/trading/equity-index/rolldates.html) |Undated|2026-09-14|Medium; primary exchange|


## Staleness map

Computed by `recon_kit.py staleness`; see [machine-readable results](staleness.json). Earliest scheduled recheck: **2026-10-13**, for vendor schema and exchange schedules. Measurement/attribution sources are due 2027-09-13; recheck them sooner if the original video or a new data source arrives. Seminal-method claims have no automatic age limit under the academic-literature pack, but their applicability remains conditional. Undated living documentation uses this run’s access date solely as the scheduling anchor, explicitly marked `last_access_not_publication` in the input; publication dates remain unknown. Recheck applicable historical calendars before admission regardless of this schedule.

Acquisition-slice freshness: provider quotes are valid only as timestamped observations and must be refreshed immediately before any order; current fees and schemas should be rechecked before use and by October 14, 2026. Dated holiday schedules remain required before admitting the proposed sessions.
