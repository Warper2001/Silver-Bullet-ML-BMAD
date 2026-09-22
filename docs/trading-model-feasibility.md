# Trading-model feasibility: readiness before training

The implemented first deliverable is an offline readiness audit and a preregistered boundary around what it may conclude. It does not train a strategy. The initial disposition is **HOLD_DATA / UNASSESSABLE power**; a completed scan is not permission to trade or test strategy returns.

## What is available

- `tools/trading_model_readiness.py`: standard-library-only CSV validation, descriptive timestamp/contract/coverage inventory, input and script hashes, CPU audit timing, and compute-runtime detection.
- `_bmad-output/preregistration_trading_model_readiness.md`: readiness-only registration, committed as `ff7fbeb74491704ea0a5c36e5ac1aff2d5cf633f` before audit implementation or measurement.
- `tests/unit/test_trading_model_readiness.py`: synthetic tests for DST, both timestamp labels, overlapping contracts, invalid rows, duplicates, missing data, output isolation, sealed-path aliases and preregistration tampering.
- `docs/reports/trading-model-feasibility/`: measured evidence and qualifications. Existing reports are immutable; reruns require fresh directories.

The audit takes repeated `--input` CSV arguments, `--output-dir`, and `--source-revision` (a full Git SHA). It writes `report.json` and `report.md`. Exit zero means the descriptive audit completed, **not** that data or power passed; always inspect the explicit gate fields. Missing and malformed datasets are reported. Unsafe paths, changed inputs, preregistration mismatches and existing destinations refuse the run.

Use the research interpreter and run outside the live `src` package. Before obtaining a revision or otherwise operating Git, inspect status and both divergence counts as AGENTS.md requires. Supply the actual committed code revision; the report records it as runner-supplied and independently hashes the script. Launch a full historical scan with `nohup nice -n 10 ... > <new-log-path> 2>&1 &`, since it can exceed 30 seconds. There are no broker calls, trader imports, dependency installations or model downloads.

## Candidate data and limits

The initial run uses three explicit inputs, without discovering or traversing holdouts:

| Input | Use | Limitation |
| --- | --- | --- |
| `data/mim_x/mnq_1min_by_contract.csv` | Primary structural inventory | Contract labels alone establish neither source authenticity nor a causal front-month roll policy. This is previously researched data. |
| `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv` | Known-defective diagnostic comparator | Documented interleaved contracts during roll periods; no contract column. Do not train on this series as if it were front month. |
| `.claude/worktrees/gapfade-splice-sensitivity/_bmad-output/diagnostics_gap_fade_splice_20260916/mnq_1min_2025_frontmonth.csv` | Explicitly named diagnostic reconstruction | The main-checkout copy is absent. This separate worktree copy must never be substituted silently. Reconstruction selected the contract by **full-session minute-count majority**, not information available at session open, and discarded row-level contract labels. |

The reconstructed file's documented SHA-256 is `f1fe5b36abba90681d8b1439a3975f94e4b4d1040093c7c0368629a807d219d4`; compare the measured hash. Do not interpret a matching hash as economic admission. Reference construction: `_bmad-output/diagnostics_gap_fade_splice_20260916/rebuild_2025_frontmonth.py` and `rebuild_meta.json` in the main checkout. The older `notional` column is ignored; it must not set MNQ PnL multipliers.

The two grid hypotheses correspond to 09:30–15:59 or 09:31–16:00 America/New_York minute labels. Counts assume an ordinary weekday session, exclude duplicate contract-days, and are descriptive only. Holidays, early closes, entirely missing sessions, bar completion, revisions and arrival times require independent evidence. Multiple contracts and overlapping files cannot be summed into independent sessions. No roll choice, interpolation, price adjustment or favourable-session selection occurs.

The existing `research/mim_comparison/data.py` offers a previous-session-volume selection reference. It is not adopted automatically: its business-day calendar, expiry approximation and regular-grid exclusions need independent validation. The older `tools/yank_frontmonth_revalidation.py` is not run/imported because its replay has different assumptions and guarded holdout-period access.

## Power and compute disposition

No justified economic target effect, admissible untouched evaluation interval or dependence-adjusted variance model has been established for this new model comparison. The audit explicitly returns **UNASSESSABLE**, not PASS or UNDERPOWERED. A valid experiment-specific power gate must follow once those quantities and the comparison allocation are registered. Repeated bars, overlapping windows, synthetic scenarios and training seeds do not create independent market outcomes. Earlier strategy power gates cannot be reused as this model's authorization.

This host has a research Python environment but no installed torch/transformers and no detected `nvidia-smi` during initial inspection. The report probes the runtime again. Only elapsed CSV processing time is measured; hashing time is outside that throughput measurement. GPU training hours and cost remain null until the exact model, context length, batch, optimizer, precision, rollout count and device have a representative benchmark. A GPU visibility check alone is not that benchmark. FP16 parameter-storage arithmetic is a weights-only lower bound and does not predict GPU fit.

No paid resources or packages have been provisioned. GPU provisioning is a later budget decision; nothing in this assessment claims that data fees or GPU time are already covered.

## Future comparison protocol — unsealed, not executable

The user-approved design starts with MNQ intraday decisions on completed 15-minute observations, target positions −1/0/+1 contract, and session-end flattening. These are proposed experimental boundaries, not optimized or adopted strategy parameters. Variable sizing and a literal LLM/news comparison follow separately.

Once the readiness requirements are resolved:

1. Register exact data and timestamp semantics, causal roll selection, untouched chronological evaluation, costs and fill assumptions, power inputs and statistical decisions before a strategy test. Calibrated decision thresholds must cite development sweeps and commits; do not insert hand-set economic thresholds into a seal.
2. Build an isolated simulator with a common observation/action interface. Observations contain only information available then; fills occur after decisions. Mark-to-market accounting includes open losses and costs. Independent code enforces position, session and account-risk constraints. Missing/stale observations cannot manufacture fills or future knowledge.
3. Establish statistical and small neural-policy baselines. Change only the market representation for the first pretrained-model comparison, holding the observation budget, actions, costs and reward constant. Predeclare the family so repeated model trials are counted.
4. Evaluate net performance, risk-adjusted performance, drawdown, turnover and cost sensitivity with chronological separation, dependence-aware uncertainty and repeated training seeds. Preserve large-winner behaviour in evaluation; don't add filters because a historical subset looked attractive. Audit base-model pretraining exposure separately from local training splits.
5. Freeze a qualifying candidate for prospective paper observation. A historical result alone never authorizes live deployment or sizing changes. FAIL and UNDERPOWERED stop the experiment without searching the reserved evaluation data for a rescue.

For numerical inputs, a pretrained market representation plus a small learned policy is the proposed starting point. [Kronos](https://github.com/shiyu-coder/Kronos) supplies candidate pretrained representations, not an established MNQ edge. [Trade-R1](https://arxiv.org/html/2601.03948v2) demonstrates return-linked LLM training but does not establish this strategy's profitability. No model is selected, downloaded or admitted by this readiness audit.
