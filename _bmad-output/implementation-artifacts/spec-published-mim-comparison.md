---
title: Published strategy versus MIM-NB research comparison
type: feature
created: 2026-09-10
status: done
route: dispatch
baseline_commit: a0ff173522ee7a8029f53ada7be16ee0f1356e7f
context: []
---
<frozen-after-approval reason="User supplied implementation plan and authorized implementation">
## Intent
Build an isolated research system evaluating published rules against deployed MIM-NB, attributing mechanisms, and separating dynamic sizing. Historical diagnosis cannot promote; prospective A/B requires 120 eligible sessions within nine calendar months. User explicitly authorized implementation and verification of this plan, without further plan approval.

## Boundaries & Constraints
Only new files under `research/mim_comparison/`, `tests/unit/mim_comparison/`, and this specification. Preserve production, services, orders, unrelated dirty workspace and historical reports. Never import study scripts or production modules with side effects. No paid acquisition, parameter search, SPY performance replication, deployment or live sizing changes. All history is exposed. Existing data only: reject missing contract provenance instead of inventing symbols. Record unavailable analysis honestly. Root agent separately writes `research/mim_comparison/evidence/` source evidence; implementation must not modify that directory.

Use two independent references: authors' vectorized executable semantics and deployed event decisions, with controlled broker fixtures and recorded decision reconciliation. Preserve full precision and flag rounded-log ambiguity. Published URL: https://concretumgroup.com/python-backtesting-beat-the-market-an-effective-intraday-momentum-strategy-for-the-sp500-etf-spy/ . No further human choices required: freeze explicit conservative operational defaults, disclose them.

## I/O & Edge-Case Matrix
| Given | When | Then |
|---|---|---|
| Contract OHLCV, timestamp label declaration and expiry metadata | audit/select | Normalize end-labeled ET; previous-session volume selects unexpired available contract; tie nearest expiry; 390 unique contiguous minutes required; duplicate/missing/early-close/invalid data excluded explicitly |
| 14 complete sessions history | next session | Deployed dimensionless sigma carries across rolls; prior close comes from selected contract; never cross-contract price returns |
| Completed signal bar | order modeling | Primary fill index i+2 open, optimistic i+1 separately; protect only after entry, adverse gaps worse than stop; 16:00 close proxy |
| Reference trigger versus actual modeled execution | guard | Stop anchored signal close ±250, guard uses deployed realized gross reference P&L ≤−1000; actual net P&L uses modeled gaps/fills and costs |
| Neutral/equality/reversal/zero volume | reference decisions | Strict inequalities; author neutral goes flat on sample; deployed opposite-band stop; zero-volume author VWAP NaN; no-lookahead |
| Shadow restart/replay or corrected bars | collection | First-observation decisions durable before next bar, no duplicate decisions; corrections never revise; missing/late unavailable for both |
| Credential-bearing process or writable production | shadow | Enforced sandbox denies credentials, sockets, production writes; isolated output only |
| 120 eligible sessions or nine-month limit | final | Bootstrap decision below; insufficient coverage incomplete/inconclusive, never automatic extension |
</frozen-after-approval>

## Code Map
- `src/research/mim_nb_live.py`: read-only reference. Marks 10:00..15:30 half-hour plus 16:00 EOD; close×volume VWAP unused for entry. UB=max(O,Cprev)+O*sigma, LB=min(O,Cprev)-O*sigma. 14 complete prior days. `_enter` signal-price anchor; `_record_trade` gross reference guard; catastrophe external/unknown/rejected events need controlled fixtures. Never import live module outside safe isolated tests.
- `study_mim_noise_bands_gate0.py`, `study_mim_noise_bands_gate1_oos.py`: read-only existing V1/V2 research, not new findings; unsafe imports.
- `data/mim_x/mnq_1min_by_contract.csv`: 141MB explicit contract,timestamp,OHLCV; use this existing history, infer quarterly MNQ third-Friday expiry with documented date boundary; inspect provenance in data/mim_x and study_mim_x2_powered.py.
- `data/processed/dollar_bars/1_minute/mnq_1min_*.csv`: available continuous CSVs lack contract column; not eligible contract-level history unless provenance independently established.
- `data/mim_nb/{bars_raw,decisions,orders,trades}.csv`: operational reconciliation only; bars ts_utc and received_at but no contract column. Read frozen snapshots; never mutate.
- Author: start labels 09:30..15:59 convert +1 minute, sample min_from_open%30; HLC3 VWAP; UB=max(O,Cprev)*(1+sigma), LB=min(O,Cprev)*(1-sigma). Sigma rolling14 min13 shift1, first day moves missing. Sample directional breakout AND VWAP, forward-fill until zero, shift1 exposures for close differences. Vol slice returns[d−15:d−1], ddof1 (14 values excludes yesterday), NaN→4x; Python round.

## Tasks & Acceptance
- [x] `research/mim_comparison/` package and CLI `audit`, `historical`, `shadow`, `evaluate`; new immutable per-run manifests, hashes of source/data/config/protocol before returns; ledgers and report; no overwrite. Include arm/config hash, contract, event/receipt timestamps, signal price, fill, quantity, costs, reason, eligibility/exclusion.
- [x] Independent references and common execution engine: A current guarded, B published guarded, C current unguarded, D published unguarded; all one contract. Gap-only, VWAP/confirmation-only, exit-only diagnostics against A; interaction combined-minus-singles.
- [x] Reports: daily net expectancy (eligible no-trades zero), paired differences, drawdown, yearly, turnover, exposure, large-winner dependence. Costs 2.24/3.24/6.24 USD round trip on actual turnover including reversals. Two timings.
- [x] Separate published-signal MNQ sizing adaptation: equity 100000, target .02, cap4, exact author lag, integer contracts with multiplier2 and bankers rounding; constant-notional comparator; realized volatility/return/drawdown/turnover/exposure, cannot promote.
- [x] Frozen prospective protocol: first full session after freeze; only A/B; complete timely data and warmup required; 120 eligible, nine months; no interim efficacy. Operational status continuous; estimate historical 120-session minimum detectable improvement (state power assumption) before collection; publish underpowering without changing horizon.
- [x] Final: 20000 stationary bootstrap draws seed7 mean block5 and sensitivity10,20; paired B−A and standalone B CIs. Support further validation only lower incremental >5 and B lower>0 plus highest-cost point estimates positive. Failure if primary incremental upper<5 or B upper<0; otherwise inconclusive including materially conflicting dependence and incomplete. Never deployment authorization.
- [x] Tests cover matrix, independent-reference agreement, log precision, no-lookahead signals/fills/sizing/selection, entry stops/gaps/guard/EOD, DST/rolls, deterministic ledgers, turnover/P&L reconciliation, sandbox and replay. Run CLI against available data; report data limitations rather than weaken protocol.
- [x] README specification matrix, mechanism report, frozen protocol, runnable shadow instructions and final decision report. Do not claim future 120 sessions completed. Do not start a persistent service.

## Implementation Notes
Existing working tree is dirty, but new research-only files fit the user's explicit isolation authorization. Prospective start must remain gated if historical contract data or operational contract provenance cannot be established. Root agent will verify generated files and run review after implementation.

## Verification
`python -m pytest tests/unit/mim_comparison -q`; exercise all research CLI commands with synthetic fixtures and real-data audit; inspect immutable manifests and safe failures.

## Spec Change Log

## Review Triage Log

| Finding | Verdict | Evidence and route |
|---|---|---|
| BH1 | false | Required archived files exist in workspace and are part of delivered new directory; review diff omitted unchanged external-source bytes, not delivery. |
| BH2 | high | Invalid first rows can fail normalization before durable journal insertion; corrected replay can become first observation. Patch durable invalid-row tombstones. |
| BH3 | high | Complete rows do not imply complete eligible decision marks; out-of-order marks can persist missing_prefix and later be counted eligible. Patch session exclusion. |
| BH4 | false | With frozen warmup and actual-time60s bound, prior RTH day cannot acquire available bars after next opening; context derives strictly prior days. Caching is a performance concern, not demonstrated context drift. |
| BH5 | high | Historical input reopened after hashing without verification. Patch before/after consumed-input and source integrity checks or immutable snapshots. |
| BH6 | high | BPF lacks AUDIT_ARCH_X86_64 check; add architecture rejection before syscall-number dispatch. |
| BH7 | false | Executed unchanged production AST with both exit attempts rejected: source still enters short and books no close. Independent oracle evidence confirms reference fidelity; preserve and document deployed quirk. |
| BH8 | false | Authors executable full_notional explicitly uses previous_aum/open, hence own-equity1x comparator is correct. README already discloses compounding; retain author convention. |
| BH9 | medium | Nanosecond timestamps pass second/microsecond checks; require exact minute flooring. |
| BH10 | medium | Existing daily scenario data permit paired/mechanism reports for all timings/costs; add summaries. |
| BH11 | high | Sequence-level published/deployed engine reference coverage absent. Add independent mixed-signal fixtures. |
| EH1 | high | Same invalid-first-observation defect as BH2; patch once with nonfinite/malformed regression cases. |
| EH2 | high | Same availability-versus-decision defect as BH3; patch once. |
| EH3 | false | Same rejected-exit source behavior as BH7; actual AST oracle disproves reference divergence. |
| EH4 | high | Same consumed-input integrity defect as BH5; live input needs observed-prefix/journal provenance rather than fixed whole-stream hash. |
| EH5 | false | Same author full_notional prior-equity convention as BH8; comparator matches executable reference. |
| VG1 | high | Pre-verified engine guard coverage gap: add two-stop loss sequence plus post-guard breakout and unguarded comparator. |
| VG2 | high | Pre-verified highest-cost veto negative-case gap; add both B and incremental-negative scenarios. |
| VG3 | high | Pre-verified actual launcher filesystem/env isolation gap; probe temporary canaries through same sandbox construction. |

## Completion evidence

Implementation complete; future prospective observation is not complete. Full suite: 40 passed in 103.45s. All actionable review findings patched; evidence-refuted findings retained with explanations. Final historical run: `20260910T210224-historical-f3950efb68`. Final audit, sandbox launch and incomplete evaluation succeeded; all 78 artifact hashes checked. See `research/mim_comparison/RESULTS.md` and `VERIFICATION.md`. Real collection awaits an identified append-only feed and qualifying fresh warmup; no orders, persistent services or deployment changes. The implementation is locally committed under the build workflow; generated immutable runs are local artifacts excluded from Git.
