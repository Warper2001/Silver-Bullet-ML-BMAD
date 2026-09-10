---
title: Reconcile MIM contract provenance into an isolated shadow feed
type: feature
status: done
baseline_commit: 20fcf79877a37fccb9390c410fd3d3f0417199c5
context: []
---
## Intent
User says continue after locating contract evidence in the project. Implement the missing research-side adapter, verify it on existing local sources, and connect it to the already frozen shadow runner. No question or permission is needed: production remains read-only. Root uses the existing bmad-build workflow; do not invoke another skill.

## Boundaries
Create only `research/mim_comparison/feed_adapter/` and `tests/unit/mim_comparison/test_feed_adapter.py`. Root owns this spec, high-level documentation and real operational runs. Never edit top-level comparison Python files: their hashes and original nine-month protocol are already frozen. Never change orders, production source, logs, services, original reports or source archives. No broker/network calls, parameter search, new paid data, or claims of prospective results. The adapter must not widen the original eligibility rules. Current historical run is `research/mim_comparison/runs/20260910T210224-historical-f3950efb68`; warmup is `data/mim_x/mnq_1min_by_contract.csv`. Warmup has a gap after Aug28; initial prospective sessions may be unavailable until current observations establish prior-session coverage. Do not splice operational bars into historical performance or reset original freeze. Ignore unrelated dirty files.

## Code Map and evidence
- `src/research/mim_nb_live.py` read-only: ChainedCsv appends fields ts_utc,open,high,low,close,volume,received_at,chain. Payload is pipe-joined values excluding chain; sha256(previous_head+'|'+payload)[:16], initial GENESIS. Source may contain known historical chain problems; record/exclude, never silently repair trust.
- `logs/mim_nb_live.log` has naive UTC millisecond timestamps (declare CLI log timezone UTC explicitly), successful HTTP GET TradeStation `/v3/marketdata/barcharts/MNQU26?interval=1&unit=Minute&barsback=60` and startup `DATA: tradestation (signal)` with `px_contract=CON.F.US.MNQ.U26`. Some historical intervals had concurrent requests for MNQM26/MNQU26; do not assign a whole file one symbol. DATA source projectx or missing/contradictory context is not qualified by TradeStation requests. Startup lines must invalidate prior DATA context until a new DATA line.
- `data/mim_nb/orders.csv` currently PLACE/CANCEL/FILL only. Code's future ROLL events put new symbol in outcome, detail AUTOROLL. These alone cannot attribute the already-fetched bar at roll: on_bar logs its bar before _maybe_roll. Request evidence governs; never use a later roll to relabel it.
- `research/mim_comparison/shadow.py`: launch accepts identified CSV and audits compatible original historical run; sandbox read-only inputs, isolated writes; finite invocation. _normalize requires contract,timestamp,OHLCV,received_at. Timeliness is both original receipt AND collector actual wall time <=60sec; old replay never qualifies. Missing bars and entire missing weekdays already exclude sessions. Do not emit malformed placeholder symbols that permanently poison entire collector.

## Design
Provide `.venv/bin/python -m research.mim_comparison.feed_adapter --bars PATH --log PATH --log-timezone UTC --state PATH`. State must be under isolated `research/mim_comparison/runs/`; inputs read-only. Finite polling invocation, no background service. Sandbox actual adapter with read-only source files, no credential/home visibility and no network, isolated state only writable. Reuse established sandbox construction if suitable but mount adapter source package explicitly; do not change core hashes. Freeze adapter sources/config/input paths and timezone in state before first attribution; reject drift on resume. Each call produces a uniquely named immutable report/manifest/exclusions/evidence snapshot, while durable state and stable feed.csv are append-only.

Reconcile each raw bar's first observation against already-written log evidence: require a unique contract among successful exact one-minute TradeStation requests within previous15 seconds of received_at, no future log timestamps, and known tradestation signal context. Track startup DATA transitions and reject contradictory contract/source context. This is log-based inferred provenance, not authenticated per-response payload identity; disclose it. Match receipt to response completion, not bar event timestamp. Preserve OHLCV, original event/receipt and add contract, attribution evidence timestamps/byte offsets and inference label. Validate values and offset-aware receipt/event; future receipts and late bars never become timely by rewriting timestamps. Reject ambiguity, absent evidence and invalid first observations permanently. Duplicates/corrections never revise first attribution. Every rejection remains auditable; skipped observations cannot count as eligible flat days.

Durability: journal raw first observations/exclusions and decisions before next raw bar; partial trailing lines remain pending until newline. Protect concurrent invocation with lock; detect source truncation/replacement or changed consumed prefix and fail closed, using recorded input prefix hashes/sizes. Efficiently index log once/incrementally rather than rescanning it for each bar; bound joins. Deterministic recovery from crash between journal and output append: no duplicate output lines, verify existing output is canonical prefix before append. CSV initial rows consumed now are historical replay, not prospective. Hash-chain breaks should be reported as exclusion evidence and must not be treated as authenticated rows; choose documented conservative policy that never repairs prior records. Avoid repeated full multi-million-row SQL processing on every poll.

## Tasks & Acceptance
- [x] Implement sandboxed CLI, frozen adapter config/source and durable append-only identified feed with per-invocation manifests and reports.
- [x] Given unique causal request and matching known DATA context, emit same OHLCV/event/receipt plus inferred contract; before/after roll requests retain their own identities.
- [x] Given ambiguous/future/absent evidence, wrong source, invalid values or chain break, permanently exclude and report; no symbol guess or corrected replay reinstatement.
- [x] Given partial writes, resume, changed input prefix, replacement/truncation, concurrent calls or interrupted output append, preserve first records, no duplicates and fail closed where integrity cannot be established.
- [x] Tests prove causal receipt join, source/startup/roll contexts, ambiguity, no timestamp rewriting, historical replay not prospective, rejection permanence, durable recovery and actual sandbox denial of production writes/credentials/network.
- [x] README documents commands, source inference limitations, current warmup/contract universe limitations, no service/deployment, and use of feed.csv with existing shadow --state.

## Verification
Run focused adapter tests. Root runs full comparison suite, real adapter, existing shadow launch and incomplete evaluate with exact frozen core. Adapter must not alter original freeze or source hashes. Report real mapping/exclusion counts, not prospective sessions. Do not execute real-data runs as implementer; root owns these.

## Implementation clarifications
The real chained bar record has only three local link failures (Jun28, Jul29, Aug6). Exclude those rows, disclose unanchored subsequent segments, and use frozen consumed-prefix hashes for first-observation integrity; do not permanently block future data or claim repaired GENESIS authentication. Root verified all 70 DATA log context records say tradestation, including latest Sep2. Core hashes still match the original historical freeze.

## Review Triage Log
| Finding | Verdict | Evidence and disposition |
|---|---|---|
| Blind1 sandbox-relative returned path | medium | Worker returns /state path and parent forwards it; translate to existing host artifact path, patch. |
| Blind2 snapshot fsync ordering | medium | snapshot_id commits after close/chmod without fsync; durable files/directory must precede cursor, patch with fault test. |
| Blind3 readable report missing | medium | Only report.json exists despite common CLI report convention; add success/failure report.md, patch. |
| Blind4 malformed log timestamp halts | false | Halt is explicit conservative fail-closed behavior with sealed error evidence; no unsafe continuation or guessed context. Operator must resolve source corruption; skipping malformed context would weaken provenance. |
| Blind5 unlimited roll transition reconstruction | low | Real source quarterly rolls between startup DATA events are bounded operationally; pathological fabricated histories are unlikely everyday use and adding a new transition policy is disproportionate. Reject per workflow low-impact rule. |
| Blind6 positive current append untested | high | Historical and initial-current rejection tests cannot catch rejection of all current appended bars; add positive and exact60-second boundary tests, patch. |
| Blind7 adapter-to-shadow integration missing | high | Operational endpoint is not exercised; prove real produced schema ingestion/replay and missing-minute exclusion using unchanged collector, patch. |
| Blind8 log mutation/partial/index recovery tests | medium | Separate log cursor/index path not covered by bar-only mutation checks; add log regressions, patch. |
| Verification1 current append path | high | Filed verified gap agrees with Blind6; patch same positive test. |
| Verification2 worker filter setup bypassed | high | Existing custom probe installs its own filter and cannot detect removal from actual main; exercise actual worker entry with substituted collector probe, patch. |
| Edge1 oversized CSV evidence | medium | Permitted request count can exceed default131072-character field capacity of downstream reader; reject oversized field permanently before emission, patch/regression. |
| Root1 duplicated production roll lines | high | Real log duplicates messages. Reproduced context_at applying identical old→new transition twice then invalidating source and rejecting valid new-contract request. Deduplicate identical timestamp/old/new transition while retaining evidence; genuinely contradictory transitions still reject. Patch/regression. |

## Review patch verification
All patch-routed findings implemented, including duplicated real-format roll messages. Full verification: `pytest tests/unit/mim_comparison -q` — 82 passed in 101.17s. Original top-level research source hashes still match the historical freeze; no returns recomputed and no protocol clock moved. Real adapter/collector initialization completed; results are recorded in `research/mim_comparison/FEED_STATUS.md`.

## Completion
Real adapter mapped78,648 and excluded7,814 records; restart preserved exact feed bytes. All emitted rows and314,029 unique original log offsets independently verified. Actual frozen shadow collector accepted feed; evaluation incomplete/inconclusive with0eligible sessions. Continuous polling not started. Production source unchanged. Original freeze/deadline preserved.

## Operational latency amendment
The first real full-feed shadow replay exceeded3minutes, exceeding the60-second arrival budget. EXPLAIN confirmed repeated full scans of4307large invalid_rows records because only identity was indexed. KEEP all frozen core/adapter sources, recorded observations, historical results and original deadline. Add a finite poll.sh wrapper with source hash freeze/lock, read-only committed-feed validation, atomic last500-complete-record CSV window, and research-journal-only event/contract index. The full feed remains authoritative and unchanged; window is a disclosed derived reader input. Tests cover0/1/499/500/501rows, partial tail, header preservation, no output outside research runs, concurrent calls, drift and index usage without row changes. Root must time a real bounded/indexed resumed poll before declaring integration ready. No background service.

## Bounded polling review triage
| Finding | Verdict | Disposition |
|---|---|---|
| PollBlind1 window/manifest interrupted publication | medium | Patch: verify matching pair before shadow dispatch; interrupted preparation fails before consumer invocation. Derived files are mutable conveniences, never authoritative evidence. |
| PollBlind2 missing fresh-checkout runs directory | medium | Patch fixture creates parent before temporary directory. |
| PollBlind3 prepare-only default poisons live freeze | medium | Patch distinct default fixture state and forbid fixture use of live state. |
| PollBlind4 incompatible same-name SQLite index | medium | Patch inspect table/columns and fail on incompatible definition. |
| PollBlind5 live orchestration untested | high | Patch test ordering, frozen arguments, failure propagation and lock coverage. |
| PollVerification1 bounded window durable handoff untested | high | Patch real collector integration across replaced windows, preserving earlier observations. |
| PollEdge | clear | No additional actionable findings. |

## Operational verification completed
All bounded polling review findings patched. Focused wrapper tests: 30 passed in5.29s. Real finite wrapper succeeded in22.339seconds versus218.41seconds for full-feed replay. Eight new after-hours bars increased mapped total to78,656; exclusions remain7,814, eligible prospective sessions remain0. All prior observations, invalid rows and flags match the original collector snapshot;38 new inventory hashes checked. Wrapper source/config frozen; original core source and protocol hashes unchanged. No continuous process started. Derived window/manifest pair is validated before dispatch; interrupted publication cannot trigger collection. Results and run links: `research/mim_comparison/FEED_STATUS.md`.

Final complete comparison suite: `pytest tests/unit/mim_comparison -q` — **112 passed in97.52s**. Shell syntax and staged whitespace checks passed.
