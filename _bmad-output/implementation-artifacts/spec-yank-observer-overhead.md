---
title: 'Measure and reduce YANK observer overhead'
type: 'refactor'
created: '2026-09-10'
status: 'done'
route: 'dispatch'
baseline_commit: 'a0ff173522ee7a8029f53ada7be16ee0f1356e7f'
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="User supplied implementation plan and authorized consolidation">

## Intent

Measure collector overhead separately from benchmark instrumentation, optimize demonstrated costs without weakening evidence, and refresh the conditional acquisition and maintenance findings. Preserve previous benchmarks. The user authorized consolidating validation branch 16330e0 into the main repository folder; that merge is complete. Continue here, preserving unrelated edits.

## Boundaries & Constraints

Always preserve decision-boundary snapshots, ordering, loaded-code/model/configuration verification, bounded memory, evidence verifier compatibility, legacy diagnostics, HOLD_VALIDATION, installed strategy, snapshots, and frozen P&L. Report incomplete capacity-1 evidence honestly. There is no numerical production latency target.

Never purchase, refresh credentials, install, restart, collect live data, retrain, access holdouts, push, or make additional merges. Do not modify unrelated dirty files. Historical acquisition is conditional on verified account-specific expired MNQ access at zero incremental cost. Maintenance remains unapproved. Never move mutable runtime reads into the writer or silently sample evidence.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| Unprofiled latency | Baseline, disabled, fully guarded; identical fixtures | Three counterbalanced repetitions each of 2,880-bar startup and 30 polls after 7,500-bar warmup | Disclose drops, coverage, environment and measurement limitations |
| Mutation | Model, configuration, code or installed wrappers change | Capture ineligible; trading return/state unaffected by observer | Preserve permanent invalidation |
| Evidence failure | Sequence discontinuity, overflow, writer failure, interrupted publication | Verifier rejects incomplete evidence | No completion/admission fiction |
| Rollback | Another owner replaces wrapper | Preserve replacement binding | Idempotent close |
| Correctness | Both shadows, accepted/rejected synthetic decisions and order intentions | Equal decisions, quantities, tick prices and final states | Fail comparison on mismatch |
| Provider gate | No verified account zero-cost evidence or coherence guarantees | Acquisition blocked; account UNKNOWN | No provider request |

</frozen-after-approval>

## Code Map

- `src/research/yank_deployed_validation/adapter.py`: PrivateSnapshot isolates pinned strategy and synthetic services. Adapter.state hashes buffer, derives warmup metadata, normalizes state; likely startup cost. Do not alter pinned strategy or state schema.
- `src/research/yank_deployed_validation/capture.py`: bounded_payload validates and copies then serializes. record serializes/loads each decision before final serialization. DecisionCapture hands strings to worker, bounds records, fails closed.
- `src/research/yank_deployed_validation/startup.py`: prepare_observation and ObservationSession check code/config/model each poll; runtime_identity traverses helpers. Cache only immutable derivations if profiling warrants; mutable bindings still reread.
- `tests/unit/yank_deployed_validation/test_evidence.py`: prepared_runtime, signed verification and failure/mutation tests. CLI loader uses private validation module namespace; avoid mismatched class identities.
- `tests/unit/yank_deployed_validation/test_startup_benchmark.py`: old benchmark preserved. with_shadows enables synthetic parity feed and bullish methods; prepared_runtime uses mutable bars fixture. Profiling is mixed with its old latency measurements.
- `tests/unit/yank_deployed_validation/test_observation_benchmark.py`: startup_bars, original shadow poll fixtures and storage monitoring.
- `tests/unit/yank_deployed_validation/test_capture_matrix.py`, `tests/research/test_yank_deployed_capture.py`: synthetic decisions/execution and failure controls to reuse.
- `docs/reports/yank-pilot-acquisition/`: prior findings, pending gate, maintenance proposal and baseline evidence. Previous 633 regressions passed except known SL5 expectation versus SL2 configuration.
- `.venv/bin/python`: existing Python 3.12 environment; no installation needed. Existing preservation script uses historical absolute paths; verify applicable pins in main repository too.

## Tasks & Acceptance

**Execution:**
- [x] Add reproducible offline benchmark harness and focused tests under `tests/unit/yank_deployed_validation/`; record wall/CPU/RSS, capture bytes, drops and coverage using capacity 1, max_bytes 64,000,000, max_nodes 4,000,000. Use separate fresh subprocesses for latency cells, both shadows and identical inputs. Keep profiling/correctness/I/O monitoring separate.
- [x] Profile a separate diagnostic run, retain attribution for extraction/copying/serialization/runtime checks/storage handoff; measure original before optimizing. Optimize only demonstrated costs in collector modules, adding meaningful safety/equivalence tests. Record comparable before/after measurements.
- [x] Run separate correctness comparisons with both shadow paths, accepted/rejected decisions and intentions. Exercise all matrix failure controls; verify no trading-thread disk waits through existing monitored interfaces, distinguishing configured bounds from observed peaks.
- [x] Check newly supplied local provider/subscription evidence. If insufficient, retain precise acquisition/coherence blockers without network calls. If sufficient, apply existing gate and nine-request ledger and replay usable archives twice within user boundaries.
- [x] Write results and verification artifacts under `docs/reports/yank-observer-overhead/`; refresh maintenance proposal with overhead, capacity limits and outstanding operator/release/key/checkpoint requirements. Keep dates only if feasible and explicitly proposed.
- [x] Run affected regressions, deterministic legacy comparisons and preservation checks; report SL5-versus-SL2 separately. Deliver local conventional commits containing only task files.

**Acceptance Criteria:**
- Given the intended limits, when comparing modes, then three counterbalanced repetitions for each workload are retained with profiling absent from latency runs and fixture/code identities recorded.
- Given measured bottlenecks, when optimizing, then evidence remains compatible and behavior equal; absence of safe improvement is an acceptable documented outcome.
- Given missing external evidence and operational approval, when reporting readiness, then acquisition stays blocked, account UNKNOWN and operations unapproved.

## Implementation Notes

The main-folder merge preserved SHA256 of four unrelated modified tracked files. Their baseline hashes are in `.git/yank-consolidation-preservation.json`. The user already authorized this plan's implementation; no further spec approval is needed. All work remains in the main repo folder. Do not remove other worktrees or their local data. Root will independently inspect preservation and review results while implementation proceeds.

### Results retained for review

All 36 official latency cells have identical fixtures/final states across modes and before/after. All 12 guarded timing cells have complete queue/write coverage; timing packages are inadmissible because public keys are not retained and steady synthetic clocks violate receipt chronology. Separate signed correctness packages pass their verifier controls with account UNKNOWN. Capacity 1 demonstrably overflows in an unpaced fixture; it remains the proposed limit.

After the review correction for UTC `fold`, the exact built-in UTC warm-cache component improves from 38.473ms to 18.116ms; cold optimized extraction is 42.162ms. The unchanged private Clock latency fixtures bypass this cache, so the smaller full-workload medians cannot establish cache benefit or production acceptance. See `docs/reports/yank-observer-overhead/README.md` and its linked raw artifacts for limitations and separate profiling.

Broad regressions before review fixes: 618 passed; the current tree after review passes 647. Both report one known SL5-versus-SL2 expectation failure and deselect three old timing tests. The reviewed-source full-startup correctness rerun also passes 2,880 transitions, both shadow paths, state/decision equality and signature/identity/checkpoint/coverage checks. After review, 38 focused tests pass and all four profile summaries regenerate byte-identically. Final shadow compatibility tests pass six cases on both committed and locally modified strategy source. Unrelated installed source drift was observed and preserved, not edited or restored by this task. Acquisition remains blocked; no provider calls or operational actions were made.

## Spec Change Log

## Review Triage Log

| Finding | Verdict | Evidence and route |
|---|---|---|
| Blind 1: comparison validation accepts incomplete guarded coverage | medium | `validate()` checks limits, identities and state but does not require a closed, zero-drop, fully written guarded capture. The current 12 guarded cells are complete, but a later incomplete run could receive the same validated label. Route: patch. |
| Blind 2: parity shadow row counts are not validated | medium | `parity_rows` is retained but unchecked; final strategy state does not include parity-log output. Expected counts are deterministically 1 for startup and 31 for steady, including warm-up. Route: patch. |
| Blind 3: disabled mode lacks an inline callable/capture assertion | false | `prepare_observation(..., enabled=False)` returns before reading or mutating the trader, and `test_disabled_startup_never_reads_trader` proves that contract. The harness passes `enabled=mode == 'guarded'`, while retained disabled cells have null coverage and zero capture bytes. The described unintended instrumentation does not occur. |
| Blind 4: dynamic loader is omitted from benchmark pins | medium | `test_evidence.py` imports and executes `src/cli/check_yank_deployed_replay.py`; its loader determines which private collector modules execute. The collector modules themselves are pinned, but a loader change can alter loading semantics without changing those hashes. Route: patch. |
| Blind 5: exact UTC cache keys merge different `fold` values | low | Python datetime equality/hash can merge exact UTC values whose `fold` attributes differ, and the cached returned hour can therefore inherit the first caller's fold. Current downstream bucket keys treat those UTC instants equally, so practical harm is negligible, but preserving the prior helper result is a direct correction. Route: patch. |
| Blind 6: no `ZoneInfo` DST-fold test | false | Every non-`timezone.utc` value takes the original uncached `isoformat`/`replace` path. The existing fixed-offset and custom-subclass tests exercise that same branch; no DST-specific bad outcome exists in the changed logic. |
| Blind 7: no individual duration for each of 30 polls | false | The approved benchmark measures the specified 30-poll workload as one interval, and no per-poll latency or eviction-spike claim is made. Adding instrumentation would define a different measurement and require rerunning all cells; the current reported outcome remains valid for its stated workload. |
| Blind 8: UTC diagnostic does not advance a rolling window | false | The diagnostic explicitly claims only repeated state projection for an unchanged 7,500-bar buffer, while separate tests establish the 8,192-entry bound and mutable-field rereads. The report makes no rolling-window or production extrapolation, so the described overclaim does not occur. |
| Blind 9: direct `--diagnostic` uses lossy standard cProfile export | medium | The CLI still advertises and accepts this path even though the retained findings demonstrate same-label loss. The authoritative profile runner avoids it by calling `cell()` with `AggregateProfile`; direct CLI use can produce a misleading profile. Route: patch. |
| Blind 10: published profile summaries lack a reproduction command | medium | The profile runner reproduces raw pstats, but `attribution()` has no CLI route and the command ledger cannot regenerate the four published attribution JSON files. Route: patch. |
| Edge 1: `--cell` and `--workload` are not required together | low | A lone `--cell` passes `None` into `cell()`, while a lone `--workload` is silently ignored by full-batch mode. This is a reachable developer-facing CLI error with a direct argument-validation fix. Route: patch. |
| Edge 2: dynamic loader is omitted from pins | medium | Independently confirms Blind 4: the executed loader is outside the recorded identity set. Route: patch with Blind 4. |
| Edge 3: incomplete guarded coverage enters validation | medium | Independently confirms Blind 1. Current artifacts are complete, but the validator does not enforce that property. Route: patch with Blind 1. |
| Edge 4: before/after comparison does not restrict code differences to the adapter | medium | Each batch is internally pinned, but cross-batch validation only compares fixtures and final states. Unrelated collector or fixture changes could be attributed to the adapter optimization. Current artifacts differ only in `adapter.py`; enforce that expectation. Route: patch. |
| Edge 5: UTC diagnostic does not guard source stability during timing | medium | The report hashes the adapter after measurement without comparing it to a pre-measurement hash. A concurrent edit can make the recorded hash identify bytes different from the loaded/timed module. Route: patch. |

## Verification

Use the existing `.venv/bin/python -m pytest` environment. Preserve historical benchmark files. Record exact commands and full git identifiers. Diagnostics and latency are separate processes. A fresh release manifest is required after collector changes; historical release tarballs remain historical evidence.
