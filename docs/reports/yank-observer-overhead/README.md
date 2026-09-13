# YANK observer overhead findings

September 10, 2026; reviewed and refreshed September 13. **HOLD_VALIDATION.** This work measures the private pinned runtime and prepares evidence for a future decision. Production observation, maintenance, purchases and credential refreshes remain unapproved.

## Repository consolidation and preservation

The user authorized bringing the validation branch into the main repository folder. Local merge `a0ff173522ee7a8029f53ada7be16ee0f1356e7f` incorporated the 14 validation-lineage commits through `16330e0`; all four existing modified tracked files were byte-identical immediately after that merge. Work continues in `/root/Silver-Bullet-ML-BMAD`. Other worktrees and their local historical data were not deleted.

The preserved snapshot and prior startup benchmark remain unchanged. The 177 historical-pin check matched 176 files; the main YANK source already differed from the earlier preserved installed-source fingerprint because of existing diagnostic edits. It then changed again outside this task during execution. This task did not edit, restore or stage that source. [Preservation evidence](preservation.json) distinguishes unchanged task inputs from concurrent installed-source drift.

Read-only service inspection found a different process from the earlier report: PID 1247178, started September 10 at 19:35 UTC, versus the old PID 455819. This task performed `systemctl show` only and no restart. [Installed-file and process evidence](installed-preservation.json) is a read-only observation, not proof of loaded-runtime identity. The benchmark uses the unchanged pinned private snapshot and does not claim equivalence to this changed installed process.

## Measurement design

The official latency batches use three counterbalanced repetitions of baseline, disabled observation and fully guarded observation for each workload. Each cell runs in a fresh process. Startup processes 2,880 synthetic provider-shaped bars; incremental cells perform an actual 7,500-bar warm-up before 30 timed polls. Observation begins after that warm-up. Both existing shadow paths execute with private in-memory sinks and a fixed 15-bar synthetic second-feed tail. This is not live parity evidence.

Fixture construction and hashing finish before timing. No decision profiler, tracing hook or filesystem monitor runs inside latency cells. Wall time, process CPU time, preparation, close, warm-up, absolute process high-water RSS, capture size, drops and coverage are retained in [original metrics](latency-before.json), [optimized metrics](latency-after.json) and the [validated comparison](comparison.json). [Host metadata](environment.json) documents the shared environment; no CPU isolation or production certification is claimed.

Latency cells use synthetic self-derived release expectations to exercise the complete guards. Their capture packages are timing diagnostics: signing public keys are not retained, so these packages cannot later serve as independently signature-verifiable admission evidence. Steady timing cells also combine the private historical strategy clock with real collector receipt times: later historical poll starts precede earlier real receipts. Complete queue/write coverage therefore does not establish verifier chronology coverage. These timing packages remain inadmissible; no verifier check was relaxed. Separate correctness runs retain public keys, independent synthetic control expectations and verifier verdicts. Neither kind supplies real account, process or production-release attestation.

Exploratory timings were superseded after finding that a synthetic shadow closure retained thousands of unused startup rows. The official harness retains only the 15 rows it returns, preserving feed behavior while removing artificial runtime-verification work. Historical benchmarks were not overwritten; exploratory results do not support the final latency comparison.

## Measured results

All 36 official cells pass fixture and final-state comparisons. The twelve guarded cells report complete queue/write coverage and zero drops. Median values below are seconds; CPU includes the process's worker thread.

| Workload | Mode | Wall: original → optimized | CPU: original → optimized |
|---|---|---:|---:|
| Startup: 2,880 bars | baseline | 36.959 → 36.297 | 36.957 → 36.295 |
| Startup: 2,880 bars | disabled | 37.510 → 38.079 | 37.507 → 38.075 |
| Startup: 2,880 bars | guarded | 76.113 → 73.828 | 76.272 → 74.021 |
| Incremental: 30 polls | baseline | 0.905 → 0.818 | 0.904 → 0.818 |
| Incremental: 30 polls | disabled | 0.830 → 0.743 | 0.830 → 0.743 |
| Incremental: 30 polls | guarded | 3.355 → 3.010 | 3.394 → 3.045 |

Startup guarded wall time decreased 3.0%, while incremental guarded time decreased 10.3%. Baseline also changed between batches. Guarded minus baseline medians changed from 39.154 to 37.531 seconds at startup and from 2.450 to 2.192 seconds per 30-poll batch. These are differences of mode medians, not paired causal estimates. Startup ranges overlap, the environment is shared, and three repetitions do not establish a production latency guarantee. Disabled-observer variation likewise does not demonstrate a speedup or a stable overhead penalty.

Guarded startup captures were 10,975,591 bytes each; incremental captures were 518,494 bytes each, unchanged before/after. Guarded RSS high-water ranges were 339,620–340,668 KiB before and 340,116–340,924 KiB after at startup, and 280,884–281,388 versus 281,604–282,060 KiB for incremental processes. There is no demonstrated memory reduction. Preparation, close, warm-up and every individual sample remain in the metric files rather than being folded into poll latency.

The separately rerun [built-in UTC component experiment](utc-state-diagnostic.json) compares complete equal states for 7,500 bars: original extraction median 38.473ms versus warm-cache optimized 18.116ms, with 45,000 hits, 7,500 misses and 7,500 retained entries. Cold optimized extraction took 42.162ms. This demonstrates a warm-cache benefit for that narrow timestamp representation, not for the official private Clock fixtures or an unmeasured live process.

## Demonstrated bottleneck and change

The separate original startup profile attributes 28.766 seconds of cumulative instrumented time to state extraction across 5,762 calls. Repeated timestamp formatting and hour derivation account for millions of calls; the remaining extraction includes buffer scanning, hashing and normalization. Copying, serialization, runtime verification and producer handoff remain visible in the retained diagnostic profiles. These cumulative categories overlap and cannot be added or substituted for unprofiled latency.

The collector now caches only immutable timestamp formatting/hour derivations, with 8,192 entries per loaded adapter module, and builds buffer rows and hourly buckets in one pass. The pinned private runtime parses bars into its Clock datetime subclass, which intentionally bypasses this cache. Its official timing change therefore cannot demonstrate the cache benefit; the combined traversal and shared-host variability also affect the result. Caching applies only to exact built-in, unfolded UTC datetime objects; folded UTC values, other timezone representations and subclasses retain the original operations. Every mutable bar value is still read on the observation thread. Runtime identity checks, model/configuration guards, snapshot ordering, copying, serialization and bounded storage handoff remain in place. No evidence sampling or mutable runtime reads moved to the writer.

The optimized startup diagnostic records 30.965 seconds in state extraction versus 28.766 seconds originally. Timestamp formatting/replacement call counts remain 4,211,773 and 4,165,882, with no cached-helper entries. The new fallback helper adds millions of Python calls under the profiler, so those instrumented durations are not an unprofiled latency comparison. No private-Clock extraction speedup is established. The separate steady diagnostic attributes 1.508 seconds to extraction, 0.108 seconds to bounded copying, 0.524 seconds to canonical serialization, 0.008 seconds to JSON decoding, 2.460 seconds to runtime identity checks and 0.068 seconds to producer handoff. These overlapping profiled costs identify remaining work; runtime guards were retained, and writer-thread disk time was not profiled. See the [startup](profile-after-startup.json) and [steady](profile-after-steady.json) diagnostic records.

The diagnostic driver aggregates distinct code objects sharing a source label after profiling stops. Standard cProfile export can overwrite those entries when the private fixture reloads modules; its lossy export is retained as supplemental evidence only. The authoritative aggregate omits caller edges rather than inventing a caller graph.

## Conditional acquisition and account evidence

No new account-specific subscription record or provider confirmation was supplied or found in the inspected local repository locations. The existing gate rejects missing zero-cost and entitlement evidence and incomplete reviewed credential/endpoint/approval bindings. [The local evidence check](provider-check.json) records its search scope and exact blockers. This task made zero authenticated requests, zero historical requests and zero credential reads or refreshes. The existing nine-request ledger remains unexecuted; no newly acquired archive or replay result is claimed.

No new ProjectX coherence evidence was found. The prior provider assessment remains the applicable evidence: separate replies or stream events do not establish an atomic cross-entity decision-boundary snapshot, ordering, missing-event detection and reconnect replay guarantees. Account admission remains **UNKNOWN**. This work did not perform a new provider-documentation search or subscribe to live events.

## Verification and capacity limits

The pre-review [broad regression run](regressions.log) passed **618 tests**. After the repository gained unrelated tests and the review fixes landed, the [current-tree run](regressions-review-final.log) passed **647 tests**. Both runs deselected three old timing tests and reported only the known SL5-versus-SL2 configuration expectation failure. The preserved configuration remains SL2. The initial broad run also exposed a stale main-folder shadow fixture after consolidation; its fixture used the old combined logger attribute. The fixture provides both committed and locally installed logger attribute names using the same synthetic trade sink, avoiding a dependency on uncommitted strategy changes. All six tests pass against both the [local source](shadow-compatibility-local.log) and the [committed source loaded in memory](shadow-compatibility-committed.log). Installed strategy code was not changed by this task.

The preserved [full-startup correctness test](full-startup-correctness.json) passed separately: 2,880 transitions, equal final state and 2,637 equal filter-decision records, both shadow paths, and successful signature/identity/checkpoint/coverage verification. The [reviewed-source rerun](full-startup-review-final.log) passes the same checks against adapter SHA256 `53d48184322834de939928f4cf08fed95bfb203626b44857c8a1134728ca471f`. This legacy test profiles decisions and uses capacity 16; its timings are excluded from the official capacity-1 latency comparison. Through its monitored `Path.open`, `builtins.open`, `os.open` and capture-write interfaces, no poll-thread filesystem access occurred and capture writes were on `yank-decision-capture`. This is not a kernel syscall audit.

The final [38 focused tests](focused-tests.log) cover cache bounds and representation/mutable-field equivalence, synthetic accepted/rejected decisions and intentions, signed single/multiple-poll packages, queue overflow, benchmark validation and diagnostic aggregation. [Retained correctness packages](focused-verification.tar.gz) include public keys and verifier verdicts. All four synthetic signed packages pass signature, identity, checkpoint and coverage checks; account remains UNKNOWN and admission is ineligible. Correctness-only quiescent pacing occurs between polls, never inside the trading path or official latency driver.

**Capacity 1 does not guarantee complete evidence.** An unpaced small fixture actually retained one of three records and dropped two, correctly invalidating coverage with queue overflow; [retained failure](capacity-one-observed-incomplete.json). The separate delayed-writer test retained 200,046 queue bytes and observed a 669,231-byte `tracemalloc` peak for a roughly 200KB payload; eleven further emits failed closed. This small-fixture peak does not establish the peak at the configured maximum. Capacity 1 bounds queued serialized records to at most 64,000,000 bytes; an in-flight writer record, observation trees, copying/serialization temporaries and the bounded timestamp cache are additional memory. Absolute RSS includes libraries, fixture construction and warm-up. No hard whole-process 64MB memory claim is made, and the production proposal remains capacity 1.

Model, configuration, loaded method/helper/observer-code mutation, sequence discontinuity, writer failure, interrupted publication and rollback ownership/idempotence are exercised by the broader validation suites. [Two repeated legacy comparisons](legacy-repeat-verification.json) are byte-identical to each other and semantically identical to the preserved prior comparison; only the collector code-hash field changed. The legacy comparison remains ineligible and HOLD_VALIDATION.

Three independent review passes covered defects, edge cases and verification gaps. They found no verification gap. Review fixes now reject incomplete official comparison coverage, validate both shadow paths, pin the dynamic loader, restrict cross-batch code changes to the adapter, preserve UTC `fold`, reject lossy direct profiling, validate CLI argument pairs and reproduce profile summaries. The fold correction does not affect the private Clock timing fixtures; the UTC component result was rerun against source commit `a21a6a7dbe7d2e4e2d6c40ac9f83dff8e5bfb9fc`.

## Reproduction and retained artifacts

[Execution commands](execution-commands.json) identify the official batches and superseded exploratory/diagnostic attempts. [Root verification commands](root-verification-commands.json) record the broader regression and legacy startup invocations. Raw batch results and diagnostic profiles are retained in the before/after archives in this report directory; temporary execution directories are not required to read the findings. Correctness packages are stored separately from timing captures.

To run another optimized latency batch from the repository root with the existing environment, choose a new output directory:

```bash
.venv/bin/python tests/unit/yank_deployed_validation/observer_overhead.py --output data/yank/observer-overhead-repro
```

The driver rejects an existing output directory and runs all 18 cells sequentially. Allow for nine full 7,500-bar warm-ups in addition to startup and timed polling. Do not run the profiler or other heavy verification alongside a latency batch. Reproducing the original collector requires its recorded source and harness pins; current HEAD alone is not the original baseline.

## Maintenance readiness

The [maintenance proposal](../yank-pilot-acquisition/maintenance-proposal.md) remains unapproved. Its September 12 window expired, and the September 14/16 evidence-session candidates were withdrawn. No replacement dates are proposed while the gates remain unresolved.

Outstanding requirements include a named operator, approved launcher and release, independently reviewed loaded-code/model/configuration and checkpoint expectations, external signer and independently distributed public key, explicit latency acceptance, contract-roll confirmation and current rollback review. Existing source/process drift and collector changes require renewed release expectations; historical release/rollback artifacts cannot establish current readiness. No deployment, installation or restart is authorized by these measurements.
