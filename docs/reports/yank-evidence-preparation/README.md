# YANK evidence admission and pilot preparation

2026-09-09 — **HOLD_VALIDATION**. Engineering preparation is complete in the isolated validation worktree from `a478d674e8d507928cec843b139f086d14d85296`. No installed strategy, original snapshot, prior audit outcome or frozen P&L was changed. No authentication, historical acquisition, spending, installation, restart, live collection, push or merge occurred.

## Completed engineering

- Versioned Ed25519 evidence binds the loaded runtime, fitted model, effective configuration, process start, complete initial checkpoint, capture bytes and ordered coverage. Offline trust keys and release expectations are supplied independently. Signatures establish trusted-collector provenance, not protection against a compromised host.
- Disabled startup verifies actual objects before wrapping. Memory-only checks bracket polls; filesystem/process-start checks occur at quiescent startup/close. Storage is bounded and asynchronous, failures invalidate coverage, rollback respects binding ownership, and publication writes the COMPLETE manifest last.
- Passive, redacted ProjectX observations preserve request/reply boundaries, failures and contradictions without changing deployed reconciliation. Independent signature, identity, checkpoint, account and coverage verdicts control admission. Legacy captures cannot be upgraded by caller declarations. Exact decision/quantity/tick differences remain separate from feature differences and unavailable execution evidence.
- The local TradeStation ledger proposes nine explicit-contract requests, April 1–May 30 warm-up/archive with May 19–30 evaluation, seven-day chunks and one-minute overlaps. Earliest request offsets are 0–480 seconds at 60-second spacing; all acquisition gates remain false and spending authorization is $0. Raw archive integrity, revisions, completion fields, alternative interval labels, calendar coverage and conditional replay streams are preserved separately.
- The [semantics report](../../yank-tradestation-semantics.md) cites official documentation and separates documented behavior from observations and unresolved semantics. The [maintenance and two-session protocol](../../yank-evidence-pilot-preparation.md) specifies independent trust inputs, disabled configuration, rollback, cost/entitlement gates and actual evidence fields.

## Verification

| Check | Result |
|---|---|
| Final relevant regression suites | **575 passed, 1 pre-existing failure**; [log](final-regressions.log) |
| Known failure | `test_load_repo_strategy_config_yaml` expects SL5; preserved YAML has SL2. Reproduced before changes: baseline 462 passed, same one failure. No parameter change made. |
| Observation benchmarks | **4 passed**; [log](observation-benchmarks.log) |
| Full native replay, twice | 13,440 polls each, zero poll errors, five modeled intentions, one simulated completed trade. Trace and report byte-identical between runs and to preserved prior results; [evidence](native-replay-verification.json). These are simulations, not actual execution. |
| Legacy capture comparison, twice | Byte-identical, three diagnostic polls, no exact/float differences; all remain UNASSESSABLE and admission ineligible; [comparison](legacy-comparison.json). |
| Preservation | 177 prior immutable pins and 42 additional local protected files: zero mismatches; [prior](preservation-prior-final.json), [local](preservation-local-final.json). |
| Review | Two fresh-context reviewers plus an explicitly disclosed implementation-agent verification self-review (platform thread limit prevented a third fresh reviewer). All confirmed findings corrected; [triage](review-triage.md). |

The 7,500-bar benchmark exercised the original private poller and both shadow method paths with full synthetic provider-shaped bars. Baseline/disabled/enabled decisions and state were equal. Poll durations were 137.21/137.77/148.28 seconds on the shared offline host. Default bounds deliberately failed coverage rather than silently accepting truncated startup evidence. JSONL writes occurred only on the storage worker. A delayed-storage queue test measured 869,966 bytes of traced peak allocations; this does not measure whole-runtime memory. See [measurements and limitations](observation-7500-benchmark.json).

With explicit capacity 1, 64,000,000-byte and 4,000,000-node limits, the 2,880-bar startup benchmark retained all 2,880 transitions and decision boundaries, with zero dropped records and 10,975,592 capture bytes; poll plus close took 71.69 seconds. See [48-hour sizing evidence](observation-48h-benchmark.json). These direct-observer benchmarks use synthetic transport and memory shadow sinks; verified-startup guards have separate integration tests. They do not qualify production latency, demonstrate actual fills or measure full-session production resource use.

Negative controls cover unsigned/altered bundles, untrusted keys, replaced models/configuration/code, process restart, incomplete/mismatched checkpoints, missing/reversed sequence timing, missing replies, stale/partial/inconsistent account state, original failed-HTTP-to-FLAT behavior, publication failure and wrapper substitution. Archive cases include partial/future bars, conflicting flags, malformed fields, revisions, duplicate contracts, missing minutes, DST, chunk boundaries and insufficient warm-up.

## Remaining evidence blockers

- Inspected ProjectX replies lack a common atomic snapshot ID across balance/orders/positions. Ordinary successful replies and apparent FLAT therefore remain **UNKNOWN**, including the deployed error-to-FLAT fallback. Engineering supports conservative admission; no actual account evidence has passed it.
- Independent production release/checkpoint/process pins, trusted public keys and externally managed signer/salt have not been provisioned. Local synthetic signing fixtures prove the mechanism only.
- `MNQM25` is a documentation-derived candidate; provider symbol spelling, expired-contract availability, entitlements, price and endpoint boundary/label semantics still require concrete verification. No May 2025 archive or independent session calendar was acquired. Calendar coverage alone does not prove complete feature warm-up.
- Historical response receipt timestamps describe acquisition now, never hypothetical 2025 arrival. Both interval interpretations remain conditional until endpoint-specific evidence resolves them.
- No actual observation sessions, broker acknowledgements/fills/cancels or production latency qualification exist from this preparation. No strategy-validation or profitability conclusion is warranted.

## Operations awaiting execution approval

Acquire the April–May pilot archive only after symbol, entitlement, cost and retry-budget details are concrete and approved. The dry-run ledger and worksheet are reviewable now; incremental spending remains unauthorized.

Install the disabled preparation overlay only after the exact next maintenance window, service/operator, independently trusted release inputs, dependencies and rollback artifact are approved. Then conduct two actual sessions under the protocol, retaining execution evidence only when it occurs. Never manufacture trades to obtain it.

The release manifest pins the local engineering commit and archive members. The archive is a disabled preparation overlay requiring the existing repository, snapshot and dependencies; it is not deployment authorization. Its hash and member verification are in `release-package-verification.json`.

Release built twice from engineering commit `f2bc13cc57b10e427e10a2544a13db463aafface`: identical SHA-256 `be824d38b6b60dafad75fb1e0b9b39915a4e8eb8e84b2ede12a68e031eaca6d0`, 48,287 bytes, 16 verified regular-file members. The local [build recipe](build-release.py) records the current HEAD in each rebuild; reproducing this exact archive requires the engineering revision inputs.
