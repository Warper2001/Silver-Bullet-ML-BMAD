---
title: Native MNQM5 volume-profile measurement audit
type: feature
created: 2026-09-14
status: done
baseline_commit: ac71b1f7237d4269255bebf4f2b27f6761eec15a
route: dispatch
review_loop_iteration: 0
context: []
---

<frozen-after-approval>

## Intent

Implement the approved native-data audit and volume-profile accuracy plan. Reuse the existing May 2025 MNQM5 acquisition and reviewed native-minute reconstruction to establish causal Globex measurement suitability and compare traded volume at price against the approved OHLCV uniform-allocation proxy.

## Boundaries & Constraints

Always use an isolated worktree, explicit local input paths, existing Python environment, bounded native decode and deterministic hash-bound outputs. Record exclusions and uncertainty. Use native capture clock, nonsnapshot T records only, integer ticks and contract volumes. Preserve repeated timestamps and native record order. The developing 70% profile follows the existing lower-price POC and adjacent-expansion tie rules. Compare both profiles at identical minute boundaries using prior completed available bars only. Halts preserve profile; sessions reset it.

Never import replay runners, live traders, credentials, sealed holdout or trade ledger. Never modify sources, live configuration, strategy parameters or install dependencies. No signals, fills, P&L, statistical power claims, threshold optimization or data purchases. Calendar uncertainty excludes sessions rather than converting an assumption into verification. Successful measurement does not unlock market evaluation.

## I/O & Edge-Case Matrix

| Scenario | Expected behavior |
|---|---|
| Native records match saved bars | Emit exact conservation, OHLC, counts and raw-digest reconciliation |
| Corrupted pins, malformed records, identity/tick mismatch | Fail nonzero; no success report |
| Partial session, unexplained gap, calendar/status conflict | Explicit per-session exclusion ledger |
| Delayed LAST event across minute boundary | Exclude unavailable snapshot; report delay without retroactive eligibility |
| Same native/proxy allocation | Identical VAL/VAH/POC |
| Concentrated volume differing from uniform | Signed/absolute tick differences with denominators |
| Existing output, symlink/hardlink collision, protected input | Reject without modifying input |

</frozen-after-approval>

## Code Map

- `tools/valentini_native_audit.py`: new standalone measurement entry point.
- `tools/valentini_reclaim.py`: reuse Profile, Bar, protected_path, strict JSON/hash helpers; do not call simulation or alter strategy constants.
- `src/research/yank_native_minute/builder.py`: reusable bounded decoder, Builder completion semantics, read_auxiliary, coverage. Load by file under private module to avoid src.research eager imports. Never import runner/replay.
- `src/research/yank_native_minute/pins.json`: acquisition file pins; ignore unrelated replay/model pins.
- Local source root `/root/Silver-Bullet-ML-BMAD/data/yank/databento-pilot-20260907`; saved reconstruction `/root/Silver-Bullet-ML-BMAD-yank-minute/data/yank/native-minute-reviewed-a`. Saved seven core hashes match prior verification. Raw files must be freshly verified.
- `tests/unit/test_valentini_native_audit.py`: isolated synthetic and filesystem tests. Existing simulator suite remains regression coverage.
- `docs/valentini-native/`: dated calendar/source evidence and run interpretation. `docs/valentini-reclaim.md`: public command and limitation update.

## Tasks & Acceptance

- [x] Add CLI with required --source-root, --reconstruction, --calendar, --output-dir. Validate inputs and publish only into a new output directory.
- [x] Stream once through pinned native files using existing builder plus capture-clock per-minute tick histograms. Verify before/after hashes, definitions and status; reconcile bars, coverage and delayed events with saved artifacts and their manifest hashes.
- [x] Load explicit dated sessions and breaks; classify every session and every observed bar. Calendar must retain source citations and unresolved evidence. Reject overlapping or malformed boundaries. Do not fabricate empty bars.
- [x] Compare eligible full-session developing profiles at pre-bar boundaries. Accumulate integer native histograms and uniform proxy, with cumulative event availability; report unavailable snapshots separately.
- [x] Write report, session ledger, per-snapshot comparison and provenance/manifest with code and input hashes. Summarize signed/absolute ticks, mean, median, quantiles and exact agreement; no adoption thresholds. Zero comparisons cannot be PASS.
- [x] Add tests for matrix, chunk invariance, snapshot/F exclusion, duplicates, conservation, halts, DST, holiday, missing minutes and event availability.
- [x] Run focused checks, benchmark bounded decode then background full scan, and publish result interpretation with retained limitations.

Acceptance criteria:
- Given validated native input, when the measurement audit runs, then every histogram conserves exact trade volume and matches independent saved bars or the command fails.
- Given incomplete/unverified evidence, when sessions are classified, then excluded sessions cannot contribute profile comparisons and every exclusion is visible.
- Given identical code/input bytes, when rerun, then deterministic artifacts match apart from explicitly operational timing logs outside artifacts.
- Given the completed study, when the report is read, then measurement suitability and remaining prerequisites for strategy testing are distinct.

## Implementation Notes

Implement code/tests/docs only within `/root/Silver-Bullet-ML-BMAD/.claude/worktrees/valentini-native`; no commits. Parent owns calendar files, actual pilot scan and final results; do not duplicate them. Approval is the user's explicit 'Implement the plan'; no new approval gate needed. Main contains unrelated untracked work; authorized isolation preserves it. No unresolved user intent or irreversible action. Parent supplies calendar evidence while implementation proceeds independently. Calendar UTC timestamps should be explicit, source-linked, with VERIFIED/UNVERIFIED per session and minute-aligned half-open breaks. Existing status transitions can occur nanoseconds after exchange boundaries: boundary-only mixed status needs separately evidenced handling, never classify all MIXED as missing data.

## Spec Change Log

## Review Triage Log

## Verification

Use `/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m pytest tests/unit/test_valentini_native_audit.py tests/unit/test_valentini_reclaim.py -q`; targeted Black, flake8 and mypy where installed. Existing native-builder tests provide additional decoder regression. Time a bounded source slice before background full scan. Inspect source conservation, exclusions, output absence of performance fields, and preserved refusal by existing market-evaluation CLI.

Initial verification: 102 focused tests passed; full native scan exited0 in126.73s, exact reconciliation of237146989 records, six eligible sessions and8274 comparisons. Independent120-minute histogram oracle and artifact accounting passed. Review pending.


Review triage (2026-09-14; every reported finding retained):

| Finding | Verdict | Evidence and disposition |
|---|---|---|
| blind-1 | medium | datetime truncates nonzero submicroseconds before alignment check; patch strict fraction validation. |
| blind-2 | medium | Calendar ESM5 identity accepted despite fixed MNQM5 native source; patch supported metadata check. |
| blind-3 | medium | End minute is outside classifier loop; premature close can pass. Patch exact closing corroboration; explicitly retain that entire inter-session intervals are not certified. |
| blind-4 | medium | Missing file guard only applies to scheduled trading minutes, leaving in-session break evidence unchecked. Patch missing source exclusion throughout session. |
| blind-5 | medium | Zero comparison report returns exit0; patch exit3 with retained explanatory report. |
| blind-6 | medium | Cached decoder identity is not tied to disk hash; patch cached-load hash binding. |
| blind-7 | low | Results artifacts are not in review diff yet; planned final evidence commit will retain full report/manifest/oracle and outputs. Direct artifact retention correction. |
| blind-8 | low | Small publication fixture mocks loading/decoding, but unit decoder tests and the completed full native CLI run plus independent120-minute byte oracle cover the composed path. Retain all actual-run evidence. A second miniature fixture combining all three native schemas requires substantial additional fixture machinery and does not close an untested delivery path; reject additional fixture scope. |
| blind-9 | medium | Only pooled summaries make session concentration harder to assess; add descriptive per-session summaries with no new threshold. |
| verification-1 | medium | Reviewer mutation dropping classifier exception argument survives all39 native tests; add parsed-calendar classification regression. |
| verification-2 | medium | Reviewer mutation quantile=0 survives all39 native tests; add independent numeric quantile oracle. |
| edge-1 | medium | Same demonstrated timestamp truncation as blind-1; patch shared root cause, retain duplicate row. |
| edge-2 | medium | Output-parent symlink replacement between validation/publication can redirect fresh artifacts; patch descriptor-relative publication and controlled regression. |
| implementation-1 | medium | Acquisition pins not cross-checked against frozen saved source list; native nontrade edits can preserve aggregate reconciliation. Current actual lists exactly match; patch equality guard plus regression. |

Preserve native/proxy causal comparison, complete source reconciliation, static protected-path checks, independent research boundary, exact boundary references and six-session exclusions. No intent change or strategy tuning is introduced by these corrections.

Final worktree verification:119 focused tests passed (56 native audit and63 existing simulator), default strict mypy, Black check and flake8 passed. Full final native scan exited0; all native artifacts reconciled. Six sessions/8274 comparisons unchanged; histograms/snapshots/observed assignments are byte-identical to initial run. The retained independent120-minute/24164-trade histogram, raw digest and volume oracle passed. All14 review rows triaged:13 addressed (one duplicate), one low additional-fixture request rejected with retained real CLI/oracle evidence, none deferred. No live files, configuration, service actions, strategy parameters, power tests or returns changed/calculated.
