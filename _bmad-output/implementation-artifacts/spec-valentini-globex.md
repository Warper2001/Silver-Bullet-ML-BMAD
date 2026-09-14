---
title: 'Gated Globex value-area reclaim research harness'
type: feature
created: 2026-09-13
status: done
route: dispatch
review_loop_iteration: 0
baseline_commit: 099044ef23e1e8a7bc6662fd2b9315c822538017
context: []
---

<frozen-after-approval>

## Intent

Implement the user's approved research plan: reproducible MNQ long-only developing Globex value-area reclaim feasibility, gated before performance evaluation. The user approved full Globex entry hours, adjacent one-minute volume comparisons, and a disclosed OHLCV proxy. Research may conclude DATA_UNSUITABLE or POWER_UNDETERMINED without strategy returns.

## Boundaries & Constraints

Always work under `/root/Silver-Bullet-ML-BMAD/.claude/worktrees/valentini-globex`. Run Python/tests with `/root/Silver-Bullet-ML-BMAD/.venv/bin/python`; install nothing. Use pure research code with no trader imports or persistent trade logging. Do not read sealed_holdout, credentials, or live ledger. Do not change any existing trader/config. Do not commit; lead handles review and commit. No new user approvals are needed for this already-approved implementation.

Use 70% volume area, tick 0.25 and $2/point, one contract. Proxy allocates bar volume uniformly over tick levels low through high. POC ties select lower price; expand contiguously toward greater adjacent volume with lower-price ties. Profile uses preceding completed bars only. Previous close must be inside pre-break VA and current close below VAL with lower volume than previous bar. Freeze VAL/VAH; track excursion low. First later close >= VAL confirms only if <= VAH and its volume > preceding volume; otherwise cancel. Enter next contiguous bar open with adverse costs only if open and slipped entry lie strictly between stop and target. Stop one tick below excursion low including reclaim; target frozen VAH. Stop-first on ambiguous bar; stop gap fills worse of stop/open minus slippage; target requires trade-through and fills target. Exit market at session end with costs. No same-bar rearm after exit/cancel; fresh break next bar allowed. Cancel pending on gap/session end. Never fabricate zero-volume bars. Invalid data rejects session, not silently drops rows.

## I/O & Edge-Case Matrix

| Input/state | Expected behavior |
|---|---|
| Complete synthetic session, valid reclaim | next-bar entry, fixed stop/target, explicit ledger |
| Future bars added/modified | earlier signals unchanged |
| Missing minute, off-tick/invalid OHLCV, duplicates | reject session before evaluation |
| Unresolved timestamp/calendar/contract evidence | reject admission; no performance |
| Absent, non-POWERED or mismatched gate | CLI evaluation refuses before simulator |
| Equal volume, first reclaim overshoots VAH | no entry |
| Session boundary, shortened session, DST | obey supplied independently defined UTC schedule; reset |

</frozen-after-approval>

## Code Map

- New `tools/valentini_reclaim.py`: standalone typed module and CLI; no service wiring.
- New `tests/unit/test_valentini_reclaim.py`: synthetic deterministic acceptance tests.
- New `docs/valentini-reclaim.md`: commands, input contracts, interpretation, limitations.
- `tools/xsmom1_power_gate.py` is conceptual firewall precedent only. Do not reuse its unrelated effect constants.
- Existing `mnq_1min_2025.csv` hash `3f20ec70885cdee6b48e6c5c7ed3254dd4cc8ce7bd8533696c5e461c75fb7822` is a dollar-aggregated file, not reliable fixed minute bars. Admit no local CSV by filename. Lead audits it separately.

## Tasks & Acceptance

**Execution:**
- [x] Implement typed Bar/Signal/Trade, causal profile and deterministic standalone simulator in the new module.
- [x] Implement metadata `audit` CLI accepting repeated CSV paths and JSON output; stream/hash inputs, record count/date range/duplicate/gap/schema/invalid-row findings, explicitly unknown label semantics/calendar/contract provenance. Protect both requested and resolved paths from sealed_holdout, including symlinks. Audit never calls signal or return code.
- [x] Implement `power` CLI consuming audit and evidence, producing strict JSON verdict and input/code hashes. Missing valid data => DATA_UNSUITABLE; missing effect or calibration => POWER_UNDETERMINED. Do not issue POWERED from an unchecked boolean or user-authored token. Current version may conservatively support terminal verdicts only; document that powered promotion requires an independently calibrated future gate.
- [x] Provide pure null/MDE utility using only mismatched session pairings, disallowing fixed points and identity, reporting dependence sensitivity. No real-data null calculation when data admission fails. Unit-test with synthetic panels; do not borrow an empirical effect.
- [x] Implement `evaluate` CLI with fail-closed admission before any performance read/execution; no force/bypass option. This version must reject forged POWERED files because it cannot yet independently establish powered promotion. Pure simulator remains executable on synthetic fixtures in tests.
- [x] Unit-test every matrix row plus conservation/ties/zero volume, low tracking, volume equality, costs, stop gaps/trade-through, simultaneous touches, deterministic nulls, audit/gate hash mismatch and forged gate.
- [x] Document exact commands, synthetic versus market-data boundary, evidence required for future powered evaluation, full Globex profile semantics, proxy bias and research-only status.

**Acceptance Criteria:**
- Given the approved mechanics, when synthetic fixtures run, then signals and fills reproduce causal hand-calculated expectations.
- Given unsuitable/unverified real data, when audit and power run, then reproducible non-performance artifacts explain blockers and evaluate exits nonzero without a trade ledger.
- Given absent transferable effect evidence, when gate runs, then it cannot label the study UNDERPOWERED or POWERED from an invented effect.
- Given any rerun with identical input/code, when JSON artifacts are generated, then their deterministic fields match and nonfinite floats are not emitted.

## Implementation Notes

User's plan approval and 'Implement the plan' authorize this scope. No additional intent gaps or irreversible actions. Initial data investigation materially constrains the feasible deliverable: build and test the requested simulator, but fail closed on current market data rather than manufacture a power calibration. The full-session calendar is an explicit external input to pure session simulation; do not infer a holiday schedule from observed bars.

## Spec Change Log

## Review Triage Log

| Finding | Verdict | Evidence | Route |
|---|---|---|---|
| blind-1 | high | Hard-link output aliases can truncate a CSV despite string path guard. Verified by reviewer synthetic samefile reproduction; duplicate of edge-1. | patch |
| blind-2 | medium | Permissive DictReader accepts unterminated quote as clean valid row. Strict parsing and regression required. | patch |
| blind-3 | medium | Duplicate JSON keys have divergent consumer semantics and last-wins acceptance. Reject duplicates. | patch |
| blind-4 | medium | Finite volume inputs can overflow accumulated profile values, violating finite-calculation expectations. Reject derived nonfinite values. | patch |
| blind-5 | medium | Doubled finite commissions may overflow and produce nonfinite trade output. Reject derived nonfinite results. | patch |
| blind-6 | medium | Repeated single derangement or zero spread is a degenerate calibration and must not imply zero detectable effect. Reject explicitly. | patch |
| blind-7 | medium | Session-close price was stamped at minute start; intrabar timestamps are not known from OHLCV. Correct known close timing and explicitly retain unknown event timing. | patch |
| blind-8 | low | Audit kind/version unchecked during evaluate. Refusal remains unconditional; add direct artifact field checks for accurate reason. | patch |
| edge-1 | high | Same hard-link overwrite as blind-1; demonstrated temp-file source truncation. | patch |
| edge-2 | medium | Intermediate symlink target can traverse protected directory despite clean final resolution. Add intermediate resolution guard. | patch |
| verification-1 | medium | Power CLI audited-data overwrite guard was disabled in memory and all tests still passed. Add collision regression for this distinct guard. | patch |
| lead-1 | medium | Open above target establishes target-first chronology; keep stop-first only for unresolved intrabar order. Add regression. | patch |
| verification-2 | medium | Replacing MDE quantile multiplier with 1.0 leaves tests passing. Add independent numeric oracle. | patch |

## Verification

`/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m pytest tests/unit/test_valentini_reclaim.py -q` from the worktree must pass. Run targeted formatter/type checks if available, without installing dependencies. Lead will run metadata audit and terminal power verdict on local files, inspect diff, and commission fresh review.

Implementation detail from exchange-source verification: explicit scheduled trading minutes must support the intraday Globex halt inside one session, retain profile/positions through it, cancel pending entries across noncontiguous minutes, and never use nonadjacent elapsed-minute volume comparisons. Synthetic halt coverage required.

Initial implementation delivered: 42 synthetic tests and targeted Black, mypy, flake8 passed per implementation report. Lead reviewed unified diff /tmp/valentini-globex-review-20260913.diff; fresh three-layer review pending. No real-data admission capability claimed.

Final verification (2026-09-14): implementation commit 530aedd5d835cbce9da7525871e14c7b5b3f2b9c merged into main; 63 synthetic tests passed in worktree and main; strict mypy/Black/flake8 passed. All 13 review entries resolved with no deferred findings. Four-file real metadata audit and terminal gate ran; power and returns were not estimated. Market-data calibrated promotion/evaluation remains explicitly unavailable, as specified. No live deployment.
