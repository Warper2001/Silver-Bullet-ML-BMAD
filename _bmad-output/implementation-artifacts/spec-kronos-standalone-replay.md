---
title: Standalone Kronos synthetic trading replay
type: feature
created: 2026-09-22
status: done
baseline_commit: 0c8c9063f51fd2c2f7d861e373a4adec1ca91073
route: dispatch
review_loop_iteration: 0
---

<frozen-after-approval>
## Intent
Implement the user's approved standalone replay plan, converting pinned pretrained forecasts into single-contract MNQ targets every eligible 15-minute close. This milestone establishes mechanics using synthetic data and three fixed comparison arms, not profitability.

## Boundaries & Constraints
Use committed mechanics preregistration. Never access real or sealed bars, credentials, live traders, services, training or downloads. Preserve smoke CLI, model pins, readiness and economic preflight. Historical tests need a separate power gate, admitted data, justified costs and frozen protocol. Mark all outputs SYNTHETIC_MECHANICS_ONLY with economic evaluation and trading authorization false.

## I/O & Edge-Case Matrix
| Scenario | Input | Result |
|---|---|---|
| Direction | 3 valid paths | Mean terminal versus observed close -> +1/-1/0 |
| Invalid forecast | Any bad path or inference exception | Flat request, retained evidence |
| Warmup | Fewer than 128 complete same-contract 15m bars | No forecasts |
| Session end | Forecast horizon crosses close | Hold last target until final-minute open flatten |
| Delayed order | Availability close + modeled latency | First minute open strictly later |
| Missing/malformed bar | Gap, duplicate, inconsistent candle | Incomplete result retaining exposure/pending |
| Roll | New contract | Restart context; abort if exposure remains |
| Position change | Repeat or reversal | Hold without costs, or close+open charging both sides |
</frozen-after-approval>

## Code Map
- tools/kronos_inference_pilot.py: reuse source/checkpoint pins, offline preparation, hash verification and validation; preserve CLI safeguards.
- tools/kronos_evaluation_preflight.py and tools/trading_model_readiness.py: retain existing gates unchanged; safe destination/digest helpers usable.
- research/kronos_replay/: new isolated package, no live imports.
- tests/test_kronos_inference*.py, tests/unit/test_*readiness.py and test_kronos_evaluation_preflight.py: regression coverage.

## Tasks & Acceptance
- [x] research/kronos_replay/engine.py: explicit minute-open sessions, incremental aggregation, execution, accounting, diagnostics. Given complete schedules and bars, when replay runs, then context is causal and all three arms share execution; future mutation never changes earlier decisions/orders.
- [x] research/kronos_replay/providers.py: ForecastProvider protocol, deterministic stub, cached adapter. Given offline assets, when forecasting repeatedly, then load once and use unchanged pins/settings/seeds, retaining all paths.
- [x] research/kronos_replay/fixtures.py and __main__.py: bundled fixture and synthetic-only CLI with manifests. Given output path, when run finishes or fails, then fresh artifacts identify synthetic scope and incomplete failures cannot report success.
- [x] tests/test_kronos_replay.py and adapter tests: hand calculations, causal boundaries, calendars, invalid data, roll and determinism. Given invalid market data, when replay aborts, then no fabricated fill or exposure reset occurs.
- [x] docs/kronos-replay.md: interfaces, timestamp/event semantics, use, limitations and next gate.

## Implementation Notes
No intent gaps or irreversible actions. Work in .claude/worktrees/kronos-replay, merge after review, reverify without service restart. Minute labels explicitly denote interval opens. Scheduled flatten is a pre-known order, independent of forecast latency. Momentum uses close versus four bars earlier with same 128-bar eligibility. Explicit schedule determines sessions; no invented holiday calendar.

Verification completed: 135 targeted tests passed (46 new replay/adapter cases and 89 existing Kronos/readiness cases); package and new tests pass flake8, package passes mypy with imported modules skipped. Three independent reviewers completed; all accepted findings fixed and tested, none deferred. Atomic projected-account validation also preserves diagnostics on numeric overflow.

Offline cached-model integration completed twice on generated bars, with 2,040 minutes, six valid paths, zero forecast errors and all arms flat at end. Forecasts, decisions, fills and equity match across runs. Final reviewed run: main checkout `_bmad-output/kronos-replay-cached-reviewed-20260922/`; manifest SHA-256 `8b585c630f9238d8f4014f120f804256ac10936778a3718f498a6478d1a38e4b`, all 14 artifact hashes and completion marker verified. No downloads, real bars, credentials, live imports, gate changes or service restarts.

## Spec Change Log

## Review Triage Log

- Verification gap 1 — medium, patch: tests checked aggregate counts but not OHLCV/amount or delivered context; a volume-sum regression could pass. Add hand-calculated recording-provider checks.
- Verification gap 2 — medium, patch: no momentum-specific assertions meant constant-zero momentum could pass. Add positive/negative/equality and exact four-bar lookback checks.
- Blind 1 — medium, patch: finite minute volumes of 1e308 overflow derived aggregates and previously complete. Reject nonfinite aggregates before admitting context.
- Blind 2 — medium, patch: large accepted slippage can create negative sell fills. Validate prospective transaction prices/accounting before mutation and retain exposure on failure.
- Blind 3 — medium, patch: shared pilot/readiness code and runtime versions were absent from provenance despite affecting output. Record hashes and versions.
- Blind 4 — medium, patch: empty CSVs lack headers and ordinary readers fail; record fixed table schemas.
- Blind 5 — medium, patch: pending causality test sliced out all differing rows, so its inputs were identical. Replace with differing complete tails and a common failure boundary retaining pending exposure.
- Blind 6 — medium, patch: same verified aggregation assertion gap as verification gap 1; one regression test addresses both findings.
- Blind 7 — medium, patch: invalid forecast tests began flat; add valid-entry-then-invalid close with costs and delayed fill.
- Blind 8 — medium, patch: latency tests used unchanging targets, leaving queued ordering/reversal/cancellation unverified. Add queued changing-target ledger test.
- Edge 1 — false in supported scope: repeated autumn DST hour occurs before NY RTH; supplied RTH schedules at 09:30 onward cannot reach the ambiguous hour. The scope is RTH, not overnight sessions; the fixture tests the changing RTH UTC offset. No overnight calendar feature added.
- Edge 2 — medium, patch: same verified aggregate overflow as blind 1; reject nonfinite derived bars.
- Edge 3 — medium, patch: fees of 1e308 per side overflow cumulative fees on exit. Validate proposed account arithmetic atomically so incomplete diagnostics remain finite with exposure intact.
- Parent lint — low, patch: remove unused test import and shorten test docstring; package type check itself passed.

## Verification
Targeted pytest for new replay and all existing Kronos/readiness tests, offline cached model run on synthetic bars, lint/type checks for new package, independent review, post-merge tests.
