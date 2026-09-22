---
title: Standalone Kronos synthetic trading replay
type: feature
created: 2026-09-22
status: ready-for-dev
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
- [ ] research/kronos_replay/engine.py: explicit minute-open sessions, incremental aggregation, execution, accounting, diagnostics. Given complete schedules and bars, when replay runs, then context is causal and all three arms share execution; future mutation never changes earlier decisions/orders.
- [ ] research/kronos_replay/providers.py: ForecastProvider protocol, deterministic stub, cached adapter. Given offline assets, when forecasting repeatedly, then load once and use unchanged pins/settings/seeds, retaining all paths.
- [ ] research/kronos_replay/fixtures.py and __main__.py: bundled fixture and synthetic-only CLI with manifests. Given output path, when run finishes or fails, then fresh artifacts identify synthetic scope and incomplete failures cannot report success.
- [ ] tests/test_kronos_replay.py and adapter tests: hand calculations, causal boundaries, calendars, invalid data, roll and determinism. Given invalid market data, when replay aborts, then no fabricated fill or exposure reset occurs.
- [ ] docs/kronos-replay.md: interfaces, timestamp/event semantics, use, limitations and next gate.

## Implementation Notes
No intent gaps or irreversible actions. Work in .claude/worktrees/kronos-replay, merge after review, reverify without service restart. Minute labels explicitly denote interval opens. Scheduled flatten is a pre-known order, independent of forecast latency. Momentum uses close versus four bars earlier with same 128-bar eligibility. Explicit schedule determines sessions; no invented holiday calendar.

## Spec Change Log

## Review Triage Log

## Verification
Targeted pytest for new replay and all existing Kronos/readiness tests, offline cached model run on synthetic bars, lint/type checks for new package, independent review, post-merge tests.
