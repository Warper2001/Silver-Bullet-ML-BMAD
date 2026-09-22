---
title: Kronos readiness for TradeStation SIM evaluation
type: feature
created: 2026-09-22
status: in-review
route: dispatch
baseline_commit: e89a14e6d48715acd8098b7ba4bc612a3fdeeb65
review_loop_iteration: 0
context: []
---
<frozen-after-approval>
## Intent
Deliver a reproducible, documentary readiness verdict for a future Kronos real-data replay targeting TradeStation SIM. The user approved implementation of the supplied plan. This milestone assesses evidence, a bounded current-data probe and conditional power; it cannot authorize strategy scoring or trading. The ultimate economic requirements are positive Kronos net expectancy and improvement over momentum.

## Boundaries & Constraints
Always work in `/root/Silver-Bullet-ML-BMAD/.claude/worktrees/kronos-readiness`. The readiness preregistration is already committed there before implementation. Preserve the synthetic replay and preflight unchanged. Use new `research/kronos_readiness` modules, tests and documentation only. No historical prices or sealed inputs; documentary reports only. No orders, services, shared auth refresh/writes or installation. All report/protocol/status outputs retain strategy_test_permitted=false and trading_authorized=false. Inputs lacking proof remain unresolved, current observations cannot authenticate historical data. Parent handles official-source archiving and actual command execution; implement commands and tests, leaving these run tasks to parent.

Freeze model/policy/three arms from the preregistration. A future candidate protocol preserves 128-bar same-contract warmup, 15-minute closes, four-step forecasts, unanimous terminal-direction rule, four-bar momentum, flat control, reset behavior, strict-after-availability fills and scheduled flattening. Costs remain separated and unadopted. Per-session one-contract outcomes are K net, M net and K-M; both means >0 required. Sharpe/drawdown separately reported in future. No historical outcome calculation.

## I/O & Edge-Case Matrix
| Scenario | Input | Behavior |
|---|---|---|
| Evidence absent/tampered | source missing or hash mismatch | explicit unresolved/refusal; no admission |
| Completion ambiguous | missing status or timezone | flag ambiguity, never infer completed from age |
| Revisions/gaps | repeated timestamp changing fields, missing minutes | retain every ordered observation, descriptive report |
| Contract mismatch/expiry | metadata differs/expired | stop before bars |
| RTH/DST/early close | verified explicit dated session evidence | correct UTC boundaries; outside/full window unavailable pending without token read |
| Auth/throttle/timeout | mocked failures | stop without refresh/retry or secret disclosure |
| Bounds | monotonic limit/180 requests | stop by 900s; poll every 5s; never catch up bursts |
| Forbidden endpoint | nonapproved URL/method/redirect | refuse network |
| Missing effect/variance | absent independent evidence | actual power UNASSESSABLE; standardized planning only |
</frozen-after-approval>

## Code Map
- `research/kronos_replay/{engine,providers,fixtures,__main__}.py`: unchanged mechanics, CachedProvider, bundled_fixture and artifact helpers. CachedProvider default cache is repo-relative; command should accept explicit main-checkout cache path for worktree runs.
- `tools/kronos_evaluation_preflight.py`: fixed EVIDENCE documentary hashes; do not modify. Reports may be untracked in worktree, so accept explicit documentary root pointing main checkout, strictly allowlist and verify these reports.
- `tools/trading_model_readiness.py`: safe_path blocks sealed data; audit findings contain provenance and minute-label ambiguity.
- `tools/kronos_inference_pilot.py`: pinned hashes/CPU/seeds; do not import live auth.
- `src/data/auth_v3.py`: plain .access_token convention only; never instantiate. Read once using Path, no writes, redaction of token in any response/error.
- Main `logs/mim_nb_live.log`: parent archives recent successful GET MNQZ26 evidence. Metadata endpoint `/v3/marketdata/symbols/MNQZ26`; bars `/v3/marketdata/barcharts/MNQZ26?interval=1&unit=Minute&barsback=3` at sim-api.tradestation.com only. Explicit unexpired candidate metadata required.

## Tasks & Acceptance
- [ ] `research/kronos_readiness/{__init__,__main__,evidence,protocol,power,probe,timing}.py` -- implement documentary assess, finite probe and offline synthetic timing commands. Fresh destination, hash manifest, readable report and immutable append-only observations. Keep interfaces simple and document CLI usage. Parent supplies archived source pack: flexible evidence source entries each with path/hash/date/url/claim, never treated as automatically proving every gap. No price file inputs in assessment.
- [ ] `tests/test_kronos_readiness*.py` -- mocked tests for matrix, guarded endpoints, token unchanged/redaction, power calculations/dependence and admission distinction.
- [ ] `docs/kronos-readiness.md` -- commands, assumptions, exact evidence needed next and prospective collection specification without launching collector.
- [ ] Parent: archive official fees/calendar/API sources and acquisition evidence, run actual finite probe during verified RTH or retain pending/blocked, run pinned synthetic timing, generate final manifested evidence and decision.
- [ ] Parent: independently review/fix, merge and rerun tests without restarting services.

Acceptance: Given existing reports and missing historical evidence, when assessed, then HOLD_EVALUATION and actual power UNASSESSABLE identify each blocker and next evidence required. Given synthetic fixture and pinned assets, when timing runs, then startup and each three-seed decision duration are separate, with no adopted latency. Given verified session and recent explicit contract evidence, when probe runs, then metadata validation precedes up to 180 bar GETs within 900s, raw response order/provider fields and request/receipt UTC/monotonic times persist; all failure conditions stop safely. Given no independently justified dollar effect/variance, conditional normal planning uses n sessions and SE inflation >=1, alpha=.025 per test, marginal power=.90; detectable standardized effect=(z(.975)+z(.90))*inflation/sqrt(n), joint lower bound=.80, never actual admission. Dollar calculations require evidence independently supporting both mean effects and variances, and paired variance uses covariance, never treats arms/seeds as independent samples.

## Implementation Notes
No intent gaps; read-only market-data GET authorized. Parent independently gathers sources while implementation proceeds. Use main `.venv/bin/python` for tests and `.venv-research/bin/python` for model timing; do not create dependencies. Full canonical prereg commit obtained by parent after commit. Child should implement all code/doc/tests and report results, not commit or run real network/capture. Parent runs/archives.

## Spec Change Log

## Review Triage Log

## Verification
`/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m pytest tests/test_kronos* tests/unit/test_kronos_evaluation_preflight.py tests/unit/test_trading_model_readiness.py` must pass. Independently review code and execute CLI artifact verification.

Planning correction: the engine uses sign(mean of three terminal closes minus observed close), not unanimity. Preserve the existing engine per user intent; the earlier descriptor is an agent transcription error. Preregistration appended correction before capture. No strategy parameter change.
