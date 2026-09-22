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


Review triage (2026-09-22; patches in separate fixes worktree while finite capture remains running):

| Finding | Verdict and evidence | Route |
|---|---|---|
| Blind1 receipt timestamp after parsing | medium: request() timestamps after sanitize/hash/parse, adding local processing. Move sampling immediately after transport. Existing capture timing remains a receipt upper bound. | patch |
| Blind2 malformed OHLC dictionaries continue | false as stated: this is a raw timing/completion observer, not price admission or scoring; missing/naive timestamps are explicitly reported ambiguous. Preserve allowed ambiguity; nonstandard JSON is separately fixed. | reject |
| Blind3 JSON NaN truncates observation | medium: json.loads accepts NaN but allow_nan=False serialization raises after file open. Reject constants and pre-serialize. | patch |
| Blind4 failed timing loses partial paths | medium: exception exits loop before paths serialized. Preserve returned/ForecastFailure paths with safe error code. | patch |
| Blind5 frozen pilot dependency omitted | medium: pilot supplies checkpoint/seed constants but was not FROZEN. Pin pilot and fixture dependencies. | patch |
| Blind6 token path can read sealed source | high: direct read_text bypasses safe_path. Guard before read with alias/path tests. | patch |
| Blind7 generic request error hides failure cause | medium: timeout/redirect/size collapse into same string. Retain safe structured category. | patch |
| Blind8 sources external to manifested output | medium: copies not in output; snapshots improve reproducibility when original paths disappear. Snapshot verified assessment sources. Probe source archive retained separately and bound by final delivery. | patch |
| Blind9 every fixed report linked to every category | medium: identities are valid but supporting locations are unspecified. Add category-specific report conclusions/key citations. | patch |
| Edge1 Unicode-escaped credential echo | high: decoded ordinary strings/keys bypass byte replacement. Recursively redact decoded token strings/keys. | patch |
| Edge2 token sealed alias | high: same verified direct-read defect as Blind6. | patch |
| Edge3 receipt after processing | medium: same verified defect as Blind1. | patch |
| Gap1 mean-terminal rule tests only unanimous | medium: current positive stub tests cannot reject voting/first-seed drift. Add mixed/negative/tie cases. | patch |
| Gap2 invalid timing outputs untested | medium: no malformed OHLC timing status/CLI test; add invalid output retention/non-success test. | patch |

Additional verification: parent found typing/lint failures in new code and extreme finite standardized-power arithmetic overflow. Fix without changing project configuration. No strategy behavior, old preflight or replay changes authorized or needed.

Follow-up review: receipt/redaction/token-path patches independently verified (19 focused tests). Reviewer found nested JSON processing could lose request metadata; patched to retain metadata with invalid-response status and safe body omission. New regression independently confirmed. All review issues requiring correction resolved; no deferred implementation findings. Probe evidence remains qualified for initial logger receipt timing and formatting-only provenance; these are retained evidence limitations, not retroactive authentication.

Final fixes verification: 201 targeted Kronos/readiness tests passed; flake8 clean; mypy clean across all seven new modules. Original capture completed180 bar GETs plus metadata in895.533s, all HTTP200; no service actions. Parent will merge verified fixes, generate final snapshot assessment and reverify main.
