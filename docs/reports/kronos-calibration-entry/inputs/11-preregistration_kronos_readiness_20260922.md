---
id: kronos_readiness_20260922
scope: readiness_only
strategy_test_permitted: false
trading_authorized: false
baseline_commit: e89a14e6d48715acd8098b7ba4bc612a3fdeeb65
---
# Kronos TradeStation SIM readiness preregistration

Committed before implementation and any capture. This milestone assesses documentary evidence, a bounded current-feed probe and conditional session-level power. No historical strategy scoring, sealed-data access, orders, training, service changes or ongoing collector is authorized.

Freeze the unchanged research/kronos_replay engine and provider at the baseline commit above: Kronos-small 901c26c1332695a2a8f243eb2f37243a37bea320; tokenizer 0e0117387f39004a9016484a186a908917e22426; upstream 67b630e67f6a18c9e9be918d9b4337c960db1e9a. Seeds 0,1,2 remain one decision ensemble, with unchanged sampling defaults. Arms: existing Kronos unanimous terminal-direction rule, four-bar momentum, always-flat. Preserve 128 same-contract completed 15-minute bars, four-bar horizon, strict-after-availability fills, contract reset and scheduled flatten. No strategy parameter changes.

Assess acquisition provenance, minute convention, completion and arrival, calendars, causal contract selection, research exposure and costs from already audited documentary reports and acquisition records. Missing evidence stays unresolved; current observations never authenticate historical files. Never reopen sealed inputs or calculate historical outcomes.

If verified RTH leaves a full capture window, request metadata for one explicit unexpired MNQ contract identified by recent successful request evidence, then poll the SIM latest three one-minute bars every five seconds for at most 900 seconds and 180 bar requests. Only required market-data GET endpoints; read existing access token without refresh or shared-state writes. Stop on authentication failure, throttling, timeout, invalid metadata, session boundary or duration cap. Preserve append-only ordered responses, request identities, provider fields, UTC receipt and monotonic times, revisions/gaps/incompleteness and descriptive delays. Otherwise record pending/blocked without starting a collector. Separately time offline pinned inference on the bundled synthetic fixture, separating startup from decision computation; no adopted latency assumption.

Future outcomes per eligible session and one MNQ: Kronos net PnL, momentum net PnL, paired difference. Both Kronos mean and incremental mean must exceed zero; economic usefulness requires independent justification. Sharpe and drawdown remain separate outcomes. Separate commission, exchange/regulatory, slippage and latency; SIM instant fills cannot validate slippage. No numerical cost or latency is adopted here.

Conditional planning: two one-sided comparisons, family alpha .05 divided equally, marginal power .90 each, at least .80 joint by union bound. Sessions are sampling units; overlapping forecasts and seeds do not increase N. Report detectable standardized effects versus N and dependence-adjusted SE. Dollar power requires independently sourced variance and useful effects, otherwise actual power UNASSESSABLE. Statistical design is not an economic threshold. No useful effect selected to fit available N.

Deliver hash-manifested evidence register, probe evidence/status, machine-readable candidate protocol, conditional power, timing, blocker list, readable HOLD_EVALUATION decision, and prospective collection specification if history remains inadmissible. All outputs deny strategy testing and trading. Verify with mocked network/credentials, existing Kronos regression tests and independent review in a worktree; merge and reverify without service restart.

## Documentary correction before capture
The frozen baseline engine (`research/kronos_replay/engine.py`, lines 339–344) takes the arithmetic mean of the three terminal closes and compares it with the last observed close. The earlier word “unanimous” is a transcription error. The baseline-code freeze governs: preserve sign(mean terminal close - observed close), including zero for equality. No code or strategy parameter has changed. This correction implements the user's explicit requirement to preserve the existing forecast-to-position rule.
