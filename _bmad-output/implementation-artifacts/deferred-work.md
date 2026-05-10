
## Deferred from: code review of spec-btc-vol-regime-gate-sprint1 (2026-05-10)

- **Asymmetric baseline comparison due to dedup** — `execute_param` applies a one-trade-per-window-per-day dedup. Removing regime-filtered setups from some days alters which setups trigger the cap on other days, making the baseline vs. regime-gated comparison subtly asymmetric. Not a bug in the gate logic, but the PF improvement vs. baseline may be partially a dedup artifact rather than a pure regime signal. Investigate if the Sprint 1 FAIL margin is close to ambiguous. [`btc_regime_gate_backtest.py:filter_by_regime`, `optimize_btc_silver_bullet.py:execute_param`]
- **ATR fallback for first 14 bars contaminates percentile ranks 30–44** — `compute_atr14` returns `bars[idx].close * 0.005` when `idx < 14`. These artificial ATR% values enter `date_atr_pct` and participate in the percentile rank window for approximately calendar days 30–44. Days 0–29 are already guarded to "medium" by `VOL_REGIME_MIN_HISTORY=30`; days 30–44 are not guarded and see a contaminated lookback. Impact is small (~15 dates out of 500+) but could produce spurious "low" labels early in the training window. [`train_btc_ml.py:build_vol_regime_map`, `train_btc_ml.py:compute_atr14`]
- **execute_param vs execute_for_ml trade counts can diverge** — Gate 3 in `btc_regime_gate_backtest.py` uses `execute_param` trade counts; `train_btc_ml.py` uses `execute_for_ml` trade counts. These executors have slightly different fill/exit logic and may not produce identical counts on the same regime-filtered setups. A Gate 3 PASS in the gate backtest does not guarantee `train_btc_ml.py` won't `sys.exit(1)`. Acceptable for Sprint 1 (gate check is standalone by design); monitor if counts diverge by > 5% when train_btc_ml.py is run after a gate-backtest PASS. [`btc_regime_gate_backtest.py:180`, `train_btc_ml.py:524`]

## Deferred from: code review of spec-btc-ml-training-silver-bullet (2026-05-10)

- **Swing point 3-bar lookahead in features** — `detect_swing_points` confirms a swing at bar `i` by checking bars `i+1..i+3`. `extract_features` guards `s["index"] < fill_idx` so the swing's *value* is historical, but the confirmation uses future bars. Pre-existing property of the base backtest's detection function; consistent across all BTC scripts. [`train_btc_ml.py:extract_features`, `backtest_btc_silver_bullet.py:detect_swing_points`]
- **Threshold derived from pooled CV fold models applied to full-train model** — Each CV fold trains on a prefix of data; the pooled OOS probabilities come from 5 differently-sized models. The final model (full train set) may have a systematically different probability scale, making the CV-derived threshold biased. Standard limitation of CV-threshold tuning; accept for now, revisit with Platt scaling or isotonic calibration if holdout performance degrades. [`train_btc_ml.py:tune_threshold`]
- **`compute_atr14` includes the fill bar itself** — ATR range includes `bars[fill_idx]`, which is the bar being filled. In closed 1-minute bars this is fully formed; consistent with how the base backtest uses bar-level data. Would change to `idx-1` in a tick-level real-time system. [`train_btc_ml.py:compute_atr14`]
- **`time_exit` trades near dataset end silently dropped** — If `fill_idx + MAX_HOLD_BARS >= len(bars)`, the exit loop range truncates and `exit_reason` stays None; trade is skipped. Pre-existing behavior from `execute_backtest` and `execute_param`; affects only last 120 bars of the dataset. [`train_btc_ml.py:execute_for_ml`]
- **`session_time_frac` distribution shifts in CST months** — CDT offset hardcoded UTC-5; CST (Nov–Mar) is UTC-6, causing `session_time_frac` values ~1.0 higher than expected. Tied to DST handling defer (already in base backtest deferred list). [`train_btc_ml.py:extract_features`]

## Deferred from: code review of spec-btc-silver-bullet-base-backtest (2026-05-10)

- **CDT/CST DST handling** — `_in_kill_zone` and `_cdt_date` hardcode UTC-5 year-round; CST (Nov–Mar) is UTC-6, causing 1hr kill-zone shift in winter data. Spec explicitly uses "CDT = UTC-5" convention matching live paper trader; fix requires spec renegotiation and adding `zoneinfo` dependency. [`backtest_btc_silver_bullet.py`]
- **Swept-check uses close vs high/low** — `_find_next_liquidity_pool` checks `close > swing_high` instead of `high > swing_high`; matches source-of-truth MNQ backtest exactly; fix would change strategy behavior. [`backtest_btc_silver_bullet.py`]
- **Sharpe excludes no-trade days** — `analyze_performance` builds daily_pnl from trade dates only, omitting zero-P&L days; inflates annualized Sharpe slightly. [`backtest_btc_silver_bullet.py`]
- **FVG/MSS coincident at offset=0** — confluence allows FVG and MSS on the same bar; change to `range(1,11)` for stricter causality if desired. [`backtest_btc_silver_bullet.py`]
- **Kill zone breakdown omits windows with zero trades** — report only shows windows that have trades; edge case if one window fires 0 trades over the backtest period. [`backtest_btc_silver_bullet.py`]
- **swing_high `>=` allows flat-top ties** — inflates MSS count on flat-price bars; matches source-of-truth behavior. [`backtest_btc_silver_bullet.py`]

## Deferred from: code review of spec-kraken-futures-paper-trader (2026-05-10)

- **`_find_next_liquidity_pool` last-bar false swing** — rightmost bar always passes `bar.high > 0` guard (right=0 sentinel), producing an artificially close TP target. Pre-existing in `paper_trade_winning_strategy.py`. [`paper_trade_kraken.py`]
- **`_is_in_kill_zone` CDT-midnight spanning windows** — helper can't handle kill zones crossing CDT 23:59→00:00. Current 3-window config is safe; guard is latent. [`paper_trade_kraken.py`]
- **`fetch_bars` interval_ms hardcoded to 60s** — non-1m intervals return fewer bars than requested. Only 1m is used today. [`src/execution/kraken/market_data/history.py`]
- **`_preload_history` opens redundant httpx client** — bypasses `KrakenHistoryClient`; duplicate TCP connection. Low impact for paper trading. [`paper_trade_kraken.py`]
- **`run()` re-entrancy** — no guard against double-call; concurrent mutation of all shared state. Not a concern in single-process usage. [`paper_trade_kraken.py`]

## Deferred from: code review of spec-silver-bullet-backtest-logic-fix (2026-05-05)

- **`take_profit_must_respect_rr_ratio` validator removed** — 2R minimum no longer structurally enforced on `TradeOrder`. Subsumed by AC2 fix (recomputing proper swing targets enforces 2R naturally). [`src/execution/models.py`]
- **`sync_quantities` validator silently corrupts partially-closed 1-contract positions** — validator only fires when `original_quantity==1 AND quantity>1`; passing `original_quantity=1` as a real "one contract" value trips the condition on multi-contract mutations. Pre-existing design fragility. [`src/execution/models.py:55–61`]
- **Datetime timezone mismatch** — `_now()` returns NY-aware datetimes; E2E test constructors use naive `datetime.now()`. Will raise `TypeError` on any path calling `TradeOrder.is_held_max_time()` directly. Pre-existing test infrastructure issue. [`tests/e2e/test_ensemble_e2e.py`]
- **`for/else` confidence tier loop fragile** — accidentally correct today because Tier 5 upper bound is 1.01 not 1.00, but will silently misbehave if `CONFIDENCE_TIERS` is ever reset. Revisit if tiers change. [`src/execution/entry_logic.py:73–81`]
- **`calculate_rr_achieved` always returns −1.0 for losses regardless of magnitude** — analytics corruption (a 5R adverse move and a 1-tick SL hit both report −1.0); no impact on trading decisions. [`src/execution/exit_logic.py`]

---

## ⚠ OPEN PROBLEM: ML filter does not achieve objective (2026-05-04)

**Finding:** Full-year 2025 backtest with deployed model (spec-6-6 Pipeline(StandardScaler+LR), threshold=0.52) produces PF 1.207 — indistinguishable from the raw unfiltered PF of 1.217. The filter passes only 9% of trades (87 / 1,019), reducing annual P&L from $6,386 to $1,133. The OOS claim of PF 2.31 (spec-6-6 AC5) was based on 17 filtered trades on the same fold used for threshold selection — statistically meaningless.

**Root cause:** AUC 0.5662 is insufficient to produce a spread in predicted probabilities wide enough for threshold selection to be meaningful. 91% of trades score between 0.50–0.52; the model has near-zero discriminative power at the operating threshold.

**Objective not met:** Epic 6 goal was ML filter that improves Profit Factor over 1.15 baseline on a held-out period. Actual result on full-year application: PF unchanged (1.207 vs 1.217 raw).

**Current best deliverable:** Raw Tier 2 strategy, no ML filter. PF 1.217, 1,019 trades/year, ~$30/day, ~$6,400/year on one MNQ contract.

**Path forward:** See `_bmad-output/planning-artifacts/research/technical-tier-2-ml-profit-factor-improvement-research-2026-05-01.md` — Post-Implementation Empirical Findings section. Minimum requirements for next attempt: AUC ≥ 0.60, pass rate ≥ 30% at operating threshold, threshold validated on a fully held-out period with no prior decisions made on it.

---

## Deferred from: code review of spec-6-3-tier-2-live-ml-filter (2026-04-30)

- **`_session_open_price` undefined before 06:00 ET** — pre-market bars cause `session_displacement = 0.0` fallback; not spec-6-3 scope. [`tier2_streaming_working.py:258-259`]
- **`_calculate_atr` called twice per detection** — redundant, no correctness impact; refactor when optimizing. [`tier2_streaming_working.py:417, 474`]
- **`.loc[i]` vs `.iloc[i]` in swing detection** — works after `reset_index(drop=True)` but fragile; low risk. [`tier2_streaming_working.py:315-317`]
- **No test for sweep-expiry path** — expiry logic untested; add in follow-up test expansion. [`tests/unit/test_tier2_ml_filter.py`]
- **`_calculate_atr` hardcoded 10.0 fallback** — guarded by 20-bar minimum; no current correctness issue. [`tier2_streaming_working.py:480`]
- **SL before TP same-bar pessimistic bias** — pre-existing pattern, deferred in prior reviews. [`tier2_streaming_working.py:360-373`]
- **`prior_setup_proximity` no lower-bound** — cannot go negative in normal flow; theoretical edge case. [`tier2_streaming_working.py:441`]
- **O(N) DataFrame rebuild on every bar** — explicitly deferred in spec-6-3 original review. [`tier2_streaming_working.py:286`]
- **`vol_ratio` sentinel 99.0 out-of-distribution** — consistent with backtest training; revisit if model performance degrades. [`tier2_streaming_working.py:426-429`]
- **`predict_proba` pass-through logs P(Success)=1.000** — misleading metric but spec says "trade proceeds"; cosmetic. [`tier2_streaming_working.py:105-107`]
- **AC2 end-to-end latency test missing** — unit test covers dominant latency; full-path integration test out of scope. [`tests/unit/test_tier2_ml_filter.py`]

## Deferred from: code review of spec-6-3-tier-2-live-ml-filter (2026-04-29)

- **O(N²) rebuild in `_update_h1_structure`** — `pd.DataFrame([vars(b) for b in self.dollar_bars])` rebuilds entire history every 1m bar; O(N²) total; cache H1 state and only recompute on H1 boundary crossing. [`tier2_streaming_working.py:217`]
- **`dollar_bars` grows unbounded** — no max-length cap; combined with O(N²) rebuild causes OOM after days of polling. Add eviction: `self.dollar_bars = self.dollar_bars[-MAX_BARS:]`. [`tier2_streaming_working.py:187`]
- **SL/TP check order in `simulate_trade` (consistently pessimistic)** — SL is always checked before TP; minor systematic pessimistic bias in backtest statistics; consistent across directions. [`backtest_zero_bias_optimized.py:129`]
- **`simulate_trade` time-exit price clamp** — `closes[min(start_idx + MAX_HOLD_BARS, n-1)]` biases time-exits toward terminal dataset price; minor backtest artifact, pre-existing. [`backtest_zero_bias_optimized.py:140`]
- **No walk-forward validation in training** — Single 80/20 split used; project has `WalkForwardOptimizer` available. Deliberate for small 672-sample dataset; document rationale explicitly. [`train_tier2_meta_labeling.py`]
- **DST gap in `_is_market_open` Sunday** — `h >= 23` is correct for EST but 1 hour late in EDT; minor; pre-existing pattern in codebase. [`tier2_streaming_working.py:160`]

## All HTF Grid Scripts — Lookback Window Boundary (> vs >=)
All three grid scripts (directional presence, momentum, spatial filter) use `close_time > window_start` (strict inequality), meaning a parent FVG that closes exactly at the window boundary is excluded. This makes "LB3" effectively mean "FVGs formed in the last 2.x bar-durations". Investigate whether changing to `>=` materially affects top combos; likely low impact but worth verifying once all grid variants are complete.

## All HTF Grid Scripts — Entry Look-Ahead Design
All three grid scripts derive `entry` from `c3_low`/`c3_high` of the current bar `i` and compare against parent FVGs with `close_time <= bar_ts` (where `bar_ts` is also bar `i`'s timestamp). This means signals and parent FVGs share the same bar boundary. In live trading, the entry would be taken on the *next* bar open. This is a known design simplification shared by all grid scripts — revisit when translating top combos to paper trading.

## Directional Presence Grid — LB40 OR pass-through effect
LB40 OR combos on wide TFs (89-min) produce 0% filter rate — the lookback window (40×89=3,560 min ≈ 59 hours) covers nearly all available history, so virtually every signal finds a matching parent FVG. These combos are numerically indistinguishable from the unfiltered baseline. Future research should cap effective lookback or test shorter windows (LB5, LB10 on the 89-min TF) to ensure the filter is actually discriminating.

## Deferred from: code review of spec-6-5-tier-2-ml-feature-phase2-pf-optimization (2026-05-01)

- W1: elif in _detect_and_enter blocks bearish signals when bullish sweep active — live trade count diverges from backtest [tier2_streaming_working.py:409]
- W2: simulate_trade evaluates SL before TP on same bar — always SL on bars spanning both levels; pessimistic bias [backtest_zero_bias_optimized.py:193-198]
- W3: Swing detection differs between backtest (all() window) and live (2-bar check) — inconsistent swing point sensitivity [backtest_zero_bias_optimized.py:42-46, tier2_streaming_working.py:329]
- W4: Transaction cost $1.80 backtest vs $0.80 live — paper P&L will appear $1/trade better than benchmark [backtest_zero_bias_optimized.py:33, tier2_streaming_working.py:48]
- W5: _update_h1_structure O(n) rebuild every bar; no cap on dollar_bars list — latency grows unboundedly over multi-week runs [tier2_streaming_working.py:296]

## Deferred from: code review of spec-6-6-tier2-doe-sample-generation-lr-model (2026-05-03)

- **W1: DOE subprocess has no timeout** — crashed backtest indistinguishable from failed-gate run; returns gate_pass=False silently. Research/offline tool, not production path. [src/research/doe_runner.py]
- **W2: Float equality for `atr_threshold` in `compute_main_effects()`** — fragile float comparison `summary[summary[col] == level]` on values like 0.10, 0.25, 0.50. Research/offline tool. [src/research/doe_runner.py]
- **W3: No AC5 verification report persisted** — AC5 validated interactively during dev; no artifact in data/reports/ demonstrates prior validation for future audits. [data/reports/]

## Deferred from: spec-paper-trader-sim-order-submission (2026-05-06)

- **Order ID parsing via Message string match** — `_submit_sim_bracket` parses entry/TP/SL order IDs by matching "Stop Market" / "Limit" in the `Message` field. Fragile if TradeStation changes message casing or format. Pre-existing pattern from `tier2_streaming_working.py:650–655`. Consider switching to `OrderType` field parsing when API docs confirm field names.
- **Market entry vs limit entry divergence** — SIM bracket uses a market order; local sim assumes FVG midpoint limit fill. SIM P&L will differ from local sim by slippage. Acceptable for paper trading comparison, but document the divergence when reporting.
- **httpx client per call** — `_submit_sim_bracket` creates a new `AsyncClient` per submission. Pre-existing pattern in this file. Consider sharing a session-level client if order frequency increases.
- **No env guard on SIM_ACCOUNT_ID** — account ID is hardcoded. The `sim-api.tradestation.com` base URL is the environment guard. Consistent with tier1/tier2 pattern.
