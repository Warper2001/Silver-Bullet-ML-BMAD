
## Deferred from: code review of spec-lr-channel-btc-signal-module (2026-05-15)

- **`compute_lr_channel` O(n·length) Python loop** — docstring incorrectly says "vectorised"; actual implementation is a Python loop with per-iteration NumPy slice/dot. Acceptable for a ~18k-bar research backtest but will be prohibitively slow at 500k+ bars. Replace with `np.lib.stride_tricks` strided view + matrix multiply for a true vectorised implementation before this module is used in any live pipeline. [`src/research/lr_channel.py:compute_lr_channel`]
- **`denom` zero-divide for `length=1`** — `denom = L*sx2 - sx*sx = 0` when length=1; produces silent NaN. Not a real use case for this strategy (lengths 300/100/30) but could silently corrupt results if someone passes length=1. Add a `if length < 2: raise ValueError` guard. [`src/research/lr_channel.py:64`]
- **`notional_value=float(row.close)` semantic mismatch** — `DollarBar.notional_value` is designed to hold dollar notional of the bar (the $50M threshold); setting it to spot price misuses the field. Pre-existing pattern from `backtest_btc_silver_bullet.py:load_csv_as_bars`. Fix when `notional_value` is first consumed downstream. [`backtest_lr_channel_btc.py:load_csv_as_bars`]
- **Daily Sharpe uses entry_ts.date() grouping** — trades entered late in a session but exiting the next day have PnL attributed to the entry date, distorting daily variance. Standard fix: group by exit date. Acceptable simplification for exploration phase. [`backtest_lr_channel_btc.py:analyze_and_print`]
- **Non-overlapping guard is 1 bar over-conservative** — `entry_bar <= last_exit_bar` skips an entry whose fill at `entry_bar+1` would be the bar after the exit. Could be `entry_bar < last_exit_bar` to allow one more entry per cycle. Impact is a few missed signals per run. [`backtest_lr_channel_btc.py:_run_sequential:114`]

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

## Deferred from: spec-mnq-paper-trader-stop-fix (2026-05-11)

- **`risk == 0` guard is now dead code** — `paper_trade_winning_strategy.py:331` checks `if risk == 0: return` after computing risk from `abs(fvg_midpoint - stop_loss)`. With `STOP_MULT=0.75` and the `fvg_gap > 0` guard above, risk is always > 0 for any valid setup. The guard is harmless but confusing; remove when next touching this block. [`paper_trade_winning_strategy.py:331`]
- **ML integration uses wrong attribute/method names** — `paper_trade_winning_strategy.py` calls `self.ml_inference.feature_engineer` (actual: `_feature_engineer`) and `self.ml_inference.predict(features)` (actual: `predict_probability(signal, horizon)` with a different call signature). ML filtering is permanently disabled by the latch until this is fixed. Requires understanding the `SilverBulletSetup → Signal → predict_probability` call chain before re-enabling. [`paper_trade_winning_strategy.py:289-305`, `src/ml/inference.py:77,133`]

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

---

## ⚠ SPRINT 1 FAIL — INVESTIGATION REPORT (2026-05-10)

**Fold 3 date range:** May 9 – July 8, 2025 (TimeSeriesSplit(5) OOS fold 3 of 5)

**What fold 3 corresponds to:** BTC consolidation at $98k–$112k after the Nov 2024 post-election pump (+43% in ~3 weeks). Fold 3 has the highest density of "low vol" days in the entire dataset: June 2025 had 12 consecutive low-regime days as daily ATR% compressed to 1.76% avg (vs 2.61% for the full fold period).

**The regime gate DOES help in fold 3:** Fold 3 baseline PF = 0.893 (losing month) → regime-gated PF = 1.368. The gate correctly identifies that low-vol days in fold 3 are bad trading days for this setup.

**Why the holdout fails (the inversion):**

The 252-day ATR percentile window still includes the Nov–Dec 2024 extreme-volatility period for every date in the holdout (Nov 2025 – May 2026). That violent pump acts as a persistent high-water mark, causing holdout days that are "normal" volatility to be classified as "low" relative to the 2024 spike. But these holdout "low" days are productive:

| Holdout trades | N | PF | WR |
|---|---|---|---|
| Regime-gated (medium only) | 71 | **1.084** | 28.2% |
| Filtered-out (extreme+low) | 48 | **1.344** | 37.5% |
| Low-day trades specifically | 15 | **3.056** | 33.3% |

The regime gate removes the best-performing trades from the holdout. No threshold setting corrects this:

| Threshold | Holdout N | Holdout PF | Gate 1 (≥1.40)? |
|---|---|---|---|
| P20/P80 | 72 | 1.169 | FAIL |
| P10/P90 | 96 | 1.319 | FAIL |
| P5/P95 | 112 | 1.129 | FAIL |
| No filter | 119 | 1.200 | FAIL |

**Root cause:** The 252-day lookback is not appropriate for a 24/7 perpetual futures market that experienced an exceptional regime change (Nov 2024 BTC election pump). The rolling window requires ~252 days to "forget" the extreme event, during which the percentile thresholds misclassify normal-vol holdout days as "low regime." This is a structural limitation of absolute-percentile vol filtering on assets with non-stationary vol distributions.

**The baseline raw strategy has edge:** Holdout PF 1.200 without any filter. The vol regime gate as designed is the wrong instrument for this dataset and market structure.

**Decision framework verdict:** SPRINT 1: FAIL. Per framework: no deployment path. Investigate and document before proceeding.

**Paths forward (human decision required):**

1. **Sprint 4 (NY PM window, 11:00–12:00 CDT):** Independent of Sprint 1 outcome. The research doc identified 16:00–17:00 UTC as the true BTC statistical edge window; NY PM 11:00–12:00 CDT = 16:00–17:00 UTC. If NY PM holdout PF ≥ 1.40 raw (no vol filter), Sprint 1's vol gate becomes irrelevant — the kill zone change alone delivers the needed edge.
2. **Shorter adaptive window (60–90 days):** A 60-day ATR percentile would "forget" the Nov 2024 event 2 months faster. But requires spec renegotiation and retesting.
3. **Regime-relative filter (z-score vs recent mean, not absolute percentile):** Filter setups where current ATR% is > 1.5 std devs above its own 60-day mean. Adapts to the local regime rather than a global distribution.
4. **Accept raw strategy, move directly to Sprints 2–3 with reduced scope:** The raw strategy has WFE=0.51 and holdout PF=1.200. It doesn't meet the terminal gate (1.40) but it has real edge. The ML layer (Sprints 2–3) could close the 0.20 PF gap without vol filtering.

**Recommendation:** Run Sprint 4 (NY PM window) first. It answers the kill-zone question with zero code changes and no holdout contamination from the regime gate. If NY PM passes, the terminal gate is met without Sprint 1's vol filter.

---

## Deferred from: code review of spec-btc-sprint3-lr-model (2026-05-10)

- **`roc_auc_score` on full train set not guarded against single-class labels** — CV loop at `train_btc_ml.py:575` wraps the AUC call in try/except; the full-train AUC on line 629 does not. If feature extraction drops enough trades to produce a label-constant train set (all win or all loss), the call raises `ValueError`. Low probability given `MIN_TRAIN_TRADES=100`, but inconsistent with the fold-level guard. [`train_btc_ml.py:629`]
- **`atr14_100` synthetic fallback for `fill_idx` 100–113** — `compute_atr14(bars, fill_idx - 100, 14)` calls `compute_atr14(bars, 0..13, 14)`, which returns `bars[idx].close * 0.005` (synthetic fallback) when `idx < 14`. The feature extraction guard `fill_idx < 100` does not prevent this; the minimum safe value is `fill_idx >= 114`. Affects a small number of early trades; `vol_expansion` is silently distorted. [`train_btc_ml.py:extract_features`]
- **`FEATURE_COLS` is dead code after Sprint 3** — `FEATURE_COLS` (16 features) is defined but no longer used anywhere in the training pipeline; `LR_FEATURE_COLS` is now the active feature set. `extract_features()` still populates all 16 keys, so the definition is only confusing, not incorrect. Delete `FEATURE_COLS` or rename to `ALL_EXTRACT_COLS` to clarify intent. [`train_btc_ml.py:296`]
- **`threshold.json` saves unvalidated fallback threshold with no flag** — When `tune_threshold` finds no qualifying threshold (< 50 filtered trades at any threshold bucket), it returns `(0.5, 0.0)`. The saved JSON has `threshold=0.5, cv_oos_pf_filtered=0.0`. A downstream consumer loading this file cannot distinguish a validated 0.5 from the fallback. Add `"threshold_validated": false` to the JSON when `best_cv_pf == 0.0`. [`train_btc_ml.py:tune_threshold`, `train_btc_ml.py:669`]
- **`mss_to_fvg_bars` can produce negative values** — `mss_to_fvg_bars = min(fvg_idx - mss["index"], 10)` has no lower-bound clamp. If `fvg_idx < mss["index"]` (e.g. due to re-labeling), the feature goes negative. Default is `5` but live values can be negative, creating an asymmetric distribution. Fix: `min(max(fvg_idx - mss["index"], 0), 10)`. Pre-existing in `extract_features()`. [`train_btc_ml.py:394`]

## Deferred from: code review of spec-tier2-wf-adaptive-threshold (2026-05-15)

- **`fvg_fill_pct` values outside [0,1]** — actual range in `doe_run_08_fullyear_features.csv` is −14.67 to 17.0. Suggest reviewing the upstream DOE feature export to ensure correct denominator in fill-percentage calculation. Unbounded outliers are absorbed by `StandardScaler` but inflate feature variance. [`data/ml_training/doe_run_08_fullyear_features.csv`]
- **Positional row alignment between feature/history CSVs** — `load_data()` copies `year_month` from `hist` to `feat` by position (`feat["year_month"] = hist["year_month"].values`). Safe while both CSVs are always exported together in the same sort order, but brittle if either file is ever re-exported independently. Add a shared index key (e.g., signal UUID or timestamp) for join-based alignment. [`backtest_tier2_wf_adaptive.py:load_data`]
