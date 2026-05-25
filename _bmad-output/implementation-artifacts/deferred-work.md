
## Deferred from: code review of 8-5-prereg-yaml-workflow (2026-05-25)

- **No test for YAML-file-missing at `oos_checkpoint` verification time** — `run_checks()` catches `FileNotFoundError` correctly but no test exercises that path; symmetric test exists on the seal side (`test_seal_yaml_config_missing_file_returns_1`). [`oos_checkpoint.py:140-144`]

## Deferred from: code review of 8-4-rolling-weekly-backtest (2026-05-25)

- **`--weeks` window anchored to last data row's timestamp, not today's date** — if the data file hasn't been refreshed recently, the window silently narrows; user expects N weeks from today. [`tools/weekly_backtest.py:_load_and_filter`]
- **`epilog` crashes with `-OO` Python flag** — `__doc__` is `None` when Python strips docstrings; `__doc__.split()` raises `AttributeError`; fix is `epilog=(__doc__ or "").split(...)`. [`tools/weekly_backtest.py:main():264`]

## Deferred from: code review of 8-3-multi-instrument-support (2026-05-25)

- **Crash recovery half-implemented: `StatePersistence` saves state on entry but `load_state()` never called on `initialize()`** — on crash, dangling bracket orders persist at TradeStation while the bot restarts with no active trade; Epic 4 Story 4-2 is the correct home for this fix. [`tier2_streaming_working.py:153, initialize()`]
- **`detect_fvg` hardcoded to MNQ `POINT_VALUE_USD=2.0` for dollar-ceiling gate** — MES ($5/pt) gaps are up to 2.5× more expensive in dollar terms; the `max_gap_dollars` filter is structurally MNQ-only; fix requires a `point_value` parameter in `detect_fvg` (AR1-protected strategy_core change). [`strategy_core.py:359`]
- **`commission_per_roundtrip=4.0` not scaled per instrument** — MES (2 contracts) incorrectly charged $4.00 instead of $1.60; acknowledged in Story 8-3 dev notes as "per-roundtrip, not per-contract"; accepted for now. [`strategy_core.py:98`]
- **CSV header TOCTOU race in `_log_trade`** — `write_header = not log_path.exists()` evaluated before open; two instances starting simultaneously can both write headers; fix is to check `file.tell() == 0` inside the open context. [`tier2_streaming_working.py:802`]
- **`_bar_processing_times` list grows unboundedly** — one append per bar, never trimmed; fix is `collections.deque(maxlen=10000)` in `__init__`. [`tier2_streaming_working.py:438`]
- **Symbol expiry codes require manual update each quarterly rollover** — `MNQM26/MESM26/M2KM26` reject new expirations (e.g., `MNQU26`) with `ValueError`; operational concern, no code fix needed now.
- **`_log_trade` drops `entry_time`** — parameter accepted but never written; only `exit_time` is recorded as `timestamp`; AC#8 doesn't specify entry vs exit; deferred pending clarification.

## Deferred from: code review of 8-2-yaml-config-externalization (2026-05-24)

- **Time-parser inconsistency: `config_loader.py` uses `split(":")` / `time(int,int)` while `prereg_seal._build_config()` uses `time.fromisoformat()`** — both handle the expected "HH:MM" format correctly; `config_loader.py` is actually more robust for single-digit hours (e.g. "9:30"); `prereg_seal._build_config()` fromisoformat would fail on "9:30" in the legacy `--config-json` path. Low priority since the YAML workflow always goes through `config_loader.py`. [`src/research/config_loader.py:51`, `prereg_seal.py:87`]
- **HASH_PATTERNS `hash_a` regex too broad** — `r"\|\s*\(a\)[^\|]+SHA-256\s*\|\s*\`([0-9a-f]+)\`"` matches any `(a) *SHA-256` row; alternation `(StrategyConfig|YAML config)` would be more precise. Low practical risk given controlled doc format. [`oos_checkpoint.py:34`]
- **Misleading FAILED message when `--config` omitted on YAML-workflow prereg** — user sees "StrategyConfig has been modified since pre-registration seal" when the real issue is that `--config strategy_config.yaml` was forgotten. Auto-detecting from the `(a) YAML config SHA-256` label in the doc is the fix. [`oos_checkpoint.py:run_checks`]
- **`yaml.safe_load(path.read_text())` without `encoding="utf-8"` in config_loader** — low probability of failure on UTF-8 Linux but not portable. Also affects `oos_checkpoint.py`'s prereg doc read. Fix: pass `encoding="utf-8"` to `read_text()`. [`src/research/config_loader.py:42`, `oos_checkpoint.py`]
- **`load_strategy_config` returns StrategyConfig defaults silently on empty/null YAML** — `if not raw: return StrategyConfig()` gives no warning; a truncated YAML file would silently use defaults. Behavior is technically correct (hash still round-trips). Add `logger.warning()` on next touch. [`src/research/config_loader.py:44`]

## Deferred from: code review of 8-1-s25-config-deployment-g025-m15-choch (2026-05-24)

- **`tuesday_exclusion` config field silently ignored in live trader** — `_detect_and_enter()` line 839 has `if bar_et.weekday() == 1: return` hardcoded; `self._strategy_config.tuesday_exclusion` is never read by `Tier2StreamingTrader`. Toggling the YAML field would have zero effect on the live system. Only `BacktestEngine` reads the config field. Fix by replacing the hardcoded check with `if cfg.tuesday_exclusion and bar_et.weekday() == 1: return`. [`tier2_streaming_working.py:839`]
- **AC5: BacktestEngine has no S25 CHoCH state machine** — The N=61 verification backtest used the no-CHoCH 15m baseline (S13 reference). The S25 CHoCH logic (`_update_m15_choch()`) lives only in `Tier2StreamingTrader`; `BacktestEngine` has only `m15_confirmation` (bar-close direction check), which is a different filter. Confirming S25 CHoCH behavior via backtest requires porting the CHoCH state machine to `BacktestEngine`. Future story. [`src/research/backtest_engine.py`, `src/research/tier2_streaming_working.py:641`]
- **AC7: Three test files have pre-existing collection errors** — `tests/unit/test_resource_monitor.py`, `test_tier2_ml_filter.py`, `test_performance_documentation.py` fail at pytest collection; `pytest tests/ -q` exits with collection errors before running. Must exclude these files or fix their import/syntax issues for AC7 to be satisfiable as written. Pre-existing since before Story 8-1. [`tests/unit/`]
- **AC1: No dedicated unit test for `StrategyConfig().min_gap_atr_ratio == 0.25`** — The only test covering this value loads the repo-root YAML file (test_config_loader.py), not the bare dataclass default. A low-cost assertion would be: `assert StrategyConfig().min_gap_atr_ratio == 0.25`. [`tests/unit/`]

## Deferred from: code review of 3-2-pre-registration-document-generator-prereg-seal (2026-05-24)

- **`_git_is_dirty()` detects the output doc itself as "dirty"** — On re-seal runs, the previously generated `_bmad-output/preregistration_*.md` shows as a modified or untracked file, causing a WARNING on every invocation. Expected workflow behavior: user seals → commits the doc → clean tree. Not a bug. [`prereg_seal.py:58-66`]
- **Silent overwrite of existing sealed document** — Re-running with the same `--name` silently replaces the doc. No guard or error. Risk: accidentally overwriting a committed pre-reg during OOS period. Add `if output_path.exists(): print("ERROR: seal already exists"); return 1` if reproducibility audit requires. [`prereg_seal.py:189`]
- **`_config_to_json()` value-mutation during `d.items()` iteration** — Replaces `time` values in-place while iterating. CPython 3 allows value mutation without structural change; no crash observed. Use `for k in list(d)` if `StrategyConfig` ever gains nested sub-dict `time` fields. [`prereg_seal.py:29-31`]
- **`_build_config()` raises `AttributeError` on unknown JSON fields** — `defaults.update(overrides)` adds unknown keys; `getattr(base, k)` then raises `AttributeError`. No user-friendly message. Add field validation before update. [`prereg_seal.py:75`]
- **`date.today()` is local wall-clock, not UTC** — Sealed date may differ from UTC by ±1 day near midnight. Use `datetime.now(timezone.utc).date().isoformat()` for audit-grade timestamps. [`prereg_seal.py:127`]
- **`output_path.write_text(doc)` uses system default encoding** — Should be `write_text(doc, encoding="utf-8")` for portability across locales. Harmless on UTF-8 Linux systems. [`prereg_seal.py:189`]
- **`_extract_holdout_dates()` silently ignores files with no 8-digit date in stem** — Unnamed holdout CSVs (e.g., `holdout_data.csv`) are excluded from date range with no warning. Acceptable given current naming convention. [`prereg_seal.py:39-44`]

## Deferred from: code review of 3-1-sealed-holdout-directory-and-protect-holdout (2026-05-24)

- **`chmod 444` bypass via root user** — Running as root bypasses DAC; chmod 444 does not prevent root writes. Documented and accepted per AC#3. If ever deployed as non-root, no fix needed. [`protect_holdout.py:57`]
- **`init()` applies no date validation before protecting** — `--init` chmods all CSVs regardless of date; subsequent `--verify` would then fail date check. AC#1 does not require init to date-validate; by design. [`protect_holdout.py:62-82`]
- **`_extract_date` regex matches first 8-digit run** — Could pick wrong date on filenames with multiple long numeric sequences. Does not affect actual holdout filename `mnq_1min_holdout_20260301_plus.csv`. Fix anchor regex with `(?<!\d)...(?!\d)` if naming convention changes. [`protect_holdout.py:23`]
- **`init()` returns 0 with zero CSVs** — `INIT PASS — 0 CSV(s) protected` is misleading when no data files exist. Edge case not covered by spec. Add exit 1 + warning if needed. [`protect_holdout.py:79`]
- **No error handling in `init()` around `os.chmod()` / `open()`** — Python raises `PermissionError` with traceback on failure; no recovery path. Acceptable for a CLI tool. [`protect_holdout.py:70,78`]
- **No subprocess test for process-restart durability (AC#4)** — `test_init_verify_roundtrip` runs in-process; smoke tests in Task 3 cover the cross-process case. chmod persistence is OS-level. [`tests/unit/test_protect_holdout.py`]
- **`ACCESS_LOG.md` mutable and untamper-evident** — Intentional per AC#6; must stay writable for future append operations. Would require cryptographic signing to make tamper-evident. [`protect_holdout.py:18`]
- **`verify()` inconsistent early-exit** — Date check fast-fails on first bad file; permission check collects all offenders. Style inconsistency, not a correctness bug. [`protect_holdout.py:35-52`]
- **`verify()` identical error for non-existent vs empty dir** — Both print "no CSV files found". Holdout dir always exists in practice. [`protect_holdout.py:31`]
- **`--init --verify` combined flags runs only `--init`** — `elif` routing silently skips verify. Users expected to use one flag at a time. [`protect_holdout.py:93-98`]

## Deferred from: code review of 2-4-volatility-regime-gate-parameterization-relaxed-filter-constants (2026-05-24)

- **AC #4 pending-timeout not behaviorally verified** — `vol_regime_15m_test.py` prints `RELAXED_CONFIG.max_pending_bars = 120` as confirmation but does not assert that a pending order placed at bar 0 is actually cancelled at bar 121. The BacktestEngine `>= config.max_pending_bars` gate is covered by the broader test suite. Add a dedicated synthetic-scenario assertion if timeout boundary verification is required for audit. [`src/research/vol_regime_15m_test.py`]

## Deferred from: code review of 2-3-m15-confirmation-layer-and-resample (2026-05-24)

- **Look-ahead bias in M15 gate for 1-min bar input** — `searchsorted(bar_ts)` on the pre-computed M15 index returns the next M15 bucket position when `bar_ts` falls mid-period; the current (incomplete) M15 bar is included and its `close` contains future data. Only affects 1-min input; the 15m bar path (used in research) is correct. Fix before calling `BacktestEngine` with 1-min bars and `m15_confirmation=True`. [`src/research/backtest_engine.py:830`]
- **`m15_confirmed=True` for all trades when `m15_confirmation=False`** — `m15_ok=True` by default; when the gate is disabled, all trades record `m15_confirmed=True`, making the field ambiguous in CSV output vs. a gate-enabled confirmed trade. Consider using `None` or a separate `m15_filter_enabled` column. [`src/research/backtest_engine.py:828`]
- **No integration test for 1-min bars with `m15_confirmation=True`** — look-ahead bias has zero test coverage; 15m bar test masks the bug. Add a BacktestEngine integration test with 1-min synthetic bars to validate the M15 slice is computed without look-ahead. [`tests/`]

## Deferred from: code review of 2-2-am-kill-zone-filter-dst-aware (2026-05-24)

- **`min_gap_atr_ratio` default mismatch** — `StrategyConfig` default is 0.25; CLAUDE.md documents live value as 0.15. Backtest and live system differ silently. Fix when strategy params are consolidated before live deployment. [`src/research/strategy_core.py:92`]
- **`make_entry_decision` no-kwargs bypass** — calling `make_entry_decision(sweep, fvg, config)` without `vol_ok` always passes the volatility filter vacuously. Research scripts may inadvertently bypass the vol gate. Audit all callers before live use. [`src/research/backtest_engine.py`]
- **`save_outputs()` empty-trades path** — writes equity curve CSV but `pnls` is empty; `list(itertools.accumulate([]))` returns `[]` which is harmless, but the equity file is created with only a header. Confusing artifact when verdict is H₀. [`src/research/kz_15m_test.py`]
- **`kill_zone_filter` TypeError on tz-naive timestamp** — `bar_timestamp.astimezone(_NY_TZ)` raises `TypeError` for tz-naive input. All BacktestEngine callers pass UTC-aware timestamps so this is unreachable in practice; add a guard before exposing in a library context. [`src/research/strategy_core.py:584`]
- **`_H1_BUFFER_BARS=7500` thin margin** — 7500 1-min bars = 125 H1 bars; minimum for H1 sweep detection is 120. Only 5-bar slack; a longer gap in the data could exhaust the buffer. Increase to 10,000 (≈167 H1 bars) when next touching backtest_engine. [`src/research/backtest_engine.py`]
- **`_compute_vol_pct` duplicates `volatility_regime_filter` internals** — rolling ATR percentile computed twice with slightly different window configs; diverges from StrategyConfig params. Refactor to call `volatility_regime_filter` directly. [`src/research/backtest_engine.py`]
- **Empty CSV → unhandled ValueError from `resample_to_h1`** — `pd.DataFrame.resample().agg()` on empty input raises ValueError; BacktestEngine has no guard. Fix before using BacktestEngine in automated pipelines. [`src/research/backtest_engine.py`]
- **Dead code `if i < 2: continue`** — this guard is shadowed by `if i < 20: continue` later in the same loop; the `< 2` branch never executes. Remove on next cleanup pass. [`src/research/backtest_engine.py`]
- **Pending timeout falls through to same-bar entry** — when a pending order times out, the code clears `active` and falls through to the entry-detection block on the same bar, potentially arming a new pending order immediately. Intentional per inline comment but creates a subtle sequence dependency. [`src/research/backtest_engine.py`]

## Deferred from: code review of 3-4-oos-verdict-report-generator (2026-05-24)

- **`_parse_prereg` called twice in `verdict()`** — `checkpoint_or_abort` internally calls `_parse_prereg`; `verdict()` calls it again to extract `hash_c`. Redundant and fragile: if the second call returns different results (unlikely race condition or malformed doc), "unknown" gets logged in ACCESS_LOG. Fix when `checkpoint_or_abort` is refactored to return its parsed hashes, or expose `_parse_prereg` result from the checkpoint call. [`oos_verdict.py:209`]

## Deferred from: code review of 3-3-oos-checkpoint-verification (2026-05-24)

- **`_config_to_json` only converts top-level `time` fields** — nested dataclass `time` values survive unconverted; `json.dumps` would raise `TypeError` if StrategyConfig ever gains a nested dataclass with a `time` field. Latent — no nested dataclasses today. Fix when StrategyConfig structure changes. [`oos_checkpoint.py:41–44`]
- **`_git_is_dirty` false-passes outside a git repo** — `git status --porcelain` exits non-zero with empty stdout when not in a repo; `bool("")` is `False`, so the function reports "clean" incorrectly. Unrealistic scenario: this script is always run from repo root. [`oos_checkpoint.py:59–63`]
- **`protect_holdout.verify()` not mocked in tests** — tests pass a `tmp_path` holdout dir to the real `verify()`, which checks date-cutoff logic (`HOLDOUT_CUTOFF = "2026-03-01"`). Test CSV named `mnq_1min_holdout_20260301_plus.csv` satisfies this accidentally. A bump to `HOLDOUT_CUTOFF` in `protect_holdout.py` would silently break these tests. Consider mocking `protect_holdout.verify` directly. [`tests/unit/test_oos_checkpoint.py`]
- **Hash regex accepts any-length hex** — `HASH_PATTERNS` uses `[0-9a-f]+` (no length constraint); a truncated hash parses silently and fails check with "mismatch" rather than "malformed hash". Comparison still fails correctly; user just sees a less helpful error. Add `{64}` quantifier for SHA-256 or an explicit length guard. [`oos_checkpoint.py:33–37`]

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

## Deferred from: code review of 2-1-bidirectional-fvg-detection-remove-bearish-only (2026-05-23)

- **`calc_sharpe` single-day edge case** — if all trades fall on one calendar day, `std` of a single-element list returns 0 or NaN; no guard before `calc_sharpe` call. Pre-existing pattern shared with `timeframe_replication.py`; low practical risk for 15m training window with 81 trades spread over 2025. [`src/research/bidir_15m_test.py:79`]
- **Timezone date bucketing for daily Sharpe uses UTC date** — `t.timestamp_entry.date()` returns UTC date; bars were resampled in `America/New_York`. Trades near midnight ET may be bucketed on the wrong calendar day, slightly distorting daily Sharpe. Pre-existing pattern from `timeframe_replication.py`. [`src/research/bidir_15m_test.py:77`]
- **`backtest_engine.py` carries an unstaged modification (git status M)** — modification predates this story and is not part of this diff; should be investigated separately to confirm it is not an accidental parameter drift. [`src/research/backtest_engine.py`]

## Deferred from: S12 adversarial review (2026-05-20)

Low-severity findings from three-agent review; symmetric or conservative-bias only — S12 verdict is unaffected.

- **H1 boundary 1-hour lag** — `build_h1_df` returns `iloc[:-1]`; H1 state updates only after a full H1 bar closes, introducing up to 1 hour of lag. Symmetric across real strategy and all 50 seeds; does not bias the S12 comparison. Fix if single-TF absolute-trade-count matters. [`s12_random_entry_control.py:build_h1_df`]
- **EDT/EST market-hours boundary** — `is_market_open` uses UTC-hour thresholds (22, 23) without DST correction; shifts market-open window ±1 hr in EDT vs EST months. Pre-existing in the deployed strategy (`tier2_streaming_working.py`); symmetric across real and random paths — both use the same `is_market_open`. [`s12_random_entry_control.py:is_market_open`]
- **O(n²) `build_h1_df`** — rebuilds H1 OHLC from scratch on every H1 boundary crossing. Acceptable for a one-shot 75k-bar run (~seconds). Incremental build needed before any live streaming use. [`s12_random_entry_control.py:build_h1_df`]
- **Daily circuit breaker absent** — deployed strategy halts at -$750/day; S12 real-strategy run has no circuit breaker. Conservative bias only (fewer real-strategy trades → real PF is pessimistic relative to live). Does not invalidate `patterns_survive` verdict. [`s12_random_entry_control.py:run_real_strategy`]
- **Fill-bar SL miss** — SL checked before TP on same bar; if entry bar simultaneously triggers both, SL takes priority. Minor optimistic bias on real strategy (a tiny subset of bars). Pre-existing in deployed system. [`s12_random_entry_control.py:run_real_strategy`]
- **`BEARISH_ONLY` not declared as a named constant** — value `True` is inline in `run_real_strategy`; behavioral compliance is present. Cosmetic. Add `BEARISH_ONLY = True` to frozen-params block if the file is revisited. [`s12_random_entry_control.py:run_real_strategy`]

## Deferred from: S13 adversarial review (2026-05-20)

Low-severity findings; no loopback required. S13 verdict (`design_phase2_ml_test`) is unaffected.

- **ATR filter is resolution-dependent** — `calc_atr` on 5-min bars produces ~3–5× larger ATR values than on 1-min bars; the `ATR_THRESHOLD * atr` filter is therefore stricter at coarser resolutions. By design per pre-registration ("same multipliers, same parameters"). FVG gaps also scale proportionally at coarser resolutions, so the effective filter rate is not obviously worse. Investigate if per-resolution trade counts are unusually low (S13: 96/32/14 at 1-min/5-min/15-min). [`s13_timeframe_replication.py:run_strategy`]
- **MAX_PENDING_BARS=240 spans 60 hr at 15-min resolution** — 240 × 15min = 3600min ≈ 2.5 calendar days. A limit order set Monday morning can remain pending Thursday morning, spanning multiple sessions. By design per pre-registration ("same multipliers"). Documented in spec comment. Revisit if 15-min edge is ever promoted to live trading. [`s13_timeframe_replication.py:run_strategy`]

## Deferred from: Program C Phase 1 — S12/S13 split (2026-05-20)

**S13 — Timeframe Replication** (`s13_timeframe_replication.py`): runs only if S12 returns `patterns_survive`.
- Resample holdout CSV to 5-min and 15-min OHLCV (open=first, high=max, low=min, close=last, volume=sum)
- At each resolution: run H1 sweep detection (H1 bars built from that resolution's bars) + bearish FVG detection with frozen params (ATR_THRESHOLD=0.5, MIN_GAP_ATR_RATIO=0.15, MAX_GAP_DOLLARS=60, SL_MULT=5.0, TP_MULT=6.0, same Tuesday block, same vol-regime gate, MAX_HOLD=60 bars of that resolution)
- 1-min result is available from S12's real-strategy run
- Report PF per timeframe, best_TF_PF = max of non-None PFs, verdict (≥ 1.1 → design_phase2_ml_test; < 1.1 → PIVOT)
- Gate: same `--preregistration 910e95c` + ACCESS_LOG append, same self-contained implementation pattern as S12
- Spec draft: `spec-program-c-phase-1-s12-s13.md` (S13 tasks already written; trim S12 tasks and rename if resuming)
