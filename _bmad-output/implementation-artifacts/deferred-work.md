
## Deferred from: code review of kraken-s27-live-execution (2026-06-04)

- **Orphaned live order on HTTP-200-then-parse-failure** — If `place_order()` gets HTTP 200 but raises `KrakenOrderError` on parse (e.g., missing `order_id` in `sendStatus`), the exchange order may have been submitted but local state clears to flat. Fix requires position reconciliation (explicitly out of spec scope). Operator must monitor Kraken UI for ghost orders. [`submission.py:96-104`, `s27_squeeze_streaming.py` entry flow]
- **UTC vs ET hour feature in ML** — `df.index[-2].hour` extracts UTC hour; the `vol_squeeze_ml_model.pkl` may have been trained on ET-hour features. If so, the live model receives systematically wrong hour values (offset by 5–6 hours). Verify training code and retrain if needed. [`s27_squeeze_streaming.py` ML features block]

## Deferred from: code review of 4-2-trade-logging-and-state-persistence-crash-recovery (2026-05-25)

- **Vol percentile computation duplicated inline** — `volatility_regime_filter()` does not expose its internal `pct_rank`; `_last_vol_regime_pct` is populated by replicating the same formula inline in `_update_h1_structure()`. Risk: the two formulas could diverge on future changes. Needs refactor of `volatility_regime_filter` to return percentile, or extraction of a shared helper. [`tier2_streaming_working.py` _update_h1_structure]
- **`exit_reason` mapping roundabout** — `_close_active_trade()` maps internal strings (`"tp"/"sl"/"time"`) back to PRD strings via hand-crafted ternary rather than using `ExitReason.value`. Values are correct; refactor would clean up the ExitReason→string→string pipeline. [`tier2_streaming_working.py:1174`]
- **`kill_zone_filter` receives `datetime` not `pd.Timestamp`** — type annotation says `pd.Timestamp` but call site passes `datetime`; works at runtime via duck-typing. Pre-existing mismatch from Story 4-1. [`tier2_streaming_working.py` _enter_trade]

## Deferred from: code review of 4-1-bracket-order-submission-and-position-reconciliation (2026-05-25)

- **`_ts_client=None` before `initialize()`** — same pre-existing pattern as `self.auth` and `self.client`; NoneType dereference if methods called before init. [`tier2_streaming_working.py` Tier2StreamingTrader.__init__]
- **Hardcoded `sim-api.tradestation.com` URLs defeat FR14 SIM→live swap** — `_BROKERAGE_BASE`, `cancel_order`, `SIM_ORDERS_URL` all hardcoded SIM; `AccountConfig.execution_mode` is unused dead data for now. [`tier2_streaming_working.py` TradeStationClient]
- **PENDING state only populates `entry_order_id`; TP/SL IDs always None in TradeState** — needed for Story 4-3 crash recovery and full bracket reconciliation. [`tier2_streaming_working.py` reconcile_state:368-372]
- **PENDING detection picks first Limit order (may be TP leg, not entry)** — acceptable while in-memory `active_trade` is authoritative; relevant if full broker-driven recovery is added. [`tier2_streaming_working.py` reconcile_state]
- **`close_position_at_market()` uses config contract count, not broker position size** — over/under-close possible on partial fill; no partial fills in TradeStation SIM currently. [`tier2_streaming_working.py` close_position_at_market]
- **Partial HTTP failure (one leg 503, other 200) gives ambiguous reconcile state** — only the outer `except` catches full failures; individual 4xx/5xx on one leg silently treated as empty data. [`tier2_streaming_working.py` reconcile_state]
- **AC#5 log format mismatch** — spec says `ERROR: API call failed — <error> — skipping bar`; implementation uses warning-level with emoji. Aspirational spec text; no log scraper active. [`tier2_streaming_working.py` TradeStationClient]
- **AC#5 error handling scope** — method-level exception handling only; pre-existing `_poll_and_process` handles the cycle-level catch. [`tier2_streaming_working.py` _poll_and_process]

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
- source_spec: none
  summary: src/ticksim/book.py — L3 OrderBook per instrument_id + apply_event(A/C/M/T/F) + queries (AD-3, AD-9, AD-22)
  evidence: split from the ticksim build; independently shippable after the foundation contract layer
- source_spec: none
  summary: src/ticksim/orders.py OrderTracker — the order lifecycle state machine + OCO groups (AD-8, AD-25)
  evidence: split from the ticksim build; depends only on the foundation schemas
- source_spec: none
  summary: src/ticksim/events.py — BookEventSource protocol + DbnMboSource + stable merge_streams (AD-18, AD-20)
  evidence: split from the ticksim build
- source_spec: none
  summary: src/ticksim/fills.py — pure decide() + BackOfQueueModel/TimePriorityModel + observe_book_event (AD-5, AD-21, AD-22)
  evidence: split from the ticksim build; depends on book + orders
- source_spec: none
  summary: src/ticksim/sim.py — SimRun discrete-event loop (AD-20), book->order seam (AD-21), deferred adverse-selection (AD-28), run manifest (AD-11)
  evidence: split from the ticksim build; the orchestration layer, depends on all leaves
- source_spec: none
  summary: src/ticksim/report.py — the three-way P&L report (AD-14, AD-24)
  evidence: split from the ticksim build
- source_spec: none
  summary: src/ticksim/parity/ — invariants.py (AD-16), part_a.py (AD-17), part_b.py, gate.py (AD-26); the acceptance gate
  evidence: split from the ticksim build; the parity gate that certifies the simulator per seal §3/Amendment 8
- source_spec: none
  summary: src/ticksim/cli.py — `simulate` and `parity-gate` entry points
  evidence: split from the ticksim build
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-foundation.md`
  summary: AD-24 wording — DOLLARS_PER_INDEX_POINT is a compile-time constant report.py imports directly from config.py; "read from the manifest's SimConfig dump" in AD-24 applies only to the configurable fee fields (commission/exch_reg). Reconcile at the report.py spec + a spine touch-up.
  evidence: blind-hunter review — a module constant does not appear in SimConfig.model_dump(), so AD-24 as literally worded is unsatisfiable. config.py placement is correct; only the AD sentence + the future consumer need aligning.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-foundation.md`
  summary: SimConfig derived-config creation (AD-15) must re-validate — Pydantic v2 model_copy(update=) skips validation. Provide a SimConfig.derive() helper or use model_validate(base.model_dump() | overrides) when study/report code lands.
  evidence: blind-hunter — AD-15 tells studies to build derived configs; model_copy bypass would let latency_ns=-1 or a float through silently.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-foundation.md`
  summary: spine AD-12 field names queue_rank_at_submit / queue_ahead_size_at_submit read "_at_submit" but AD-22 computes them at the arrival tick — consider renaming to _at_arrival in a spine polish pass (not a code defect; impl matches the AD).
  evidence: blind-hunter — semantic/name mismatch; deferred because it is a spine-AD wording change, not a diff defect.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-book.md`
  summary: [ADDRESSED in spec-ticksim-events.md] commit a small truncated .dbn.zst fixture slice so a real-record fold runs in the normal unit suite.
  evidence: verification-gap review — the only real-data verification of the MBO folder currently skips silently on any checkout without the untracked fixture.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-book.md`
  summary: book.py max_transient_cross_ns is a single global max but cross_start_ns is per-instrument — for a multi-instrument fold the manifest loses which instrument crossed. Make it per-instrument when multi-instrument lands.
  evidence: blind-hunter — fine for single-instrument H1, undocumented gap for later.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-book.md`
  summary: test_ticksim_imports.py constrains only src.* imports, not third-party — AD-4 limits ticksim's new deps to databento + sortedcontainers but nothing fails if a module imports an unlisted third-party package.
  evidence: verification-gap — pre-existing limitation; book.py is the first module to add third-party imports under the unenforced rule.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-book.md`
  summary: a "fills observed vs subsequent resting-size reductions" cross-check to catch stream gaps where the C/M following an F is missing (total_size stays inflated with no detection). Belongs with sim.py's observe_book_event / manifest work.
  evidence: blind-hunter — the F-as-no-op design is otherwise blind to a dropped reduction record.

## Deferred from: code review of spec-ticksim-order-tracker (2026-08-29)

- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-order-tracker.md`
  summary: **[RESOLVED 2026-08-29] AD-25 bracket semantics.** Alex chose the leg-aware cascade: an `EXIT`-leg fill cancels the other live group members; an `ENTRY`-leg fill cascades nothing. Spine AD-25 + the spec frozen block + `_cascade_oco` + tests all amended (commit on feat/ticksim-fill-simulator). No longer pending.
  evidence: blind-hunter + edge-case-hunter review; touched the frozen spine AD-25 + spec frozen block, surfaced to Alex, resolved same day.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-order-tracker.md`
  summary: OrderSnapshot carries only the spec's acceptance fields + `kind`. `fills.decide(book, tracker, clock_ns, config)` may need `arrival_best_bid_dbn`/`arrival_best_ask_dbn`, `oco_group_id`, `trade_id` on the snapshot. Add whatever that signature turns out to require when the fills.py slice lands.
  evidence: blind-hunter — snapshot field set can't be finalized until the queue-model interface is written.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-order-tracker.md`
  summary: `activate_arrivals` / `expire_all` full-scan `self._orders` every call (O(n) per tick, ~O(n²) per run). Fine for an H1 run (thousands of orders); add an `arrival_ts_ns`-keyed index if a study ever drives many orders per tick.
  evidence: edge-case-hunter — perf, not correctness.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-order-tracker.md`
  summary: no per-order time-in-force / validity window — `expire_all(now_ns)` force-expires every live order. AD-13(b) only requires interval-end expiry; per-order TIF (GTD/GTC/IOC) is not modelled. Revisit only if a strategy needs order-level expiry.
  evidence: blind-hunter — documented scope limit.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-order-tracker.md`
  summary: `sim.py` must translate an `IntentAction.CANCEL` intent-log record into `tracker.cancel(order_id, now_ns)` — the tracker takes `cancel(order_id, now_ns)`, not a CANCEL `OrderIntent`. Fields on a CANCEL record beyond `order_id` are not validated by the tracker. Handle at the sim.py intent-replay seam.
  evidence: edge-case-hunter — asymmetry with submit()/replace(); acceptable but must be wired in sim.py.

## Deferred from: code review of spec-ticksim-fills (2026-08-29, review-1)

- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-fills.md`
  summary: a `PASSIVE_LIMIT` intent priced *through* the opposite side (i.e. actually marketable on arrival). `fills.py` routes purely on `kind`, so it sits as passive waiting for trade volume at its aggressive price rather than crossing. Part A (AD-16/17) replays real broker orders — a limit entry that the broker filled immediately on arrival must be classified `marketable_limit`, not `passive_limit`, by the intent producer, or it becomes a parity miss. Decide at the Part A slice whether `_passive_fill` should defensively cross or the producer contract just forbids it.
  evidence: edge-case-hunter + blind-hunter review; a unified "route by actual marketability" model was judged out of scope for this slice (frozen spec separates the three kinds).
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-fills.md`
  summary: two (or more) same-side marketable orders in a single `decide` tick each walk the full displayed book independently — combined fills can exceed a level's real size. Consistent with "no own-order market impact" and H1 submits one entry at a time, but revisit if a strategy fires multiple marketable orders per tick.
  evidence: edge-case-hunter — `_walk_book` is per-order; the book is not depleted between orders in the same tick.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-fills.md`
  summary: a `MARKETABLE_LIMIT` order whose limit is not crossable at its arrival tick — `_walk_book` breaks immediately, the order stays working with zero queue modelling (`observe_book_event` only tracks `PASSIVE_LIMIT`), and it crosses whenever price later reaches it. Same "route by actual marketability" gap as the passive-priced-through item. Clarify producer contract at the Part A / H1 slice.
  evidence: blind-hunter review.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-fills.md`
  summary: AD-21's literal tie-break is "earlier `add_ts_ns`, or equal `add_ts_ns` and earlier `sequence`". `counts_resting_order` / `book.queue_ahead_size(strict=True)` ignore `sequence` — defensible (our order carries no vendor sequence at submit, so the tie is always resolved "our order last") but the deviation isn't covered by the AD-21 amendment. Fold a one-line note into AD-21 in a spine polish pass.
  evidence: blind-hunter — consistent with book.py's own `queue_ahead_size` comment; not a code defect.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-fills.md`
  summary: `observe_book_event` no-ops on `R` (book clear). An `R` wipes resting liquidity ahead of working passives; their `queue_ahead` is left stale. Treated as an exchange halt/reset (prereg §2.2 not-modelled list, AD-13 mask territory) — but if a session ever contains a mid-RTH `R`, `sim.py` should zero working passives' `queue_ahead` at that seam.
  evidence: edge-case-hunter.

## Deferred from: spec-ticksim-sim planning split (2026-08-29)

- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-sim.md`
  summary: **[RESOLVED 2026-08-30] AD-28 `adverse_selection` deferred-check queue.** Built as step 6 of `SimRun._loop` — `spec-ticksim-sim-adverse.md` (done). Alex pinned the §2.1 predicate: any point in the 1 s window, same-side quote moves away, latched on book ticks. Spine AD-28 amended.
  evidence: planning split from the sim.py slice; predicate needed a human decision.

## Deferred from: code review of spec-ticksim-sim (2026-08-29, review-1)

- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-sim.md`
  summary: `OrderOutcome.queue_rank_at_submit` is set to the contracts-ahead total (== `queue_ahead_size_at_submit`), not a true order-count rank. A real rank needs `QueueModel.queue_ahead_size` (or a sibling) to return the number of resting orders ahead, not just their summed size. Revisit if a study wants rank as a distinct signal. spine AD-12 field-name note also open (`_at_submit` vs `_at_arrival`).
  evidence: blind-hunter + verification-gap — flagged since the foundation slice; `queue_ahead_size` returns a total.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-sim.md`
  summary: OCO-cascade cancels are surfaced only as a manifest counter (`oco_cascade_cancel_count`) and `logger.debug`; the per-fill cascaded-id list the spec's step 5 says to "record" is not machine-readable beyond `finalize()`'s terminal states. Fine for parity (recoverable) — add a structured cascade log to the manifest only if a consumer needs the linkage.
  evidence: blind-hunter.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-sim.md`
  summary: the intent log is not validated against per-order terminal state within its own timeline (e.g. `SUBMIT o1; CANCEL o1; CANCEL o1` passes up-front validation; the 2nd cancel is dropped at runtime when the deferred effect finds o1 terminal). Safe (no crash, broker-realistic) but a stricter producer-contract check could fail-fast. A REPLACE can un-terminalize, so "terminal in the log" is fuzzy — revisit at the Part A intent-log reconstruction slice.
  evidence: edge-case-hunter + verification-gap.

## Deferred from: code review of spec-ticksim-sim-adverse (2026-08-30, review-1)

- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-sim-adverse.md`
  summary: **multi-partial passive fills** — the AD-28 check is enqueued only on the *completing* fill (its `ts_ns` / `px_dbn`), because `tracker.set_adverse_selection` requires a `FILLED` order. Adverse moves between a passive order's first partial and its completion, and the filled quantity of a passive order that partial-fills then EXPIRES/CANCELS, are not measured. Fix would need per-partial `_AdverseCheck`s with the `set_adverse_selection` call deferred until FILLED (or a per-partial adverse field). Rare for 1–5-lot H1 (fills in one shot at the touch). Revisit if a study shows meaningful passive partials.
  evidence: edge-case-hunter + blind-hunter; forced by the `set_adverse_selection` FILLED contract.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-sim-adverse.md`
  summary: `parity/invariants.py` (a later slice) should pick up an AD-28 structural invariant: `adverse_selection is True ⇒ terminal_state == FILLED and kind == passive_limit`; a `MARKETABLE`/`MARKETABLE_LIMIT` outcome never has `adverse_selection`. Currently enforced only inline in `sim._step_fills`; no independent check.
  evidence: blind-hunter.

## Deferred from: code review of spec-ticksim-report (2026-08-30, review-1)

- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-report.md`
  summary: `ThreeWayReport` carries `config_primary` / `config_optimistic` as provenance but no `study_id`, source-manifest SHA-256, intent-log SHA, or simulator commit — AD-11 makes the manifest the unit of reproducibility. `cli.py` persists the full manifests alongside the report; if a study wants the linkage inside the report object, add a `provenance` field there (cli-slice concern).
  evidence: blind-hunter.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-report.md`
  summary: no `ThreeWayReport.from_dict` — `to_dict()` is one-way. Add a round-tripping constructor if a consumer needs to reload a persisted report (rather than re-running `build_report`).
  evidence: blind-hunter (`to_dict` "round-trips" language) + verification-gap.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-report.md`
  summary: an `OpenPosition` (entry filled, exit never) whose entry fill was `adverse_selection`-flagged loses that signal — `adverse` is only recorded on `RoundTrip`. A `.adverse` field on `OpenPosition` would preserve it. Rare; §2.2 exposure detail is already captured (size/px/ts).
  evidence: blind-hunter.

## Deferred from: code review of spec-ticksim-parity-invariants (2026-08-30, review-1)

- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-invariants.md`
  summary: **invariant 6 (strict causality / no-lookahead) has no post-hoc `OrderOutcome` form.** `check_fill_causality` only checks the *trace* (fills non-decreasing, `>= arrival_ts_ns`). The real AD-20 property — "only book events with `ts_event <= clock` decided any fill" — must be enforced in `parity/part_b.py`, which owns the book/event stream: e.g. re-drive the sim with the event source truncated at each fill's `ts_ns` and assert the same fill, or assert every consumed event's `ts_event <= fill.ts_ns`. Not this slice.
  evidence: verification-gap + blind-hunter — the module docstring and spec Design Notes both name it a construction guarantee; part_b must add the live check.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-invariants.md`
  summary: **invariant 4's time-series monotonicity is not checked on Part B's synthetic orders.** `check_queue_position` checks only the serialized endpoint (both fields present + non-negative for a worked passive). The "non-increasing until terminal" series is asserted only in `test_ticksim_orders.py` on hand-built tracker sequences. `parity/part_b.py` should, for each synthetic passive order, capture the `queue_ahead_size` series across ticks and assert monotone-non-increasing until fill/cancel — the ≥1000-order gate is where it gets exercised against real book churn.
  evidence: verification-gap.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-invariants.md`
  summary: **invariant 5's liquidity half** ("no fill without book liquidity at/through the price") is a `fills.py` construction guarantee (`_walk_book` only emits for `level_size > 0`; passive fill only when `cum_trade_vol_since_arrival - queue_ahead > 0`), verified in `test_ticksim_fills.py`. `parity/part_b.py` MAY add a book cross-check at each fill tick since it has the book. Confirmed as the resolution to the spec's "Ask First" at CHECKPOINT 1.
  evidence: planning CHECKPOINT 1 (option b) + all three reviewers.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-invariants.md`
  summary: `check_no_price_improvement` raises "unverifiable" when a marketable order filled with the crossed-side arrival quote `None` (fail-closed, deviates from frozen-block iteration-0 "→ None"). If Part A/B surfaces legitimate `None`-quote fills (thin book, warm-up edge) this may need softening to a counted-and-recorded miss rather than a hard `InvariantViolation`. Revisit when Part B runs against real MNQ data.
  evidence: Spec Change Log #4; Suggested Review Order #3.

## Deferred from: planning of spec-ticksim-parity-part-a (2026-08-30, token split)

- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-part-a.md`
  summary: `run_part_a` — the MBO-window runner tying reconstruction to the sim: build a `DbnMboSource` per trade's ±90-min window, call `sim.simulate(source, trade.intents, PRIMARY, [window_ns])` (and `OPTIMISTIC` for the reported-only errors), feed the outcomes to `compare_fills`, price every `leg_unfilled` miss as the same-side touch at `exit_ts` + 1 tick adverse slip (AD-17), then `aggregate`. Needs AD-7 widened `parity → events` (inline note; part_b needs it too). Ships with `tests/integration/test_ticksim_parity_part_a.py` (@integration) reconstructing the 3 yank 2026-06-22 trades and running against `data/tick/_test/glbx-mdp3-20260622.mbo.dbn.zst`. Full N≥28 run is post-Tranche-1-purchase.
  evidence: part-a planning token check — frozen block hit ~1650+ tokens with the runner included; the pure core (reconstruct + compare + aggregate, imports only orders/config, fixture-tested) is the independently-mergeable half. User approved the split 2026-08-30.

## Deferred from: code review of spec-ticksim-parity-part-a (2026-08-30, review-1 → intent_gap loopback)

- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-part-a.md`
  summary: Part A does not exercise resting-order queue behaviour at all — mim-nb's `otype 4` protective stop is always cancelled (never fills, never graded) and yank rows have no resting limit legs, so Part A is a pure market-order fill-price replay. Back-of-queue position, partial fills vs real trade volume, and non-through-limit passive fills are covered only by Part B's ≥1000 synthetic orders. If a live bot later runs real resting limit entries/exits, Part A's sample should be extended to grade them (v2 / §A8.4).
  evidence: blind-hunter + the human's loopback-1 "minimal replay" decision — accepted scope, recorded so the gate's coverage is honest.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-part-a.md`
  summary: `compare_fills` assumes a single `RealFill` per `(order_id, leg)` and `reconstruct_mim_nb` raises on a second `FILL` row for one order_id. The real Part A sample is 1–2 lot and fills in one shot, but yank is 2ct and future bots larger. Partial real fills would need per-order fill accumulation in the lifecycle walk and a multi-fill `RealFill` list per leg.
  evidence: all three reviewers; forced by the current single-fill data shape.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-part-a.md`
  summary: `data/mim_nb/orders.csv` (and `trades.csv`) carry a `chain` hash-chain integrity column; `reconstruct_mim_nb` ignores it. A truncated or tampered ledger is accepted silently. Chain validation belongs in the `run_part_a` loader (this slice takes already-parsed rows), which should verify the chain before reconstruction and refuse a broken ledger.
  evidence: blind-hunter + project memory (`data/mim_nb` logs are hash-chained).
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-part-a.md`
  summary: `run_part_a` (deferred slice 2) is where the routed-exit-leg vs `OrderTracker` OCO-cascade interaction actually bites, where the `leg_unfilled` miss magnitude is priced (touch @ exit_ts + 1 tick slip, AD-17), and where `trades.db` timestamp timezone (verified `+00:00` UTC-aware now, but assert at load) and nanosecond resolution (source is µs — no ns lost from these ledgers) must be re-checked against real MBO windows.
  evidence: verification-gap + edge-case-hunter — carried forward to the runner slice.

## Deferred from: code review of spec-ticksim-parity-part-a (2026-08-30, review-2 / patch round)

- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-part-a.md`
  summary: Part A grades fill *price* only — `compare_fills` ignores fill timestamps entirely. A sim that fills a leg far from the real fill time at a coincidentally close price passes Part A clean. The prereg §A8.2 tolerances are all price (MAE/p90/bias in ticks), so this is in-seal, but `run_part_a` (slice 2) should add a time-divergence sanity bound (e.g. flag any leg whose sim fill ts is > N seconds from the real fill ts) as a non-verdict-bearing diagnostic, and v2 (§A8.4) could tighten it.
  evidence: blind-hunter r2 (raised twice) — a real class of simulator bug currently invisible to the gate.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-part-a.md`
  summary: the reconstructed entry+exit legs share one `oco_group_id` (AD-25). `OrderTracker._cascade_oco` cancels other live group members (incl. an unfilled entry) when an EXIT-leg fills. For the real mim-nb/yank data the exit is hours after the entry fill so this never bites, but `run_part_a`'s integration test should exercise a 2-member entry+exit OCO group through the tracker under latency to confirm the exit-fill cascade does not void an already-filled (or still-in-flight) entry.
  evidence: blind-hunter r2 — no test currently covers an entry+exit OCO group end-to-end through the tracker.

## Deferred from: code review of spec-ticksim-parity-run-part-a (2026-08-30, review-1 / patch round)

- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-run-part-a.md`
  summary: `tests/integration/test_ticksim_parity_run_part_a_integration.py` (the only test that drives `run_part_a` over a real re-iterable `DbnMboSource` + a dense real book — real DBN iteration, `_touch_at` finding touches in real depth, `unresolved_misses == 0` on real data) skips whenever the window fixture `data/tick/_test/glbx-mdp3-20260622.mbo.dbn.zst` is absent, which it is in this worktree and in CI (`TICKSIM_REQUIRE_FIXTURE` unset). Run it (and set `TICKSIM_REQUIRE_FIXTURE=1` in a fixture-carrying CI lane) once Tranche 1 data lands; regenerating the capture may also need `_WindowSource`'s lead margin re-tuned.
  evidence: verification-gap + blind-hunter r1 — the real-data path has never executed.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-run-part-a.md`
  summary: `_touch_at` re-walks the whole window source once per `leg_unfilled` miss (CHECKPOINT-1: kept simple, O(misses × window) is negligible at Part A's N≈28 × ≤1 miss). If Part A v2 (§A8.4, N≥100) or a future high-miss window makes this hot, amortise to a single sorted-BBO-timeline pass with a bisect per miss.
  evidence: CHECKPOINT-1 decision — recorded as the revisit condition.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-run-part-a.md`
  summary: `_window_span`'s "span every `RealFill.ts_ns`" term is load-bearing only for mim-nb-reconstructed trades (where `_build_mim_trade` sets `RealFill.ts_ns` to the FILL row ts, later than the PLACE/submit ts); no unit or integration test uses a trade whose real fill ts differs from its submit ts (integration is yank-only, where they're equal). The 5-min `PART_A_WINDOW_PAD_NS` absorbs realistic market-order fill latency so the observable impact is ~nil, but add a mim-nb-shaped fixture when the real ledger is available.
  evidence: verification-gap r1.

## Deferred from: code review of spec-ticksim-parity-part-b (2026-08-31, review-1 → intent_gap loopback)

- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-part-b.md`
  summary: `_bookwalk.BookReplay`'s multi-instrument guard is lazy — it only fires for a second `instrument_id` that appears at or before the `advance_to` cutoff. A foreign contract that appears only later in the window is not caught (same limitation as `sim`'s own lazy multi-instrument check). A `BookReplay.scan_single_instrument()` full-pass mode would close it; not needed while callers filter to front-month.
  evidence: edge-case-hunter r1.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-part-b.md`
  summary: Part B's `valid_intervals` is one spanning `[lo-pad, hi+pad)` window. The prereg generates ≥1000 orders "across the Tranche 1 data" = 28 windows / 24 calendar days; a single interval marks orders valid during overnight / inter-window gaps. If per-window bucketing is wanted for the real run it must come from the slice-2 generator emitting orders only inside real session windows, or `run_part_b` taking `valid_intervals` explicitly. Revisit at slice 2 / the real gate run.
  evidence: blind-hunter r1.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-part-b.md`
  summary: `part_b` no longer needs `_bookwalk`, but the slice-2 synthetic-order generator will (BBO sampling to pick realistic `limit_px_dbn` for `passive_limit` / `marketable_limit` orders). `BookReplay` is the primitive for that.
  evidence: loopback-1 KEEP rationale.

## Deferred from: code review of spec-ticksim-parity-part-b (2026-08-31, review-2 / patch round)

- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-part-b.md`
  summary: `PART_B_COVERAGE_NOTE` (which invariants Part B verifies post-hoc vs construction-guaranteed) is carried on `PartBResult.coverage_note` but nothing forces a consumer to surface it. `gate.py` (next slice) MUST render the note into the §A8.2 gate report so a gate reader knows the ≥1000-order battery is a scaled post-hoc check, not a re-derivation of the sim's internal ordering.
  evidence: blind-hunter r2 — flagged as a slice-2/gate.py requirement so the note doesn't become decorative.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-part-b.md`
  summary: `run_part_b` does not eagerly scan `source` for a second `instrument_id` — it relies on `sim`'s lazy multi-instrument check, which misses a foreign contract appearing only outside the padded submit-ts window. Same stance as `run_part_a` (caller filters front-month). A `BookReplay.scan_single_instrument()` full-pass or an eager `{ev.instrument_id for ev in source}` check in the gate driver would close it; revisit at the real gate run when the front-month filter's correctness matters most.
  evidence: edge-case-hunter r1+r2, blind-hunter r1.

## Deferred from: code review of spec-ticksim-cli-simulate (2026-08-31, review-1 / patch round)

- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-cli-simulate.md`
  summary: `simulate` with `--instrument-id` omitted reads the whole `.dbn.zst` twice — once for `detect_front_month`, once for the sim run. On the ~22.5M-record parity fixture that doubles decompression. Round-1 patch documents it (`--help` recommends passing `--instrument-id` for large windows) but a caching solution (return the first-pass events, or a peek-and-tee source) is deferred — RAM cost of holding 22.5M `BookEvent`s in memory needs measuring first.
  evidence: blind-hunter + verification-gap r1.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-cli-simulate.md`
  summary: the `simulate` integration test (`tests/integration/test_ticksim_cli.py`) skips without `data/tick/_test/glbx-mdp3-20260622.mbo.dbn.zst`, so `FrontMonthSource` over a REAL `DbnMboSource` (real `class_rank`, real re-iterability, real auto-detect pass over a `.dbn.zst`) is never exercised in CI. Run it (and set `TICKSIM_REQUIRE_FIXTURE=1` in a fixture-carrying lane) once Tranche 1 data lands.
  evidence: verification-gap r1.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-cli-simulate.md`
  summary: cli.py has three more subcommands to build in later slices: `report` (AD-14 3-way P&L from two outcome+manifest pairs — re-adds the `report` import edge), `parity-gate` (reconstruct trades → run_part_a + run_part_b → gate.build_amendment_stub → append-only stub), and the §5 integrity preflight scan (gate.py slice) wired into `parity-gate`. Plus the synthetic-order generator (`generate_synthetic_orders`, BBO sampling via `_bookwalk.BookReplay`) feeding `parity-gate`'s Part B. None can run a real verdict until the Tranche 1 MBO windows are purchased.
  evidence: planning splits across the parity build.

## Deferred from: code review of spec-ticksim-parity-synthetic (2026-08-31, review-1)

- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-synthetic.md`
  summary: `_OVERGEN_FACTOR = 2.0` is a guess at how many limit candidates a real Tranche-1 window drops. `generate_synthetic_orders` prices BBO at `submit_ts_ns`; the sim fills at `+250ms` (PRIMARY latency), so the marketable-vs-passive label is nominal. Once Tranche 1 data lands: check the `logger.debug` drop-rate breakdown on a real window and re-tune `_OVERGEN_FACTOR`; if the nominal-label drift matters for a study, add an `advance_to(submit_ts_ns + latency_ns)` variant.
  evidence: blind-hunter + verification-gap r1.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-synthetic.md`
  summary: the `n>=1000` full-book Part B battery (generator + `run_part_b` over a real GLBX capture) is only in `tests/integration/test_ticksim_parity_synthetic.py`, which skips without `data/tick/_test/glbx-mdp3-20260622.mbo.dbn.zst`. Run it (with `TICKSIM_REQUIRE_FIXTURE=1`) once Tranche 1 lands; also fix the integration test to start `gen_source` with the same warm lead-in as `sim_source` (currently the generator prices off a cold/partial book while the sim uses a warm one) and to scan the full file (not first 400k records) for the front-month `iid`.
  evidence: verification-gap + blind-hunter r1.

## Deferred from: code review of spec-ticksim-parity-integrity (2026-08-31, review-1)

- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-integrity.md`
  summary: `preflight_integrity`'s BBO-watch cross state machine partly duplicates `book`'s own `cross_start_ns` timer; on an `R` (clear) event the two could momentarily disagree. Round-1 patch requires recomputing the module's open-cross var after every folded event (incl. a clear) to stay in sync. If a future `book` change alters the cross-timer semantics, re-check this. Cleaner long-term: expose `book`'s persistent-cross episodes as a queryable list so the module doesn't re-derive durations at all.
  evidence: blind-hunter r1.
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-integrity.md`
  summary: `_MAX_GAP_NS` (inter-event-gap halt threshold) and `_TRADE_BBO_TOLERANCE_TICKS` are guesses. Once Tranche 1 data lands, run `preflight_integrity` over each real window and calibrate: the 2026-06-22 test window's `bbo_cross_rate` should be ≈0.014% (Amendment 9); the degraded days 2026-05-24 / 2026-07-30 windows are where a real persistent cross / large gap would show — use those to confirm the thresholds don't false-positive on normal degraded-but-usable data.
  evidence: verification-gap + blind-hunter r1 (integration test skips; real-window figures unverified).
- source_spec: `_bmad-output/implementation-artifacts/spec-ticksim-parity-integrity.md`
  summary: correlating a flagged anomaly's `ts_event` with whether it fell on a `degraded` day would be the highest-value triage output, but needs ts→date arithmetic this module deliberately avoids (`datetime` banned). The parity-gate CLI slice (which already has the window→date mapping) could do the correlation when it assembles the amendment stub.
  evidence: blind-hunter r1.

## Deferred: the `cli.py parity-gate` subcommand — the last capstone slice (2026-08-31)

- source_spec: (to be written — `spec-ticksim-cli-parity-gate.md`)
  summary: The one remaining code slice. `ticksim parity-gate` wires every built module into the §A8.2 append-only amendment stub. All dependencies exist and are merged (`part_a` + `part_a_runner`, `synthetic` + `part_b`, `integrity`, `gate`, `cli.FrontMonthSource`). Cannot run a real verdict until the 28 Tranche-1 MBO windows are purchased (~$68, `tick_infra_tranche1_purchase_guide.md`).
  Design captured while the pieces are fresh:
  * **Part A**: read `data/mim_nb/orders.csv` → `part_a.reconstruct_mim_nb(rows)`; read `data/trades.db` `trades` where `trader_id='trader-yank' AND timestamp>='2026-06-17'` (and any new mim-nb rows) → `part_a.reconstruct_trades_db_row(row)` per row. A `--windows` JSON maps each trade_id (or a ±90-min window key) → its `.dbn.zst` path + the front-month `instrument_id` + any degraded-day tag (regenerate from `~/.claude/jobs/960bda86/tmp/parity_windows.py`). `source_for(trade) = FrontMonthSource(DbnMboSource(path), iid)` clipped to the trade's window. `part_a_runner.run_part_a(trades, source_for) -> PartAResult`.
  * **Part B**: pick one dense window (or concatenate a few via `events.merge_streams`), `synthetic.generate_synthetic_orders(source, lo, hi, n=1000, seed=<cli arg>)` → `part_b.run_part_b(intents, source) -> PartBResult`.
  * **Integrity**: `integrity.preflight_integrity(FrontMonthSource(...), degraded_days=[...])` per window → `integrity.format_integrity` → concatenate into the `integrity:` string for `gate.build_amendment_stub`.
  * **Verdict + stub**: `gate.evaluate(part_a, part_b)`; `gate.build_amendment_stub(part_a, part_b, amendment_number=<cli>, cycle_number=<cli>, integrity=<combined>, trader_by_trade_id=<derived from reconstruction>, date=<cli or TBD>)` → write the `.md` (append-only — the function returns text, the CLI writes a NEW file, the analyst appends to the seal).
  * **CHECKPOINTs for that slice**: (a) does a FLAGGED integrity report block the gate verdict, or just annotate the stub? (b) Part B over one window vs several merged? (c) `--windows` JSON schema. (d) trader_by_trade_id: derive from the `mimnb-` prefix vs a real map.
  * Edge: `cli` would gain `parity` (all the parity siblings) — a large widening; consider a `parity/gate_cli.py` helper module inside `parity/` instead so `cli.py` stays thin.
  evidence: planning — every upstream slice is done; only the wiring + the data purchase remain.

## Deferred: `parity-gate` slice — merged 2026-09-01

- source_spec: `spec-ticksim-parity-gate-cli.md`
  summary: The capstone slice shipped. These are the knowingly-unfinished edges.
  * **The three round-2 reviewer subagents never ran** — all three (blind-spot, edge-case, verification-gap) died on an API session rate limit. The round-2 pass was done inline by the main session and found + fixed two untested round-1 patches (source memoisation, BOM tolerance), but it is not a substitute for the adversarial pass. Re-run `bmad-review` over `gate_cli.py` + the `cli.py parity-gate` delta when the limit clears.
  * **The integration test has never actually executed.** `tests/integration/test_ticksim_parity_gate.py` skips cleanly because `data/tick/_test/glbx-mdp3-20260622.mbo.dbn.zst` is not present in the worktree. Every parity-gate assertion to date is against in-memory hand-built sources. Run it before trusting a real verdict.
  * `test_broken_pipe_preserves_fail_exit_code` monkeypatches `cli.os.dup2` to a no-op so the handler's real-fd redirect cannot clobber pytest's capture. The exit-code path is genuinely asserted; the redirect's actual side effect is not.
  * `_is_type_checking_test` (`tests/unit/test_ticksim_imports.py`) matches any `ast.Attribute` with `.attr == "TYPE_CHECKING"`, not specifically a `typing` binding. Unreachable today; tightening needs import-alias tracking.
  * `--out` pointing at an existing **directory** with `--force` surfaces as a generic `_CliError` from `os.replace`, not a targeted message.
  * `_run_parity_gate`'s `getattr(args, "windows_parsed", None) or _read_windows(...)` fallback exists only for direct `_run_parity_gate` calls that bypass `_validate_parity_gate_args`; it is untested and, in the CLI path, unreachable.
  * `--config optimistic` is asserted at the `run_parity_gate` level (`test_config_forwarded_to_both_runners`) but not end-to-end through `cli.main`.
  evidence: review round 2, inline (2026-09-01).

## Deferred: §A8.2 gate RUNTIME — measured, and it is the binding practical constraint (2026-09-01)

- source_spec: `spec-ticksim-parity-gate-cli.md` (post-merge measurement, not a code defect)
  summary: The parity-gate integration test was executed for the first time against the real `data/tick/_test/glbx-mdp3-20260622.mbo.dbn.zst` capture (364 MB compressed, 22.5M records, 2.5h of MNQ tape). Measured on the KVM4 box (4 vCPU):
  * raw `DbnMboSource` decode: **~297,000 rec/s**
  * decode + `book.apply_event` fold: **~111,000 rec/s** (folding costs ~2.7x)
  * the test's nominal "90-minute" window (`anchor-30min .. anchor+60min`) actually contains **~20.3M records — essentially the whole capture**, so `_ClippedSource`'s `hi_ns` break buys almost nothing here.
  * a single folding pass over that window: **~3 minutes**. `run_parity_gate` makes ~9 such passes (one per Part A trade, one per unfilled leg, one for `generate_synthetic_orders` pricing, one for `run_part_b`'s `simulate`, one for integrity, plus the test's 2 anchor passes) -> a ~27-minute floor.
  * **CONFIRMED: the test PASSED in 2h57m33s (10,652s) for ONE window.** The excess over the 27-min fold floor is `run_part_b`'s `simulate` over 1000 synthetic orders — per-event order matching, not decompression — which dominates.
  What this means for the real §A8.2 run: at ~3h per window, **28 Tranche-1 windows is ~83 hours serial (3.5 days) -- against a §4 kill criterion of 15 working days for THREE revision cycles.** One bad cycle would eat a quarter of the clock in compute alone. Before the purchase is spent, decide one of:
  1. Size Part B's window deliberately — the prereg requires >=1000 synthetic orders, *not* a 20M-event window. A 5-10 minute dense RTH slice satisfies §A8.2 Part B and cuts the dominant cost by ~10x. `--synthetic-window` already makes this a CLI argument, so no code change is needed — just point it at a narrow window entry.
  2. Profile `sim.simulate`'s per-event hot path (the resting-order scan) before running the real gate.
  3. Parallelise across windows (each window is independent; Part A is embarrassingly parallel per trade).
  Option 1 is free and should be the default. **Do not budget the real gate run assuming it is minutes.**
  evidence: measured 2026-09-01 on the merged `feat/ticksim-fill-simulator` @ 937073b.

### Update 2026-09-01: the integration test PASSED (first real end-to-end gate execution)

`tests/integration/test_ticksim_parity_gate.py::test_run_parity_gate_over_the_2026_06_22_window`
**passed in 2h57m33s** against the real 22.5M-record capture. This is the first
time `run_parity_gate` has executed end to end on genuine MNQ MBO data rather
than in-memory hand-built sources, and it exercised the full chain: Part A over
3 reconstructed trades, Part B over 1000 synthetic orders, the integrity
preflight, `gate.evaluate`, and `build_amendment_stub` (the stub's
`# Amendment 1 -- Parity gate result (cycle 1)` header asserted).

The deferred item "the integration test has never actually executed" is now
**CLOSED**. What remains is the runtime consequence above: ~3h/window x 28
windows = ~83h serial, so **narrow `--synthetic-window` before the real run**.
