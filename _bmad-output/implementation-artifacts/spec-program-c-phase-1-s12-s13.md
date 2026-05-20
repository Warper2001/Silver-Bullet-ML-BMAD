---
title: 'Program C Phase 1 — S12 Random-Entry Control'
type: 'feature'
created: '2026-05-20'
status: 'done'
baseline_commit: '415d3a3ffdd14b9b59942d996fece0eebdab85d0'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Program C Phase 1 requires a random-entry control test (S12) on the sealed holdout before any Phase 2 work. No script exists, so the S12 verdict cannot be computed.

**Approach:** Implement a self-contained `s12_random_entry_control.py` that loads the holdout CSV, applies the pre-committed filters, runs 50 random-entry seeds and the real strategy on the same window, and emits a verdict per the pre-registered S12 decision tree.

## Boundaries & Constraints

**Always:**
- Self-contained — no imports from `backtest_tier2_1year_validation.py` or `src/research/tier2_streaming_working.py`; replicate all logic inline
- Frozen parameters (match `preregistration_phase1.md` exactly): `SL_MULT=5.0`, `TP_MULT=6.0`, `MAX_HOLD=60`, `ATR_PERIOD=20`, `VOL_LOOKBACK=120`, `VOL_THRESH=0.75`, `ATR_THRESHOLD=0.5`, `MIN_GAP_ATR_RATIO=0.15`, `MAX_GAP_DOLLARS=60.0`, `MNQ_DOLLAR=2.0`, `ENTRY_PCT=0.5`, `BEARISH_ONLY=True` (real strategy), `MAX_PENDING_BARS=240`
- ACCESS_LOG gate (SHA validate + append) fires before any holdout bar is read; same pattern as Phase 0.5 (`backtest_tier2_1year_validation.py:66–141`), accessor label `"s12 script"`
- Report written to `data/reports/s12_<YYYYMMDD_HHMMSS>.txt` and printed to stdout
- Verdict text quotes the exact pre-committed thresholds from `preregistration_phase1.md`

**Ask First:**
- If `data/sealed_holdout/ACCESS_LOG.md` is missing — halt and explain Phase 0.4 setup before reading any bars

**Never:**
- Adjust any frozen parameter after observing results
- Run the 50-seed loop before the ACCESS_LOG gate has passed and been appended

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| No `--preregistration` flag | holdout bars present | Exit 1; stderr: "pre-registration" + "ACCESS_LOG" | Print full instructions to stderr |
| Bad SHA format | `--preregistration zzz` | Exit 1 with SHA in message | Regex guard before any subprocess call |
| SHA not in git | valid hex, unknown commit | Exit 1 "SHA not found" | Subprocess returncode check |
| Valid SHA, no prereg file | commit exists, no `preregistration*.md` | Exit 1 listing expected patterns | Parse `git show --name-only` output |
| All checks pass | valid SHA + prereg file | Run S12; append ACCESS_LOG; write report | — |
| Seed produces 0 trades | all bars filtered or no signals | Row shows "N/A"; seed excluded from median/pct calc | Guard div-by-zero in `profit_factor` |

</frozen-after-approval>

## Code Map

- `data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv` — input (columns: timestamp, open, high, low, close, volume, notional)
- `data/sealed_holdout/ACCESS_LOG.md` — appended once per valid invocation
- `backtest_tier2_1year_validation.py:66–141` — gate helpers reference; copy verbatim, change accessor label
- `backtest_tier2_1year_validation.py:146–167` — CSV loader pattern; adapt to simple `Bar` namedtuple (timestamp, open, high, low, close)
- `_bmad-output/preregistration_phase1.md` — S12 exact procedure (steps 1–11) and decision rule; implement to the letter

## Tasks & Acceptance

**Execution:**

- [x] `s12_random_entry_control.py` — new file; structure:
  1. **Constants block**: all frozen params, `HOLDOUT_CSV`, `HOLDOUT_CUTOFF = datetime(2026,3,1,utc)`, `ACCESS_LOG_PATH`, `_PREREG_PATTERNS` (same two globs as Phase 0.5)
  2. **Gate helpers** copied from `backtest_tier2_1year_validation.py:66–141` with accessor label changed to `"s12 script"` in `append_access_log`
  3. **`Bar` namedtuple** `(timestamp, open, high, low, close)`; **`load_holdout_bars() -> list[Bar]`** via `csv.DictReader`; ISO timestamp with UTC tzinfo; sort by timestamp
  4. **`calc_atr(bars, end_idx, period=20) -> float`**: 20-bar SMA True Range ending at `end_idx` (exclusive); return `10.0` if fewer than `period` bars; TR = `max(H−L, |H−prev_C|, |L−prev_C|)`
  5. **`build_h1_df(bars, up_to_idx) -> pd.DataFrame`**: resample `bars[:up_to_idx]` to 1-hour OHLC via pandas `resample("1h").agg(open=first, high=max, low=min, close=last).dropna()`; return `iloc[:-1]` (completed bars only)
  6. **`VolRegimeTracker`**: stateful; `update(h1_df) -> bool` computes H1 ATR (20-bar SMA TR on h1_df rows), appends to rolling history (cap `VOL_LOOKBACK=120`), returns True when `len >= 20 and pct_rank > VOL_THRESH`
  7. **`is_market_open(ts_utc) -> bool`**: replicate `backtest_tier2_1year_validation.py:356–363` exactly (Saturday closed; Sunday ≥ 23:00; Friday < 22:00; Mon–Thu ≠ 22:00)
  8. **`run_real_strategy(bars) -> float | None`**: single sequential pass; update H1/vol state on each H1 boundary crossing; detect bearish H1 sweep (last completed H1 bar: `high > confirmed_swing_high and close < confirmed_swing_high`; confirmed swings use 2-bar left+right radius ≥ 2 bars before last H1; sweep active 6 H1 hours); detect bearish 3-bar FVG on `bars[i-2:i+1]` (`c1.low > c3.high and c2.close < c2.open`); apply 3 FVG filters (ATR_THRESHOLD, MAX_GAP_DOLLARS, MIN_GAP_ATR_RATIO); enter limit at FVG midpoint snapped to 0.25 tick; SL = `entry + gap * SL_MULT`, TP = `entry − gap * TP_MULT`; fill: `bar.high >= entry`; advance: check SL first (`bar.high >= sl`), then TP (`bar.low <= tp`), then time-stop at `bar.close` after `MAX_HOLD` bars; P&L = `(entry − exit) * MNQ_DOLLAR`; return `profit_factor(pnl)` or `None` if 0 trades
  9. **`run_random_seed(bars, seed) -> float | None`**: fresh `VolRegimeTracker` per seed; `rng = np.random.default_rng(seed)`; at each bar (no active trade): check `is_market_open`, not Tuesday (ET weekday), not vol_regime_high; flip `rng.integers(2)` (0=SHORT, 1=LONG); `atr = calc_atr(bars, i+1)`; SHORT: entry=`bar.close`, `sl=entry+5*atr`, `tp=entry−6*atr`; LONG: `sl=entry−5*atr`, `tp=entry+6*atr`; NO pending period (market order); simulate until TP/SL/time-stop; P&L = `(entry−exit)*MNQ_DOLLAR` for SHORT, negated for LONG; return `profit_factor(pnl)` or `None` if 0 trades
  10. **`profit_factor(pnl) -> float`**: `sum(p>0) / |sum(p<0)|`; return `float("inf")` if no losses
  11. **`main()`**: argparse `--preregistration`; gate check; `load_holdout_bars`; `run_real_strategy` → `real_pf`; run seeds 0–49 → `seed_pfs`; filter `None` for percentile (report as N/A in table); `median = np.median(finite)`, `p90 = np.percentile(finite, 90)`; S12 verdict: `real_pf > p90` → `patterns_survive`; `real_pf > median` → `PIVOT (ambiguous)`; else → `PIVOT`; build + print report; write to `data/reports/s12_<timestamp>.txt`

**Acceptance Criteria:**
- Given valid `--preregistration 910e95c`, when script runs, then `ACCESS_LOG.md` gains one new row and `data/reports/s12_*.txt` is written with per-seed PF table, median, 90th-pct, real-strategy PF, and one S12 verdict line
- Given no flag, when holdout bars are loaded, then exit 1 with stderr containing "ACCESS_LOG" and "pre-registration"
- Given a seed with 0 trades, when report is generated, then that row shows N/A and the seed is excluded from median/percentile
- Given `real_pf > p90`, when verdict prints, then output contains `patterns_survive`
- Given `median < real_pf <= p90`, when verdict prints, then output contains `PIVOT` and `ambiguous`
- Given `real_pf <= median`, when verdict prints, then output contains `PIVOT`

## Design Notes

**Cold start symmetry:** Script loads only the holdout CSV. H1 and vol-regime history starts cold at bar 0 (2026-03-01). First ~20 H1 bars produce no vol-regime gate. This is symmetric across real strategy and all 50 seeds — does not bias the S12 comparison.

**Real strategy: bearish-only / gap-based SL-TP.** Random seeds: coin-flip direction / ATR-based SL-TP. This matches pre-registration §S12 steps 3–6 exactly.

**Tuesday check:** convert `bar.timestamp` to US/Eastern before calling `.weekday() == 1`, matching the deployed strategy's `bar_et.weekday()` check.

**VolRegimeTracker fresh per seed:** H1 resampling is deterministic (same bars), so the vol_regime state can be shared across seeds by pre-computing it before the seed loop. Build `vol_regime_per_bar: list[bool]` in one pass, then each seed reads from this list.

## Verification

**Commands:**
- `.venv/bin/python s12_random_entry_control.py --help` — expected: `--preregistration` shown
- `.venv/bin/python s12_random_entry_control.py 2>&1 | head -8; echo "Exit: $?"` — expected: access-denied message, exit 1
- `.venv/bin/python s12_random_entry_control.py --preregistration 910e95c 2>&1 | tail -25` — expected: 50-seed table + verdict + report path printed
- `ls -lh data/reports/s12_*.txt` — expected: new report file created in this run
- `tail -4 data/sealed_holdout/ACCESS_LOG.md` — expected: new row with `s12 script` and today's date

## Spec Change Log
