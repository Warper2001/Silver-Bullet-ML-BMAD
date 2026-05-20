---
title: 'Program C Phase 1 — S13 Timeframe Replication'
type: 'feature'
created: '2026-05-20'
status: 'done'
baseline_commit: '39e4adf'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** S12 returned `patterns_survive`, requiring S13 timeframe replication before any Phase 2 work can begin. No script exists; the S13 verdict cannot be computed.

**Approach:** Implement `s13_timeframe_replication.py` that resamples the sealed holdout to 5-min and 15-min bars, runs the frozen Tier2 strategy at all three resolutions (1-min, 5-min, 15-min), and emits the pre-committed S13 verdict.

## Boundaries & Constraints

**Always:**
- Same preregistration gate as S12: `--preregistration <sha>` validates SHA, confirms commit contains `_bmad-output/preregistration*.md`, appends ACCESS_LOG with accessor label `"s13 script"` — before any holdout bar is read
- Frozen parameters match `preregistration_phase1.md` exactly: `SL_MULT=5.0`, `TP_MULT=6.0`, `MAX_HOLD=60` (bars at each resolution), `ATR_PERIOD=20`, `VOL_LOOKBACK=120`, `VOL_THRESH=0.75`, `ATR_THRESHOLD=0.5`, `MIN_GAP_ATR_RATIO=0.15`, `MAX_GAP_DOLLARS=60.0`, `MNQ_DOLLAR=2.0`, `ENTRY_PCT=0.5`, `BEARISH_ONLY=True`, `MAX_PENDING_BARS=240`
- Self-contained: copy all helpers from `s12_random_entry_control.py` verbatim; do not import it
- Report written to `data/reports/s13_<YYYYMMDD_HHMMSS>.txt` and stdout
- Re-run the 1-min strategy (do not hardcode S12 result); all three resolutions produced from the same `load_holdout_bars()` call

**Ask First:**
- If `data/sealed_holdout/ACCESS_LOG.md` is missing — halt and explain Phase 0.4 setup

**Never:**
- Read holdout bars before the ACCESS_LOG gate passes
- Adjust any frozen parameter after observing results

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| No `--preregistration` flag | bare invocation | Exit 1; stderr contains "pre-registration" and "ACCESS_LOG" | Print instructions |
| Bad / unknown SHA | `--preregistration zzz` | Exit 1 with SHA format or "not found" message | Regex + `git cat-file` check |
| Valid SHA | `910e95c` | Run 3 TFs, append ACCESS_LOG, write report | — |
| Resolution with 0 trades | all bars filtered at that TF | PF shows N/A in table; excluded from `best_TF_PF` | Guard `None` in max() |
| `best_TF_PF ≥ 1.1` | any TF PF ≥ 1.1 | verdict: `design_phase2_ml_test` | — |
| `best_TF_PF < 1.1` | all TFs PF < 1.1 | verdict: `PIVOT` | — |

</frozen-after-approval>

## Code Map

- `data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv` — input (1-min bars; columns: timestamp, open, high, low, close, volume, notional)
- `data/sealed_holdout/ACCESS_LOG.md` — appended once per valid invocation
- `s12_random_entry_control.py` — gate helpers and strategy logic reference; copy verbatim, do not import
- `_bmad-output/preregistration_phase1.md` — S13 exact procedure §S13 steps 1–6 and decision rule

## Tasks & Acceptance

**Execution:**

- [x] `s13_timeframe_replication.py` — new file; structure:
  1. **Constants block** — all frozen params, `HOLDOUT_CSV`, `HOLDOUT_CUTOFF = datetime(2026,3,1,utc)`, `ACCESS_LOG_PATH`, `REPORTS_DIR`, `_PREREG_PATTERNS`
  2. **Gate helpers** — copy `verify_preregistration` and `append_access_log` verbatim from `s12_random_entry_control.py:67–130` with accessor label changed to `"s13 script"`
  3. **`Bar` namedtuple + `load_holdout_bars()`** — copy verbatim from `s12_random_entry_control.py:132–153`
  4. **`resample_bars(bars, tf_min) -> list[Bar]`** — build DataFrame from `bars`, `df.index = timestamps`, `df.resample(f"{tf_min}min").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()`, return list of `Bar` namedtuples; raise `ValueError` if empty result
  5. **Shared helpers** — copy `calc_atr`, `build_h1_df`, `calc_h1_atr`, `VolRegimeTracker`, `is_market_open`, `is_tuesday_et`, `snap_tick`, `detect_bearish_sweep`, `profit_factor` verbatim from S12
  6. **`run_strategy(bars) -> tuple[float|None, int]`** — copy `run_real_strategy` verbatim from `s12_random_entry_control.py:282–368`; rename only; H1 boundary check `bar.timestamp.replace(minute=0, second=0, microsecond=0)` is resolution-agnostic and requires no changes
  7. **`build_report(results, sha) -> str`** — `results = {1: (pf, n), 5: (pf, n), 15: (pf, n)}`; table of TF / PF / Trades; `best_TF_PF = max(pf for pf,_ in results.values() if pf is not None)` (or None if all None); S13 verdict line per pre-registration §S13 decision rule
  8. **`main()`** — argparse `--preregistration`; gate check (`verify_preregistration` + `append_access_log`) before loading any bars; `bars_1m = load_holdout_bars()`; `bars_5m = resample_bars(bars_1m, 5)`; `bars_15m = resample_bars(bars_1m, 15)`; run `run_strategy` on each; `build_report`; print; write `data/reports/s13_<timestamp>.txt`

**Acceptance Criteria:**
- Given valid `--preregistration 910e95c`, when script runs, then ACCESS_LOG.md gains one new `s13 script` row and `data/reports/s13_*.txt` is written with PF for each of 3 TFs, `best_TF_PF`, and one S13 verdict line
- Given no flag, when invoked, then exit 1 with stderr containing "ACCESS_LOG" and "pre-registration"
- Given a resolution with 0 trades, when report is generated, then that row shows N/A and it is excluded from `best_TF_PF`
- Given `best_TF_PF ≥ 1.1`, when verdict prints, then output contains `design_phase2_ml_test`
- Given `best_TF_PF < 1.1`, when verdict prints, then output contains `PIVOT`

## Design Notes

**H1 boundary detection is resolution-agnostic.** The `bar.timestamp.replace(minute=0, ...)` pattern fires once per hour regardless of bar resolution. `build_h1_df` uses pandas `resample("1h")` which aggregates correctly from 5-min or 15-min input.

**MAX_HOLD=60 bars scales with resolution.** At 5-min resolution this is 5 hours; at 15-min it is 15 hours. The pre-registration fixes the multiplier (60), not the wall-clock duration. The same `MAX_HOLD` constant applies unchanged.

**Resampling preserves H1 boundary alignment.** MNQ 1-min bars are on-the-minute. 5-min and 15-min resample intervals align to 5:00/10:00/... and 15:00/30:00/... respectively, which are sub-multiples of the hour — H1 bars from `resample("1h")` are the same regardless.

## Verification

**Commands:**
- `.venv/bin/python s13_timeframe_replication.py --help` — expected: `--preregistration` shown
- `.venv/bin/python s13_timeframe_replication.py 2>&1 | head -5; echo "Exit: $?"` — expected: access-denied message, exit 1
- `.venv/bin/python s13_timeframe_replication.py --preregistration 910e95c 2>&1 | tail -20` — expected: 3-row TF table + verdict + report path printed
- `ls -lh data/reports/s13_*.txt` — expected: new report file created in this run
- `tail -3 data/sealed_holdout/ACCESS_LOG.md` — expected: new row with `s13 script` and today's date

## Spec Change Log

## Suggested Review Order

**Gate & Access Control**

- `--preregistration` guard fires before any holdout bars load; exit-1 path
  [`s13_timeframe_replication.py:484`](../../../s13_timeframe_replication.py#L484)

- SHA validation + ACCESS_LOG append; accessor label `"s13 script"`
  [`s13_timeframe_replication.py:67`](../../../s13_timeframe_replication.py#L67)

**Resampling (new logic)**

- `resample_bars()` — only new function vs S12; pandas resample to 5-min / 15-min
  [`s13_timeframe_replication.py:140`](../../../s13_timeframe_replication.py#L140)

**Strategy Simulation**

- `run_strategy()` — resolution-agnostic; H1 boundary via timestamp truncation
  [`s13_timeframe_replication.py:323`](../../../s13_timeframe_replication.py#L323)

- H1 boundary detection works at any sub-hourly resolution via `replace(minute=0)`
  [`s13_timeframe_replication.py:333`](../../../s13_timeframe_replication.py#L333)

**Verdict & Report**

- `build_report()` — best_TF_PF = max of non-None PFs; S13 threshold 1.1
  [`s13_timeframe_replication.py:404`](../../../s13_timeframe_replication.py#L404)

**Audit Trail**

- ACCESS_LOG entry for S13 run (verdict recorded)
  [`data/sealed_holdout/ACCESS_LOG.md:82`](../../../data/sealed_holdout/ACCESS_LOG.md#L82)
