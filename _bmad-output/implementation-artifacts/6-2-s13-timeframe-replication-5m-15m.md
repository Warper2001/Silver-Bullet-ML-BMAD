# Story 6.2: S13 Timeframe Replication (5m / 15m)

Status: review

## Story

As Alex (researcher),
I want to run the Silver-Bullet strategy on 5-minute and 15-minute resamples of the 2025 MNQ data and compare PF/WR/Sharpe against the 1-min baseline (PF=0.937),
so that I have pre-registered cross-timeframe evidence for the Program C pivot-vs-survive verdict (Story 6.3).

## Acceptance Criteria

1. Pre-registration doc `_bmad-output/preregistration_s13_timeframe.md` is written **and committed to git** before any replication code runs. Hypothesis: "If the FVG+H1-sweep edge is real, PF should be consistently > 1.0 at 5m and 15m resamples."
2. `src/research/timeframe_replication.py` is implemented — resamples `mnq_1min_2025.csv` to 5m and 15m OHLCV, runs `BacktestEngine` on each, computes PF/WR/Sharpe/trade-count per timeframe.
3. PF, WR, Sharpe, and trade count are computed for the 5m and 15m runs.
4. Results are compared against the 1-min baseline (PF=0.937, 129 trades).
5. Verdict report `_bmad-output/s13_verdict_<YYYYMMDD>.md` is produced with: per-timeframe metrics table, pattern consistency assessment, and preliminary verdict.
6. No modifications to `src/research/strategy_core.py`, `src/research/backtest_engine.py`, or `src/research/tier2_streaming_working.py`.

## Tasks / Subtasks

- [x] Task 1 — Write and commit pre-registration doc (AC #1)
  - [x] Create `_bmad-output/preregistration_s13_timeframe.md`
  - [x] Include hypothesis, data source, timeframes (5m, 15m), StrategyConfig defaults (unchanged), no-modification guardrail
  - [x] `git add -f _bmad-output/preregistration_s13_timeframe.md && git commit` — record SHA
  - [x] Write the SHA into the pre-reg doc

- [x] Task 2 — Implement `src/research/timeframe_replication.py` (AC #2)
  - [x] Load 1-min bars from `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv`
  - [x] Resample to 5m and 15m using pandas (open=first, high=max, low=min, close=last, volume=sum)
  - [x] Save resampled bars to temp files, run `BacktestEngine` on each
  - [x] Compute PF, WR, Sharpe, trade count for each timeframe
  - [x] CLI: `.venv/bin/python src/research/timeframe_replication.py`

- [x] Task 3 — Run and collect results (AC #3, #4)
  - [x] Execute script; capture per-timeframe output
  - [x] Compare against 1-min baseline

- [x] Task 4 — Produce verdict report (AC #5)
  - [x] Write `_bmad-output/s13_verdict_20260523.md` with metrics table and pattern consistency assessment

- [x] Task 5 — Verify no strategy code modified (AC #6)
  - [x] Confirm only `timeframe_replication.py` was created in `src/research/`

## Dev Notes

### Program C Scope Guardrail

Do NOT modify `strategy_core.py`, `backtest_engine.py`, or `tier2_streaming_working.py`. Use `StrategyConfig()` with no arguments on all runs.

Pre-register before running. Force-add with `git add -f` (directory gitignored, consistent with S12/S25/S26 pattern).

### Resampling Pattern

```python
import pandas as pd

bars_1m = pd.read_csv(
    "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv",
    parse_dates=["timestamp"],
)
# tz handling identical to BacktestEngine._load_bars()
if bars_1m["timestamp"].dt.tz is None:
    bars_1m["timestamp"] = bars_1m["timestamp"].dt.tz_localize("UTC")
else:
    bars_1m["timestamp"] = bars_1m["timestamp"].dt.tz_convert("UTC")
bars_1m["timestamp"] = bars_1m["timestamp"].dt.tz_convert("America/New_York")
bars_1m = bars_1m.set_index("timestamp").sort_index()
bars_1m = bars_1m.drop(columns=["notional"], errors="ignore")

bars_5m = (
    bars_1m.resample("5min")
    .agg(open=("open","first"), high=("high","max"), low=("low","min"),
         close=("close","last"), volume=("volume","sum"))
    .dropna(subset=["open","high","low","close"])
)
bars_15m = (
    bars_1m.resample("15min")
    .agg(open=("open","first"), high=("high","max"), low=("low","min"),
         close=("close","last"), volume=("volume","sum"))
    .dropna(subset=["open","high","low","close"])
)
```

BacktestEngine expects a CSV path — write resampled bars to temp files:
```python
import tempfile, os
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
    bars_5m.to_csv(f)
    path_5m = f.name
```
Then `BacktestEngine(path_5m).run()`. Delete temp files after.

### Known Behavioral Deltas at 5m / 15m

These are expected and do not invalidate the test:

| Behavior | 1m baseline | 5m | 15m |
|---|---|---|---|
| FVG span | 3 min | 15 min | 45 min |
| `max_hold_bars=60` | 60 min hold | 300 min | 900 min |
| `max_pending_bars=240` | 4 hr pending | 20 hr pending | 60 hr pending |
| H1 resample | 60 bars/H1 | 12 bars/H1 | 4 bars/H1 |
| M1 ATR in detect_fvg | true 1-min ATR | 5-min ATR (larger) | 15-min ATR (larger) |

Document all deltas in the verdict report. Do not adjust StrategyConfig to compensate.

### BacktestEngine API

```python
from src.research.backtest_engine import BacktestEngine
from src.research.strategy_core import StrategyConfig, calc_profit_factor, calc_sharpe

engine = BacktestEngine(csv_path, config=StrategyConfig())
trades = engine.run()  # list[TradeRecord]
pnls = [t.pnl_usd for t in trades]
```

Temp file must have tz-aware timestamps in the `timestamp` column — BacktestEngine._load_bars() handles UTC→ET conversion.

When writing the resampled DataFrame to CSV, call `bars_5m.reset_index()` first so `timestamp` becomes a column (not the index). BacktestEngine expects `timestamp` as a named column in the CSV.

### Metrics

```python
pf = calc_profit_factor(pnls)
wr = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0.0
# Daily Sharpe — aggregate to daily first
daily = {}
for t in trades:
    d = t.timestamp_entry.date()
    daily[d] = daily.get(d, 0.0) + t.pnl_usd
sharpe = calc_sharpe(list(daily.values()))
```

### 1-min Baseline (from Epic 1)

| Metric | Value |
|---|---|
| Trades | 129 |
| PF | 0.937 |
| WR | ~46% |
| Sharpe | ~0 |
| TIME_STOP exits | ~65% |

### File Structure

```
_bmad-output/preregistration_s13_timeframe.md   ← NEW (commit BEFORE running)
src/research/timeframe_replication.py            ← NEW
_bmad-output/s13_verdict_<YYYYMMDD>.md          ← NEW (script output)
```

### References

- Epic 6 guardrails: `_bmad-output/implementation-artifacts/epic-6-program-c-phase-1-falsification.md`
- S12 pre-reg format: `_bmad-output/preregistration_s12_random_entry.md`
- BacktestEngine: `src/research/backtest_engine.py`
- strategy_core: `src/research/strategy_core.py`
- Data: `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv`

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6 (2026-05-23)

### Debug Log References

- Pre-registration commit: `5fde2d254277ab5b2943d608a1e8833d5a7243e2` (before any code)
- Temp file fix: BacktestEngine._load_bars() expects UTC timestamps in CSV; resampled ET bars must be converted to UTC before writing with `dt.tz_convert("UTC")`. Otherwise pandas `parse_dates=["timestamp"]` leaves column as object type.
- 5m run completed in ~5 min (67,610 bars); 15m run in ~2 min (23,595 bars)

### Completion Notes List

1. Pre-registration sealed at commit `5fde2d254277ab5b2943d608a1e8833d5a7243e2` before any replication code ran. Force-added with `git add -f` (directory gitignored; consistent with S12/S25/S26 pattern).
2. `timeframe_replication.py` resamples 1m→5m and 1m→15m in-process, writes temp UTC CSV, runs `BacktestEngine(tmp_path, config=StrategyConfig())`, cleans up temp file after.
3. **Results**: 5m — 86 trades, PF=1.026, WR=0.465, Sharpe=0.202 | 15m — 61 trades, PF=1.179, WR=0.475, Sharpe=1.373
4. **Consistency criterion SATISFIED**: both 5m AND 15m PF > 1.0. H₁ (alternative hypothesis) is supported.
5. **VERDICT: PATTERNS SURVIVE** — FVG+H1-sweep edge survives timeframe resampling.
6. Exit breakdown notable: 15m TIME_STOP drops from ~65% (1m) to 11% (7/61) — coarser bars give trades more time to resolve via TP/SL rather than timing out. This supports genuine pattern structure.
7. No modifications to `strategy_core.py`, `backtest_engine.py`, or `tier2_streaming_working.py`. Verified via `git status --short src/research/`.

### File List

- `_bmad-output/preregistration_s13_timeframe.md` — NEW (committed at `5fde2d2`)
- `src/research/timeframe_replication.py` — NEW
- `_bmad-output/s13_verdict_20260523.md` — NEW (produced by running the script)
