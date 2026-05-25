# Story 8.4: Rolling Weekly Backtest Script

Status: done

## Story

As Alex (the researcher),
I want a `tools/weekly_backtest.py` script that backtests the current config on the most recent post-holdout 1-min data,
so I can get same-day feedback on how the deployed strategy behaves on fresh data each week.

## Background

**Why this matters:** With ~5-8 live trades/week (3 instruments), same-day assessment requires a backtest on recently-fetched data. The weekly workflow: fetch fresh 1-min bars → run `weekly_backtest.py` → review PF/WR/Sharpe → decide if config change is warranted.

**Post-holdout constraint:** Only data AFTER the holdout cutoff (`2026-05-19`) is valid. The `data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv` ends ON `2026-05-19` (it IS the holdout period). Fresh data must be fetched from TradeStation via a separate download step before running this tool.

**Data resolution (per symbol):** The tool looks for CSVs using the convention `data/processed/dollar_bars/1_minute/{prefix}_1min_{year}.csv`. If no post-holdout data is found in the date-filtered window, it outputs `N=0 (no post-holdout data)` for that symbol — not an error.

**BacktestEngine API:** `BacktestEngine(csv_path: str, config: StrategyConfig)` → `list[TradeRecord]`. `TradeRecord` has `pnl_usd: float` and `exit_reason: str`. The engine hardcodes MNQ point value; for MES/M2K a `point_value` scaling step is needed (noted as a future enhancement — out of scope for this story).

## Acceptance Criteria

1. `tools/weekly_backtest.py` exists and is runnable: `PYTHONPATH=. .venv/bin/python tools/weekly_backtest.py --help` exits 0.
2. `--config <yaml>` (optional, default `strategy_config.yaml`): loads `StrategyConfig` via `config_loader.load_strategy_config`.
3. `--weeks N` (default 4): uses the most recent N weeks of data after the holdout cutoff.
4. `--symbols MNQM26[,MESM26,M2KM26]` (default `MNQM26`): comma-separated list of symbols to backtest.
5. `--holdout-cutoff YYYY-MM-DD` (default `2026-05-19`): data before this date is excluded.
6. `--data-file <path>` (optional): override the CSV data file path for a single symbol (useful for testing). When provided with multiple symbols, applies to the first symbol only.
7. Per-symbol output table (one row per symbol + one POOLED row if multiple symbols):
   ```
   Symbol  | N  | PF   | WR   | TIME_STOP%
   MNQM26  | 8  | 1.34 | 0.50 | 12%
   POOLED  | 8  | 1.34 | 0.50 | 12%
   ```
8. When a symbol's data file is not found or has zero trades in the post-holdout window, outputs `N=0 (no data)` for that row — not an error exit.
9. All existing tests pass — `python -m pytest tests/unit/test_config_loader.py tests/unit/test_strategy_core_detection.py -q` green.

## Tasks / Subtasks

- [x] Task 1: Create `tools/` directory and `tools/weekly_backtest.py` skeleton (AC: #1)
  - [x] `tools/` created at repo root
  - [x] `tools/weekly_backtest.py` created with full argparse CLI
  - [x] `--help` exits 0 ✓

- [x] Task 2: Implement data loading and date filtering (AC: #2–#6)
  - [x] `_find_data_file(symbol)` — searches `{prefix}_1min_{year}[_ytd].csv` for current and prior year
  - [x] `_load_and_filter(csv_path, holdout_cutoff, weeks)` — loads CSV, UTC→NYC tz convert, filters post-cutoff, last N weeks
  - [x] `_write_temp_csv(df)` — writes to `tempfile.NamedTemporaryFile`, cleaned up after BacktestEngine run
  - [x] Missing file → N=0 "no data file"; no post-holdout rows → N=0 "no post-holdout data in window"

- [x] Task 3: Implement metrics computation and output table (AC: #7, #8)
  - [x] Per-symbol metrics: PF, WR, TIME_STOP%; N=0 rows print `—` for all metrics with note
  - [x] Output table with header, separator, per-symbol rows
  - [x] POOLED row printed when multiple symbols have N>0 data

- [x] Task 4: Wire together and smoke test (AC: #1–#8)
  - [x] Smoke test result: `--holdout-cutoff 2024-12-15 --weeks 2` → N=5, PF=0.259, WR=0.200, TIME_STOP=40% ✓
  - [x] Default (real post-holdout) → N=0 "0 trades in window" (correct: 2026 YTD ends at holdout date) ✓

- [x] Task 5: Run existing tests to confirm no regressions (AC: #9)
  - [x] 78 tests pass; no regressions

## Dev Notes

### Data File Resolution Convention

```python
SYMBOL_DATA_PREFIXES = {
    "MNQM26": "mnq",
    "MESM26": "mes",
    "M2KM26": "m2k",
}
DATA_DIR = Path("data/processed/dollar_bars/1_minute")

def _find_data_file(symbol: str, year: int) -> Path | None:
    prefix = SYMBOL_DATA_PREFIXES.get(symbol)
    if prefix is None:
        return None
    for y in (year, year - 1):
        p = DATA_DIR / f"{prefix}_1min_{y}_ytd.csv"
        if p.exists():
            return p
        # Also check without _ytd suffix
        p2 = DATA_DIR / f"{prefix}_1min_{y}.csv"
        if p2.exists():
            return p2
    return None
```

### Date Filtering

```python
def _load_and_filter(
    csv_path: Path,
    holdout_cutoff: str,
    weeks: int,
) -> pd.DataFrame | None:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    df = df.set_index("timestamp").sort_index()

    cutoff = pd.Timestamp(holdout_cutoff, tz="America/New_York")
    df = df[df.index > cutoff]

    if len(df) == 0:
        return None

    # Take last N calendar weeks
    end_ts = df.index[-1]
    start_ts = end_ts - pd.Timedelta(weeks=weeks)
    df = df[df.index >= start_ts]

    return df if len(df) > 0 else None
```

### Writing Temp CSV for BacktestEngine

BacktestEngine takes a CSV path (not a DataFrame). Write the filtered df to a temp file:

```python
import tempfile

def _write_temp_csv(df: pd.DataFrame) -> Path:
    df_out = df.reset_index()
    df_out["timestamp"] = df_out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df_out.to_csv(tmp.name, index=False)
    return Path(tmp.name)
```

### BacktestEngine API

```python
from src.research.backtest_engine import BacktestEngine
from src.research.strategy_core import StrategyConfig

engine = BacktestEngine(str(tmp_csv_path), config)
trades = engine.run()
# trades: list[TradeRecord]
# TradeRecord fields: pnl_usd (float), exit_reason (str)
```

### Metrics Calculation

```python
def _metrics(trades) -> dict:
    n = len(trades)
    if n == 0:
        return {"N": 0, "pf": float("nan"), "wr": float("nan"), "tstop_pct": float("nan")}
    pnls = [t.pnl_usd for t in trades]
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    wr = sum(1 for p in pnls if p > 0) / n
    tstop = sum(1 for t in trades if t.exit_reason == "TIME_STOP") / n * 100
    return {"N": n, "pf": pf, "wr": wr, "tstop_pct": tstop}
```

### Output Table Format

```
Symbol  |  N  |   PF   |  WR   | TIME_STOP%
MNQM26  |   8 |  1.340 | 0.500 |       12%
POOLED  |   8 |  1.340 | 0.500 |       12%
```

### Smoke Test Command

To test the tool works end-to-end (using 2025 data with a Dec 2024 cutoff):
```bash
PYTHONPATH=. .venv/bin/python tools/weekly_backtest.py \
  --holdout-cutoff 2024-12-15 \
  --weeks 2 \
  --symbols MNQM26 \
  --data-file data/processed/dollar_bars/1_minute/mnq_1min_2025.csv
```

Expected: runs without error; shows N≥0; if N=0, gracefully says "no data in window".

To test with no data (real post-holdout scenario):
```bash
PYTHONPATH=. .venv/bin/python tools/weekly_backtest.py --symbols MNQM26
```
Expected: `MNQM26  |   0 | — (no data in post-holdout window)`

### Point Value Limitation

`BacktestEngine` computes `pnl_usd` using `strategy_core.POINT_VALUE_USD = 2.0` (MNQ-specific). For MES/M2K, the P&L values from the engine are incorrect. This tool only claims accuracy for MNQ in this story — MES/M2K rows will show `pf/wr/tstop` based on MNQ point value until a future story adds per-instrument scaling.

### References

- `src/research/backtest_engine.py` lines 530–616: BacktestEngine API
- `src/research/config_loader.py`: `load_strategy_config(yaml_path)` 
- `data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv`: only goes to 2026-05-19
- `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv`: use for smoke testing

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

### Completion Notes List

- `tools/weekly_backtest.py` created: all 5 CLI args (`--config`, `--weeks`, `--symbols`, `--holdout-cutoff`, `--data-file`), data file auto-discovery, UTC→NYC tz conversion, `_load_and_filter`, `_write_temp_csv` + cleanup, `_run_all`, `_print_table`
- Graceful N=0 handling: "no data file", "no post-holdout data in window", "0 trades in window" — never crashes
- Smoke test confirmed: N=5 trades in last 2 weeks of Dec 2024 from 2025 data file
- Default run (real post-holdout) correctly reports N=0 — no post-2026-05-19 data yet exists (must fetch from TradeStation)
- Point value limitation documented: engine uses MNQ POINT_VALUE_USD=2.0; MES/M2K P&L would be inaccurate until future story adds per-instrument scaling

### Review Findings (2026-05-25)

**Patched (7 items — all applied):**
- [x] P1 HIGH: `DATA_DIR` relative to CWD — silent failure when run from `tools/`; fixed to `Path(__file__).parent.parent / ...`
- [x] P2 MED: `_compute_metrics` dead code (only called from dead `_run_symbol`) — removed
- [x] P3 MED: `_run_symbol` dead code (never called from `main` or `_run_all`) — removed
- [x] P4 MED: `pooled_trades_flat: list = []` unused variable in `_print_table` — removed
- [x] P5 LOW: `import os` unused — removed
- [x] P6 LOW: POOLED row WR column missing `>6` width specifier — fixed to `{wr:>6.3f}`
- [x] P7 HIGH: Duplicate POOLED block in `main()` (different unaligned format); `_print_table` already prints POOLED — `main()` block removed

**Deferred (2 items):**
- [ ] `--weeks` window anchored to last data row's timestamp, not today's date; stale data files produce narrower windows silently
- [ ] `epilog` crashes with `-OO` Python flag (strips docstrings → `__doc__` is `None` → `.split()` raises AttributeError)

**Dismissed (6 items):**
- `--data-file` applies to first symbol only — AC#6 explicitly documents this behavior
- `--config` silent fallback to defaults — documented in help text
- MES/M2K point value inaccuracy — documented in Dev Notes as known limitation
- AC#8 note string wording — acceptable intent match (graceful N=0 handling works correctly)
- POOLED only shown for multi-symbol — correct per AC#7
- `_write_temp_csv` temp file name collision — `NamedTemporaryFile` generates unique names

### File List

- `tools/weekly_backtest.py` (new)
