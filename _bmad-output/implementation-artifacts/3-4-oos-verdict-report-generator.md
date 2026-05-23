# Story 3.4: OOS Verdict Report Generator (oos_verdict.py)

Status: in-progress

## Story

As Alex,
I want an `oos_verdict.py` script that reads the sealed holdout data, computes OOS metrics, and produces a Go/No-Go verdict report,
So that the final performance verdict is tamper-evident, traceable to the pre-registration commit, and unambiguous.

## Acceptance Criteria

1. Given `oos_verdict.py` is invoked,
   When its very first action executes,
   Then it calls `checkpoint_or_abort()` from `oos_checkpoint.py` — if any check fails, it aborts before reading a single byte of holdout data (AR8)

2. Given all checkpoint checks pass and holdout files are read,
   When each file in `data/sealed_holdout/` is accessed,
   Then a timestamped entry is appended to `data/sealed_holdout/ACCESS_LOG.md` with: timestamp, prereg SHA, accessor script name, purpose, and result (AR8)

3. Given the holdout OOS backtest completes,
   When `oos_verdict.py` computes metrics,
   Then it calculates: realized Profit Factor (using `calc_profit_factor` from `strategy_core`), annualized Sharpe (using `calc_sharpe` with daily returns grouped by exit date), maximum drawdown percentage (using `calc_max_drawdown_pct` on pre-accumulated equity curve), and total trade count N (FR26)

4. Given pre-registered thresholds (PF ≥ 2.0, Sharpe ≥ 1.5, max DD ≤ 10%, N ≥ 200),
   When realized metrics are compared against thresholds,
   Then verdict is `GO` if all four pass, `NO-GO` if any metric fails at N ≥ 200, or `INCONCLUSIVE` if N < 200

5. Given the stopping rule (halt if PF < 1.1 after first 100 OOS trades),
   When `oos_verdict.py` is run with N = 105 and PF = 1.04,
   Then the report flags `STOPPING_RULE_TRIGGERED` regardless of other metrics — stopping rule takes priority over all other verdicts

6. Given the verdict is computed,
   When `oos_verdict.py` writes its output,
   Then the report is saved to `data/reports/oos_verdict_<YYYYMMDD>_<HHMMSS>.md` containing: prereg document path, git commit at seal time (hash_c from prereg), realized metrics table, threshold comparison table, and the final verdict (`GO` / `NO-GO` / `INCONCLUSIVE` / `STOPPING_RULE_TRIGGERED`)

## Tasks / Subtasks

- [x] Task 1 — Implement `oos_verdict.py` at repo root (ACs #1–#6)
  - [x] Define module-level constants: `HOLDOUT_DIR`, `HOLDOUT_CSV`, `ACCESS_LOG`, `REPORTS_DIR` and threshold constants
  - [x] Add `_append_access_log(access_log: Path, prereg_sha: str, result_summary: str) -> None` — appends a table row to ACCESS_LOG.md matching the existing format (see Dev Notes)
  - [x] Add `_compute_metrics(trades: list) -> tuple[float, float, float, int]` — returns `(pf, sharpe, max_dd_pct, n)` using the three `strategy_core` functions; group trades by exit date for daily returns; pre-accumulate equity for max_dd_pct
  - [x] Add `_determine_verdict(pf: float, sharpe: float, max_dd_pct: float, n: int) -> str` — applies rules in priority order: STOPPING_RULE first, then N-gate, then GO/NO-GO
  - [x] Add `_write_report(report_path: Path, prereg_path: Path, hashes: dict, trades: list, pf: float, sharpe: float, max_dd_pct: float, n: int, verdict: str) -> None` — writes the markdown verdict report
  - [x] Add `verdict(prereg_path: Path, holdout_csv: Path = HOLDOUT_CSV, access_log: Path = ACCESS_LOG, reports_dir: Path = REPORTS_DIR) -> int` — orchestrates the full run; AC #1 FIRST action is `checkpoint_or_abort(prereg_path)`; append to ACCESS_LOG after computing metrics; return 0 if GO, 1 otherwise
  - [x] Add `main()` with argparse: `--prereg <path>` (required); calls `sys.exit(verdict(args.prereg))`
  - [x] `verdict()` and all helper functions accept path parameters for testability (same pattern as `oos_checkpoint.checkpoint()`)

- [x] Task 2 — Unit tests `tests/unit/test_oos_verdict.py` (all ACs)
  - [x] Define `make_trades(n_wins, n_losses, ...)` fixture factory — MagicMock objects with `.pnl_usd` and `.timestamp_exit`
  - [x] `test_checkpoint_called_first_on_failure`: mock `checkpoint_or_abort` to raise `SystemExit(1)`, mock `BacktestEngine.run` to track calls → verify BacktestEngine never called (AC #1)
  - [x] `test_verdict_go`: N=260 (250 wins × $200, 10 losses × $100) → PF=50, MaxDD=2%, Sharpe high → `"GO"` (AC #4)
  - [x] `test_verdict_no_go_pf`: N=300, PF=1.143 (above stopping rule, below threshold) → `"NO-GO"` (AC #4)
  - [x] `test_verdict_no_go_sharpe`: mock `_compute_metrics` returns (2.5, 1.0, 0.05, 300) → `"NO-GO"` (AC #4)
  - [x] `test_verdict_no_go_maxdd`: 200 wins × $400 then 100 losses × $100 → MaxDD=12.5% → `"NO-GO"` (AC #4)
  - [x] `test_verdict_inconclusive`: N=150 trades → `"INCONCLUSIVE"` regardless of metrics (AC #4)
  - [x] `test_verdict_stopping_rule_at_105_pf_1_04`: N=105, PF<1.0 → `"STOPPING_RULE_TRIGGERED"` (AC #5)
  - [x] `test_stopping_rule_overrides_inconclusive`: N=105, PF<1.1 → `"STOPPING_RULE_TRIGGERED"` not `"INCONCLUSIVE"` (AC #5 priority)
  - [x] `test_stopping_rule_not_triggered_below_100`: N=99, PF<1.1 → `"INCONCLUSIVE"` (stopping rule requires N≥100)
  - [x] `test_access_log_appended`: run `verdict()` with tmp_path; assert ACCESS_LOG has `oos_verdict.py` and `OOS verdict run` (AC #2)
  - [x] `test_report_file_created`: run `verdict()` with tmp reports_dir; file `oos_verdict_*.md` exists, contains `"Profit Factor"`, `"VERDICT"`, hash_c (AC #6)
  - [x] `test_compute_metrics_daily_grouping`: two trades on same date → 1 daily return (sharpe=0 for <2 samples)
  - [x] `test_zero_trades_inconclusive`: empty trade list → `"INCONCLUSIVE"`, N=0, no crash
  - [x] `test_access_log_appended_before_report`: verify log write precedes report write (AR8 ordering)
  - [x] `test_determine_verdict_logic`: direct unit test of all four verdict branches
  - [x] Run: `.venv/bin/python -m pytest tests/unit/test_oos_verdict.py -v` → 15/15 pass

- [x] Task 3 — Full regression test suite
  - [x] 80/80 tests pass with no regressions

## Dev Notes

### File location (AR5)

`oos_verdict.py` lives at the **repo root** — same level as `oos_checkpoint.py`, `protect_holdout.py`, and `prereg_seal.py`. Do NOT put it in `src/`.

### AR8 — First action is non-negotiable

The **very first statement** in `verdict()` (before opening any file, before any import of backtest modules) must be:

```python
from oos_checkpoint import checkpoint_or_abort
checkpoint_or_abort(prereg_path)
```

If `checkpoint_or_abort` raises `SystemExit(1)`, execution stops and no holdout byte is ever read. Tests must verify that `BacktestEngine.run()` is never called when the checkpoint fails.

### BacktestEngine import — CRITICAL

`BacktestEngine` is in **`src/research/backtest_engine.py`**, NOT in `strategy_core.py`. Import as:

```python
from src.research.backtest_engine import BacktestEngine, TradeRecord
```

`BacktestEngine.__init__` takes `csv_path: str` (not Path). `BacktestEngine.run() -> list[TradeRecord]`.

`TradeRecord` is a frozen dataclass with fields:
- `timestamp_exit: pd.Timestamp` — needed to group by date for daily Sharpe
- `pnl_usd: float` — the per-trade P&L in USD

### Metric function imports and input contracts

```python
from src.research.strategy_core import calc_profit_factor, calc_sharpe, calc_max_drawdown_pct
```

**`calc_profit_factor(pnls: list[float]) -> float`**
- Input: `[t.pnl_usd for t in trades]`
- Returns `float('inf')` when no losing trades; `0.0` when no winning trades

**`calc_sharpe(daily_returns: list[float]) -> float`**
- Input: **DAILY returns** (NOT per-trade PnL). Group by `t.timestamp_exit.date()` and sum pnl_usd per day:
  ```python
  from collections import defaultdict
  daily: dict = defaultdict(float)
  for t in trades:
      daily[t.timestamp_exit.date()] += t.pnl_usd
  daily_returns = list(daily.values())
  ```
- Returns annualized Sharpe: `sqrt(252) * mean / std`; returns `0.0` for <2 samples

**`calc_max_drawdown_pct(equity: list[float]) -> float`**
- Input: **pre-accumulated equity curve** (cumulative P&L). NOT per-trade PnL.
  ```python
  import itertools
  equity = list(itertools.accumulate(t.pnl_usd for t in trades))
  ```
- Returns a fraction: `0.073` means 7.3% drawdown. Threshold comparison: `max_dd_pct <= 0.10` for ≤ 10%

### Verdict logic — exact priority order

```python
THRESHOLD_PF = 2.0
THRESHOLD_SHARPE = 1.5
THRESHOLD_MAX_DD_PCT = 0.10   # fraction — 10%
THRESHOLD_N = 200

STOPPING_RULE_N = 100
STOPPING_RULE_PF = 1.1

def _determine_verdict(pf, sharpe, max_dd_pct, n):
    # Priority 1: stopping rule (overrides everything)
    if n >= STOPPING_RULE_N and pf < STOPPING_RULE_PF:
        return "STOPPING_RULE_TRIGGERED"
    # Priority 2: insufficient sample
    if n < THRESHOLD_N:
        return "INCONCLUSIVE"
    # Priority 3: all metrics pass → GO
    if pf >= THRESHOLD_PF and sharpe >= THRESHOLD_SHARPE and max_dd_pct <= THRESHOLD_MAX_DD_PCT:
        return "GO"
    # Priority 4: any metric fails → NO-GO
    return "NO-GO"
```

### ACCESS_LOG append format

The ACCESS_LOG has a **pipe-delimited Markdown table** with columns:
`| Date | SHA (pre-registration) | Accessor | Purpose | Result |`

The **hash_c** from the prereg doc (the git SHA at time of sealing) is the "SHA (pre-registration)". The result row format must match the existing table rows exactly:

```python
ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
row = (
    f"| {ts} | `{hash_c}` | oos_verdict.py | OOS verdict run — {prereg_path.name} "
    f"| {verdict}: PF={pf:.4f}, N={n}, Sharpe={sharpe:.3f}, MaxDD={max_dd_pct:.1%} |"
)
# Append with leading newline — file may not end with newline
with open(access_log, "a") as f:
    f.write("\n" + row)
```

Append **after** computing the verdict (so the log row includes the full result), but always before writing the final report and before printing to stdout. This ensures the audit trail is complete even if report writing fails.

### Report file format and location

Output path: `data/reports/oos_verdict_{YYYYMMDD}_{HHMMSS}.md`

```python
ts = datetime.now(timezone.utc)
report_path = reports_dir / f"oos_verdict_{ts.strftime('%Y%m%d_%H%M%S')}.md"
```

Minimum required sections in the report (AC #6):
1. Title and timestamp
2. Pre-registration reference: path to prereg doc, hash_c (sealed commit SHA), hash_a (config SHA)
3. Metrics table: realized PF, Sharpe, MaxDD%, N vs. threshold for each
4. Threshold comparison table: pass/fail per metric
5. Stopping rule status (triggered or not)
6. **Final verdict line: `## VERDICT: {verdict}`** (test will assert `"VERDICT"` in report)

### ACCESS_LOG path — do not hardcode write to real file in tests

The `verdict()` function must accept `access_log: Path` as a parameter (defaulting to `HOLDOUT_DIR / "ACCESS_LOG.md"`). Tests pass `tmp_path / "ACCESS_LOG.md"` — never write to the real `data/sealed_holdout/ACCESS_LOG.md` in unit tests.

Similarly, `holdout_csv: Path` and `reports_dir: Path` are parameters with defaults. Tests pass tmp paths with fake data.

### Testability pattern for BacktestEngine

In tests, mock the run method so no real holdout file is needed:

```python
from unittest.mock import MagicMock, patch
import pandas as pd

def make_trades(n_wins, n_losses, win_pnl=200.0, loss_pnl=-100.0):
    """Return a list of simple TradeRecord-like objects."""
    # Use MagicMock with spec so attribute access works
    trades = []
    base_date = pd.Timestamp("2026-03-05", tz="America/New_York")
    for i in range(n_wins + n_losses):
        t = MagicMock()
        t.pnl_usd = win_pnl if i < n_wins else loss_pnl
        t.timestamp_exit = base_date + pd.Timedelta(days=i)
        trades.append(t)
    return trades

# In test:
with (
    patch("oos_verdict.checkpoint_or_abort"),   # bypass gate
    patch("oos_verdict.BacktestEngine") as mock_engine,
):
    mock_engine.return_value.run.return_value = make_trades(200, 50)
    rc = verdict(prereg_doc, holdout_csv=tmp_holdout, access_log=tmp_log, reports_dir=tmp_reports)
```

Note: `checkpoint_or_abort` is called **before** BacktestEngine is instantiated, so patch the function in the `oos_verdict` module's namespace (not in `oos_checkpoint`).

### Import guard — PYTHONPATH

All imports of `src.*` require `PYTHONPATH=.` when running from repo root:

```bash
PYTHONPATH=. .venv/bin/python oos_verdict.py --prereg <path>
```

In tests, `conftest.py` already adds the repo root to sys.path, so no additional setup is needed.

### Do NOT modify

- `src/research/strategy_core.py` — metric functions are already correct
- `oos_checkpoint.py` — the public API is stable; `checkpoint_or_abort` is the contract
- `protect_holdout.py` — holdout protection is handled by oos_checkpoint check (e)
- `prereg_seal.py` — pre-registration sealing is a separate concern

### Holdout CSV path

The actual holdout data file is:
`data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv`

This is the file `BacktestEngine` will be pointed at. The path is a module-level constant in `oos_verdict.py`.

### Previous story learnings (3.1–3.3)

- Use `Path.read_text()` / `open(..., "a")` — no binary modes needed for MD/CSV files
- Module-level constants for paths; all public functions accept them as optional params for testability
- `subprocess.run` for git operations (same pattern as oos_checkpoint.py)
- All tests use `tmp_path` and `unittest.mock.patch` — never touch real `data/sealed_holdout/`
- `_parse_prereg` is already implemented in `oos_checkpoint.py` — import and reuse it to extract hash_c

## Dev Agent Record

### Implementation Plan

Module-level imports for `checkpoint_or_abort`, `BacktestEngine`, and metric functions — required for `patch()` to work in tests (lazy imports inside functions are not patchable by module path). Public `verdict()` function accepts all paths as optional parameters matching the `oos_checkpoint.checkpoint()` testability pattern. Execution order strictly: checkpoint → parse hashes → run backtest → compute metrics → determine verdict → append log → write report.

### Debug Log

- Initial lazy imports failed patching: `patch("oos_verdict.checkpoint_or_abort")` raised `AttributeError` since the name didn't exist in module namespace. Fixed by moving all three import groups to module level.
- `test_verdict_go` initially returned NO-GO: wins-then-losses created MaxDD=16.7% (above 10% threshold). Fixed by using 250 wins × $200 then 10 losses × $100 (MaxDD=2%).
- `test_verdict_no_go_pf` triggered stopping rule: 150/150 equal trades gave PF=1.0 < 1.1. Fixed to 160/140 giving PF=1.143.
- `test_verdict_no_go_sharpe` triggered stopping rule (PF=1.008 < 1.1). Fixed by mocking `_compute_metrics` directly to return (2.5, 1.0, 0.05, 300).
- `test_verdict_no_go_maxdd` returned GO (MaxDD=0%): losses-before-wins means equity never positive when losses happen, so `calc_max_drawdown_pct` computes 0%. Fixed to wins-first (MaxDD=12.5%).

### Completion Notes

`oos_verdict.py` implemented at repo root (AR5). All 6 ACs satisfied:
- AC #1: `checkpoint_or_abort()` is first module-level import and first call in `verdict()`. Verified by test that BacktestEngine is never called on checkpoint failure.
- AC #2: `_append_access_log()` appends pipe-delimited Markdown table row to ACCESS_LOG.md after computing metrics and before writing report.
- AC #3: `_compute_metrics()` groups trades by `timestamp_exit.date()` for daily Sharpe; pre-accumulates equity with `itertools.accumulate` for `calc_max_drawdown_pct`.
- AC #4/5: `_determine_verdict()` applies stopping rule first, then N-gate, then GO/NO-GO.
- AC #6: Report written to `data/reports/oos_verdict_{YYYYMMDD}_{HHMMSS}.md` with prereg hashes, metrics table, threshold table, stopping rule status, and VERDICT line.
15 new tests, 80 total tests pass, 0 regressions.

## File List

- [x] `oos_verdict.py` (NEW — repo root)
- [x] `tests/unit/test_oos_verdict.py` (NEW)

## Change Log

| Date | Change |
|---|---|
| 2026-05-24 | Story created — Story 3.4, Epic 3 final story |
| 2026-05-24 | Implementation complete — `oos_verdict.py` + 15 unit tests; 80/80 regression suite passes |

## Status

review
