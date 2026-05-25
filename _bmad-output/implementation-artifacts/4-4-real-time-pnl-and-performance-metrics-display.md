# Story 4.4: Real-Time P&L and Performance Metrics Display

Status: done

## Story

As Alex,
I want real-time Profit Factor, Sharpe, max drawdown, trade count, and equity curve output computed from the cumulative OOS trade log and displayed after each completed trade,
So that I can monitor strategy performance continuously without manually running analysis scripts.

## Acceptance Criteria

**AC#1 — Shared metric functions used (no reimplementation):**
Given the OOS trade log CSV with N completed trades,
When `calc_profit_factor()`, `calc_sharpe()`, `calc_max_drawdown_pct()` from `strategy_core` are called on the accumulated PnL list,
Then metrics are returned using the same shared functions as the backtest engine — no reimplemented calculation paths (FR30, FR31, FR32).

**AC#2 — Metrics displayed after each trade:**
Given metrics are computed after each new trade closes,
When they are logged,
Then output includes: `PF: 2.14 | Sharpe: 1.67 | MaxDD: 7.3% | Trades: 79` (FR30–FR33).

**AC#3 — Equity curve written to disk after each trade:**
Given the equity curve,
When it is written to disk after each trade close,
Then `logs/equity_curve.csv` is updated (appended) with columns `timestamp, cumulative_pnl_usd, trade_count` (FR34, NFR17).

**AC#4 — Filter decision log per bar:**
Given each polling bar where a filter decision is made,
When the bar is processed through `_detect_and_enter()`,
Then one row is appended to `logs/tier2_filter_log.csv` containing:
`bar_timestamp, h1_sweep_active, kill_zone_active, vol_regime_blocked, m15_confirmed, fvg_detected, action`
where `action` is one of `ENTER / SKIP / HOLD / EXIT` (FR35).

**AC#5 — Zero-trade case shows N/A without error:**
Given a session with 0 closed trades,
When metrics are displayed,
Then: `PF: N/A | Sharpe: N/A | MaxDD: 0.0% | Trades: 0` — no division-by-zero (FR30).

**AC#6 — Unit tests:**
Given `tests/unit/test_realtime_metrics_display.py`,
When pytest runs it,
Then all tests pass covering:
- Metric display with N=0 (N/A fallbacks)
- Metric display with N>0 (correct PF/Sharpe/MaxDD from strategy_core)
- Equity curve CSV written with correct columns
- Filter log row written with correct fields and action values

## Tasks / Subtasks

- [x] Task 1: Add `_log_trade_metrics()` method to `Tier2StreamingTrader` (AC: #1, #2, #5)
  - [x] Import `calc_profit_factor`, `calc_sharpe`, `calc_max_drawdown_pct` from `strategy_core`
  - [x] `_log_trade_metrics()` uses `self.completed_trades` for pnl accumulation
  - [x] Handles N=0 → "N/A" with no division-by-zero
  - [x] Called at end of `_close_active_trade()` after `append_trade`

- [x] Task 2: Add `_write_equity_curve()` method (AC: #3)
  - [x] Writes `logs/equity_curve.csv` with columns `timestamp, cumulative_pnl_usd, trade_count`
  - [x] Appends one row per trade; creates file+header if not exists
  - [x] Wrapped in try/except — write failure swallowed

- [x] Task 3: Expand filter decision log (AC: #4)
  - [x] `_log_filter_decision()` added to `Tier2StreamingTrader`
  - [x] Writes to `logs/tier2_bar_decisions.csv` with all required fields
  - [x] Called from `_detect_and_enter()`: HOLD on active trade, SKIP on vol_regime_high, ENTER/SKIP on FVG result

- [x] Task 4: Write unit tests (AC: #6)
  - [x] 10 tests in `tests/unit/test_realtime_metrics_display.py` — all pass

- [x] Task 5: All tests pass, no regressions

## Dev Notes

### AR3 — Single-File Mandate

All new methods go in `src/research/tier2_streaming_working.py`. Do NOT create helper modules.

### Metric Accumulation

`TradeLogger` writes to `logs/tier2_filter_log.csv` (old MetaLabeling log path) and `TradeRecord` is the canonical per-trade record. For in-session metric accumulation, read pnl from `self.completed_trades` (list of `CompletedTrade` objects already maintained in `Tier2StreamingTrader`). This avoids re-reading the CSV on every trade.

```python
pnls = [t.pnl for t in self.completed_trades]
```

`calc_sharpe` takes **daily** returns, not per-trade. For the streaming display, use per-trade PnL as a proxy (same behavior as backtest engine's `per_trade_sharpe`). This is consistent with the backtest code (see `backtest_tier2_1year_validation.py:176`).

### Equity Curve Path

Use `Path(__file__).parent.parent.parent / "logs/equity_curve.csv"` — same convention as the MetaLabelingFilter's filter log at line 644.

### Filter Decision Log Extension

The existing `MetaLabelingFilter._log_decision()` writes `timestamp, filter_decision, probability, threshold` to `logs/tier2_filter_log.csv`. Story 4-4 adds a SEPARATE, richer per-bar log — do NOT modify `MetaLabelingFilter`. Add `_log_filter_decision()` to `Tier2StreamingTrader` writing to `logs/tier2_filter_log.csv` with the extended columns.

To avoid column conflicts, write to a separate file: `logs/tier2_bar_decisions.csv` with columns `bar_timestamp, h1_sweep_active, kill_zone_active, vol_regime_blocked, m15_confirmed, fvg_detected, action`. This keeps the ML filter log clean.

### Zero-Trade Safety

```python
if not pnls:
    logger.info("PF: N/A | Sharpe: N/A | MaxDD: 0.0%% | Trades: 0")
    return
pf = calc_profit_factor(pnls)
pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
sh = calc_sharpe(pnls)
cum_pnl = list(itertools.accumulate(pnls))
dd = calc_max_drawdown_pct(cum_pnl)
logger.info("PF: %s | Sharpe: %.2f | MaxDD: %.1f%% | Trades: %d", pf_str, sh, dd*100, len(pnls))
```

### Strategy Core Import

Add at top of file (near existing `from src.research.strategy_core import ...`):
```python
from src.research.strategy_core import (
    ...existing imports...,
    calc_profit_factor,
    calc_sharpe,
    calc_max_drawdown_pct,
)
```

Check line ~23 for the current strategy_core import block.

### Key File Locations

- `src/research/tier2_streaming_working.py` — all new methods added here
- `tests/unit/test_realtime_metrics_display.py` — new test file
- `logs/equity_curve.csv` — new log file (created at runtime; no fixture needed)
- `logs/tier2_bar_decisions.csv` — new filter decision log (created at runtime)

## Dev Agent Record

### File List

- `src/research/tier2_streaming_working.py` — modified
- `tests/unit/test_realtime_metrics_display.py` — new

### Change Log

- 2026-05-25: Story created
