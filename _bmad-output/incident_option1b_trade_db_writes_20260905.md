# Near-miss: Option 1b backtest runs write to a live-shaped SQLite path

**Date:** 2026-09-05
**Severity:** near-miss, no actual data loss or contamination confirmed.

## What happened

`Tier2StreamingTrader._close_active_trade()` logs every closed trade via `src.monitoring.trade_db.TradeDatabase()`, which defaults to `db_path="data/trades.db"` — a **relative** path, with no constructor injection point. `backtest_tier2_1year_validation.run_backtest()`'s mocked-broker setup does not mock or redirect this — every backtest run using this engine writes real rows to whatever `data/trades.db` resolves to under the process's CWD.

Discovered when a run launched from `tools/` (CWD mismatch) raised `sqlite3.OperationalError: unable to open database file` — `tools/data/` doesn't exist. Investigating why exposed the underlying issue: earlier runs launched from the worktree root **succeeded** by writing 286 synthetic YANK trade rows into `.claude/worktrees/post-r3-options-research/data/trades.db` — a file that did not exist before this session and was created fresh by `sqlite3.connect()`.

## What was checked immediately

- **Real ledger (`/root/Silver-Bullet-ML-BMAD/data/trades.db`): confirmed untouched.** `trader-yank` row count before and after: **1854, unchanged**. File mtime (2026-09-05 00:01:24) predates this session's later diagnostic runs. Confirmed on a **different inode** (542467) from the worktree-local file (7876832) — no shared storage, no risk of the isolated file's writes bleeding into the real one.
- The worktree-local file contained **only** `trader-yank` rows (286), no other trader IDs — consistent with it being a fresh, empty file that only ever received this session's synthetic writes, not a corrupted copy of the real ledger.
- **The one run launched with CWD = the main checkout root** (the original timing pilot, `cd /root/Silver-Bullet-ML-BMAD && ...`, killed after ~27 min) is the one invocation that *could* have written to the real path — direct evidence (unchanged row count, older mtime) says it did not reach a trade-close event before being killed.

## Fix

`tools/option1b_yank_horizon_sweep.py::_run_one` now monkeypatches `TradeDatabase.__init__` to always use a fresh `tempfile.mkdtemp()` path, unconditionally, regardless of what's passed or what the CWD is — every worker process gets its own throwaway db file, deleted with the temp dir. This is a workaround at the call site, not a fix to `trade_db.py`/`tier2_streaming_working.py` themselves (out of scope for this research task) — **any other script that calls `run_backtest()` from a CWD where `data/` exists is still exposed to the same risk** and should apply the same pattern or a proper upstream fix (an injectable `db_path` on `Tier2StreamingTrader`, or a `--no-persist` backtest flag) before being trusted for parallel/CI-style runs.

## Cleanup performed

- Deleted the worktree-local `data/trades.db` (286 synthetic rows, confirmed isolated).
- No other stray `.db`/WAL/SHM files found; `git status` on `data/` is clean.
