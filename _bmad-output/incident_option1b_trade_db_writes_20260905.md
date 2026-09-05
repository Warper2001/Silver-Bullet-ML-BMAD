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

---

## Follow-up (2026-09-05): second write vector, and the actual research impact

Digging into "what did this bug affect" surfaced a **second, worse-pathed write** and then established the impact is nonetheless near-zero.

### Second vector — `logs/tier2_trade_log.csv`

`TradeLogger.append_trade()` also appends every closed trade to `logs/tier2_trade_log.csv` at a path derived from `__file__` (`Path(__file__).parent.parent.parent / "logs" / ...`), **not** CWD. So this one resolved to the **main checkout's** `logs/` regardless of where the run was launched — the sqlite monkeypatch did not cover it, and this session's runs did append synthetic rows there (visible as 2×/4× consecutive-duplicate rows from the parallel workers).

**But that file was already a heavily-duplicated dump before this session:** 7,778 data rows, only 4,337 unique — **3,441 pre-existing excess duplicates**, some rows appearing up to 35 times, spanning the full 2025-01 → 2026-08 backtest history. It's the write target of *every* `run_backtest`/Tier2 backtest ever run in this project, it's **gitignored** (transient by design), and it is read only by ad-hoc perf-reporting scripts (`check_traders_perf.py`, `robust_report.py`, `report_from_logs_final.py`) — any report run against it was already wrong by ~44% duplication independent of this session. This session's contribution is an increment (~a few hundred row-writes, mostly duplicates of rows already present), not the origin of the problem.

**Latent risk worth flagging separately:** `migrate_trades.py` imports this CSV into `data/trades.db` as `trader-yank` / `backfilled` — the pipeline that produced the 1,841 `backfilled` trader-yank rows now in the ledger (migrated ~2026-06-23). Those rows predate this session by months; this session did not touch them. But if a migration is ever re-run against this CSV in its current state, it would import the accumulated duplicate garbage.

### Impact on this session's research findings: none

- **`data/trades.db` (the canonical ledger): confirmed untouched** — `trader-yank` 1854 rows (8 legacy + 1841 `backfilled` + 5 `realtime`), unchanged before/after, different inode from the isolated worktree file this session's sqlite writes hit.
- **`Tier2StreamingTrader` / `run_backtest` does not read `trades.db` or the CSV mid-run** (verified: `trade_db` is imported only at the `append_trade` write site, no read query anywhere in `tier2_streaming_working.py`). PF/trade results are computed from the in-memory `trades` list returned by `run_backtest`, so the write side effects cannot feed back into a backtest's numbers.
- **Option 1a (GAP-1)** and **Option 2 (overnight hold)** use `backtest_gap_fade`'s standalone pandas engine and my own script respectively — neither touches `Tier2StreamingTrader` at all.
- **Option 1b** produced no comparison result; its one confirmed number (baseline PF 2.125) is an in-memory computation, unaffected, and is filed as non-decision-bearing reference.
- **Options 3 & 4** are analysis of pre-existing `trades.db` rows (`trader-mim-nb`: 24, unchanged; `trader-yank`: the pre-session `backfilled` population) — not touched by this session.

### Net

The bug is real and the second (CSV) vector makes it worse-pathed than first documented, but its effect on research is a hygiene problem in a gitignored, already-corrupted transient log — not a corruption of any analysis input or any Option's verdict. The upstream fix (injectable `db_path` / a `--no-persist` backtest mode covering *both* the sqlite and CSV writes) is still needed before parallel/CI-style Tier2 backtesting is safe.
