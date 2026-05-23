# Story 2.2: AM Kill Zone Filter (09:30–11:00 ET, DST-Aware) at 15m

Status: review

## Story

As Alex (researcher),
I want to activate the AM kill zone filter in BacktestEngine and measure its impact on the 15m strategy at the training window,
so that I can determine whether restricting entries to 09:30–11:00 ET improves PF over the bearish-only 15m baseline (61 trades, PF=1.179).

## Acceptance Criteria

1. Pre-registration doc `_bmad-output/preregistration_s_kz_15m.md` written **and committed** before running any backtest. Hypothesis: "Kill zone restricted bearish 15m trades (09:30–11:00 ET) show PF > 1.3 with N ≥ 15."
2. `enable_kill_zone_filter: bool = False` added to `StrategyConfig` in `strategy_core.py` (default `False` — no behavior change for existing callers).
3. `backtest_engine.py` updated to block entry when `config.enable_kill_zone_filter=True` and `kill_zone_filter()` returns `False`. One-liner guard added after `kz = kill_zone_filter(bar_ts, config)`.
4. `src/research/kz_15m_test.py` implemented — loads 2025 training CSV, resamples to 15m, runs `BacktestEngine(StrategyConfig(bearish_only=True, enable_kill_zone_filter=True))`, verifies all entries are in [09:30, 10:59 ET] (DST-aware), compares against S13 baseline.
5. Verdict report `_bmad-output/s_kz_15m_verdict_<date>.md` produced with N, PF, WR, Sharpe, exit breakdown, DST verification result, and H₁/H₀ verdict.
6. Integration test `tests/integration/test_kill_zone_filter_integration.py` added: runs BacktestEngine with `enable_kill_zone_filter=True` on synthetic data and asserts all returned trades have `kill_zone_active=True`.
7. No modifications to `tier2_streaming_working.py`.

## Tasks / Subtasks

- [x] Task 1 — Pre-register and commit (AC #1)
  - [x] Write `_bmad-output/preregistration_s_kz_15m.md` with hypothesis, data, config snapshot, stopping rule
  - [x] `git add -f _bmad-output/preregistration_s_kz_15m.md && git commit` — record SHA: df66bd9
  - [x] Write SHA into pre-reg doc (follow-up commit 6d5e086)

- [x] Task 2 — Add `enable_kill_zone_filter` to StrategyConfig (AC #2)
  - [x] Add `enable_kill_zone_filter: bool = False` to StrategyConfig in `strategy_core.py` after `commission_per_roundtrip`
  - [x] Confirm no existing tests break (the new field has a default, so all existing `StrategyConfig()` calls are unaffected)

- [x] Task 3 — Wire kill zone blocking in BacktestEngine (AC #3)
  - [x] In `backtest_engine.py`, after `kz = kill_zone_filter(bar_ts, config)` (line ~818), add: `if config.enable_kill_zone_filter and not kz: continue`
  - [x] Run existing integration tests to verify no regressions

- [x] Task 4 — Implement `src/research/kz_15m_test.py` (AC #4)
  - [x] Copy `load_and_resample()` pattern exactly from `bidir_15m_test.py`
  - [x] Run BacktestEngine with `StrategyConfig(bearish_only=True, enable_kill_zone_filter=True)` (KZ-filtered run)
  - [x] Run BacktestEngine with `StrategyConfig(bearish_only=True, enable_kill_zone_filter=False)` (baseline verification — reproduced exactly 61 trades)
  - [x] DST verification: PASS — all 5 entries in [09:30, 11:00) ET
  - [x] Compute metrics for both runs (PF, WR, Sharpe, N, exit_counts)
  - [x] Apply H₁/H₀ decision logic (PF > 1.3 AND N ≥ 15)

- [x] Task 5 — Produce verdict report (AC #5)
  - [x] Write `_bmad-output/s_kz_15m_verdict_20260523.md` with full results table and verdict

- [x] Task 6 — Add integration test (AC #6)
  - [x] Write `tests/integration/test_kill_zone_filter_integration.py`
  - [x] Run with `.venv/bin/python -m pytest tests/integration/test_kill_zone_filter_integration.py -v` — 3/3 passed

- [x] Task 7 — Full test suite verification (AC #7)
  - [x] `.venv/bin/python -m pytest tests/ -x -q` — 46 passed (no regressions)

## Dev Notes

### 15m Reframe Context

Story 2.2 is Epic 2 story 2, reframed for 15m per `_bmad-output/phase2_verdict_20260523.md`. The original epic spec described kill zone activation for the 1m strategy; this implementation runs the same experiment at the OOS-validated 15m timeframe.

### Bearish-Only Baseline (reference from Story 2.1 / S13)

| Metric | Value |
|---|---|
| Trades | 61 |
| PF | 1.179 |
| WR | 0.475 |
| Daily Sharpe | 1.373 |
| TIME_STOP % | 11% |

Story 2.1 (bidirectional) failed — `bearish_only=True` is load-bearing. Story 2.2 keeps `bearish_only=True` and adds kill zone restriction.

### Kill Zone Infrastructure — What Already Exists (DO NOT REIMPLEMENT)

**`strategy_core.py` already has:**

```python
# In StrategyConfig (lines 96-98):
kill_zone_start_et: time = time(9, 30)
kill_zone_end_et: time = time(11, 0)
commission_per_roundtrip: float = 4.0  # ← ADD enable_kill_zone_filter AFTER THIS

# Standalone function (lines 584-600):
_NY_TZ = zoneinfo.ZoneInfo("America/New_York")

def kill_zone_filter(bar_timestamp: pd.Timestamp, config: StrategyConfig) -> bool:
    """Return True iff bar_timestamp falls in [kill_zone_start_et, kill_zone_end_et) ET."""
    bar_ny = bar_timestamp.astimezone(_NY_TZ)
    bar_time = bar_ny.time()
    return config.kill_zone_start_et <= bar_time < config.kill_zone_end_et
```

DST correctness is fully tested — **do not touch `test_strategy_core_killzone.py`**, 18 boundary tests already cover spring-forward 2026-03-08 and fall-back 2026-11-01.

**`backtest_engine.py` already has (lines ~817-830):**

```python
# Kill zone (logged, not a blocking filter — matches reference)
kz = kill_zone_filter(bar_ts, config)

# Entry decision
entry = make_entry_decision(sweep, fvg, config, vol_ok=vol_ok_cached)
if entry is None:
    continue

# Arm pending trade
active = entry
active_ts = bar_ts
active_gap = fvg.gap_size
active_sweep_ago = sweep.bars_ago
active_kz = kz    # ← already stored in TradeRecord
```

`TradeRecord.kill_zone_active: bool` is already a field — populated and written to CSV.

### Exact Changes Required

#### `strategy_core.py` — StrategyConfig (one line added)

```python
# BEFORE (current state):
    commission_per_roundtrip: float = 4.0

# AFTER:
    commission_per_roundtrip: float = 4.0
    enable_kill_zone_filter: bool = False  # if True, blocks entries outside kill zone
```

No imports needed — `bool` is a builtin.

#### `backtest_engine.py` — Entry detection loop (two lines added)

```python
# BEFORE (current state, lines ~817-818):
            # Kill zone (logged, not a blocking filter — matches reference)
            kz = kill_zone_filter(bar_ts, config)

# AFTER:
            kz = kill_zone_filter(bar_ts, config)
            if config.enable_kill_zone_filter and not kz:
                continue  # outside kill zone — skip this entry candidate
```

The comment "not a blocking filter" becomes stale after this change — remove it.

### Research Script Pattern (`src/research/kz_15m_test.py`)

Follow `bidir_15m_test.py` exactly. Key constants:

```python
CSV_PATH = "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
PRE_REG_SHA = "<filled in after pre-reg commit>"

BASELINE = {"trades": 61, "pf": 1.179, "wr": 0.475, "sharpe": 1.373}
PF_THRESHOLD = 1.3
MIN_TRADES = 15

import zoneinfo
NY_TZ = zoneinfo.ZoneInfo("America/New_York")
```

**Two BacktestEngine runs:**

```python
# Run 1 — KZ filtered
engine_kz = BacktestEngine(tmp_path, config=StrategyConfig(
    bearish_only=True, enable_kill_zone_filter=True
))
kz_trades = engine_kz.run()

# Run 2 — Full window (baseline verification)
engine_full = BacktestEngine(tmp_path, config=StrategyConfig(
    bearish_only=True, enable_kill_zone_filter=False
))
full_trades = engine_full.run()
```

**DST verification assertion (required by AC #4):**

```python
from datetime import time as dtime
for t in kz_trades:
    entry_ny = t.timestamp_entry.astimezone(NY_TZ).time()
    assert dtime(9, 30) <= entry_ny < dtime(11, 0), (
        f"Entry outside kill zone: {entry_ny} for trade {t.timestamp_entry}"
    )
print(f"DST verification PASS — all {len(kz_trades)} entries in [09:30, 11:00) ET")
```

**Load and resample pattern — copy verbatim from `bidir_15m_test.py`:**

```python
def load_and_resample() -> pd.DataFrame:
    bars = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    if bars["timestamp"].dt.tz is None:
        bars["timestamp"] = bars["timestamp"].dt.tz_localize("UTC")
    else:
        bars["timestamp"] = bars["timestamp"].dt.tz_convert("UTC")
    bars["timestamp"] = bars["timestamp"].dt.tz_convert("America/New_York")
    bars = bars.set_index("timestamp").sort_index()
    bars = bars.drop(columns=["notional"], errors="ignore")
    return (
        bars.resample("15min")
        .agg(open=("open","first"), high=("high","max"),
             low=("low","min"), close=("close","last"), volume=("volume","sum"))
        .dropna(subset=["open","high","low","close"])
    )
```

**UTC temp file pattern (critical — BacktestEngine expects UTC timestamps in CSV):**

```python
def run_backtest(bars_15m: pd.DataFrame, config: StrategyConfig) -> list:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df_out = bars_15m.reset_index()
            df_out["timestamp"] = df_out["timestamp"].dt.tz_convert("UTC")  # CRITICAL
            df_out.to_csv(f, index=False)
            tmp_path = f.name
        return BacktestEngine(tmp_path, config=config).run()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
```

**`metrics()` function — reuse from `bidir_15m_test.py`** (copy verbatim — PF, WR, Sharpe, exit_counts).

### Integration Test Design

```python
# tests/integration/test_kill_zone_filter_integration.py
"""Integration test: BacktestEngine blocks outside-kill-zone entries when
enable_kill_zone_filter=True."""

import os, tempfile
import pandas as pd
import pytest
from src.research.backtest_engine import BacktestEngine
from src.research.strategy_core import StrategyConfig

import zoneinfo
NY_TZ = zoneinfo.ZoneInfo("America/New_York")

def _make_minimal_csv(tmp_path: str) -> str:
    """Write a minimal 1-min CSV spanning 2025-01-15 with bars inside and outside
    the 09:30-11:00 ET kill zone."""
    # Use the real 2025 CSV if available, otherwise skip
    # Alternatively: create synthetic bars for a single day
    ...
```

**Simpler approach**: Run BacktestEngine on the actual 2025 CSV with `enable_kill_zone_filter=True`, then check all trades. If no trades present due to data constraints, use `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv` directly (same as the research script).

```python
def test_kill_zone_filter_blocks_non_kz_entries():
    """All trades produced with enable_kill_zone_filter=True must be in kill zone."""
    import math, os, tempfile
    import pandas as pd
    from src.research.backtest_engine import BacktestEngine
    from src.research.strategy_core import StrategyConfig
    import zoneinfo

    CSV_PATH = "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
    if not os.path.exists(CSV_PATH):
        pytest.skip("Training CSV not available")

    NY_TZ = zoneinfo.ZoneInfo("America/New_York")
    bars = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    # Use only 2 weeks to keep test fast
    bars = bars.head(2 * 390)  # ~2 weeks of 1-min bars
    if bars["timestamp"].dt.tz is None:
        bars["timestamp"] = bars["timestamp"].dt.tz_localize("UTC")
    bars = bars.set_index("timestamp").sort_index()
    bars = bars.drop(columns=["notional"], errors="ignore")
    bars_15m = (
        bars.resample("15min")
        .agg(open=("open","first"), high=("high","max"),
             low=("low","min"), close=("close","last"), volume=("volume","sum"))
        .dropna(subset=["open","high","low","close"])
    )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df_out = bars_15m.reset_index()
            df_out["timestamp"] = df_out["timestamp"].dt.tz_convert("UTC")
            df_out.to_csv(f, index=False)
            tmp_path = f.name

        config = StrategyConfig(bearish_only=True, enable_kill_zone_filter=True)
        engine = BacktestEngine(tmp_path, config=config)
        trades = engine.run()

        from datetime import time as dtime
        for t in trades:
            entry_ny = t.timestamp_entry.astimezone(NY_TZ).time()
            assert dtime(9, 30) <= entry_ny < dtime(11, 0), (
                f"Trade outside kill zone at {entry_ny}"
            )
        assert all(t.kill_zone_active for t in trades)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
```

### Verdict Report Format

Follow `s_bidir_15m_verdict_20260523.md` format. Include:
- Pre-registration SHA
- Results table: KZ-filtered vs Full-window vs S13 baseline
- DST verification: PASS/FAIL + count
- Consistency criterion bullets
- Verdict: H₁ SUPPORTED or H₀ SUPPORTED

### Pre-Registration Discipline

**CRITICAL: commit pre-reg doc BEFORE running any backtest script.**

Pre-reg commit command:
```bash
git add -f _bmad-output/preregistration_s_kz_15m.md && git commit -m "pre-register S-KZ-15m: AM kill zone filter at 15m"
```

SHA goes into the pre-reg doc and into `PRE_REG_SHA` constant in `kz_15m_test.py`.

### Long-Running Script Execution

Use nohup pattern (per memory — CPU is slow, script takes minutes):
```bash
nohup bash -c 'PYTHONPATH=. .venv/bin/python src/research/kz_15m_test.py > /tmp/kz_15m.log 2>&1' &
until grep -q -E "(VERDICT:|Error|Traceback)" /tmp/kz_15m.log 2>/dev/null; do sleep 10; done && cat /tmp/kz_15m.log
```

### What NOT to Change

- `tier2_streaming_working.py` — live system, governed by S26 pre-registration
- `kill_zone_start_et` / `kill_zone_end_et` defaults — stay 09:30/11:00 ET
- `test_strategy_core_killzone.py` — DST tests already complete and passing
- `bearish_only` default — stays `True` (load-bearing per Story 2.1)

### References

- S13 baseline: `_bmad-output/s13_verdict_20260523.md` (15m bearish-only)
- Story 2.1 result: `_bmad-output/s_bidir_15m_verdict_20260523.md` (bidirectional fails)
- Pre-registration pattern: `_bmad-output/preregistration_s_bidir_15m.md`
- Resample + temp-file pattern: `src/research/bidir_15m_test.py`
- Kill zone tests: `tests/unit/test_strategy_core_killzone.py`
- Kill zone function: `src/research/strategy_core.py:584`
- BacktestEngine entry loop: `src/research/backtest_engine.py:817`

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6 (2026-05-23)

### Debug Log References

- Pre-registration commit: `df66bd9` (SHA fill-in: `6d5e086`)

### Completion Notes List

1. Pre-registration sealed at `df66bd9` before any backtest ran.
2. **VERDICT: H₀ SUPPORTED.** AM kill zone fails both criteria at 15m:
   - Count: 5 trades (need ≥ 15) — kill zone window captures almost none of the 61 baseline trades
   - PF: 0.826 ≤ 1.3
   - DST verification: PASS (all 5 entries correctly in [09:30, 11:00) ET)
3. **Root cause:** At 15m resolution the 09:30–11:00 ET window spans only 6 bars. The strategy's H1 sweep lookback fires at H1 bar boundaries (typically 09:00 or 10:00), and the corresponding FVG on the 15m chart often appears outside the kill zone window. The filter is far too restrictive at 15m.
4. **Implication for Epic 2:** The AM kill zone in its current form (09:30–11:00 ET) is not viable as a filtering approach at 15m. The full-window bearish 15m strategy (61 trades, PF=1.179) should proceed to Story 2.3 (M15 confirmation layer) for the next power-recovery attempt.
5. `enable_kill_zone_filter: bool = False` is retained in StrategyConfig with default `False` — no behavior change for any existing caller. The infrastructure is in place for future experiments or live-system use (S26 governs that separately).
6. Full window baseline verification reproduced exactly 61 trades / PF=1.179, confirming BacktestEngine parity.

### File List

- `_bmad-output/preregistration_s_kz_15m.md` — NEW (committed at `df66bd9`)
- `src/research/strategy_core.py` — MODIFY (added `enable_kill_zone_filter: bool = False` to StrategyConfig)
- `src/research/backtest_engine.py` — MODIFY (kill zone blocking guard after `kz = kill_zone_filter(bar_ts, config)`)
- `src/research/kz_15m_test.py` — NEW
- `_bmad-output/s_kz_15m_verdict_20260523.md` — NEW (produced by running script)
- `tests/integration/test_kill_zone_filter_integration.py` — NEW
