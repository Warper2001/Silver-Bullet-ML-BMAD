# Story 2.3: M15 Confirmation Layer and Resample (15m Reframe)

Status: review

## Story

As Alex (researcher),
I want M15 bar resampling and a confirmation check that verifies the prior M15 bar closes in the H1 sweep direction before entry,
So that false entries on structurally misaligned 15m bars are filtered out and I can measure whether this restriction improves PF over the bearish-only 15m baseline.

> **15m Reframe Note:** Stories 2.1 (bidir, H₀) and 2.2 (kill zone, H₀) both failed. This story follows the same pre-registration-first research discipline at 15m. The M15 confirmation at 15m checks whether the bar immediately preceding the FVG candidate closes in the H1 sweep direction — a structural alignment filter.

## Acceptance Criteria

1. Pre-registration doc `_bmad-output/preregistration_s_m15conf_15m.md` written **and committed** before running any backtest. Hypothesis: "M15-confirmed bearish 15m trades (prior bar closes bearish) show PF > 1.3 with N ≥ 15."
2. `resample_to_m15(bars: pd.DataFrame) -> pd.DataFrame` added to `strategy_core.py` — same aggregation as `resample_to_h1` but `"15min"` frequency. Same canonical AR9 schema output.
3. `M15Confirmation` frozen dataclass added to `strategy_core.py` with fields `confirmed: bool` and `direction: Direction | None = None`.
4. `check_m15_confirmation(h1_sweep: SweepSignal, m15_bars: pd.DataFrame) -> M15Confirmation` added to `strategy_core.py` — returns `M15Confirmation(confirmed=True, direction=Direction.BEARISH)` when bearish sweep + last M15 bar closes bearish (`close < open`); returns `M15Confirmation(confirmed=False)` otherwise. Empty `m15_bars` returns `M15Confirmation(confirmed=False)`.
5. `m15_confirmation: bool = False` added to `StrategyConfig` (default `False` — no behavior change for existing callers).
6. `BacktestEngine.run()` wired: when `config.m15_confirmation=True`, computes M15 confirmation from completed M15 bars before the current bar and blocks entry if `confirmed=False`. `TradeRecord.m15_confirmed` populated with the actual confirmation result (not always `False`).
7. `tests/unit/test_strategy_core_m15.py` added: covers `resample_to_m15` schema/aggregation, `check_m15_confirmation` (bearish confirmed, bearish rejected, bullish confirmed, empty bars, doji edge case).
8. `src/research/m15_conf_test.py` implemented: loads 2025 training CSV, resamples to 15m, runs two BacktestEngine passes (m15_confirmation=True vs False), verifies all M15-confirmed trades have `m15_confirmed=True`, applies H₁/H₀ decision logic (PF > 1.3 AND N ≥ 15).
9. Verdict report `_bmad-output/s_m15conf_15m_verdict_<date>.md` produced with N, PF, WR, Sharpe, exit breakdown, confirmation verification result, and H₁/H₀ verdict.
10. No modifications to `tier2_streaming_working.py`. StatePersistence (FR37) is Epic 4 / live system scope — out of scope here.

## Tasks / Subtasks

- [x] Task 1 — Pre-register and commit (AC #1)
  - [x] Write `_bmad-output/preregistration_s_m15conf_15m.md` with hypothesis, data, config snapshot, stopping rule
  - [x] `git add -f _bmad-output/preregistration_s_m15conf_15m.md && git commit` — record SHA: cfe7cb3
  - [x] Write SHA into pre-reg doc (follow-up commit e17d4b4)

- [x] Task 2 — Add M15 infrastructure to `strategy_core.py` (ACs #2, #3, #4, #5)
  - [x] Add `M15Confirmation` frozen dataclass (after `ExitDecision`, before internal helpers section)
  - [x] Add `resample_to_m15()` function (after `resample_to_h1`)
  - [x] Add `check_m15_confirmation()` function (after `kill_zone_filter`)
  - [x] Add `m15_confirmation: bool = False` to `StrategyConfig` (after `enable_kill_zone_filter`)

- [x] Task 3 — Wire M15 confirmation in `BacktestEngine` (AC #6)
  - [x] Add imports: `check_m15_confirmation`, `resample_to_m15`
  - [x] Pre-compute `full_m15 = resample_to_m15(bars)` after `full_h1 = resample_to_h1(bars)`
  - [x] Add `active_m15: bool = False` to active-trade state variables
  - [x] In entry detection: compute `m15_ok`, pass `m15_conf=m15_ok` to `make_entry_decision`
  - [x] `active_m15 = m15_ok` set after entry arm
  - [x] Update `_append_trade` signature to accept `m15_ok: bool`; use `m15_confirmed=m15_ok`
  - [x] Update both `_append_trade` call sites to pass `active_m15`
  - [x] Run integration tests to verify no regressions — 12/12 passed

- [x] Task 4 — Unit tests for new `strategy_core` functions (AC #7)
  - [x] Write `tests/unit/test_strategy_core_m15.py`
  - [x] Verify `resample_to_m15` schema and aggregation with synthetic 1-min bars
  - [x] Verify `check_m15_confirmation` for all cases: bearish confirmed, bearish rejected, bullish confirmed, empty bars, doji
  - [x] Run: `.venv/bin/python -m pytest tests/unit/test_strategy_core_m15.py -v` — 12/12 passed

- [x] Task 5 — Implement `src/research/m15_conf_test.py` (AC #8)
  - [x] Copy `load_and_resample()` and `metrics()` pattern from `kz_15m_test.py`
  - [x] Run A: `StrategyConfig(bearish_only=True, m15_confirmation=True)` — 61 trades, PF=1.179
  - [x] Run B: `StrategyConfig(bearish_only=True, m15_confirmation=False)` — 61 trades (baseline reproduced)
  - [x] Verify all trades in Run A have `m15_confirmed=True` — PASS (61/61)
  - [x] Apply H₁/H₀ decision logic — H₀ SUPPORTED (PF 1.179 ≤ 1.3)

- [x] Task 6 — Produce verdict report (AC #9)
  - [x] Write `_bmad-output/s_m15conf_15m_verdict_20260523.md` from script output

- [x] Task 7 — Full test suite verification (AC #10)
  - [x] `.venv/bin/python -m pytest tests/unit/test_strategy_core_m15.py tests/integration/test_baseline_backtesting.py tests/integration/test_kill_zone_filter_integration.py tests/unit/test_strategy_core_killzone.py -q` — 52 passed, 0 failed

## Dev Notes

### 15m Research Context

This is the third Epic 2 power-recovery experiment at 15m:
- **Story 2.1** (bidir): H₀ — bullish FVG drag, bearish_only=True confirmed load-bearing
- **Story 2.2** (kill zone): H₀ — 09:30–11:00 ET captures only 5/61 trades at 15m
- **Story 2.3** (M15 confirmation): hypothesis — requiring the previous 15m bar to close in the sweep direction filters false entries

Baseline: `bearish_only=True`, 61 trades, PF=1.179, WR=0.475, Sharpe=1.373 (2025 training window).

### Bearish-Only Baseline

| Metric | Value |
|---|---|
| Trades | 61 |
| PF | 1.179 |
| WR | 0.475 |
| Daily Sharpe | 1.373 |
| TIME_STOP % | ~11% |

### What M15 Confirmation Means at 15m

In the live system (H1→M15→M1 hierarchy), M15 confirmation checks whether the last completed 15-minute bar closes in the H1 sweep direction before the 1-min FVG entry.

At 15m research: the bars ARE 15m bars. So M15 confirmation checks whether bar `i-1` (the last completed 15m bar before the current FVG candidate at bar `i`) closes in the H1 sweep direction. For a bearish setup: bar i-1 close < open → confirmed.

The filter is asking: "Is the most recent completed 15m candle also bearish?" This is a structural alignment check — it requires that momentum is still pointing in the sweep direction.

### Infrastructure Already in Place

**`strategy_core.py` — DO NOT reinvent:**
- `resample_to_h1()` at line ~194 — exact same aggregation pattern, just change `"1h"` → `"15min"` for `resample_to_m15`
- `SweepSignal` dataclass at line ~113 — `h1_sweep.direction` is a `Direction` enum member
- `_validate_bars()` at line ~146 — helper for validating DataFrame schema; DO NOT add to check_m15_confirmation (it expects empty m15_bars to return M15Confirmation(confirmed=False) without raising)
- `_NY_TZ = zoneinfo.ZoneInfo("America/New_York")` at line ~500 — shared module-level constant
- `kill_zone_filter()` at line ~585 — pattern for what a new module-level filter function looks like
- `StrategyConfig` at line ~70 — `enable_kill_zone_filter: bool = False` is the last field (line 99); add `m15_confirmation` AFTER it

**`backtest_engine.py` — DO NOT reinvent:**
- `full_h1 = resample_to_h1(bars)` at line ~629 — add `full_m15 = resample_to_m15(bars)` immediately after
- `active_kz: bool = False` at line ~638 — add `active_m15: bool = False` immediately after
- Entry detection at line ~817 — `kz` guard is at line 817-819; add M15 guard AFTER the kz block, BEFORE `make_entry_decision`
- `make_entry_decision(sweep, fvg, config, vol_ok=vol_ok_cached)` at line ~822 — change to `make_entry_decision(sweep, fvg, config, vol_ok=vol_ok_cached, m15_conf=m15_ok)`; `make_entry_decision` already handles `**filter_results` so any falsy value blocks entry
- `active_kz = kz` at line ~830 — add `active_m15 = m15_ok` immediately after
- `_append_trade()` at line ~929 — currently `m15_confirmed=False` hardcoded at line 962; add `m15_ok: bool` parameter and use it
- Two call sites for `_append_trade` (lines ~711 and ~739) — both need `active_m15` passed

### Exact Code Changes — strategy_core.py

#### 1. M15Confirmation dataclass (add after ExitDecision, before internal helpers)

```python
@dataclass(frozen=True)
class M15Confirmation:
    """M15 bar confirmation result. Returned by check_m15_confirmation (Story 2.3)."""

    confirmed: bool
    direction: Direction | None = None
```

Place this AFTER the `ExitDecision` dataclass (line ~139) and BEFORE the `# Internal helpers` comment block (line ~141).

#### 2. StrategyConfig — new field (add after enable_kill_zone_filter)

```python
    enable_kill_zone_filter: bool = False  # if True, blocks entries outside kill zone
    m15_confirmation: bool = False  # if True, blocks entries where prior M15 bar misaligns with H1 sweep
```

#### 3. resample_to_m15 (add immediately after resample_to_h1)

```python
def resample_to_m15(bars: pd.DataFrame) -> pd.DataFrame:
    """Resample bars to 15-minute OHLCV candles.

    Same aggregation as resample_to_h1 but at 15-minute frequency.
    No timezone conversion performed (AR19); bars must arrive tz-aware.

    Parameters
    ----------
    bars:
        Canonical bars with tz-aware DatetimeIndex named ``timestamp`` (AR9).

    Returns
    -------
    pd.DataFrame
        15-min OHLCV with tz-aware DatetimeIndex named ``timestamp``.
        Aggregation: open=first, high=max, low=min, close=last, volume=sum.
        Periods with no data are dropped.

    Raises
    ------
    ValueError
        On empty input, NaN in OHLCV/volume, or missing columns.
    """
    _validate_bars(bars, min_rows=1)
    df = bars.copy()
    if not (isinstance(df.index, pd.DatetimeIndex) and df.index.name == "timestamp"):
        df = df.set_index("timestamp")
    m15 = (
        df[["open", "high", "low", "close", "volume"]]
        .resample("15min")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open", "high", "low", "close"])
    )
    m15.index.name = "timestamp"
    return m15
```

#### 4. check_m15_confirmation (add after kill_zone_filter, before metric functions)

```python
def check_m15_confirmation(
    h1_sweep: SweepSignal,
    m15_bars: pd.DataFrame,
) -> M15Confirmation:
    """Check if the last completed M15 bar closes in the H1 sweep direction.

    For a bearish sweep: confirmed when last M15 bar close < open (closes bearish).
    For a bullish sweep: confirmed when last M15 bar close > open (closes bullish).
    A doji (close == open) is NOT confirmed for either direction.

    Returns M15Confirmation(confirmed=False) when m15_bars is empty — the caller
    must decide whether to block or allow entry on insufficient history.

    Parameters
    ----------
    h1_sweep:
        The active H1 liquidity sweep (from detect_liquidity_sweep).
    m15_bars:
        Completed M15 bars up to (but not including) the current bar.
        Must have the canonical AR9 schema (timestamp index, OHLCV columns).

    Returns
    -------
    M15Confirmation
        confirmed=True + direction when last bar aligns; confirmed=False otherwise.
    """
    if len(m15_bars) == 0:
        return M15Confirmation(confirmed=False)

    last = m15_bars.iloc[-1]
    close = float(last["close"])
    open_ = float(last["open"])

    if h1_sweep.direction == Direction.BEARISH:
        confirmed = close < open_
    else:
        confirmed = close > open_

    return M15Confirmation(
        confirmed=confirmed,
        direction=h1_sweep.direction if confirmed else None,
    )
```

### Exact Code Changes — backtest_engine.py

#### 1. Imports — add to the from strategy_core import block

```python
from src.research.strategy_core import (
    POINT_VALUE_USD,
    Direction,
    EntryDecision,
    ExitDecision,
    M15Confirmation,          # NEW
    StrategyConfig,
    SweepSignal,
    calc_atr,
    check_exit,
    check_m15_confirmation,   # NEW
    detect_fvg,
    detect_liquidity_sweep,
    kill_zone_filter,
    make_entry_decision,
    resample_to_h1,
    resample_to_m15,          # NEW
    volatility_regime_filter,
)
```

#### 2. Pre-compute full_m15 (in run(), after full_h1)

```python
full_h1 = resample_to_h1(bars)
full_m15 = resample_to_m15(bars)          # NEW
```

#### 3. Active-trade state variables

```python
active_kz: bool = False
active_m15: bool = False                   # NEW
```

#### 4. Entry detection — M15 guard (add after kz guard, before make_entry_decision)

```python
            kz = kill_zone_filter(bar_ts, config)
            if config.enable_kill_zone_filter and not kz:
                continue  # outside kill zone — skip this entry candidate

            # M15 confirmation gate
            m15_ok = True
            if config.m15_confirmation:
                m15_idx = int(full_m15.index.searchsorted(bar_ts))
                m15_slice = full_m15.iloc[:m15_idx]
                if len(m15_slice) >= 1:
                    m15_ok = check_m15_confirmation(sweep, m15_slice).confirmed
                else:
                    m15_ok = False

            # Entry decision
            entry = make_entry_decision(sweep, fvg, config, vol_ok=vol_ok_cached, m15_conf=m15_ok)
```

#### 5. Active-trade state capture

```python
            active_kz = kz
            active_m15 = m15_ok                # NEW
```

#### 6. _append_trade signature update

Add `m15_ok: bool` parameter after `kz: bool`:

```python
    def _append_trade(
        self,
        trades: list[TradeRecord],
        active: EntryDecision,
        ts_entry: pd.Timestamp | None,
        ts_exit: pd.Timestamp,
        exit_dec: ExitDecision,
        gap: float,
        sweep_ago: int,
        kz: bool,
        m15_ok: bool,        # NEW
        vol_pct: float,
    ) -> None:
```

Change `m15_confirmed=False` to `m15_confirmed=m15_ok`.

#### 7. _append_trade call sites — add active_m15 argument

Both existing calls (lines ~711 and ~739) need `active_m15` inserted between `active_kz` and `active_vol_pct`:

```python
                            self._append_trade(
                                trades,
                                active,
                                active_ts,
                                bar_ts,
                                exit_dec,
                                active_gap,
                                active_sweep_ago,
                                active_kz,
                                active_m15,      # NEW
                                active_vol_pct,
                            )
```

### Research Script Pattern (`src/research/m15_conf_test.py`)

Copy `load_and_resample()` and `metrics()` verbatim from `kz_15m_test.py`. Key constants:

```python
CSV_PATH = "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
PRE_REG_SHA = "<filled after pre-reg commit>"
BASELINE = {"trades": 61, "pf": 1.179, "wr": 0.475, "sharpe": 1.373}
PF_THRESHOLD = 1.3
MIN_TRADES = 15
```

**Two BacktestEngine runs:**
```python
# Run A — M15-confirmed (m15_confirmation=True)
m15_trades = run_backtest(bars_15m, StrategyConfig(bearish_only=True, m15_confirmation=True))

# Run B — Full window verification (m15_confirmation=False), should ≈ 61 trades
full_trades = run_backtest(bars_15m, StrategyConfig(bearish_only=True, m15_confirmation=False))
```

**M15 confirmation verification (required by AC #8):**
```python
for t in m15_trades:
    assert t.m15_confirmed, f"Trade not M15-confirmed: {t.timestamp_entry}"
print(f"M15 confirmation verification PASS — all {len(m15_trades)} trades have m15_confirmed=True")
```

**H₁/H₀ decision:**
```python
pf_pass = m15_m["pf"] > PF_THRESHOLD
n_pass = m15_m["trades"] >= MIN_TRADES
if pf_pass and n_pass:
    verdict = f"H₁ SUPPORTED — M15-confirmed 15m shows PF > {PF_THRESHOLD} with N ≥ {MIN_TRADES}"
else:
    verdict = "H₀ SUPPORTED — fails: ..."
```

**Run command (nohup pattern — backtest takes minutes):**
```bash
nohup bash -c 'PYTHONPATH=. .venv/bin/python src/research/m15_conf_test.py > /tmp/m15_conf.log 2>&1' &
until grep -q -E "(VERDICT:|Error|Traceback)" /tmp/m15_conf.log 2>/dev/null; do sleep 15; done && cat /tmp/m15_conf.log
```

### Unit Test Design (`tests/unit/test_strategy_core_m15.py`)

Follow the exact pattern of `tests/unit/test_strategy_core_detection.py` (same imports, `make_1min_bars` helper, `NY_TZ` constant).

```python
"""Unit tests for strategy_core M15 functions (Story 2.3).

Covers: resample_to_m15, check_m15_confirmation (all direction + edge cases).
"""
from src.research.strategy_core import (
    Direction, M15Confirmation, SweepSignal,
    check_m15_confirmation, resample_to_m15,
)
```

Key test cases:

| Test | Setup | Expected |
|---|---|---|
| `test_resample_to_m15_schema` | 60 1-min bars | 4 or fewer 15-min bars, DatetimeIndex named timestamp |
| `test_resample_to_m15_aggregation` | 15 bars: first open=100, max high=105, min low=98, last close=102, sum volume | Single 15m bar with those values |
| `test_check_m15_bearish_confirmed` | Bearish sweep + m15 bar close=99 < open=100 | M15Confirmation(confirmed=True, direction=Direction.BEARISH) |
| `test_check_m15_bearish_rejected` | Bearish sweep + m15 bar close=101 > open=100 | M15Confirmation(confirmed=False, direction=None) |
| `test_check_m15_bullish_confirmed` | Bullish sweep + m15 bar close=101 > open=100 | M15Confirmation(confirmed=True, direction=Direction.BULLISH) |
| `test_check_m15_doji_bearish` | Bearish sweep + m15 bar close=100 == open=100 | M15Confirmation(confirmed=False, direction=None) |
| `test_check_m15_empty_bars` | Empty DataFrame | M15Confirmation(confirmed=False, direction=None) |

**Creating a SweepSignal for tests:**
```python
import pandas as pd
import zoneinfo
NY_TZ = zoneinfo.ZoneInfo("America/New_York")

def make_sweep(direction: Direction) -> SweepSignal:
    return SweepSignal(direction=direction, bars_ago=1, sweep_price=100.0)

def make_m15_bar(open_: float, close: float, ts: str = "2025-06-02 09:00:00") -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        [pd.Timestamp(ts, tz=NY_TZ)], name="timestamp"
    )
    return pd.DataFrame(
        {"open": open_, "high": max(open_, close) + 0.5,
         "low": min(open_, close) - 0.5, "close": close, "volume": 1000},
        index=idx,
    )
```

### Out of Scope

- `tier2_streaming_working.py` — live system; NOT to be touched
- `StatePersistence` / FR37 — Epic 4 (live trading system) territory
- Changing `kill_zone_start_et` / `kill_zone_end_et` defaults
- Changing `bearish_only` default (confirmed load-bearing by Story 2.1)
- Holdout access — training window 2025 only
- Changing `enable_kill_zone_filter` default (stays `False`)

### Pre-Registration Discipline

**CRITICAL: commit pre-reg doc with `git add -f` BEFORE any backtest simulation runs.**

Pattern from Story 2.2:
```bash
git add -f _bmad-output/preregistration_s_m15conf_15m.md && git commit -m "pre-register S-M15CONF-15m: M15 confirmation filter at 15m"
```

SHA goes into the pre-reg doc AND into `PRE_REG_SHA` constant in `m15_conf_test.py`.

### References

- S13 baseline: `_bmad-output/s13_verdict_20260523.md` (15m bearish-only)
- Story 2.1 pattern: `src/research/bidir_15m_test.py`, `_bmad-output/s_bidir_15m_verdict_20260523.md`
- Story 2.2 pattern: `src/research/kz_15m_test.py`, `_bmad-output/s_kz_15m_verdict_20260523.md`
- Pre-registration pattern: `_bmad-output/preregistration_s_kz_15m.md`
- Load/resample pattern: `src/research/kz_15m_test.py::load_and_resample()`
- `resample_to_h1` (to copy): `src/research/strategy_core.py:194`
- `kill_zone_filter` (pattern for new function placement): `src/research/strategy_core.py:585`
- `enable_kill_zone_filter` guard pattern: `src/research/backtest_engine.py:817`
- `_append_trade` method: `src/research/backtest_engine.py:929`
- Detection unit test pattern: `tests/unit/test_strategy_core_detection.py`
- Kill zone unit test pattern: `tests/unit/test_strategy_core_killzone.py`

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6 (2026-05-23)

### Debug Log References

(none)

### Completion Notes List

- H₀ SUPPORTED: M15 confirmation filter is not selective at 15m — all 61 baseline trades already have prior bearish 15m bar. Filter returns PF=1.179 (identical to baseline), N=61. Root cause: in a bearish H1 sweep regime, FVGs structurally form in sequences of bearish bars so the prior bar is almost always bearish.
- Full test suite: 52 passed (test_strategy_core_m15.py × 12, test_baseline_backtesting.py × 14, test_kill_zone_filter_integration.py × 3+, test_strategy_core_killzone.py × 18+)
- Pre-registration sealed at SHA cfe7cb3 before any simulation run; SHA committed in follow-up e17d4b4
- `tier2_streaming_working.py` not modified (confirmed)

### File List

- `_bmad-output/preregistration_s_m15conf_15m.md` — NEW
- `src/research/strategy_core.py` — MODIFY (add M15Confirmation, resample_to_m15, check_m15_confirmation, m15_confirmation field)
- `src/research/backtest_engine.py` — MODIFY (import, full_m15, active_m15, M15 guard, _append_trade update)
- `src/research/m15_conf_test.py` — NEW
- `_bmad-output/s_m15conf_15m_verdict_20260523.md` — NEW
- `tests/unit/test_strategy_core_m15.py` — NEW
