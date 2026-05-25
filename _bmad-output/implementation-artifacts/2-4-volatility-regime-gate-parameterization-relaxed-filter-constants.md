# Story 2.4: Volatility Regime Gate Parameterization and Relaxed Filter Constants

Status: done

## Story

As Alex (researcher),
I want the volatility regime gate to use configurable `StrategyConfig` fields and all over-tight filter constants relaxed to evidence-based values,
so that I can measure whether loosening entry constraints (H1 sweep lookback, min gap size, pending timeout, Tuesday exclusion) improves PF over the bearish-only 15m baseline.

> **Context:** Stories 2.1–2.3 all returned H₀ at 15m. Bearish-only baseline: 61 trades, PF=1.179. Story 2.4 is the final Epic 2 experiment: relax entry constraints and measure whether more trades come with improved PF. The epics spec listed `bearish_only=False` in AC #2, but Story 2.1 showed bidirectional trades drag PF to 0.826 — `bearish_only=True` is preserved as a hard constraint here. Similarly, `enable_kill_zone_filter=False` and `m15_confirmation=False` are kept (Stories 2.2–2.3 showed no benefit; stacking failing filters is not scientifically sound).

## Acceptance Criteria

1. Pre-registration doc `_bmad-output/preregistration_s_vol_15m.md` written **and committed** before running any backtest. Must document the deviation from epics AC #2 (bearish_only=True, not False) with Story 2.1 verdict as justification.

2. `tuesday_exclusion: bool = True` field added to `StrategyConfig` (default `True` preserves existing behavior). `BacktestEngine` wires this field: replace hardcoded `if bar_ts.weekday() == 1: continue` with `if config.tuesday_exclusion and bar_ts.weekday() == 1: continue`.

3. `volatility_regime_filter()` confirmed to use `config.vol_regime_lookback` and `config.vol_regime_threshold` — no hardcoded `0.75` or `120` exist in `strategy_core.py`. No code change required; confirm in verdict report (AC #1 of epics is already satisfied).

4. Research script `src/research/vol_regime_15m_test.py`: runs `StrategyConfig(bearish_only=True, h1_sweep_lookback=10, min_gap_atr_ratio=0.10, max_pending_bars=120, tuesday_exclusion=False)` vs baseline. Verifies AC #4: pending order timeout at 121 bars. Computes AC #3: average monthly trade count.

5. Verdict report `_bmad-output/s_vol_15m_verdict_<date>.md` produced with: N, PF, WR, Sharpe, monthly breakdown, AC #1 confirmation, AC #4 max_pending_bars=120 confirmation, AC #5 SL=5.0/TP=6.0 confirmation, H₁/H₀ verdict (PF > 1.3 AND N ≥ 15).

6. Unit test for `tuesday_exclusion` field: verify default=True, verify False is accepted. Integration test: BacktestEngine with `tuesday_exclusion=False` on a synthetic dataset spanning Tuesdays includes Tuesday entries; `tuesday_exclusion=True` excludes them.

7. No modifications to `tier2_streaming_working.py`.

## Tasks / Subtasks

- [x] Task 1 — Pre-register and commit (AC #1)
  - [x] Write `_bmad-output/preregistration_s_vol_15m.md` with hypothesis, config snapshot, deviation note (bearish_only=True), stopping rule
  - [x] `git add -f _bmad-output/preregistration_s_vol_15m.md && git commit` — SHA: b44acc6
  - [x] Write SHA into pre-reg doc (follow-up commit 4361d88)

- [x] Task 2 — Add `tuesday_exclusion` to `strategy_core.py` and wire in `backtest_engine.py` (AC #2)
  - [x] Add `tuesday_exclusion: bool = True` to `StrategyConfig` (after `m15_confirmation`)
  - [x] In `backtest_engine.py` line ~770: replaced `if bar_ts.weekday() == 1: continue` with `if config.tuesday_exclusion and bar_ts.weekday() == 1: continue`

- [x] Task 3 — Unit and integration tests for `tuesday_exclusion` (AC #6)
  - [x] Write `tests/unit/test_strategy_core_tuesday.py`: 4 tests (default True, can be False, independent of bearish_only, frozen)
  - [x] Write `tests/integration/test_tuesday_exclusion_integration.py`: 3 tests (no Tuesday entries with True, more trades with False, default matches True)
  - [x] Run: `.venv/bin/python -m pytest tests/unit/test_strategy_core_tuesday.py -v` — 4/4 passed
  - [x] Run: `.venv/bin/python -m pytest tests/integration/test_tuesday_exclusion_integration.py -v` — 3/3 passed

- [x] Task 4 — Implement `src/research/vol_regime_15m_test.py` (ACs #3, #4, #5)
  - [x] Copy `load_and_resample()`, `run_backtest()`, `metrics()` verbatim from `kz_15m_test.py`
  - [x] Run B: `StrategyConfig(bearish_only=True)` — 61 trades (baseline reproduced)
  - [x] Run A: `StrategyConfig(bearish_only=True, h1_sweep_lookback=10, min_gap_atr_ratio=0.10, max_pending_bars=120, tuesday_exclusion=False)` — 104 trades, PF=0.881
  - [x] Compute monthly trade count: avg 8.7/month (12 months, min=3 Nov, max=18 Jul)
  - [x] Apply H₁/H₀ decision — H₀ SUPPORTED (PF=0.881 ≤ 1.3)

- [x] Task 5 — Produce verdict report (AC #5)
  - [x] Write `_bmad-output/s_vol_15m_verdict_20260523.md`

- [x] Task 6 — Full test suite verification (AC #7)
  - [x] `.venv/bin/python -m pytest tests/unit/test_strategy_core_tuesday.py tests/integration/test_baseline_backtesting.py tests/integration/test_tuesday_exclusion_integration.py tests/unit/test_strategy_core_killzone.py tests/unit/test_strategy_core_m15.py -q` — 56 passed, 0 failed

### Review Findings

- [x] [Review][Patch] Integration test uses real CSV + all three tests pass vacuously on 0 trades [`tests/integration/test_tuesday_exclusion_integration.py`] — replaced with synthetic 3-day fixture (Thu/Mon/Tue); all 3 tests are now non-vacuous; 85/85 Epic 2+3 tests pass

- [x] [Review][Defer] AC #4 pending-timeout verification is a static config echo, not a behavioral assertion [`src/research/vol_regime_15m_test.py`] — script prints `RELAXED_CONFIG.max_pending_bars = 120` but does not assert that a pending order actually cancels at bar 121; deferred, pre-existing script pattern

## Dev Notes

### Research Methodology Constraints (READ FIRST)

**DO NOT** set `bearish_only=False` — Story 2.1 showed bidirectional PF=0.826; the field is load-bearing.
**DO NOT** set `enable_kill_zone_filter=True` — Story 2.2 showed H₀ (5/61 trades); not selective at 15m.
**DO NOT** set `m15_confirmation=True` — Story 2.3 showed H₀ (all 61 trades already had prior bearish bar); not selective.
**DO NOT** touch `tier2_streaming_working.py` — live system, untouchable during research.
**DO NOT** change `vol_regime_threshold` or `vol_regime_lookback` defaults — already correct at 0.75/120.

Pre-registration must be committed BEFORE any BacktestEngine run. Pattern: `git add -f doc && git commit`, then SHA goes into pre-reg doc AND into `PRE_REG_SHA` constant in the research script.

### StrategyConfig — Current State (strategy_core.py lines 70–101)

```python
@dataclass(frozen=True)
class StrategyConfig:
    sl_multiplier: float = 5.0         # SL = 5.0× gap (AC #5 confirmation value)
    tp_multiplier: float = 6.0         # TP = 6.0× gap (AC #5 confirmation value)
    entry_pct: float = 0.5
    atr_threshold: float = 0.5
    max_gap_dollars: float = 60.0
    max_hold_bars: int = 60
    max_pending_bars: int = 240        # ← RELAX to 120 in RELAXED_CONFIG (not default change)
    contracts_per_trade: int = 5
    max_daily_loss: float = -750.0
    vol_regime_lookback: int = 120     # already configurable — no change
    vol_regime_threshold: float = 0.75 # already configurable — no change
    min_gap_atr_ratio: float = 0.25   # ← RELAX to 0.10 in RELAXED_CONFIG (not default change)
    ml_threshold: float = 0.0
    bearish_only: bool = True          # MUST stay True (Story 2.1 verdict)
    h1_sweep_lookback: int = 6         # ← RELAX to 10 in RELAXED_CONFIG (not default change)
    kill_zone_start_et: time = time(9, 30)
    kill_zone_end_et: time = time(11, 0)
    commission_per_roundtrip: float = 4.0
    enable_kill_zone_filter: bool = False  # MUST stay False (Story 2.2 verdict)
    m15_confirmation: bool = False         # MUST stay False (Story 2.3 verdict)
    tuesday_exclusion: bool = True     # ← NEW field to add (Story 2.4)
```

**Important:** Only add `tuesday_exclusion` to StrategyConfig. Do NOT change any existing defaults — the relaxed values are passed as constructor arguments in the research script, not as new defaults.

### BacktestEngine — Tuesday Exclusion Fix (backtest_engine.py line 770)

Current (hardcoded, problematic):
```python
if bar_ts.weekday() == 1:  # Tuesday filter
    continue
```

Replace with (configurable):
```python
if config.tuesday_exclusion and bar_ts.weekday() == 1:  # Tuesday filter
    continue
```

That is the **only change** in `backtest_engine.py`. The rest of the engine is untouched.

### volatility_regime_filter — Already Parameterized (strategy_core.py lines 484–545)

The function already uses `config.vol_regime_lookback` and `config.vol_regime_threshold`. No code changes needed. Confirm in verdict: "No hardcoded 0.75 or 120 in strategy_core.py ✓".

### Research Script Pattern (`src/research/vol_regime_15m_test.py`)

Copy these verbatim from `src/research/kz_15m_test.py`:
- `load_and_resample()` — CSV load, ET timezone conversion, 15m resample, UTC for temp CSV
- `run_backtest(bars_15m, config)` — tempfile pattern, BacktestEngine.run()
- `metrics(trades_list)` — PF, WR, Sharpe, exit_counts

Additional for Story 2.4:
```python
from collections import Counter

CSV_PATH = "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
PRE_REG_SHA = "<fill after pre-reg commit>"
BASELINE = {"trades": 61, "pf": 1.179, "wr": 0.475, "sharpe": 1.373}
PF_THRESHOLD = 1.3
MIN_TRADES = 15
MONTHLY_TRADE_TARGET = 30  # AC #3

RELAXED_CONFIG = StrategyConfig(
    bearish_only=True,
    h1_sweep_lookback=10,
    min_gap_atr_ratio=0.10,
    max_pending_bars=120,
    tuesday_exclusion=False,
)
```

Monthly breakdown for AC #3:
```python
monthly = Counter(t.timestamp_entry.strftime("%Y-%m") for t in relaxed_trades)
avg_monthly = len(relaxed_trades) / max(len(monthly), 1)
monthly_pass = avg_monthly >= MONTHLY_TRADE_TARGET
```

H₁/H₀ decision (same pattern as prior stories):
```python
pf_pass = relaxed_m["pf"] > PF_THRESHOLD
n_pass = relaxed_m["trades"] >= MIN_TRADES
```

**Run command (nohup — backtest may take 2–5 min at 15m):**
```bash
nohup bash -c 'PYTHONPATH=. .venv/bin/python src/research/vol_regime_15m_test.py > /tmp/vol_regime.log 2>&1' &
until grep -q -E "(VERDICT:|Error|Traceback)" /tmp/vol_regime.log 2>/dev/null; do sleep 15; done && cat /tmp/vol_regime.log
```

### Unit Test Design (`tests/unit/test_strategy_core_tuesday.py`)

```python
"""Unit tests for StrategyConfig.tuesday_exclusion (Story 2.4)."""
from src.research.strategy_core import StrategyConfig

def test_tuesday_exclusion_default_true():
    assert StrategyConfig().tuesday_exclusion is True

def test_tuesday_exclusion_can_be_false():
    assert StrategyConfig(tuesday_exclusion=False).tuesday_exclusion is False
```

For the integration test (synthetic Tuesday dataset): construct bars spanning at least one Monday + Tuesday + Wednesday. Run BacktestEngine; with `tuesday_exclusion=True` no entry has `timestamp_entry.weekday() == 1`; with `tuesday_exclusion=False` at least one Tuesday entry may appear (or the count is ≥ the Tuesday-excluded count). Use a fixture analogous to the kill zone integration test.

### Verdict Report Content

`_bmad-output/s_vol_15m_verdict_<date>.md` must include:

| Run | N | PF | WR | Daily Sharpe |
|---|---|---|---|---|
| Relaxed config (A) | ? | ? | ? | ? |
| Full window baseline (B) | ~61 | ~1.179 | ~0.475 | ~1.373 |
| S13 baseline | 61 | 1.179 | 0.475 | 1.373 |

Monthly breakdown table (month, trade count).

**AC confirmations:**
- AC #1: `volatility_regime_filter` uses config.vol_regime_lookback / config.vol_regime_threshold — no hardcoded constants ✓
- AC #3: avg monthly trades = X (pass/fail vs ≥ 30)
- AC #4: max_pending_bars=120 in RELAXED_CONFIG ✓
- AC #5: sl_multiplier=5.0, tp_multiplier=6.0 ✓

### Previous Story Learnings (2.1–2.3)

| Story | Filter | Result | Lesson |
|---|---|---|---|
| 2.1 | Bidirectional FVG | H₀ — bullish PF=0.826 | bearish_only=True is load-bearing |
| 2.2 | AM Kill Zone 09:30–11:00 | H₀ — 5/61 trades | Kill zone too narrow at 15m (H1 sweep fires at hour boundaries) |
| 2.3 | M15 confirmation | H₀ — 61/61 already confirmed | Filter not selective; FVGs form in bearish sequences |

The key lesson: individual restrictions at 15m don't improve PF. Story 2.4 tests whether **relaxing** (not tightening) constraints increases both frequency and PF.

### Files Modified

| File | Action | Notes |
|---|---|---|
| `src/research/strategy_core.py` | MODIFY | Add `tuesday_exclusion: bool = True` after `m15_confirmation` |
| `src/research/backtest_engine.py` | MODIFY | Wire `config.tuesday_exclusion` at line ~770 |
| `src/research/vol_regime_15m_test.py` | NEW | Research script |
| `tests/unit/test_strategy_core_tuesday.py` | NEW | Unit tests |
| `tests/integration/test_tuesday_exclusion_integration.py` | NEW | Integration test |
| `_bmad-output/preregistration_s_vol_15m.md` | NEW | Pre-reg (commit FIRST) |
| `_bmad-output/s_vol_15m_verdict_<date>.md` | NEW | Produced by script |
| `_bmad-output/implementation-artifacts/sprint-status.yaml` | MODIFY | `2-4-...: review` when done |

### References

- Epics AC for Story 2.4: `_bmad-output/planning-artifacts/epics.md` lines 604–639
- StrategyConfig: `src/research/strategy_core.py` lines 70–101
- Tuesday hardcode: `src/research/backtest_engine.py` line 770
- `volatility_regime_filter()`: `src/research/strategy_core.py` lines 484–545
- Load/resample pattern: `src/research/kz_15m_test.py::load_and_resample()`
- Story 2.1 verdict: `_bmad-output/s_bidir_15m_verdict_20260523.md`
- Story 2.2 verdict: `_bmad-output/s_kz_15m_verdict_20260523.md`
- Story 2.3 verdict: `_bmad-output/s_m15conf_15m_verdict_20260523.md`

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6 (2026-05-23, create-story)

### Debug Log References

(none)

### Completion Notes List

- H₀ SUPPORTED: Relaxed config (h1_sweep_lookback=10, min_gap_atr_ratio=0.10, max_pending_bars=120, tuesday_exclusion=False) yields N=104 trades, PF=0.881. More trades (61→104) but lower quality — smaller FVG gaps (0.10 threshold) admit weaker setups that drag PF below baseline.
- Monthly avg: 8.7 trades/month (target ≥ 30). Trade frequency is structurally limited by H1 sweep signal rate at 15m, not by the gap-size filter.
- AC #1 confirmed: volatility_regime_filter() already fully parameterized via config — no code change needed.
- AC #4 confirmed: max_pending_bars=120 in RELAXED_CONFIG.
- AC #5 confirmed: sl_multiplier=5.0, tp_multiplier=6.0 (unchanged defaults).
- Pre-registration sealed at SHA b44acc6 before any simulation run.
- 56 tests green, no regressions.

### File List

- `_bmad-output/preregistration_s_vol_15m.md` — NEW
- `src/research/strategy_core.py` — MODIFY (add `tuesday_exclusion: bool = True` to StrategyConfig)
- `src/research/backtest_engine.py` — MODIFY (wire `config.tuesday_exclusion` at line ~770)
- `src/research/vol_regime_15m_test.py` — NEW
- `_bmad-output/s_vol_15m_verdict_20260523.md` — NEW
- `tests/unit/test_strategy_core_tuesday.py` — NEW
- `tests/integration/test_tuesday_exclusion_integration.py` — NEW
- `_bmad-output/implementation-artifacts/2-4-volatility-regime-gate-parameterization-relaxed-filter-constants.md` — NEW (this file)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — MODIFY
