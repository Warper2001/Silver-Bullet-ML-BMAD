# Pre-Registration: S-KZ-15m — AM Kill Zone Filter at 15m

**Registered:** 2026-05-23
**Story:** 2.2 — AM Kill Zone Filter (09:30–11:00 ET, DST-Aware) at 15m
**Status:** SEALED — do not modify after git commit

---

## Hypothesis

**H₁ (alternative):** Restricting bearish 15m FVG+H1-sweep entries to the AM kill zone
(09:30–11:00 ET) produces PF > 1.3 with N ≥ 15 trades on the 2025 training window.

**H₀ (null):** KZ-restricted PF ≤ 1.3 OR N < 15.

Pass criterion: **both** conditions must be true:
1. `pf > 1.3`
2. `n >= 15`

Rationale for thresholds:
- PF > 1.3 represents a directional improvement over the baseline PF=1.179 with meaningful margin.
- N ≥ 15 is the minimum sample for any interpretation of the PF (avoids 3-trade flukes).

---

## Data

- **Training window:** `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv` (full year 2025)
- **Resampled in-process** to 15m (same ET→resample→UTC temp-file pattern as `bidir_15m_test.py`)
- **Sealed holdout NOT accessed:** `data/sealed_holdout/` is off-limits for this experiment.

---

## Config Snapshot

```python
StrategyConfig(
    bearish_only=True,           # load-bearing — confirmed by Story 2.1
    enable_kill_zone_filter=True,  # NEW field — blocks entries outside 09:30–11:00 ET
    # all other fields at defaults:
    kill_zone_start_et=time(9, 30),
    kill_zone_end_et=time(11, 0),
)
```

No other StrategyConfig parameters are changed. The `enable_kill_zone_filter` field is newly
added with default `False`; setting it `True` activates the existing `kill_zone_filter()` function
as a blocking gate in `BacktestEngine`.

---

## Comparison Baseline

S13 training window (Story 6.2 / Epic 7 Phase 2 pre-reg context):

| Metric | Value |
|---|---|
| Trades (N) | 61 |
| Profit Factor | 1.179 |
| Win Rate | 0.475 |
| Daily Sharpe | 1.373 |
| TIME_STOP % | ~11% |

Bearish-only at 15m. No kill zone restriction.

---

## Analysis Plan

1. Commit this document before running any simulation.
2. Add `enable_kill_zone_filter: bool = False` to `StrategyConfig`.
3. Wire blocking guard in `BacktestEngine` after `kz = kill_zone_filter(bar_ts, config)`.
4. Run `src/research/kz_15m_test.py`:
   - **Run A (KZ-filtered):** `StrategyConfig(bearish_only=True, enable_kill_zone_filter=True)`
   - **Run B (full window):** `StrategyConfig(bearish_only=True, enable_kill_zone_filter=False)` — verification baseline, should ≈ 61 trades.
5. DST verification: assert all entries in Run A have `timestamp_entry` in [09:30, 11:00) ET.
6. Apply H₁/H₀ decision rule to Run A.
7. Write verdict report.

---

## Stopping Rule

No re-testing or parameter tweaking after unblinding. If H₀ is supported, the result stands.
The next experiment must be pre-registered independently before running.

---

## Pre-Registration SHA

`df66bd9` (full: `df66bd9e…` — see `git log --oneline -1`)
