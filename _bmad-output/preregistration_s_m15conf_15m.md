# Pre-Registration: S-M15CONF-15m — M15 Confirmation Filter at 15m

**Registered:** 2026-05-23
**Story:** 2.3 — M15 Confirmation Layer and Resample (15m Reframe)
**Status:** SEALED — do not modify after git commit

---

## Hypothesis

**H₁ (alternative):** Requiring the prior 15m bar to close in the H1 sweep direction (M15
confirmation) produces PF > 1.3 with N ≥ 15 trades on the 2025 training window, relative to
the bearish-only 15m baseline (61 trades, PF=1.179).

**H₀ (null):** M15-confirmed PF ≤ 1.3 OR N < 15.

Pass criterion: **both** conditions must be true:
1. `pf > 1.3`
2. `n >= 15`

Rationale for thresholds (same as Stories 2.1 and 2.2, for comparability):
- PF > 1.3 represents directional improvement over the baseline PF=1.179 with meaningful margin.
- N ≥ 15 is the minimum sample for any PF interpretation.

---

## Prior Results (Epic 2 at 15m)

| Story | Filter | N | PF | Verdict |
|---|---|---|---|---|
| S13 baseline | bearish_only=True | 61 | 1.179 | — |
| 2.1 (bidir) | bearish_only=False | 81 total | 0.985 | H₀ |
| 2.2 (kill zone) | 09:30–11:00 ET | 5 | 0.826 | H₀ |

M15 confirmation is the third filter attempt. `bearish_only=True` is confirmed load-bearing.

---

## Data

- **Training window:** `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv` (full year 2025)
- **Resampled in-process** to 15m (same pattern as previous Epic 2 scripts)
- **Sealed holdout NOT accessed:** `data/sealed_holdout/` is off-limits.

---

## Config Snapshot

```python
StrategyConfig(
    bearish_only=True,          # load-bearing — confirmed by Story 2.1
    m15_confirmation=True,      # NEW field — blocks entries where prior 15m bar misaligns
    # all other fields at defaults
)
```

The `m15_confirmation` field is newly added with default `False`; existing callers are unaffected.

**What M15 confirmation checks:**
- For a bearish H1 sweep: the most recent completed 15m bar before the FVG candidate must close
  bearish (close < open). A doji (close == open) is NOT confirmed.
- Implemented in `check_m15_confirmation(h1_sweep, m15_slice)` in `strategy_core.py`.

---

## Comparison Baseline

| Metric | Value |
|---|---|
| Trades (N) | 61 |
| Profit Factor | 1.179 |
| Win Rate | 0.475 |
| Daily Sharpe | 1.373 |
| TIME_STOP % | ~11% |

---

## Analysis Plan

1. Commit this document before running any simulation.
2. Add `M15Confirmation`, `resample_to_m15()`, `check_m15_confirmation()` to `strategy_core.py`.
3. Add `m15_confirmation: bool = False` to `StrategyConfig`.
4. Wire blocking guard in `BacktestEngine` after FVG detection.
5. Run `src/research/m15_conf_test.py`:
   - **Run A (M15-confirmed):** `StrategyConfig(bearish_only=True, m15_confirmation=True)`
   - **Run B (full window):** `StrategyConfig(bearish_only=True, m15_confirmation=False)` — baseline verification, should ≈ 61 trades.
6. Verify all trades in Run A have `m15_confirmed=True`.
7. Apply H₁/H₀ decision rule to Run A.
8. Write verdict report.

---

## Stopping Rule

No re-testing or parameter tweaking after unblinding. If H₀ is supported, the result stands.
The next experiment must be pre-registered independently before running.

---

## Pre-Registration SHA

`[to be filled after git commit]`
