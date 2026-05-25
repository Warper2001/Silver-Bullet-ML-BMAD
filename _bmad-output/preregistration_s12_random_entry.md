# Pre-Registration: S12 Random-Entry Control Test
**Registered:** 2026-05-22
**Authored by:** Alex (warper2001@gmail.com)
**Experiment ID:** S12
**Status:** SEALED — frozen at commit time. No modifications after commit.

---

## Purpose

S12 is the Program C Phase 1 falsification experiment: establishing the null distribution of
Profit Factor for a direction-matched random-entry strategy over the 2025 training window.

The actual Silver-Bullet strategy (H1 sweep + M1 FVG, bearish only) achieved **PF=0.937**
over the full 2025 year (129 trades) using the deterministic `BacktestEngine` (Epic 1).

If the strategy has real edge, its PF should lie in the upper tail of the random-entry
distribution. If it does not, the null hypothesis (random entry = same performance) cannot
be rejected, and Program C mandates pivoting.

---

## Hypothesis

**H₀ (null):** The Silver-Bullet bearish-FVG strategy's observed PF=0.937 on 2025 MNQ 1-min
data is consistent with what a direction-matched random-entry strategy would produce under the
same exit rules and market conditions.

**H₁ (alternative):** The strategy's PF lies in the upper tail (> 90th percentile) of the
random-entry null distribution, indicating a real pattern above chance.

---

## Architecture / Config Snapshot

All parameters from `StrategyConfig()` defaults — **no modifications**:

| Parameter | Value | Notes |
|---|---|---|
| `sl_multiplier` | 5.0 | SL = entry ± 5 × gap_size |
| `tp_multiplier` | 6.0 | TP = entry ± 6 × gap_size |
| `entry_pct` | 0.5 | FVG midpoint |
| `atr_threshold` | 0.5 | FVG ≥ 0.5 × M1 ATR |
| `max_gap_dollars` | 60.0 | USD ceiling |
| `max_hold_bars` | 60 | Time-stop at 60 M1 bars |
| `max_pending_bars` | 240 | Not used in random (immediate entry) |
| `contracts_per_trade` | 5 | MNQ contracts |
| `max_daily_loss` | -750.0 | Daily circuit breaker |
| `vol_regime_lookback` | 120 | H1 bars |
| `vol_regime_threshold` | 0.75 | 75th ATR percentile |
| `min_gap_atr_ratio` | 0.25 | H1 ATR ratio gate |
| `bearish_only` | True | Direction: bearish only |
| `h1_sweep_lookback` | 6 | H1 bars |
| `commission_per_roundtrip` | 4.0 | USD |

**Data:** `data/processed/mnq_1min_2025.csv` (training window only). Sealed holdout NOT accessed.

---

## Decision Rule (verbatim from `_bmad-output/problem-solution-2026-05-20.md`)

| Condition | Verdict |
|---|---|
| Strategy PF < median of random-entry PFs | **PIVOT** — choose from pivot menu P1-P5 |
| Strategy PF > 90th percentile of random-entry PFs | **PATTERNS SURVIVE** — unlock Epic 2 |
| Strategy PF in 50th–90th percentile | **AMBIGUOUS = TREATED AS FAIL = PIVOT** |

**Strategy PF to compare:** 0.937 (BacktestEngine full-year 2025 run, Epic 1).

---

## Simulation Protocol

- **N:** 100 independent simulations
- **Random seeds:** 0–99 (numpy `np.random.default_rng(seed)`)
- **Entry gates shared with real strategy:** H1 sweep active (within last 6 H1 bars),
  vol regime filter passes, daily circuit-breaker not tripped, not Tuesday
- **Entry decision:** uniform random coin flip with probability `p_enter` calibrated to
  match the real strategy's per-bar entry rate (baseline: 129 trades / candidate_bars)
- **Entry price:** `bar.close` at the bar where entry is triggered (conservative proxy)
- **gap_size for SL/TP sizing:** `atr_threshold × calc_atr(recent_m1_bars)` (minimum FVG size)
- **Exit logic:** `strategy_core.check_exit()` unchanged (SL → TP → TIME_STOP resolution order)
- **Direction:** Bearish only (matching real strategy)

---

## Analysis Plan

1. Run N=100 simulations; compute PF, WR, Sharpe per simulation
2. Build empirical distribution across 100 runs
3. Report: min, p25, median, p75, p90, max for each metric
4. Locate strategy PF=0.937 in distribution; compute percentile rank
5. Apply decision rule verbatim; record preliminary verdict

---

## Stopping Rule

Do not run additional simulations after observing results. N=100 is fixed. No cherry-picking
of seeds or additional tuning based on observed distribution.

---

## Freeze SHA

*(To be filled in by dev agent after `git commit` of this file)*

Commit SHA: `7ffb3e0b712f4265478b21ae0e583e57f1249f4e`
