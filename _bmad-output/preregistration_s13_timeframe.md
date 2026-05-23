# Pre-Registration: S13 Timeframe Replication — 5m / 15m
**Registered:** 2026-05-23
**Authored by:** Alex (warper2001@gmail.com)
**Experiment ID:** S13
**Status:** SEALED — frozen at commit time. No modifications after commit.

---

## Purpose

S13 is the second Program C Phase 1 falsification experiment: testing whether the
Silver-Bullet FVG+H1-sweep pattern survives resampling to coarser timeframes.

Rationale: a genuine microstructure edge driven by real order-flow patterns should
manifest (at degraded but non-zero signal strength) on adjacent timeframes. If the
pattern exists only at 1-minute resolution and vanishes at 5m and 15m, that is
consistent with noise-fitting to the specific bar granularity rather than a
persistent market structure.

Baseline: the 1-minute strategy produced PF=0.937 over 2025 (129 trades) using the
deterministic `BacktestEngine` (Epic 1, pre-committed StrategyConfig defaults).

---

## Hypothesis

**H₀ (null):** The pattern does not survive resampling — PF at 5m and/or 15m is ≤ 1.0,
consistent with no edge at those granularities.

**H₁ (alternative):** PF > 1.0 consistently at both 5m and 15m, suggesting the pattern
is robust to timeframe choice and not a 1-min granularity artifact.

**Consistency criterion:** Both 5m AND 15m must show PF > 1.0 for H₁ to be supported.
If either is ≤ 1.0, H₀ is supported (pattern is timeframe-specific = consistent with noise).

---

## Architecture / Config Snapshot

Same `StrategyConfig()` defaults as the 1-min baseline — **no modifications**:

| Parameter | Value |
|---|---|
| `sl_multiplier` | 5.0 |
| `tp_multiplier` | 6.0 |
| `atr_threshold` | 0.5 |
| `max_gap_dollars` | 60.0 |
| `max_hold_bars` | 60 |
| `max_pending_bars` | 240 |
| `contracts_per_trade` | 5 |
| `max_daily_loss` | -750.0 |
| `vol_regime_lookback` | 120 |
| `vol_regime_threshold` | 0.75 |
| `min_gap_atr_ratio` | 0.25 |
| `bearish_only` | True |
| `h1_sweep_lookback` | 6 |
| `commission_per_roundtrip` | 4.0 |

Note: `max_hold_bars=60` means 300 min at 5m and 900 min at 15m — far longer than the
1-min 60-min equivalent. `max_pending_bars=240` means 20 hr at 5m and 60 hr at 15m.
These behavioral deltas are accepted; StrategyConfig is NOT adjusted.

**Data:** `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv`, resampled in-process
to 5m and 15m OHLCV. Sealed holdout NOT accessed.

**Resampling:** pandas `resample()` with `open=first, high=max, low=min, close=last,
volume=sum`, then `dropna()` on OHLC. No forward-fill.

---

## Analysis Plan

1. Resample 1-min bars to 5m and 15m in-process (no data leakage)
2. Run `BacktestEngine(path, config=StrategyConfig())` on each resampled dataset
3. Compute PF, WR, Sharpe, trade count per timeframe
4. Compare against 1-min baseline (PF=0.937, 129 trades)
5. Apply consistency criterion: both 5m AND 15m PF > 1.0 required to support H₁
6. Record result in `_bmad-output/s13_verdict_<date>.md`

---

## Stopping Rule

Results are recorded as-is. No re-running with adjusted parameters. No selection of
favorable subsets.

---

## Freeze SHA

*(To be filled in by dev agent after `git commit` of this file)*

Commit SHA: `__FILL_AFTER_COMMIT__`
