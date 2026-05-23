# Pre-Registration: S-BIDIR-15m — Bidirectional FVG at 15m Resolution
**Registered:** 2026-05-23
**Authored by:** Alex (warper2001@gmail.com)
**Experiment ID:** S-BIDIR-15m
**Status:** SEALED — frozen at commit time. No modifications after commit.

---

## Purpose

Story 2.1 (first Epic 2 story, reframed for 15m per phase2_verdict_20260523.md).

Tests whether removing the `bearish_only=True` constraint at 15m resolution
meaningfully increases statistical power (trade count) while preserving edge
(PF > 1.0). This is the statistical power recovery experiment — the FVG+H1-sweep
pattern produces only 61 trades/year at 15m (bearish-only), which is barely
enough for meaningful analysis. Adding bullish setups should roughly double
the opportunity set.

**Not tested in this story:** whether bidirectional at 15m works on the holdout.
Training-window validation only. A subsequent pre-registered holdout test is
required before deployment.

---

## Hypothesis

**H₁ (alternative):** Bidirectional 15m FVG detection recovers ≥ 1.5× trade
count (≥ 91 trades vs 61 bearish-only) while maintaining PF > 1.0.
Both directions must contribute positive PF individually to support H₁.

**H₀ (null):** Trade count increase < 1.5×, or PF drops ≤ 1.0, or one
direction shows PF < 1.0 (drag without signal).

**Consistency criterion:** BOTH bearish and bullish components must show PF > 1.0
individually for the bidirectional result to be considered directionally consistent.

---

## Architecture / Config Snapshot

| Parameter | Value | Change from baseline |
|---|---|---|
| `bearish_only` | **False** | changed (was True) |
| All other StrategyConfig fields | defaults | unchanged |

Bearish-only baseline (S13 training): 61 trades, PF=1.179, WR=0.475, Sharpe=1.373.

**Data:** `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv`, resampled
to 15m in-process. Sealed holdout NOT accessed.

---

## Analysis Plan

1. Commit this pre-registration.
2. Resample 1m → 15m (same `resample_bars("15min")` from timeframe_replication.py).
3. Run `BacktestEngine(tmp_csv, config=StrategyConfig(bearish_only=False))`.
4. Compute total PF/WR/Sharpe/count + per-direction breakdown.
5. Apply consistency criterion.
6. Record result in `_bmad-output/s_bidir_15m_verdict_<date>.md`.

---

## Stopping Rule

Results recorded as-is. No re-running with adjusted parameters.

---

## Freeze SHA

Commit SHA: `__FILL_AFTER_COMMIT__`
