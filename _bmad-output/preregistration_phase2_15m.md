# Pre-Registration: Phase 2 — 15m FVG+H1-sweep Sealed Holdout OOS Test
**Registered:** 2026-05-23
**Authored by:** Alex (warper2001@gmail.com)
**Experiment ID:** Phase2-15m
**Status:** SEALED — frozen at commit time. No modifications after commit SHA is used to access holdout.

---

## Purpose

This is the Phase 2 pre-registration for Program C. It gates access to
`data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv`.

**Context:** Phase 1 (Epic 6) produced a PIVOT verdict on the training window
(2025 data). However, the pivot selection was P1 (15m timeframe) based on S13
evidence showing the FVG+H1-sweep pattern survives timeframe resampling
(5m PF=1.026, 15m PF=1.179 on 2025 training data).

This Phase 2 test is the definitive OOS validation: does the 15m signal
hold up on the sealed holdout period (2026-03-01 to 2026-05-19)?

---

## Prior Holdout Access Disclosure

The holdout has been accessed before. The relevant prior 15m access:

| Date | SHA | Result |
|---|---|---|
| 2026-05-20 | `910e95c` | 15m PF=1.8157, **14 trades** (old S13, insufficient sample) |

The prior result had only 14 trades — far below statistical significance.
This test is the dedicated, properly-scoped 15m OOS validation.

No other prior holdout accesses inform the hypothesis or threshold set here.

---

## Hypothesis

**H₁ (alternative):** The FVG+H1-sweep bearish pattern at 15m resolution produces
PF > 1.1 on the sealed holdout period, confirming the pattern is a genuine
structural signal that generalises out-of-sample.

**H₀ (null):** PF ≤ 1.1 on the holdout — the pattern does not generalise
out-of-sample at 15m resolution.

---

## Architecture / Config Snapshot

`StrategyConfig()` defaults — **no modifications for 15m**:

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

**Behavioral deltas at 15m (documented, not compensated):**
- `max_hold_bars=60` means 900 min (15 hr) at 15m — far wider time window
- `max_pending_bars=240` means 60 hr pending — very long
- H1 resample has only 4 bars per H1 bar (coarse)
- Bar ATR is 15-min ATR (larger than 1-min)

These deltas are accepted and not adjusted.

**Data:** `data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv`
(75,081 bars, 2026-03-01 to 2026-05-19)

**Resampling:** 1m → 15m using pandas `resample("15min")` with
`open=first, high=max, low=min, close=last, volume=sum`, `dropna()` on OHLC.

---

## Analysis Plan

1. Commit this pre-registration. Record SHA here.
2. Load holdout CSV. Resample to 15m.
3. Write resampled bars to UTC temp CSV.
4. Run `BacktestEngine(tmp_path, config=StrategyConfig())`.
5. Compute PF, WR, Sharpe, trade count.
6. Apply pass/fail threshold.
7. Append access log entry to `data/sealed_holdout/ACCESS_LOG.md`.
8. Write verdict report `_bmad-output/s_phase2_15m_verdict_<date>.md`.

---

## Pass/Fail Threshold

**PASS:** PF > 1.1 on sealed holdout at 15m.
**FAIL:** PF ≤ 1.1.

Threshold pre-committed in `_bmad-output/phase1_verdict_20260523.md` § Epic 7 Outline.

---

## Sample Size Caveat

Expected ~13 trades (holdout is ~21% of training year; training had 61 trades at 15m).
This is a small sample. The verdict must acknowledge the sample size limitation.
A PF > 1.1 result on 13 trades is encouraging but not conclusive — it is the
best evidence available given the holdout window.

---

## Stopping Rule

Results recorded as-is. No re-running with adjusted parameters.
No selection of favorable subsets (e.g., "exclude the bad month").

---

## Freeze SHA

Commit SHA: `__FILL_AFTER_COMMIT__`
