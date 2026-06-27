# BUG: YANK live ML filter silently disabled (fail-open) since 2026-06-17

**Severity:** High (real-money config integrity) — **not** a blow-up risk.
**Status:** Diagnosed, fix proposed, **NOT deployed.** Awaiting Alex's decision.
**Found:** 2026-06-27, via the ML drift-check requested after the OOF tuning probe.
**Component:** `src/research/yank_streaming_working.py` (live YANK combine trader).

## Summary

YANK's ML meta-label filter (sealed config: `ml0.50`) has been a **silent no-op on
the real Topstep combine since the 2026-06-17 Phase-D redeploy**. Every signal is
passed through. YANK has been trading **effectively no-ML**, not the validated
`ml0.50` config. Per the sealed 2026 OOS holdout, `ml0.50` (PF 1.60) outperforms
no-ML (PF 1.32), so the bug has YANK running its **weaker, non-validated** config
on real money. It is *not* a safety/blow-up issue (no-ML was still profitable in the
holdout); it is a **config-integrity + underperformance** bug that ran silently for
~10 days.

## Symptom

- `logs/tier2_filter_log.csv`: every recent decision is `ALLOWED, 1.0, 0.5`.
- `logs/yank_streaming_working.log`: repeated
  `WARNING ML inference failed: The feature names should match those that were
  passed during fit. — returning pass-through`.
- `logs/yank_ml_canary.csv`: the 3 logged live trades all show `ml_proba=1.0`.
- `data/trades.db` `trader-yank`: `ml_proba` NULL on all rows (separate logging gap).

## Root cause — feature-schema mismatch + fail-open fallback

The deployed model `models/xgboost/tier2_meta_labeling_model.pkl` is a
`Pipeline(StandardScaler + LogisticRegression)` fit on **18 features**:
`atr, gap_size, volume_ratio, et_hour, day_of_week, signal_direction,
session_displacement, adr_pct_used, fvg_to_sweep_bars, prior_setup_proximity,
h1_trend_slope, sin_hour, cos_hour, session_volume_ratio, fvg_fill_pct,
bar_body_ratio, sweep_window_vol, slope_direction_match`.

But the live filter's selector lists only **8** (`yank_streaming_working.py:669`):
```python
FEATURE_COLS = [
    'fvg_fill_pct', 'sweep_window_vol', 'volume_ratio', 'signal_direction',
    'h1_trend_slope', 'atr', 'session_displacement', 'session_volume_ratio',
]
```
`predict_proba` does `pd.DataFrame([features])[self.FEATURE_COLS]` → an 8-column
frame is handed to the 18-feature pipeline → sklearn raises *"feature names should
match those passed during fit."* The `except` block **fails open**:
```python
except Exception as e:
    logger.warning(f"ML inference failed: {e} — returning pass-through")
    return 1.0          # <-- silently disables the filter; 1.0 >= 0.50 → ALLOW
```
So every call returns `1.0`, every signal is ALLOWED, and the only trace is a
WARNING line. The model output range is actually **0.37–0.71** (AUC 0.523) — it
*never* legitimately returns 1.0, which is why `1.0` is the unmistakable fingerprint
of the fallback.

**Note:** `_extract_features` already computes and returns **all 18 features**
correctly (the method comment even says "Log all 18 feature values"). The defect is
*purely* that `FEATURE_COLS` selects 8 of them. Fix scope is a one-block edit.

## Git forensics (the regression chain)

| Date | Commit | Event |
|---|---|---|
| 2026-05-31 | `31669a7` | "fix: ML schema — update FEATURE_COLS to 18 features, retrain model, fix disable logic" — applied to **`tier2_streaming_working.py`** + retrained the 18-feature model (current pkl). |
| 2026-06-11 | `629c196b` | "deploy mnq/mes statistical arbitrage combine trader…" — wrote/forked **`yank_streaming_working.py`** carrying the **stale 8-feature `FEATURE_COLS`**. The 05-31 fix was never carried into the new YANK fork. |
| 2026-06-16 | `b61c494` | added the passive ML drift canary (which later recorded the `1.0`s — the canary worked as designed). |
| 2026-06-17 04:47 | `76814e9` | "Phase D cutover: deploy YANK ProjectX port…" — **restarted the live YANK bot on the real combine**, loading the 06-11 broken code. |
| 2026-06-17 22:16 | — | first `ML inference failed` warning in the live log. |

Diagnosis: **`yank_streaming_working.py` is a fork of `tier2_streaming_working.py`
that did not carry the 2026-05-31 18-feature ML-schema fix.** The defect entered the
code on 06-11 and went live on the 06-17 combine restart. `tier2_streaming_working.py`
has the correct 18-feature `FEATURE_COLS` and is unaffected (and it runs ML disabled
anyway, so its identical fail-open is harmless there).

## Why it stayed silent

1. **Fail-open by design:** `return 1.0` on inference error → the filter disables
   itself and *passes* trades instead of refusing them.
2. **WARNING-level only:** no alert/halt; the PF guardrail (PF<0.90 after N≥20) is the
   only actuator, and it can't tell "ML off" from "ML on."
3. **No startup schema check:** nothing validates `FEATURE_COLS == model.feature_names_in_`
   at boot, so a stale fork starts cleanly and runs no-ML indefinitely.

## Proposed fix (NOT YET APPLIED)

### 1. Primary — align `FEATURE_COLS` to the model (the actual bug)
`yank_streaming_working.py:669` — replace the 8-feature list with the 18 the model
expects (copy from `tier2_streaming_working.py`, in `feature_names_in_` order):
```python
FEATURE_COLS = [
    'atr', 'gap_size', 'volume_ratio', 'et_hour', 'day_of_week',
    'signal_direction', 'session_displacement', 'adr_pct_used',
    'fvg_to_sweep_bars', 'prior_setup_proximity', 'h1_trend_slope',
    'sin_hour', 'cos_hour', 'session_volume_ratio', 'fvg_fill_pct',
    'bar_body_ratio', 'sweep_window_vol', 'slope_direction_match',
]
```
`_extract_features` already produces all 18, so no feature-computation work is needed.

### 2. Defensive — fail CLOSED + startup self-check (so this can't recur silently)
`predict_proba` except-branch — block instead of allow on error:
```python
except Exception as e:
    logger.error(f"ML inference FAILED — BLOCKING trade (fail-closed): {e}")
    return 0.0   # was 1.0 (fail-open). 0.0 < threshold → trade is filtered.
```
(Keep the `model is None → return 1.0` path: that is the *intentional* "ML disabled"
config, distinct from an error.)

Add to `MetaLabelingFilter.__init__`, after the model loads:
```python
if self.model is not None and hasattr(self.model, "feature_names_in_"):
    fitted = list(self.model.feature_names_in_)
    if set(fitted) != set(self.FEATURE_COLS):
        raise RuntimeError(
            f"ML schema mismatch: FEATURE_COLS({len(self.FEATURE_COLS)}) != "
            f"model.feature_names_in_({len(fitted)}). Refusing to start with a "
            f"silently-disabled filter. Missing: {sorted(set(fitted) - set(self.FEATURE_COLS))}")
```
This single assertion would have aborted the 06-17 boot instead of running no-ML for
10 days.

## Validation plan (BEFORE any deploy)

1. Apply fix #1; replay recent live bars (or a short backtest) and confirm
   `predict_proba` now returns values in **~0.37–0.71**, not `1.0`. Grep the new
   decision log: no `,1.0,` rows.
2. Confirm the live feature vector matches the model's expected range (no NaN /
   out-of-distribution inputs that would still error).
3. Verify fix #2: a deliberately wrong `FEATURE_COLS` raises at startup (unit test).
4. Treat this as **restoring the sealed `ml0.50` config** that live had silently
   diverged from. Recommend a short pre-registration note recording: the divergence
   window (2026-06-17 → fix date), that live ran no-ML, and that ml0.50 is being
   restored — so the live-performance record is correctly partitioned.

## Deployment

**Do not deploy without Alex's go-ahead.** Restoring the filter changes which trades
YANK takes with real combine money. Steps when approved: apply fixes → validate (above)
→ restart `trader-yank` → confirm decision log shows real probas → monitor first
session. The MNQ live-performance data from 2026-06-17 to the fix date should be
labeled **no-ML** in any analysis (including the floor monitor's PF history).
