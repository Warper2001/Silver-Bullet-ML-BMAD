# Pre-Registration: yank_sl2tp8_ml050

**Generated:** 2026-06-15
**Experiment ID:** yank_sl2tp8_ml050

---

## Purpose & Context

This seal re-syncs the **running YANK live config** with a pre-registration. The live
`strategy_config.yaml` had drifted *unsealed* from the S25 seal (S25 = SL5.0/TP6.0/ML-disabled;
running = SL2.0/TP8.0). Investigation 2026-06-13/15 also established:
- A SL×TP×ML grid favored wider stops, but that was a **2025 in-sample mirage** — wide-SL fails the 2026 holdout. No SL/TP change is justified; SL2.0/TP8.0 (the running values) are sealed as-is.
- Faithful re-validation (real trader, atr 0.5, all live gates, 2025-05→2026-02): **no-ML PF 1.007 (breakeven)**. ML@0.50 showed PF 1.577 but that window is **in-sample** for the meta-model (trained on full-year-2025 `doe_run_08`), so it is **not** evidence of OOS benefit.
- `ml_threshold` corrected 0.75 → 0.50 (0.75 was a dead value never used for gating; effective gate is 0.50 from `models/xgboost/tier2_threshold.json`).

The one open question this pre-registration tests: **does the ML meta-label filter @0.50 add
value out-of-sample?** The only clean test is the 2026 holdout, where the model (2025-trained) is OOS.

## Hypothesis

**H₁ (alternative):** On the 2026 model-OOS window, the ML filter @0.50 improves the YANK
config: `PF_ml(2026) > PF_noML(2026)` **and** `PF_ml(2026) ≥ 1.20`.

**H₀ (null):** The ML filter does not improve OOS performance:
`PF_ml(2026) ≤ PF_noML(2026)` **or** `PF_ml(2026) < 1.20`.

## Test Procedure (pre-committed)

Run the faithful backtest twice over the sealed range, citing this commit's SHA:
```
backtest_tier2_1year_validation.py --preregistration <sha> --ml-threshold 0.50
backtest_tier2_1year_validation.py --preregistration <sha> --ml-threshold 0.00
```
Compare the **2026 monthly rows only** (model is OOS on 2026; trained on full-year 2025).

## Decision Rule (Pre-committed, Immutable After Seal)

| Criterion | Threshold |
|---|---|
| Primary — OOS ML benefit | `PF_ml(2026) > PF_noML(2026)` AND `PF_ml(2026) ≥ 1.20` → keep `ml_threshold=0.50` |
| Null outcome | otherwise → revert `ml_threshold` to **0.0** (ML disabled, per S25 verdict) |
| Minimum OOS sample | `N_2026(ml) ≥ 25`; below that the result is **INCONCLUSIVE** → default to ML disabled |
| Forward live stop | Disable ML and re-review if live YANK PF < 0.90 after N ≥ 20 live trades |

**Caveats (acknowledged before reading results):** low frequency (~10 trades/mo faithful; 2026
OOS N likely 30–60, filtered fewer); the strategy is ~breakeven without ML; the 2025 in-sample
ML PF (1.577) is NOT evidence. A single small-N holdout is weak — treat the verdict as directional.

---

## Configuration Snapshot

| Field | Value |
|---|---|
| sl_multiplier | 2.0 |
| tp_multiplier | 8.0 |
| entry_pct | 0.5 |
| atr_threshold | 0.5 |
| max_gap_dollars | 60.0 |
| max_hold_bars | 60 |
| max_pending_bars | 240 |
| contracts_per_trade | 5 |
| max_daily_loss | -750.0 |
| vol_regime_lookback | 120 |
| vol_regime_threshold | 0.75 |
| min_gap_atr_ratio | 0.25 |
| ml_threshold | 0.5 |
| bearish_only | True |
| h1_sweep_lookback | 6 |
| kill_zone_start_et | 09:30 |
| kill_zone_end_et | 11:00 |
| commission_per_roundtrip | 4.0 |
| enable_kill_zone_filter | True |
| m15_confirmation | True |
| tuesday_exclusion | True |
| enable_ifvg_fallback | False |
| funding_rate_filter_enabled | False |
| funding_rate_short_threshold | 0.03 |
| funding_rate_long_threshold | -0.02 |
| enable_breakeven_stop | False |
| breakeven_trigger_r | 2.0 |
| enable_trailing_stop | False |
| trailing_stop_mult | 1.5 |

---

## Holdout Data Range

- **Directory:** `data/sealed_holdout/`
- **Start date:** 2026-03-01
- **End date:** 2026-03-01

---

## Integrity Hashes

| Hash | Value |
|---|---|
| (a) YAML config SHA-256 | `faef1c740ed753449796dc948cce15f41722e0b8b60dbb99df3d82e37d1d8b52` |
| (b) strategy_core.py SHA-256 | `4824e46a35228afd4b600ce8ac631e0dc8e43d8b195274848303ec4b68c23598` |
| (c) Git HEAD commit | `36f74e6611d0666bfabffd61d45709303bc0f819` |

*Hash (a): SHA-256 of `strategy_config.yaml` file bytes.*
*Hash (b): SHA-256 of `/root/Silver-Bullet-ML-BMAD/src/research/strategy_core.py` source bytes.*
*Hash (c): `git rev-parse HEAD` at seal time — commit this document to make it tamper-evident.*
