# Results: YANK SL2/TP8 ml0.50 — 2026 OOS holdout verdict

**Date:** 2026-06-15
**Pre-registration:** `_bmad-output/preregistration_yank_sl2tp8_ml050.md`
**Seal commit:** `138cab1b31d064555ede4c9c07503399a743893f`
**Test:** `backtest_tier2_1year_validation.py --preregistration 138cab1 --ml-threshold {0.50,0.00}`
(faithful trader, atr 0.5 + all live gates; model trained on full-year 2025 → all 2026 is OOS)

## Outcome — H₁ SUPPORTED → keep ml_threshold = 0.50

**2026 model-OOS window (Jan–May 2026):**

| Arm | N | PF (aggregate) | Net P&L |
|---|---|---|---|
| ML @0.50 | 54 | ≈1.60 | **+$6,917** |
| No-ML (0.0) | 70 | ≈1.32 | +$3,064 |

The ML filter removed 16 trades whose net contribution was ≈ −$3,850 (net losers), lifting PF
and roughly doubling net P&L.

**Decision rule check (pre-committed):**
- `PF_ml(2026) > PF_noML(2026)` → 1.60 > 1.32 ✓
- `PF_ml(2026) ≥ 1.20` → ✓
- `N_2026(ml) ≥ 25` → 54 ✓
- **Verdict: KEEP `ml_threshold = 0.50`.** (Null outcome would have reverted to ML disabled.)

**Full-period reference (2025-05-19 → 2026-05-19, N=82 ml / 107 no-ML):** ml@0.50 PF 1.557 /
+$7,804 vs no-ML PF 1.135 / +$1,748. (Note: the 2025 portion is in-sample for the model; the
OOS verdict above rests on the 2026 rows only.)

## Caveats
- Low N (54 OOS ML trades); a single small-sample holdout — treat as directional, monitor live.
- Per the forward stop in the prereg: disable ML and re-review if live YANK PF < 0.90 after N ≥ 20.
- Live wiring: the YANK trader (`yank_streaming_working.py`) already gates at 0.50 (loaded from
  `tier2_threshold.json`); the YAML now documents 0.50 (was a dead 0.75). The `tier2_streaming_working.py`
  variant loads 0.0 — confirm which process is the live YANK service if behavior matters.
