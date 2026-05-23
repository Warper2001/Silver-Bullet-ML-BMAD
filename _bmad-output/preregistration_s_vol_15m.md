# Pre-Registration: S-VOL-15m — Relaxed Filter Constants at 15m

**Registered:** 2026-05-23
**Git SHA (sealed):** `b44acc6`
**Experiment ID:** S-VOL-15m

---

## Hypothesis

**H₁ (alternative):** Bearish-only 15m trades with relaxed entry parameters
(`h1_sweep_lookback=10`, `min_gap_atr_ratio=0.10`, `max_pending_bars=120`,
`tuesday_exclusion=False`) show **PF > 1.3 AND N ≥ 15** over the 2025 training
window, representing a directional improvement over the bearish-only baseline.

**H₀ (null):** Relaxed-config PF ≤ 1.3 OR N < 15. The relaxed parameters do
not produce a reliably profitable regime on the training data.

---

## Decision Rule (Pre-committed, Immutable After Seal)

| Criterion | Threshold | Source |
|---|---|---|
| Profit Factor | PF > 1.3 | Consistent with Stories 2.1–2.3 |
| Minimum trades | N ≥ 15 | Statistical floor |
| Both must pass | AND logic | |

**Stopping rule:** A single BacktestEngine run on the full 2025 training window.
No parameter sweeps, no cherry-picking of sub-periods.

---

## Configuration Snapshot (Exact, Immutable)

```python
RELAXED_CONFIG = StrategyConfig(
    bearish_only=True,          # Story 2.1 verdict: must stay True
    h1_sweep_lookback=10,       # was 6 — relax to detect sweeps further back
    min_gap_atr_ratio=0.10,     # was 0.25 — allow smaller FVG gaps
    max_pending_bars=120,        # was 240 — halve pending timeout
    tuesday_exclusion=False,     # was True — include Tuesday entries
    # All other fields at StrategyConfig defaults:
    # sl_multiplier=5.0, tp_multiplier=6.0, entry_pct=0.5
    # max_hold_bars=60, vol_regime_lookback=120, vol_regime_threshold=0.75
    # enable_kill_zone_filter=False (Story 2.2: H₀, not applied)
    # m15_confirmation=False (Story 2.3: H₀, not applied)
)
```

---

## Deviation from Epics Spec

The epics AC #2 listed `bearish_only=False` in the "full Epic 2 configuration."
This pre-registration intentionally **deviates** from that spec:

- **Story 2.1 verdict** (SHA `e43c0aa`): bidirectional FVG at 15m shows bullish
  PF = 0.826. `bearish_only=True` is confirmed load-bearing. Setting
  `bearish_only=False` would introduce a known losing component.

- **Stories 2.2 and 2.3 verdicts**: kill zone (H₀) and M15 confirmation (H₀)
  are not applied. Stacking failed filters compounds signal dilution.

This deviation is scientifically motivated: we are testing whether **relaxing**
entry constraints (not stacking failed restrictors) improves PF. The deviation
is documented here prior to any simulation run.

---

## Data

- **Training file:** `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv`
- **Resampled:** 1-min → 15-min in-process (same pattern as Stories 2.1–2.3)
- **Period:** Full 2025 calendar year (training window; holdout NOT accessed)
- **Sealed holdout:** `data/sealed_holdout/` — NOT accessed in this story

---

## Baseline for Comparison

| Metric | Value | Source |
|---|---|---|
| Trades | 61 | S13 / bearish-only 15m, 2025 training |
| Profit Factor | 1.179 | S13 baseline |
| Win Rate | 0.475 | S13 baseline |
| Daily Sharpe | 1.373 | S13 baseline |

---

## AC Confirmations (Pre-committed)

- **AC #1 (vol regime parameterized):** `volatility_regime_filter()` already
  uses `config.vol_regime_lookback` and `config.vol_regime_threshold`. No
  hardcoded `0.75` or `120` in `strategy_core.py`. Confirmed before any code
  changes.
- **AC #4 (max_pending_bars):** `RELAXED_CONFIG.max_pending_bars = 120`.
- **AC #5 (TP/SL finalized):** `sl_multiplier=5.0`, `tp_multiplier=6.0`
  (StrategyConfig defaults, unchanged).

---

## What This Pre-Registration Seals

Any subsequent changes to thresholds, timeframes, or configuration after this
commit would constitute a protocol violation. The script `src/research/vol_regime_15m_test.py`
must reproduce the config snapshot above verbatim.
