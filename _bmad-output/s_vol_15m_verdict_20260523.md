# S-VOL-15m Verdict — 20260523

## Pre-Registration
Sealed at git SHA: `b44acc6`
Doc: `_bmad-output/preregistration_s_vol_15m.md`

## Relaxed Configuration

```python
StrategyConfig(
    bearish_only=True,
    h1_sweep_lookback=10,   # was 6
    min_gap_atr_ratio=0.10, # was 0.25
    max_pending_bars=120,    # was 240
    tuesday_exclusion=False, # was True
)
```

## Results

| Run | N | PF | WR | Daily Sharpe |
|---|---|---|---|---|
| Relaxed config (A) | 104 | 0.881 | 0.442 | -1.060 |
| Full window baseline (B) | 61 | 1.179 | 0.475 | 1.373 |
| S13 baseline | 61 | 1.179 | 0.475 | 1.373 |

## Exit Breakdown

| | TP | SL | TIME_STOP |
|---|---|---|---|
| Relaxed config | 44 | 48 | 12 |
| Full window baseline | 29 | 25 | 7 |

## Monthly Trade Breakdown (Run A)

| Month | Trades |
|---|---|
| 2025-01 | 6 |
| 2025-02 | 10 |
| 2025-03 | 7 |
| 2025-04 | 5 |
| 2025-05 | 13 |
| 2025-06 | 5 |
| 2025-07 | 18 |
| 2025-08 | 15 |
| 2025-09 | 6 |
| 2025-10 | 8 |
| 2025-11 | 3 |
| 2025-12 | 8 |
| **Average** | **8.7** |

## AC Confirmations

- **AC #1** (vol regime parameterized): `volatility_regime_filter()` uses `config.vol_regime_lookback` and `config.vol_regime_threshold` — no hardcoded constants ✓
- **AC #3** (monthly trade count): avg 8.7 trades/month — ✗ FAIL (target ≥ 30)
- **AC #4** (max_pending_bars=120): RELAXED_CONFIG.max_pending_bars = 120 ✓
- **AC #5** (SL/TP finalized): sl_multiplier=5.0, tp_multiplier=6.0 ✓

## Pass Criteria

- PF > 1.3: ✗ (0.881)
- N ≥ 15: ✓ (104)

## Verdict

**H₀ SUPPORTED — fails: PF 0.881 ≤ 1.3**
