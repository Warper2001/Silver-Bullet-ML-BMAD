# Pre-Registration: Program C Phase 1 Falsification Tests
**Registered:** 2026-05-20
**Authored by:** Alex (warper2001@gmail.com)
**Status:** ACTIVE — this document is frozen at commit time. No amendments after the commit SHA is used to access the sealed holdout.

---

## Purpose

This document is the access token for `data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv`.

It pre-commits, in full detail and before any holdout data is observed, the hypotheses being tested, the exact test procedures, and the decision rule that will be applied to the results. Nothing in the decision rule or test procedure may be changed after this document is committed.

If you are reading this while contemplating a "small" adjustment to the test — you are experiencing the failure mode Program C was designed to prevent. The bypass-inoculation section in `_bmad-output/problem-solution-2026-05-20.md` describes exactly this moment. Re-read it before proceeding.

---

## Frozen Parameters

The following parameters define the deployed Tier2 strategy as of commit `6116ba3`. These are **locked** — no parameter may be changed based on holdout observations:

### Risk / Exit
| Parameter | Frozen Value |
|---|---|
| `SL_MULTIPLIER` | 5.0 |
| `TP_MULTIPLIER` | 6.0 |
| `ENTRY_PCT` | 0.5 (FVG midpoint) |
| `MAX_HOLD_BARS` | 60 bars (1 hr from fill) |
| `MAX_PENDING_BARS` | 240 bars (4 hr wait for limit fill) |
| `CONTRACTS_PER_TRADE` | 5 (sizing only; does not affect PF) |

### Filters
| Parameter | Frozen Value |
|---|---|
| Direction | Bearish only (`BEARISH_ONLY = True`) |
| Tuesday | Blocked (`weekday() == 1` returns early) |
| Volatility regime | Block when H1 ATR percentile > 0.75 over 120-bar rolling window |
| `MIN_GAP_ATR_RATIO` | 0.15 (FVG gap ≥ 15% of H1 ATR) |
| `ATR_THRESHOLD` | 0.5 (FVG gap ≥ 0.5× 1-min ATR) |
| `MAX_GAP_DOLLARS` | $60.00 |
| ML filter | Disabled (`ML_THRESHOLD = 0.0`) |
| LR regime filter | Disabled (no `lr_regime_config.json` deployed) |
| Seasonal block | Disabled (`BLOCKED_MONTHS` empty) |

### H1 Structure
| Parameter | Frozen Value |
|---|---|
| H1 sweep window | 6 H1 bars after detection |
| Swing confirmation radius | 2 bars left + 2 bars right |
| Swing confirmation lag | ≥ 2 H1 bars before current bar |

---

## Test S12: Random-Entry Control

### Hypothesis
> The real strategy's profit factor on the sealed holdout period will exceed the 90th percentile of a 50-seed random-entry distribution produced under identical time filters, position sizing, and exit logic.

The null hypothesis is: the strategy's PF is indistinguishable from pure-noise entries (i.e., the FVG + H1-sweep pattern contributes no directional information beyond what a random coin flip would achieve under the same operating conditions).

### Exact Test Procedure

1. Load `data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv`.
2. Apply the same bar-level filters the real strategy applies:
   - Skip Tuesday bars (`bar_et.weekday() == 1`)
   - Skip when outside market hours (same `_is_market_open()` logic)
   - Skip when volatility regime is high (`_vol_regime_high = True`, computed identically to the deployed strategy using `VOL_REGIME_LOOKBACK=120`, `VOL_REGIME_THRESHOLD=0.75`)
   - No position already open
3. At each bar that survives the filters, flip a fair coin (using `numpy.random.default_rng(seed)`):
   - Heads → enter SHORT
   - Tails → enter LONG
4. Entry price = bar close (market order approximation for random entries; not limit).
5. SL = entry ± 5 × ATR₂₀ (same 20-bar ATR as the deployed strategy).
6. TP = entry ± 6 × ATR₂₀.
7. Hold up to `MAX_HOLD_BARS = 60` bars from entry; time-stop at bar close if neither TP nor SL hit.
8. Each bar evaluated once; if a trade is active, skip signal generation.
9. Run 50 seeds (`seed = 0, 1, 2, …, 49`). Record PF for each seed.
10. Compute: median PF across 50 seeds, 90th-percentile PF across 50 seeds.
11. Run the real strategy on the same holdout window (via `backtest_tier2_1year_validation.py --preregistration <this-commit-SHA>`). Record real strategy PF.

### S12 Decision Rule (pre-committed)

| Condition | Verdict |
|---|---|
| Real strategy PF **< median** of 50 random PFs | **PIVOT** — no detectable signal above noise |
| Real strategy PF **between median and 90th percentile** | **PIVOT** — ambiguous; cannot claim edge |
| Real strategy PF **> 90th percentile** of 50 random PFs | **patterns_survive** — proceed to S13 |

"Ambiguous = PIVOT" is intentional. The bar for claiming edge is high: the strategy must beat 90% of random-entry outcomes under the same conditions. Beating 50% (median) is not sufficient.

---

## Test S13: Timeframe Replication

### Hypothesis
> If S12 returns `patterns_survive`, the H1-sweep + FVG pattern on the sealed holdout will produce PF ≥ 1.1 on at least one of the three tested timeframes (1-min, 5-min, 15-min), confirming the pattern is not purely an artifact of the 1-minute bar resolution.

S13 runs only if S12 returns `patterns_survive`. If S12 returns PIVOT, S13 is skipped.

### Exact Test Procedure

1. Load the sealed holdout CSV.
2. For **1-min**: run the real strategy as-is (`backtest_tier2_1year_validation.py`). This is already available from the S12 real-strategy run.
3. For **5-min**: resample holdout CSV to 5-min OHLCV bars (standard OHLCV resample: open=first, high=max, low=min, close=last, volume=sum). Run H1 sweep detection (H1 bars resampled from 5-min, same logic). Run FVG detection on 5-min bars with frozen parameters scaled to 5-min resolution: same multipliers (ATR_THRESHOLD=0.5, MIN_GAP_ATR_RATIO=0.15, MAX_GAP_DOLLARS=$60). Same direction filter, same Tuesday block, same vol-regime gate.
4. For **15-min**: same as 5-min but resample to 15-min.
5. Record PF for each of the three timeframes.
6. `best_TF_PF` = max(PF_1min, PF_5min, PF_15min).

### S13 Decision Rule (pre-committed)

| Condition | Verdict |
|---|---|
| `best_TF_PF` < 1.1 | **PIVOT** — pattern does not generalise across timeframes |
| `best_TF_PF` ≥ 1.1 | **design_phase2_ml_test** — pattern survives; proceed to honest ML hypothesis |

---

## Combined Decision Tree

```
S12 result
├── PIVOT → stop. Execute pivot menu.
└── patterns_survive
    └── S13 result
        ├── best_TF_PF < 1.1  → PIVOT. Execute pivot menu.
        └── best_TF_PF ≥ 1.1  → design_phase2_ml_test
```

---

## Pivot Menu

If any branch returns PIVOT, choose **exactly one** option and commit the decision in `_bmad-output/pivot_decision.md` before beginning new work:

| Code | Option |
|---|---|
| P1 | Different timeframe on MNQ throughout (5-min or 15-min bars as primary signal; full re-spec) |
| P2 | Different asset (ES, NQ, CL, or liquid equity ETF) |
| P3 | Different strategy family (mean reversion, pure momentum, range breakout) |
| P4 | Buy-and-hold benchmark study (understand what passive alternatives deliver; inform realistic return targets) |
| P5 | Invest in infrastructure for future honest validation (data quality, execution simulation, monitoring) |

No new strategy development may begin on the current family (ICT Silver Bullet + FVG + H1 sweep) after a PIVOT verdict.

---

## What Is Not Pre-Committed (and therefore not permitted)

- Running S13 if S12 returns PIVOT
- Adjusting any frozen parameter after observing holdout results
- Re-running with a different date range after seeing the S12/S13 outcomes
- Choosing a different random-entry baseline methodology after seeing the real-strategy holdout PF
- Choosing a different timeframe baseline methodology after seeing the 1-min holdout PF
- Interpreting an ambiguous S12 result (50th–90th percentile) as "patterns_survive"

---

## Acknowledgement

By committing this document, the author pre-commits to all decision rules above. Any deviation from this document constitutes a methodology violation and must be disclosed in full in the session notes and `data/sealed_holdout/ACCESS_LOG.md`.

*This document is intentionally difficult to amend — that is its purpose.*
