# Results: DIX × GEX Swing-Horizon Interaction (SPX) — FAIL

**Run:** 2026-06-19 · against frozen snapshot `dix_gex_seal_data_20260619.csv` (SHA-256 `46994107…54ba2a`, integrity verified at runtime) · prereg seal commit `a55166c` · script `backtest_dix_gex_swing.py`.

## Verdict: **FAIL** (per pre-registered decision rule, section 3)

The DIX × GEX interaction does not predict forward 10-day SPX returns at the bar set before the data was examined. No resurrection by sub-slicing (prereg kill rule).

## Numbers

Unconditional mean 10-day forward SPX return = **+0.527%**.

| | β3 (DIX×FRAG) | one-tailed p | corner excess (DIX80th & GEX20th) | gate |
|---|---|---|---|---|
| **In-sample 2011–2019** (n=2123, ~212 blocks) | **+0.00063** (right sign) | **0.139** (need <0.05) | **+0.287%** (need ≥+0.8%) | **FAIL** |
| **Sealed holdout 2020–2026** (n=1613, single look) | **−0.00079** (sign FLIPS) | 0.717 | +0.097% | — |

- In-sample: the interaction is the *hypothesized* sign but **statistically weak (p=0.14)** and **economically below the +0.8% floor (+0.29%)**. Fails the dual gate on both counts.
- Holdout: the interaction coefficient **flips negative** — no directional consistency.

## Secondary (descriptive only — not a decision input)

Coiled-spring cell (DIX>80th & GEX<20th, in-sample): **44 independent episodes**, mean 10d return **+1.10%** vs unconditional +0.49% (excess +0.61%) — *but* episode-bootstrap 95% CI **−0.11% .. +2.20% crosses zero**. The intriguing cell mean is not distinguishable from noise, and it is descriptive-only by design.

## Interpretation

This is a clean, honest negative. The pre-registration did its job: the descriptive "coiled-spring" cell (+1.1% mean) is the kind of in-sample story that *looks* like an edge and would have been tempting to trade — but the primary, properly-powered, look-ahead-free interaction test does not clear significance in-sample and reverses sign out-of-sample. Consistent with the broader research prior that DIX/GEX do not yield a robust tradeable edge, even at the swing horizon where the signal is native.

**Disposition:** DIX-GEX-SWING **CLOSED**. Do not deploy. The SqueezeMetrics dataset has now been tested as (a) an intraday MNQ filter and (b) a swing-horizon interaction — both negative. Any future use requires a genuinely new, pre-registered mechanism, not a re-cut of this tape.
