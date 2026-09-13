# Wedge and measured-move power gates — results (2026-09-13)

- **Plan:** `analysis_plan.md`, sha256 `b5287cb3…b601`. It was hash-pinned before any computation. The one pre-run edit was adding the same-bar cancel rule, made after the synthetic detector tests and before hashing.
- **Inherited gate:** the H2/L2 gate at commit 4482d37, sha256 `35bc2de4…87ef`, verified at run time.
- **Window:** 2025-01-02 → 2026-02-27, 297 sessions, MNQ 5-minute RTH bars. No bar on or after 2026-03-01 survived the filter.
- **Firewall:** the sealed holdout was not opened. No statistic of price after a real fill was computed. All dispersion comes from 288 placebo shifts, with the identity pairing refused.
- **Detector checks:** both detectors were verified on synthetic bars before the run, covering the fill, the cancel on a break of C, and the long/short mirror.

## Verdicts: both UNDERPOWERED on the primary arm

Primary arm: pivot strength s = 1, 1R target, $5.80 cost, central θ = 0.10R.

| Construct | N (upper / lower) | Per session | μ_net | σ (cluster) | Power (upper / lower) | Years for 80% | Verdict |
|---|---|---|---|---|---|---|---|
| **W: wedge reversal** | 371 / 286 | 1.25 | +$4.87 | $87.6 | 28.3% / 24.0% | 6.4 | **UNDERPOWERED** |
| **M: MM target fade** | 889 / 501 | 2.99 | +$3.85 | $88.2 | 36.6% / 25.2% | 4.3 | **UNDERPOWERED** |

**Why this differs from H2/L2:**
- Frequency is not the problem here. H2/L2 had 0.19 events per session; W and M have 1.25 and 2.99.
- **Costs and a short window are the problem.** Cost is 9.3% of mean R for W and 10.0% for M. So the gross edge only breaks even at 0.054R (W) and 0.060R (M).
- That leaves about $4–5 of net edge per trade against σ ≈ $88. At that ratio the test needs about 2,000 trades for W and 3,200 for M, which is **4–6 years** of data at these rates. The window has 1.2 years.

## Power across the declared effect sizes (s = 1, 1R)

| θ (gross R) | W: power (upper) | W: years for 80% | M: power (upper) | M: years for 80% |
|---|---|---|---|---|
| 0.05 pessimistic | cost-bound (μ −$0.47) | — | cost-bound (μ −$0.97) | — |
| **0.10 central** | **28.3%** | **6.4** | **36.6%** | **4.3** |
| 0.20 optimistic | 96.2% | 0.6 | 99.8% | 0.3 |
| 0.10 at $11.60 cost | cost-bound | — | cost-bound | — |
| 0.20 at $11.60 cost | 69.0% | 1.6 | 83.1% | 1.1 |

**Holdout confirmation** (55 sessions, projected) is weak even at the optimistic edge:

| Edge | W | M |
|---|---|---|
| central | 11.8% | 13.9% |
| optimistic | 43.0% | 62.5% |

## Pre-declared sensitivity arms (reported only; they cannot change the verdicts)

| Arm | N | Cost / R | μ_net at central | Power at central | Years for 80% | Power at optimistic |
|---|---|---|---|---|---|---|
| W, s = 2 | 170 | 7.1% | +$6.56 | 22.4% | 9.3 | 81.9% |
| W, s = 3 | 91 | 6.4% | +$7.39 | 17.2% | 14.9 | 61.9% |
| M, s = 2 | 462 | 7.3% | +$6.89 | 42.4% | 3.5 | 99.4% |
| M, s = 3 | 262 | 5.9% | +$9.62 | 40.5% | 3.7 | 97.8% |

- **Stronger pivots trade frequency for risk size.** Wider R shrinks the share lost to costs, but N falls faster.
  - For W, power at central falls as s rises.
  - For M it moves little: 36.6%, 42.4% and 40.5% for s = 1, 2, 3. The fewest years needed, 3.5, is at s = 2.
- **The 2R target lowers power in every arm.** It raises σ by about 20% while μ_net stays the same (the same θ in R).
- **Same-bar ambiguities in M** (a bar both breaking C and trading through D, resolved as a cancel): 9, 3 and 2 for s = 1, 2, 3. That is under 1% of events.

## Reading

- **At the plan's central edge, neither construct is testable on the window this shop uses.** At the optimistic edge, the ~60%-at-1:1 claim typical of course material, both would be detectable within the window.
- **So the gate can't rule them out the way it did H2/L2.** A shortfall caused by the window's length can be fixed with more data; H2/L2's shortfall came from the construct firing too rarely.
- **Data needed at the central edge:** about 4.3 years of MNQ 5-minute RTH history for M, and about 6.4 for W.
  - On disk and unsealed today there are about 2.2 years: 2024 (`mnq_5min_2024.csv`, not used here) plus 2025–Feb 2026.
  - How far back TradeStation's intraday MNQ/NQ history goes has **not been checked**.
- **A longer window is a new construct-plus-window and needs its own plan and gate.** Per the inherited section 6, UNDERPOWERED is terminal for the constructs *as specified here*.
- **Regime shifts would complicate a longer test.** arXiv:2605.11423 reports MNQ intraday behaviour flipping from 2024 to 2025, so a pooled multi-year mean test would need a stationarity check.
- **Priors are low for both constructs.** Both are counter-trend fades, and every earlier MNQ fade here failed after costs (VWAP reversion, VPOC-1, ORB reversion). Nothing in this gate changes that; it only sizes the test.

## Deviations from the plan

1. The plan was hash-pinned but **not committed to git** before the run (the same deviation as the H2/L2 gate).
2. The same-bar cancel rule was added to the plan after the synthetic detector tests and before the hash was pinned and any real-data statistic was computed.

No post-hoc analyses were run.

## Outputs

- `analysis_plan.md`
- `power_gate.py`
- `results.json`
- this file
