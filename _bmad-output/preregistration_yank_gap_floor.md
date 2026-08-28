# Pre-Registration: YANK-FLOOR — Loosen the FVG Gap Floor

**Registered:** 2026-08-28
**Status:** SEALED at commit time. Append-only amendments.

---

## 1. Diagnosis — the binding gate is not the one this project has been fixing

YANK earns **13.85 bps net against 0.51 bps friction — 27× headroom — and has traded
11 times since June** (6 / 3 / 2 across Jun / Jul / Aug). That asymmetry has never been
explained. This seal follows a measurement, not a hunch.

**Gate funnel**, streamed from `logs/tier2_bar_decisions.csv`, 70,461 bars from 2026-06-01:

| stage | bars | % of bars | % of previous |
|---|---|---|---|
| H1 sweep active | 30,286 | 42.98% | — |
| + vol regime not blocked | 22,528 | 31.97% | 74.38% |
| + M15 CHoCH confirmed | 11,308 | 16.05% | 50.20% |
| **+ FVG detected** | **13** | **0.0184%** | **0.11%** |

**The FVG gate destroys 99.89% of everything that reaches it.** Every upstream gate is
healthy. Also established while looking: `kill_zone_active` is computed and logged but
**never gates** in `yank_streaming_working.py` — inert, exactly as the room found for
tier2. It is not a constraint and is not touched here.

**Which of the FVG's three size filters?** Measured over 6,936 structural 3-bar bearish
gaps in the live shadow log (2026-06-15 → 2026-08-28):

| filter | passes | |
|---|---|---|
| `gap ≥ 0.5 × 1-min ATR` | 27.9% | |
| `gap ≤ 0.426 × H1 ATR` (**ceiling**) | **99.6%** | not binding |
| `gap ≥ 0.25 × H1 ATR` (**floor**) | **1.5%** | **binding** |

Distribution of `gap / H1_ATR`: median **0.034**, p90 0.112, p99 0.290. **The live floor of
0.25 sits near the 98th percentile of the actual gap distribution.** Removing the ceiling
entirely buys 1.32×; the floor is worth up to 22×.

This corrects a load-bearing assumption in this project. The 2026-08-07 finding and the
gap-ceiling denomination work (PR #37, `YANK_MAX_GAP_ATR_RATIO=0.426`) targeted the
**ceiling**. That fix was correct — it made the window scale-free and non-empty — but the
ceiling was never the thing suppressing frequency.

## 2. Stage 1 — EXPLORATION, disclosed as such

Per `feedback_derive_dont_assert_one_knob` ("sweep to find the best value first"), a
9-cell sweep of `min_gap_atr_ratio` was run on **2025 — data YANK's derivation has already
seen**. This is exploration and is **not** evidence.

| floor | N | freq/day | PF | bps | ratio |
|---|---|---|---|---|---|
| **0.250 (live)** | 80 | 0.220 | **0.944** | −0.45 | **−0.9×** |
| 0.200 | 128 | 0.353 | 1.080 | 0.53 | 1.0× |
| 0.175 | 165 | 0.455 | 1.116 | 0.71 | 1.4× |
| 0.150 | 217 | 0.598 | 1.227 | 1.27 | 2.5× |
| 0.125 | 259 | 0.713 | 1.224 | 1.16 | 2.3× |
| **0.100** | **336** | **0.926** | **1.300** | **1.44** | **2.8×** |
| 0.075 | 395 | 1.088 | 1.177 | 0.78 | 1.5× |
| 0.050 | 474 | 1.306 | 1.186 | 0.74 | 1.5× |
| 0.025 | 506 | 1.394 | 1.197 | 0.75 | 1.5× |

Two things the sweep says, of very different strength:

- **Strong and not a grid artifact:** the live floor of **0.25 is the worst cell tested and
  its PF is below 1.0**. Every loosened cell beats it. That ordering does not depend on
  picking a winner.
- **Weak and explicitly a grid maximum:** 0.10 is the argmax of 9 cells. It sits in a
  plateau (0.15 / 0.125 / 0.10 all PF 1.22–1.30) rather than being a lone spike, which
  helps — but it is still selected from the same data it was measured on, the identical
  multiple-comparisons exposure recorded in the LRC seal.

## 3. The sealed change — ONE KNOB

> **`min_gap_atr_ratio`: 0.25 → 0.10.**

Nothing else moves. `max_gap_atr_ratio` stays 0.426, `atr_threshold` stays 0.5,
sl/tp stay 2.0/8.0, the sweep/CHoCH/vol/Tuesday gates are untouched, direction stays
bearish-only.

## 4. POWER STATEMENT — fixed before the OOS run

Per `preregistration_intraday_momentum_mnq.md` §A1.5. Derived from the 2025 (spent) run at
floor 0.10: per-trade sd = **$509.22** at 5 contracts.

Expected OOS sample: 0.926 trades/day × ~1,461 days ≈ **1,350 trades**.

| | value |
|---|---|
| Smallest mean detectable at t=2.0 | **$27.72/trade (0.940 bps)** |
| Detectable at 80% power | **$38.81/trade (1.315 bps)** |
| 2025-observed mean at this floor | **$42.56/trade (1.443 bps)** |

**The test is adequately powered for an effect of the magnitude Stage 1 observed**
($42.56 > $38.81). Declared in advance: if the OOS effect is materially smaller than 2025's,
this test may not resolve it, and that outcome is reported as underpowered rather than as a
null.

## 5. Out-of-sample data — never seen by YANK

`data/mim_x/mnq_1min_2021_2024_frontmonth.csv` — **1,415,732 bars, 2021-01-03 → 2024-12-31**,
front-month continuous, built from the 23-contract history fetched on 2026-08-28 for MIM-X2.
Each contract contributes only its active window `[3rd Friday −3 months, 3rd Friday]`, so no
bar is spliced across a roll.

**YANK's derivation has never touched 2021–2024.** Its seals and sweeps used 2025 and the
2026 holdout. This is a genuinely virgin four-year window, and it exists only because the
MIM-X2 fetch happened to pull it.

## 6. Primary metric and decision rule

The question is comparative — *should the floor be loosened* — so both floors run on the
same OOS bars.

**Primary:** mean net P&L per trade at floor **0.10** on the OOS window, with a two-sided
t-test, **and** the same figures at floor 0.25 for comparison.

| Condition | Verdict | Pre-committed action |
|---|---|---|
| 0.10 mean > 0, t ≥ 2.0, **and ≥ 1.53 bps** (3× friction), **and** PF(0.10) > PF(0.25) | **PASS** | Record. Triggers §8. **No live change without a deployment seal.** |
| 0.10 mean > 0, t ≥ 2.0, but < 1.53 bps | **MARGINAL** | Record. Real but sub-scale. No deployment. |
| 0.10 mean > 0, t < 2.0 | **UNPROVEN** | Record. No deployment. |
| 0.10 mean ≤ 0 | **FAILS** | The loosening does not survive OOS. Record; close; **no second floor value is tried on this data** (§7.1). |

**Reported, never decision-bearing:** per-year breakdown (a result carried by one year is
disclosed in the verdict line, per this session's September-2025 lesson); the full floor
sweep on OOS data for shape; win rates; exit-reason mix.

## 7. What we will NOT do

1. **No second floor value** if 0.10 fails OOS. One knob, one test, one value.
2. No re-sweep on the OOS data to find a better floor. The OOS sweep is reported for shape
   only and may not be acted on.
3. No adjustment of the ceiling, `atr_threshold`, sl/tp, or any structure gate.
4. No dropping of years, contracts, or trades.
5. No lowering of the 3× economic bar to convert MARGINAL into PASS.
6. **No change to the live bot under any outcome of this document.** A PASS authorises a
   deployment pre-registration, not a deployment.
7. No pooling of the 2025 sweep results with the OOS results.

## 8. Successor trigger

A **PASS** authorises drafting a deployment pre-registration, which must address: YANK's
clock restarting at zero (a floor change is a strategy change, per
`preregistration_yank_gap_ceiling_denomination.md`); the daily-loss breaker at −$300, which
was derived at the old trade frequency and will bind differently at ~4× the trades; and
whether 2 contracts remains correct at the new frequency.

## 9. Disclosed limitations

- 0.10 is a **grid maximum from 9 cells on spent data**. The OOS test is the only thing that
  makes it more than that, and a PASS should be read as "this specific loosening survived
  one clean window", not "0.10 is optimal".
- The backtest runs 5 contracts and −$750 daily loss (`BASE_CONFIG`); live YANK runs 2
  contracts and −$300. **PF and bps are size-invariant; dollar columns are not**, and the
  daily breaker will bind differently live.
- 2021–2024 covers a materially different volatility era than 2025–2026. A PASS across it is
  evidence of robustness; a FAIL could be regime rather than the floor, and those are not
  separable here.
- The 2025 sweep and this OOS test share the same engine and the same bugs, if any.

## 10. Values fixed at seal time

| Item | Value |
|---|---|
| git HEAD at seal | `14549d1` |
| OOS bars | `data/mim_x/mnq_1min_2021_2024_frontmonth.csv`, 1,415,732 rows |
| Sweep script | `yank_gap_floor_sweep.py` |
| Change under test | `min_gap_atr_ratio` 0.25 → 0.10, one knob |
| Expected OOS N | ~1,350 |
| Detectable @ t=2 / @80% power | $27.72 / $38.81 per trade |
| Economic bar | 1.53 bps (3 × 0.51) |

---

# Amendment 1 — Result (2026-08-28)

Append-only. Original sealed text unedited.
Script: `study_yank_gap_floor_oos.py` | Reports: `data/reports/yank_gap_floor_oos_20260828.txt`,
`data/reports/yank_gap_floor_sweep_2025_20260828.txt`. Run under seal `ed8b226`.

## A1.1 Verdict — FAILS

1,415,732 bars, 2021-01-03 → 2024-12-31 (1,457 days), both floors on the same bars.

| floor | N | /day | PF | WR | net | mean | t | bps | ratio |
|---|---|---|---|---|---|---|---|---|---|
| **0.250 (live)** | 500 | 0.343 | **1.064** | 30.4% | +$4,808 | **+$9.62** | 0.514 | 0.33 | 0.6× |
| **0.100 (candidate)** | 1,609 | 1.104 | **0.925** | 26.8% | −$12,946 | **−$8.05** | −1.149 | −0.27 | −0.5× |

The candidate's mean is **negative**, so per §6 the verdict is **FAILS**. Recorded; closed;
per §7.1 **no second floor value is tried on this data**.

## A1.2 The relationship did not merely weaken — it INVERTED

This is the finding, and it is cleaner than a simple failure.

| floor | 2025 PF (sweep) | 2021–2024 PF (OOS) |
|---|---|---|
| 0.250 | **0.944** (worst cell) | **1.064** (best cell) |
| 0.200 | 1.080 | 1.044 |
| 0.150 | 1.227 | 0.993 |
| 0.125 | 1.224 | 0.960 |
| **0.100** | **1.300** (best cell) | **0.925** (worst cell) |
| 0.075 | 1.177 | 0.983 |
| 0.050 | 1.186 | 0.988 |

In 2025, PF **rose** monotonically as the floor loosened, peaking at 0.10. Across 2021–2024
it **falls** monotonically as the floor loosens, and the live floor of 0.25 is the best cell
tested. **The argmax and the argmin swapped places.**

That is the signature of a grid maximum fitted to one regime — the exact exposure §2
disclosed and §9 warned about. Stage 1's confident-looking ordering ("every loosened cell
beats the live floor") reversed completely on four years of unseen data.

## A1.3 The test was adequately powered, so this null is informative

§4 fixed, before the run: detectable at t=2.0 = **$27.72/trade**, at 80% power = **$38.81**,
against a 2025-observed effect of **$42.56**. Realised OOS N was **1,609**, *above* the
~1,350 assumed, so power was slightly better than sealed.

The candidate came in at **−$8.05/trade**. An effect of the magnitude Stage 1 reported would
have been detected comfortably. It is not there.

Note the null is a **null, not a reversal**: t = −1.149 is not significant. The honest claim
is "no positive effect of the 2025 magnitude", not "loosening actively loses money".

## A1.4 The uncomfortable second finding: neither floor clears the economic bar

The live configuration was not the control group anyone expected.

At floor **0.25** on 2021–2024: PF 1.064, **+$9.62/trade, t = 0.514, 0.33 bps — 0.6× friction.**

**The live floor does not clear 1× friction, let alone the project's 3× bar, and its
t-statistic is 0.514.** On four years of unseen data YANK's own sealed configuration is
statistically indistinguishable from zero and economically below its own costs.

This is outside the question §6 asked and is therefore **not a verdict on YANK** — the seal
tested a floor change, not YANK's viability, and 2021–2024 is a different volatility era
from the one YANK was derived in (§9). But it is the most consequential number produced
today and it should not be buried in a secondary table.

## A1.5 Per-year, candidate floor (NOT decision-bearing, §6)

| year | N | mean | net |
|---|---|---|---|
| 2021 | 456 | −$6.81 | −$3,105 |
| 2022 | 377 | +$4.99 | +$1,882 |
| 2023 | 420 | **−$28.12** | −$11,809 |
| 2024 | 356 | +$0.24 | +$86 |

Three of four years are negative or ~zero, and 2023 carries most of the damage. Per the
September-2025 lesson this is disclosed, not acted on.

## A1.6 What this closes

**Closes:** loosening YANK's FVG floor to 0.10. Not viable out-of-sample, and §7.1 forbids
trying another value on this data.

**Does not close:** the diagnosis in §1, which stands and is independent of this result —
the FVG gate really does destroy 99.89% of qualifying bars, the floor really is the binding
filter, the ceiling really is not, and `kill_zone_active` really is inert. Those are
measurements, not hypotheses.

**Raises, for its own seal:** A1.4. If YANK's live configuration is 0.6× friction and
t = 0.514 across four years, the question is no longer how to make it fire more often.

**Live bot unmodified throughout.** No parameter changed, no service restarted, nothing
deployed.

---

# Amendment 2 — Correction to A1.1/A1.4 bps figures (2026-08-28)

Append-only. **The verdict in A1.1 is unchanged.** This corrects arithmetic, not a conclusion.

A1 normalised P&L to basis points using **$59,000** notional per contract — the *2026* MNQ
level, hard-coded in `study_yank_gap_floor_oos.py`. The test window is **2021–2024**, where
the mean MNQ index level was **15,186**, giving notional per contract of **$30,371**.

Friction scales with notional too, so both sides of the ratio move:

| | A1 reported | corrected |
|---|---|---|
| notional/contract | $59,000 | **$30,371** |
| friction | 0.51 bps | **0.823 bps** ($2.50 RT) |
| 3× economic bar | 1.53 bps | **2.469 bps** |
| live floor 0.25 | 0.33 bps, 0.6× | **0.633 bps, 0.77×** |
| candidate 0.10 | −0.27 bps, −0.5× | **−0.530 bps, −0.64×** |

**Nothing that mattered changes.** The candidate's mean is still negative (FAILS), t = −1.149
and t = 0.514 are unaffected by any notional choice, and the live floor is still below 1×
friction and far below the 3× bar. But the numbers were wrong and are corrected here rather
than left to be quoted.

**Lesson:** a bps denominator is a property of the *era being measured*, not of the current
market. Carrying today's notional back four years is the same class of error as A2.2's
threshold-across-regimes finding in GAP-V.
