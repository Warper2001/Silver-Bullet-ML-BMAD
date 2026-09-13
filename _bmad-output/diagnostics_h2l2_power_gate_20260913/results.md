# H2/L2 second-entry power gate — results (2026-09-13)

- **Plan:** `analysis_plan.md`, sha256 `1e08bb4e…2076`. It was written before any computation, and the script checks its hash before running.
- **Inputs:** `mnq_1min_2025.csv` and `mnq_1min_2026_ytd.csv`, with their sha256 recorded in `results.json`.
- **Window:** 2025-01-02 → 2026-02-27, 297 RTH sessions, 21,925 five-minute bars. The script asserts that no bar on or after 2026-03-01 survived.
- **Firewall:** nothing under `data/sealed_holdout/` was opened. No statistic of price after a real fill was computed.
  - Every dispersion figure comes from placebo pairings, 288 shifts of 5 to ND−5 sessions.
  - The identity pairing is refused, and a synthetic test confirmed the refusal.

## Verdict: UNDERPOWERED, in every pre-declared cell

| Target | Cost | μ_net at θ = 0.10R | Power, in-sample (N_upper / N_lower) | Verdict |
|---|---|---|---|---|
| **1R (primary)** | **$5.80 (primary)** | **+$7.98** | **14.3% / 14.0%** | **UNDERPOWERED** |
| 2R | $5.80 | +$7.98 | 12.6% / 12.3% | UNDERPOWERED |
| 1R | $11.60 | +$2.18 | 6.9% / 6.8% | UNDERPOWERED |
| 2R | $11.60 | +$2.18 | 6.6% / 6.5% | UNDERPOWERED |

**This is terminal for the construct as specified** (plan section 6). No return has been computed, so the 2025–Feb 2026 window is unspent for this construct.

## Why it fails

**Frequency.**
- Only **56 events** passed the filters in 297 sessions: 0.19 per session, 23 long and 33 short.
- Thinning to one trade at a time leaves 53, so position overlap is not what binds.

**The test needs about 1,030 trades.**
- That is for 80% power at the central edge (1R target, primary cost).
- Placebo σ is **$103 per trade** (day-clustered; iid gives $103 too, so there is no clustering penalty).
- Against a net edge of $8, that means about **22 years** of 5-minute MNQ data at the observed rate.

**Risk is wide and skewed.**
- R has a median of $76.50 and a mean of $137.75, with p10 $32.75 and p90 $466.
- 20% of events risk more than $150. These are mostly the high-volatility months of 2025.
- Cost averages 8.5% of R, so the gross edge only breaks even at **0.042R** (0.084R at double cost).
- **The pessimistic effect, 0.05R, barely clears costs:** it would need about 1,166 years. At double cost it is negative.

| θ (gross, R) | Power, in-sample (1R, primary cost) | Years of data for 80% power |
|---|---|---|
| 0.05 pessimistic | 5.9% | ~1,166 |
| **0.10 central** | **14.3%** | **~22** |
| 0.20 optimistic | 47.5% | ~2.9 |

A holdout-only confirmation (55 sessions, projected) would have 8.2% power at the central edge.

**Even the optimistic 0.20R**, the ~60%-at-1:1 figure typical of course material, is only 47.5% powered on the in-sample window. It would need about 2.9 years.

## Exploratory, not pre-registered (`exploratory_frequency.py`, written after the verdict)

**Question:** is the verdict an artefact of translating Wade's 4-tick EMA proximity literally to MNQ as 1.00 pt? The same firewall, effect sizes and costs apply. This section cannot change the verdict above.

| Variant | N | Per session | Power at central | Years for 80% at central | Power at optimistic |
|---|---|---|---|---|---|
| A: plan primary (proximity 1.00 pt) | 56 | 0.19 | 14.3% | 21.7 | 47.5% |
| B: proximity scaled by price (4.00 pt) | 61 | 0.21 | 13.9% | 23.1 | 47.3% |
| C: no proximity filter | 136 | 0.46 | 12.7% | 28.6 | 56.4% |
| D: raw H2/L2, no filters at all | 1,019 | 3.43 | 18.3% | 13.3 | 99.6% |
| ES 5-min, literal 4 ticks, sized as MES | 49 | 0.23 | 2.3% | cost-bound | 11.1% |
| ES 5-min raw H2/L2, sized as MES | 774 | 3.60 | 0.0% | cost-bound | 13.3% |

**What the exploratory runs show:**
- **The proximity translation is not what binds.** Scaling it by price (B) barely changes N.
- **Relaxing the filters doesn't rescue the central case.** Each relaxation adds trades but shrinks R, so the $5.80 cost takes a larger share. At central θ, no variant falls below 13 years.
- **The one powered cell is the raw count at the optimistic edge (D).** That is not Wade's construct, and the edge it assumes is the marketed claim. Picking that variant now would be selection after seeing the gate.
- **On Wade's own instrument (ES, traded as MES), the central edge is cost-bound.** A 5-minute ES signal bar risks about $26 on MES, so $5.80 is about 22% of R.

## Deviations from the plan

1. `exploratory_frequency.py` was written and run after the verdict, and is labelled as such.
2. The plan was hash-pinned but **not committed to git** before the run, so the only evidence it came first is its pinned hash.

## Outputs

- `analysis_plan.md`
- `power_gate.py`
- `results.json`
- `exploratory_frequency.py`
- `exploratory_frequency.json`
- this file
