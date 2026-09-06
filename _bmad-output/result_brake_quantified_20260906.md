# The brake, quantified — joint combine MC with halt triggers active

**Date:** 2026-09-06
**Script:** `tools/joint_mc_with_brakes.py`
**Why:** the sealed engine never modelled the two derived halt triggers (they were derived *from* instrumented runs, not inputs *to* them — see `correction_option3_and_joint_mc_rerun_20260905.md`). This runs them as real halts so the brake's cost/benefit is measured rather than argued.

**Triggers:** halt when combined equity ≤ trailing floor + $500, or when combined PF < 0.70 at the 30-trade checkpoint.

## Results (constrained primary pool, N_SIM = 20,000)

| config | brakes | pass | blow | frozen | timeout |
|---|---|---|---|---|---|
| MIM solo | off | 51.9% | 34.0% | — | 14.1% |
| MIM solo | **ON** | 43.6% | **1.2%** | 50.2% | 4.9% |
| MIM 1 : YANK 1 | off | 54.2% | 27.2% | — | 18.6% |
| MIM 1 : YANK 1 | **ON** | 47.2% | **0.8%** | 44.0% | 7.9% |
| **MIM 1 : YANK 2 (deployed)** | off | 61.3% | 28.6% | — | 10.0% |
| **MIM 1 : YANK 2 (deployed)** | **ON** | **52.0%** | **0.8%** | 43.9% | 3.4% |
| MIM 1 : YANK 3 | off | 64.0% | 32.0% | — | 4.0% |
| MIM 1 : YANK 3 | **ON** | 52.6% | **1.6%** | 44.8% | 1.0% |

Brakes-off cells reproduce the earlier run exactly — validation intact.

## The exchange rate, and it's consistent

| config | blow avoided | pass surrendered | ratio |
|---|---|---|---|
| MIM solo | −32.8pp | −8.3pp | ~4.0 : 1 |
| 1 : 1 | −26.4pp | −7.0pp | ~3.8 : 1 |
| **1 : 2 (deployed)** | **−27.9pp** | **−9.4pp** | **~3.0 : 1** |
| 1 : 3 | −30.4pp | −11.4pp | ~2.7 : 1 |

**The brake nearly eliminates the catastrophic outcome** — blow rate collapses to under 2% in every configuration — at a cost of roughly 7–11pp of pass rate. Across all four sizings you give up about one point of success to avoid roughly three points of ruin.

## Three things that change how to read this

**1. The 52.0% is a lower bound, not an expectation.** A halt is modelled as *permanent* — the path ends "frozen." That was chosen deliberately so the brake couldn't be flattered by inventing a human who re-enables it. In reality frozen ≠ dead: the account is alive and above the floor. **In Topstep terms this distinction is the whole game** — a blown account is permanently ineligible for funding (exactly what happened to 23884932, "exceeded max loss limit and remains ineligible"), whereas a self-imposed halt is reviewable and resumable. True brakes-on performance sits somewhere between 52.0% and 61.3%, depending entirely on operational review.

**2. That makes 43.9% "frozen" the number carrying the most weight — and its meaning is an operations question, not a modelling one.** If a halt fires and nobody looks at it for a week, frozen behaves like failed. If it's reviewed same-day and correctly resumed, frozen is far better than blown. The brake's real value is therefore contingent on there *being* a review process.

**3. With brakes on, sizing stops mattering much.** Pass is 47.2 / 52.0 / 52.6 across 1:1 / 1:2 / 1:3 while blow sits near 1% throughout. The brake dominates the sizing lever. Notably **1:3 buys almost nothing over 1:2** (52.6 vs 52.0) — no case for increasing size. And YANK still earns its place: MIM solo with brakes is 43.6% vs 52.0% at 1:2.

## Honest caveats

- With brakes on, the deployed 1:2 pass rate (52.0%) falls **below the sealed ADOPT gate's >54% threshold**. That gate was written for a no-brake world, so applying it here is arguably a category error — but it should not be silently ignored.
- The PF trigger fires once, at the first day-end past 30 trades. A continuously-evaluated version would halt more often; not tested here (one knob at a time).
- The floor trigger's $500 buffer is **less than MIM-NB's $1,000 single-trade cat-stop**, so it can only fire once there is already less than one trade's risk of room left. It is a late brake by construction.

## Read

On these numbers the brake is a strongly favourable trade — near-total elimination of permanent ineligibility for a modest, and probably overstated, pass-rate cost. The case against it is not the arithmetic; it's the operational question of what actually happens when a halt fires, and the separate structural problem (`review_option4_combine_vehicle_fit_20260904.md`) that MIM-NB risks $1,000 per trade against a $2,000 total allowance on this vehicle regardless of brakes.
