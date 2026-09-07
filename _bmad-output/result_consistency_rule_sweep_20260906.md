# Consistency rules vs a tail-capture strategy — exhaustive sweep

**Date:** 2026-09-06
**Script:** `tools/consistency_rule_sweep.py`
**Why:** the vehicle survey found firms cap best-day profit at 30–50% of total. MIM-NB's profit arrives in ~3 fat days out of 163, so this could disqualify it regardless of drawdown. Never previously tested.

Grid: consistency {30/40/50/100%} × formulation {A, B} × target {$1,500, $3,000} × floor {trailing $2,000, static $2,000} × config {MIM solo, MIM 1:YANK 2}. N_SIM=10,000, 90-day horizon.

## Headline: the formulation matters far more than the percentage

Firms word this rule two non-equivalent ways:

- **A — best day < pct × *realised profit*.** The cap **grows as you profit**, so a blocked path can trade its way out.
- **B — best day < pct × *profit target*.** A **fixed dollar cap**. Breach it once and you are permanently blocked, no matter how much you go on to earn.

Same nominal "30% rule", deployed config, $1,500 target, trailing:

| formulation | pass | hit target but blocked |
|---|---|---|
| **A** (vs realised profit) | **56.6%** | 15.1% |
| **B** (vs target) | **3.1%** | 66.5% |

**An 18× difference from wording alone.** A firm advertising a *50%* rule under formulation B can be far worse than one advertising *30%* under A. **Establishing which formulation a firm uses is more important than comparing their percentages.**

## Full grid — MIM 1 : YANK 2 (deployed)

TRAILING $2,000 — pass% / hit-target-but-blocked%

| target | formulation | 30% | 40% | 50% | 100% |
|---|---|---|---|---|---|
| $1,500 | A | 56.6 / 15.1 | 69.2 / 5.7 | 74.1 / 2.3 | 79.3 / 0.0 |
| $1,500 | **B** | **3.1 / 66.5** | 15.3 / 56.1 | 35.6 / 38.3 | 61.0 / 16.4 |
| $3,000 | A | 51.0 / 10.6 | 58.1 / 3.7 | **61.5 / 0.5** | 62.1 / 0.0 |
| $3,000 | **B** | 23.5 / 38.0 | 37.6 / 24.1 | 37.6 / 24.1 | 62.1 / 0.0 |

STATIC $2,000 — same shape, uniformly better

| target | formulation | 30% | 40% | 50% | 100% |
|---|---|---|---|---|---|
| $1,500 | A | 59.4 / 24.1 | 73.0 / 10.8 | 78.8 / 5.2 | 84.2 / 0.1 |
| $1,500 | **B** | **3.1 / 80.3** | 15.5 / 68.1 | 36.7 / 47.1 | 64.2 / 19.9 |
| $3,000 | A | 53.7 / 13.2 | 61.8 / 5.1 | **66.1 / 0.8** | 66.9 / 0.0 |
| $3,000 | **B** | 24.4 / 42.5 | 39.6 / 27.2 | 39.6 / 27.2 | 66.9 / 0.0 |

The bolded $3,000/A/50% cells are the current Topstep configuration, and they reproduce the earlier runs (61.5% trailing, 66.1% static) — validation intact.

## Why identical cells appear — verified, not assumed

Several thresholds return *exactly* the same result (e.g. $3,000/B at 40% and 50%). That looked like a bug, so it was tested directly. Best-day distribution across paths that reached +$3,000 profit:

| best day | share of target-reaching paths |
|---|---|
| < $900 | 43.4% |
| **$900 – $1,200** | **0.0%** |
| **$1,200 – $1,500** | **0.0%** |
| ≥ $1,500 | 56.6% |

Median best day at target: **$1,581**.

**The distribution is bimodal with an empty band between $900 and $1,500.** MIM-NB trades 1ct with $1,000-scale per-trade outcomes, so a day is either "one good trade" (~$400–800) or "a big one" (≥$1,500) — nothing in between. Caps at $900/$1,200/$1,500 all land inside a region no path occupies, so they behave identically.

**Consequence: the consistency rule is binary for this strategy, not a gradual tax.** There is no threshold to tune into. And **56.6% of all successful paths reach target via a single day that breaches a 50%-of-$3,000 cap** — the fat day that creates the win is the same fat day the rule punishes.

## Correction to my own output

The script's legend calls the 100% column "no consistency rule." **That is only true under formulation A.** Under B, 100% means a cap of 1.0 × target — still a real constraint ($1,500 or $3,000), which is why the B/100% cells (e.g. 61.0% / 16.4%) don't match A/100% (79.3% / 0.0%). The B row has no true no-rule reference. Recorded rather than silently left to mislead.

## What this means for the vehicle search

1. **Ask every firm which formulation they use, before comparing percentages.** This single question dominates the drawdown comparison.
2. **A low target paired with a strict rule is the worst combination**, and it's exactly TradeDay's static profile ($1,500 target, ~30% rule). If that rule is formulation B, the product is effectively unpassable for this strategy — **3.1%**. The static floor and absent daily-loss limit would be irrelevant.
3. **The current Topstep setup (50%, formulation A, $3,000) is close to the best case available on this axis** — 61.5% trailing / 66.1% static, with only 0.5–0.8% of paths blocked by the rule. Whatever else is wrong with the vehicle, its consistency rule is nearly non-binding for us.
4. **Static still helps everywhere** (+3–5pp of pass at equal rule), so the floor argument survives — it is just now clearly second in priority to the consistency formulation.

**Net: the vehicle search should be re-ordered.** Consistency formulation first, then floor regime, then allowance. Shopping on allowance alone could buy a product that is structurally unpassable for a strategy whose edge is concentrated in single days.
