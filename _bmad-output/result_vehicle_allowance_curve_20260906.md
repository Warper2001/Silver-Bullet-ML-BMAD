# How much drawdown allowance does the strategy actually need?

**Date:** 2026-09-06
**Script:** `tools/vehicle_allowance_curve.py`
**Method:** strategy held fixed, MLL varied, under both floor regimes. Profit target held at $3,000 + the 50%-consistency rule so **the allowance is the only moving variable**. N_SIM=10,000, 90-day horizon, constrained primary pool.

## 1. The requirement, read straight off the drawdown distribution

Max drawdown over a 90-day path with **no floor constraining it**:

| config | p50 | p75 | p90 | p95 | p99 | worst |
|---|---|---|---|---|---|---|
| MIM solo (1ct) | $1,625 | $2,434 | $3,339 | $3,955 | $5,190 | $8,112 |
| MIM 1 : YANK 2 | $1,511 | $2,286 | $3,114 | $3,707 | $5,029 | $8,250 |

**Against the current $2,000 allowance.** The median path draws down $1,625 — 81% of the entire MLL — and **p75 exceeds the whole allowance.** Roughly one path in four suffers a drawdown bigger than the account can absorb, from ordinary variance alone, independent of whether the edge is real.

**Corroboration worth noting:** MIM-NB's *realised* live max drawdown was **$2,386**, which lands almost exactly at **p75** of this simulated distribution. The observed event was not an unlucky outlier — it was a textbook-typical path. The model independently predicts what already happened.

## 2. The curves

### MIM solo (1ct)

| allowance | TRAILING blow | pass | STATIC blow | pass |
|---|---|---|---|---|
| $1,000 | 58.5% | 37.3% | 38.6% | 47.7% |
| $1,500 | 45.8% | 46.4% | 25.3% | 53.5% |
| **$2,000** | **33.3%** | **52.5%** | 16.1% | 55.6% |
| $2,500 | 22.8% | 55.1% | 10.1% | 56.4% |
| $3,000 | 14.2% | 56.3% | 6.5% | 56.6% |
| $4,000 | 4.7% | 56.6% | 2.4% | 56.7% |
| $5,000 | 1.2% | 56.7% | 0.6% | 56.7% |
| $7,500+ | 0.0% | 56.7% | 0.0% | 56.7% |

### MIM 1 : YANK 2 (deployed)

| allowance | TRAILING blow | pass | STATIC blow | pass |
|---|---|---|---|---|
| $1,000 | 53.8% | 43.3% | 33.9% | 55.9% |
| $1,500 | 40.5% | 54.4% | 20.5% | 63.1% |
| **$2,000 (current)** | **28.7%** | **61.5%** | **12.1%** | **66.1%** |
| $2,500 | 18.8% | 65.1% | 7.1% | 67.0% |
| $3,000 | 11.0% | 66.7% | 4.4% | 67.4% |
| $4,000 | 3.2% | 67.5% | 1.5% | 67.6% |
| $5,000 | 1.0% | 67.6% | 0.4% | 67.6% |
| $7,500+ | 0.0% | 67.6% | 0.0% | 67.6% |

## 3. Findings

**More allowance is not a defensive trade-off — it raises profitability too.** On the deployed config, $2,000 → $4,000 trailing takes blow from 28.7% → 3.2% *and* pass from 61.5% → 67.5%. Both axes improve; there is no cost side. **The undersized container is currently costing ~6 points of pass rate** by killing paths that would otherwise have reached target. This matters directly to a profitability-first mandate: the floor is a profit constraint, not merely a risk constraint.

**Static beats trailing decisively, at every allowance — larger than intuition suggested.** (I predicted before running this that the gap might be modest; that was wrong.) On the deployed config at an identical $2,000: blow **28.7% → 12.1%** and pass **61.5% → 66.1%**. Same capital, same strategies, same per-trade risk — only the floor rule differs.

**Useful equivalence: a static $2,000 ≈ a trailing $3,000–3,500.** The ratchet costs roughly $1,000–1,500 of effective allowance.

**The knee is $3,000–4,000, and past $5,000 more room buys literally nothing.** Pass saturates (56.7% solo / 67.6% joint) because the binding constraint stops being survival and becomes the strategy's own ability to reach target. So the requirement is a specific number, not "as big as possible":

- **~$4,000 trailing**, or **~$2,500–3,000 static**, to make the floor a non-issue (blow ≤ ~3%)
- the current **$2,000 trailing is roughly half-sized**

**Diversification confirmed again, from a new angle:** the 1:2 joint config has a *tighter* drawdown distribution than MIM solo (p90 $3,114 vs $3,339) despite carrying more contracts. The documented near-zero YANK↔MIM correlation showing up in worst-case path behaviour, not just in average returns.

## 4. Caveat, stated because it flatters the top of the curve

The profit target is held at $3,000 throughout so the MLL is isolated. **Real vehicles scale the target with account size** — a larger funded account demands more profit, so the upper end of these curves is not free money. The trailing-vs-static comparison at *equal* allowance is unaffected by this, since target is identical on both sides; it is the only comparison here that is fully apples-to-apples.
