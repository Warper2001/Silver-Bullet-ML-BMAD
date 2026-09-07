# Result: vehicle sizing — target-scaled tier sweep (2026-09-07)

Follow-on to `result_vehicle_allowance_curve_20260906.md`, which flagged its own
gap: it varied the trailing MLL while holding the profit target fixed at $3,000,
but real Topstep tiers scale target and MLL together. This closes that gap.

Engine: `tools/tier_vehicle_sweep.py` (this session; reuses `tools/joint_combine_mc.py`
+ `tools/joint_combine_mc_constrained.py` from the sealed 06-17 engine unmodified —
same trade pools, same 1:2 deployed sizing, same per-strategy DLL, same seed).
Not a new pre-registration: this is a sensitivity check closing a disclosed gap in
an already-sealed result, not a new hypothesis test.

## Tiers (current 2026 Topstep Combine pricing)

| Tier | Target | Trailing MLL | Monthly (promo) |
|---|---|---|---|
| 50K (current, acct 26556101) | $3,000 | $2,000 | $49 |
| 100K | $6,000 | $3,000 | $99 |
| 150K | $9,000 | $4,500 | $149 |

Source: [proptradingvibes.com](https://proptradingvibes.com/blog/topstep-trading-combine-rules), [tradecovex.com](https://tradecovex.com/guides/topstep-combine-account-sizes-profit-targets-2026), checked 2026-09-07.

## First pass was wrong — caught before shipping the number

Initial run used the sealed engine's `MAX_DAYS=90` unchanged. Result: pass rate
*collapsed* going up the tiers (150K joint: pass 4.8%, "run"/unresolved 93.1%).
That's a modeling artifact, not a finding — confirmed directly with Topstep
([help.topstep.com](https://help.topstep.com/en/articles/8284197-trading-combine-parameters)):
**the Combine has no time limit.** `MAX_DAYS=90` silently censored any path that
hadn't hit the (now 2-3x larger) target within 90 days as neither pass nor blow.
Re-run at `HORIZON_DAYS=900` (~3.5 yr, run% → ~0 at every tier) below.

## Result (N=10,000, horizon=900d, seed=42)

**MIM solo (1ct):**

| Tier | Target/MLL | Pass | Blow | Median days to pass |
|---|---|---|---|---|
| 50K (current) | $3,000 / $2,000 | 61.1% | 38.9% | 52 |
| 100K | $6,000 / $3,000 | 71.5% | 28.5% | 140 |
| 150K | $9,000 / $4,500 | 86.0% | 13.9% | 248 |

**MIM 1 : YANK 2 (deployed):**

| Tier | Target/MLL | Pass | Blow | Median days to pass |
|---|---|---|---|---|
| 50K (current) | $3,000 / $2,000 | 68.3% | 31.7% | 46 |
| 100K | $6,000 / $3,000 | 80.4% | 19.6% | 118 |
| 150K | $9,000 / $4,500 | 92.4% | 7.6% | 203 |

Monotonic in both directions at every real tier: **pass rate rises, blow rate
falls, all the way up.** The 09-06 intuition survives target-scaling — it was
the 90-day horizon that was wrong, not the original allowance argument.

## What this does NOT settle

- **Time-to-pass triples to quadruples** (46d → 203d joint, 50K→150K). No real
  penalty for that under Topstep's actual no-time-limit rule, but it's real
  calendar exposure — 200+ trading days (~10 months) is a long window over which
  to assume the live edge stays stationary. This shop has watched several
  "edges" die inside a single year; a 150K bet is a bet the current live set
  (YANK + MIM-NB) holds up for most of a year, not ~2 months.
- Contract caps scale too (50K caps at 5ct / 100K at 10ct / 150K at 15ct) —
  irrelevant at the current 1:2 (3ct total) sizing, not modeled as a constraint
  here since it never binds.
- Consistency rule (`best_day < 0.5 * profit`) is modeled; the account-level
  Topstep Daily Loss Limit itself is not — only each strategy's own internal
  DLL is (unchanged from the sealed engine, not a gap introduced here).
- Subscription cost delta is immaterial (~$100/mo) against the P&L stakes.

## Bottom line

The vehicle-sizing case for going bigger holds up under the correct model, and
is stronger than the flawed first pass suggested. The open question isn't
"does more allowance help" — it does, cleanly — it's whether Alex wants to
trade a ~2-month combine cycle for a ~7-month one at 150K (or ~4-month at 100K)
in exchange for a meaningfully lower blow rate and higher pass rate.
