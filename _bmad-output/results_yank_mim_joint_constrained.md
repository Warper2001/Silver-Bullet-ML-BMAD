# Results: Joint Combine MC under Topstep-Session-Constrained YANK

**Date:** 2026-06-17
**Why:** §6 pre-cutover check #3 found that the authorizing joint MC used *unconstrained 24h YANK*, which cannot trade as-is on a Topstep combine (auto-flatten 15:10 CT, no entries 15:08–17:00 CT, no overnight carry). This re-runs the analysis on a YANK series that obeys those rules.
**Engines:** `tools/build_yank_topstep_constrained.py` (series transform), `tools/joint_combine_mc_constrained.py` (MC + trigger re-derivation). 20,000 sims, primary common-day bootstrap.

## Constrained YANK series
Force-flatten any position open at 15:10 CT at the real 1-min price; drop entries 15:08–17:00 CT; keep evening/Globex trades (allowed). Impact on the 82 seal trades: **7 force-flattened, 1 blocked, 74 unchanged**; window P&L (5ct) **$7,804 → $6,316 (−19%)**. (This corrects an earlier erroneous −42% estimate that wrongly counted allowed evening Globex entries as blocked and used calendar-midnight instead of the 15:10 CT session boundary.) Constrained YANK remains net-positive.

## Joint MC — constrained vs unconstrained (primary)

| Size (MIM:YANK) | Unconstrained | **Constrained (governs)** |
|---|---|---|
| MIM-only baseline | 54.0% / 33.3% | 54.0% / 33.3% |
| 1:1 | 56.4% / 25.6% | **54.0% / 27.8%** |
| **1:2** | 64.8% / 26.2% | **61.2% / 29.2%** |
| 1:3 | 67.6% / 29.2% | 63.7% / 32.2% |

(pass% / blow%)

## Decision (sealed gate: smallest YANK size with pass% > 54% AND blow% ≤ 33%)
- **1:1 → NO** (pass 54.0% = baseline, no improvement).
- **1:2 → QUALIFIES** (pass 61.2% > 54%, blow 29.2% ≤ 33%).
- 1:3 → qualifies but larger and nearer the blow ceiling (32.2%).

**Verdict: ADOPT MIM 1ct : YANK 2ct — unchanged.** Under the faithful constraint, 1:2 is now the *smallest* qualifying size (the constraint removed 1:1's marginal pass edge), which independently confirms the 1:2 choice. The joint still beats MIM-solo on both axes (pass +7.2pp, blow −4.1pp), with a thinner margin than the unconstrained 64.8%/26.2%.

## Triggers — re-derived on constrained data, both UNCHANGED
- **Distance-to-floor:** P(blow) crosses 50% at the $500–750 band (47.6%) vs $250–500 (60.4%) → **d* = floor + $500** (same as unconstrained).
- **Combined PF @ 30 trades:** crosses 50% at 0.6–0.7 (51.2%) vs 0.7–0.8 (46.6%) → **PF* = 0.70** (same).

## Still required before cutover (live-code gap)
The analysis now models a flattened YANK; the **live YANK bot must be made to match** — add: flat by 15:10 CT, no new entries 15:08–17:00 CT, no carry across the close. Until the live code enforces this, live behavior ≠ this (re-validated) analysis. This is part of the YANK→ProjectX execution-port work.
