# Option 3 — Correlation-aware sizing: status check, not a new test

**Date:** 2026-09-04
**Parent plan:** `_bmad-output/research_plan_post_r3_options_20260904.md`, Option 3.

## This is already done — don't redo it

Before building anything, I checked `project_yank_mim_correlation_portfolio.md`. Option 3 as scoped in the plan doc (test a correlation-aware allocation vs. fixed sizing) **already happened, properly, and is live:**

- Correlation confirmed near-zero via a faithful apples-to-apples recompute: daily Pearson **+0.015** (p=0.82), monthly **−0.167** (p=0.59) — the two edges genuinely don't fire together.
- A pre-registered joint-combine Monte Carlo (`_bmad-output/preregistration_yank_mim_joint_combine_mc.md`, re-seal `75fc1eb`, engine `tools/joint_combine_mc.py`) tested sizing ratios {1:1, 1:2, 1:3 MIM:YANK} against combine pass/blow rates.
- Session-constrained re-run (accounting for Topstep's 15:10 CT auto-flatten): **1:2 → 61.2% pass / 29.2% blow**, beating both the 54%/33% baseline and 1:1.
- **Deployed 2026-06-17 at MIM 1ct : YANK 2ct**, sealed `185d5f2`, with two *derived* halt triggers computed specifically to justify choosing 1:2 over the more conservative 1:1: distance-to-floor ≤ trailing-floor **+$500**, and combined PF **< 0.70** at a 30-trade checkpoint.

Redoing that analysis now would be pure duplication. Closing Option 3 as "already validated" — **except for one thing worth flagging.**

## The gap this review actually found

The 1:2 sizing's case over 1:1 rested on those two derived triggers reducing blow-tail risk (the MC's own framing: pass-rate gain was a timeout→pass conversion, not a blow-rate cost, *because the triggers were there to catch it first*). Twelve days after that sizing went live, `project_combine_floor_gating_removed_20260729.md` records: **`combine-floor-monitor.service` — stopped and disabled.** That is the service the 06-17 deployment prereg named as the enforcement point for exactly those two derived triggers.

So: **the account is currently running the 1:2 sizing whose safety case depended on triggers that are no longer active**, and the sizing decision has not been re-examined against that fact. This isn't a new problem discovered today — the same 07-29 memory already flagged the parallel issue for MIM-NB's own gating — but nobody has connected it specifically to the *joint sizing ratio's* justification, as distinct from each bot's individual risk gate.

## What this means, concretely

- The 1:2 ratio may well still be the right call — but the number that was used to choose it over 1:1 (29.2% blow vs whatever 1:1 scored under the same constrained model) assumed active enforcement that isn't there anymore.
- This directly compounds Option 4's finding: MIM-NB's own realized drawdown ($2,386) already exceeds the account's MLL ($2,000) with **no** floor gating active. The joint sizing sits on top of that same now-absent safety layer.

## Recommendation

Two options, not mutually exclusive:

1. **Re-enable specifically the two derived triggers** (distance-to-floor +$500, PF<0.70) without reversing the rest of the 07-29 decision (the cat-stop philosophy, the "let it run" stance on day-to-day P&L swings) — these two were sized narrowly to protect the *sizing choice itself*, not to re-impose general buffer-gating on entries.
2. **Re-run the constrained joint MC with triggers OFF** (the current actual regime) to get an honest current blow-rate estimate for 1:2 vs 1:1 vs solo — if 1:2's edge over 1:1 evaporates without the triggers, that's the direct evidence needed to either restore them or revert to 1:1.

Either is cheap relative to the other options in this plan — the MC engine (`tools/joint_combine_mc.py`) and both prereg/results docs already exist; this would be a parameter change and a re-run, not new infrastructure. Flagging rather than doing it here, since it touches live sizing on real capital and deserves its own explicit go-ahead.
