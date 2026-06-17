# Pre-cutover Verification Results — YANK+MIM-NB Joint Combine Deployment

**Date:** 2026-06-17
**Against:** sealed deployment prereg `preregistration_yank_mim_joint_deployment.md` (185d5f2), §6 binding checks.
**Outcome: BLOCKED — check #3 fails. Cutover does not proceed.**

## Check 1 — ML threshold effective = 0.50 → ✅ PASS
`models/xgboost/tier2_threshold.json` = 0.5. Live YANK log confirms `ML threshold loaded from JSON: 0.5` + `ML Filter: ACTIVE | threshold=0.5`. The backtest `--ml-threshold` wiring hazard does not affect the live path (loads from JSON). Live YANK is genuinely running ml0.50.

## Check 2 — YANK day-deactivation at DLL → ✅ PASS
`yank_streaming_working.py` has a real daily circuit breaker (`check_and_update`): sets `_daily_halted=True` when daily P&L ≤ `max_daily_loss` (= −$750, confirmed in `strategy_config.yaml` + `strategy_core.py`), blocks new entries while halted, persists across restart. Not merely logged. (−$750 is tighter than the MC's modeled −$1,000 → conservative.)

## Check 3 — Topstep overnight/Globex permitted → ❌ FAIL (blocker)
Authoritative Topstep rule (help.topstep.com): trading day runs Sun 5:00 PM CT → weekday **3:10 PM CT close**; Globex/overnight trading IS allowed (reopens 5:00 PM CT), BUT **all positions auto-flatten at 3:10 PM CT, no new entries after 3:08 PM CT, no carrying across the close**.

YANK has **no session-close flatten** — it trades 24h with ≤60-min holds. Two consequences:
1. **Operational:** positions open at 3:10 PM CT get uncontrolled Topstep auto-liquidation; entries after 3:08 PM CT would be rejected.
2. **Analytical (the bigger problem):** the authorizing joint MC used the *unconstrained* YANK seal series. Quantified impact of the Topstep flatten on the 82 seal trades:
   - held across 15:10 CT (force-flattened): **5**
   - exit on a different CT day (overnight carry, force-flattened): **4**
   - entered after 15:08 CT (blocked): **10**
   - **total altered: 18 / 82 = 22% of trades, carrying $3,273 of $7,804 = 42% of YANK's P&L.**

The 64.8% pass / 26.2% blow authorization is therefore **not valid for the live combine** — it modeled a YANK that cannot trade as-is on Topstep.

## Required resolution (before any cutover)
1. Add Topstep-session compliance to YANK: flat by 3:08 PM CT, no new entries after 3:08, no carry across 3:10 PM CT (mirrors MIM-NB's EOD-flat, shifted to the Topstep close).
2. **Re-derive YANK's backtest P&L under that constraint** (a new constrained series; the seal series is unconstrained 24h).
3. **Re-run the joint MC** on the constrained YANK → true joint pass/blow.
4. Re-derive the distance-to-floor and combined-PF triggers if the distribution shifts.
5. Re-seal the deployment prereg against the corrected analysis, THEN proceed.

This is a structural finding, not a tuning issue: YANK-on-the-combine ≠ YANK-as-validated. The standalone YANK paper track record (24h, unconstrained) is also not combine-representative.

---

## RESOLUTION (2026-06-17) — re-validated, deployment re-sealed

**Impact correction:** the initial estimate (22% trades / 42% P&L) was WRONG — it counted allowed evening Globex entries (17:00–23:00 CT) as blocked and used calendar-midnight instead of Topstep's 15:10 CT session boundary. **Corrected impact: 8/82 trades (7 force-flattened + 1 blocked), YANK window P&L $7,804 → $6,316 = −19%.** Still net-positive.

**Re-validation** (`results_yank_mim_joint_constrained.md`, tools `build_yank_topstep_constrained.py` + `joint_combine_mc_constrained.py`): constrained joint MC at 1:2 = **61.2% pass / 29.2% blow** (vs 54.0%/33.3% solo and the unconstrained 64.8%/26.2%). **ADOPT still holds; 1:2 is now the *smallest* qualifying size** (constrained 1:1 only matches baseline). Both derived triggers unchanged: distance-to-floor $500, combined-PF 0.70.

**Disposition:** §6 #3 moves from BLOCKER to RESOLVED-IN-DESIGN. The Topstep flatten is now in the YANK spec (§2) and modeled in the authorizing MC. The one remaining binding item is **live-code enforcement** of the 15:10 CT flatten / 15:08 entry cutoff (part of the YANK→ProjectX execution port). Checks #1 and #2 remain PASS. Deployment re-sealed.
