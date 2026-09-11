# Correction to Option 3 + joint combine MC re-run

**Date:** 2026-09-05
**Supersedes the central claim of:** `_bmad-output/review_option3_sizing_status_20260904.md`
**Script:** `tools/joint_mc_live_accurate_dll.py` (wraps the sealed engine on branch `worktree-joint-mc-prereg`)

## The correction

The Option 3 review claimed the deployed **1:2 (MIM 1ct : YANK 2ct)** sizing "depended on" the two derived halt triggers (distance-to-floor +$500, combined PF < 0.70) that `combine-floor-monitor.service` enforced until it was disabled 2026-07-29 — and therefore that the sizing decision was resting on a safety net that no longer exists.

**Reading the engine shows that is wrong on the load-bearing point.** `joint_combine_mc.simulate()` never modelled those triggers. What it models is: one shared $50k account, a trailing floor (`floor = min(50000, max(floor, bal − 2000))`), blow on `bal <= floor`, Topstep's pass/consistency rule (profit ≥ $3000 with best_day < 0.5 × profit), and per-strategy **daily** loss limits. The triggers were derived *afterwards*, from instrumented runs of that same simulation (`joint_combine_mc_constrained.instrumented()` records per-day states; `crossing()` finds where P(eventual blow) crosses 50%). They were an operational alarm overlay computed **from** the MC, never an input **to** it.

So the sealed pass/blow numbers were produced in a no-trigger world — which is exactly today's world. **Disabling the triggers does not invalidate the sizing decision.** I overstated that.

## Re-run results (constrained primary pool, N_SIM=20,000)

| sizing | pass % | blow % | median days |
|---|---|---|---|
| MIM solo | 51.9% | 34.0% | 45 |
| MIM 1 : YANK 1 | 54.2% | 27.2% | 48 |
| **MIM 1 : YANK 2 (deployed)** | **61.3%** | **28.6%** | 41 |
| MIM 1 : YANK 3 | 64.0% | 32.0% | 35 |

Reproduces the sealed constrained figures (memory records 61.2% / 29.2% at 1:2) to within MC noise — engine validated.

**Against the sealed ADOPT gate (pass > 54% AND blow ≤ 33%): the deployed 1:2 still passes**, with no trigger overlay assumed anywhere in the calculation. 1:2 buys +7.1pp of pass rate over 1:1 for +1.4pp of blow rate. That trade still looks right.

## What the trigger removal actually costs

Not the sizing verdict. The triggers were meant to catch trouble *before* the modelled blow paths played out — an extra layer intended to push realised blow risk **below** the modelled ~28.6%. Without them, ~28.6% is the operative expectation rather than a ceiling. That is a real loss of protection, just a different (and smaller) claim than the one I made.

## Side finding, and it sharpens Option 4

Running the live-accurate daily loss limits (MIM −$1000 per `DLL_GUARD_USD`, YANK −$750 per `max_daily_loss`) instead of the engine's hardcoded −$1000/−$1000 produced **byte-identical results**. Verified why:

| strategy | worst single-day P&L in the pool | live DLL |
|---|---|---|
| YANK | −$217 (1ct) / −$435 (2ct) / **−$652 (3ct)** | −$750 |
| MIM | **−$1,002 (1ct)** | −$1,000 |

**YANK's daily circuit breaker has never come within ~$98 of firing at any sizing in the entire historical pool** — it is effectively decorative at deployed size. MIM's binds on exactly one day, by $2.

The 2026-07-29 memo describes the post-gating-removal state as protected by "two static daily caps" (MIM −$1000, YANK −$750). This says one of those two has never been reachable and the other is a hair's-breadth trigger on a single historical day. **The account's remaining automatic protection is thinner than that description implies** — which strengthens, independently, Option 4's finding that MIM-NB's realised $2,386 drawdown already exceeded the $2,000 MLL.

## Net recommendation

- **Sizing (1:2): no change indicated.** It remains inside its sealed authorization on numbers that never assumed the triggers.
- **The live-risk question is not sizing, it's protection depth.** Between the never-binding YANK cap, the barely-binding MIM cap, no floor gating, and a realised drawdown that has already exceeded the MLL once, the account currently has essentially no automatic brake between a bad sequence and the floor. That is the item worth a decision — re-enabling the two derived triggers is the cheapest way to restore one, and it does not require revisiting the "let it run / take the cat-stop" philosophy that the 07-29 call was actually about.
