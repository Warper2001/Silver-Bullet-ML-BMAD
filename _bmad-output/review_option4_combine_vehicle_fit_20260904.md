# Option 4 — Combine-vs-tail-capture vehicle fit: decision memo

**Date:** 2026-09-04
**Parent plan:** `_bmad-output/research_plan_post_r3_options_20260904.md`, Option 4.
**Type:** strategic review — not a Gate 0 backtest. Data is real (queried from `data/trades.db` and project memory), the recommendation is a judgment call for Alex, not a pass/fail verdict.

## The mismatch, quantified

MIM-NB's edge is documented tail-capture — a handful of fat-tail days carry the strategy, most days are noise. That shape needs to *survive* through real variance to collect the rare payoff. Pulling MIM-NB's actual live trade history (`trades.db`, `trader-mim-nb`, 2026-06-11 → 2026-09-04, N=24 trades / 22 trading days):

| metric | value |
|---|---|
| Total realized P&L | +$300.50 |
| Worst single trade | −$1,000.00 |
| Worst 5 days | −$1,000, −$514.50, −$500, −$500, −$500 |
| **Max drawdown (daily-close basis)** | **$2,386.00** |
| Current combine trailing MLL | **$2,000.00** |

MIM-NB's own **realized** worst drawdown — not a hypothetical, not a backtest, an actual sequence that happened live — already exceeds the combine's entire MLL budget by $386, in under three months of live trading. Daily-close drawdown is a floor on the true worst case, not a ceiling (an intraday peak-to-trough could be larger) — so this is if anything an understatement.

## Current combine risk architecture (as of this review)

Per `project_combine_floor_gating_removed_20260729.md`: on 2026-07-29 Alex made a **deliberate, informed decision** to remove all floor-derived gating from both bots ("*let MIM-NB blow the account*"), keeping only two static daily caps — MIM −$1,000, YANK −$750 — plus a monitor that computes the trailing floor correctly but is report-only (`FLOOR_MONITOR_REPORT_ONLY=1`, reaffirmed the same day). This review does not second-guess that call; it's presenting what it implies given the number above.

**What that architecture does and doesn't cover:** the static daily caps stop a *single bad day* from doing more than $1,000/$750 of damage. They reset every day and never look at the cumulative trailing floor — so a sequence of moderate days (well under either daily cap individually) can walk equity down toward the $2,000 MLL with nothing watching in real time. That is structurally what happened to the *previous* account (23884932, blown 2026-07-06) — the proximate cause was a floor-tracking bug (`project_combine_blown_20260706.md`), but the underlying exposure — equity swings sized like MIM-NB's real drawdowns colliding with a $2,000 trailing budget — is not the bug. The bug determined *when* it happened; the collision was coming either way given the strategy's realized variance.

## The vehicle already in hand for exactly this problem

`project_topstep_labs_2k.md` already identified and priced a **static-MLL** account ($25K size, $1K static MLL, $75 one-time) and noted: *"Static-MLL accounts are immune to this entire failure class"* (`project_combine_blown_20260706.md`). That vehicle was earmarked for **GAP-1** — a strategy whose whole shape (thin, frequent, mean-reverting, same-day exit) is a poor match for a trailing account's cumulative-drawdown sensitivity anyway, but whose per-trade risk is small and steady, exactly what a *static* cap tolerates well.

That pairing may be backwards. GAP-1 doesn't need tail-survival room — its risk is per-trade and bounded (2× gap stop). **MIM-NB is the one that structurally needs room to survive a drawdown before its tail day arrives**, and a static MLL that never ratchets down would give it that room in a way the current $50K trailing combine — with all its floor protection now off — does not.

## Options, plainly

1. **Status quo.** Accept that MIM-NB is running on a vehicle whose MLL its own realized drawdown has already exceeded once, with no active protection between it and a repeat. Alex's 07-29 call means this is a live, chosen tradeoff, not an oversight — but it should be a *reviewed* one, not a forgotten one.
2. **Re-enable floor/buffer gating on MIM-NB specifically.** Reverses part of the 07-29 decision. Trades the "let it run and take the cat-stop" philosophy for survival odds; the historical cost was already measured once — buffer-gating blocked 4 consecutive qualifying signals over $0.88 of shortfall in July, a real opportunity cost to weigh against the $2,386 drawdown fact above.
3. **Move MIM-NB (or split its size) onto a static-MLL vehicle** — the Topstep Labs 2K account already priced, or a similar static-cap product — instead of, or alongside, the trailing $50K combine. Structurally immune to the exact failure class that already killed one account.
4. **Run MIM-NB self-funded**, no MLL at all, sized to Alex's own risk tolerance rather than a funded-account rule that was never designed around a tail-capture return profile. Removes the vehicle mismatch entirely at the cost of using own capital instead of a funded account's.
5. **Swap the pairing**: GAP-1 → the static-MLL account (it was already slated there and fits better than originally framed), MIM-NB → self-funded or a materially larger/looser-drawdown vehicle, freeing the $50K trailing combine for strategies whose risk profile is genuinely bounded per-trade rather than tail-dependent.

## Recommendation

Option 5 is the one this data actually points at: it's not "protect MIM-NB more," it's "MIM-NB and the current trailing-MLL combine are the wrong pair, independent of gating settings." Gating (options 1–2) treats the symptom; re-pairing strategies to vehicles (options 3–5) treats the mismatch this memo measured. This is Alex's call, not a Gate 0 verdict — flagging it with the number that makes it concrete: **the drawdown already happened once, live, and it was bigger than the account's entire risk budget.**
