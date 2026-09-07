# Verdict — Option 6: MNQ calendar seasonality

**Date:** 2026-09-06
**Pre-registration:** `_bmad-output/preregistration_option6_calendar_seasonality.md` (sealed commit `6775d0d`, before any P&L was computed)
**Script:** `tools/option6_calendar_seasonality.py`
**Data:** 1,300 RTH sessions, 2021-01-04 → 2026-06-11 (~5.4 years), day-session open→close, 1ct, $4.00 RT.

## Result

Unconditional always-long baseline: **+$4.18/day**, +$5,436 total.
Bonferroni across 7 tests: α = 0.00714 → cells must clear the **99.29th** null percentile.

| cell | N | mean $/day | null p99.29 | ex-top5 cell vs base | verdict |
|---|---|---|---|---|---|
| Turn-of-month | 264 | +23.57 | 55.67 | +3.87 vs −3.35 | FAIL |
| **Mon** | 240 | **+56.49** | **58.97** | +36.60 vs −3.35 | **FAIL (near-miss)** |
| Tue | 272 | −5.45 | 58.24 | −19.44 | FAIL |
| Wed | 267 | +7.68 | 56.03 | −21.31 | FAIL |
| Thu | 262 | **−33.94** | 59.11 | −58.40 | FAIL |
| Fri | 259 | +0.78 | 55.72 | −15.26 | FAIL |
| Pre-holiday | 52 | +62.62 | 144.18 | +4.53 vs −3.35 | descriptive only (N floor, pre-declared) |

**VERDICT: FAIL — no gate-eligible cell passes all four checks.**

## The seal earned its keep

**Monday would have been called an edge under a looser design.** It beats the baseline 13.5× (+$56.49/day vs +$4.18), and it survives the fat-day check comfortably (+$36.60/day with the top 5 days removed, against a baseline that goes *negative* at −$3.35). On an uncorrected 95th-percentile null it would almost certainly have passed.

It fails on the **Bonferroni-corrected** bar: $56.49 against a required $58.97. That is a near-miss, not a refutation — and the honest reading is "not established at this N," not "definitively nothing." Which is precisely why the correction was fixed in the seal before looking: with 7 cells on the table, a 95th-percentile pass is roughly what you'd expect from noise alone.

The reason nothing clears is visible in the null thresholds themselves (~$56–59/day): **daily MNQ P&L is noisy enough that a random 240–270-day sample routinely produces means in that range.** Effect sizes here, even the real-looking ones, are small relative to daily variance at this sample size.

## Two observations recorded, explicitly not acted on

Per the stopping rule (no re-slicing, no post-hoc interactions), neither of these is pursued on this seal:

1. **Monday (+$56.49) and Thursday (−$33.94) are the two largest divergences, in opposite directions.** A long-Monday/short-Thursday pair is an obvious next thought — and is exactly the post-hoc construction the seal forbids. It would need its own pre-registration, ideally with a reason to expect it beyond "it showed up here."
2. **Curious echo, no claim attached:** this project already runs a *validated* Thursday-short (crypto — `project_kraken_thursday_short_20260621`). MNQ Thursday being the worst weekday here is a coincidence worth noting and nothing more; different asset, different mechanism, and this cell failed its own gate.

## Incidental finding worth keeping

**The always-long day-session baseline is barely viable on its own:** +$4.18/day gross of nothing but the $4 RT cost, and it turns **negative (−$3.35/day) once the top 5 days are removed.** Over 5.4 years, MNQ's day-session drift is a handful of days plus noise.

Read alongside Option 2 (overnight always-long: PF 1.154, also collapsing without its top 5 nights), the pair says something coherent: **neither leg of the 24h session — day or overnight — carries a robust standalone directional drift for MNQ after costs.** Both are weakly positive and both are tail-dependent. That's a more useful takeaway than either verdict alone.

## Disposition

- **Option 6 closed, FAIL.** No re-slicing on this seal.
- **Option 5 (volatility risk premium) not attempted — data-blocked, not failed.** The repository contains no options chains, no VIX/VX history, and no implied-volatility series. Testing it requires acquiring one of: NQ or QQQ options chains with IV, or VIX index + VX futures term-structure history. It is additionally **not combine-eligible** (options are not tradeable on the Topstep futures combine), so it is a self-funded-capital decision before it is a research task. Deliberately not approximated with a futures-only proxy, which would test a different hypothesis under the VRP label.
