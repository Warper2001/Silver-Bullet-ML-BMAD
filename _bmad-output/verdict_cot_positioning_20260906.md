# Verdict — COT positioning (E-mini Nasdaq-100)

**Date:** 2026-09-06
**Pre-registration:** `_bmad-output/preregistration_cot_positioning.md` (sealed commit, before any return was computed)
**Script:** `tools/cot_positioning_test.py`
**Data:** CFTC contract `209742`, 1,420 weekly reports 1999–2026; MNQ prices 2021-01 → 2026-06. **284 aligned weekly observations**, publication-lag corrected.

## Result

| | |
|---|---|
| **Spearman(z, forward weekly return)** | **rho = +0.0192, p = 0.7471** |
| Most net-LONG decile (n=29) | +$335.07/wk |
| Most net-SHORT decile (n=29) | +$261.21/wk |
| Contrarian decile spread | **−$73.86/wk** (needs > +$4.00) |
| Unconditional always-long baseline | +$113.60/wk |

| Gate 0 check | result |
|---|---|
| N ≥ 200 (284) | PASS |
| sign is NEGATIVE (rho = +0.0192) | **FAIL** |
| p < 0.05 (p = 0.7471) | **FAIL** |
| decile spread > $4.00 (−$73.86) | **FAIL** |
| net-short leg beats baseline | PASS *(spurious — see below)* |

**VERDICT: FAIL.**

## Reading it

**This is a textbook null, not a near-miss.** rho = +0.019 with p = 0.75 means there is no monotonic relationship between large-speculator net positioning and forward weekly MNQ returns over this window — not a weak one, not a wrong-signed one worth investigating. Nothing.

**The trap, named so nobody walks into it later:** the contrarian spread is −$73.86/wk, which invites "so invert it and earn $74/week." **No.** With rho ≈ 0 and p = 0.75 there is no relationship to invert, and each decile holds 29 observations. That spread is noise, and treating a null result's residual as a momentum edge is precisely the relabelling the seal pre-emptively forbade.

**One of my own Gate 0 checks passed for a bad reason, and it's worth recording.** "Net-short-decile leg beats baseline" passed (+$261.21 vs +$113.60). Given rho ≈ 0, that carries no information — it is a 29-observation bucket in a market that rose throughout. The check was too weak as written: it can pass on noise whenever the underlying correlation is absent. A better formulation would have required the leg to beat baseline *conditional on the correlation test passing first*. Noted for future seals rather than quietly ignored.

**Why every number is positive:** the always-long baseline is +$113.60/wk. MNQ rose substantially over 2021–2026 and that drift dominates every bucket. This is exactly why the seal required beating the baseline rather than beating zero — against zero, all three columns would have looked like edges.

## Incidental observation, heavily caveated

The **weekly** always-long baseline (+$113.60/wk ≈ +$5,900/yr at 1ct) is far stronger than either intraday leg measured earlier today — the day session was +$4.18/day (~+$21/wk, `verdict_option6_calendar_seasonality_20260906.md`) and the overnight hold was weak and tail-dependent (`verdict_option2_overnight_hold_20260905.md`). Longer holding captures the drift that intraday slicing destroys, consistent with the June edge-headroom finding.

**This is not a lead.** It is "MNQ went up during a bull market," measured over a single 5.4-year window with no regime variation. Any strategy claim built on it would need a far longer and more varied sample. Recorded as an observation only.

## Disposition

**COT positioning closed, FAIL.** Per the stopping rule: no threshold sweep, no lookback re-tuning, no second bite via commercials-as-smart-money. Each of those would be a new seal requiring new reasoning, and the mechanism test failing at rho ≈ 0 gives no reason to expect a threshold variant to succeed.

This also closes the last untested component of the original Option 6 ("COT positioning / calendar-seasonality"), whose seasonality half failed separately on 2026-09-06.
