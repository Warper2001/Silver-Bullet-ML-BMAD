# VRP-1 Phase 0 — VERDICT 2026-09-07: ⛔ UNDERPOWERED (TERMINAL)

**Seal:** `_bmad-output/preregistration_vrp_gate0.md`, commit `d1afdf5`, §5 Phase 0.
**Instruction:** Alex, "run the power gate".
**Script:** `tools/vrp_phase0_power_gate.py`.
**Outcome:** the sealed rule — *"If MDE > 0.5, the study closes UNDERPOWERED and nothing else
runs"* — is met. **VRP-1 is closed. No Phase 1, no Phase 2, no holdout access.**
**The sealed holdout was never constructed and no VX price series was evaluated for effect.**

---

## 1. Result

Development window 2007-06-21 → 2018-12-31, front-month VX series built per the sealed roll rule
(3 trading days before settlement), from **156 individual monthly contracts**.

| quantity | value |
|---|---|
| N daily observations | 2,775 |
| effective span | **11.01 years** |
| annualized realized volatility | 83.6% *(context only)* |
| **minimum detectable Sharpe (α=0.05 two-sided, 80% power)** | **0.844** |
| sealed bar to detect | **0.500** |
| **achieved power at SR = 0.50** | **38.2%** |
| span required for 80% power at SR = 0.50 | **31.4 years** |

**MDE 0.844 > bar 0.500 → UNDERPOWERED.**

## 2. There is no data rescue — this is the important part

The obvious objection is "spend the holdout too, or pull more history." Neither works:

| span used | MDE Sharpe | power at SR = 0.50 |
|---|---|---|
| dev only (11.01 y) | 0.844 | 38.2% |
| dev + holdout (18.69 y) | 0.648 | 58.0% |
| **every VX bar that has ever existed** (~22.7 y, VX launched 2004) | **0.588** | **66.4%** |

**Detecting an annualized Sharpe of 0.50 at 80% power requires 31.4 years. VIX futures have
existed for about 22.** No amount of additional data, and no finer sampling, closes that gap —
this is the XSMOM-1 lesson restated on a new instrument: **span buys power; sampling frequency does
not.** 2,775 daily observations look like a large sample and are not; for a test about a *mean*,
what counts is the 11 years they span.

**So the study is not merely underpowered as scoped — it is underpowered at its own bar against
the entire history of the instrument.** That is a stronger and cleaner close than a Phase 1 FAIL
would have been.

## 3. What this does and does not say

**It does NOT say VRP has no edge.** No effect size was measured. The firewall held: the script
computes σ and N and *deletes the return series before any mean can be taken* — no Sharpe, no
sign, no P&L was computed at any point, by construction rather than by discipline.

**It says this design cannot resolve an edge of the size this shop requires.** Had Phase 1 run and
returned, say, a Sharpe of 0.6, that result would have been uninterpretable — indistinguishable
from noise at 38% power — and the temptation would have been to treat it as a finding. This gate
cost about ten minutes and prevented that, before the holdout was even built.

## 4. The one thing that must NOT happen next

**Do not raise the bar to 0.844 so the study passes.** The 0.5 bar was inherited from TSC-1
specifically so it could not be tuned to a candidate, and moving it after seeing the power
calculation is the textbook version of this shop's most-documented failure
(`feedback_iteration_loop_pattern`).

If VRP is to be pursued at all, the only legitimate routes are:

1. **A new seal with a pre-declared, independently justified bar.** Note honestly that a bar near
   0.85 is *not* absurd for VRP — published short-VX Sharpes are often quoted in the 0.5–1.0 range
   — but such a bar must be justified from external evidence **before** any result is seen, and it
   materially narrows what counts as success: at that bar, a "real but modest" premium is
   deliberately out of scope.
2. **A genuinely higher-signal construct** (e.g. conditioning on contango depth, which was Phase 2)
   — but that must be sealed as its own primary hypothesis with its own power gate, not smuggled
   in as a rescue for a failed one.
3. **Accept the closure.** Given routes 1 and 2 both start from "VRP needs a bar this shop has
   never used, on an instrument whose entire history is shorter than the test requires," this is
   the honest default.

## 5. Verdict

**VRP-1: CLOSED, UNDERPOWERED (terminal).** Nothing deployed, no holdout constructed or read, no
effect size observed. Option 5 returns to the closed list — no longer "data-blocked" (the data is
demonstrably there and reachable) but **resolution-blocked**: the instrument's entire history is
too short to test the hypothesis at the required effect size.

The reusable output is the feasibility work, which stands regardless:
`_bmad-output/feasibility_vrp_data_20260907.md` — ~20 years of `$VIX.X`/`$VXN.X`/`@VX`, a 9-contract
term structure, individual contracts back to 2007, and Mini-VIX (VXM, $100/pt) as a correctly
sized instrument. Any *future* volatility hypothesis starts from that, and must clear a power gate
first.
