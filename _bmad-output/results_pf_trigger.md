# Results: Derived Combined-PF Halt Trigger (joint 1:2)

**Run date:** 2026-06-17
**Engine:** `tools/derive_pf_trigger.py` (reuses sealed joint MC pool/rules), 20,000 sims, locked MIM 1ct : YANK 2ct, primary common-day bootstrap.
**Motivation:** the deployment prereg's `combined PF < 0.70 after 30 trades` was an *inherited, undeived* placeholder. Derive the PF threshold. One knob (PF value); N held at the inherited 30-trade checkpoint.

## Pre-stated objective (fixed before computing)
Fire the halt-and-review when the account is **more likely to blow than to pass** — P(eventual blow | running combined PF at the 30-trade checkpoint ≤ x) > 0.50. Read x off the crossing; do not pick it by hand.

## Result — P(eventual blow | running combined PF at trade 30)
84% of sims reach 30 combined trades (the ~3% fast-blows terminate earlier); blow-rate among those reaching 30 = 23.1%.

| Running PF @ 30 trades | P(eventual blow) |
|---|---|
| 0.0–0.5 | 81.2% |
| 0.5–0.6 | 64.3% |
| **0.6–0.7** | **51.7%**  ← crosses 50% |
| 0.7–0.8 | 42.7% |
| 0.8–0.9 | 38.9% |
| 0.9–1.0 | 34.0% |
| 1.0–1.2 | 28.8% |
| 1.2–1.5 | 22.4% |
| 1.5–2.0 | 13.3% |
| >2.0 | 5.3% |

## Derived trigger
**Halt-and-review if running combined PF after 30 trades < 0.70.** The 50% break-even sits exactly at the 0.6–0.7 / 0.7–0.8 boundary: everything below 0.70 is >50% blow, everything above is <50%. So **PF* = 0.70**.

The inherited 0.70 is therefore **empirically calibrated**, not arbitrary — the derivation confirms it as the joint-case break-even. It is upgraded from "placeholder" to "derived." The **N = 30** checkpoint remains inherited from prereg 7939eed (a separate knob, not re-derived in this pass, per one-knob discipline).

## Caveats
- Calibrated on the same thin joint pool (18 both-traded days, both MNQ); the value is derived but inherits that data fragility.
- Halt-and-**review** breaker (human looks), not auto-recovery — a low PF near the floor cannot itself create a pass.
- Companion to the distance-to-floor trigger (`results_floor_trigger.md`); the two catch different failure shapes (slow edge-decay vs. proximity to the kill line).
