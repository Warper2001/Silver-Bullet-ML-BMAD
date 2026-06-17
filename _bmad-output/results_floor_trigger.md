# Results: Derived Distance-to-Floor Halt Trigger (joint 1:2)

**Run date:** 2026-06-17
**Engine:** `tools/derive_floor_trigger.py` (reuses sealed joint MC pool/rules), 20,000 sims, locked MIM 1ct : YANK 2ct, primary common-day bootstrap.
**Motivation:** replace the *asserted* `correlation > 0.30` trigger and the *asserted* absolute `$48,400` trigger with a single, empirically-derived distance-to-floor circuit breaker. One knob (distance-to-floor); correlation demoted to observe-only. See [[feedback_derive_dont_assert_one_knob]].

## Pre-stated objective (fixed before computing)
Fire the halt-and-review when the account enters the zone where it is **more likely to blow than to pass** — i.e. P(eventual blow | current distance-to-floor) > 0.50. Read the threshold off the crossing; do not pick it by hand.

## Result — P(eventual blow | start-of-day distance-to-floor)

| Distance-to-floor | P(eventual blow) |
|---|---|
| $0–250 | 74.7% |
| $250–500 | 58.9% |
| **$500–750** | **45.5%**  ← crosses below 50% |
| $750–1000 | 36.4% |
| $1000–1250 | 29.1% |
| $1250–1500 | 23.8% |
| $1500–1750 | 20.3% |
| $1750–2000 | 17.3% |
| $2000–2500 | 13.9% |
| $2500+ | 2.0% |

Run check: pass 64.8% / blow 26.2% (matches the sealed joint-MC headline).

## Derived trigger
**Halt-and-review when combined equity ≤ current trailing floor + $500.** The 50% break-even sits between the $250–500 band (58.9%) and the $500–750 band (45.5%); $500 is the smallest band at/under 50%. The curve is smooth and monotonic, so the value is robust, not a knife-edge.

## Why this beats the asserted values it replaces
- The old absolute **$48,400** = only $400 of room at start, which the curve shows is already **~59% blow** — fires late, deep in the death zone. It also stops meaning "$400 of room" once the trailing floor ratchets up; expressing the trigger relative to the *current* floor fixes that.
- The old **correlation > 0.30** was never tested, ~20× the realized 0.015 (likely never fires = theater), and measures a lagging/noisy/scale-blind proxy rather than the dollars-at-floor that actually blow the account.

## Caveats
- Calibrated on the same thin joint pool (18 both-traded days, both MNQ); the *value* is derived but inherits that data fragility. It is a halt-and-**review** circuit breaker (human looks), not an auto-recovery mechanism — near the floor, halting cannot itself create a pass.
- This tunes ONE knob. The combined-PF<0.70 and slippage/replay triggers are unchanged and not re-derived here.
