# Option B Gate 0 Scout — VERDICT: OOS FAIL, CLOSED

**Date:** 2026-07-03
**Prereg:** `_bmad-output/preregistration_option_b_impulse_aftermath.md`, sealed at commit `f2b8850` BEFORE any P&L evaluation (only event counts were known at seal time).
**Script:** `tools/option_b_gate0_scout.py` (one tz-handling bugfix applied post-seal; no spec change).
**Holdout access:** logged in `data/sealed_holdout/ACCESS_LOG.md` with prereg SHA before the OOS run; single shot, executed verbatim.

---

## Verdict chain

**Dataset gate: PASS.** IS (2025) impulse events: K=4 → 618/yr, K=6 → 235/yr, K=8 → 99/yr. Frequency is not the constraint (unlike the FOMC event-fade scout).

**Gate 0 (IS 2025): PASS — all four sealed criteria.**

- Selected cell (mechanical rule: max net PF, N≥60): **K8 / follow / H60** — N=99, net PF **1.605**, expectancy **+$41.80/trade** (1ct, $6 RT costs in).
- Null test: beat the 95th percentile of 200 random-entry samples (1.569) — narrowly (4.5% of null ≥ selected). This thinness was the first warning sign.
- K-robustness: K4/follow/H60 PF 1.318, K6/follow/H60 PF 1.493 — direction consistent.
- Fat-day check: ex-top-3-days total +$425 (PF 1.062) — passed the sealed bar (> $0), but barely; second warning sign.
- Structural observation: **all 12 follow cells profitable, all 12 fade cells losing** on IS. 2025 impulse aftermath was uniformly continuation.

**OOS (2026-01-01 → 06-11, single shot): FAIL.**

| segment | N | total | exp/trade | win% | PF |
|---|---|---|---|---|---|
| OOS all | 546 | −$34 | −$0.06 | 49% | **0.999** |
| 01-01 → 02-28 (open) | 475 | −$420 | −$0.88 | 49% | 0.980 |
| 03-01 → 06-11 (sealed window) | 71 | +$386 | +$5.43 | 51% | 1.106 |
| ex-top-3-days | 516 | −$2,596 | — | — | 0.893 |

Sealed rule: FAIL if net PF < 1.00 → **0.999 = FAIL. Option B is CLOSED on this dataset; no re-sweeps.** (Boundary-adjacent to the MARGINAL band, recorded for honesty; the rule stands as sealed.)

## Post-mortem (why it died)

1. **Regime instability of the event definition:** 99 events/year on 2025 data became **546 events in 5.4 months** of 2026 — the war regime mass-produced qualifying impulse bars. The rolling-median baseline adapts too slowly to a persistent vol regime shift; the "anomalous impulse" concept degenerates into "any war-week bar cluster."
2. **The 2025 edge was null-adjacent to begin with:** selected PF 1.605 vs random-entry 95th pct 1.569. In a strongly trending year, impulse-follow at fixed horizons is barely distinguishable from being long the tape at random times with the same direction mix.
3. **Expectancy collapse, not cost death:** OOS gross expectancy was ~+$5.94/trade before the $6 cost — the aftermath drift signal shrank to exactly cost-sized. Consistent with the program's edge-headroom findings: the structure exists but the capturable residual at 60s/fixed-horizon granularity is cost-order.
4. The sealed-window segment (war regime) was mildly positive (PF 1.106) while the calmer Jan–Feb was negative — directionally consistent with the Option C finding (shock regimes are friendlier to vol-harvesting) but at N=71 vs the sealed full-window rule, it is a subgroup observation, not a result. Chasing it would be the S26-KZ-subgroup pattern; it would need a fresh prereg with prospective data and is NOT recommended given expectancy is cost-order even in the favorable segment.

## Program state after this scout

- **Option A (chase the post): rejected** (structural, innovation-strategy doc).
- **Option C (policy-shock throttle): dead** (H-C1 rejected, 2026-07-03).
- **Option B (impulse aftermath): dead** (Gate 0 passed, OOS failed, closed 2026-07-03).

The "Trump/news edge" inquiry is now fully resolved with three documented kills and zero deployments. Reusable assets banked: the frozen policy-shock calendar (`data/macro/policy_shock_calendar_2025_2026.csv` + windows), the impulse-event scanner (`tools/option_b_gate0_scout.py`), and the finding that MNQ 1-min impulse aftermath is follow-not-fade but cost-order at retail granularity.

Priorities return to: GAP-1 promotion review (direction-split check), S25 accrual, HG copper slippage decision (~Jul 2 data).
