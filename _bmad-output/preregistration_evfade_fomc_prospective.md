# Pre-Registration: EVFADE-FOMC — Prospective Accrual of the FOMC Fade

**Date sealed:** 2026-09-07, BEFORE any forward FOMC event is observed. The first eligible event
is **2026-09-16**.
**Author:** Alex (session run at Alex's instruction: "set up the prospective accruals").
**Lineage:** `project_mnq_event_fade_scout_20260615` — which parked this candidate and named this
exact remedy: *"the disciplined path is pre-register the FOMC-fade direction and collect forward."*
**Sibling pattern:** `prereg_gc_cpi_prospective.py` (tamper-evident event prereg) and
`gap_velocity_prospective_tracker.py` (observation-only accrual). This seal follows both.

---

## 1. The honest framing — read before anything else

This is a **post-hoc subgroup selected from N=8**. The 2026-06-15 scout tested an aggregate MNQ
event fade (thin, PF 1.12, and **parameter-unstable** — a K/M sensitivity grid swung +$39 to −$41,
which is noise, not edge) and then observed an event-type split:

| event type | N | scout result |
|---|---|---|
| **FOMC** | ~9 | **fades strongly**: +$59.63/ct, PF 4.66 |
| NFP | ~8 | trends (fade −$48.49; momentum +$44.01) |
| CPI | ~8 | wash |

**The FOMC-fade direction was chosen AFTER seeing that split.** The in-sample PF 4.66 is therefore
*the observation that motivates this test*, not evidence for it, and it cannot be cited as
validation. **The prospective accrual is the primary and only validity check.** A weak or negative
forward result means FOMC was a noise artefact of nine observations.

## 2. Hypothesis

**H-EVFADE1.** Following the FOMC statement release (14:00 ET), MNQ's initial impulse over the
first K minutes **reverts**, such that a position taken against that impulse and held M minutes is
profitable net of cost.

## 3. Frozen specification — nothing here may change

Taken verbatim from the scout's primary cell (`K3/M30`); **not re-tuned**, because the scout
already showed the K/M surface is unstable and re-picking a cell now would be fitting to the very
instability that discredited the aggregate.

| parameter | value |
|---|---|
| instrument | MNQ, 1 contract |
| event | FOMC statement, 14:00 ET (tier-1 only) |
| reference price | close of the 14:00 ET 1-minute bar |
| impulse window **K** | **3 minutes** |
| direction | **fade** — short if impulse up, long if impulse down |
| entry | close of the 14:03 ET bar |
| hold **M** | **30 minutes** → exit at close of 14:33 ET bar |
| cost | **$2.24 per round turn** (scout's cost basis) |
| eligibility | the day's MNQ 1-min series must cover 14:00–14:33 ET with no gap > 2 min |

## 4. Sample, timeline, and the stopping rule

**FOMC meets 8× per year.** Forward schedule is fixed and committed at
`data/macro/fomc_calendar_forward.csv` (source: federalreserve.gov), 11 events through 2027-12-08.

- **N target: 30.** At 8 events/year this is reached in **≈ 3.75 years — approximately Q2 2030.**
- **Stopping date: 2030-12-31.** If N < 30 by then the study closes `INSUFFICIENT_SAMPLE`.
- **One interim look at N = 15**, which may only PASS. A null at the interim is INCONCLUSIVE and
  the accrual continues. (Borrowed from GAP-V2's two-look design so the interim cannot be used to
  stop for futility and then be re-argued.)

**This timeline is stated up front because it is the single most important fact about this study.**
`project_mnq_event_fade_scout_20260615` closed with exactly this verdict — *"great geometry but a
fatal validation timeline"* — and the timeline has not improved. **Accrual is not progress.**
Nobody may treat a partial ledger as evidence of anything.

## 5. Sealed decision rule (evaluated only at N=15 interim or N≥30 final)

At the look, on the accrued prospective events **only** (no scout data pooled in):

- **PASS:** mean net P&L per event > $0 **AND** one-sample t-test p < 0.05 (two-sided) **AND**
  net PF > 1.10. → authorizes drafting a deployment pre-registration; nothing trades from this.
- **FAIL:** mean net P&L per event ≤ $0 at N ≥ 30 → EVFADE-FOMC closed; the scout's split is
  recorded as a nine-observation artefact.
- **INCONCLUSIVE:** anything else at N ≥ 30 → PARK; no deployment path; no re-slicing.
- **INSUFFICIENT_SAMPLE:** N < 30 at the stopping date → closed with no verdict on the hypothesis.

**No subgroup may rescue a failing result** (no "only when VIX high", no "only cut days", no K/M
re-sweep). **No parameter may be re-tuned mid-accrual.** One shot at each look.

## 6. Power — stated, not assumed

**This study is NOT powered to detect a modest effect, and that is disclosed now rather than
discovered at the look.** With MNQ day-session σ ≈ $400–530 per contract, N=30 can only detect a
per-event effect of roughly **$200+**. The scout's observed +$59.63 is **far below** that.

So the arithmetic is: **if the true FOMC fade effect is the size the scout measured, N=30 will not
confirm it.** This accrual can only return PASS if the true effect is very large — which, given
the scout's estimate came from nine observations and is almost certainly winner's-cursed upward,
is unlikely.

**Why seal and run it anyway:** it costs one systemd timer, it places no orders, it is the only
instrument on the board generating genuinely *new* evidence rather than re-examining old, and a
sealed null in 2030 is still a real answer. **But it is explicitly NOT a candidate edge, and it
must not be counted as one in any portfolio or research-status summary.** If a decision is needed
sooner than 2030, this study cannot supply it.

## 7. Observation-only guarantee

The tracker (`evfade_fomc_prospective_tracker.py`) is a **pure observation instrument**, following
`gap_velocity_prospective_tracker.py`'s stated principle: it accrues raw per-event facts and
reports **only the sample count and progress to target** — **no running mean, no PF, no verdict** —
because watching those accrue is how a decision rule gets chosen to fit the data it will be tested
on. It reads market data read-only, **places no orders**, and touches no live trading state.
Idempotent on event date.

## 8. Disclosures

1. Post-hoc subgroup from N≈9 (§1). The prospective result is the only validity check.
2. K=3/M=30 frozen from the scout's primary cell, deliberately not re-optimised.
3. Cost $2.24/RT is the scout's assumed basis, **not** a fresh measurement. Before any deployment
   it must be re-measured prospectively (HG/PL method).
4. Forward FOMC dates are the Fed's *tentative* published schedule; the tracker must tolerate a
   rescheduled or unscheduled (emergency) meeting, and an unscheduled meeting is **not** eligible
   (the hypothesis is about the scheduled 14:00 statement).
5. No forward FOMC event has been observed at seal time; the first is 2026-09-16.

## 9. Out of scope

The NFP-momentum and CPI cells (each would need its own seal); the aggregate event fade (already
found unstable); K/M sweeps; any deployment; sizing beyond 1 contract; and the **MON-1 Monday
seasonality accrual**, closed the same day as timeline-infeasible — see
`verdict_mon1_timeline_infeasible_20260907.md`.
