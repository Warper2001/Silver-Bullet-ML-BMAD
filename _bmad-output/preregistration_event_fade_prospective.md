# Pre-registration — MNQ Econ-Event Fade/Trend, Prospective

**Sealed:** 2026-09-02
**Status:** OBSERVATION ONLY. No capital. No live orders. Paper ledger only.
**Supersedes:** the "PARKED" disposition in `project_mnq_event_fade_scout_20260615`.

---

## Why this is being un-parked

The scout (2026-06-15) found real structure with excellent cost geometry and
parked it for one stated reason, quoted verbatim from the record:

> "Great headroom but a fatal validation timeline for a combine you want to pass
> this quarter."

That reason no longer exists. On 2026-09-02 the shop committed to **edge first,
combine later**, and separately established that a combine cannot validate an
edge at this trade frequency anyway (a combine window contains too few trades to
distinguish edge from noise, so it fails in both directions).

The candidate was never tested to failure. It was shelved against a deadline that
has since been withdrawn. Under the eligibility rule adopted the same night — *a
shelved strategy may be revived if and only if it was killed for a criterion since
explicitly and prospectively changed, and stays dead if it was killed for failing
validation* — this qualifies.

## The hypothesis (frozen)

From the scout's event-type split, which had economic logic but N=8–9 per cell:

- **H1 (FOMC):** the 14:00 ET knee-jerk overshoots and reverts. **FADE** the impulse.
- **H2 (NFP):** the 08:30 ET pre-market print sets direction. **FOLLOW** the impulse.
- **CPI:** no directional hypothesis. Logged for completeness, excluded from the
  decision rule.

## Frozen mechanics

| Parameter | Value |
|---|---|
| Instrument | MNQ (front month) |
| Pre-event reference | close of the event minute |
| Impulse window K | 3 minutes |
| Hold M | 30 minutes |
| Entry | at close of minute K, direction per hypothesis |
| Exit | at close of minute K+M, or session end, whichever first |
| Cost | $2.24 per round trip (1 contract) |
| Size | 1 contract, notional only — nothing is executed |

No parameter above may be changed without a new pre-registration. The scout
already demonstrated this family is parameter-unstable in aggregate (a K/M grid
swung +$39 to −$41), which is precisely why K and M are frozen here rather than
swept.

## What counts

**Only tier-1 events occurring on or after the seal date, 2026-09-02.** The
historical events that generated this hypothesis (13 NFP, 13 CPI, 9 FOMC through
2026-05-13) are the discovery sample and are permanently excluded from the
decision. Starting N is therefore **zero**, by construction.

## Decision rule (both directions, fixed in advance)

Evaluated **per event type independently**, at **N ≥ 30 events of that type**:

- **PASS** — mean net P&L per contract > $0 AND profit factor > 1.30
- **FAIL** — otherwise

At roughly 9 FOMC and 13 NFP tier-1 events per year, N=30 is about **3.3 years
for FOMC** and **2.3 years for NFP**. That is the honest cost, stated up front.
There is no interim readout that authorizes anything, and no partial-N result may
be used to promote, kill, or re-parameterize this candidate. A tracker that has
run for eight months and is "looking good" means nothing under this rule.

## Promotion criteria (what passing actually earns)

Adopted per the 2026-09-02 finding that this shop's process had kill gates and no
promotion gates, which turned a pipeline into a queue. Written now, before any
result exists:

- **On PASS:** the candidate is promoted to **SIM execution** automatically, with
  no further meeting — same instrument, same frozen mechanics, 1 contract.
- Promotion to **live capital** remains a separate, sealed decision requiring its
  own pre-registration. Passing this rule does not authorize risking money.
- **On FAIL:** the candidate is dead. It does not get a re-sweep, a sub-period, or
  a "what if we changed K" — those are the moves the shop's own graveyard is
  made of.

## Operational honesty requirement

The calendar file currently ends **2026-05-13** and therefore contains **no events
after the seal date**. The tracker MUST fail loudly when it has no forward
coverage rather than run clean and log nothing — a tracker that looks healthy
while measuring nothing is the exact failure mode this shop named on 2026-08-08
and hit again with BTC-CARRY on 2026-09-02.

Ledger: `data/event_fade/prospective_trades.csv`
Tracker: `tools/event_fade_tracker.py`
