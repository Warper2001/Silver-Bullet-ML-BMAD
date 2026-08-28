# Pre-Registration: GAP-V2 — Velocity-Conditioned Gap-Down Response, Prospective Decision Rule

**Registered:** 2026-08-28
**Status:** SEALED at commit time. Append-only amendments.
**Required by:** `preregistration_gap_velocity_conditioned.md` (GAP-V) **§A3.3**, which opened
a prospective accrual with **no committed decision rule** and stated that one must be fixed
before the ledger reaches target. This is that document.

---

## 1. Why GAP-V could not carry its own decision rule

GAP-V tested whether GAP-1's fade is velocity-dependent on gap-downs. It returned
INSUFFICIENT_SAMPLE (N_high = 3 against a floor of 12), and **§A2.2 diagnosed why**: the
split point (0.9421%, the median of 2025's qualifying gap-downs) sat at the **83rd
percentile** of the 2023/24 test era. 2024 Sep–Nov contained *no* qualifying gap-down above
it. An absolute percentage derived in one volatility regime lands in the tail of another.

That left GAP-V's prospective phase in a bind recorded honestly in §A3.3: carrying 0.9421%
forward repeats the defect, and re-deriving it after seeing outcomes is forbidden by §8.1.
**Neither move was legitimate, so the tracker was built to commit to no split at all** and
to report only a sample count.

## 2. The resolution: remove the threshold entirely

A threshold cannot import a regime if there is no threshold.

> **Primary test: Spearman rank correlation ρ between `gap_pct` and net P&L per trade,
> across every accrued GAP-1 LONG trade in the prospective window.**

This tests H1 directly — *does higher velocity predict a worse fade outcome* — using **all**
the data, with **no split point, no subgroup floor, and no free parameter**. It is invariant
to the gap distribution's scale and location, so a volatility regime shift cannot corrupt it
the way §A2.2 describes. It also disposes of the N≥12-per-subgroup question §A3.3 raised:
there are no subgroups.

**Hypothesis, unchanged from GAP-V §3:**

- **H1:** ρ < 0 — larger gap-downs are followed by worse outcomes for the fade.
- **H0:** ρ = 0.

**One-sided**, because the direction is pre-specified by the external mechanism
(arXiv:2605.04004 §7.2: *"gap-down days with high Kalman velocity tend to continue
downward"*). It may not be flipped after seeing results.

## 3. POWER STATEMENT — fixed before any outcome is seen

Per the rule established in `preregistration_intraday_momentum_mnq.md` §A1.5 ("compute the
minimum detectable effect size at the planned N BEFORE sealing"). Fisher-z, one-sided,
α = 0.025 per look, 80% power:

| N | smallest \|ρ\| detectable at 80% power |
|---|---|
| **24** | **0.545** (large) |
| 30 | 0.492 |
| 40 | 0.431 |
| **47** | **0.399** (moderate) |
| 60 | 0.355 |
| 100 | 0.277 |

Sample required for a given true effect, at GAP-1's observed 0.190 longs/calendar-day:

| true ρ | N | elapsed |
|---|---|---|
| 0.55 | 24 | ~4.1 months |
| 0.50 | 29 | ~5.0 months |
| **0.40** | **47** | **~8.1 months** |
| 0.30 | 85 | ~14.7 months |

**Declared in advance: N=24 — GAP-V's original target — can only detect a LARGE effect
(ρ ≥ 0.545).** A null at N=24 would mean "no large effect", not "no effect". That is exactly
the misreading MIM-X fell into, and it is why this seal does not stop at 24.

## 4. Two looks, alpha spent in advance

| look | trigger | α (one-sided) | can conclude |
|---|---|---|---|
| **Interim** | ledger reaches **N = 24** | 0.025 | **PASS only.** A non-significant result at this look is **INCONCLUSIVE**, never a stop. |
| **Final** | ledger reaches **N = 47** | 0.025 | PASS, H0, or INCONCLUSIVE per §5. |

Familywise false-positive rate across both looks: **4.94%**. No further looks are permitted;
peeking at the ledger between triggers is forbidden by §6.

The interim look is deliberately **asymmetric** — it can stop early for a strong effect but
cannot kill the hypothesis — because at N=24 a null carries almost no information (§3).

## 5. Decision rule

Let ρ̂ be the observed Spearman correlation and p its one-sided p-value.

| Condition | Verdict | Pre-committed action |
|---|---|---|
| p ≤ 0.025 **and** ρ̂ < 0 | **MECHANISM TRANSFERS** | Record. Triggers §8. **No live change, no GAP-1 modification.** |
| Final look, p > 0.025, ρ̂ < 0 | **DIRECTIONALLY CONSISTENT, UNPROVEN** | Record. Thread closed. No successor, no further accrual. |
| Final look, ρ̂ ≥ 0 | **H0 — MECHANISM DOES NOT TRANSFER** | Record as a null. GAP-1 unchanged. Thread **CLOSED**. |
| Stopping date reached with N < 47 | **INSUFFICIENT** | Record. No verdict. Do not lower the target to manufacture one. |

**Genuine do-nothing branch:** three of the four outcomes end with GAP-1 untouched and the
thread closed. Only the first triggers anything, and what it triggers is a document.

**Secondary, reported and NEVER decision-bearing:** the median-split subgroup means in
dollars (for interpretability only); win rates; per-quarter breakdown; and the same figures
at $2.00 vs $3.00 friction. Per this session's September-2025 lesson, if the result is
carried by one quarter that is disclosed in the verdict line.

## 6. What we will NOT do

1. No threshold, split point, or velocity band is introduced. §2 is the whole test.
2. No look at the ledger outside the two triggers in §4.
3. No lowering of N=47 if accrual is slow. §5 has an INSUFFICIENT branch for that.
4. No switch to two-sided, or sign flip, after seeing ρ̂.
5. No substitution of a different velocity proxy (ATR-scaled, Kalman, pre-market range).
   `gap_pct` is the measure, as in GAP-V §6.
6. No pooling of GAP-V's 18 retrospective trades or GAP-1's 24 pre-window live trades.
   The window opened 2026-08-27T22:05:19Z and nothing before it is admissible.
7. No inclusion of gap-**up** (SHORT) trades. Gap-downs only, as in GAP-V §8.5.
8. No change to GAP-1, its config, its bot, its clock, or its own N≥30 decision rule.

## 7. Stopping date and dependency risk

**Stopping date: 2027-12-31.** If N=47 is not reached by then, §5 returns INSUFFICIENT.

**Disclosed dependency:** this test consumes GAP-1's live long trades. GAP-1 carries its own
sealed rule (`preregistration_gap_fade_panic_open.md`): *"PF < 1.00 at N ≥ 30: STOP. Archive
strategy."* GAP-1 stood at N=24 total on 2026-08-27. **If GAP-1 is archived under its own
rule, GAP-V2's population stops growing and this seal returns INSUFFICIENT at the stopping
date.** That is an accepted outcome, pre-committed here; GAP-1's decision rule takes
precedence and must not be bent to keep this test alive.

## 8. Successor trigger

A **MECHANISM TRANSFERS** verdict authorises exactly one thing: drafting a new
pre-registration for a velocity-conditioned gap strategy, carrying its own power statement
and its own cost model.

It does **not** authorise: modifying GAP-1; suppressing GAP-1's high-velocity longs; trading
the continuation short; changing size; or promoting anything to a funded account.

**Economic gate, pre-committed before any number is seen** (per `preregistration_ofi_bar_level.md`
§2, where statistical significance at large N proved worthless): a significant ρ establishes
only that velocity *ranks* outcomes. Any successor must additionally show that acting on it
produces **≥ 1.53 bps of notional per trade (3 × the 0.51 bps friction benchmark)**. A
rank correlation is not an edge.

## 9. Values fixed at seal time

| Item | Value |
|---|---|
| git HEAD at seal | `14ada68` |
| Prospective window opened | 2026-08-27T22:05:19+00:00 (GAP-V Amendment 2, `1baf5cf`) |
| Ledger | `data/gap_velocity/prospective_trades.csv` |
| Ledger state at seal | **0 trades** |
| Tracker | `gap_velocity_prospective_tracker.py`, `gap-velocity-prospective.timer` (daily 06:30 UTC) |
| Interim look | N = 24 |
| Final look | N = 47 |
| α per look | 0.025 one-sided (familywise 4.94%) |
| Stopping date | 2027-12-31 |

## 10. Required tracker change

`gap_velocity_prospective_tracker.py` currently targets **N = 24** and prints a
refusal-to-analyse notice on reaching it. It must be updated to reflect §4: announce the
**interim** look at N=24 and the **final** look at N=47, and continue accruing past 24
rather than treating it as the end. This is a reporting change only — the ledger, the
natural key, and the observation-only discipline (no subgroup statistics; GAP-V §A3.2) are
unchanged.
