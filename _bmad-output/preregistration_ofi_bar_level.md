# Pre-Registration: OFI-1 — Bar-Level Signed Order Flow on MNQ

**Registered:** 2026-08-28
**Status:** SEALED at commit time. Append-only amendments.

---

## 1. Why this test is the most dangerous one in this program

Every other test this week had its rule specified by someone else — GAP-1's seal, the JFE
paper, the falsification study. **This one has no external specification.** There is no
published strategy for bar-aggregated signed volume; the research run found that the entire
space is vendor marketing with no empirical base (`research/technical-hf-order-flow-
strategies-2026-08-27/research.md` §1).

That means unlimited tuning freedom: horizon, threshold, filter, direction, session. This
seal exists to remove that freedom **before** any number is seen.

## 2. THE INVERTED POWER PROBLEM — the governing design fact

MIM-X failed because it was **underpowered**: it could not detect the effect it tested.
This test has the opposite pathology and it must be stated plainly.

Measured before the run, from instrument properties only:

| Quantity | Value |
|---|---|
| Usable bars | **2,028,942** |
| Bars per OFI decile | ~202,894 |
| sd of 1-bar forward return | 3.723 bps |
| **Decile mean detectable at t=2.0** | **0.0165 bps** |
| **Friction benchmark** | **0.51 bps** |
| **Project's 3× economic bar** | **1.53 bps** |

**This test is ~93× more statistically sensitive than it is economically relevant.** At
N=2M it will return overwhelming t-statistics for effects worth nothing. A p-value here is
not evidence of anything tradeable.

**Therefore the decision rule in §5 is ECONOMIC, not statistical.** Statistical significance
is reported but is explicitly and permanently NOT decision-bearing in this document.

## 3. Data and signal

Source: `data/mim_x/mnq_1min_by_contract.csv` — 2,028,965 MNQ 1-min bars, 23 quarterly
contracts, 2020-12-17 → 2026-08-28, fetched 2026-08-28 for MIM-X2.

Verified before sealing: `upvol + downvol == volume` on **100.000%** of bars — the venue's
signing is complete, with no unclassified volume.

Signal, one definition, no free parameter:

> **OFI_t = (UpVolume_t − DownVolume_t) / TotalVolume_t**, in [−1, +1].

Measured distribution (instrument property): mean −0.0007, sd 0.2203, p1 −0.557, p99 +0.556.

Forward returns are computed **within a single contract** (`groupby(contract)`), so no
return crosses a roll boundary.

## 4. Primary horizon — derived, not chosen for convenience

The research run established (F1.1, arXiv:2505.17388 on CSI 300 index futures) that OFI
prediction shows "negligible differences across various historical windows when the
prediction horizon ranges from 0.5 seconds to 10 minutes, while significant variations
emerge when the prediction window exceeds 1500 ticks (12 minutes)."

> **Primary horizon h = 10 bars (10 minutes)** — the outer edge of the literature-supported
> plateau. It is taken from the external finding, before any outcome here was computed.

Horizons h ∈ {1, 5, 30, 60} are computed and reported as **secondary**. Acting on any of
them requires a **new seal** (§6.2). They exist to characterise decay, not to be shopped.

## 5. Primary metric and ECONOMIC decision rule

Bars are sorted into deciles by OFI. Two legs:

- **Long leg edge** = mean forward h-bar return of the **top** OFI decile, in bps
- **Short leg edge** = **−1 ×** mean forward h-bar return of the **bottom** OFI decile, in bps

| Condition | Verdict | Pre-committed action |
|---|---|---|
| **Both legs ≥ 1.53 bps** (3× friction) | **PASS** | Record. Triggers §7. No deployment. |
| Both legs ≥ 0.51 bps but either < 1.53 | **MARGINAL** | Record. No deployment, no successor seal — a 1× edge is not a business. |
| Either leg < 0.51 bps | **FAILS** | Bar-level signed OFI does not carry tradeable directional content at h=10 on MNQ. Record; close. |

Directional asymmetry is permitted in reporting but **not** in the verdict: if one leg passes
and the other fails, the verdict is **FAILS**. Rescuing a strategy by dropping its losing
direction is the documented failure pattern in this project's memory, and §6.3 forbids it.

**Reported, never decision-bearing:** the regression `r_fwd = a + b·OFI` (β, t, R²); decile
monotonicity; per-year figures; RTH-only figures; and all secondary horizons.

## 6. What we will NOT do

1. No change to the OFI definition (e.g. volume-normalised vs tick-normalised) after seeing results.
2. No promotion of a secondary horizon to primary. h=10 is fixed by §4.
3. **No single-direction rescue.** Both legs must clear, or the verdict is FAILS.
4. No session filter (RTH-only) introduced to rescue a failing result — all bars are primary.
5. No threshold tuning (top 5%, top 1%, |OFI| > x) after seeing decile results.
6. No conditioning on volatility, time-of-day, or regime added post hoc.
7. No citation of a t-statistic as evidence of a tradeable effect (§2).

## 7. Successor trigger

A **PASS** authorises drafting a successor pre-registration for a tradeable strategy, which
must itself carry a holding-period cost model and its own power statement. It does not
authorise deployment, sizing, or any change to a live system.

## 8. Disclosed limitations

- **`UpVolume/DownVolume` is uptick/downtick classified volume, not book-level OFI.** The
  academic literature the horizon was derived from measures order-book depth changes. These
  are related but distinct quantities; a null here does not refute the book-level literature.
- Classification accuracy in futures is reported at ~72.8–92.6% depending on study (research
  §1). Imperfect signing attenuates any true effect toward zero, so a null is
  conservative and a PASS is not inflated by it.
- MNQ only, 2020-12 → 2026-08. No cross-instrument or earlier-era claim is made.
- Decile sorting uses the full-sample OFI distribution, which is mildly forward-looking in
  the strict sense. Disclosed rather than corrected, because the OFI distribution is a stable
  instrument property (sd 0.2203) and re-deriving it rolling would introduce a window choice
  — a free parameter this seal is trying to avoid. Any PASS must re-test with rolling deciles.

## 9. Values fixed at seal time

| Item | Value |
|---|---|
| git HEAD at seal | `af031bf` |
| Bar file | `data/mim_x/mnq_1min_by_contract.csv`, 2,028,965 bars |
| Usable bars / decile size | 2,028,942 / ~202,894 |
| Primary horizon | h = 10 bars |
| Economic bar | 1.53 bps per leg (3 × 0.51) |
