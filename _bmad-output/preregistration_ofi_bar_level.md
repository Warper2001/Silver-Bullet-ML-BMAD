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

---

# Amendment 1 — Result (2026-08-28)

Append-only. Original sealed text unedited.
Script: `study_ofi_bar_level.py` | Report: `data/reports/ofi_bar_level_20260828_153726.txt`
Run under seal `fedaf84`.

## A1.1 Verdict — FAILS

N = 2,028,735 bars with a forward 10-bar return, 23 contracts.

| leg | edge | required |
|---|---|---|
| LONG (top OFI decile mean) | **−0.0252 bps** | ≥ 1.53 bps |
| SHORT (−1 × bottom OFI decile mean) | **−0.0653 bps** | ≥ 1.53 bps |

**Both legs are negative.** Neither clears even 1× friction (0.51 bps), let alone 3×.
Verdict per §5: **FAILS**. Recorded; closed.

Decile means of forward 10-bar return (bps), decile 0 → 9:
`0.0653, 0.0497, 0.0746, 0.0325, 0.0667, 0.0380, 0.0343, 0.0387, 0.0055, −0.0252`

There is a **rough downward trend** across deciles — high buying pressure is followed by
slightly *lower* forward returns. So to the extent any signal exists it is **mildly
contrarian, not momentum**. It is non-monotone, and the whole range spans **0.10 bps**
against 0.51 bps of friction. Not actionable in either direction.

## A1.2 §2 called this exactly, and that is the transferable result

The seal predicted, before the run, that at N≈2M the test would be "~93× more statistically
sensitive than economically relevant" and "WILL return overwhelming t-statistics for effects
worth nothing."

Observed:

```
r_fwd(10) = a + b*OFI :  beta = -0.1045 bps per unit OFI   t = -2.8   R^2 = 0.000004
```

**t = −2.8 is conventionally "significant". R² = 0.000004 means OFI explains four parts per
million of forward-return variance.** A one-standard-deviation OFI move (0.2203) implies
**−0.0230 bps** — about one twenty-second of the friction cost.

Had this document gated on p-values, it would have concluded that signed order flow
"significantly predicts" MNQ returns. §6.7 forbade that in advance. **This is the cleanest
demonstration in the whole program of why an economic threshold must be pre-committed
separately from a statistical one.**

## A1.3 No horizon rescues it

All secondary horizons were computed and all fail (§6.2 forbids promoting any of them):

| h (bars) | long leg | short leg | spread |
|---|---|---|---|
| 1 | −0.0214 | −0.0286 | −0.0499 |
| 5 | −0.0286 | −0.0580 | −0.0866 |
| **10 (primary)** | **−0.0252** | **−0.0653** | **−0.0905** |
| 30 | +0.0026 | −0.0630 | −0.0605 |
| 60 | −0.0006 | −0.1439 | −0.1445 |

Every spread is negative and every leg is at least 7× below friction. The failure is not a
horizon-selection artifact.

## A1.4 The null is conservative — and correcting for it does not close the gap

§8 disclosed that imperfect trade classification attenuates any true effect toward zero.
Correcting for it explicitly, using the research's reported accuracy range:

| classification accuracy | attenuation factor (2p−1) | implied true effect | still short of friction by |
|---|---|---|---|
| 0.728 (worst study) | 0.456 | ≤ 0.1432 bps | **3.6×** |
| 0.844 (mid) | 0.688 | ≤ 0.0949 bps | **5.4×** |
| 0.926 (futures best) | 0.852 | ≤ 0.0766 bps | **6.7×** |

Even under the **most generous** attenuation correction, the effect remains **3.6× short of
1× friction** and ~11× short of the 3× bar. The gap is not a measurement problem.

## A1.5 What this closes, and what it does not

**Closes:** bar-aggregated signed order flow as a standalone directional signal on MNQ, at
every horizon tested, over 2020-12 → 2026-08. This was the last live candidate from the
research run, and — unlike MIM-X — the sample is large enough that the null is decisive
rather than uninformative.

**Does not close:** (a) the book-level OFI literature, which measures a different quantity
(§8) at sub-second resolution and is not tested here; (b) OFI as a *conditioning* variable
on some other signal, which is a different hypothesis needing its own seal; (c) any
instrument other than MNQ.

**Does not support:** the mildly contrarian sign. §6.3 forbids the single-direction rescue,
both legs failed, and a 0.10 bps decile range is inside the noise of anything tradeable.

## A1.6 Status

Closed. Nothing deployed, no live system touched, no parameter changed. The bar file
retains its `upvol`/`downvol` columns and remains available for any future hypothesis —
but this seal answers the direct-directional question in the negative.
