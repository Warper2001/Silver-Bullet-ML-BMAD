# Pre-Registration: GAP-V — Velocity-Conditioned Gap-Down Response on MNQ

**Registered:** 2026-08-27
**Authored by:** Alex (warper2001@gmail.com), drafted in BMAD party session
**Status:** SEALED at commit time. Append-only amendments; original text never edited.

---

## 1. Transparency Disclosure — read this first

This hypothesis has a **post-hoc origin and the origin is contaminated**. Stated in full:

1. On 2026-08-27 an external paper (§2) was read which reports that MNQ **gap-down days
   with high velocity tend to continue downward** rather than revert.
2. Immediately after reading it, the 12 live GAP-1 **long** trades (the leg that fades
   gap-downs) were split at their **median gap_pct**, *after their outcomes were known*,
   producing: small gap-downs N=6 net **+$766.50** (WR 67%), large gap-downs N=6 net
   **−$207.50** (WR 17%).
3. That split is **not evidence**. It is a post-hoc subgroup split of 12 observations at a
   threshold chosen after seeing the answer — the exact failure pattern recorded in this
   project's memory under "never restrict-to-favorable-subset and call it validation."

**Consequence, pre-committed:** the 24 live GAP-1 trades (12 long) are **excluded from
every test in this document**. They generated the hypothesis; they cannot also test it.
They may be cited only as narrative motivation, never as a result.

This disclosure follows the precedent of `hcvwap_v3_longonly` (2026-06-09), which declared
its long-side split as in-sample-observed before running its OOS test.

---

## 2. External Source

Mesfin, M. (2026). *Structural Limits of OHLCV-Based Intraday Signals in MNQ Futures:
A Systematic Falsification Study.* arXiv:2605.04004.

Two results from that paper bear on GAP-1:

| Strategy | Entry | N | Mean Net (pts) | T | WR | Verdict |
|---|---|---|---|---|---|---|
| Gap Fill Fade | 09:30 | 238–245/yr | −1.92 | −0.44 | 48.1% | FAIL |
| Gap Fill Fade | 09:45 | 238–245/yr | −1.31 | −0.32 | 47.2% | FAIL |
| Gap Fill Fade | 10:00 | 238–245/yr | −2.24 | −0.59 | 47.9% | FAIL |
| Gap Continuation Short | 09:30 (Kalman v>2.5) | 22 | **+14.52** | **+3.23** | 68.2% | FAIL — N<30 |

Stated mechanism (§7.2, verbatim): *"gap-down days with high Kalman velocity tend to
continue downward."*

**Honest reading of that table, pre-committed before we run anything:**
- Gap Fill Fade is GAP-1's thesis, tested unconditionally at ~240 events/yr over 2021–2025,
  and is **negative at all three entry times**.
- But its T-statistics (−0.32 to −0.59) are **nulls, not kills** — the author calls them
  "indistinguishable from noise." This does not falsify GAP-1.
- His friction is 2.0 pts ($4.00); GAP-1 realistically pays ~$3.00. His best gap-fill cell
  (−1.31 pts) is approximately breakeven at our cost.
- He faded **every** gap; GAP-1 requires **≥0.5%**. Different hypotheses.
- His bars are 5-minute; GAP-1 is 1-minute.
- His Kalman velocity filter is **not reproducible from the paper** — no specification is
  given. We therefore do **not** copy his threshold (see §5).

---

## 3. Hypothesis

> **H1.** Among MNQ RTH sessions opening ≥0.5% **below** the prior RTH close, the GAP-1
> fade response (LONG at the open) has **lower mean net P&L per trade on high-velocity
> gap-downs than on low-velocity gap-downs**, where velocity is proxied by gap magnitude.

**H0:** No difference in mean net P&L between the two velocity subgroups.

One-sided. The direction is pre-specified by the external mechanism in §2 and may not be
flipped after seeing results.

**This tests a *conditioning variable*, not a new strategy.** A PASS says GAP-1's fade
thesis is velocity-dependent. It does **not** say the continuation trade is profitable —
that is a separate question this document does not ask (§9).

---

## 4. Relationship to GAP-1 — this is NOT an amendment

`preregistration_gap_fade_panic_open.md` (sealed 2026-06-25) states
*"No parameter amendments after this commit"* and, under **What We Will NOT Do**:

> 2. No gap threshold tuning after seeing Gate 0 results. 0.5% is locked.
> 5. No data subsetting ("strategy works better in trending/volatile months").
> 6. No direction asymmetry split ("only take longs" or "only take shorts").

GAP-1's seal **anticipated and forbade exactly this class of move.** Accordingly:

- **GAP-1 is not modified by this document.** Its parameters, its live bot, its N≥30
  decision rule and its clock all continue **unchanged**.
- `trader-gap-fade` is **not restarted, reconfigured, or paused** by this registration.
- This is a **separate, new hypothesis** carrying its own name (GAP-V) and its own clock.
- No result here may be applied to GAP-1 without a **new** strategy pre-registration.

---

## 5. Derivation of the velocity threshold — derived, not asserted

Per this project's standing rule (`feedback_derive_dont_assert_one_knob`), no hand-set
threshold. One knob, set by a stated rule, on data that is **already spent** so the test
window stays clean:

> **VELOCITY_SPLIT** := the **median `gap_pct`** among all qualifying gap-**down** sessions
> (gap ≤ −0.5%) in `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv`,
> computed under GAP-1's frozen qualification rules (§6).

2025 is already spent — GAP-1's own disclosure states the 2025 dataset "has been seen."
Using it to fix a split point adds no new contamination, and keeps the 2023/2024 test
window untouched.

The threshold is computed **once**, recorded in §11 before any outcome is examined, and
never adjusted. Kalman velocity is **not** used: it is unreproducible from the source (§2).

---

## 6. Test window and frozen parameters

**Test data — never used for any gap analysis:**

| File | Coverage | SHA-256 (first 16) |
|---|---|---|
| `mnq_1min_2023_sepnov.csv` | 2023-09-01 → 2023-12-01 | `cb9a346af16ad443` |
| `mnq_1min_2024_sepnov.csv` | 2024-09-01 → 2024-11-29 | `2bc03d33ff856c6a` |

**Contamination statement, stated honestly rather than claimed pristine:** these files
exist in the repo and have been read by *other* strategies —
`study_mim_noise_bands_gate0.py`, `study_mim_classic_gate0.py`,
`backtest_octnov_seasonal.py`, `download_mnq_octnov.py` (verified by grep, 2026-08-27).
**No gap-direction analysis has ever used them**, and no gap outcome in these windows has
been examined by the author or in this session. They are clean *for this hypothesis*; they
are not virgin data in general.

**Frozen — inherited verbatim from GAP-1 §Frozen Parameters, no re-tuning:**

| Parameter | Frozen Value |
|---|---|
| Qualifying gap | RTH open ≤ −0.5% vs prior RTH close (gap-DOWN only) |
| Response tested | LONG at RTH open (GAP-1's fade) |
| Entry price | Open of the 09:30 ET 1-min bar |
| Target | Prior RTH close |
| Stop | 2.0 × gap_abs beyond RTH open |
| Time stop | 13:00 ET bar open |
| Friday exclusion | YES |
| Max trades/day | 1 |
| Prior RTH close | Close of the 15:59 ET 1-min bar |
| Min RTH bars | 300 in prior session |
| Contract size | 1 MNQ ($2/point) |
| **Friction** | **$3.00 round-trip** (commission + ~1 tick; GAP-1 enters at market) |
| Velocity proxy | `gap_pct` magnitude |
| Split point | VELOCITY_SPLIT per §5 |

---

## 7. Primary metric and decision rule

**Primary metric:** Δ = mean net P&L/trade (LOW-velocity subgroup) − mean net P&L/trade
(HIGH-velocity subgroup), across the pooled 2023 + 2024 test window.

**Test:** one-sided Welch's t-test on the subgroup means, α = 0.05. Reported alongside
N per subgroup, both means, both PFs, and a bootstrap 90% CI on Δ.

| Condition | Verdict | Pre-committed action |
|---|---|---|
| N_low < 12 **or** N_high < 12 | **INSUFFICIENT_SAMPLE** | Record. No verdict. Do **not** widen the window, lower the gap threshold, or pool in live trades. |
| Δ ≤ 0 | **H0 — mechanism does not transfer** | Record as a null. GAP-1 unchanged. Thread **CLOSED**. No re-slicing by another velocity proxy. |
| Δ > 0, p > 0.05 | **DIRECTIONALLY CONSISTENT, UNPROVEN** | Record. GAP-1 unchanged. Eligible for a prospective successor seal (§9); **no live change**. |
| Δ > 0, p ≤ 0.05 | **MECHANISM TRANSFERS** | Record. GAP-1 **still unchanged**. Triggers §9. |

**Secondary (reported, never decision-bearing):** per-year breakdown 2023 vs 2024. If the
result is carried entirely by one year, that is disclosed in the verdict line — the
September-2025 lesson from this session's friction re-screen.

---

## 8. What we will NOT do

1. No adjustment of VELOCITY_SPLIT after seeing any outcome.
2. No substitution of a different velocity proxy (ATR-scaled, Kalman, pre-market range)
   if `gap_pct` fails. One knob, one test.
3. No pooling of the 24 live GAP-1 trades into any arm.
4. No widening of the test window if N is short — INSUFFICIENT_SAMPLE is a real outcome.
5. No test of the gap-**up** leg. This registration is gap-downs only.
6. No change to GAP-1, its config, its bot, or its clock, under any outcome.
7. No lowering of the 0.5% gap threshold to manufacture sample.
8. No re-running with a different friction assumption to move a borderline result.

---

## 9. Successor trigger — what a PASS does and does not authorize

A **MECHANISM TRANSFERS** verdict authorizes exactly one thing: drafting a **new**
pre-registration for a velocity-conditioned gap strategy, to be tested **prospectively**.

It does **not** authorize: modifying GAP-1; suppressing GAP-1's high-velocity longs;
trading the continuation short; changing position size; or promoting anything to a funded
account.

Explicitly recorded so a passing result cannot sit in a drawer **or** be over-read: this
document tests whether a *conditioning variable* has signal. Whether the *opposite* trade
is profitable is a different hypothesis requiring its own seal, its own data and its own
friction accounting.

---

## 10. Cost and sample-size disclosure

At GAP-1's observed 2025 IS rate (N=117 over ~250 trading days ≈ 0.47 qualifying
trades/day, roughly half of them gap-downs), the ~126 trading days in the pooled test
window are expected to yield **on the order of 30 gap-down sessions total**, split across
two subgroups. **This test is underpowered by construction** and may well return
INSUFFICIENT_SAMPLE. That is disclosed *before* running, and is a pre-committed outcome
rather than a reason to widen anything.

Purely prospective testing was considered and rejected as the primary route: GAP-1 fires
~0.19 longs/day live, so N=30 gap-down trades would take **~7–8 months**.

---

## 11. Values fixed at seal time

| Item | Value |
|---|---|
| git HEAD at seal | `bfbc4c304d19e34967b911017b14f04fa4b371bf` |
| `gap_fade_live.py` SHA-256 (first 16) | `d5715ba2285827d6` |
| `backtest_gap_fade.py` SHA-256 (first 16) | `404240bea5f0a877` |
| `mnq_1min_2025.csv` SHA-256 (first 16) | `3f20ec70885cdee6` |
| GAP-1 live trades at seal | N=24 (12 long / 12 short), 2026-06-25 → 2026-08-27 |
| VELOCITY_SPLIT | **0.9421%** — set by Amendment 1, 2026-08-27 |

VELOCITY_SPLIT is computed from 2025 per §5 and appended as **Amendment 1** *before* any
test-window run. The test script must refuse to execute while the placeholder is present.

---

## 12. Analysis script

To be written as `study_gap_velocity_conditioned.py`. It must:
- refuse to run if §11 VELOCITY_SPLIT is unset;
- read only the two test files in §6 for outcomes;
- read `mnq_1min_2025.csv` only to compute VELOCITY_SPLIT;
- never read `data/trades.db`;
- emit N, mean, PF and Δ per subgroup plus the per-year split, to
  `data/reports/gap_velocity_conditioned_<timestamp>.txt`.

---

# Amendment 1 — VELOCITY_SPLIT fixed (2026-08-27)

Appended **before** any test-window run, per §11. Original text above is unedited.

## A1.1 Derived value

Computed per §5 on `mnq_1min_2025.csv` (SHA-256 `3f20ec70885cdee6`) under GAP-1's
frozen qualification rules, loaded directly from `backtest_gap_fade.py`
(SHA-256 `404240bea5f0a877`, matching §11) so the rules cannot drift from the engine.

| Item | Value |
|---|---|
| 2025 sessions with a valid prior RTH close | 207 |
| Qualifying gap-DOWN sessions (≥0.5%, non-Friday) | **34** |
| min / p25 / max gap_pct | 0.5130% / 0.6495% / 4.3689% |
| **VELOCITY_SPLIT (median)** | **0.9421%** |

Assignment: `gap_pct < 0.9421%` → **LOW** velocity; `gap_pct ≥ 0.9421%` → **HIGH**.
This value is now frozen and may not be adjusted (§8.1).

## A1.2 Correction to the §10 sample-size estimate — disclosed before running

§10 estimated "on the order of 30 gap-down sessions" in the pooled test window. That
estimate was **too optimistic** and is corrected here, before any outcome is seen.

The 2025 rate is 34 qualifying gap-downs over ~250 trading days = **0.136/day**. The
pooled 2023+2024 test window is ~126 trading days, implying **≈17 gap-down trades**,
or roughly **8–9 per subgroup**.

That is **below the N≥12 per-subgroup floor in §7.** On the derivation-era rate alone,
this test is expected to return **INSUFFICIENT_SAMPLE** before it is run.

Recorded rather than acted upon. Per §8.4 and §8.7 the window will **not** be widened and
the 0.5% threshold will **not** be lowered to manufacture sample. The test is run as
sealed, and INSUFFICIENT_SAMPLE — if returned — is the honest result, not a failure to
be engineered around.

## A1.3 Incidental defect found while loading GAP-1's logic

`backtest_gap_fade.py:44` resolves its repo root as
`Path(__file__).resolve().parents[3]`, which raises `IndexError` when the file sits at
the repository root. It only resolves inside `.claude/worktrees/<name>/`. **GAP-1's own
sealed backtest script cannot currently be run from the repo root.**

Not fixed here — this seal ships no code, and GAP-1 is not modified (§4). Recorded as a
defect for separate action. The study script patches that single line in memory and
asserts the remainder of the source is byte-identical before executing it.

---

# Amendment 2 — Result (2026-08-27)

Append-only. Original sealed text and Amendment 1 unedited.

Script: `study_gap_velocity_conditioned.py`
Report: `data/reports/gap_velocity_conditioned_20260827_220408.txt`
Run under seal e5b40f6 + Amendment 1 5f08c5d. VELOCITY_SPLIT = 0.9421%,
re-derived by the script and confirmed OK against Amendment 1.

## A2.1 Verdict — INSUFFICIENT_SAMPLE

| subgroup | N | net | mean/trade | PF | WR |
|---|---|---|---|---|---|
| LOW (gap < 0.9421%) | 15 | −$399.50 | −$26.63 | 0.785 | 53.3% |
| HIGH (gap ≥ 0.9421%) | **3** | +$279.00 | +$93.00 | 5.612 | 66.7% |

**N_high = 3 against a §7 floor of 12. Verdict: INSUFFICIENT_SAMPLE.**

Per §7 the pre-committed action is: **record, no verdict.** The window is not widened,
the 0.5% threshold is not lowered, and the 24 live GAP-1 trades are not pooled in
(§8.3, §8.4, §8.7). No claim about H1 is made or implied.

The descriptive difference (Δ = −$119.63/trade, i.e. HIGH outperforming LOW, the
*opposite* sign to H1) is recorded for completeness and **is not a result**: it rests on
three observations. It must not be cited, and in particular must not be read as evidence
against the external mechanism.

Total qualifying trades: **18**. Amendment 1 predicted ≈17 before the run.

## A2.2 Why it is underpowered — a design defect in §5, not just a short window

Amendment 1 attributed the expected shortfall to calendar length. That was **incomplete**.
The deeper cause is a distributional shift between the derivation era and the test era:

| era | N | median gap | mean gap | ≥ 0.9421% | share |
|---|---|---|---|---|---|
| 2025 (derivation) | 34 | 0.942% | 1.292% | 17 | 50.0% |
| 2023 Sep–Nov (test) | 8 | 0.752% | 0.886% | 3 | 37.5% |
| 2024 Sep–Nov (test) | 10 | 0.562% | 0.587% | **0** | **0.0%** |
| **pooled test** | **18** | **0.663%** | **0.720%** | **3** | **16.7%** |

**The 2025-derived median sits at the 83rd percentile of the test era.** 2024 Sep–Nov
contains *no* qualifying gap-down above it at all.

§5 chose to derive the split on already-spent data to keep the test window clean. That
reasoning was sound on contamination and **wrong on stationarity**: it assumed the gap
magnitude distribution is stable across eras. It is not. 2025 contained a tariff-driven
volatility regime (including a 6% single-day Nasdaq drop in April); Sep–Nov 2023 and 2024
were comparatively calm. A median-split derived in a violent year cannot partition a calm
one near its middle — by construction it lands in the tail.

This is a **generalizable methodological finding for this project**: a threshold derived
on era A and applied to era B inherits era A's volatility regime. Any future seal using
this pattern should either derive the split *within* the test era (accepting the
contamination and disclosing it), or use a **distribution-relative** rule — e.g. "the
median of the test era's own qualifying gaps" — rather than an absolute percentage.

Recorded, not acted upon. Re-deriving the split now, having seen these outcomes, is
exactly what §8.1 and §8.2 forbid.

## A2.3 Status

GAP-V is **not** closed and **not** advanced — it is a recorded failure of statistical
power. GAP-1 remains unmodified: no parameter, config, service or clock was touched, and
`data/trades.db` was never read by the study.

Open routes, none selected here:
1. **Prospective accrual** — ~7–8 months to N=30 gap-down longs at GAP-1's live rate.
2. **Acquire unseen data with comparable volatility** — the source paper notes Databento
   MNQ history at ~$42/quarter; pre-2021 data would also supply the volatility range
   2023/24 Sep–Nov lacks.
3. **Leave as recorded** — a null of power, not of effect, closing nothing.
