# Pre-Registration: S26-EXIT — ATR Floor, Paired Exit Experiment

**Registered:** 2026-08-27
**Authored by:** Alex (warper2001@gmail.com), drafted in BMAD party session
**Status:** SEALED at commit time. Append-only amendments; original text never edited.

---

## 1. Why this experiment exists

Diagnosis (2026-08-27, this session): s26's August collapse to PF 1.003 is **not** entry
decay. Win rate is stable (47.8% → 43.9% → 43.5%), `ml_proba` is stable (.659/.688/.674),
position size is constant. What collapsed is the payoff ratio (1.54 → 2.35 → **1.30**).

Mechanism: `s26_soft_fvg_streaming.py:32` sets `self.length = 20`. The stop is
`2 × ATR` over **twenty one-minute bars**. A 20-minute volatility estimate compresses in
calm and is then travelled straight through on expansion. August evidence: SL
mean/median ratio **1.95** (vs 1.25 and 1.31 in June/July); worst single loss **496.6
points against a 48.1-point median stop**.

The same defect class appears in two sibling bots (gap-fade time-stop: 6/6 `fill` exits
positive vs 18 `time` exits at −$2,977; MIM-NB guard exits = 100% of its losses). s26 is
chosen as the test bed because it has **165 paired entries** — the only sample in the shop
large enough — and because it is paper, so being wrong costs nothing.

---

## 2. Design — paired, nothing dropped

The naive test ("the winners had a property") is a post-hoc favourable-subset split, the
failure pattern in this project's memory. This design avoids it structurally:

- **All 165 live entries are frozen inputs** — timestamp, direction, entry price. Nothing
  is filtered, dropped or re-selected. Not the good months, not August 20–21.
- Both arms replay the **same** 165 entries through the **same** engine.
- **One knob** differs: the ATR used to place SL and TP.
- Comparison is **per-trade against itself** — trade *i* under A vs trade *i* under B.

There is no subset to cherry-pick because there is no subset.

**Cost-blindness cancels.** s26's ledger records raw price deltas with no fees
(`s26_soft_fvg_streaming.py:173`), so the *absolute* P&L of either arm is fiction. But
both arms take the identical 165 trades and therefore carry identical friction, so the
**difference** is clean. This is the only reason the experiment is meaningful on a
cost-blind ledger, and it is also why no arm's absolute PF may be quoted as a result.

---

## 3. Arms

| | Arm A (live) | Arm B (variant) |
|---|---|---|
| ATR used for SL/TP | `ATR20` | **`max(ATR20, ATR60)`** |
| SL | entry − dir × ATR × 2.0 | same formula |
| TP | entry + dir × ATR × 4.0 | same formula |
| Max hold | 60 bars | 60 bars |
| Entries | the same frozen 165 | the same frozen 165 |

---

## 4. Derivation of the floor — derived, not asserted

Per `feedback_derive_dont_assert_one_knob`, and per the lesson recorded in
`preregistration_gap_velocity_conditioned.md` §A2.2 **earlier today**: an *absolute*
threshold derived in one volatility regime lands in the tail of another. BTC moved
62,326 → 67,140 (max 80,088) across this window, so any fixed price floor would be wrong
by construction.

The floor is therefore **self-scaling and carries no new constant**:

> **ATR floor := `ATR60`** — the same true-range average over **60 bars**, where 60 is
> `self.max_hold`, a parameter **already frozen in the live strategy**.

Rationale, stated a priori: a stop should not be set from a volatility estimate covering a
*shorter* horizon than the position's own exposure. The strategy already declares that
exposure — 60 bars. The floor horizon is read off the strategy's own frozen config rather
than chosen by us. **No value is swept, tuned, or selected.**

---

## 5. Preconditions — all verified BEFORE this seal, results recorded here

The experiment was gated on the replay being faithful. Kraken bars for
2026-05-25 → 2026-08-28 were fetched (136,692 bars,
`data/s26_replay/kraken_1m_2026-05-25_2026-08-28.csv`) and tested:

| Precondition | Result |
|---|---|
| **Venue stability** — do fetched bars still match what the bot traded? | **165/165 entry prices reproduced EXACTLY** (offset +0 open ≡ offset −1 close). Kraken does **not** revise its 1-minute history. |
| **ATR reproducibility** — can the bot's own ATR be recomputed? | **141/141** TP/SL trades reproduce the live-implied ATR to **0.0000%** relative error. |
| **Arm A regression** — does the engine reproduce live outcomes? | **165/165 exit reasons match.** 159/165 P&L exact (<$0.01). Replay net 3,089.80 vs live 3,085.80 — **$4.00 drift on $3,085.80 (0.13%)**. |

**Disclosed replay limitation:** all 6 P&L discrepancies are `TIME_STOP` exits differing
by one bar's close (max $28), an artifact of the live bot's 60-second poll cadence versus
a clean bar-index walk. **Every SL and TP exit reproduces exactly** — which is what this
experiment turns on, since only SL/TP levels differ between arms. Both arms use the
identical engine, so this noise is common-mode and cancels in the paired difference.

This contrasts deliberately with the 2026-08-06 TradeStation finding (145 of 390 bars
revised). The corroborate-don't-repair rule was applied and the venue **passed**.

---

## 6. Primary metric and decision rule

**Primary:** Wilcoxon signed-rank test on the per-trade paired difference
`d_i = pnl_B(i) − pnl_A(i)`, across all 165 pairs. **Two-sided, α = 0.05.**

Two-sided is deliberate. The mechanism predicts fewer stop-outs under B, but a wider stop
also produces larger losses when hit; the net P&L direction is **genuinely uncertain** and
is not pre-specified.

| Condition | Verdict | Pre-committed action |
|---|---|---|
| p > 0.05 | **NO DIFFERENCE** | The floor does not change outcomes. Record. Thread closed. No second variant on this data. |
| p ≤ 0.05 and median(d) < 0 | **B IS WORSE** | A real and reportable result. The 20-bar ATR is not the defect. Record; close. |
| p ≤ 0.05 and median(d) > 0 | **FLOOR HELPS** | Record. Triggers §8. **No live change.** |

**Secondary — mechanism check, never decision-bearing:** change in the count of `SL`
exits, and change in the SL mean/median ratio, A vs B. These test whether the *stated
mechanism* moved, independent of whether P&L did.

---

## 7. What we will NOT do

1. No second exit variant on these 165 trades if B fails. One knob, one test.
2. No sweep of the floor horizon (30/90/120 bars). 60 is derived from `max_hold`, §4.
3. No dropping of trades, months, or 2026-08-20/21 from either arm.
4. No re-running with different `sl_mult`/`tp_mult`. Both frozen at 2.0/4.0.
5. No quoting either arm's absolute PF or net P&L as evidence of profitability — the
   ledger is cost-blind (§2).
6. No change to the live `trader-s26` bot under any outcome of this experiment.
7. No re-derivation of the floor after seeing results.

---

## 8. Successor trigger — and the fee gate that binds it

A **FLOOR HELPS** verdict authorizes **only** drafting a successor pre-registration.

It does **not** authorize deploying the floor, restarting `trader-s26`, or treating s26 as
viable. The binding constraint is unchanged and is recorded here so a passing result
cannot be misread:

> s26's gross edge is **2.985 bps of notional per trade** against Kraken PF_XBTUSD
> round-trip fees of **4 bps (maker/maker) to 10 bps (taker/taker)**. Break-even requires
> ~2.99 bps round-trip — better than Kraken's best maker tier.

**Any exit variant must lift gross edge past ~4 bps round-trip before s26 is a live
candidate at all.** That bar is pre-committed here, before any number is seen. A verdict
that B beats A on gross price deltas says the exit layer leaks; it does **not** say s26
makes money.

The wider purpose is explicit: s26 is the **test bed for whether this shop's exit layer
leaks generally**. A result here informs gap-fade and MIM-NB — which is why the sample
size mattered more than the bot's own viability.

---

## 9. Values fixed at seal time

| Item | Value |
|---|---|
| git HEAD at seal | `03bc783` |
| Bar file SHA-256 (first 16) | `e2c34c09f1d956ca` |
| `s26_soft_fvg_streaming.py` SHA-256 (first 16) | `2399e760e02bf9ed` |
| Frozen entries | 165 (`trader-s26`, timestamp ≥ 2026-06-01) |
| Arm A regression | 165/165 reasons, 159/165 P&L exact, $4.00 net drift |

---

# Amendment 1 — Result (2026-08-27)

Append-only. Original sealed text unedited.

Script: `study_s26_exit_atr_floor.py` | Report: `data/reports/s26_exit_atr_floor_20260827_221517.txt`
Run under seal `be099b9`.

## A1.1 Verdict — NO DIFFERENCE

**Wilcoxon signed-rank, two-sided, on 165 paired differences: p = 0.94042.**

| arm | net | mean/trade | PF | WR | TP | SL | TIME |
|---|---|---|---|---|---|---|---|
| A (ATR20) | 3,089.80 | 18.73 | 1.381 | 45.5% | 58 | 83 | 24 |
| B (floor) | 3,536.07 | 21.43 | 1.421 | 47.3% | 58 | 79 | 28 |

mean(d) = **+2.70**, median(d) = **+0.00**, non-zero differences **69 of 165**.

Per §6 the pre-committed action is: **record, close the thread, no second variant on
this data** (§7.1). The ATR floor does not change outcomes.

The floor was **not inert** — it bound (ATR60 > ATR20) on **73 of 165 entries (44.2%)**.
It had ample opportunity to matter and did not.

## A1.2 The reason this design was worth building

Arm B's aggregate net is **+$446.27 better than Arm A — a 14.4% improvement** — and its
PF is higher (1.421 vs 1.381) and its win rate is higher (47.3% vs 45.5%).

**Every one of those numbers is noise.** The median paired difference is exactly zero and
p = 0.94.

Had this been run as an unpaired before/after comparison — the way the original naive
version of this experiment would have been run — the honest-looking conclusion would have
been *"the ATR floor improves s26 by 14%."* The paired design is the only thing standing
between that sentence and the record. Recorded here as the clearest demonstration this
project has produced of why aggregate P&L comparisons on the same strategy are untrustworthy.

## A1.3 The mechanism did not move either

The §6 secondary check is more damaging to the hypothesis than the primary:

| | Arm A | Arm B | Δ |
|---|---|---|---|
| SL exits | 83 | 79 | **−4** |
| TP exits | 58 | 58 | 0 |
| TIME exits | 24 | 28 | +4 |
| SL mean/median ratio | 1.363 | **1.407** | **worse** |

Binding on 44.2% of entries, the floor converted **four** stop-outs into time-stops and
left the TP count untouched. And the SL fat-tail ratio — the very statistic that motivated
the diagnosis (August 1.95 vs 1.25/1.31) — got **slightly worse**, not better.

**Interpretation, pre-committed as a null:** the August blow-throughs were not caused by
ATR20 being small *relative to ATR60*. On the days that mattered, both estimates were
small: the market expanded beyond what **any** recent-history volatility estimate could
anticipate. That is a regime event, not a calibration error — consistent with this
session's finding that s26's August damage was six trades on 2026-08-20/21 during a
violent rally.

The §1 diagnosis correctly identified *where* the loss occurred. It was **wrong about
why**. Stop width was not the defect.

## A1.4 What this does and does not close

**Closes:** the ATR-floor hypothesis for s26. No second variant is run on these 165
trades (§7.1), and no live change is made (§7.6). `trader-s26` continues unmodified.

**Does not close:** the exit-layer thesis for the sibling bots. gap-fade's defect is a
**time-stop** (6/6 `fill` exits positive vs 18 `time` exits at −$2,977) and MIM-NB's is a
**categorical guard** (100% of losses). Those are different mechanisms and neither was
tested here.

But the honest reading is that the thesis is **weaker than it looked**. s26 was chosen as
the test bed precisely because it was the only sample large enough to give a well-powered
answer, and the well-powered answer is **no**. The sibling evidence remains N=24 and N=20
with no paired test — which is where this session found it.

**Not triggered:** §8. The fee gate is untouched and s26 remains an instrument that cannot
clear its own costs (2.985 bps gross vs 4–10 bps round-trip).
