# Pre-Registration: MIM-X2 — Market Intraday Momentum on MNQ, Adequately Powered

**Registered:** 2026-08-28
**Status:** SEALED at commit time. Append-only amendments.
**Supersedes in scope (not in record):** `preregistration_intraday_momentum_mnq.md` (MIM-X),
which returned an **underpowered null** — its A1.3 showed the entire JFE effect range sat
below its detection threshold. MIM-X's result stands as recorded; this is the powered redo.

---

## 1. What changed, and why this seal exists

MIM-X tested the Baltussen/Da/Lammers/Martens rule (JFE 142 (2021) 377–403) on N=303 MNQ
sessions and returned mean net −$12.04/trade. That verdict was **uninformative**: at N=303
and sd $122.97 the smallest detectable mean at t=2 was $14.13/trade, while the paper's
Sharpe range implies only $6.74–$13.40. The test could not have seen the effect.

MIM-X Amendment 1 §A1.5 set the corrective rule for this project:

> "compute the minimum detectable effect size at the planned N BEFORE sealing, and state it
> in the seal alongside the decision rule. A verdict threshold without a power statement can
> manufacture a confident null."

**This seal complies with that rule. §4 is the power statement, and it is written before any
outcome has been computed.**

## 2. Data — newly acquired, not the two local CSVs

23 quarterly MNQ contracts (H21 … U26) fetched from TradeStation, **2,028,965 one-minute
bars**, 2020-12-17 → 2026-08-28. Stored `data/mim_x/mnq_1min_by_contract.csv`.

**Roll handling — no splice.** Each contract has an ACTIVE WINDOW `[3rd Friday of
expiry_month − 3 months, 3rd Friday of expiry_month]`. Both `r_ROD` and `r_LH` are computed
**within a single contract's own series**; the first qualifying session of each contract is
dropped for lack of a prior close inside that contract. No return is ever computed across a
roll boundary.

## 3. Frozen specification — identical to MIM-X, deliberately unchanged

| Item | Value |
|---|---|
| Instrument | MNQ, 1 contract, $2.00/point |
| `r_ROD` (signal) | close(15:30 ET) / close(prior session 15:59 ET) − 1, same contract |
| `r_LH` (traded) | close(15:59 ET) / close(15:30 ET) − 1 |
| Position | LONG if `r_ROD > 0`, SHORT if `r_ROD < 0`, skip if exactly 0 |
| Entry / Exit | close of 15:30 ET bar / close of 15:59 ET bar |
| Trades/day | exactly 1 |
| **Primary friction** | **$3.00 round trip** |
| Secondary friction | $2.00 RT, reported, never decision-bearing |
| Window | 2020-12-21 → 2026-08-27 |

**No parameter is changed from MIM-X.** The only differences are sample size and data
source. If the rule is altered here, the comparison to MIM-X is destroyed.

## 4. POWER STATEMENT — computed and fixed before the run

Counted from the data **without applying the signal's sign** — the quantities below are
properties of the instrument, not of the strategy, so nothing about the outcome is known:

| Quantity | Value |
|---|---|
| Qualifying sessions, **N** | **1,392** |
| sd of the last-30-min dollar move, 1 MNQ | **$99.08** |
| Minimum detectable effect at t=2.0 (≈50% power) | **$5.31/trade** |
| Effect detectable at 80% power | **$7.43/trade** |

Against the JFE claim translated to per-trade terms at this sd:

| JFE annualised Sharpe | implied mean/trade | power here |
|---|---|---|
| 0.87 (bottom of range) | $5.43 | **~50% — marginal** |
| 1.30 (mid) | $8.11 | **>80% — adequate** |
| 1.73 (top) | $10.80 | **>95% — strong** |

**Declared before the run:** this test is adequately powered for the mid and top of the
JFE range and only **marginally** powered at its bottom. A null result must therefore be
reported as "no effect of the claimed mid-to-upper magnitude", **not** as "no effect".

## 5. Primary metric and decision rule

**Primary:** mean **net** P&L/trade at $3.00 RT, two-sided t-test over 1,392 pairs.

| Condition | Verdict | Pre-committed action |
|---|---|---|
| mean net ≤ 0 | **FAILS** | The effect is absent or negative on MNQ at deployable cost. Record; close; no variant search. |
| mean net > 0, t < 2.0 | **POSITIVE, UNPROVEN** | Record; eligible for a prospective seal; no deployment. |
| mean net > 0, t ≥ 2.0 | **SURVIVES** | Record. Triggers §8. Still no deployment. |

**Secondary, never decision-bearing:** the paper's regression `r_LH = a + b·r_ROD` (β, t, R²);
win rate; gross figures; edge in bps against the 0.51 bps friction benchmark; per-year
breakdown; and results at $2.00 friction.

**Per-year breakdown is reported but explicitly NOT decision-bearing**, per this session's
September-2025 lesson: a result carried by one year is disclosed as such in the verdict line.

## 6. What we will NOT do

1. No change to the 30-minute window, the predictor, or the friction if the result fails.
2. No volatility, day-of-week, session or regime filter added after seeing results.
3. No dropping of years, contracts, or qualifying sessions.
4. No re-run at lower friction to rescue a negative. $3.00 is primary.
5. No sizing above 1 contract.
6. No re-derivation of the roll convention after seeing results.
7. **No further redo of this test at larger N.** 1,392 sessions is the available history;
   if this is inconclusive, the answer is that MNQ cannot settle it, not another sample.

## 7. Disclosed limitations

- Single instrument. JFE's claim is cross-market (60+ futures); an MNQ result does not
  generalise to it either way.
- Window 2020-12 → 2026-08 is **entirely outside** the paper's 1974–2020 sample. Genuinely
  out-of-sample in time — a strength for validity, and it means a null cannot distinguish
  "decayed since 2020" from "never present in MNQ".
- 0DTE option volume grew substantially over this window, which plausibly *increases* the
  dealer-gamma mechanism the paper identifies. Stated as an unverified prior, not a finding.

## 8. Successor trigger

A **SURVIVES** verdict authorises drafting a prospective pre-registration only. Not
deployment, not sizing, not a combine promotion. Noted: entry 15:30 / exit 15:59 ET is flat
by the close — no overnight or weekend exposure, so unlike LRC this is venue-compatible in
principle.

## 9. Values fixed at seal time

| Item | Value |
|---|---|
| git HEAD at seal | `816878a` |
| Bar file | `data/mim_x/mnq_1min_by_contract.csv`, 2,028,965 bars, 23 contracts |
| Session file | `data/mim_x/sessions.csv`, N=1,392 |
| Source | Baltussen, Da, Lammers & Martens, JFE 142 (2021) 377–403 |

---

# Amendment 1 — Result (2026-08-28)

Append-only. Original sealed text unedited.
Script: `study_mim_x2_powered.py` | Report: `data/reports/mim_x2_powered_20260828_152100.txt`
Ledger: `data/reports/mim_x2_trades.csv`. Run under seal `b890967`.

## A1.1 Verdict — FAILS

N = 1,390 qualifying sessions (2 of the sealed 1,392 had `r_ROD` exactly 0 and were
skipped per §3), 2020-12-21 → 2026-08-27, 23 contracts.

| | total | mean/trade | t | WR | PF |
|---|---|---|---|---|---|
| GROSS (no costs) | −$2,929 | **−$2.11** | −0.793 | 48.5% | 0.9403 |
| **NET @ $3.00 RT (primary)** | −$7,099 | **−$5.11** | −1.921 | 47.5% | 0.8614 |
| NET @ $2.00 RT (secondary) | −$5,709 | −$4.11 | −1.545 | 47.8% | 0.8869 |

Mean net at primary friction ≤ 0 → **FAILS** per §5. Recorded; closed; no variant search;
and per §6.7 **no third attempt at larger N**.

## A1.2 This time the null is informative — which is the whole point of the redo

The sealed §4 power statement, fixed before the run: minimum detectable effect at t=2.0
was **$5.31/trade**, 80%-power threshold **$7.44**, against a JFE-implied range of
**$5.43–$10.80**.

Observed **gross** edge: **−$2.11/trade**.

The mid ($8.11) and top ($10.80) of the JFE range were detectable at >80% and >95% power
respectively. Neither appeared. **Therefore: market intraday momentum of the claimed
mid-to-upper magnitude is absent on MNQ over 2020-12 → 2026-08.**

That is a genuine finding, and it is exactly what MIM-X could not deliver. The contrast is
the lesson: same rule, same instrument, same specification — N=303 produced a
verdict that meant nothing, N=1,390 produced one that means something.

**Scope it honestly.** §4 declared the bottom of the range ($5.43) only ~50% powered, so a
*weak* version of the effect is **not excluded**. The correct statement is "no effect of the
claimed mid-to-upper magnitude", not "no effect".

## A1.3 It remains a null, not a reversal — and the β sign is unstable across samples

Gross t = **−0.793**. The negative is **not significant**. Nothing here supports "intraday
momentum is inverted on MNQ".

The paper's own regression:

| | MIM-X (N=303) | MIM-X2 (N=1,390) |
|---|---|---|
| β | **+0.00069** | **−0.00278** |
| t(β) | +0.063 | −0.499 |
| R² | 0.00001 | 0.000179 |

**The sign flipped between samples and neither estimate is significant.** That instability is
itself evidence that β is estimating noise rather than a small true effect. Note also that
the net point estimate moved *toward* zero as N grew (−$12.04 → −$5.11), which is the
behaviour expected when the truth is approximately zero and the small-sample figure was
noise.

## A1.4 Per-year — reported, NOT decision-bearing (§5)

| year | N | mean net | total |
|---|---|---|---|
| 2020 | 7 | −$10.29 | −$72 |
| 2021 | 247 | +$0.74 | +$184 |
| 2022 | 246 | **+$6.96** | +$1,713 |
| 2023 | 244 | −$4.79 | −$1,168 |
| 2024 | 245 | −$11.91 | −$2,918 |
| 2025 | 241 | **−$18.17** | −$4,379 |
| 2026 | 160 | −$2.87 | −$458 |

There is a visible monotone-ish deterioration from 2022 to 2025. **It is not a finding and
must not be quoted as one.** §5 pre-committed this table as non-decision-bearing precisely
to prevent that, and each year carries only ~245 sessions — a per-year detectable effect of
~$12.65 at t=2, so most individual years are themselves underpowered. Any decay claim would
require its own seal and its own power statement.

## A1.5 What can and cannot be concluded

**Can:** on MNQ, over a window entirely outside the paper's sample, the JFE intraday-momentum
effect does not appear at the mid-to-upper end of its claimed magnitude, and does not survive
$3.00 round-trip friction at 1 contract.

**Cannot:** (a) refute Baltussen et al. — their claim is cross-market over 60+ futures and
1974–2020, and a single-instrument out-of-sample null does not overturn it; (b) exclude a
weak (~Sharpe 0.87) version, which was only ~50% powered here; (c) separate "decayed since
2020" from "never present in MNQ specifically" — §7 disclosed this in advance and the result
does not resolve it.

## A1.6 Status

Closed. No live system touched, no parameter changed, nothing deployed. The 2M-bar
contract-stitched MNQ history acquired for this test (`data/mim_x/mnq_1min_by_contract.csv`,
2020-12 → 2026-08, with `upvol`/`downvol` columns retained) is a durable asset and is
reusable for any future test at this horizon — notably the bar-level signed-order-flow axis,
which is the remaining live candidate from the research run.
