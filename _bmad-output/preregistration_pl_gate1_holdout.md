# Pre-Registration: PL (Platinum) Gate-1 — One-Shot Sealed-Holdout Test

**Date sealed:** 2026-09-07 (committed BEFORE any holdout access; no bar of PL data on/after
2026-03-01 has been evaluated by any strategy code — see Disclosures)
**Author:** Alex (run at Alex's explicit instruction: "run the platinum holdout test")
**Sibling precedent:** `preregistration_hg_gate1_holdout.md` (seal `fbd7afe`) — this document
mirrors its structure and decision-rule shape deliberately; deviations are marked **[PL-SPECIFIC]**.

## Lineage — how this test became authorized

1. **Exploratory result (frozen, never re-run):** YANK cross-instrument fan-out batch 2,
   2026-06-26. Seal `8f33f12` / `cc17543`.
2. **Cost basis (measured, prospective):** `precommit_pl_slippage_measurement_2026-06.md`
   (`cc17543`) + Amendment 1 `precommit_pl_slippage_amendment1_plv26.md` (`0934500`)
   → **PASS**, `pl_slippage_verdict_20260705.md` (`895d0b5`).
   That verdict's sealed consequence reads: *"Authorized: WRITE a Gate-1 pre-registration for
   the frozen PL structural strategy on its sealed holdout … Before finalizing that prereg, the
   combine-fit gate must be evaluated."*
3. **Combine-fit gate (required precondition): evaluated, FAILED** —
   `pl_combine_fit_verdict_20260705.md`. PL busts the Topstep 50K $2,000 trailing MLL
   structurally (worst single stop −$1,914 = 96% of the buffer; no CME micro exists to size
   down to). That verdict **scoped its own close**: *"this kills PL for the $2,000-trailing-MLL
   50K combine specifically … PL could only ever be revisited on an account whose drawdown
   buffer is large relative to a −$1,900 single-trade risk."*
4. **[PL-SPECIFIC] Vehicle change is the reason this test is now live.** The precondition in (2)
   is satisfied: combine-fit *was* evaluated before finalizing this prereg. It failed for one
   vehicle, and this Gate-1 is run under the vehicle the combine-fit verdict itself named as the
   revisit condition — a **TradeStation SIM / large-buffer account with no trailing MLL**, where a
   −$1,914 single-trade stop is a normal loss rather than an account-ending event.

**Why this is not goalpost-moving (stated for the audit trail).** The holdout tests the *signal*,
which is vehicle-independent. Combine-fit was a *vehicle* question, answered honestly and
recorded as a FAIL for that vehicle. Nothing about the signal hypothesis, the frozen engine, the
cost basis, or the decision-rule shape is being relaxed here — the rule below is the HG rule and
the cost is the measured one. What changed is which account the eventual deployment would target,
and that change makes the test *possible*, not *easier*. If a reader disagrees, the falsifiable
claim is unchanged: net PF < 1.00 on the holdout closes PL permanently, on every vehicle.

---

## Hypothesis (H-PL1)

The frozen YANK structural engine (bearish-FVG + H1-sweep + M15-CHoCH, ML-off, structural mode),
which produced gross PF 1.344 on full-size platinum in the exploratory window, retains a net edge
≥ the program bar on the unseen holdout period 2026-03-01 → 2026-06-12 at the **measured** all-in
cost of $34.00/RT.

## Frozen inputs (nothing below may change between seal and run)

- **Engine/config:** the repo's frozen YANK structural path —
  `backtest_tier2_1year_validation.py --instrument pl --structural`, ML disabled
  (`--ml-threshold 0.0`). No parameter, gate, or exit differs from the run that produced the
  frozen IS trade list. Structural overrides (per `STRUCTURAL_OVERRIDES`): daily circuit breaker
  off, $-gap ceiling off, commission 0 (costs applied offline), 1 contract.
- **Instrument:** full platinum PL, 50 troy oz, **$50/pt, tick $0.10 = $5.00/contract**. No CME
  micro exists. 1 contract.
- **Frozen IS reference:** `data/reports/backtest_1year_20260626_025416.csv` — **N=101,
  gross PF 1.344, gross total +$6,265, gross avg +$62.03/trade**, window 2025-05-19 → 2026-02-28.
- **Cost basis:** all-in **$34.00/RT** per contract (measured pooled median spread $30.00 = 6.0
  ticks + $4.00 commission; Amendment 1 PASS, n=26,158 samples over 6 qualifying RTH sessions).
  Sensitivity cost **$44.00/RT** (worst qualifying session median) — reported, **non-binding**.
- **Holdout window:** 2026-03-01 → 2026-06-12 (end of available data). Data source:
  `data/processed/dollar_bars/1_minute/pl_1min_2025_2026.csv`, whose ≥2026-03-01 rows must
  match the sealed reference copy `data/sealed_holdout/pl_1min_holdout_20260301_plus.csv`
  (integrity check, step 2). Only the file's first/last timestamp have been observed pre-seal
  (to fix the window end); no prices or structure examined.

## Derivation of the decision bar (computed, not hand-set)

From the parent seal `precommit_pl_slippage_measurement_2026-06.md`, applying a per-trade all-in
cost `c` to the **frozen IS trade list**:

| quantity | value | source |
|---|---|---|
| cost ceiling for net PF ≥ 1.10 | **$41.71/RT (8.3 ticks)** | parent seal, computed from frozen list |
| pure breakeven (net PF = 1.00) | $62.02/RT (12.4 ticks) | parent seal |
| **measured** all-in cost | **$34.00/RT** | Amendment 1 PASS |
| ⇒ frozen IS net at measured cost | **net PF ≈ 1.141, +$28.03/trade** | $62.03 − $34.00 |

The binding clause for a full $50/pt contract is **net PF**, not net $/trade (the `net $/trade ≥ $2`
clause is satisfied at any plausible cost and is therefore non-binding — parent seal's own note).

**Expected N (context, not a criterion):** IS rate 101 trades / 9.37 months ≈ **10.8/month** →
≈ **36 expected** over the 3.4-month holdout. The N ≥ 15 floor below guards against a thin-sample
verdict, identical to HG's reasoning (HG: ~10.1/month → ~34 expected, floor 15).

## Protocol (three steps, in order)

**Step 1 — Reproduction gate (no holdout involved).**

```
PYTHONPATH=. .venv/bin/python backtest_tier2_1year_validation.py --instrument pl --structural \
  --ml-threshold 0.0 --start 2025-05-19 --end 2026-02-28
```

Must reproduce the frozen IS result: **N=101, gross PF 1.344 (±0.005), gross total +$6,265 (±$5)**,
and a trade list matching `backtest_1year_20260626_025416.csv` on (entry_time, direction, pnl).
**If it does not reproduce, ABORT — no holdout access.** Reproduction-gate iterations are unlimited
(they touch no holdout data); the holdout run remains single-shot.

**Step 2 — Holdout integrity check (metadata only).** Verify the working file's rows timestamped
≥ 2026-03-01 are identical to `data/sealed_holdout/pl_1min_holdout_20260301_plus.csv` on OHLCV
(row count + content diff; timestamp-separator and float-repr noise in the `notional` column are
tolerated, per the HG precedent). Mismatch → ABORT; the sealed 444 copy is authoritative.

**Step 3 — One-shot holdout run.**

```
PYTHONPATH=. .venv/bin/python backtest_tier2_1year_validation.py --instrument pl --structural \
  --ml-threshold 0.0 --start 2026-03-01 --end 2026-06-12 \
  --preregistration <SHA of the commit sealing this document>
```

The script verifies the SHA and appends the access record to `data/sealed_holdout/ACCESS_LOG.md`
(that log change is committed). Net P&L is computed offline from the emitted trade list:
`net_i = gross_i − $34.00` per trade (1 contract). **One run. No re-runs, no parameter changes,
no subgroup selection, regardless of outcome.** The result is recorded in ACCESS_LOG and in a
verdict doc either way.

## Sealed decision rule (evaluated at $34.00/RT all-in)

- **PASS:** net PF ≥ 1.10 **AND** N ≥ 15 **AND** ex-top-3-days net total > $0
  → authorizes drafting a **deployment pre-registration** for prospective paper trading
  (TradeStation SIM, PLV26 or the then-front contract, **1 contract**). Deployment itself remains
  a separate gate requiring Alex's explicit go; nothing trades from this result.
- **INSUFFICIENT:** N < 15 (regardless of PF) → PARK; no verdict on the edge; option to log
  prospectively on paper.
- **MARGINAL:** net PF 1.00–1.10 (N ≥ 15) → PARK; no deployment path; may only be revisited with
  fresh prospective data under a new prereg.
- **FAIL:** net PF < 1.00 (N ≥ 15) → **PL closed as a net candidate on every vehicle**;
  cross-instrument claim logged as "structural edge did not survive the holdout at measured cost."

**Disposition of a result that clears net PF but fails the fat-day clause.** PASS requires all
three clauses, so such a result grants **no deployment authorization**. But the sealed FAIL clause
is *net PF < 1.00*, which such a result does not meet. Therefore that case is **PARK**, with the
same disposition as MARGINAL — **not** a close. This is stated now so it cannot be argued either
way after the fact.

**Specification of "ex-top-3-days":** trades are grouped by **exit date** (the date the P&L is
realized), the three highest net-P&L days are removed, and the remaining net total must be > $0.
Fixed here, before the run.

Reported but **non-binding**: net PF and expectancy at the $44.00/RT sensitivity cost; net $/trade;
win rate; exit-type mix; monthly P&L; max drawdown; worst single trade (context for the
large-buffer vehicle claim). A boundary result is read at face value — a 1.09 net PF is
MARGINAL/PARK, not "almost passed" (Option B precedent).

## [PL-SPECIFIC] Pre-run expectation, recorded before spending the one-shot

Running the analyzer over the **frozen IS trade list** (no holdout data involved) reproduces the
parent seal exactly — N=101, gross PF 1.3440, gross total +$6,265, net PF 1.1411, +$28.03/trade,
worst trade −$1,914, max DD $4,890 — and reveals one fact the parent chain recorded only as the
shorthand "tail 130%":

> **The frozen in-sample reference itself FAILS the ex-top-3-days clause.**
> Top 3 days = +$4,252, +$2,352, +$1,386 = **+$7,990**, against a net total of **+$2,831**.
> Ex-top-3-days net = **−$5,159**. PL's in-sample edge is entirely carried by three days out of
> nine months; the other ~9.1 months lose money net of measured cost.

Three consequences, all fixed before the run:

1. **The clause is NOT relaxed.** It is inherited verbatim from the HG seal. Dropping or softening
   a criterion after seeing the candidate fail it is precisely this shop's most-documented failure
   pattern (`feedback_iteration_loop_pattern`), and the vehicle change does not license it —
   fat-day dependence is a survival problem on a combine and a *return-quality* problem on a
   large-buffer account, but it is a problem on both.
2. **The expected outcome of this test is therefore NOT PASS → PARK**, and it is recorded here in
   advance. The clause gets *harder*, not easier, on the holdout: removing 3 days from a 3.4-month
   window removes proportionally more than removing 3 days from a 9.4-month window. A subsequent
   PASS would mean the holdout showed a **broader** edge than the in-sample did — a genuinely
   surprising and therefore genuinely informative result. A subsequent fat-day PARK is the
   expected result and must not be written up as a discovery.
3. **What this test is actually for, then.** The unknown being purchased with the one-shot is the
   *primary economics on unseen data*: does net PF at the measured $34.00/RT stay above 1.00 (the
   close line) or above 1.10 (the deployment line), and at what N. Copper's holdout answered that
   question with gross PF 0.563 — a collapse, not a tail problem. Whether PL collapses the same
   way is not knowable from the in-sample and is worth the one-shot, which has **no alternative
   use** (this sealed slice can only ever test PL).

## Disclosures (contamination and known-context — do not drop these)

1. **[PL-SPECIFIC] Lowered portability prior, twice over.** Copper — the cleaner sibling
   candidate, with a *better* cost margin relative to its ceiling — passed its slippage gate and
   then **FAILED its Gate-1 holdout** (`328cdaf`: N=26, gross PF 0.563, net PF 0.463; gross itself
   went negative). The PL slippage verdict already recorded this as direct evidence against the
   cross-instrument thesis and instructed that a PL holdout be treated as **a lower-base-rate bet
   than copper was**. That instruction stands and is restated here at seal time.
2. **[PL-SPECIFIC] The operator knows the sibling failed on this exact calendar window.** HG's
   Gate-1 covered 2026-03-01 → 2026-06-12, the identical window, and failed. The operator also
   knows the 2026 macro regime (Iran war, Mar–May 2026) and MNQ-instrument results on the same
   window. **No PL price data in the window has been examined.** This knowledge cannot bias the
   run (the engine is frozen and the rule is sealed) but it does mean a PL FAIL is weak *new*
   evidence about the war regime, while a PL PASS would be notable precisely because its sibling
   failed there.
3. **Thin book — the measured spread is a best-case 1-lot fill.** Median platinum book is **1×1**.
   The $34.00/RT clears the frozen ceiling, but depth beyond 1 lot is unproven. This Gate-1 and
   any deployment that follows are **1 contract only**; a size increase would require a fresh
   depth measurement, not an extrapolation.
4. **Working file physically contains the holdout rows.** `pl_1min_2025_2026.csv` spans
   2025-05-01 → 2026-06-12; only its first and last timestamps were viewed (to fix the window
   end). The script-enforced prereg gate blocks strategy evaluation on ≥2026-03-01 bars without
   a valid SHA.
5. **Cost is not fit to holdout prices.** The slippage capture ran 2026-06-26 → 07-03, entirely
   *after* the holdout window.
6. **Engine copy used.** Run from worktree `post-r3-options-research`, whose engine carries
   `68225eb` (backtest trade-log persistence gated behind `TradeLogger(persist=)`) — so this
   replay writes no rows to `data/trades.db`. Verified before the run
   (`tools/verify_no_persist_fix.py`).
7. **Runtime is not reproducible on this box** (`project_tier2_backtest_engine_gotchas_20260905`);
   no run-count or wall-clock estimate is relied upon by this protocol.

## Kill precedents acknowledged

S26 net-cost death (costs inside the primary metric, at measured not assumed values); MIM-NB
fat-day fragility (ex-top-3 criterion); S26-KZ subgroup pattern (no segment or subgroup may
rescue a failing full-window result); the iteration-loop pattern (single sealed shot, no
re-entry, no restrict-to-favorable-subset); Option B boundary honesty; HG's own FAIL (accepted
without re-runs or subgroup rescue — the same discipline binds here).

## Out of scope

Deployment (a separate prereg + Alex's go); position sizing beyond 1 contract; the Topstep combine
(closed for PL by `pl_combine_fit_verdict_20260705.md`, and this prereg does not reopen it);
correlation with the live book and floor-monitor integration (deployment-prereg territory).
