# Pre-Registration: VRP-1 — Volatility Risk Premium on VX Term Structure

**Date sealed:** 2026-09-07 — committed BEFORE any strategy P&L is computed on any VX data.
**Author:** Alex (session run at Alex's instruction: "pre-register the VRP study").
**Feasibility basis:** `_bmad-output/feasibility_vrp_data_20260907.md` (commit `ed8c7e1`).
**Lineage:** Option 5 of `research_plan_post_r3_options_20260904.md`, closed 2026-09-06 as
DATA-BLOCKED, reopened 2026-09-07 when the VIX/VX path was verified reachable.
**Precedent seals this one deliberately mirrors:** `preregistration_hg_gate1_holdout.md` (`fbd7afe`)
for the gate taxonomy and the cost-ceiling pattern; `project_xsmom1_power_gate_20260907` for the
firewalled power gate. Deviations are marked **[VRP-SPECIFIC]** and justified.

---

## 1. Hypothesis

**H-VRP1 (primary).** A short position in the front-month VIX future, held continuously and rolled
on a fixed schedule, earns a positive risk-adjusted return over the development window — the
volatility risk premium — that survives realistic retail transaction costs at **one Mini-VIX (VXM)
contract**.

**H-VRP2 (secondary, one knob, only if H-VRP1 passes and only under a separate amendment).** That
return is improved by conditioning entry on the term-structure basis (contango depth).

## 2. Vehicle — stated up front, because it is load-bearing

**TradeStation SIM / any account whose drawdown buffer is large relative to a single adverse VX
move.** Declared buffer for all survivability arithmetic: **$50,000, 1 VXM contract.**

**This study is NOT for the Topstep combine and must not be repurposed to it without a new seal.**
Short volatility is a *negative-skew* strategy: it collects small and loses large. A trailing-MLL
combine is structurally hostile to exactly that, and `pl_combine_fit_verdict_20260705` is the
worked example of what happens when this is checked late. Checking it first, here, is deliberate.

## 3. Instrument and data construction

- **Economics:** Mini-VIX (`VXM`), **$100/index point**, exchange CBOEF. Full VX ($1,000/pt) is
  explicitly **rejected** — at 10× the risk it reproduces the platinum sizing failure.
- **Price series:** **individual monthly contract series only** (`VX{M}{YY}`), never the stitched
  continuous `@VX`. **[VRP-SPECIFIC] This is a correctness requirement, not a preference:** the
  roll yield *is* the signal, and a back-adjusted continuous series either bakes it in or removes
  it. Verified available back to **VXM08 (2007-06-21)**, including `VXG18`/`VXH18` (Feb-2018
  Volmageddon) and `VXH20`/`VXM20` (COVID).
- **Term structure:** per date, contracts ordered by days-to-final-settlement → M1, M2, M3…
- **Spot reference:** `$VIX.X` (5,000 daily bars, 2006-10-25 →).
- **Roll rule (FIXED, not swept in this seal):** hold M1; at the close **3 trading days before
  final settlement**, close the short and open the next contract. A roll-timing sweep is a
  different knob and requires its own seal.
- **Disclosure — pre-2020 VXM is a simulation.** VXM launched 2020-08. Before that, VXM economics
  are computed from VX prices × 1/10. This is *exact* in index terms (same underlying, same
  settlement) but it does **not** simulate VXM liquidity or spread. Hence §7's mandatory
  prospective slippage gate; no cost may be assumed from this construction.

## 4. Windows — split declared before any P&L

| window | dates | contains |
|---|---|---|
| **Development (in-sample)** | 2007-06-21 → 2018-12-31 | GFC 2008, **Feb-2018 Volmageddon** |
| **Sealed holdout (one-shot)** | 2019-01-01 → 2026-09-07 | **COVID Mar-2020**, 2026 war regime |

**[VRP-SPECIFIC] Both windows deliberately contain a major short-vol disaster.** Neither can pass
by dodging tail events — the failure mode that makes published VRP backtests look better than the
trade. The holdout is written to `data/sealed_holdout/vx_term_structure_20190101_plus.csv`,
`chmod 444`, and registered in `ACCESS_LOG.md` at construction time. **No strategy statistic of any
kind may be computed on holdout rows before Phase 3.** Construction code must assert its output
never contains a P&L column.

## 5. Phases — strictly in order, each gated on the last

**Phase 0 — POWER GATE (firewalled; spends no data).** Using only the realized volatility of the
daily front-contract return series and N, compute the **minimum detectable annualized Sharpe** at
80% power, α = 0.05, two-sided. **If MDE > 0.5, the study closes UNDERPOWERED and nothing else
runs.** No P&L is computed in this phase. (XSMOM-1 pattern: a design that could not have seen an
edge is closed before it consumes anything. Note its lesson — a critical value is not an MDE.)

**Phase 1 — Gate 0 on the development window.** Primary spec only (unconditional short M1, fixed
roll). Outputs the full **cost curve**: `c*@0.5` and `c*@0.0` = the all-in $/round-turn at which
net annualized Sharpe falls to 0.5 and to 0.

**Phase 2 — one knob (separate amendment, only if Phase 1 PASSES).** Contango-basis threshold on
`(M1 − spot)/spot`, swept over a grid declared in that amendment, and required to beat the 95th
percentile of a random-threshold null. Not authorized by this document.

**Phase 3 — sealed holdout, ONE SHOT.** Single run, no re-runs, no parameter changes, no subgroup
selection, result recorded either way.

**Phase 4 — prospective VXM slippage measurement** (HG/PL method) — see §7.

**Phase 5 — deployment pre-registration.** Separate document, requires Alex's explicit go. Nothing
trades from any phase above.

## 6. Sealed decision rule — Phase 1 (development window)

**PASS requires ALL of:**

1. **Gross annualized Sharpe ≥ 0.5** — the bar is **inherited from TSC-1**, which was judged
   against exactly this 0.5 threshold (`project_tsc1_term_structure_carry_result`: "best deadzone
   Sharpe +0.169 vs 0.5 bar"). Not a number invented for this study.
2. **N ≥ the Phase-0 power requirement.**
3. **Beats the 95th percentile of a random-sign null** — identical holding periods and roll dates,
   position sign drawn at random. **[VRP-SPECIFIC]** This is the load-bearing test. It guards the
   exact failure mode of every published VRP result: that "short a persistently contangoed asset"
   is indistinguishable from "the sample happened to be a vol-decay regime." TSC-1 died on this
   same structural point (persistent one-sided contango, not an oscillating signal) — a fact that
   *lowers* the prior here and is recorded as such.
4. **Survivability at 1 VXM on the $50,000 declared buffer: max drawdown ≤ 12% of buffer
   (= $6,000).** The 12% is **inherited from this repo's own `config.yaml`
   `risk.max_drawdown_percent` default**, not chosen for this study.
5. **[VRP-SPECIFIC] Breadth clause, replacing the fat-day clause.** Net P&L **excluding the best
   10% of months** must remain > $0. The HG/PL `ex-top-3-days > 0` clause is **deliberately not
   reused**: it was designed to catch *tail-carried* edges, and a premium-harvesting strategy has
   the opposite shape by construction — many small gains, rare large losses. Testing a
   negative-skew strategy for tail-carriage would be the wrong test asked in the wrong direction.
   The right broadness question is whether the premium accrues steadily, which clause 5 asks.
6. **Crisis survivability (binding).** Reported separately for 2008-09→2008-12, **2018-02**, and
   (holdout only) 2020-02→2020-04: the strategy must not breach the declared buffer at 1 contract
   in any of them. **A strategy that is profitable only by being ruined once fails**, regardless of
   aggregate Sharpe.

**Cost handling.** Phase 1 is computed gross; the binding cost comparison is deferred to Phase 4
by design (no VXM cost card exists, and none may be assumed). **PASS is conditional: the measured
VXM all-in cost must be ≤ `c*@0.5`.** This is the HG/PL pattern — measure the cost, then compare
it to a ceiling derived from the trade list, never the reverse.

**Taxonomy** (mirrors HG/PL, with the gap PL exposed closed in advance):
- **PASS** — all six clauses + `measured ≤ c*@0.5` → authorizes Phase 2/3.
- **MARGINAL** — Sharpe in [0, 0.5) → PARK; no deployment path; revisit only under a new seal.
- **NOT PASS → PARK** — Sharpe ≥ 0.5 but a robustness clause (4, 5 or 6) fails → parked, **not
  closed**. Stated now so it cannot be argued either way later.
- **FAIL** — gross Sharpe < 0, or `c*@0.5` below any plausible VXM cost → VRP-1 closed.
- **UNDERPOWERED** — Phase 0 fails → closed as a design that could not have seen an edge.

## 7. Gates that bind before anything trades

1. **VXM slippage measured prospectively**, HG/PL method: quote capture over ≥5 qualifying
   sessions, thresholds derived from the frozen Phase-1 trade list *before* any quote is analyzed.
   **Do not assume a cost.** Carry the twice-paid lesson: **detect contract rolls by
   quote-staleness / spread-sanity, never by sample count** — dead front months keep answering the
   quote endpoint at full sample parity (MHGN26, PLN26).
2. **Trade-permission check.** Data entitlement ≠ trade entitlement — proven in this very probe by
   the options `403 Missing required scope`. Confirm CFE/VXM is tradeable on the target account
   **before** any execution work.
3. **Basis risk, if `$VXN.X` is ever used as a signal.** VX/VXM track **VIX (S&P 500)**, not VXN
   (Nasdaq-100). A Nasdaq-flavoured signal on an S&P instrument is a real basis mismatch and would
   need its own seal; **this study does not use `$VXN.X`.**

## 8. Disclosures

1. **This is a replication, not a discovery.** VRP is among the most published effects in finance
   and the operator knows its canonical blowups (XIV termination Feb-2018, Mar-2020). That prior
   is not data snooping, but it does mean a positive result here is *confirmation of a known
   effect*, and the only genuinely open question is **whether it survives retail cost at one VXM
   contract**. It earns no methodological shortcut for being well known.
2. **The sibling prior is negative.** TSC-1 (term-structure carry) FAILED Gate 0 on MGC/SIL/MNQ,
   specifically because contango was persistent and one-sided rather than oscillating. VX is a
   different market with a genuine structural reason for its premium, but the failure mode is the
   same family — clause 3 exists to catch it.
3. **The whole book's edges are tail-carried** (`project_fanout_rerun_and_pl_abort_20260907`: all 8
   instruments have negative ex-top-3-days). VRP is the opposite exposure. That is a
   diversification argument *and* a new risk this shop has no operational experience managing.
4. **No VX/VXM price data has been examined for strategy behaviour.** Only symbol availability,
   bar counts and date ranges were observed (`tools/probe_vrp_feasibility.py`), plus four quote
   prints recorded in the feasibility memo.
5. **Runtime unpredictability** on this box is documented; no phase relies on a wall-clock estimate.

## 9. Kill precedents acknowledged

The iteration-loop pattern (single sealed shot, no re-slicing, no restrict-to-favourable-subset);
S26's net-cost death (cost inside the primary metric, measured not assumed); Option B's boundary
honesty (a near-miss is PARK, not "almost passed"); HG's Gate-1 FAIL accepted without subgroup
rescue; and **PL's abort — a frozen reference is only frozen if a committed SHA regenerates it.**
**[VRP-SPECIFIC] Consequence of that last one, binding here:** every artifact this study freezes
must be regenerated from a committed SHA *at freeze time*, and the freeze must record the exact
command plus that SHA. No artifact is cited later unless that regeneration was verified.

## 10. Out of scope

Phase 2's contango filter (needs its own amendment); roll-timing sweeps; full-size VX; `$VXN.X`
signals; options-chain strategies (blocked at 403); any combine deployment; position sizing beyond
1 contract.
