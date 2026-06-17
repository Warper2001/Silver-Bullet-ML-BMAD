# Pre-Registration: YANK + MIM-NB Joint Combine Monte Carlo

**Generated:** 2026-06-17
**Experiment ID:** yank-mim-joint-combine-mc
**Pre-registration base commit:** fd2bb26 (seal commit SHA recorded by the commit itself)
**Status:** SEALED (re-seal v2). The joint MC has not been computed under this corrected spec. Per §7, the first seal (bfecfd1) omitted MIM-NB's deployed 500-pt cat-stop from the input series (it modeled the pre-cat-stop V2 at a 48.4% baseline instead of the deployed 54%); §3 is corrected and the experiment is re-sealed before the governing run.
**Objective (fixed by Alex, 2026-06-17):** maximize Topstep 50K combine **PASS probability**. Steady-income optimization is explicitly deferred until there is account headroom. This experiment is scored on pass% / blow%, not on Sharpe or expectancy.

---

## 0. Why this experiment (one paragraph)

MIM-NB at 1 contract passes the Topstep 50K combine MC at **~54% pass / ~33% blow / ~13% run-on** (single-strategy, already observed). The binding failure mode is **path lumpiness against a trailing drawdown ratchet**: the edge is ~$32/ct concentrated in ~3 fat-tail days of 163, so on the other ~160 days the account is a near-driftless walk near a one-sided absorbing floor. Sizing MIM-NB *up* makes this worse (2ct moved blow ~17%→33%) because it scales path variance against a fixed floor. The one lever that reduces path variance **without** reducing drift is an **uncorrelated second return stream**. YANK (the other independently-validated MNQ edge) is effectively uncorrelated with MIM-NB. This experiment asks, pre-registered: **does running YANK alongside MIM-NB on one combine account raise pass% above the 54% single-strategy baseline without raising blow% above its 33% baseline?**

This is **diversification of two independently held-out-validated edges**, not restrict-to-favorable-subset. Neither strategy's internal logic is touched.

---

## 1. Integrity Disclosure (read before the spec)

- **No joint MC has been run.** The decision gate (§5) is being fixed *before* the first joint simulation.
- **Inputs are already-observed series**, reused verbatim from each strategy's own sealed validation — combining them introduces no new look at held-out data. The realized correlation/vol numbers in §3 are descriptive statistics of those same series and were computed for sizing only; they do not constitute a new OOS test.
- **No new fitting is permitted.** Sizing is set by an inverse-volatility rule derived from the series' own daily vols (§4), not searched for the best pass%. The sizing grid is frozen to **YANK ∈ {1, 2, 3} contracts with MIM-NB fixed at 1ct** — three points, no interpolation, no extension.
- **Known weakness, disclosed:** the joint pool is thin. YANK has 47 traded days; MIM-NB and YANK both traded on only **18 common days** in the overlap window. Both edges are MNQ, so tail-regime correlation can rise above the benign ~0.015 measured here. The primary method (§4) preserves whatever joint structure the realized data contains; a sensitivity arm tests the optimistic (independent) case so the two bracket the truth.

---

## 2. Hypothesis

**H₁:** Adding YANK to the MIM-NB combine account at a vol-balanced size lifts Topstep 50K MC **pass% above 54%** while keeping **blow% at or below 33%**, at some frozen size in {YANK 1, 2, 3}ct × {MIM 1ct}.

**H₀:** The joint config does not beat the single-strategy baseline on pass% at equal-or-lower blow% — i.e. YANK's added per-trade cost and own variance offset its diversification benefit on one shared floor. If H₀ holds at all three sizes, the joint-on-one-account branch closes and the fallback is Victor's parallelism lever (two separate combine accounts; not tested here).

---

## 3. Inputs (Frozen)

**MIM-NB** — deployed V2 variant **+ 500-pt catastrophe stop** (the live config), net of $2.24/ct ($1.12 pts) per completed trade, $2/pt.
- Dev 2025: `data/reports/mim_nb_gate0_v2_2025.csv` — sha256 `fb26140b0c1d88ab…`
- OOS 2026: `data/reports/mim_nb_gate1_v2_2026oos.csv` — sha256 `027741fe2a926694…`
- 163 traded days, 2025-01-24 … 2026-05-19. Per-trade `pnl_pts`, grouped by ET `day`.
- **Cat-stop (binding):** the gate CSVs are the raw V2 trades *without* the deployed catastrophe stop. The live bot runs `CAT_STOP_PTS=500`, so each trade's point P&L is floored at −500 before cost: `net1ct = (max(pnl_pts, −500) − 1.12) × 2`. This is the variant that produces the **54% / 33% single-strategy baseline** used in the gate (verified: cap=500 → 54.0%/33.3%; cap=None → 48.4%/41.4%). YANK's own SL2 stop is already baked into its backtested trades, so no analogous cap is applied to YANK.

**YANK** — sealed config SL2 / TP8 / ml_threshold 0.50, seal run 181838.
- Trades (primary input): `data/reports/backtest_1year_20260615_181838.csv` — sha256 `88c026b0c113712e…`
- Daily P&L (cross-check only): `data/reports/equity_curve_1year_20260615_181838.csv` — sha256 `3356f1b39955d810…`
- 48 traded days, 2025-05-22 … 2026-05-04. Native size 5ct.
- **Cost normalization (binding):** the YANK backtest netted only `commission_per_roundtrip=$4.00` at 5ct = **$0.80/ct**, whereas MIM-NB and the combine use the TopstepX **$2.24/ct**. To cost both engines identically, YANK is **re-costed per trade** to $2.24/ct: `net_1ct = (pnl_5ct + 4.00)/5 − 2.24`. This uses the per-trade file (not the daily equity curve) so trade count per day is exact. YANK trade timestamps are UTC; the **`day` key is the ET date of `entry_time`** to align with MIM-NB's ET-session days.

**Realized joint statistics (descriptive, sizing-only — computed 2026-06-17):**

| Quantity | Value |
|---|---|
| MIM-NB daily P&L std, 1ct (own traded days / overlap) | $452 / **$303** |
| YANK daily P&L std, 1ct | **$142** |
| Daily return correlation (overlap, flat=0) | **+0.0148** |
| Monthly return correlation | −0.167 (N=13, n.s.) |
| Inverse-vol YANK size at MIM=1ct (overlap / full) | **2.1** / 3.2 ct |
| Current live ratio (MIM:YANK) | 1 : 5 (over-weights YANK ~2.4×) |
| Days both strategies traded (overlap) | 18 |

---

## 4. Method (Frozen)

**Combine engine:** identical rules to the sealed single-strategy MC (`study_mim_noise_bands_gate2_mc.py`), applied to **one shared account**:
- Start $50,000. **Pass** = balance ≥ $53,000 (+$3,000) AND best single day < 50% of total profit (consistency rule).
- **MLL floor** starts $48,000; ratchets at END OF DAY to `min(50_000, max(floor, EOD_balance − 2_000))`; breach checked **per-trade against current equity** → BLOW.
- **DLL:** if a strategy's day P&L ≤ −$1,000 after a trade, that strategy deactivates for the day; the account lives. (Applied per-strategy; the shared MLL still sees both.)
- Costs: $2.24/ct per completed trade, already in the net series; scales linearly with contracts.
- 5,000 sims, 90-day cap → STALL if neither pass nor blow. Same RNG discipline (fixed seed).

**Joint bootstrap — the one new design choice, frozen:**
- **Primary (correlation-preserving):** block-bootstrap by **common calendar day** over the overlap window (2025-05-22 … 2026-05-04). **Day pool = the set of ET dates in the overlap window on which at least one strategy traded** (union of traded days); dates where neither traded are excluded, consistent with the single-strategy engine's traded-day sampling. Each sim draws days from this pool with replacement; on a drawn day, **both** strategies contribute their realized trades for that exact ET date (0 if a strategy was flat). Intraday order preserved within each strategy; the two strategies' trades on the drawn day are interleaved by timestamp so the shared floor sees the true within-day path. This preserves the realized ~0 daily correlation and the actual co-occurrence (or not) of fat-tail days.
- **Sensitivity (independence upper bound):** draw each strategy's day **independently** from its own full traded-day pool (MIM 163 days, YANK 48 days). Assumes zero correlation (justified by +0.015) and uses each strategy's full history; the optimistic diversification bound. Within a sim-day there is no shared calendar, so **MIM-NB's trades are processed before YANK's** (disclosed modeling choice; affects only the within-day floor-breach path). Report alongside primary; **the primary governs the decision.**

**Aggregate-floor note (binding):** the trailing floor is computed on **combined** account equity, not per-strategy. Vol-balancing is therefore on combined drawdown — exactly what the joint MC measures. Do not substitute standalone-vol balancing.

**Sizing grid (frozen):** YANK ∈ {1, 2, 3} ct, MIM-NB = 1 ct. The vol-balanced prior is YANK ≈ 2 (overlap inverse-vol 2.1, invvol 2.27) — i.e. the experiment is centered on 1:2 with 1:1 (conservative) and 1:3 (aggressive) as the bracket. No other sizes.

---

## 5. Decision Gate (Frozen, set before any run)

For each frozen size, the **primary** bootstrap yields (pass%, blow%, median days-to-pass).

- **ADOPT** the joint config at the smallest YANK size where **pass% > 54%** (single-strategy baseline) **AND blow% ≤ 33%** (single-strategy baseline). Prefer the smaller size on ties (conservative; Mary's tail-correlation caveat).
- **Tie-break / report:** if multiple sizes qualify, the chosen size is the smallest qualifying one; all three are reported.
- **CLOSE** the joint-on-one-account branch if **no** frozen size satisfies both conditions under the **primary** method (even if the sensitivity arm looks better — the optimistic arm cannot rescue a failing primary).
- **Caveat reporting (mandatory):** report the gap between primary and sensitivity pass% as the diversification-fragility indicator. If sensitivity passes but primary fails, the conclusion is "diversification benefit exists only under independence we cannot guarantee" → still CLOSE.

This gate is about **pass-probability only**. Sharpe, expectancy, and income-smoothness are explicitly out of scope for the decision (objective deferred per §0).

---

## 6. What this does NOT decide

- **Execution path.** YANK is currently a separate TradeStation *paper* account at 5ct; MIM-NB is the only strategy on the real ProjectX combine (23884932). If this experiment says ADOPT, a **separate deployment pre-registration** is required to (a) port YANK execution onto the combine account at the chosen down-sized contract count, and (b) define joint halt triggers on the shared floor. This document does not authorize any live change.
- **Parallel-combine strategy** (run YANK on a second combine account). That is Victor's capped-downside optionality lever and a different bet structure; it needs its own analysis (a 54% single-account pass implies ~79% across two independent accounts, but two fees and two ratchets).

---

## 7. Stopping rule

Run once, at the three frozen sizes, both bootstrap arms. Record pass%/blow%/median-days for each. Apply §5. No re-runs with different seeds, windows, or sizes. If a bug is found, fix, re-seal (new commit), and re-run all six cells.
