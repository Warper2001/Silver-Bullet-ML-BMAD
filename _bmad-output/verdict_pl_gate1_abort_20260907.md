# PL Gate-1 — VERDICT 2026-09-07: ⛔ ABORTED AT STEP 1. HOLDOUT NOT SPENT.

**Seal:** `_bmad-output/preregistration_pl_gate1_holdout.md`, commit `e2a6bda`.
**Instruction:** Alex, "run the platinum holdout test" → "go ahead".
**Outcome:** the sealed protocol's **Step 1 reproduction gate FAILED**, whose stated consequence
is *"ABORT — no holdout access."* That was honoured.
**`data/sealed_holdout/pl_1min_holdout_20260301_plus.csv` was never opened. `ACCESS_LOG.md` has no
new row, because no access occurred.**

---

## 1. What the reproduction gate found

The seal required the in-sample window (2025-05-19 → 2026-02-28, `--instrument pl --structural
--ml-threshold 0.0`) to reproduce the frozen reference `backtest_1year_20260626_025416.csv`:
N=101, gross PF 1.344 (±0.005), gross total +$6,265 (±$5).

| | frozen reference (2026-06-26) | reproduction (2026-09-07) |
|---|---|---|
| N | 101 | **129** |
| gross PF | 1.3440 | **1.2262** |
| gross total | +$6,265.00 | **+$5,295.00** |
| win rate | 47.5% | 46.5% |
| exits | sl=42 time=52 tp=7 | sl=56 time=65 tp=8 |
| max DD (gross) | $3,380.00 | $3,380.00 |

**The signal path is intact.** 100 of the frozen 101 trades reproduce byte-exact on
(entry_time, direction, pnl); one frozen trade (2025-12-05T13:04) is absent; **29 trades are
present now that were not in the frozen list.** Zero shared entry-times disagree on P&L.

## 2. What was ruled out — by evidence, not inference

Each of these was a live hypothesis; each was killed by a specific check, and several of my own
intermediate conclusions were wrong before the right answer appeared. Recorded so the next reader
does not re-walk them.

| hypothesis | how it died |
|---|---|
| Trailing-DD halt removal (`e56bc6a`) — a $2,000 Topstep floor that `STRUCTURAL_OVERRIDES` never neutralizes, plausibly suppressing trades after PL's −$2,560 June 2025 drawdown | **Tested, not assumed.** Restored the pre-`e56bc6a` halt behind an env flag and re-ran: still **N=129, PF 1.226, $+5,295**, identical. The halt only sets `_daily_halted`, which resets each day, and PL trades ~0.5/day — so it removes almost nothing. **This was my initial published root cause and it was wrong.** |
| Data drift (CSV rebuilt/backfilled) | `pl_1min_2025_2026.csv` mtime is **2026-06-12 17:46**, predating the frozen run (06-26). Same file, and both runs load the same 206,083 bars over the same range. |
| Gap-ceiling denomination change (`71978d0`) | Opt-in. `max_gap_atr_ratio` still defaults to `0.0` at HEAD, so `detect_fvg` takes the unchanged `max_gap_dollars` branch — and structural mode sets that to 1e12 (ceiling off) in both eras. |
| `BacktestEngine` m15_confirmation tz fix (`600a1fd`) | Different engine. This harness drives `Tier2StreamingTrader`, not `strategy_core.BacktestEngine`. |
| Trade-log persistence fix (`68225eb`) | `append_trade()` is called *after* the trade is already appended to `completed_trades`; results come from that in-memory list. It cannot drop a trade. |
| Concurrency / one-position-at-a-time gate | **0 of 29** extra entries fall inside a frozen trade's hold window. |
| Day-of-week gating | Extras spread Mon 9 / Wed 11 / Thu 3 / Fri 5 / **Tue 1** — no weekday gate shape. |

## 3. The decisive test

`git archive cc17543` (the commit that *sealed the fan-out*, the repo state at the frozen run's
own timestamp) was extracted to a clean tree with the data symlinked and writes isolated, and the
identical command was run:

> **Frozen-era code returns N=129, gross PF 1.226, +$5,295 — byte-identical to HEAD.**

The two trade lists are identical as files: `diff` of
`data/reports/pl_frozen_era_cc17543_20260907.csv` (cc17543) against
`data/reports/backtest_1year_20260907_025458.csv` (HEAD) reports **no differences at all**. Both
are committed as evidence.

**There is no code drift.** The frozen 101-trade PL reference is **not reproducible from any
committed state of this repository**. It was produced on 2026-06-26 02:54 by working-tree code,
configuration, or arguments that were never committed, and no record of what they were survives.

## 4. Corrected in-sample economics, and why this closes the test

Recomputing the cost ceiling from the **reproducible** trade list (`pl_gate1_net_analysis.py`,
validated against the parent seal's own published numbers before use):

| | frozen (N=101, unreproducible) | **corrected (N=129, reproducible)** |
|---|---|---|
| gross PF | 1.3440 | 1.2262 |
| gross avg / trade | +$62.03 | +$41.05 |
| net PF @ measured $34.00/RT | 1.1411 | **1.0352** |
| net $/trade @ $34.00 | +$28.03 | **+$7.05** |
| net PF @ $44.00 sensitivity | 1.0882 | **0.9857** (negative) |
| ex-top-3-days net @ $34.00 | −$5,159 | **−$7,081** |
| **c\* for net PF ≥ 1.10** | **$41.71/RT** | **$21.73/RT** |
| c\* for net PF ≥ 1.00 | $62.03/RT | $41.05/RT |

The parent slippage seal's PASS was exactly `measured $34.00 ≤ ceiling $41.71`. **Against the
reproducible fingerprint the ceiling is $21.73 and the measured cost is $34.00 — the slippage gate
FAILS by a wide margin.** PL does not clear the precondition that authorized this Gate-1 at all,
so the holdout question never legitimately arises.

The top-3 days are *identical* in both lists (+$4,252, +$2,352, +$1,386). The 29 recovered trades
are pure drag: they add ~28% more trades and *reduce* gross P&L by $970.

## 5. Retroactive impact on the parent chain

- **`pl_slippage_verdict_20260705.md` (PASS) is VOID as an authorization.** Its thresholds
  (`c* = $41.71`, breakeven `$62.02`) were derived from the unreproducible 101-trade list. The
  measurement itself (pooled median spread $30.00 → all-in $34.00/RT over 26,158 samples) is
  untouched and remains good evidence; only the ceiling it was compared against was wrong.
- **`pl_combine_fit_verdict_20260705.md` (FAIL) stands, and is if anything strengthened.** It
  replayed the 101-trade path, but the corrected path has the same worst single trade
  (−$1,914) and a *larger* net max drawdown ($5,644 vs $4,890).
- **The `--structural` mode's own claim needs an audit.** It advertises neutralizing "dollar-scaled
  / path-dependent gates," but it demonstrably does not neutralize the Topstep trailing floor
  (`check_trailing_dd`, tier2_streaming_working.py:1375, `topstep_trailing_dd_amount` default
  $2,000). That turned out not to explain this divergence, but it is a real gap in a mode whose
  entire purpose is scale-invariance, and it silently applies to every cross-instrument run.
- **Every instrument in the fan-out is suspect for the same reason.** SI/YM/RTY/HG/ES/GC/PL
  fingerprints all came from that same uncommitted working tree. A full re-run on current code is
  in flight; results land in `verdict_fanout_rerun_20260907.md`.

## 6. Verdict

**PL Gate-1: ABORTED. No holdout access. The sealed slice remains unspent and one-shot.**

PL is **not** closed by a holdout FAIL — it never reached one. It is closed at the precondition:
*at the measured cost of $34.00/RT, the reproducible in-sample structural fingerprint yields net
PF 1.035, below the 1.10 bar its own slippage seal required, and below it at a cost ceiling
($21.73) far under what is actually paid.* Reopening PL would require a new hypothesis and a new
seal, not a re-run of this one.

## 7. The transferable lesson

A pre-registration's reproduction gate is not ceremony. This one cost ~2 hours of compute and it
caught a case where **the frozen artifact that motivated an entire measurement campaign, a
slippage PASS, a combine-fit gate and a holdout authorization could not be regenerated by the
repository at all.** Had the gate been skipped — or had reproduction been "fixed" by hunting for a
config that reproduces 101 — a one-shot sealed holdout would have been spent validating a number
no committed code produces.

**Rule to carry forward:** a frozen reference is only frozen if a committed SHA regenerates it.
Freeze the *command and the SHA*, and verify regeneration at freeze time — not months later when
something else depends on it.
