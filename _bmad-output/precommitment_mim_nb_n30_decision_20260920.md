# Pre-Commitment: how MIM-NB's N=30 will be read, written before trade 30

**Date:** 2026-09-20. **Status:** committed BEFORE trade 30 lands (28 completed trades at writing).
No parameter, threshold, config, unit file, or sizing changes. The sealed triggers —
`30 completed trades with net PF < 0.70` and `account equity ≤ $48,400`
(`preregistration_mim_nb_live_deployment.md` §4) — are unchanged and are not re-derived here.
This document only fixes **which number counts and how each outcome is read**, so neither is decided
after seeing trade 30. Format follows `precommitment_gap_fade_n30_decision_20260917.md`.

**Evidence base:** `_bmad-output/diagnostics_mim_nb_n30_race_20260920/` at commit `05e468c`
(`analyze.py`, `results.json`, `REPORT.md`). It corrects the 09-17 diagnostic, whose N=26 / PF 0.689
was a mis-windowed ledger; the true state is **N=28, net −$753.00, PF 0.848**.

## 1. What counts as "the N" — fixed now

1. **The counter (D1)** is MIM-NB's completed round trips in `data/mim_nb/trades.csv` since
   2026-06-11: one row = one trade; "trade 30" is the 30th row. It includes every row already there
   (retired-account rows and `EXTERNAL_*` closes: they were real fills and are strategy evidence,
   not funding evidence). **YANK trades are excluded. An account reset does not reset it.**
   `trades.db` with `write_mode='realtime'` is **not** the counting rule — it drops the first two
   trades.
2. **The PF of record** is Σwins / Σ|losses| of `pnl_usd` in that ledger (gross of commission, as the
   sealed reference is). A fee-adjusted PF is reported beside it. **If either is < 0.70 the trigger is
   treated as fired** — conservative, and no tolerance band is introduced.
3. **Evaluation semantics.** The sealed text is silent on whether N=30 is a single look. It is read
   as *at N=30 and at every later completed trade, the first time PF < 0.70* — the same semantics
   `combine_floor_monitor.py` implements. This is an interpretation, disclosed here; it gives the rule
   more chances to fire, never fewer.
4. **The monitor's PF trigger (D3)** — MIM-NB + YANK rows in `trades.db` since the current account's
   start, report-only — is a *separate tripwire with its own N* (12 today, PF 0.447). It neither counts
   toward D1 nor overrides it. Never quote one as the other.

## 2. What the sealed D1 rule can actually do at N=30 (from the diagnostics)

- To fire at N=30 the next two trades must lose **> $1,048.29 combined (> $524 each)**. The live
  cat-stop caps a stop-out at $500; two full cat-stops give PF **0.7057** — still above 0.70.
  P(fires exactly at N=30) = **0.4% / 0.0% / 0.0%** (sealed-500pt ref / truncated ref / live-250pt).
  Expected arrival of trade 30: **~2026-09-26** (5–95%: 09-21 → 10-07).
- On a fresh 30-trade sample the rule trips **12.9%** of the time when the sealed net edge (+$31.99)
  is real, 25.6% at zero edge, 42.1% at −$31.99, and misses a −$51/trade strategy **47%** of the time.
  It is a weak discriminator in both directions.

## 3. Reading at trade 30 — every branch, decided now

| Ledger PF of record | Reading |
|---|---|
| **< 0.70** (either gross or fee-adjusted) | **The sealed trigger fires as written.** Same session: report it to Alex with this table and the ledger. The recommendation on record is the sealed one — halt MIM-NB, log why, then review (§4: "halt the bot first"). Halting is Alex's action (no automation exists; AGENTS.md requires asking). **Continuing is an explicit logged override, not "the rule did not apply."** 0.70 is not re-derived afterwards. |
| **≥ 0.70** (≈ 99.6% likely) | **No halt — and not evidence.** (a) Passing a rule that trips a sealed-edge strategy 12.9% of the time and misses a −$51/trade one 47% of the time proves nothing. (b) Seal §5's upgrade needs PF *tracking the OOS 1.30*; 0.848 does not. **The upgrade is not declared at N=30 under any outcome that does not track it,** and "passed N=30" is never cited as validation. The any-time evaluation in §1.3 simply continues. |

Any future claim that MIM-NB's edge is *confirmed* needs a power calculation on the live 250-pt
config's own trade list first. None exists today: the OOS 1.30 reference is the 500-pt variant
(`preregistration_mim_nb_catstop_250.md`: "a new OOS benchmark requires a fresh backtest… not
performed"). This document adopts no confirmatory N; that number must be derived, not set here.

## 4. Capital events — independent of the edge question

Live at writing: equity $48,917.42; MLL floor $48,298.96 (buffer $618.46); sealed equity line
$48,400 is **$517.42** away; one cat-stop lands at $48,417.42. From the diagnostics, P(touch $48,400 /
touch the floor) within 2 trades ≈ 8–18% / 4–14%; within 10 trades ≈ 30–61% / 25–56% (i.i.d.,
MIM-NB only, YANK and DLL not modelled).

1. **Equity ≤ $48,400** is a *capital event*, reported the same session. It is **not** an edge
   verdict, does **not** count as D1 firing, and D1 firing does not depend on it. Deployment §4 makes
   it a halt-and-review trigger; the review is Alex's, and a decision to continue is recorded as an
   override. (Halting authority was removed 2026-07-29 and the monitor is report-only since
   2026-08-04 — both accepted; this does not reopen them.)
2. **Floor breach.** Only the Topstep dashboard is authority; `canTrade` is not (it stayed `true`
   through the 07-06 breach and 38 days beyond — memory `project-combine-blown-20260706`). Any breach
   signal is treated as a breach until Alex confirms otherwise; the burden of proof runs the other way.
   On confirmation: recommend halting MIM-NB and YANK (asking first, per AGENTS.md); follow the reset
   checklist in that memory (`PROJECTX_ACCOUNT_ID` in every unit, `COMBINE_EPOCH_START_FALLBACK` in
   `mim_nb_live.py`; `floor_state.json` self-archives — never hand-edit it). **D1 continues across the
   reset; D3's window resets by design.** Whether to buy another combine is Alex's decision with no
   default here — the vehicle question is open.

## 5. Disclosures carried into any reading

- The 2026-09-15 −$355 trade was the AUTOROLL contamination bug (fixed and deployed 09-17, merge
  `1ae682e`). D1 includes it; report the PF with and without it.
- D1's PF rests on four trades: the 07-29→08-04 run is +$2,279.50; without it D1 is N=24, net
  −$3,032.50, PF 0.388. Current account 26556101, MIM-only: N=10, −$1,441.50, PF 0.284. These are
  descriptive splits, not thresholds or selection rules; report them beside the PF of record.
- The sealed reference (500-pt, gross) is not the live config (250-pt). The truncated reference is
  optimistic and the live-config sample (N=25) is the thing under test.
- `data/mim_nb/decisions.csv` and `orders.csv` hash chains are broken (2026-06-29, 2026-07-29 —
  `project-mim-nb-chain-breaks-20260919`); `trades.csv` is clean and is the counter's source.

## 6. What this document does not do

It does not reopen the 2026-09-06 "brake stays OFF" decision, propose gating, change a threshold,
touch a unit or config, or claim anything about MIM-NB's edge. **1 MNQ is the smallest contract, so
no sizing option exists — only on or off.** GAP-1's N=30 reading is separate and stands unchanged
(`precommitment_gap_fade_n30_decision_20260917.md`).
