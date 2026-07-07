# Pre-Registration: MIM-NB Cat-Stop 250-vs-500 Prospective Shadow Ledger

**Generated:** 2026-07-06
**Experiment ID:** mim-nb-catstop-shadow-ledger
**Type:** Knowledge-only prospective test. NO live config change. `CAT_STOP_PTS = 250`
stays deployed for the duration; any change afterward requires a NEW prereg.
No sealed holdout data is accessed or spent.

**Status:** SEALED 2026-07-07 — Alex chose **Design B (Mary's fixed-N=10 dual
criterion)**. Design A (SPRT) is struck and may not be revisited for this experiment.
The event clock starts at the first genuine (broker-verified) cat-stop AFTER this
commit. Ledger tool: `tools/mim_catstop_shadow_ledger.py`.

---

## 0. Motivation — the derivation debt

Provenance of the current 250pt cat-stop:

- Prereg `6957daa` ran the honest 2-arm sweep: **250pt FAILED Monte Carlo**
  (42.3% combine pass, OOS PF → 1.09); **500pt PASSED** (54%). 500 was deployed.
- Prereg `30bc6a8` (2026-06-25) cut 500 → 250 on an **asserted** rule
  ("max single-trade loss ≤ 25% of DD budget") — a derive-don't-assert violation
  acknowledged at the 2026-07-06 mechanics roundtable. That seal also disclosed
  that 500's MC advantage came largely from truncating one dev-period outlier.
- Retrospective counterfactual (2026-07-06, all 3 live cat-stop events:
  06-24, 06-25\*, 07-02): 250-arm total −$2,000 vs 500-arm −$2,649; **0/3
  recoveries** between 250 and 500 pts. Favors 250, but retrospective, N=3,
  and \*06-25 is approximate (bot offline, bar gaps, widened EOD fallback).

The roundtable verdict: keep 250 live, but repay the derivation debt with a
**prospective, pre-registered** test rather than leaving the choice asserted.

## 1. Question

On days when a live MIM-NB position's adverse excursion reaches ≥250 pts, does
price recover often enough before the 500pt level / EOD that a 500pt cat-stop
outperforms the 250pt cat-stop?

## 2. Why 250 stays live during the test (observatory argument)

The 250 arm is the **unbiased observatory**: with 250 live, every excursion
≥250 pts is observed and the 500-arm outcome is computable from recorded bars.
With 500 live, days where price recovers between 250 and 500 would be censored
(no event logged at 250). Bars are exogenous to our fills (1ct MNQ does not move
the market), so the counterfactual arm is deterministic given recorded bars.

## 3. Event definition and shadow-ledger mechanics (both designs share this)

- **Event:** a live MIM-NB entry whose adverse excursion from entry reaches
  ≥250 MNQ pts (i.e., the live cat-stop fires, per `data/mim_nb/trades.csv`
  exit reason `CAT_STOP*`). Offline/gap days (like 06-25) COUNT as events but
  are flagged `approximate` and excluded from the primary criterion if bar
  coverage <90% of the entry→EOD window.
- **250-arm outcome:** the realized live trade PnL (−$500/ct + slippage as logged).
- **500-arm outcome:** computed from `data/mim_nb/bars_raw.csv` (schema
  `ts_utc,open,high,low,close,volume,received_at,chain`, hash-chained) by walking
  bars from the live entry: exit at the first bar whose low (long) crosses
  entry−500pts → −$1,000/ct; else if session DLL-equivalent (−$1,000 at 500-arm
  scale, per 6957daa's paired constants) triggers → that; else EOD close at the
  15:30 ET mark (fallback: last bar within 240 min before, flagged approximate).
- **Recovery:** a 500-arm outcome strictly better than the 250-arm outcome.
- **Ledger:** append one row per event to `data/mim_nb/shadow_catstop.csv`
  (`date,entry_px,side,arm250_pnl,arm500_pnl,recovered,approximate,chain`) —
  written by an offline tool run after each event day, never by the live bot.
- **Clock starts:** first event AFTER the seal commit. The retrospective
  events are context only and do NOT count.
- **Event authenticity (added 2026-07-07):** an event counts ONLY if the broker-side
  stop order actually filled (verify via ProjectX Trade/search, as in the halt-review).
  The 2026-07-06 "cat-stop" was an EXTERNAL FLATTEN (order #3229490103; the bot's stop
  was canceled unfilled) — it is excluded, making the genuine retrospective count 3
  (06-25≈, 07-02, 07-07), not 4. Externally-closed positions are logged in the shadow
  ledger with `recovered=NA, approximate=true` and excluded from both designs' counts.

## 4. Sealed design — Design B chosen by Alex 2026-07-07

*(Design A — Quinn's SPRT — was struck at seal time per the pick-one rule. Its spec
is preserved in git history at draft commit cf4975c for provenance; it is not
available for this experiment.)*

### Design B — Mary's fixed-N dual criterion (CHOSEN)

- Fixed N = 10 non-approximate events; no interim looks.
- Revert-to-500 requires BOTH:
  1. **Paired delta:** Σ(arm500 − arm250) > 0 over the 10 events; AND
  2. **MC re-run:** the 6957daa Monte Carlo re-executed on the pooled
     (dev + live-to-date) trade set with the dev outlier winsorized at the
     99th percentile — the 500 arm's combine pass-rate must exceed the 250
     arm's by ≥ 5 percentage points.
- Any other result → 250 confirmed; ledger continues passively for context.
- Recommended because it is bounded, has no sequential-peeking machinery, and
  matches the project's fixed-N gate style.

At current event frequency (3 events in ~25 calendar days of live trading),
N = 10 projects to roughly 2–4 months.

## 5. Outcome handling

- This experiment produces KNOWLEDGE ONLY. Regardless of verdict, `CAT_STOP_PTS`
  changes require a new pre-registration commit citing this ledger.
- If MIM-NB halts permanently (combine blown/passed) before N is reached, the
  test ends INCONCLUSIVE at whatever N stands; partial results are reportable
  but trigger no action.

## 6. Integrity hashes

| Hash | Value |
|---|---|
| (a) `src/research/mim_nb_live.py` SHA-256 | `e2a9b70399684a48176dd61be438d0bf8f90a11f8d9525fe6142c2cdb7e200d2` |
| (b) Git HEAD at draft time | `22a9ea4cfefef9bac25c5781156dad9a48a85232` |
| (c) `bars_raw.csv` schema | `ts_utc,open,high,low,close,volume,received_at,chain` (hash-chained rows) |

*Commit this document (with §4 choice made) to make it tamper-evident.*
