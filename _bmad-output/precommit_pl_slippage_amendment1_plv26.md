# Amendment 1 to PL Slippage Pre-Commitment — binding symbol PLV26

**Date:** 2026-07-05 (DRAFT — must be committed/sealed BEFORE the amended analysis is run)
**Parent:** `_bmad-output/precommit_pl_slippage_measurement_2026-06.md` (sealed 2026-06-26, commit `cc17543`)
**Reason:** the parent's binding instrument (PLN26, July) died into delivery mid-capture; this corrects a contract-selection error. No threshold, window, estimator, or evidence requirement changes.
**Sibling precedent (verbatim structure):** `precommit_hg_slippage_amendment1_mhgu26.md` (copper, commit `0861e2f`, sealed 2026-07-04).

## Verbatim outcome of the parent rule (recorded first, for the record)

`analyze_pl_quotes.py` (parent, unmodified) run 2026-07-05 on the full capture (2026-06-26 → 2026-07-03, 6 qualifying RTH sessions ≥ 3,000 valid samples each):

- Binding symbol stayed **PLN26** (roll clause did NOT trigger: sample counts remained exactly equal — PLN26 26,158 vs PLV26 26,158 — because the dead contract kept streaming stale quotes).
- PLN26 per-session median spread widened monotonically as it went to delivery: $45 (9t) 06-26 → $85 (17t) 06-29 → $385 (77t) 06-30 → $345 (69t) 07-01 → $535 (107t) 07-02 → $540 (108t) 07-03. **Pooled all-in $369.00/RT (73 ticks).**
- **Formal result: FAIL** under the parent rule (pooled all-in $369.00 > $41.71; worst session $544 > the $62.02 breakeven guard).

**Classification: INVALID MEASUREMENT, not cost evidence.** The July full-size platinum contract (PLN26) entered its delivery/expiry window during the capture period and ceased to be a traded market; its quotes are stale placeholder values, visibly non-market (a spread that balloons 9→108 ticks and never mean-reverts, at 1×1 size). The parent's roll clause keyed on *sample density*, which cannot detect this failure mode — a dead contract still answers the quote endpoint. This is the **identical artifact that hit the copper capture (MHGN26)**; the lesson was banked in the HG Amendment 1 seal: **detect rolls by quote-staleness/spread-sanity, never by sample count, and never bind a slippage measurement on a near-expiry front month.**

## Amendment (single change)

- **Binding symbol: PLV26** (October full-size platinum — the actual front/liquid contract for the whole capture window; it was captured in parallel from day one under the parent protocol as the designated context/roll target).
- Everything else is inherited **unchanged and un-retuned** from the parent seal:
  valid-sample definition, RTH window 09:30–15:55 ET, qualifying session ≥ 3,000 valid samples, minimum 5 qualifying sessions, `c = pooled_median_spread + $4.00`, **PASS iff c ≤ $41.71/RT (8.3 ticks) AND every qualifying session's all-in median ≤ $62.02/RT** (the pure-breakeven guard). These thresholds were derived from the frozen PL trade list on 2026-06-26, before any quote was captured — the amendment cannot and does not touch them.
- Analysis performed only by `analyze_pl_quotes_amendment1.py` (to be committed with this seal) — a copy of the frozen analyzer with the binding symbol set to PLV26 and the roll clause removed (no further roll is possible; capture auto-stops 2026-07-08, and October is the front once July delivers). One run; no second attempt regardless of outcome.

## Contamination disclosure

- The parent analyzer's non-binding context line already printed PLV26's **pooled** RTH median: **$30.00 (6.00 ticks)** → all-in $34.00/RT — so this amendment is written knowing the *pooled* number likely passes ($34.00 ≤ $41.71). Mitigations, identical to the copper amendment: (a) the thresholds pre-date all data and are untouched; (b) the binding-symbol switch is *forced by instrument death*, not chosen among candidates (PLV26 was the pre-designated roll target in the parent seal); (c) the **per-session PLV26 medians and the session-level breakeven guard have NOT been examined at seal time** — they can still fail the measurement, and platinum's spread was noted in the parent seal as more variable than copper's, so this guard is a live risk here (unlike copper, which was dead-stable at 2 ticks).

## Lowered-prior disclosure (specific to this candidate)

This amendment is authored **after copper — the cleaner sibling candidate — failed its Gate-1 holdout** (`328cdaf`: N=26, gross PF 0.563, net PF 0.463; the signal did not transfer out-of-sample, war-regime window). Copper's failure is direct evidence *against* the cross-instrument portability thesis and **lowers the prior** that platinum's frozen structural edge will survive its own holdout. This seal clears **only the slippage gate**; it does not raise the portability prior. The downstream Gate-1 holdout decision rule stands on its own pre-registered thresholds, but a rational reader should weight a platinum PASS-then-holdout as a lower-base-rate bet than copper was, and platinum additionally carries an **unresolved combine-fit gate** (below) that copper did not. Pursued at Alex's explicit direction with this understood.

## Consequences (inherited verbatim from the parent)

- **PASS** → authorizes **WRITING** a Gate-1 pre-registration for the frozen PL structural strategy on its sealed holdout (`data/sealed_holdout/pl_1min_holdout_20260301_plus.csv`, protected since 2026-06-12, UNTOUCHED), using the **measured** PLV26 spread as cost basis (not the old 13-tick assumption). The holdout run itself and any deployment remain **separate gates requiring Alex's explicit go**. Nothing deploys from this measurement; no holdout data is touched by it.
- **Combine-fit is a SEPARATE downstream gate (inherited):** full-size PL is 50 troy oz, **$50/pt, $5/tick, no CME micro**; contract notional ≈ $78K vs the 50K Topstep account, and per-trade SL $-risk on the frozen SL2×-gap exits must be checked against the daily-loss / trailing-DD limits. A slippage PASS does NOT clear this; it must be evaluated before any Gate-1 holdout prereg is finalized.
- **FAIL** → PL reclassified gross-only; no holdout spent; logged as "orthogonal structural edge real but not cost-survivable at full-contract size given platinum illiquidity."
- Context stats reported but non-binding: p75/p90 spread, % at ≤ 8 ticks, median bid/ask sizes (1-lot adequacy), the non-binding PLN26 comparison, spread by hour.

One run of the amended analysis; no second attempt regardless of outcome.
