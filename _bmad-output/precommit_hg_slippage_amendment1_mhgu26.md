# Amendment 1 to HG Slippage Pre-Commitment — binding symbol MHGU26

**Date:** 2026-07-04 (sealed before the amended analysis is run)
**Parent:** `_bmad-output/precommit_hg_slippage_measurement_2026-06.md` (sealed 2026-06-26, commit 8f33f12)
**Reason:** the parent's binding instrument died mid-capture; this corrects a contract-selection error. No threshold, window, estimator, or evidence requirement changes.

## Verbatim outcome of the parent rule (recorded first, for the record)

`analyze_hg_quotes.py` run 2026-07-04 on the full capture (2026-06-26 → 2026-07-03):

- Binding symbol stayed **MHGN26** (roll clause did NOT trigger: sample counts remained ~equal because the dead contract kept streaming stale quotes).
- MHGN26 per-session median spread: $23.13 on 06-26, then a **frozen constant $663.75 (531 ticks, p75 = median, bid 1 × ask 1) for every session 06-29 → 07-03**. Pooled all-in $665.25/RT.
- **Formal result: FAIL** under the parent rule.

**Classification: INVALID MEASUREMENT, not cost evidence.** The July micro-copper contract (MHGN26) entered its delivery/expiry window during the capture period and ceased to be a traded market; its quotes are stale placeholder values, visibly non-market (a constant 531-tick spread with the p75 equal to the median for four straight sessions). The parent's roll clause keyed on *sample density*, which cannot detect this failure mode — a dead contract still answers the quote endpoint. **Lesson (logged for all future slippage precommits): detect rolls by quote-staleness/spread-sanity, never by sample count, and never bind a slippage measurement on a near-expiry front month.** Same artifact hit the PL capture (PLN26).

## Amendment (single change)

- **Binding symbol: MHGU26** (September micro copper — the actual front/liquid contract for the whole capture window; it was captured in parallel from day one under the parent protocol as the designated context/roll target).
- Everything else is inherited **unchanged and un-retuned** from the parent seal:
  valid-sample definition, RTH window 09:30–15:55 ET, qualifying session ≥ 3,000 valid samples, minimum 5 qualifying sessions, `c = pooled_median_spread + $1.50`, **PASS iff c ≤ $4.63/RT AND every qualifying session's all-in median ≤ $5.63/RT**. These thresholds were derived from the frozen HG trade list on 2026-06-26, before any quote was captured — the amendment cannot and does not touch them.
- Analysis performed only by `analyze_hg_quotes_amendment1.py` (this commit) — a copy of the frozen analyzer with the binding symbol set to MHGU26 and the roll clause removed (no further roll is possible; capture auto-stops 2026-07-08, and Sept is the front).

## Contamination disclosure

- The parent analyzer's non-binding context line already printed MHGU26's **pooled** RTH median: $2.50 (2.00 ticks) — so the amendment is written knowing the pooled number likely passes. Mitigations: (a) the thresholds pre-date all data and are untouched; (b) the binding-symbol switch is forced by instrument death, not chosen among candidates (MHGU26 was the pre-designated roll target in the parent seal); (c) the per-session medians and the session-level breakeven guard — which can still fail the measurement — have NOT been examined at seal time.
- The 2026-07-03 session in this repo's memory recorded a preliminary "back-month passes" read. This amendment formalizes that contingency under seal rather than acting on it informally.

## Consequences (inherited verbatim)

- **PASS** → authorizes WRITING a Gate-1 pre-registration for HG on its sealed holdout (`data/sealed_holdout/hg_1min_holdout_20260301_plus.csv`) using the measured spread as cost basis. Nothing deploys; the holdout run and any deployment remain separate gates requiring Alex's go.
- **FAIL** → HG reclassified gross-only; no holdout spent; portability claim logged as "structural edge real but not cost-survivable at micro size."
- One run of the amended analysis; no second attempt regardless of outcome.
