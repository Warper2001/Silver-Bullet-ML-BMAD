# HG (Copper) Slippage Decision — VERDICT 2026-07-04

**Parent seal:** `precommit_hg_slippage_measurement_2026-06.md` (2026-06-26, commit 8f33f12 — thresholds derived from the frozen HG trade list before any quote was captured).
**Amendment 1 seal:** `precommit_hg_slippage_amendment1_mhgu26.md` (commit `0861e2f`, sealed 2026-07-04 BEFORE the amended run; binds MHGU26 after MHGN26 died into delivery mid-capture).
**Analyzers:** `analyze_hg_quotes.py` (parent, run verbatim first) and `analyze_hg_quotes_amendment1.py` (single amended run).
**Data:** `data/quotes/hg_quote_capture.csv`, 2026-06-26 → 2026-07-03, 6 qualifying RTH sessions (≥3,000 valid samples each; minimum was 5).

---

## Verdict chain

**1. Parent rule, verbatim: FAIL — but INVALID MEASUREMENT.** Binding MHGN26 (July) went to delivery mid-capture and streamed a frozen constant $663.75 / 531-tick spread (p75 = median, 1×1 size) for four straight sessions. The sample-count roll clause could not fire — a dead contract still answers the quote endpoint. Recorded as a contract-selection error, not cost evidence. **Lesson banked: detect rolls by quote-staleness/spread-sanity, never sample density; never bind a slippage measure on a near-expiry front.**

**2. Amendment 1 (MHGU26, thresholds inherited unchanged): 🟢 PASS.**

| Session | n | median spread | all-in | p75 |
|---|---|---|---|---|
| 2026-06-26 | 4,346 | $2.50 | $4.00 | $2.50 |
| 2026-06-29 | 4,359 | $2.50 | $4.00 | $2.50 |
| 2026-06-30 | 4,364 | $2.50 | $4.00 | $2.50 |
| 2026-07-01 | 4,359 | $2.50 | $4.00 | $2.50 |
| 2026-07-02 | 4,362 | $2.50 | $4.00 | $2.50 |
| 2026-07-03 | 4,367 | $3.75 | $5.25 | $106.25 |

- Pooled (n=26,157): **median $2.50 = 2.00 ticks → all-in $4.00/RT ≤ $4.63 ✅**
- Session guard: all all-in medians ≤ $5.63 (worst $5.25) ✅
- Depth context: 80.8% of samples at ≤2 ticks; median book 3×3 — adequate for 1-lot.

**Caveats (recorded, non-blocking):**
- 07-03 was the July-4th half-day: COMEX metals closed early, but capture polled to 15:55 ET, so that session's afternoon contains post-close widened/stale quotes (hence the $106.25 p75 and the 3-tick median). It still passed the frozen guard, and the frozen rule counts it; five normal sessions were dead-stable at exactly 2 ticks.
- The margin is thin by construction: $4.00 vs $4.63 ceiling. At measured cost, the frozen exploratory HG result nets **PF 1.151, +$2.63/trade** (from the parent's pre-computed cost curve). This is a "worth one holdout shot" edge, not a fat one.
- Amendment contamination disclosed in the seal: the parent's context line had already printed MHGU26's pooled median before the amendment was written; thresholds pre-date all data and the per-session guard had not been examined at seal time (and genuinely could have failed on 07-03 — it came within $0.38 of the guard).

## Consequence (per the sealed rule)

**Authorized: WRITE a Gate-1 pre-registration for the frozen HG structural strategy on its sealed holdout** (`data/sealed_holdout/hg_1min_holdout_20260301_plus.csv`), using the **measured** cost basis (all-in $4.00/RT median; sensitivity at $5.25 worst-session) — not the old assumption. The holdout run itself and any deployment remain separate gates requiring Alex's explicit go. Nothing is deployed by this verdict; no holdout data has been touched.

Platinum (PLV26) hit the same dead-front artifact (PLN26) and by the 07-03 preliminary read would pass its own bar — it would need its own amendment seal before a formal verdict; separate decision, plus the unresolved combine-fit question (full 50oz contract, ~$78K notional).
