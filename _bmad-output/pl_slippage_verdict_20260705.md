# PL (Platinum) Slippage Decision — VERDICT 2026-07-05

**Parent seal:** `precommit_pl_slippage_measurement_2026-06.md` (2026-06-26, commit `cc17543` — thresholds derived from the frozen PL trade list before any quote was captured).
**Amendment 1 seal:** `precommit_pl_slippage_amendment1_plv26.md` (commit `0934500`, sealed 2026-07-05 BEFORE the amended run; binds PLV26 after PLN26 died into delivery mid-capture).
**Analyzers:** `analyze_pl_quotes.py` (parent, run verbatim first) and `analyze_pl_quotes_amendment1.py` (single amended run).
**Data:** `data/quotes/pl_quote_capture.csv`, 2026-06-26 → 2026-07-03, 6 qualifying RTH sessions (≥3,000 valid samples each; minimum was 5).

---

## Verdict chain

**1. Parent rule, verbatim: FAIL — but INVALID MEASUREMENT.** Binding PLN26 (July) went to delivery mid-capture and streamed a spread that ballooned $45 (9t) → $540 (108t) monotonically over the week, never mean-reverting, at 1×1 size; pooled all-in $369.00/RT (73t). The sample-count roll clause could not fire — the dead contract held exact sample parity with the live one (26,158 vs 26,158) because it kept answering the quote endpoint. Recorded as a contract-selection error, not cost evidence. Identical artifact to copper's MHGN26; lesson already banked (detect rolls by quote-staleness/spread-sanity, never sample density).

**2. Amendment 1 (PLV26, thresholds inherited unchanged): 🟢 PASS.**

| Session | n | median spread | all-in | p75 |
|---|---|---|---|---|
| 2026-06-26 | 4,348 | $35.00 (7.0t) | $39.00 | $40.00 |
| 2026-06-29 | 4,357 | $30.00 (6.0t) | $34.00 | $35.00 |
| 2026-06-30 | 4,361 | $30.00 (6.0t) | $34.00 | $35.00 |
| 2026-07-01 | 4,361 | $30.00 (6.0t) | $34.00 | $30.00 |
| 2026-07-02 | 4,363 | $35.00 (7.0t) | $39.00 | $40.00 |
| 2026-07-03 | 4,368 | $40.00 (8.0t) | $44.00 | $630.00 |

- Pooled (n=26,158): **median $30.00 = 6.00 ticks → all-in $34.00/RT ≤ $41.71 ✅**
- Session guard: all all-in medians ≤ $62.02 (worst $44.00) ✅
- Depth context: 83.3% of samples at ≤8 ticks; median book 1×1 (thin — see caveats).

**Caveats (recorded, non-blocking):**
- 07-03 was the July-4th half-day: COMEX metals closed early, but capture polled to 15:55 ET, so that session's afternoon contains post-close widened/stale quotes (hence the $630 p75). The session *median* still held at 8t/$44 all-in and passed the frozen guard; five normal sessions were stable at 6–7t. Same artifact copper hit on 07-03.
- **Median book is 1×1** — platinum is genuinely thin. The spread passes, but a 1-lot fill at the quoted spread is the *best* case; larger size or fast markets would slip worse. This is the "liquidity mirage" risk flagged in the parent seal; the measurement clears the spread bar but does not prove depth beyond 1 lot.
- Margin vs ceiling: all-in $34.00 vs $41.71 ceiling — **more headroom than copper had** ($4.00 vs $4.63). At measured cost the frozen exploratory PL result nets **PF ≈ 1.141, +$28.03/trade** (gross avg $62.03/trade − $34.00, per the parent's pre-computed cost curve; N=101).
- Amendment contamination disclosed in the seal: the parent's context line had already printed PLV26's pooled median ($30.00) before the amendment was written; thresholds pre-date all data and the per-session guard had not been examined at seal time.

## Two disclosures carried from the seal (do not drop these)

1. **Lowered portability prior.** This PASS was produced *after* copper — the cleaner sibling candidate — **failed its Gate-1 holdout** (`328cdaf`: N=26, gross PF 0.563, net PF 0.463, signal did not transfer OOS). Copper's failure is direct evidence against the cross-instrument thesis. This verdict clears **only the slippage gate**; it does not raise the odds that PL's frozen structural edge survives its own holdout. Treat a PL holdout as a lower-base-rate bet than copper was.
2. **Combine-fit is an unresolved SEPARATE gate.** Full-size PL is 50 troy oz, $50/pt, $5/tick, no CME micro; contract notional ≈ $78K vs the 50K Topstep account, and per-trade SL $-risk on the frozen SL2×-gap exits must be checked against the daily-loss / trailing-DD limits. A slippage PASS does **not** clear this — it must be evaluated *before* any Gate-1 holdout prereg is finalized.

## Consequence (per the sealed rule)

**Authorized: WRITE a Gate-1 pre-registration for the frozen PL structural strategy on its sealed holdout** (`data/sealed_holdout/pl_1min_holdout_20260301_plus.csv`, protected 2026-06-12, UNTOUCHED), using the **measured** cost basis (all-in $34.00/RT median; sensitivity at $44.00 worst-session) — not the old 13-tick assumption. **Before finalizing that prereg, the combine-fit gate (above) must be evaluated.** The holdout run itself and any deployment remain separate gates requiring Alex's explicit go. Nothing is deployed by this verdict; no holdout data has been touched.
