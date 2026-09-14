# Roll-splice correction (C1) for the H2/L2 and wedge/MM power gates

**Written:** 2026-09-14, after the contamination diagnostic and before any corrected re-run. It is committed with `corrected_rerun.py` before that script runs; the script pins this file's SHA-256.

**The diagnostic** (`contamination_check.py`, output `contamination_results.json`) computed only these, with no placebo, power or post-event price:
- session contract purity
- 5-minute bar ranges
- each event's entry-known risk

## 1. What the diagnostic found

- **27 of the gates' 297 RTH sessions are roll-week sessions whose 1-minute input interleaves two contracts.** Dates are in `contamination_results.json`: Mar 3 and 5, Jun 2–12, Sep 1–10 and Dec 1–10, all 2025.
- The CSV has 5,583 mixed-contract rows, matching `docs/reports/yank-provenance-closure/README.md`.
- **Every interleaved session contains a fake 5-minute bar of at least 234.5 points**, the calendar spread. The clean-session median is 93.5.
- The largest clean-session bars are genuine market events: 2025-04-07 and 04-09 (tariffs) and 2025-08-22 (Jackson Hole).

**Events in interleaved sessions:**

| Construct | Events there | Events with R$ > $400 there |
|---|---|---|
| H2/L2 | 12 of 56 | **8 of 8** |
| Wedge s1 | 30 of 371 | 14 of 17 |
| MM fade s1 | 43 of 889 | 20 of 23 |

**Clean-session mean R$** is $80 vs $138 for H2/L2, $88 vs $107 for wedge, and $86 vs $97 for MM.

**Second-order effect (H2/L2 only):** its EMA21 filter runs continuously. At each roll, the EMA carries the ~240-point contract gap into the next session for about 21+ bars. The wedge/MM gate uses no EMA.

**A second, separate defect** was found while checking the loader's other input, and before writing this section. Also a data-only check:
- The Jan–Feb 2026 rows of `mnq_1min_2026_ytd.csv` are **not the front month**. Their closes differ from the raw MNQH26 closes at the same timestamps by a median 223.25 points, with zero exact matches. That fits the deferred MNQM26, which `scripts/update_eod_data.py` targets.
- The CSV has 29,157 rows for those two months against 56,100 raw front-month minutes, consistent with a thinly traded back month.
- So about 39 of the gates' sessions run on an illiquid deferred contract. There is also a ~247-point jump at the 2025→2026 file boundary.
- **Also noted:** 2,973 of the 86,545 RTH rows in 2025 (3.4%) are dollar-aggregated over several minutes.

## 2. Two re-runs, each changing only the input bars

Both re-run the committed gates' **unchanged** `main()` code paths. Only the bar loader (`load_5min`) is replaced.

**C1a, splice removal (attribution).** The same hash-verified CSVs, with:
1. The 27 interleaved sessions dropped at the 1-minute level, before resampling and before the EMA. A session is interleaved if its raw RTH minutes in `/root/mnq_historical.json` carry more than one contract label, which is the diagnostic's definition.
2. The H2/L2 EMA21 restarted at the first bar of each contract segment, using the gate's own rule (`adjust=False`, seeded on the first bar).
   - Segments for 2025 sessions come from the raw labels.
   - All 2026 CSV sessions form one segment, MNQM26 per the check above.

C1a shows how much the roll splices alone moved each verdict. The back-month and dollar-aggregation defects stay in.

**C1b, front-month rebuild (the corrected gate).** The 1-minute input is rebuilt from the raw TradeStation front-month records for the same window, 2025-01-01 → 2026-02-28. That means fixed one-minute bars, close-stamped, and front month only (H26 in Jan–Feb 2026). Then:
- the same 27 interleaved sessions are dropped
- the EMA21 is restarted at each contract change
- the gates' own RTH filter, 5-minute resampling, slots, cutoff assert and everything downstream apply unchanged
- holiday half-days stay in, as in the originals

**Unchanged in both:** window, constructs, filters, geometry, costs, effect sizes, placebo, power formula and verdict rules. Any other difference from the original scripts is a bug.

## 3. How the result is recorded

- **The C1b verdict supersedes the original** for each construct, because the original inputs contained fabricated and deferred-contract prices. C1a is reported to attribute the change.
- The originals' `results.md` files get an appended note pointing here. Their files and hashes are otherwise untouched.
- **If a verdict is unchanged,** it is recorded as robust to the contamination.
- **If a verdict improves** (for example to MARGINAL or POWERED), that is not adoption. The construct still needs its own pre-registration before any test.
- **If a verdict worsens,** it stands.
- Prose claims in the original results that relied on the risk tail are marked as affected. The known one is H2/L2's "20% of events risk more than $150, mostly the high-volatility months of 2025".

## 4. What C1 will not do

- Compute any statistic of price after a real event.
- Read `data/sealed_holdout/`, or any bar on or after 2026-03-01.
- Change anything beyond section 2, or re-run the H2/L2 exploratory variants.
