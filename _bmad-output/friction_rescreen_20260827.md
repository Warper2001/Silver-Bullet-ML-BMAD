# Friction Re-Screen of the MNQ/MES Combine Graveyard

**Date:** 2026-08-27
**Question:** The ~15 buried combine candidates were killed by Topstep combine gates
(win-rate floor, ≥1.0/day frequency, $2k trailing drawdown). That is a different
question from *does this have an edge*. Re-score them on economics alone.

**Artifacts:** `tools/friction_rescreen.py`, `data/reports/friction_rescreen_20260827.txt`

---

## 1. The systematic defect

All **17** graveyard studies hard-code `COMMISSION = 4.80` round-trip and model no
slippage separately:

```
study_5min_trend_pullback.py, study_es_mnq_dollar_divergence.py, study_es_mnq_stat_arb.py,
study_es_mnq_wide_tp.py, study_gc_post_catalyst.py, study_hcvwap.py, study_hcvwap_v2.py,
study_hcvwap_v3_longonly.py, study_lunch_window_oscillation.py, study_mes_orb_reversion.py,
study_mes_orb_reversion_v2.py, study_pdh_pdl_breakout.py, study_stat_arb_large_div.py,
study_stat_arb_short_only.py, study_vol_compression_15min.py,
study_vol_compression_breakout.py, study_volume_profile_poc.py
```

Live MNQ books **$2.00/contract round-trip**. Verified against the live tape: for
`trader-yank`, `delta × $2 × 2ct` minus logged P&L equals exactly `−$4.00` on every
clean trade (2 contracts × $2.00).

Market/stop entries realistically pay ~**$3.00** (commission + ~1 tick of spread).
Limit entries (YANK, gap-fade) pay ~$2.00.

**Every marginal verdict in the graveyard was scored against a cost 60–140% too high.**

## 2. What changes

Re-scored at $3.00 RT (conservative — assumes market entry, pays the spread):

| candidate | N | /day | EV@4.80 | PF@4.80 | EV@3.00 | PF@3.00 | ratio | verdict |
|---|---|---|---|---|---|---|---|---|
| HCVWAP v2 (5m, SD2.0/S20) | 76 | 0.26 | +8.62 | 1.300 | **+10.42** | **1.378** | 3.5× | **CLEARS 3×** |
| HCVWAP v2 (5m, SD2.0/S15) ◀ primary | 89 | 0.30 | +6.89 | 1.290 | **+8.69** | **1.386** | 2.9× | MARGINAL |
| HCVWAP v2 (5m, SD2.5/S10) | 114 | 0.38 | +4.93 | 1.280 | **+6.73** | **1.412** | 2.2× | MARGINAL |
| MES ORB Rev v2 (S12/ORB25) | 15 | 0.07 | +12.53 | 1.360 | +14.33 | 1.423 | 4.8× | CLEARS 3× (N=15) |
| ES/MNQ Stat Arb short [IS] | 350 | 3.60 | +6.45 | 1.268 | +8.25 | 1.354 | 2.8× | MARGINAL |
| HCVWAP v1 (1m, SD2.5/TP10) | 335 | 1.12 | −0.66 | 0.920 | +1.14 | 1.155 | 0.4× | THIN |
| HCVWAP v1 (1m, SD2.0/TP12) ◀ primary | 481 | 1.61 | −1.83 | 0.810 | **−0.03** | **0.997** | −0.0× | exactly breakeven |
| PDH/PDL Breakout | 280 | 0.94 | −8.45 | 0.796 | −6.65 | 0.835 | −2.2× | LOSES |
| 5-min NQ Trend-Pullback | 200 | 0.67 | −8.79 | 0.775 | −6.99 | 0.816 | −2.3× | LOSES |
| MES ORB Reversion v2 ◀ primary | 32 | 0.15 | −11.40 | 0.700 | −9.60 | 0.738 | −3.2× | LOSES |
| ES/MNQ Stat Arb short [OOS] | 198 | 3.60 | −7.70 | 0.765 | −5.90 | 0.814 | −2.0× | LOSES |
| HCVWAP v3 long-only [OOS] | 10 | 0.18 | −3.72 | 0.850 | −1.92 | 0.918 | −0.6× | LOSES |
| HCVWAP v1 MES ◀ primary | 258 | 1.20 | −6.61 | 0.700 | −4.81 | 0.769 | −1.6× | LOSES |

**Verdict flips: 4.** HCVWAP v2 (all three cells) and the MES ORB v2 grid cell go from
"failed" to economically viable. HCVWAP v1's primary spec lands on *exactly* breakeven
(PF 0.997) — the highest-frequency candidate in the graveyard sits at precisely zero.

**Verdict holds: 9.** PDH/PDL, trend-pullback, MES ORB v2 primary, both OOS failures,
and HCVWAP v1 MES remain clearly negative at any plausible cost.

## 3. The N question — answered, and the answer is no

Alex's question was whether the graveyard contains a candidate with s26-like frequency
(1.88/day) that also survives its costs.

Exactly one candidate ever had both: **ES/MNQ Stat Arb short-only**, 3.60/day,
+$8.25/trade at corrected cost, PF 1.354, ratio 2.8×.

That is the **in-sample** row. Its OOS (2026-03-01 → 2026-05-19) is −$5.90/trade,
PF 0.814. It died on a genuine out-of-sample test, not on a combine rule.

Everything that clears its costs at corrected pricing fires at **0.07–0.38/day**:
789–4,286 days to reach N=300.

## 4. Caveat that stops HCVWAP v2 being oversold

HCVWAP v2's edge is entirely its long leg — in-sample long N=60, WR 38.3%, PF 1.87,
+$18.73/trade; short N=29, PF 0.39, −$17.61/trade. The combined spec failed the
combine's win-rate gate because the short leg dragged it down.

**That long leg was already tested out-of-sample, as HCVWAP v3.** It failed:
N=10, WR 30.0%, PF 0.85, −$3.72/trade (−$1.92 at corrected cost). The v3 report
carries its own disclosure that the long-side split was observed in-sample first.

So HCVWAP v2 is not a resurrection. It is the one candidate whose *original* death
certificate was a combine-rule artifact rather than an economic verdict — and whose
only OOS test, at N=10, failed.

## 5. Correction to the hypothesis that motivated this screen

The screen was run to test a claimed inverse law between frequency and edge, observed
across the 6 live bots. Across these 13 graveyard candidates:

**Spearman(frequency, EV@3.00) = −0.209**, N=13.

Weak and not significant at this N. The pattern is real in the live portfolio; the
graveyard does **not** support stating it as a general law. Recorded here so the
stronger claim does not propagate.

## 6. Not covered

Vol Compression (1-min and 15-min), Volume Profile POC Fade, Lunch-Window Oscillation,
and the pair-divergence survey are not in this table — their reports are not in the same
summary format and were not tabulated. Memory records Vol Compression as PF 1.46–1.96
(1-min) and 1.77 (15-min), killed purely on frequency (0.05–0.27/day), so corrected
costs would improve them but cannot address the N question. **This screen is not
exhaustive.**

## 7. What this does and does not authorize

Does **not** authorize deploying, promoting, or re-tuning anything. A candidate passing
a friction screen on in-sample numbers has cleared one arithmetic objection, nothing more.
Any next step on HCVWAP v2 requires its own pre-registration and a genuinely fresh
prospective window — the 2026-03-01/05-19 holdout has been accessed 33 times and is spent.

---

# PART II — Screen completed (2026-08-27, same session)

Section 6 above flagged four families as not covered. They are now covered. Their
studies print to stdout and were never saved to `data/reports/`, so each was
**re-run** — once as-is (reproducing the original verdict) and once with
`COMMISSION = 3.00`. The pair survey needed no re-run: it is the only study in the
graveyard that costed correctly (`commission_rt: 1.24` for MNQ-class plus an explicit
`slippage_stress_rt` clause), so its published numbers are already realistic.

## 8. The four remaining families

| candidate | N | /day | WR | EV@4.80 | EV@3.00 | PF@3.00 | ratio | verdict |
|---|---|---|---|---|---|---|---|---|
| Volume Profile POC (VA70/TP=POC) | 6,880 | **23.09** | 14.2% | +0.22 | **+2.02** | 1.100 | 0.7× | THIN → killed on concentration |
| Pair MNQ–ES SHORT_A (thr50/sm1.0) | 372 | 1.74 | 60.2% | +11.28 | +11.28 | 1.419 | **3.8×** | already @$1.24 |
| Pair MNQ–ES SHORT_A (thr30/sm1.0) | 1,181 | 5.52 | 56.0% | +2.06 | +2.06 | 1.100 | 0.7× | already @$1.24 |
| Pair MNQ–ES LONG_A (thr40/sm1.0) | 697 | 3.26 | 49.5% | −6.35 | −6.35 | 0.795 | −2.1× | LOSES |
| Vol Compression 1-min (CF.3/MB3) | 46 | 0.15 | 34.8% | +9.90 | **+11.70** | 1.580 | **3.9×** | clears, glacial |
| Vol Compression 15-min (CF.5/MB3) | 15 | 0.05 | 66.7% | +23.29 | **+25.09** | 1.840 | **8.4×** | clears, N=15 |
| Lunch-Window Oscillation | 1,151 | 3.88 | 16.4% | −4.28 | −2.48 | 0.810 | −0.8× | LOSES |

## 9. Volume Profile POC Fade — the only new high-frequency positive, and it fails

At 23.09 trades/day and N=6,880 this was the strongest candidate the screen could have
produced: 12× s26's frequency, N=300 reachable in **13 days**, and positive at corrected
cost (+$2.02/trade, PF 1.10).

The concentration test kills it.

| | N | net | mean/trade |
|---|---|---|---|
| All 14 months | 6,880 | $13,876 | +$2.02 |
| minus Sept 2025 | 6,469 | $510 | **+$0.08** |
| minus Sept + Dec 2025 | 6,091 | −$6,056 | **−$0.99** |
| minus top 3 months | 5,556 | −$9,678 | −$1.74 |

**September 2025 alone is 96.3% of all profit.** Nine of fourteen months have negative
mean P&L. At a 14.2% win rate and 23 trades a day, the $3.00 cost assumption is also
generous — a strategy entering that often pays queue position and impact that no
backtest here models.

## 10. The cross-cutting finding: September 2025

POC Fade is not the only strategy whose edge lives in one month. Independently, across
different setup classes, resolutions and instruments:

| strategy | Sept 2025 |
|---|---|
| Volume Profile POC Fade | +$32.52/trade — **96.3% of its total profit** |
| Vol Compression 1-min | +$43.52/trade (N=9) — best month; annual total only $456 |
| Vol Compression 15-min | +$162.54/trade (N=2) — best month; annual total only $349 |
| HCVWAP v1 | WR 68.2% — recorded in memory as "the one clean mean-reversion month" |
| PBC Track B | WR 88.9% — against 14.3% in July and 16.7% in October |

Five independent fade/reversion strategies all peak in the same month. September 2025
was a mean-reversion regime, and a substantial part of what looked like edge across the
graveyard is **one regime month appearing five times**. Any future candidate in this
family should be scored with September 2025 held out, not because that month is invalid,
but because it is not independent evidence across strategies that all share it.

## 11. Final answer to the question that prompted the screen

*Is there a buried candidate with s26-like frequency that also survives its costs?*

Two rows qualify on frequency + economics: **Pair MNQ–ES SHORT_A at thr50 (1.74/day,
+$11.28/trade, 3.8×)** and **ES/MNQ Stat Arb short-only in-sample (3.60/day, +$8.25,
2.8×)**.

They are the same strategy family — ES/MNQ divergence, short leg only. That family was
tested out-of-sample as Gate 2 and **failed**: WR 57.8% → 45.7%, PF 1.268 → 0.765,
−$7.70/trade (−$5.90 at corrected cost), with March marginal, April losing and May
collapsing. Its short-only direction was also selected after observing in-sample
asymmetry, disclosed at the time.

**So the answer is no.** The high-frequency, positive-economics quadrant contains exactly
one strategy family, and it has already been tested prospectively and failed. Everything
else that clears 3× friction fires between 0.05 and 0.38 times a day.

## 12. What the completed screen delivered

1. A cost error in **17 of 18 studies** — every marginal verdict in the graveyard was
   decided against a price 60–140% too high.
2. **Four verdict flips** (HCVWAP v2 × 3 cells, MES ORB v2 grid cell) plus two
   confirmations that Vol Compression's edge is real and was killed purely on frequency.
3. **HCVWAP v2** as the one candidate whose death certificate was a combine-rule
   artifact — with the caveat (§4) that its profitable long leg already failed OOS at N=10.
4. The **September 2025 dependency**, which is new and applies to the whole fade family.
5. A definitive **no** to the high-N question.

Nothing here authorizes a deployment, a promotion, or a re-tune. Any next step needs its
own pre-registration and a genuinely fresh prospective window.
