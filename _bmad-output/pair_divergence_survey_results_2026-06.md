# Cross-Pair Divergence-Fade Survey — Gate 0 Results

**Date:** 2026-06-12
**Pre-registration:** `_bmad-output/precommit_pair_divergence_survey_2026-06.md`
(commit `b54fb08`, committed before any new-pair simulation ran)
**Harness:** `study_pair_divergence_survey.py` | **Config:** `pair_survey_config.yaml`
**Full log:** `logs/pair_survey_gate0.log` | **Grid CSV:** `_bmad-output/pair_survey_gate0_grid.csv`
**Dev window:** 2025-05-01 → 2026-02-28 (hard-coded). Sealed holdout (≥ 2026-03-01) untouched.

## Verdict: NO new pair qualifies for Gate 1

All five candidate pairs fail the pre-committed decision rule in both
directions. The MNQ/ES control reproduced the validated template exactly
(N=633, WR=58.0%, PF=1.27, freq=2.96/d, worst-mo=38.5%) and was the only
pair-direction to qualify — the harness and the decision rule discriminate
correctly.

## Cross-pair summary (primary spec $40 thr / 1.0× stop, ranked by net edge density)

| Pair | Dir | N | WR | BE WR | BE stress | PF | Avg$/trade | $/day/ct | Worst-mo | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| PL–GC | LONG | 3,987 | 54.0% | 55.0% | 67.5% | 1.05 | +$2.05 | +$41.00 | 49.1% | 🔴 FAIL (and info-only) |
| SI–GC | LONG | 2,880 | 53.6% | 54.7% | 67.2% | 1.05 | +$1.91 | +$26.51 | 47.7% | 🔴 FAIL |
| **MNQ–ES** | **SHORT** | **633** | **58.0%** | **56.0%** | **57.2%** | **1.27** | **+$6.57** | **+$19.44** | **38.5%** | **🟢 control (live)** |
| RTY–ES | LONG | 67 | 52.2% | 51.6% | 52.8% | 1.29 | +$4.28 | +$1.34 | 0.0% | 🔴 FAIL |
| RTY–ES | SHORT | 88 | 45.5% | 51.6% | 52.8% | 1.01 | +$0.13 | +$0.05 | 27.3% | 🔴 FAIL |
| YM–ES | SHORT | 107 | 51.4% | 51.6% | 52.8% | 0.97 | −$0.70 | −$0.35 | 33.3% | 🔴 FAIL |
| YM–ES | LONG | 105 | 53.3% | 51.6% | 52.8% | 0.93 | −$1.54 | −$0.75 | 16.7% | 🔴 FAIL |
| HG–GC | SHORT | 231 | 45.9% | 54.7% | 57.8% | 0.92 | −$2.01 | −$2.19 | 25.0% | 🔴 FAIL |
| HG–GC | LONG | 254 | 47.6% | 54.7% | 57.8% | 0.81 | −$4.64 | −$5.56 | 22.2% | 🔴 FAIL |
| MNQ–ES | LONG | 697 | 49.5% | 56.0% | 57.2% | 0.79 | −$6.35 | −$20.69 | 36.5% | 🔴 FAIL (known) |
| SI–GC | SHORT | 2,904 | 47.0% | 54.7% | 67.2% | 0.80 | −$8.21 | −$115.15 | 37.9% | 🔴 FAIL |
| PL–GC | SHORT | 4,008 | 49.1% | 55.0% | 67.5% | 0.86 | −$6.12 | −$123.33 | 43.1% | 🔴 FAIL |

## Kill reasons per pair

**SI–GC (silver/gold).** SHORT (fade silver outperformance): WR 47.0%, −$8.21/trade —
silver spikes are momentum, not reversion; fading them loses outright. LONG (fade silver
underperformance): gross-positive (PF 1.05, +$1.91/trade at 13.9/day) and passes 4/5
gates, but WR 53.6% < 54.7% breakeven and is hopeless against the slippage-stressed
67.2% bar ($40 stop = only 8 SIL ticks; 1 tick/side of slippage is $10 against a
$3.74-commission trade). **Kill: cost floor.**

**RTY–ES (small vs large caps).** The closest analog to the validated edge does not
exist at viable frequency: a $40 M2K stop = 8 RTY points of 5-bar divergence, which
occurs only 0.3–0.4×/day (gate: 1.0). What trades there are skew gross-positive
(LONG PF 1.29; 7/8 short cells positive EV) but N is too small and worst-month WR is
0–27%. **Kill: frequency** — RTY/ES simply doesn't decouple intraday the way MNQ/ES
does. (M2K's $5/pt also means tradeable divergences are 8× rarer in index points than
MNQ's $2/pt equivalent.)

**YM–ES (Dow vs S&P).** Flat-to-negative both directions (PF 0.93–0.97), frequency
0.5/day, worst-month 17–33%. Coupling is too tight to leave a fade-able residual after
the beta hedge. **Kill: no edge + frequency.**

**HG–GC (copper/gold).** Negative both directions (PF 0.81–0.92, WR 46–48% vs 54.7%
breakeven). The macro spread wanders rather than mean-reverts at 1-min horizon.
**Kill: no edge** — confirming the weakest-coupling prior.

**PL–GC (platinum/gold, informational only).** Same shape as SI–GC: long side
gross-positive (PF 1.05, +$2.05/trade at 20/day, passes 4/5 gates) but WR 54.0% < 55.0%
breakeven and the thin-book slippage stress (67.5%) is disqualifying — and this was
pre-declared informational-only (no micro contract; full PL on a $2k trailing-DD combine
is unsizeable). **Kill: cost floor + structure.**

**Pit-hours sensitivity (labeled exploratory, metals only):** 08:30–13:25 ET is *worse*
or flat for every metals pair-direction (best cell: PL–GC LONG PF 1.00). The RTH result
is not a session artifact.

## Lessons confirmed / new

1. **The asymmetry flips by asset class.** In index pairs the SHORT side (fade
   outperformance of the high-beta leg) wins; in all three metals pairs the LONG side
   (fade underperformance vs gold) is the better direction. Consistent story: metals
   rallies are flow-driven and continue; metals *lags* vs gold partially re-converge.
   Testing both directions per pre-commitment is what caught this — direction priors do
   not transfer between pairs.
2. **The cost floor is the binding constraint on metals microstructure.** SI/PL long
   fades have a real gross edge (~PF 1.05 at 14–20 trades/day) but ~$2/trade against a
   $3.74–$4 commission and $10/RT slippage stress. This is the S26 verdict shape again
   (gross PF 1.13, +$1.09/ct vs $6 costs): 1-min mean reversion edges in non-index
   products sit below the retail cost floor.
3. **MNQ/ES remains special.** Its divergence is frequent (3/day), large in dollar
   terms relative to tick size (a $40 stop = 80 MNQ ticks vs 8 SIL ticks), and
   asymmetric in the direction costs favor. None of the five candidates shares all
   three properties.

## Addendum (2026-06-12, prereg `9f4ae9a`): BTC–ETH — FAIL both directions

Run after the parent survey under the identical frozen protocol (addendum doc:
`precommit_pair_survey_addendum_btc_eth_2026-06.md`; log:
`logs/pair_survey_btc_eth.log`). Signal = Kraken perps (continuous, no rolls);
economics = CME MBT (0.1 BTC, $2.84 RT, tick $2.50); MET excluded as traded leg
(PV makes the frozen grid unreachable); pair-level beta clip [0,200]
pre-registered (template [0,10] would have silently broken the hedge at BTC/ETH
price scale). Control re-run after the harness change: bit-identical (N=633,
WR 58.0%, PF 1.27).

| Dir | N | WR | BE WR | BE stress | PF | Avg$/trade | $/day/ct | Worst-mo | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| SHORT BTC | 101 | 53.5% | 53.6% | 59.8% | 1.16 | +$2.60 | +$0.87 | 33.3% | 🔴 FAIL |
| LONG BTC | 114 | 50.9% | 53.6% | 59.8% | 0.92 | −$1.62 | −$0.61 | 42.9% | 🔴 FAIL |

**Kill: frequency + cost floor.** The RTY–ES shape again: a $40 MBT stop = 400
BTC index points (~0.5%) of 5-bar beta-hedged divergence occurs only 0.33×/day
(gate: 1.0). The short side (fade BTC outperformance vs ETH — same direction as
the index asymmetry) is gross-positive (PF 1.16, +$2.60/trade) but sits 0.1pt
below even the unstressed breakeven WR, with worst-month 33.3% and no path to
the 59.8% stressed bar. At $30 the frequency only reaches 0.85/day with the
edge collapsing (+$0.40/trade). BTC–ETH residuals at 1-min are either too small
to clear MBT's cost floor or too rare to matter. **Pair CLOSED** per the
addendum decision rule.

## Addendum 2 (2026-06-12, prereg `023d0de`): 5-minute extension — NO qualification; family CLOSED across timeframes

One-shot 5m re-test of all 7 pairs (√5-scaled grid, primary $80/1.0×, log:
`logs/pair_survey_5m.log`, grid: `pair_survey_gate0_grid_5m.csv`). The 1-min
control re-ran bit-identical (N=633, WR 58.0%, PF 1.27) after the resample
harness change.

**The mechanism hypothesis was confirmed — and still didn't clear the bar.**

| Pair | Dir | N | WR | BE / BE-stress | PF | Avg$ | $/day | Verdict |
|---|---|---|---|---|---|---|---|---|
| **SI–GC** | **LONG** | **462** | **57.8%** | **52.3% / 58.6%** | **1.27** | **+$12.60** | **+$27.85** | 🟡 **passes all 5 Gate 0 gates; fails ONLY the slippage-stress clause by 0.8pt** |
| PL–GC | LONG | 632 | 52.4% | 52.5% / 58.8% | 1.05 | +$2.83 | +$8.80 | 🔴 sub-breakeven (info-only anyway) |
| MNQ–ES | SHORT | 169 | 55.6% | 53.0% / 53.6% | 1.17 | +$7.66 | +$6.05 | 🔴 freq 0.81/d + worst-mo 33.3% — 1-min is MNQ's scale |
| HG–GC | LONG | 89 | 57.3% | 52.3% / 53.9% | 1.37 | +$11.03 | +$4.63 | 🔴 freq 0.43/d |
| YM–ES | LONG | 35 | 68.6% | 50.8% / 51.4% | 1.97 | +$20.97 | +$3.41 | 🔴 freq 0.17/d (tiny N) |
| BTC–ETH | LONG | 50 | 52.0% | 51.8% / 54.9% | 1.06 | +$2.06 | +$0.34 | 🔴 freq + stress |
| (all SHORT metals/crypto, RTY both) | | | | | | | | 🔴 negative or tiny |

**SI–GC long deep-dive (the near-miss):** cost-fraction mechanism worked as
hypothesized — avg edge scaled $1.91 → $12.60/trade while frequency held
(13.9/d → 2.21/d but each trade 6.6× larger), PF 1.05 → 1.27, 10/10 months
≥ 45% WR floor, monotone improvement with threshold ($100 cell: WR 59.2%,
PF 1.36, +$18.11). The sole failing clause is the frozen 1-tick/side SIL
slippage stress: measured 57.8% vs required 58.6% — a 0.35-SE gap (N=462),
statistically indistinguishable, and the house precedent (HCVWAP v3: WR 30.0%
vs gate 30.2% = DEAD) is that the gate is the gate. SIL's real spread is
plausibly 1–2 ticks ($5–10/side), so the stress assumption is realistic, not
conservative.

**Disposition per addendum-2 rule: the divergence-fade family is CLOSED across
timeframes for these pairs.** The grid observation that edge rises with
threshold is recorded but explicitly NOT actionable (banned sweep). The one
legitimate future path, requiring a fresh pre-registration and Alex's explicit
decision: empirically measure live SIL spread/slippage (prospective quote
capture, no historical re-test), and only if measured cost < the stressed
assumption would a re-evaluation of this exact frozen result be justified.
Direction asymmetry note: at 5m the LONG side dominated nearly everywhere —
consistent with the 1-min metals finding.

## Disposition

- All five pairs **CLOSED** per the pre-committed rule — no parameter sweeps beyond the
  frozen grid, no re-tests on other sessions/timeframes without a new pre-registration.
- Sealed holdout slices (si/hg/pl/rty/ym/gc ≥ 2026-03-01, chmod 444, ACCESS_LOG
  entries) remain sealed and unconsumed — available if a future pre-registered
  hypothesis needs a one-shot OOS test on these instruments.
- The live MNQ/MES stat-arb bot and MIM-NB bot are unaffected; no config touched.
- Data assets gained: SI, HG, PL, RTY, YM 1-min bars 2025-05-01 → 2026-06-12
  (`data/processed/dollar_bars/1_minute/*_1min_2025_2026.csv`, ~285k–366k bars each).
