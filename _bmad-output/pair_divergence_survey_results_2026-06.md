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

## Disposition

- All five pairs **CLOSED** per the pre-committed rule — no parameter sweeps beyond the
  frozen grid, no re-tests on other sessions/timeframes without a new pre-registration.
- Sealed holdout slices (si/hg/pl/rty/ym/gc ≥ 2026-03-01, chmod 444, ACCESS_LOG
  entries) remain sealed and unconsumed — available if a future pre-registered
  hypothesis needs a one-shot OOS test on these instruments.
- The live MNQ/MES stat-arb bot and MIM-NB bot are unaffected; no config touched.
- Data assets gained: SI, HG, PL, RTY, YM 1-min bars 2025-05-01 → 2026-06-12
  (`data/processed/dollar_bars/1_minute/*_1min_2025_2026.csv`, ~285k–366k bars each).
