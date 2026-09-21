# MIM-NB sealed engine re-run on the 2025 bars (2026-09-21)

**Read this first: 2025 is the sealed DEV window, so this run is in-sample and cannot bear on decay.**
The noise-bands prereg names `mnq_1min_2025.csv` (md5 3ba83a32…) as the Gate-0 dev set and `mnq_1min_2026_ytd.csv` as the
one-shot OOS, already spent. My last message called a 2025-01 to 2026-05 re-run "data the strategy never saw"; that was wrong.
Nothing from 2026 was read here.

Files: `run_sealed_2025.py`, `results.json`, `trades_*.csv`, `rebuild_2025_frontmonth.py` (+ `rebuild_meta.json`, `mnq_1min_2025_frontmonth.csv`).
Engine: sealed `run_catstop()` lifted verbatim by AST from `study_mim_nb_catstop.py` (sha `210518d6…`, prereg 6957daa), at S=250 (live) and S=500 (sealed original), sealed cost 1.12 pt/trade.
No gate applied, no threshold tested, no parameter varied.

## Parity: the engine reproduces the sealed 2025 trades exactly
Against the 2025 rows of `data/reports/mim_nb_catstop_s{250,500}_pooled.csv` (committed 2026-06-11): N=114 both, sum of points
3,221.75 (S250) and 3,734.75 (S500), **every row identical**. Front-month rebuild reproduced the frozen CSV byte for byte (sha 3f20ec70…) before use.

## Results, S=250 (the live config), 2025 in-sample

| | frozen CSV | front-month rebuild |
|---|---|---|
| N / net PF / net $ | 114 / **1.496** / +$6,188 | 114 / **1.543** / +$6,573 |
| expectancy per trade | +$54.28 | +$57.66 |
| net per trade, bp (95% CI) | **+14.1** (-5.7 to +39.2), SD 123 bp | +14.8 (-4.7 to +39.8) |
| win rate / payoff | 54.4% / 1.25 | 54.4% / 1.29 |
| daily Sharpe, 224 sessions (95% CI) | **+1.30** (-0.81 to +2.78) | +1.38 (-0.74 to +2.83) |
| exits | 13 cat-stops (-250 each), 101 EOD (+64 avg) | 12 / 102 |

S=500: PF 1.612 / 1.629, +$63 / +$64 per trade, daily Sharpe +1.51 / +1.53.

- **The roll-week defect does not flatter MIM-NB.** The corrected bars are slightly *better* (PF 1.54 vs 1.50), matching the 09-16 triage. Excluding the 44 defect sessions: N=107, PF 1.593.
- **Halves (fixed in advance): H1 PF 1.93 (N=47, +$108/trade, +29.7 bp) vs H2 PF 1.16 (N=67, +$17/trade, +3.2 bp).** The weakening starts inside the dev window. Front-month: 1.92 vs 1.23. Descriptive, with wide intervals.
- **Five days carry it.** Removing the 5 best days (2025-02-21, 02-27, 04-04, 04-09, 10-10) leaves N=109, **PF 0.90, -$1,236, -2.8 bp/trade** (front-month PF 0.93). April 2025 alone is +$3,686 on 14 trades; 04-04 and 04-09 are the tariff-crash days. This is the fat-tail profile the seal disclosed.
- Monthly net: Jan -632, Feb +1,344, Mar +1,009, Apr +3,686, May -417, Jun +84, Jul -446, Aug +926, Sep -597, Oct +1,495, Nov +257, Dec -521.

## What this says about the decay report
- **Not evidence for or against decay.** 2025 is the window the spec was developed and gated on (it had to clear N≥100, PF≥1.10 to be built), so a positive number here is the expected outcome, not a finding.
- **The ordering is what selection bias predicts:** 2025 dev PF 1.50 (S250) -> 2026 OOS PF 1.30 (S500 benchmark, spent) -> live 2026-06 to 09 PF 0.996 (N=29). Dev > OOS > live happens with or without real decay.
- **On 2025 itself the two do not line up:** the report says 2025-26 is "below zero" on SPY/ES, while MIM-NB is +1.3 Sharpe in-sample. They are different specs; one untested reason is that the report's 2% vol targeting shrinks size on exactly the crash days that carried MIM-NB's fixed 1 contract. That is a hypothesis, not a result.
- **Statistical resolution:** the in-sample daily Sharpe CI (-0.8 to +2.8) already includes zero. Detecting the report's +2.6 bp/trade at SD 123 bp needs about 13,800 trades; even a 2025-sized +14 bp needs about 470.

## Next step, if you want a real read on decay
The only data MIM-NB never saw is **MNQ front-month 2021-2024** (`data/mim_x/mnq_1min_2021_2024_frontmonth.csv`, clean, about 4 years x 114 trades/yr ≈ 456 trades). That is roughly enough to see a 2025-sized effect (about 470 needed) and nowhere near enough for the report's +2.6 bp. It would be a one-shot confirmatory test, so under the repo's policy it needs, before any run: a pre-registration, an outcome-blind power gate, and thresholds derived from a cited artifact (this one qualifies as a source for the effect and SD), not hand-set. It would spend the last unseen historical window, so it should be your call. Not run.

## Side finding, not investigated
`mnq_1min_2026_ytd.csv` md5 is now 30bc05a8… but the noise-bands seal pinned 4ec175dd… for that file. It was modified after sealing (mtime 2026-06-11 23:31). The 2025 file still matches its seal. Any use of the 2026 file as "the sealed OOS" needs that difference explained first.

**Resolved 2026-09-21:** the md5 change is cosmetic for MIM-NB. The file was extended to 2026-06-11 and gained a `notional` column after the seal, and the
sealed engine on it reproduces the sealed 2026 trades exactly (50/50 at S250 and S500, through 2026-05-19). See `diagnostics_mnq_2026_ytd_md5_20260921/REPORT.md`.
