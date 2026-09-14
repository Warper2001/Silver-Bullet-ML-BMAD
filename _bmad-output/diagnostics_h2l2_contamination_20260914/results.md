# H2/L2 and wedge/MM power gates: contamination check and corrected re-runs (2026-09-14)

- **Plan:** `correction_plan.md`, sha256 `18a16b33…6302`. It was committed with the scripts and the diagnostic output in `3ecfb8f` before any corrected run.
- **Guard:** the replica loader, with both corrections switched off, reproduced the original gate's bars exactly (`assert_frame_equal`). Only the bar loader differs. Both gates' committed `main()` ran unchanged.
- **Firewall:** as in the originals. No price after a real event was computed, and nothing under `data/sealed_holdout/` was opened.
- **Inputs:** the CSVs are hash-verified against the originals' `results.json`. The raw `/root/mnq_historical.json` hash is in `corrected_rerun_meta.json`.

## 1. The contamination is real (`contamination_results.json`)

**Roll-week interleaving.**
- **27 of 297 sessions** (Mar 3 and 5, Jun 2–12, Sep 1–10 and Dec 1–10, 2025) have 1-minute input that alternates between two contracts minute by minute.
- Every one of those sessions contains a fake 5-minute bar of at least **234.5 points**. The clean-session median is 93.5.
- The largest clean-session bars are real events: 2025-04-07 (868 points) and 04-09 (tariffs), and 2025-08-22 (Jackson Hole).
- The CSV has 5,583 mixed-contract rows, matching the provenance report.

**Back month in 2026.** Jan–Feb 2026 in `mnq_1min_2026_ytd.csv` is **not the front month**.
- Closes differ from the raw MNQH26 closes by a median **223.25 points**, with 0% exact matches.
- The file has 29,157 rows against 56,100 front-month minutes, so it is a thin deferred contract, most likely MNQM26.
- About 39 of the gates' sessions are affected.

**Dollar aggregation.** 2,973 of the 86,545 RTH rows in 2025 (3.4%) span several minutes.

| Events (primary arm) | In interleaved sessions | With R$ > $400 in interleaved sessions |
|---|---|---|
| H2/L2 | 12 of 56 | **8 of 8** |
| Wedge s1 | 30 of 371 | 14 of 17 |
| MM fade s1 | 43 of 889 | 20 of 23 |

## 2. Corrected verdicts: unchanged on every primary arm

**C1b** (the rebuild from raw front-month bars, 270 sessions) supersedes the originals, per plan section 3. **C1a** (splices dropped from the CSVs) differs from C1b only slightly, so the roll splices account for nearly all of the change.

| | Original | C1a | **C1b** |
|---|---|---|---|
| **H2/L2** (1R, $5.80, θ = 0.10R) | | | |
| N (upper / lower) | 56 / 53 | 45 / 43 | **47 / 44** |
| Mean R$ (p90) | $137.75 ($466) | $81.8 ($137) | **$81.4 ($147)** |
| Breakeven gross θ | 0.042R | 0.071R | **0.071R** |
| σ (cluster) | $102.9 | $83.1 | **$82.1** |
| μ_net | +$7.98 | +$2.38 | **+$2.34** |
| Power (upper) | 14.3% | 7.3% | **7.4%** |
| Years for 80% | ~22 | ~180 | **~173** |
| Optimistic 0.20R: power / years | 47.5% / 2.9 | 21.4% / 9.1 | **22.1% / 8.6** |
| Verdict | UNDERPOWERED | UNDERPOWERED | **UNDERPOWERED** |
| **Wedge s1** | | | |
| N | 371 | 341 | **333** |
| Mean R$ | $106.7 | $88.0 | **$89.9** |
| μ_net | +$4.87 | +$3.00 | **+$3.19** |
| Power (upper) | 28.3% | 17.2% | **18.0%** |
| Years for 80% | ~6.4 | ~13.6 | **~12.4** |
| Optimistic: power / holdout-only | 96.2% / 43.0% | 86.3% / 34.2% | **87.3% / 34.9%** |
| Verdict | UNDERPOWERED | UNDERPOWERED | **UNDERPOWERED** |
| **MM fade s1** | | | |
| N | 889 | 846 | **861** |
| Mean R$ | $96.5 | $86.0 | **$88.3** |
| μ_net | +$3.85 | +$2.80 | **+$3.03** |
| Power (upper) | 36.6% | 25.3% | **27.8%** |
| Years for 80% | ~4.3 | ~6.9 | **~6.0** |
| Optimistic: power / holdout-only | 99.8% / 62.5% | 99.0% / 56.2% | **99.4% / 58.7%** |
| Verdict | UNDERPOWERED | UNDERPOWERED | **UNDERPOWERED** |

**H2/L2 sensitivities:** the double-cost cells (1R and 2R) move from UNDERPOWERED to **COST-BOUND** under both corrections. The 2R primary-cost cell stays UNDERPOWERED.

## 3. What changes in the original write-ups

**The splices inflated risk.** At a fixed 0.10R edge, a larger risk means a larger net dollar edge, so all three constructs looked more testable than they are.

**H2/L2** (`diagnostics_h2l2_power_gate_20260913/results.md`):
- The claim "20% of events risk more than $150, mostly the high-volatility months of 2025" was mostly roll splices. On C1b it is 10.6%.
- "About 22 years" is about 173. "The optimistic 0.20R would need about 2.9 years" is about 8.6.
- The exploratory variants (B–D and the ES rows) ran on the same contaminated CSV and were **not** re-run.
- `es_1min_2025_2026.csv` was not checked for the same defect.

**Wedge/MM** (`diagnostics_wedge_mm_power_gate_20260913/results.md`):
- "4–6 years" of data needed at the central edge becomes **about 6–12 years** (MM about 6, wedge about 12).
- "At the optimistic edge both would be detectable within the window" still holds: 87% and 99%.

## 4. Beyond these gates (not investigated here)

- **72 repo scripts read `mnq_1min_2026_ytd.csv`, and 116 read `mnq_1min_2025.csv`.** They include the Tier1/Tier2 research backtests and MIM-NB studies.
- Any of them run on 2025 roll weeks sees fake ±240-point bars. An FVG, sweep or ATR construct can fire on those bars.
- Any of them run on Jan–Feb 2026 is on a thin deferred contract.
- Whether that moved any sealed result is **unknown**. It needs its own check, construct by construct.

## Outputs

- `contamination_check.py` and `contamination_results.json`
- `correction_plan.md`
- `corrected_rerun.py` and `corrected_rerun_meta.json`
- `C1a/{h2l2,wedge_mm}/results.json` and `C1b/{h2l2,wedge_mm}/results.json`
- this file
