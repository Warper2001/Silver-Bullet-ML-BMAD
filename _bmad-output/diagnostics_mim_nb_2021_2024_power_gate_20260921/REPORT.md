# Power gate — one-shot test of the sealed MIM-NB engine (S=250) on MNQ front-month 2021-2024 (2026-09-21)

**Verdict under the pre-committed rule: UNDETERMINED.** At the best-case effect the test sits on the 80% line
(power 0.75 to 0.83). At any effect below that best case it is underpowered. Outcome-blind: the engine was never run on the
target; the file was read only to count sessions. `data/sealed_holdout/` not touched.
Files: `power_gate.py` (rule pre-committed in its docstring), `power_gate_output.md`, `power_verdict.json`, `sensitivity_size_calibrated.json`.

## The unseen window is smaller than I said last turn
998 sessions have both a 09:31 and a 16:00 bar. **124 fall in 2023 Sep-Nov and 2024 Sep-Nov, which `study_mim_noise_bands_gate0.py` already ran
as non-gating diagnostics**, so they are not unseen. Removing them, a 14-session sigma warm-up and 16 roll-boundary sessions leaves **844 usable
sessions**. At the two observed trade rates (0.414/session live, 0.509/session in the 2025 dev window) that is **350 to 430 trades**, not the ~456 I quoted.

## Result
Effect anchors are per-trade standardised (d = mean/SD, net of the sealed 1.12 pt cost). Power is simulated (20,000 draws) from the 2025 dev
trade shape, which is extremely right-skewed (skew 5.5, kurtosis 47.5: 13 cat-stops at -250 pts, five days carry the profit), so it is not assumed normal.

| effect anchor | d | mean | trades for 80% power (simulated) | power @350 trades | @430 trades |
|---|---|---|---|---|---|
| **ceiling: 2025 dev, in-sample** | 0.115 | +14.1 bp | **395** | **0.75** | **0.83** |
| 0.75 x ceiling | 0.086 | +10.6 bp | 737 | 0.48 | 0.57 |
| 0.50 x ceiling | 0.057 | +7.1 bp | 1,737 | 0.24 | 0.29 |
| 0.25 x ceiling | 0.029 | +3.5 bp | 7,202 | 0.09 | 0.10 |
| third-party report, +2.6 bp | 0.021 | +2.6 bp | 13,363 | 0.06 | 0.07 |
| live ledger, +0.31 bp | 0.004 | +0.5 bp | > 20,000 | 0.03 | 0.03 |

- **Smallest effect the window can see** (normal approx., 80% power): d = 0.12-0.13, i.e. **1.04-1.16x the 2025 in-sample effect**.
- **The verdict depends slightly on the test.** The nominal t-test is conservative on this skewed shape (size 2.5%, not 5%). With a size-calibrated
  critical value (bootstrap null, t* 1.39-1.42) power at the ceiling is 0.84 / 0.90, which would read POWERED there; at 0.75x it is 0.60 / 0.67, at 0.5x 0.34 / 0.38.
  Either way, every anchor below the ceiling fails.
- **Clustering** (DEFF 1.5) raises the trades needed at the ceiling from about 470 to 700 (normal approx.), which would make it clearly underpowered.
  The headline used DEFF 1.0, the most favourable setting.

## What this means
- The ceiling is the *in-sample* 2025 estimate: it comes from the window the spec was gated on and rests on five days, so it is upward-biased. The test
  can confirm a pre-2025 edge only if the true edge is about as large as that estimate. If it is three-quarters as large, it is roughly a coin flip.
- **A null result would not show the edge is absent**, only that it is below about d 0.1. **A pass would show an edge existed in 2021-24, not that it decayed.**
  A decay claim needs a contrast (2021-24 vs 2025+), and the live window (N=29) has no resolving power against it.
- The report's +2.6 bp effect is out of reach in every window on hand (about 13,400 trades).

## Caveats
- The 2021-24 file is true 1-minute time bars; the sealed 2025 dev file is dollar-aggregated despite its name. The spec was developed on the second and
  the live bot runs on the first, so how well the 2025 effect transfers is unknown.
- Trade rate per session is measured on 2025 and on the live ledger, not on the target (counting entries would mean running the engine on it).
- MIM-X2 already looked at this window for a different rule (Baltussen intraday momentum): its per-year MNQ means run +$0.74 (2021), +$6.96 (2022), -$4.79 (2023), -$11.91 (2024).
  That is a different signal, so it does not un-see the window for MIM-NB, but it is weak prior evidence that the family was already weakening by 2023-24.
- 2.0-year-style leashes do not apply here: the data is finite, so the limit is the window itself.

## If you decide to run it
It needs a pre-registration first (sealed engine sha `210518d6…`, target file sha, the excluded months, S=250, cost 1.12 pt, one-sided alpha 0.05, a
size-calibrated test computed on the target's own trades, and descriptive PF / ex-top-5-days reported but not gating), citing this artifact and its commit.
My view: it is a low-odds, one-way test that spends the last unseen historical window, and its answer would not change any live decision, which is driven by the
$618 buffer and the floor rules, not by an edge estimate. I would not spend it unless you want the pre-2025 baseline for its own sake.
