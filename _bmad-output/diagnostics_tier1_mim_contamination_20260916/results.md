# Tier1 / MIM-NB (and GAP-1) contamination triage — 2026-09-16

Follow-on to `_bmad-output/diagnostics_h2l2_contamination_20260914/` and `_bmad-output/diagnostics_tier2_contamination_20260914/`, which found two defects in the MNQ 1-minute CSVs: 2025 roll weeks interleave two contracts minute by minute (27 RTH sessions, fake ±240-point bars), and Jan–Feb 2026 in `mnq_1min_2026_ytd.csv` is the deferred MNQM26 contract, not the front month. The holdout's first ~2 weeks are copies of the same rows.

**Method:** saved trade lists from the original runs were tagged by segment (`tag_results.py`, output `tag_results.json`). No replay, no bars and no holdout files were read for the MIM-NB section. The GAP-1 section re-ran a pure, no-write parity replay on 2025 data only.

## Verdict by consumer

| Consumer | Live? | Exposure | Does the conclusion change? |
|---|---|---|---|
| **MIM-NB** gates 0/1 and cat-stop | **yes** (1ct combine) | 4–18% of P&L on contaminated segments | **No — removing it improves every file** |
| **Tier1** backtests | no | all 2025-window studies include the 27 sessions | Not load-bearing; already superseded |
| **GAP-1** (adjacent, checked because it reads the same file) | **yes** | sealed Gate-0 ran on the contaminated 2025 file | **Unresolved — and its parity check no longer reproduces the seal at all** |

## 1. MIM-NB: exposed, but the contamination worked against it

Segments: `roll_week_2025` (interleaved sessions), `JanFeb26_back_month`, `Mar1-11_back_month` (holdout rows from the same deferred contract), versus clean 2025 and `Mar12+_front_month`.

| Result file | N | PF as published | Contaminated N | Contaminated pts | Clean-only N | **Clean-only PF** |
|---|---|---|---|---|---|---|
| `mim_nb_gate0_v1_2025` | 182 | 1.154 | 2 | −187.2 | 180 | **1.185** |
| `mim_nb_gate0_v2_2025` | 114 | 1.509 | 2 | −192.0 | 112 | **1.556** |
| `mim_nb_gate1_v1_2026oos` | 82 | 1.384 | 22 | +127.0 (9.3%) | 60 | **1.491** |
| `mim_nb_gate1_v2_2026oos` | 62 | 1.659 | 17 | +385.0 (18.2%) | 45 | **1.715** |
| `mim_nb_catstop_s500_pooled` | 164 | 1.541 | 19 | +193.0 (4.2%) | 145 | **1.586** |
| `mim_nb_catstop_s250_pooled` | 164 | 1.382 | 19 | −217.5 | 145 | **1.476** |

**Reading:**
- **Gate 0 (2025 in-sample):** only 2 trades of 182 / 114 fall in interleaved sessions, and both are losers. The fake bars cost MIM-NB points; they did not manufacture its edge.
- **Gate 1 (2026 OOS):** most of it is already front-month. 60 of 82 (v1) and 45 of 62 (v2) trades are after the March 12 roll, and those subsets are **stronger** than the published totals.
- **The Jan–Feb back month did flatter v2** (14 trades, +713 pts, PF 2.69 in that segment), but dropping it still leaves PF 1.715 on 45 clean trades.
- **Every file's clean-only PF is higher than its published PF.** MIM-NB's verdicts stand.

**Caveat:** these are the original runs' saved trades, re-scored. An exact corrected re-run would need front-month bars for Jan–Feb 2026 and the holdout period, so it needs its own pre-registration and `ACCESS_LOG` entry, exactly as YANK's did (seal `da82cfc`). Recommended only if someone wants to quote MIM-NB's OOS numbers precisely; the direction is already clear.

## 2. Tier1: exposed, not load-bearing

- 16 `backtest_tier1_*.py` scripts read `mnq_1min_2025.csv`. Any 2025-window run includes the 27 interleaved sessions (3.13% of the file's rows).
- **No live entry file references Tier1**, and no live seal depends on it.
- The saved Tier1 results (`data/reports/tier1_*.json`) are from 2026-04-15, before the 2026-05-20 methodology reset that already marks earlier performance claims tentative. They are bar-classification studies (11k–16k classified bars, 65–81% "win rates"), not executable trade backtests.
- **No re-runs done or recommended.** If a Tier1 claim is ever revived, rebuild its bars front-month first.

## 3. GAP-1: live operation is safe, but two things need attention

**Live operation is unaffected.** `gap_fade_live.py` reads `mnq_1min_2025.csv` only under `--replay`, its parity mode. The live path takes streaming bars, and since 2026-09-13 has its own bar recorder.

**(a) The parity check no longer reproduces the seal.** Replaying the unchanged 2025 CSV today:

| | N | WR | PF | Net |
|---|---|---|---|---|
| Today's replay | 77 | 64.9% | 2.017 | $7,861 |
| The tool's own sealed expectation | 78 | 62.8% | 1.760 | $6,462 |

The script says a mismatch means the live path has drifted from the pre-registered spec. **This is not a contamination effect** — it is the same file the seal used. Candidates: the 2026-08-19 gap-ceiling denomination change, or the Z26 roll edits. It deserves its own look.

**(b) Sensitivity to the contaminated sessions.** Dropping the 27 interleaved dates from the 2025 file moves the replay to N=77, WR 62.3%, **PF 1.885**, Net $6,950.
- **This is not a clean isolation.** GAP-1 fades the opening gap against the prior session's close, so removing whole sessions also changes the next session's reference. It bounds the sensitivity (PF moves about 0.13) rather than measuring the contamination alone.
- A proper correction rebuilds 2025 from raw front-month records, keeping every session. That is a separate, pre-registered job.

## 4. Other files checked

| File | Finding |
|---|---|
| `mnq_1min_2023_sepnov.csv`, `mnq_1min_2024_sepnov.csv` | Clean. 3–4 jumps over 100 points, on event dates (FOMC, earnings), not clustered in roll weeks. |
| `data/mim_x/mnq_1min_2021_2024_frontmonth.csv` | No interleaving. Carries one unadjusted roll gap per quarter (194–275 pts at the Globex open on roll day), as recorded on 2026-09-14. |
| `es_1min_2025_2026.csv` | No 2025 interleaving signature (checked separately 2026-09-16). Its 2026 rows are unchecked, and there is no raw ES source in the repo. |
| `mnq_5min_2024.csv` | **Suspect and unused.** 886 close-to-close jumps over 100 points in 33.8k bars, concentrated in thin overnight hours (02:00–07:00 ET) at normal 5-minute spacing. No script reads it. Left alone; do not adopt it without an audit. |

## Outputs

- `tag_results.py`, `tag_results.json`
- this file
