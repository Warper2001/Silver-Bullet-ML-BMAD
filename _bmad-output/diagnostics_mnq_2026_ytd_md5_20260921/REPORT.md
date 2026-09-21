# mnq_1min_2026_ytd.csv: md5 mismatch vs the MIM-NB seal (2026-09-21)

**Verdict: cosmetic for MIM-NB. The current file reproduces the sealed 2026 trades exactly through the sealed OOS end (2026-05-19).**
Script: `check_ytd_equivalence.py` (print-only, exits 0 when identical). Re-run:
`.venv/bin/python _bmad-output/diagnostics_mnq_2026_ytd_md5_20260921/check_ytd_equivalence.py`

## What was found
| | sealed | now |
|---|---|---|
| md5 | `4ec175dd…` (pinned in `preregistration_mim_noise_bands.md`) | `30bc05a8…` |
| header | not recorded | `timestamp,open,high,low,close,volume,notional` |
| last bar | before the 2026-05-19 OOS end | 2026-06-11T23:32 UTC |
| mtime | seal ≈ 2026-06-11 04:53 | 2026-06-11 23:31 (after the seal, the same day) |
| git | none | one add, `744642e` (2026-08-06, "track the frozen sigma warmup series"), 127,551 lines |

The sealed bytes are **not recoverable**: no prefix of the current file hashes to `4ec175dd`, and git has no earlier version. So the test is
functional instead of byte-level.

## Test
The sealed engine (`study_mim_nb_catstop.py`, sha `210518d6…`, `run_catstop` lifted verbatim by AST) run on the current file, compared with the
2026 rows the sealed run itself wrote to `data/reports/mim_nb_catstop_s{250,500}_pooled.csv` on 2026-06-11, before the file changed.

| variant | sealed 2026 trades | current file, same days | sum of points | identical |
|---|---|---|---|---|
| S=250 | 50 (last day 2026-05-19) | 50 | 347.00 vs 347.00 | **yes** |
| S=500 | 50 | 50 | 855.75 vs 855.75 | **yes** |

The current file also yields 13 (S250) and 12 (S500) further trades after 2026-05-19: the bars appended after the seal.

## Reading
- The change is an **extension through 2026-06-11 plus a `notional` column** (the sigma-warmup series the live bot needs). The loader uses only
  `timestamp, open, high, low, close, volume`, so the column cannot affect the engine.
- The spent Gate-1 OOS evidence for MIM-NB stands on this file. "The sealed OOS" should be described as *functionally* identical, not byte-identical.

## Limits
- **Verified for MIM-NB only.** Other seals that pin the same md5 (e.g. MIM-Classic, which the noise-bands seal says shares its data files) were not checked.
- Equivalence is shown for the strategy's inputs (RTH OHLCV) through 2026-05-19, not for every column or every row.
- The file still has the known pre-2026-03-12 back-month contract issue (AGENTS.md). That is a data-quality fact about the OOS window, unchanged by this finding.
