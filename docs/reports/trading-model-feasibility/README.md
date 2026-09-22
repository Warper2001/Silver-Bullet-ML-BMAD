# Trading-model readiness evidence — 2026-09-22

The current evidence is [the reviewed audit](run-20260922-reviewed/report.md), with machine-readable observations in `run-20260922-reviewed/report.json` and artifact hashes in `run-20260922-reviewed/COMPLETE.json`.

**Disposition: HOLD_DATA; power UNASSESSABLE; GPU training NOT_MEASURED. No strategy test or deployment is authorized.**

## Measurements

| Explicit input | Rows | Observed weekday dates | Dates with any full regular RTH grid | Elapsed CSV audit |
| --- | ---: | ---: | ---: | ---: |
| MIM-X contract-labeled minute history | 2,028,965 | 1,481 | 1,415 | 17.940 s |
| Legacy 2025 CSV, known splice defects | 289,230 | 260 | 207 | 2.152 s |
| Diagnostic reconstructed 2025 CSV | 281,645 | 260 | 207 | 2.119 s |

All three inputs had zero structurally invalid OHLCV/timestamp rows. This **does not** clear the known data defects. The primary file has 43 dates with more than one observed contract. Both 2025 files lack contract columns, so their mixed-contract counts are unassessable. A parser cannot discover contract substitutions that preserve valid numbers.

Both timestamp hypotheses produce the listed regular-grid date counts; this agreement does not prove the label convention. Counts are not independent evidence and must not be summed across files/contracts/windows. A regular weekday grid is not an authenticated historical exchange calendar; complete holidays/early closes and entirely absent sessions are not settled here.

Measured primary and reconstructed file hashes match the prior documentary fingerprints. The primary history is already researched; it is not generally untouched. The reconstructed file's full-session minute-count contract choice is retrospective and cannot become a causal training roll rule without further work.

The reported seconds measure elapsed CPU CSV processing, excluding the separate input-hash passes. They are not model-training timings. Python 3.12.3 reports four logical CPUs; the research interpreter has no torch or transformers, and `nvidia-smi` is unavailable. No GPU is inferred from that inspection and no GPU throughput or training cost is estimated.

## Provenance and verification

- Readiness-only preregistration: `ff7fbeb74491704ea0a5c36e5ac1aff2d5cf633f`.
- Reviewed audit code: `cfd191e548cd6a731ec7576ff188d1a1807be602`; the report also binds the actual script bytes by SHA-256. The CLI records the runner-supplied revision as unverified; this run supplied the freshly committed revision.
- Independent review produced seven actionable findings. Input-scope, UTC alignment, strict CSV parsing, invalid-price duplicate detection, invalid-row diagnostics, revision qualification and interrupted-publication fixes were implemented. Self-review also changed missing-contract mixed-date counts from zero to unassessable.
- Verification: 36 synthetic tests; targeted mypy and flake8; both report hashes checked against COMPLETE.json; no strategy returns or performance tests.
- The earlier `run-20260922/` is retained as **superseded pre-review diagnostic output**. It lacks a completion marker and must not be consumed as the reviewed deliverable. In particular, its missing-contract mixed-date zeros and unqualified revision label were corrected in the reviewed run.

## Reproduce the reviewed measurement

From the isolated worktree, with the committed code revision above and the main checkout's research interpreter:

```bash
nohup nice -n 10 /root/Silver-Bullet-ML-BMAD/.venv-research/bin/python \
  tools/trading_model_readiness.py \
  --input /root/Silver-Bullet-ML-BMAD/data/mim_x/mnq_1min_by_contract.csv \
  --input /root/Silver-Bullet-ML-BMAD/data/processed/dollar_bars/1_minute/mnq_1min_2025.csv \
  --input /root/Silver-Bullet-ML-BMAD/.claude/worktrees/gapfade-splice-sensitivity/_bmad-output/diagnostics_gap_fade_splice_20260916/mnq_1min_2025_frontmonth.csv \
  --output-dir docs/reports/trading-model-feasibility/run-20260922-reproduction \
  --source-revision cfd191e548cd6a731ec7576ff188d1a1807be602 \
  > /tmp/trading-model-readiness-reproduction.log 2>&1 &
wait $!
```

Reruns require a new destination and must use the actual code revision after any code change. Run the repository's status and divergence checks before obtaining that revision. Data hashes can be compared across runs; elapsed timing is not reproducible. `wait` keeps the background process attached to an automation runner's lifetime without changing its `nohup` protection.

The [runbook and future comparison protocol](../../trading-model-feasibility.md) list the evidence needed to move beyond this readiness stage. None of the 15-minute/one-contract proposal, future risk thresholds or pretrained model choices has been adopted into a live strategy.

## Tree cleanup preservation

Cleanup commit `f2524c0d12b36ed61175cceb9da766e4d9942ae8` ignores the research virtual environment and live `data/evfade_fomc/` accrual; both remain on disk. Three pre-existing acquisition files were preserved in named stash `3ee76834c17cd0a08d96e9534c3f2979f1b57c69`, rather than introducing tests with missing legacy imports into the active tree. Their original untracked bytes are in that stash's third parent and are recoverable. No live ledger or environment was deleted.
