# Reconstruction note — Thursday-short, 2026-08-27

**Date of this note:** 2026-09-02
**Files touched (live repo, not this worktree):** `data/thursday_ts/decisions.csv`, `data/thursday_ts/trades.csv`
**Tool:** `tools/reconstruct_thursday_aug27.py` (imports the live `ChainedCsv` class directly, so the hash chain it produced is what the real bot would have written)
**Verified:** `tools/verify_chain.py --file data/thursday_ts/trades.csv --file data/thursday_ts/decisions.csv` — both chains PASS post-reconstruction.

## What happened

`trader-thursday-short.service` entered a real short (1 MBTU26 + 32 METU26) at
2026-08-27 00:03:41 UTC. The service restarted at 06:49:54 UTC that same day; its
shutdown handler force-closed the position at 06:49:50-51 UTC (`reason="shutdown"`),
6h46m into what the strategy design calls a full 24h hold to 23:05 UTC. This is the
only such early-shutdown exit in the log's entire history — a one-off restart landing
inside an open-position window, not a recurring pattern.

Neither event ever reached `decisions.csv` or `trades.csv`: both files' last row was
still 2026-07-23, with a file mtime of 2026-08-28 16:05 — a day after the Aug 27 trade
closed. Both files are git-tracked while being live-appended, the same failure class
already named in `ChainedCsv`'s own docstring (and responsible for the 2026-08-06
gap-fade incident): a branch checkout or merge in the shared repo can revert a
live-appended file out from under the process writing to it.

## What was reconstructed, and its source

Every value below is quoted or computed directly from `logs/thursday_short.log`
(line numbers as of 2026-09-02):

| Field | Value | Source |
|---|---|---|
| Entry time | 00:03 UTC | line 50988, confirm timestamp |
| MBTU26 entry / exit | 79315.00 / 79130.00 | lines 50988, 52756 |
| METU26 entry / exit | 2516.00 / 2501.50 | lines 50988, 52760 |
| Sizing | 1 MBTU26 + 32 METU26 | line 50983 |
| Exit time / reason | 06:49 UTC / `shutdown` | lines 52752, 52756, 52760 |
| MBT ret/pnl | +23.32bps / +$18.50 | recomputed from entry/exit, matches log's `+23.3bps $+18.50` at log's 1-decimal display precision |
| MET ret/pnl | +57.63bps / +$46.40 | recomputed from entry/exit, matches log's `+57.6bps $+46.40` |
| **Net P&L, this Thursday** | **+$64.90** | sum of both legs |

## What was deliberately NOT reconstructed

- **`lr_slope20_bpd` / `lr_slope40_bpd`** — left blank on both new rows.
  `fetch_btc_lr_slopes()` only logs on failure (a WARNING, absent that day); on
  success it writes silently to the CSV and nothing else. The actual values were
  never in the log to recover — blank is the honest answer, not a guess.
- **`counterfactuals.csv`** — left untouched. A `shutdown`-reason exit defers its
  counterfactual write until a later poll resolves it against the 23:05 mark
  (`_exit()`'s `cf_pending` path); that in-memory pending state was itself lost in
  the same restart that broke the ledger, so there is no code path — real or
  reconstructed — that could have produced this row. No 23:05 mark for
  MBTU26/METU26 that day appears anywhere in the log to reconstruct it from.

## Effect on the decision rule

N toward the strategy's pre-registered N≥30-Thursdays PASS/FAIL bar is now **5**,
not 4. The realized +$64.90 for 2026-08-27 is a genuine but **understated** result
relative to what the design's full 24h hold would have shown — it reflects 6h46m of
market exposure, not 23h — and should be read with that caveat if it is ever
compared against the other four Thursdays' full-hold numbers.

## Backups

Pre-reconstruction copies of all three `data/thursday_ts/*.csv` files were taken
before this ran; available on request (not committed — they are a superset of the
git history for the same rows and add nothing to the repo).
