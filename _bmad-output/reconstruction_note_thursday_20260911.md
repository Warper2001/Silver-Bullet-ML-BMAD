# Reconstruction note — Thursday-short, 2026-08-27 and 2026-09-03

**Date of this note:** 2026-09-11
**Files touched (live repo):** `data/thursday_ts/decisions.csv`, `data/thursday_ts/trades.csv`
**Tool:** `tools/reconstruct_thursday_ledger_20260911.py` (imports the live `ChainedCsv`
directly, so the hash chain it produced is what the real bot would have written)
**Verified:** `tools/verify_chain.py` — all three ledgers PASS post-reconstruction, at
**7 decisions / 14 trades / 12 counterfactuals**.
**Supersedes in effect:** `reconstruction_note_thursday_20260827.md` — that reconstruction
was correct and was subsequently destroyed (see below). This one restores it.

## What happened — this is the second loss of the same data

The 2026-08-27 trade was reconstructed once already, on 2026-09-02, and merged to `main`
(`00203e1` → `e6c8682`). It was then destroyed again.

Commit `27e4e1e` ("record thursday-short live trades through 2026-09-03", 2026-09-08)
wrote the 09-03 rows onto a parent tree that **predated** the 08-27 reconstruction, and
never merged into `main` (`git merge-base --is-ancestor 27e4e1e origin/main` → false).
The live files on disk were byte-identical to that stale result plus the bot's own later
09-10 appends. Net loss:

| File | Lost |
|---|---|
| `decisions.csv` | 08-27 `ENTERED` **and** 09-03 `ENTERED` |
| `trades.csv` | both 08-27 legs (09-03's legs were restored by `27e4e1e`) |
| `counterfactuals.csv` | nothing recoverable (see below) |

Same "git-tracked while live-appended" failure class named in `ChainedCsv`'s own
docstring — now at least the fourth occurrence (gap-fade 2026-08-06, thursday-short
2026-08-27, and this). This time it arrived via a parallel orphaned commit rather than a
branch checkout, i.e. *after* the files were nominally untracked on `origin/main` — the
live checkout is 15 commits behind and never received that protection.

**Why the chain did not catch it:** `verify_chain.py` proves no row was *edited*, not that
the file is *complete*. `ChainedCsv._read_tail()` re-reads the head from disk before every
append, so the bot's 09-10 write correctly chained onto the stale tail. The result is an
internally consistent chain over an incomplete file. Nothing was asking the completeness
question — the restart pre-registration §7 now does.

## What was reconstructed, and its source

All values quoted or computed from `logs/thursday_short.log`.

**2026-08-27** (unchanged from the 2026-09-02 reconstruction; lines 50975–52767):

| Field | Value | Source |
|---|---|---|
| Entry time | 00:03 UTC | line 50988 |
| MBTU26 entry / exit | 79315.00 / 79130.00 | lines 50988, 52756 |
| METU26 entry / exit | 2516.00 / 2501.50 | lines 50988, 52760 |
| Sizing | 1 MBTU26 + 32 METU26 | line 50983 |
| Exit time / reason | 06:49 UTC / `shutdown` | lines 52752, 52756, 52760 |
| **Net P&L** | **+$64.90** | +$18.50 MBT, +$46.40 MET |

**2026-09-03** (decisions row only — its trades and counterfactuals rows survived):

| Field | Value | Source |
|---|---|---|
| ts_utc | 2026-09-03T00:02:01.528000+00:00 | line 58626, confirm timestamp |
| Marks | MBTU26 @ 77575.0 / METU26 @ 2399.0 | line 58626 |
| Sizing | 1 MBTU26 + 32 METU26 | line 58620 |
| `lr_slope20_bpd` / `lr_slope40_bpd` | 117.717 / 62.325 | copied from the surviving 09-03 rows in `trades.csv` — the bot writes one fetch to both files, so this is the real value, not a recomputation |

## What was deliberately NOT reconstructed

- **08-27 `counterfactuals.csv`** — a `shutdown` exit defers the counterfactual write until
  a later poll resolves it against the 23:05 mark (`_exit()`'s `cf_pending` path). That
  in-memory state died in the same restart, and no 23:05 mark for MBTU26/METU26 that day
  appears in the log. The real bot could not have resolved it either. This is why the
  counterfactual file is legitimately two rows shorter at 12.
- **08-27 `lr_slope20_bpd` / `lr_slope40_bpd`** — left blank. `fetch_btc_lr_slopes()` logs
  only on failure, so a successful fetch's values were never written to the log.

## Append order

The restored rows sit at the **end** of each file, after the 09-10 rows, so the files are
no longer in chronological order. This is deliberate: an append-only hash chain records
the order rows were *written*, and rewriting the file to interleave them would invalidate
every subsequent chain hash — destroying the tamper-evidence to cosmetically fix sort
order. The `thursday` column is the authoritative date.

## Effect on the decision rule

**None.** Per `_bmad-output/preregistration_kraken_thursday_short_restart.md` (sealed
2026-09-11), all seven traded Thursdays are **VOID** and the prospective accrual restarts
at N=0 from 2026-09-17. This reconstruction exists for audit completeness only — so the
ledger states what actually happened, and the voided sample is excluded by date rather
than by erasure.

True traded history: **7 Thursdays, −$569.05 net, 4W/3L** — the four winners are small
(+$64.90 to +$277.00) and the three losers include both double-stop days (−$506.55,
−$783.70), which is why a majority of winning Thursdays still nets −$569.

## Backups

Pre-reconstruction copies of all three `data/thursday_ts/*.csv`, plus `data/trades.db`,
are at `/root/backups/fleet_fixes_20260911/`.
