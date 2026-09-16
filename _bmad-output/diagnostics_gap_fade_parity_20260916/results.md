# GAP-1 parity "drift" — diagnosed and fixed (2026-09-16)

**Reported by:** `_bmad-output/diagnostics_tier1_mim_contamination_20260916/results.md`. Replaying `mnq_1min_2025.csv` through `gap_fade_live.py --replay` gave N=77, WR 64.9%, PF 2.017, Net $7,861, while the file's own docstring said the sealed result was N=78, WR 62.8%, PF 1.760, Net $6,462. The docstring says a mismatch means the live path has drifted from the pre-registered spec.

## Verdict: nothing drifted. The expectation string was wrong.

**The strategy code was never at fault, and no live behaviour changed.**

### 1. The code never produced the claimed numbers

Replaying the same CSV at **every commit** of `src/research/gap_fade_live.py`, back to the commit that created it:

| Commit | Date | Result |
|---|---|---|
| `1c117c3` (file created) | 2026-06-25 | N=77, WR 64.9%, PF 2.017, Net $7,861 |
| `07dcda4`, `7c9bc0a`, `b9f45ed`, `63de39f`, `2508825`, `3ffd122`, `1552ed8` | 06-25 → 09-11 | identical |
| HEAD | 2026-09-16 | identical |

So no change to this file ever moved the result, and the August gap-ceiling amendment (a YANK change) never touched it.

### 2. The sealed study's own functions agree with the replay, trade for trade

`compare_impls.py` runs `backtest_gap_fade.build_session_map` / `run` — the sealed Gate-0 study's own code — against the replay's loop on the same 2025 bars:

| Implementation | N | WR | PF | Net |
|---|---|---|---|---|
| Sealed study functions | 77 | 64.9% | 2.017 | $7,861 |
| `_run_replay` loop | 77 | 64.9% | 2.017 | $7,861 |

**Zero trades only in one, zero differing outcomes or P&L.** The replay is a faithful transcription of the sealed study.

### 3. The sealed trade list settles it

The Gate-0 run saved its trades: `data/reports/gap_fade_20260625_205328.csv`, 117 trades spanning 2025-01-06 → 2026-06-11.

| Slice of the sealed list | N | WR | PF | Net |
|---|---|---|---|---|
| **2025 rows only** | **77** | **64.9%** | **2.017** | **$7,861** |
| All rows (the sealed Gate-0 headline) | 117 | 62.4% | 1.761 | $9,878 |

The 2025 subset of the sealed run *is* today's replay output, to the dollar. The docstring's "N=78, WR 62.8%, PF 1.760, Net $6,462" matches neither row: it reads as the combined run's WR/PF pasted next to a 2025-ish N and an unrelated net (the combined long side was +$6,437). It was a hand-typed error made when the file was created, and it has been wrong ever since.

## The fix

`src/research/gap_fade_live.py`, documentation and the `--replay` path only — **no live logic touched** (diff: a constant, the docstring, one print, the CLI help):

1. **`SEALED_PARITY_2025`** records the true baseline (N=77, WR 64.9%, PF 2.017, Net $7,861) and cites its source, the 2025 rows of the sealed trade list.
2. **The replay now checks itself.** On `mnq_1min_2025.csv` it prints `PARITY: PASS` or `MISMATCH` with both tuples, instead of leaving a human to compare against a prose line. Verified: `PARITY: PASS — got (77, 64.9, 2.017, 7861), sealed baseline (77, 64.9, 2.017, 7861)`.
3. **The docstring and `--help`** now state the corrected baseline, note that the sealed Gate-0 headline (N=117, PF 1.761) is the 2025+2026 combined run, and record what was wrong before.

**No restart needed.** The running `trader-gap-fade` process is unaffected: nothing on its decision path changed, and the replay is an offline mode. The new text takes effect the next time the unit restarts for any reason.

## What this does not settle

- **The roll-splice sensitivity is still open.** Dropping the 27 interleaved 2025 sessions moves the replay to PF 1.885. That bounds it, but does not isolate it, because removing sessions also shifts the next session's prior-close reference. A clean measurement rebuilds 2025 from raw front-month records, keeping every session. Unchanged by this fix.
- **The sealed Gate-0 figure itself (N=117, PF 1.761) spans 2025+2026-YTD**, so it includes the back-month Jan–Feb 2026 rows and the post-cutoff window. Re-scoring it on corrected bars would need its own pre-registration.

## Outputs

- `compare_impls.py`, `compare_results.json`
- this file
