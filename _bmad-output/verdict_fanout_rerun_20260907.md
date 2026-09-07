# Cross-Instrument Fan-Out — RE-RUN ON REPRODUCIBLE CODE, 2026-09-07

**Trigger:** Alex, "re-run the fan-out for all seven instruments", following
`verdict_pl_gate1_abort_20260907.md` — which established that the June 2026 fan-out's frozen PL
reference is not reproducible from any committed state of this repo.
**Scope:** in-sample only. Window 2025-05-19 → 2026-02-28, `--structural --ml-threshold 0.0`,
1 contract, gross (structural mode zeroes commission; costs applied offline).
**No sealed holdout was read. No new hypothesis is tested. This is a record correction.**

Runs: 8 (the 7 fan-out instruments + the MNQ reference), 4-way parallel, all rc=0.

**Evidence manifest** — the harness emits timestamp-named files that do not record the
instrument, so the mapping is pinned here. All committed under `data/reports/`.

| inst | trade list | N | gross PF |
|---|---|---|---|
| mnq | `backtest_1year_20260907_151842.csv` | 151 | 1.159 |
| si | `backtest_1year_20260907_142800.csv` | 107 | 1.033 |
| ym | `backtest_1year_20260907_143800.csv` | 174 | 1.187 |
| rty | `backtest_1year_20260907_143915.csv` | 152 | 1.068 |
| hg | `backtest_1year_20260907_142514.csv` | 129 | 1.192 |
| es | `backtest_1year_20260907_151713.csv` | 147 | 1.046 |
| gc | `backtest_1year_20260907_150905.csv` | 113 | 0.920 |
| pl | `backtest_1year_20260907_151402.csv` | 129 | 1.226 |

Reproduce the table with `tools/fanout_rerun_table.py` and `tools/fanout_cost_headroom.py`.
The PL row is independently corroborated: `data/reports/pl_frozen_era_cc17543_20260907.csv`
(frozen-era code) is byte-identical to the HEAD run.

---

## 1. The corrected table

| inst | N | gross PF | gross $/trade | c\* breakeven | c\* @ netPF 1.10 | measured cost | net PF | net $/trade | tail-3-day | ex-top-3-days | corr→MNQ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **mnq** (ref) | 151 | 1.159 | $8.88 | $8.88 | $3.11 | ~$6.00 † | 1.048 | +$2.88 | 155% | −$734 | 1.000 |
| **si** | 107 | 1.033 | $4.07 | $4.07 | $0.00 | — | — | — | 941% | −$3,660 | −0.134 |
| **ym** | 174 | 1.187 | $2.68 | $2.68 | $1.17 | — | — | — | 125% | −$118 | +0.079 |
| **rty** | 152 | 1.068 | $0.88 | $0.88 | $0.00 | — | — | — | 224% | −$164 | +0.029 |
| **hg** | 129 | 1.192 | $3.23 | $3.23 | $1.47 | **$4.00** | **0.960** | **−$0.77** | 130% | −$126 | +0.244 |
| **es** | 147 | 1.046 | $1.06 | $1.06 | $0.00 | — | — | — | 474% | −$585 | +0.262 |
| **gc** | 113 | 0.920 | −$3.06 | — | — | — | — | — | −346% | −$1,544 | −0.003 |
| **pl** | 129 | 1.226 | $41.05 | $41.05 | $21.73 | **$34.00** | **1.035** | **+$7.05** | 154% | −$2,865 | +0.057 |

`c*` = the all-in $/RT cost at which net PF falls to the stated bar (computed from each trade
list, not assumed). "tail-3-day" = the top 3 exit-days' share of total gross P&L; >100% means the
rest of the sample is net negative.
† MNQ's $6.00 is the **assumed** S26-era 1-min cost card, not a measurement made for this engine —
treat its net row as indicative only. HG's $4.00 and PL's $34.00 *are* measured
(`hg_slippage_verdict_20260704.md`, `pl_slippage_verdict_20260705.md` Amendment 1).

## 2. What this replaces

None of the June numbers reproduce. Every original run's trade count is far below the re-run's on
identical data, code, models and defaults:

| instrument | June N / gross PF | re-run N / gross PF |
|---|---|---|
| hg | 95 / 1.439 | **129 / 1.192** |
| pl | 101 / 1.344 | **129 / 1.226** |
| es | 106 / 1.499 (probable) | **147 / 1.046** |
| gc | 81 / 0.944 (probable) | **113 / 0.920** |
| si | 81 / 0.930 (probable) | **107 / 1.033** |
| rty | 118 / 1.033 (probable) | **152 / 1.068** |
| mnq ref | 110 / 1.173 (probable) | **151 / 1.159** |

Only HG and PL are identified with certainty (their frozen filenames are named in the sealed
docs); the rest are matched by N/PF against the figures in
`project_yank_cross_instrument_copper`, hence "probable". The mapping does not affect the
conclusion — **every** June run is low by 20–45 trades.

**What was eliminated as the cause** (see the PL verdict for the full chain): committed code
(`git archive cc17543` reproduces HEAD's PL list *byte-for-byte*), data (`pl_1min_2025_2026.csv`
mtime 2026-06-12 predates the runs), ML model artifacts (unchanged since 2026-05-31 / 06-17),
and `ml_threshold` defaults (0.0 at cc17543 and at HEAD alike). The residue is uncommitted
working-tree code or command arguments from 2026-06-25/26 that were never recorded and cannot
be recovered.

**A forensic detail worth keeping:** the HG (`..._225218.csv`) and PL (`..._025416.csv`) trade
lists — the two lists later cited as "frozen" — both have mtime **2026-06-26 04:33**, hours after
their own `.txt` reports were written, and later than every other artifact in the batch. Something
rewrote exactly those two files after the fact. Their `.txt` reports already carried N=95 and
N=101, so the low counts originate at run time rather than in that rewrite — but a "frozen"
artifact that was silently rewritten after its run is not a frozen artifact.

## 3. What the corrected numbers say

**The cross-instrument portability thesis does not survive.** The June claim was "2 of 7
orthogonal, cost-surviving candidates (HG, PL) plus a correlated survivor (ES)". On reproducible
code:

- **HG (copper) never cleared cost in-sample at all.** Breakeven needs ≤$3.23/RT; the measured
  cost is $4.00/RT → **net PF 0.960, −$0.77/trade**. The candidate that justified a slippage
  campaign and a spent one-shot holdout was under water in-sample the whole time. Its Gate-1
  holdout FAIL (net PF 0.463) is no longer a surprise — and the holdout was spent on the strength
  of a fingerprint (gross PF 1.439) that no committed code produces.
- **ES was never a "best edge".** $1.06 gross per trade — breakeven at $1.06/RT, below any real
  MES cost. The June "netPF 1.281" is not reproducible. **This retracts the ES suggestion I made
  earlier in this session.**
- **PL is the only instrument with real cost headroom**, and it still fails: $41.05 gross/trade
  against $34.00 measured leaves net PF 1.035, under the 1.10 bar its own slippage seal required
  (which needs ≤$21.73/RT).
- **SI, YM, RTY, GC are dead on arithmetic** — $4.07, $2.68, $0.88 and *negative* gross per trade
  respectively, before any cost.
- **MNQ, the reference, is the second-best of the eight** (net PF 1.048, +$2.88/trade at assumed
  cost). Consistent with `project_edge_headroom_screen_20260615`: the structure works best on the
  instrument it was derived from.

**The strongest single finding — and it is unanimous:**

> **All eight instruments have a NEGATIVE ex-top-3-days gross P&L.** Every tail-3-day share is
> ≥125% (GC is negative outright). Remove each instrument's three best exit days and *not one*
> of them is profitable — before costs.

This is not a platinum quirk. Across eight instruments, six of them uncorrelated with MNQ, the
YANK structural engine produces edges that are entirely tail-carried at this sample size. It
matches the independent finding in `project_post_r3_options_pass_20260906` that neither leg of
MNQ's 24-hour session carries robust drift once its top days are removed.

## 4. Consequences

1. **`project_yank_cross_instrument_copper`'s headline numbers are retracted.** The track was
   already closed with "zero live candidates"; it is now closed for a stronger reason — the
   in-sample evidence that opened it was unreproducible, and on reproducible code no instrument
   qualifies.
2. **Do not cite any June 2026 fan-out figure again.** Cite this table.
3. **PL stays closed** at its precondition (`verdict_pl_gate1_abort_20260907.md`); its sealed
   holdout remains **unspent**.
4. **`--structural` needs a fix.** It does not neutralize the Topstep trailing floor
   (`check_trailing_dd`, `tier2_streaming_working.py:1375`, `$2,000` default) despite existing to
   strip dollar-scaled path-dependent gates. It was tested and does not affect these results
   (restoring the pre-`e56bc6a` halt changed nothing), but a scale-invariance mode with a live
   dollar gate inside it is a latent defect.
5. **Reproduction gates become mandatory at freeze time, not at use time.** A frozen reference is
   only frozen if a committed SHA regenerates it. Freeze the command *and* the SHA, and verify
   regeneration when freezing — this batch was frozen in June and only discovered to be
   irreproducible in September, after it had authorized a measurement campaign, a slippage PASS,
   a combine-fit gate, a spent copper holdout and a platinum holdout authorization.
