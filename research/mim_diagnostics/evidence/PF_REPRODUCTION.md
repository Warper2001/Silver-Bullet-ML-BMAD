# PF shortlist transformations

These instructions reproduce descriptive transformations only. They do not run strategies, reassign exits, access protected data or query prospective P&L. The source is `research/mim_diagnostics/runs/20260912T194421-run-3346059bd9/trades.csv` in the diagnostics worktree. Require its SHA256 to equal `a32f91fcbfdbcb75db5698a095e1afbe5550563c505a48e3805281aa8b5d4547`; also check the source completion hash in `pf-opportunity-snapshot-20260912.json`. Read the existing `summary.json` only after matching its recorded hash.

## Exact historical CSV recipe

1. Read rows in source order using Python `csv.DictReader`. Convert `net`, `gross`, `costs`, `sampled_mfe`, `sampled_mae` with Python `float`. Empty/missing/non-numeric values abort; do not coerce to zero or drop rows. Source integrity checks establish these particular inputs are finite. A generalized reimplementation must reject nonfinite values as well.
2. Assign `outcome = winner` for net > 0, `loser` for net < 0, otherwise `flat` (exact zero). Assign favorable flag `sampled_mfe > costs`; assign adverse flag `sampled_mae > 0`. Both are gross-dollar excursions over the original full life, including entry/terminal fills and the originally permitted closes. No new intrabar reconstruction occurs.
3. Iterate reasons in `['ALL'] + sorted(unique exit_reason)`. For each reason, iterate outcomes in `['ALL','winner','loser','flat']`. Select rows whose reason and outcome match unless that axis is ALL. Preserve source order inside each group. Keep empty groups. The resulting 16 rows include margins; they are not a mutually exclusive 16-part partition.
4. Output these columns, in this exact order:

| Column | Formula on selected group g |
|---|---|
| exit_reason, outcome | Loop labels |
| trades | `len(g)` |
| gross, costs, net | Python `sum(t[field] for t in g)` |
| winning_dollars | `sum(max(t['net'], 0) for t in g)` |
| losing_dollars | `-sum(min(t['net'], 0) for t in g)` |
| sampled_favorable_beyond_cost_count | Sum of favorable Boolean flags |
| sampled_adverse_gross_count | Sum of adverse Boolean flags |

5. Serialize with Python `csv.DictWriter`, its default dialect with `lineterminator="\n"` and the header, opening the new output with `newline=''` and exclusive mode `x`. Never overwrite the versioned artifact. Use a new directory beneath package `runs/` for a reproduction. The initially generated default CRLF was normalized to LF before commit. Python's original integer/float serialization and signed-zero behavior should be retained if comparing bytes; compare numerical values at absolute tolerance $1e-8 for accounting checks.
6. The ALL/ALL row must reconcile to 801 trades and $21,889.76 net. For a complete partition, either sum the three non-ALL exit reasons at outcome ALL, or sum winner/loser/flat at reason ALL. Each must reproduce ALL/ALL. Winning dollars are $98,516.12; losing dollars $76,626.36; `net_pf = winning_dollars / losing_dollars`. This baseline has a positive denominator; do not replace undefined PF with a finite value in other inputs.

The verification record was produced by independently rebuilding each keyed group from the hash-bound CSV and checking every numeric output field, not just grand totals.

## Portfolio metadata snapshot recipe

`pf-opportunity-snapshot-20260912.json` retains the two literal SQL queries. Open the original `data/trades.db` through a SQLite file URI with `?mode=ro`, issue `BEGIN`, and execute both queries in that single read transaction. Record UTC transaction start and end timestamps. No P&L column is queried.

- Map the grouped query's six outputs to `trader_id, write_mode, execution_mode, rows, timestamp_min_text, timestamp_max_text`. Preserve SQL NULL as JSON null. The MIN/MAX values are explicitly lexical source-text extrema, not normalized chronological bounds.
- Iterate the second query's rows in increasing ID. Decode metadata with `json.loads(metadata)` when nonempty, else `{}`. In the executed snapshot all decoded values were dictionaries. Invalid JSON or a non-dictionary aborts this exact transformation rather than silently suppressing a marker. `Counter.update(obj.keys())` counts each key once per row for each trader; values are not emitted.
- If `obj.get('backfill')` or `obj.get('correction')` is truthy, retain only `id, trader_id, timestamp, execution_mode, backfill_marker=bool(obj.get('backfill')), correction_marker=bool(obj.get('correction'))`. This intentionally tests Python truthiness, not a separately validated marker schema: the artifact flags rows for reconciliation and never determines eligibility from that flag alone. Unflagged rows are not thereby proven authentic.
- Serialize the captured aggregates, metadata-key counts and reduced markers to JSON with indentation 2 and a trailing newline. Preserve the original observation timestamps. No raw metadata notes, credentials, broker payloads or prospective outcomes are copied.

A live database can append or receive corrections after the observation. Repeating these queries produces a **new timestamped observation**, not a reproduction of prior historical database state. The prior JSON is the retained observation evidence; no full database backup or raw per-trade metadata snapshot was retained, so later reviewers cannot independently reconstruct every original row from the artifact alone. Do not imply that source-file hashing fixes this limitation. Investigate disputed rows against their authentic archived trade/order evidence in the bounded reconciliation route.

## Route evidence and source hashes

`pf-route-evidence-20260912.json` manually transcribes the cited saved reports' execution figures and candidate dispositions. Each path is checked against its recorded SHA256 before use; its claims are traceable to those documents, not recomputed broker results. The original audit did not establish running-process versions. Changing source documents later does not revise the recorded observation; preserve the older source revision when resolving any mismatch.
