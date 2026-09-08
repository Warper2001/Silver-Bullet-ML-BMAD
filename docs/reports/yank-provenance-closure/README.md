# YANK provenance closure pass

**HOLD_VALIDATION — data suitability remains BLOCKED.** All six questions have a supported disposition; all remain unresolved at the level of authenticated provenance for the exact frozen input. This local pass found a historical writer candidate that reproduces the frozen CSV **byte-for-byte**, but no original invocation or raw acquisition receipt binding that writer to the frozen hashes.

This is an incremental addition to the preserved [provenance report](../yank-bar-provenance/report.md) and [pilot completion](../yank-pilot-completion/README.md), based on clean starting revision `b82c098719de1646b787a8b7f69b6e82435122ae`. Prior conclusions and artifacts are unchanged. [Claims](claims.json) separate observed facts, limitations and missing evidence for each question; [source identities and line excerpts](sources.json) and the [search inventory](search-inventory.json) make their references reviewable.

## Newly supported evidence

The reachable `1minuteDB` history contains `scripts/generate_1min_dollar_bars_2025.py` at full revision `d54b1a4a07a31dabe1f5bfd715793dbce9cfe4b8`. It is absent from the current worktree and was not among the prior saved acquisition/converter search paths. The [retained source](historical-writer.py.txt), lines 17–19, 72–135, specifies factor 20, threshold 50,000,000, the exact target filename, whole-record accumulation and last-constituent timestamp emission. This is a stronger candidate than the previously inspected HDF5 converter or live tick transformer.

At the same revision, `MIGRATION_EXECUTION_STATUS.md` lines 11–15 reports 351,628 source records and 289,230 generated bars; `scripts/migrate_to_1min_2025.py` lines 76–84 calls the generator. These historical assertions corroborate intended use. They contain no original input/output hash binding or execution attestation. Exact excerpts and Git identities are retained in [sources.json](sources.json).

The [controlled candidate check](check-historical-candidate.py) runs that historical generator's unchanged `main` with only the hash-verified 2025 extract supplied through a substituted JSON loader and with output confined to a temporary directory. It does not load the mixed-year JSON or run a strategy. The resulting 22,318,944 bytes have SHA256 `3f20ec70885cdee6b48e6c5c7ed3254dd4cc8ce7bd8533696c5e461c75fb7822`, exactly the frozen CSV hash. The current `/root/mnq_historical.json` also matches the repository raw hash `e7aed8ba786436ba80f4b081d3b7a4ee97b06bd3e8cc347c035547ec57dcb924`.

These checks establish current file identity and historical-code compatibility. They do not establish who ran the original transformation, which raw bytes existed at that time, or where those bytes originated. The candidate uses floats and original input order; the prior independent reconstruction uses Decimal and explicit ordering. Agreement for this input is not a general equivalence claim.

## Dispositions and exact blockers

| Question | Status | Evidence and remaining requirement |
| --- | --- | --- |
| Authenticated raw origin | Unresolved | The candidate reads a local file. Recover an authenticated provider receipt/response chain binding the raw hash to endpoint/version, request ID, symbols, dates, interval, session and adjustment settings; include assembly and response hashes if combined. |
| Exact historical writer | Unresolved | Byte-compatible historical candidate and matching count assertions now located. Recover a contemporaneous invocation/build record binding source revision/hash, command, environment and parameters to both frozen input and output hashes. |
| Timestamp interval meaning | Unresolved | Last-constituent label inheritance remains established. Recover versioned semantics for the authenticated source endpoint/export, including start/end boundaries, clock/timezone, inclusivity and partial/corrected bars. Saved chart/ASCII documentation cannot select the frozen interpretation. |
| Observation availability | Unresolved | Historical file processing records no original delivery time. Recover source-record receipt/publication timestamps, clock and finalization/revision policy. A documented release guarantee must remain distinct from actual client arrival. |
| Contract mapping and roll/adjustment policy | Unresolved | 5,583 mixed-label bars remain established. Candidate drops `Contract` and adds no price adjustment. Recover effective-dated provider/exchange mappings, request symbols, roll/overlap/deduplication rules and upstream adjustment settings/factors bound to the frozen acquisition. |
| Session and acquisition completeness | Unresolved | All retained 2025 constituents reconcile; this does not certify acquisition coverage. Recover the original session/calendar, requested windows, page counts and hashes, terminal checkpoints, retry/failure and assembly records, then reconcile expected eligible intervals. |

The candidate's reported “completeness” is the fraction of **output bars reaching the dollar threshold** (lines 141–144). It does not measure missing requests, pages or source minutes. Its year filter, zero-high/low checks and exception skipping (lines 38–64) also do not define a session policy. The historical migration note's 100% assertion cannot close acquisition completeness.

The [saved pilot request](sources.json) identifies a separate Databento `GLBX.MDP3` / `MNQM5` MBO request for May 19–31, 2025. Its request and native metadata do not authenticate the original raw `MNQM25` alias, continuous contract policy or timing. Matching fields, prices or absence of contrary evidence are insufficient for closure.

## Search and verification scope

The [inventory](search-inventory.json) records exact commands, outputs, exit codes and full local ref revisions. It extends the prior saved search to reachable generator history, candidate references and migration records; reviews local downloader/client/transformation code; inventories acquisition metadata filenames under `data/yank`, `data/raw` and `data/processed`; and checks new main-worktree commodity acquisition code for exact frozen-input references. The seven metadata candidates in that inventory all belong to the separate Databento pilot. No original frozen-file receipt or hash-bound writer invocation was located in these searches. This is a bounded negative result, not a claim about every filesystem location, unreachable Git object or remote archive. No remote refs were fetched.

[Verification](verification.json) records 137 successful comparisons against existing input/artifact hashes, both preserved provenance runs, original audit and saved repeat, integrated audit findings, both corrected native outputs, frozen replay archives and model identity. Prior research pins and saved search artifacts match their recorded hashes. The independent lineage checker passed under optimized Python on preserved `verified-run1`: 289,230 rows, 351,628 constituents used once each, 29,520 multi-record bars and 5,583 mixed-contract bars, with every CSV field and inherited label matching. [All 44 provenance tests passed](provenance-tests.xml), with two existing Pydantic deprecation warnings.

Reproduce the small checks from `/root/Silver-Bullet-ML-BMAD-yank-minute` using existing local evidence:

```sh
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m pytest tests/unit/yank_bar_provenance -q
python3 -O docs/reports/yank-bar-provenance/verify-lineage.py /root/Silver-Bullet-ML-BMAD-yank-provenance/.local-provenance/verified-run1
python3 docs/reports/yank-provenance-closure/check-historical-candidate.py
```

No production API, audit command, strategy setting, timestamp interpretation, hypothetical fill or frozen P&L changed. The eight archived arm orders, five cases, 30 scenarios, 11 supported/19 unassessable outcomes, 0/100/500 ms delays and 240-opportunity pending windows retain their prior qualifications. Corrected native modeled net results remain −$818 / −$873 for no-ML / ML0.50; those diagnostic results do not revise the frozen orders. Execution-gap investigation remains a separate next task.

No acquisition, vendor contact, network access, sealed holdout inspection or non-2025 price analysis occurred. Opaque input hashing includes complete raw/native bytes; price decoding in the new compatibility check is limited to the verified 2025 extract. The full native audit was not rerun. Provenance closure remains blocked on the concrete records listed above; the compatible writer discovery alone does not authorize a validation upgrade.
