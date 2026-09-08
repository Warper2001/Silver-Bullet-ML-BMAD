# YANK purchased-data execution pilot

**PASS_AUDIT_CHECKS — HOLD_VALIDATION.** Both corrected final runs parsed 237,146,989 native MBO records and produced byte-identical canonical artifacts. All 48 audit tests and the existing 133 replay/accounting/policy tests passed. The frozen acquisition manifests, original bars, and both archived order/event arms remain unchanged.

Read the [findings](final-run1/report.md), [case results and provenance](final-run1/report.json), [source event extracts](final-run1/event-extracts.jsonl), [minute reconciliation](final-run1/reconciliation.jsonl), and [verification hashes](verification.json). The saved independent repeat is in `/root/Silver-Bullet-ML-BMAD/docs/reports/yank-execution-pilot/final-run2/`; every canonical file matches the first run. These are historical results from the original audit commit. See the [integrated completion report](../yank-pilot-completion/README.md) for fresh verification on the minute-pilot branch.

Eight arm orders reduce to five execution cases, each evaluated under two bar-label interpretations and three fixed delays. Eleven scenarios have conservative supporting trade-through evidence. Nineteen are unassessable: one arrival falls inside an incomplete event, and the May 28 pending lifetimes include non-trading intervals and locked/crossed book evidence. Supporting evidence is conditional on no market impact and does not establish hypothetical fill quantity or queue position. No partial fills or revised strategy returns are inferred.

For no-ML order 4 on May 28, under end labels the archived 20:22 fill/exit bar maps to 20:21–20:22 UTC. Native records show:

| Evidence | Capture time UTC | Price | Zero-based native record |
|---|---|---:|---:|
| First touch | 20:21:27.942583595 | 21408.00 | 19638095 |
| First strictly higher trade | 20:21:27.942601066 | 21408.25 | 19638105 |
| First stop crossing | 20:21:40.073985304 | 21440.75 | 19652120 |

All three references are in `native/GLBX-20260907-NWS3PA9QPX/glbx-mdp3-20250528.mbo.dbn.zst` under the pinned acquisition. Under start labels, the corresponding 20:22–20:23 interval begins with a trade already beyond both entry and stop; same-event ordering remains ambiguous. Neither interpretation proves an actual fill. The status feed confirms that the earlier interruption ended at 20:20:05.063483407 UTC.

The independent capture-time OHLC check matches the report: end labels cover 12,697 original minutes, with 10,631 exact OHLC matches and 2,066 differences. Start labels cover 12,688 of 12,698 minutes, with no exact OHLC matches. Empirical alignment does not independently establish the original timestamp convention; both remain in the results.

Historical P&L remains reference evidence, as archived:

| Case | Arm attribution | Historical P&L |
|---|---|---:|
| 1 | no-ML and ML0.50, order 1 | -$459 |
| 2 | no-ML and ML0.50, order 2 | -$204 |
| 3 | no-ML and ML0.50, order 3 | -$489 |
| 4 | no-ML, order 4 | -$329 |
| 5 | ML0.50, order 4 | -$384 |

The historical $4 cost on each exit is not a verified broker charge. No aggregate or revised strategy return is calculated.

Run the audit from the isolated pilot worktree using a fresh output directory:

```sh
cd /root/Silver-Bullet-ML-BMAD-yank-minute
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m src.cli.check_yank_execution_pilot --output-dir /tmp/yank-pilot-check
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m pytest tests/unit/yank_execution_pilot -q
```

Compare every canonical artifact from two fresh runs with `cmp`, including `artifacts.json`, `report.json`, `report.md`, `event-extracts.jsonl`, and `reconciliation.jsonl`. The artifact manifest hashes the other four files. Each report also records pinned input hashes, code hashes, and decoder versions. Runtime processing uses only local files.

Existing regression verification was run from `/root/Silver-Bullet-ML-BMAD-yank-replay`:

```sh
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m pytest tests/unit/yank_signals tests/unit/yank_replay tests/unit/test_strategy_core_consistency.py tests/unit/test_strategy_core_scaling.py -q
```

Three independent review lenses identified gaps in timing diagnostics, coverage, reconciliation qualifications, and failure cleanup. All were corrected and tested. Inspection of the purchased-data results additionally caught the SDK's boolean status representation; actual `databento_dbn.StatusMsg` regression tests now cover it. Superseded trial outputs were removed, and both accepted runs use the corrected code. No implementation findings were deferred.
