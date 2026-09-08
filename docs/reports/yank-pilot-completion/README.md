# YANK pilot completion

**HOLD_VALIDATION.** This branch consolidates the frozen-order audit, bar-provenance investigation, corrected native minute replay and May 28 event/book follow-ups. Completion means reproducible local execution evidence; original provenance and hypothetical execution remain unresolved.

The original frozen-order audit was already completed in commit `84f2382ed238ff66f42c470bf8021c0fd7e4e4e2`. Its [five historical artifacts](../yank-execution-pilot/final-run1/artifacts.json) are preserved byte-for-byte and match the saved independent repeat. The earlier statement that this audit was unfinished was incorrect. This integration restores its module, command, tests and documentation into `feat/yank-native-minute-pilot`, adapting only local input paths, output-tree protection and private command loading. No frozen strategy setting changes.

| Evidence | Scope and finding | Remaining limit |
| --- | --- | --- |
| [Frozen-order audit](../yank-execution-pilot/README.md) / [fresh integrated findings](run-a/report.md) | Eight archived arm orders, five cases, 30 scenarios: both label interpretations × 0/100/500 ms delays; full 240 subsequent original-bar opportunities. 11 supported, 19 unassessable. | Strict trade-through is conditional on no impact; queue positions and actual fills are unobserved. Full-window status, coverage and book gaps remain. |
| [Bar provenance](../yank-bar-provenance/report.md) | Construction lineage reproduced; end-label capture alignment has 10,631 exact OHLC matches and 2,066 differences. Suitability remains BLOCKED. | Raw origin, interval/availability semantics, authenticated aliases and roll policy are unresolved. Empirical agreement cannot select a timestamp interpretation. |
| [Corrected native minute replay](../yank-native-minute/findings.md) | 13,440 actual capture-minute bars from 7,526,752 native T prints; 17,280 coverage rows. Complete-event availability and prefix timing retained. Both frozen arms reconcile. | Separate diagnostic dataset with limited warm-up and modeled OHLC execution; it does not revise archived orders or P&L. |
| [May 28 event timeline](../yank-native-minute/may28-timeline.md) | For three corrected-replay cases, entry-through prints precede stop crossings; the no-ML 20:21 minute has distinct events about 12.13 seconds apart. | Specific crossing evidence cannot establish hypothetical fills or remove gaps elsewhere in a pending window. |
| [May 28 displayed books](../yank-native-minute/may28-book.md) | Nine arrival observations (three cases × three delays) show passive short limits; three pre-through observations add same-side displayed-size context. | Twelve complete observed books do not certify full-day or full-pending-window assessability, queue priority or execution. |

The frozen audit's May 28 no-ML order 4 retains **both conditional interpretations**. End labels map its archived 20:22 fill/exit bar to 20:21–20:22 UTC, with entry evidence before stop evidence. Start labels map it to 20:22–20:23, where the first relevant print already crosses entry and stop in the same event. The narrower follow-ups do not replace either interpretation or shorten the 240-opportunity pending lifetime. Later evidence cannot authenticate the original bar labels.

Corrected minute-replay economics remain exactly as previously published: two closed modeled trades per arm, no terminal contracts, no-ML net −$818 / equity $49,182 and ML0.50 net −$873 / equity $49,127. Each arm carries $8 in modeled closed-trade charges. The historical $4-per-close assumption is not a verified broker fee. The same-bar ambiguity annotation remains; no hypothetical partial fills, changed fills or revised frozen-order P&L are introduced.

## Verification

[Verification evidence](verification.json) records the two fresh audit runs, canonical SHA256 values, unchanged original artifacts and saved repeat, all input and archive hashes, code hashes, complete finding/schedule equality and unchanged native replay artifacts and sources. Only `code_hashes` differs from the original JSON report, reflecting the command/path integration; the event extracts, reconciliation and Markdown findings retain original bytes. Both fresh runs parse 237,146,989 native MBO records and produce identical canonical bytes.

The [test results](audit-minute-tests.xml) contain 48 restored audit tests, one fresh-interpreter import-isolation regression and 83 native minute/book tests (132 total). The [frozen regression results](frozen-tests.xml) contain 133 passing replay/accounting/policy tests, freshly run in the pinned replay checkout. Total: 265 passing tests; only existing Pydantic deprecation warnings. An optimized-Python published-ledger verification independently reconfirms both corrected replay arms. The integration review found no actionable issues in the authorized fixed-worktree scope; nothing deferred.

Reproduce using only existing local data and the existing environment. Use new output paths on each run:

```sh
cd /root/Silver-Bullet-ML-BMAD-yank-minute
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m src.cli.check_yank_execution_pilot --output-dir /tmp/yank-completion-reproduce-a
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m src.cli.check_yank_execution_pilot --output-dir /tmp/yank-completion-reproduce-b
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -O docs/reports/yank-pilot-completion/verify-integration.py /tmp/yank-completion-reproduce-a /tmp/yank-completion-reproduce-b
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m pytest -q tests/unit/yank_execution_pilot tests/unit/yank_native_minute
```

The verifier requires the saved original repeat, both retained corrected native builds and pinned sibling checkouts at the paths used by the original local research. The [audit command](../../yank-execution-pilot.md) exposes only `--output-dir`; it loads privately and does not execute the eager research initializer. Canonical audit files omit run paths and wall-clock times. The separate verification evidence records local run locations for reproduction.

No acquisition, network call, parameter tuning, model change, push or merge is part of this completion. `PASS_AUDIT_CHECKS`, `PASS_PROVENANCE_CHECKS`, `PASS_DATA_CHECKS`, `REPLAY_COMPLETE` and `PASS_INTEGRATION_CHECKS` describe checks within their stated scopes; **HOLD_VALIDATION** remains the research conclusion.
