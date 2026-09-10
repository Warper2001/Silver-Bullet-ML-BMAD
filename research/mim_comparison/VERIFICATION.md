# Verification record — 2026-09-10

The final implementation passed **40 tests in 103.45 seconds** using:

```bash
.venv/bin/python -m pytest tests/unit/mim_comparison -q
```

Coverage includes independent deployed AST broker fixtures and published reference sequences; precision/rounded-log ambiguity; guard boundaries, stops, gaps, reversals and EOD; contract selection, rolls, timing and sizing lookahead; missing, duplicate, malformed, weekend and out-of-order observations; restart/correction durability; deterministic accounting; highest-cost decision vetoes; authenticated synthetic 120-session finalization/tampering; and actual bubblewrap filesystem/credential isolation plus native/x32/compat socket restrictions.

Three review lenses were triaged in the implementation specification. Actionable findings were patched and verified. Two challenged behaviors were preserved after checking original executable evidence: the deployed rejected-exit reversal quirk and the author's compounding own-equity full-notional comparator. These are disclosed, not production changes.

Final CLI runs:

- [Historical comparison](runs/20260910T210224-historical-f3950efb68/report.md): real existing contract-level data; all arms, mechanisms, timing/cost scenarios and separate sizing.
- [Data audit](runs/20260910T210333-audit-d784c4b668/report.md): same existing source and explicit end-label declaration.
- [Sandbox launch smoke test](runs/20260910T210419-shadow-91095a6bd2/report.md): header-only identified CSV, **zero observations**. This is a launcher test, not real collection.
- [Incomplete final decision report](runs/20260910T210450-evaluate-967b6ea2cb/report.md): zero eligible sessions; no efficacy verdict or deployment authorization.

All 78 inventoried artifact hashes across those four runs were independently checked after completion. Frozen research source snapshots are inside each run. Generated run directories remain local and are excluded from Git.

The production MIM source remains SHA256 `313cbb546d322d13140cd1f47d8c5adf2b44930b5ba8ad3f8eeef61489af85d5`. The existing historical input remains SHA256 `ff76aefca405dd94359b15223c57710f4e7f01f245880426a60d0f934c6f5bea`. No production code, orders, services or prior reports were changed by this implementation.

Real prospective collection is deferred: the available live CSV lacks contract provenance, and fresh timely coverage/warmup must qualify under the frozen protocol. The smoke-test journal must not be treated as a real feed or prospective evidence. No background collector was started. Future 120-session observation and subsequent execution validation remain outstanding by design.
