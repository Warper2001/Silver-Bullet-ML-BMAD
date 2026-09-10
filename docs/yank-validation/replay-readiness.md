# Deployed replay readiness

Decision: HOLD_VALIDATION. The offline adapter executes class/function bodies from the exact versioned installed snapshot. It is not the frozen research replay. Safe private imports substitute auth, HTTP, model/config paths, persistence, trade database and side-effect loggers before construction. The constructor runs; initialize/start/stop are denied. Original polling, parsing, session counters, filters, modeled fills and recovery functions remain authoritative. Model bytes, source dependencies, runtime packages, input streams, replay code and manifest are pinned and checked again after execution.

CLI (existing commands are unchanged):

```
/root/Silver-Bullet-ML-BMAD/.venv/bin/python src/cli/check_yank_deployed_replay.py \
  --manifest docs/yank-validation/native-admission.json \
  --input-dir /root/Silver-Bullet-ML-BMAD-yank-minute/data/yank/native-minute-reviewed-a \
  --snapshot docs/yank-validation/snapshot/v1 \
  --output-dir /tmp/NEW-yank-deployed-run
```

Admission requires explicit source/request identities, file hashes, contract mapping, adjustment policy, interval, availability, session, coverage, warm-up and initial risk/execution state. Native coverage counts and continuous minute intervals are verified against pinned bars and coverage streams; empty coverage rows are never converted into bars. No annual file, holdout, auth or remote requests are read. Native completed-event availability is retained as integer nanoseconds, with the Python datetime clock rounded upward to microseconds. Raw response ordering, revisions and all native references remain in trace inputs.

Engineering fidelity and data readiness differ. Per-decision traces retain accepted-buffer identity, sweep/M15 latches, expiry clocks, session ranges, risk, active state, feature vectors/model probabilities, order intentions and discrete transitions. Consumed-history evidence measures LR 1,950 rows, minimum 20/full 120 positive H1 ATR observations, partial completed H1 buckets, and up to 20 preceding calendar-day ranges. Missing preinterval warm-up blocks full feature readiness. Initial native replay uses one bar as synthetic backfill; the deployed startup requests 48 hours. Native finalized minutes do not authenticate TradeStation labels, partial-bar responses, receipt latency or revision behavior.

All native 2025 use is development-only. Explicit data contract MNQM5 uses installed MNQU26 instrument specifications for this diagnostic, not a claim of same-expiration feed equivalence. Scheduler omissions are preserved fixed-UTC behavior; data coverage and actually consumed signal rows must be assessed separately. No profitability, broker execution, account funding or risk-headroom readiness follows from matching traces.

Synthetic initial risk state is named SYNTHETIC_UNKNOWN_ACCOUNT; its zero P&L is a test input, not observed equity. OBSERVED_STATE requires a complete checkpoint plus redacted account-evidence identity/time. Such declarations remain UNVERIFIED; a supplied hash and timestamp do not authenticate an account observation. Live equivalence remains unassessable until independent attestation is integrated. Checkpoints contain every normalized strategy state field and the full accepted bar buffer, bind to the snapshot hash, rebuild snapshot dataclasses, and validate derived state/buffer identity. Missing fields are rejected; no flat-account fallback is used. Recovery uses the original snapshotted method and an equivalent privately injected TradeState type. Unknown broker answers remain unknown even where the original recovery code makes a protective active-position assumption.

Execution stubs record intentions only. Explicit observed method-return sequences can inject actual returned order IDs without a remote call; unused or reordered replies fail closed. Acknowledgements, fills, and rejection observations are separate venue evidence. No P&L from this diagnostic replaces frozen evidence, and no terminal liquidation is added.

The private runtime is an isolation boundary for this known, hash-pinned source, not a sandbox for arbitrary Python. Private import allowlists, memory-only persistence/logging, denied auth/HTTP constructors and snapshot-restricted paths prevent the source's real capabilities from being constructed. Tests cover pin refusal, attempted infrastructure use, original descriptor invocation at equal state, nonzero entry/fill/exit intentions, pending expiry, missing/revised/out-of-order bars, stale guards, DST, contract labels and recovery.

## Measured development diagnostic

Two complete runs produced byte-identical `trace.jsonl` and `report.json` outputs. Each read all 13,440 admitted native bars and emitted 13,440 poll records with zero processing errors. The preserved fixed-UTC scheduler admitted 12,909 polls and skipped 531; raw skipped inputs remain in the trace. The final buffer has 7,500 rows, 129 completed H1 buckets (six partial), 120 positive ATR observations and 11 preceding calendar-day ranges. LR and minimum/full volatility history eventually become ready, but preinterval warm-up and full 20-day ADR readiness remain blocked.

The diagnostic generated five modeled order intentions and one completed simulated trade. These are engineering observations, not broker acknowledgements/fills or economic validation. No terminal liquidation was added. The [native report](../reports/yank-deployed-validation/native-report.json) retains independent readiness assessments and exact code/input hashes. The full trace is retained losslessly in `docs/reports/yank-deployed-validation/native-trace.jsonl.gz`; its uncompressed SHA256 is `9a0ddb3df90fb55e14e417a239025b8e56f3574a81c9a3d5a81f4280ba16f29d`. Final output directories were `/tmp/yank-deployed-reviewed-a` and `/tmp/yank-deployed-reviewed-b`; earlier interrupted directories are provisional and excluded.
