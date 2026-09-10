# Reproducing the local verification

Run from `/root/Silver-Bullet-ML-BMAD-yank-minute` using `/root/Silver-Bullet-ML-BMAD/.venv/bin/python`. The acquisition, frozen replay sibling and model must exist at their pinned local paths; no network or API key is used.

```sh
cd /root/Silver-Bullet-ML-BMAD-yank-minute
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m pytest tests/unit/yank_native_minute -q
/root/Silver-Bullet-ML-BMAD/.venv/bin/python src/cli/check_yank_native_minute.py --output-dir data/yank/native-minute-reproduce-a
/root/Silver-Bullet-ML-BMAD/.venv/bin/python src/cli/check_yank_native_minute.py --output-dir data/yank/native-minute-reproduce-b
diff -qr data/yank/native-minute-reproduce-a data/yank/native-minute-reproduce-b
/root/Silver-Bullet-ML-BMAD/.venv/bin/python docs/reports/yank-native-minute/compute-raw-trade-digests.py
/root/Silver-Bullet-ML-BMAD/.venv/bin/python docs/reports/yank-native-minute/probe-native-events.py
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -O docs/reports/yank-native-minute/verify-native-oracle.py data/yank/native-minute-reproduce-a/bars.jsonl data/yank/native-minute-reproduce-a/exchange-diagnostic.jsonl data/yank/native-minute-reproduce-a/coverage.jsonl
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -O docs/reports/yank-native-minute/verify-published-replay.py data/yank/native-minute-reproduce-a/replay.json
```

Use fresh output names on each repetition. The raw-byte collector writes `/tmp/yank-minute-raw-digests.json`; the event probe writes `/tmp/yank-minute-event-probe.json`. The native checker also requires the hash-pinned prior independent measurement at the path declared in its source. These are fixed local verification utilities. The published-ledger checker reads sibling `bars.jsonl` to derive interval identities, prefix availability and required mark coverage independently of the replay annotations. Optimized Python negative tests ensure malformed artifacts fail without relying on assertions.

The retained legacy result is 133 passing tests in `/root/Silver-Bullet-ML-BMAD-yank-replay` at revision `b6626622e3468c932722409530120d60f29c6ce6`:

```sh
cd /root/Silver-Bullet-ML-BMAD-yank-replay
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m pytest tests/unit/yank_signals tests/unit/yank_replay tests/unit/test_strategy_core_consistency.py tests/unit/test_strategy_core_scaling.py -q
```

All 65 new tests pass. Three independent review layers completed; accepted findings were fixed and retested, with no deferred findings. The final reviewed pair is `native-minute-reviewed-a` and `native-minute-reviewed-b`. Full canonical hashes, unchanged input hashes, native comparison results, published-ledger checks and checker source hashes are retained in [parent-verification.json](parent-verification.json). Large generated outputs remain ignored locally. Passing these checks does not remove `HOLD_VALIDATION`.
