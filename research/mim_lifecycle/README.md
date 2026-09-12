# MIM-NB trade lifecycle

Reproducible descriptive analysis of the frozen unchanged A baseline, primary timing, one contract. No alternative strategy returns, orders or live changes.

From repository root:

```bash
.venv/bin/python -m research.mim_lifecycle
.venv/bin/python -m pytest tests/unit/mim_lifecycle -q
```

For longer runs, use `nohup` with a log and monitor completion as required by the repository instructions. Every invocation writes a new local ignored `runs/` directory. Source, numerical runtime, input hashes and frozen intent are bound before analysis. Success outputs include path and landmark ledgers, summary JSON, Markdown/HTML reports and standalone SVG charts. A failed invocation is sealed with `failure.json`.

Run verification from Python with `research.mim_lifecycle.artifacts.verify(path, complete=True)`; `verify_inputs()` separately rechecks the original evidence. Results are conditional on the modeled fills and one-minute data. The stop-minute path is unknown; its OHLC range is excluded. See [RESULTS.md](RESULTS.md) for findings and research next steps.

Sealing makes files read-only and records their hashes. Verification detects changes or replacement; writable directories and privileged access mean this is tamper detection, not tamper prevention.
