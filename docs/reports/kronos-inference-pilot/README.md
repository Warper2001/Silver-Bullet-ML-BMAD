# Pretrained Kronos-small: local inference feasible

On 2026-09-22 the existing 24,741,376-parameter model, plus its pretrained tokenizer, generated three four-bar MNQ forecast paths on one CPU thread. No training or fine-tuning was performed.

The [verified offline run](run-20260922-verified/report.json), using runner commit `d2b8315`, measured **0.383 seconds median inference latency** and **563.7 MiB process peak RSS**. All 12 forecast candles were finite and geometrically valid. Repeating seed zero reproduced the warmup exactly. These measurements describe this small workload, not sustained-load performance or execution latency.

The [input context](run-20260922-verified/context.csv), [raw forecasts](run-20260922-verified/forecasts.csv), and [artifact hashes](run-20260922-verified/COMPLETE.json) are retained. The initial [online run](run-20260922/report.json), from runner commit `c8b9a29`, is also retained rather than overwritten. Checkpoint hashes captured on that initial pinned download were frozen before the offline repeat and checked before loading and after prediction; they detect subsequent cache changes, not independent publisher authentication.

## Interpretation

This passes a local inference smoke test, **not a strategy test**. No future actual prices, forecast accuracy, trades, PnL, costs or statistical edge were evaluated. Historical pretraining overlap is unknown. The amount input is a synthetic proxy and minute end-label semantics remain an assumption. Forecasts are continuous model outputs, not tick-rounded executable orders.

Kronos is now a research-queue candidate for the same admission process as other traders: establish data semantics and causal availability; define the forecast-to-position policy; run a power gate and commit preregistration; evaluate on genuinely unseen/prospective data against simple baselines with realistic costs; then validate paper execution and portfolio contribution. It has not been added to a live service or approved for capital allocation.

## Reproduce

Research dependencies installed only in `.venv-research`:

```sh
.venv-research/bin/python -m pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu
.venv-research/bin/python -m pip install huggingface_hub==0.33.1 einops==0.8.1 safetensors==0.6.2 tqdm==4.67.1
.venv-research/bin/python tools/kronos_inference_pilot.py --offline --output-dir docs/reports/kronos-inference-pilot/new-run
```

Omit `--offline` for the initial public checkpoint/source download. The destination must not exist. Exact numpy/pandas and other core runtime versions are in the report. Run from this checkout or its designated worktrees so the fixed audited input path resolves correctly. Use a fresh process and a distinct report directory for each invocation. Verified source files are staged and atomically published without replacing a concurrent winner; separate processes can share the source cache. Handled interruptions remove only that process's staging file; abrupt termination may leave an ignored `.part` file, never partial importable source. Existing corrupt final entries are preserved and refused, requiring operator investigation. Source-cache symlink aliases are refused. This publication mechanism targets the existing Linux filesystem with hard-link support.

Verification was extended with 17 mock-backend orchestration and cache concurrency/interruption tests, covering offline propagation, checkpoint pins and tampering, frozen evaluation mode, sampling, seed repetition, manifests and invalid-output retention. Nonfinite or geometrically invalid forecasts now consistently report `INFERENCE_INVALID_OUTPUT` and exit nonzero while retaining diagnostic evidence. Warmup predictions are now validated as well, retained in `warmup.csv`, and bound by the completion manifest; a bad warmup cannot be hidden by valid timed predictions. Prior measured reports above remain unchanged; new synthetic tests establish software behavior, not new market evidence.
