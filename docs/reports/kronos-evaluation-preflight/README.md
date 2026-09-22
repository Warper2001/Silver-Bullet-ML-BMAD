# Kronos: next economic-evaluation gate

The inference pilot passed. This follow-up asks whether the current evidence supports testing a trading edge. It does **not** compute any strategy returns or look at raw prices.

## What the next gate established

The pinned model and tokenizer revisions both have public commit timestamps on September 9, 2025. The existing audited data end on August 28, 2026 at 11:19 New York time, a partial RTH day. Excluding that partial day and starting after the later revision date leaves **at most 252 weekdays**, September 10, 2025–August 27, 2026. This generous ceiling still includes exchange holidays and unverified gaps. It is neither 252 confirmed sessions nor 252 independent, untouched outcomes.

None of those dates is admitted as untouched for this candidate. Previous local research used the same file; moving the date boundary past the checkpoint does not undo that exposure. The actual eligible count is unknown, not established to be zero. The earlier February 2025 inference example is development-only.

Under an illustrative one-sided normal known-variance test with 5% family alpha, 80% target power, one comparison and no standard-error inflation, annual net Sharpe 1 requires about **1,559 daily observations (6.18 252-day years)**; Sharpe 2 requires **390 observations (1.55 years)**. These are assumptions, not predicted Kronos performance, adopted thresholds, or an empirical power result. Dependence, multiplicity, unknown variance and non-normal returns can change the requirements. See the generated report for the complete sensitivity grid.

**Disposition: HOLD_EVALUATION; actual strategy power UNASSESSABLE.** Do not label the model economically UNDERPOWERED at an arbitrarily selected effect, or pass it using the most optimistic planning scenario. There is no justified target effect or admitted evaluation population yet. More intraday windows, stochastic seeds or model samples do not manufacture more independent market history.

## Evidence and source audit

Documentary inputs are pinned by SHA-256 in `tools/kronos_evaluation_preflight.py`; the program reads only those two report JSON files. It never opens the CSV paths named inside them. The earlier report remains a historical snapshot: its old torch-availability observation is superseded by the successful inference pilot, not silently rewritten.

Public metadata checked September 22, 2026:

- [Kronos-small pinned commit history](https://huggingface.co/api/models/NeoQuasar/Kronos-small/commits/901c26c1332695a2a8f243eb2f37243a37bea320): `901c26c1332695a2a8f243eb2f37243a37bea320`, `2025-09-09T14:10:26Z`, “Update README.md”.
- [Tokenizer pinned commit history](https://huggingface.co/api/models/NeoQuasar/Kronos-Tokenizer-base/commits/0e0117387f39004a9016484a186a908917e22426): `0e0117387f39004a9016484a186a908917e22426`, `2025-09-09T14:10:02Z`, “Update README.md”.
- These are conservative **revision dates**, not weight-training dates or independently attested first-publication dates. The model history also lists an earlier June 30 “add model” commit. We did not authenticate that earlier commit's weight identity, so do not move the boundary backward on its title alone.
- The [Kronos paper](https://arxiv.org/html/2508.02739v1#sec3) describes broad financial-market pretraining. The exact pinned checkpoint's MNQ membership and temporal exclusions have not been authenticated. No assertion of MNQ exclusion follows from the paper or model card.

Local provenance evidence: `research/mim_comparison/evidence/source-manifest.json` and `reference-audit.md` document file identity and prior exposure. `_bmad-output/preregistration_mim_x2_powered.md` attributes acquisition to TradeStation, but the exact acquisition writer/receipt for this CSV was not located. `research/mim_comparison/data.py` offers a previous-session-volume contract-selection reference; its calendar and expiry assumptions remain unadmitted. Narrative end-label conventions do not authenticate this specific file or historical decision-time availability.

## Concrete continuation path

1. Build a development-only, next-bar-fill replay for a single frozen forecast-to-position policy alongside no-position and simple non-model controls. Test its accounting and causality on synthetic data first. Existing researched history may support development and justified calibration, but cannot be renamed untouched evidence. Run the required experiment-specific power gate and preregister before any real-data strategy test, including development performance tests.
2. Establish authenticated bar labels, completion/arrival semantics, exchange calendar, causal contract selection, fees and slippage. Choose a minimum useful effect from independent economic/cost justification or allowed calibration evidence, not from the reserved test outcomes. Resolve comparison-family allocation and dependence model.
3. Register and collect genuinely prospective observations with forecasts fixed before outcomes, keeping the checkpoint and policy immutable. This needs an approved data source and a separate observation-only collector implementation; none is started by this preflight. No broker orders or live services are implied.
4. Run the empirical power gate on the registered eligible population before scoring returns. Only a qualifying subsequent economic evaluation and paper-execution validation can support trader-pool admission. Small early prospective samples can expose operational faults; they do not establish profitability.

## Reproduce

```sh
.venv-research/bin/python tools/kronos_evaluation_preflight.py --output-dir docs/reports/kronos-evaluation-preflight/new-run
```

Exit **2 is the expected blocked gate**, not a processing failure or PASS. A usable output requires `COMPLETE.json` and matching hashes for both reports. Destinations must be fresh children of this report directory. No credentials, downloads, raw market data, training, inference, economic scoring or live processes are used by this command. The generic normal-theory sensitivity helper is not an admission API; the fixed-evidence gate deliberately cannot return PASS.
