# Kronos-small inference pilot

This pilot runs the existing pretrained model. There is no training or fine-tuning. Technical feasibility earns a place in the research queue; profitability and admission to the trader pool still require the usual preregistered, powered strategy evaluation and prospective execution evidence.

## Fixed smoke-test design

Use the already audited contract-labelled `data/mim_x/mnq_1min_by_contract.csv`, contract MNQH25, and cutoff 2025-02-03 19:30 UTC (14:30 New York). Feed 128 completed RTH 15-minute bars and request four future 15-minute candles, ending at 15:30 local time. No later price enters the model and no actual future price is compared against the forecasts. This development example is selected before forecasts are observed; it is not untouched evidence.

Minute labels are assumed to be interval ends for this smoke test. Context aggregation requires exact 15-minute grids, finite consistent OHLCV, a single explicit contract, and no duplicate timestamps. NY local timestamps supply the model's calendar features. Missing full intraday buckets inside the selected context refuse the run; exchange holiday/calendar and provider availability authentication remain unresolved for later economic testing.

The upstream predictor derives `amount` as volume times mean OHLC when absent. We explicitly reproduce that proxy; it is neither authenticated turnover nor MNQ cash notional. Forecasts retain the upstream outputs without projecting them into valid candles: nonfinite/negative outputs or OHLC inconsistencies are reported. Seed variation is illustrative, not a calibrated confidence interval.

The fixed workload is one warmup plus one path per seed 0, 1 and 2, on a single CPU thread. Sampling uses upstream defaults (temperature 1, top-p 0.9, top-k 0). These are operational demonstration settings, not hand-selected trading thresholds or a sealed economic specification.

## Reproducibility and isolation

- Upstream source: `shiyu-coder/Kronos`, commit `67b630e67f6a18c9e9be918d9b4337c960db1e9a`.
- Model: `NeoQuasar/Kronos-small`, revision `901c26c1332695a2a8f243eb2f37243a37bea320`.
- Tokenizer: `NeoQuasar/Kronos-Tokenizer-base`, revision `0e0117387f39004a9016484a186a908917e22426`.
- Downloads are public and pinned; source bytes are hash checked, model files use safetensors, and no Hugging Face token is supplied.
- Dependencies and downloaded assets live in `.venv-research`; the live `.venv` is untouched. The inference process cannot submit orders and does not import trader code.
- Record checkpoint/config/input/context hashes, library versions, explicit device/thread settings, loading time, inference timings, process peak RSS and each raw forecast. Local input lineage is a provenance record, not a claim of an unbiased historical test.

Official references: [Kronos implementation](https://github.com/shiyu-coder/Kronos/tree/67b630e67f6a18c9e9be918d9b4337c960db1e9a), [small checkpoint](https://huggingface.co/NeoQuasar/Kronos-small/tree/901c26c1332695a2a8f243eb2f37243a37bea320), [tokenizer](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base/tree/0e0117387f39004a9016484a186a908917e22426).

## Relation to the earlier readiness audit

The earlier HOLD_DATA and unassessable-power findings constrain an economic strategy test. They do not prevent this unscored inference demonstration. Its successful completion would show that the model can run locally on MNQ-shaped data; it would not establish forecasting skill, expected profit, or authorization to add a live trader.
