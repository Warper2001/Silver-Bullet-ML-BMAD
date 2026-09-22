# Kronos pretrained strategy candidate

Date: 2026-09-22
Status: INFERENCE_FEASIBLE / ECONOMIC_VALIDATION_PENDING

Intent: treat an existing pretrained model as another possible strategy in the trader pool, not a separate standard of evidence. No training is required for this initial candidate.

Evidence: `docs/reports/kronos-inference-pilot/README.md` and its immutable raw run artifacts. Kronos-small generated valid MNQ-shaped forecasts locally in approximately 0.38 seconds per four-bar path using one CPU thread. This is technical feasibility only.

Next gate: define the causal input and forecast-to-position policy, comparison baselines, costs and risk constraints; run the required power gate and commit preregistration before an economic strategy test. Resolve the previous data-readiness findings relevant to that test and checkpoint pretraining overlap. Prefer prospective evidence where historical contamination cannot be ruled out. Do not select thresholds or claim an edge from the smoke-test window.

Admission: evidence of robust incremental returns after costs, adequate statistical power, untouched evaluation, reliable paper execution, and portfolio compatibility, subject to the same existing controls as every trader. No service, orders, strategy parameters, training or live allocation changed.
