# MIM profit-giveback experiment

This package implements a research-only sequence for the scheduled-mark giveback hypothesis. It imports no broker, collector, service, or live-trader module.

```bash
.venv/bin/python -m research.mim_giveback power --source-run <completed-pf-run> --data <contract-bars>
.venv/bin/python -m research.mim_giveback inventory --power-run <powered-run>
.venv/bin/python -m research.mim_giveback sweep --inventory-run <inventory-run> --source-run <completed-pf-run> --data <contract-bars>
.venv/bin/python -m research.mim_giveback freeze --sweep-run <development-pass-run>
.venv/bin/python -m research.mim_giveback collect --protocol <committed-freeze-run> --data <future-receipt-tagged-bars> [--prior-run <collection-run>]
.venv/bin/python -m research.mim_giveback evaluate --run <collection-run>
.venv/bin/python -m research.mim_giveback verify --run <any-run>
```

`power` is the only initially permitted stage. A non-`POWERED` verdict is terminal and later commands refuse to proceed. Run manifests use checkout-portable bindings and recheck inputs at seal time. Collection accepts only strict, timely appends after the committed freeze and exposes coverage only. Evaluation uses the frozen source, rules, and historical-data hash; it exposes efficacy after exactly 500 eligible sessions, while insufficient coverage at 30 months is `INCONCLUSIVE`.
