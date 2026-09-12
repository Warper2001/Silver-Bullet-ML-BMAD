# MIM-NB payoff diagnostics

Describe the unchanged arm A / delay 2 recorded baseline, separate entry timing from P&L accumulation, and inventory data feasibility. No broker/collector entrypoint, strategy engine, acquisition, holdout access or parameter test runs here. Inputs resolve to the original checkout; all invocation writes stay in fresh direct `runs/` children in this isolated worktree.

From this worktree:

```sh
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m research.mim_diagnostics audit
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m research.mim_diagnostics run
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m research.mim_diagnostics verify --run research/mim_diagnostics/runs/<run-id>
```

`audit` and `run` accept `--source-run`, `--data`, and an optional fresh `--output` direct runs child. Alternate paths still require the approved completion and data hashes. Unsafe inputs, symlinks to holdout, corrupt inventories and overwrites fail closed. Failures after directory creation remain sealed with `failure.json`; they are not accepted by `verify`. Original sources are read-only.

For runs or checks expected to exceed 30 seconds, create a fresh direct runs child for logs and use `nohup`, for example:

```sh
mkdir research/mim_diagnostics/runs/my-checks
nohup /root/Silver-Bullet-ML-BMAD/.venv/bin/python -m research.mim_diagnostics run > research/mim_diagnostics/runs/my-checks/run.log 2>&1 < /dev/null &
job_pid=$!
wait "$job_pid"
```

Validation:

```sh
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m pytest tests/unit/mim_diagnostics tests/unit/mim_comparison tests/unit/mim_lifecycle tests/unit/mim_robustness -q
```

Each run freezes source snapshots, compact evidence, input hashes, runtime versions and [definitions](definitions.md). The 141 MB bar input is hash-bound rather than duplicated. Mutable prospective journals are queried read-only for counts and coverage, then preserved as observation metadata; later normal collection does not invalidate that snapshot.

Open `report.html` directly in a browser for offline charts and tables. `report.md` carries the same narrative, feasibility corrections and future research specifications. `events.csv` retains source execution order and both event/fill timestamps. `trades.csv` contains reconciled closed trades and sampled/range excursions; `minute.csv` includes all 515,970 eligible minute marks, position, event fees and exposure bounds. `accrual.csv` links trade/minute contributions; `partitions.csv` contains complete attribution partitions; `daily.csv` preserves flat sessions; `scenarios.csv` only displays alternatives already recorded by the source. `exclusions.csv` is copied byte-for-byte. JSON files include the full machine-readable summary, inventory and provenance.

Verification independently checks mandatory inventory, source/data hashes and snapshots, reconstructs trade payoff from fills and minute accrual from exact source closes, checks event fees and positions, reconciles all complete partitions, and enforces pinned session/trade/exit counts and totals. Analytical CSV/summary outputs are deterministic for identical frozen inputs; observation timestamps and manifest identifiers intentionally vary.
