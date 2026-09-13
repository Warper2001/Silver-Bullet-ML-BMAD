# Portfolio PF improvement shortlist workflow

This package executes the approved read-only sequence: reconcile existing MIM/GAP execution evidence, stop if a current repairable economic defect is proven, otherwise build the fixed MIM scheduled-mark feasibility ledger and the three-venue commodity-carry stop/go packet.

Run from the repository root:

```bash
.venv/bin/python -m research.pf_improvement audit
.venv/bin/python -m research.pf_improvement run
.venv/bin/python -m research.pf_improvement verify --run research/pf_improvement/runs/<run>
```

`audit` and `run` also expose `--mim-data`, `--gap-data`, `--diagnostic-run`,
and `--carry-data`. Their defaults are the reviewed source snapshot; a different
path fails closed until its inventory receives a separate review and binding.

Each invocation creates a fresh direct child under `runs/`, binds the approved input and implementation hashes, records runtime/defaults, writes normalized outputs without raw account identities, independently verifies the result, and seals every file read-only. Failed invocations retain a sealed `failure.json`; existing output paths, symlinks, path escapes, sealed-holdout paths and changed inputs are refused.

The workflow never imports a live trader or strategy engine. It calculates no candidate exit or carry return, selects no parameter, sends no question, and does not invoke the portfolio decay monitor.
