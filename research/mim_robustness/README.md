# Fixed MIM-NB Sharpe experiments

Isolated historical comparison of A (unchanged MIM-NB), R (regression agreement), E (path efficiency), F (fresh-breakout reset), and P (breakout persistence). Definitions, costs, risk controls and inference are fixed in the BMad protocol; this is not parameter optimization. Existing history is exposed and cannot validate deployment.

From repository root:

```bash
.venv/bin/python -m research.mim_robustness audit
.venv/bin/python -m research.mim_robustness run
.venv/bin/python -m research.mim_robustness evaluate --run research/mim_robustness/runs/COMPLETED_RUN
.venv/bin/python -m pytest tests/unit/mim_robustness -q
```

`audit` and `run` default to the exact approved historical run and contract CSV; explicit `--baseline-run` / `--data` locations must have the approved hashes. Every invocation creates a new directory under this package's ignored `runs/`. Baseline reconciliation must pass before any candidate simulation. `run` performs the entire fixed batch and evaluation; `evaluate` independently verifies and re-evaluates an immutable completed run. It refuses audit, failed, modified or source-incompatible runs. Candidate counts and parameters are not CLI options.

Source, input, configuration and protocol hashes are frozen before returns. Ledgers, reports and failure artifacts use exclusive creation and read-only sealed inventories. This detects changes, not tampering by a privileged administrator. Commands import no broker or legacy executable study scripts and never write production files or activate services. Production risk controls and the frozen prospective A/B study are unchanged.

Primary daily Sharpe uses fixed $10,000 capital, zero risk-free return, sample standard deviation and sqrt(252) annualization across identical eligible sessions with no-trade days zero. Missing dates remain excluded. Daily closing-equity drawdown does not bound intraday losses. Historical shortlist is conditional evidence for possible separate prospective validation only. See `report.md`, `report.html` and `results.json` in each completed run for all candidates, uncertainty and limitations.

Exact baseline compatibility preserves its queued-reversal behavior: an accepted reversal can fill after a catastrophe stop during modeled latency, unless the daily guard deactivates trading. A regression fixture documents this inherited convention; these experiments do not change it.
