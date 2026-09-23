# Kronos baseline backtest — blocked, 2026-09-23

**BACKTEST_BLOCKED / PARK_PENDING_EVIDENCE.** Historical inference and scoring were not run. None of the three admission requirements is satisfied. Actual power and eligible session count remain unknown; zero sessions are admitted. This is a completed documentary decision packet, not a completed backtest or preregistration.

| Requirement | Specific failure | Smallest actionable remedy |
| --- | --- | --- |
| Historical population | The identified file has prior research exposure; acquisition-linked timestamps, availability, calendar and causal contract selection remain unadmitted. No untouched interval is established. | Supply an existing, credential-free provenance/exposure dossier binding exact bytes and dates to original receipts, contract decisions and session schedules. If that evidence does not exist, use a separately authorized prospective population. |
| Execution assumptions | Public tariffs do not establish the account plan. Applicable slippage, latency and operating-cost treatment are missing. | Obtain the applicable dated account fee schedule and independent execution evidence, then document and freeze all cost components and allocation before scoring. |
| Statistical admission | Neither comparison has an independently supported positive planning effect or paired long-run variance/dependence and nuisance-uncertainty evidence. | Supply applicable independent evidence or a reviewed, policy-compatible blinded bounding procedure, then run a gate for the eligible population. More sessions alone cannot resolve unspecified effects and uncertainty. |

The [evidence assessment](evidence.md) identifies exact sources and their limits. [decision.json](decision.json) records the constraints and refusal. [prospective-shadow.md](prospective-shadow.md) proposes a bounded next step without starting collection. [verification.md](verification.md) records coverage and limitations; [input-register.json](input-register.json) fingerprints documentary/code dependencies. `COMPLETE.json` fingerprints this packet; Git anchors its identity.

Your objective remains **any positive net profit plus an improvement on four-bar momentum**. Capital is $30,000, and the $5,000 loss-from-start preference is an assessment criterion only. It does not create a strategy stop, a variance bound or a numerical planning effect. Weekly/monthly reviews record progress and do not alter the experiment. Missing evidence does not demonstrate that the investment constraints are impossible, so the recommendation is **park**, with no claim of profitability, failure or economic UNDERPOWERED status.

No performance figures are available: net profit, costs, turnover, paired advantage, uncertainty, drawdown and loss below starting capital are **not measured**, rather than zero. For a future admitted run, with cumulative net PnL `P(t)` including the initial zero, equity is `30000 + P(t)`, loss below starting capital is `max(0, -min(P(t)))`, and maximum drawdown is `max_t(max_{s<=t} P(s) - P(t))`. Report trading-net and fully allocated operating-cost results separately. Compare loss-from-start with $5,000 without changing exits. Mark incomplete sessions and unresolved exposure as unknown; marks are not executed exits or complete results.

Reproduce integrity checks offline from the repository root:

```sh
.venv/bin/python -B docs/reports/kronos-backtest/verify.py
```

Successful verification means the blocked packet is intact. It grants no historical, collection or trading admission. No existing permission flag, synthetic CLI, model, strategy or service was changed.
