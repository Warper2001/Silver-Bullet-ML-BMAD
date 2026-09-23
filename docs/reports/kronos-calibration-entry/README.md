# Kronos calibration-entry decision — 2026-09-23

**PARK_PENDING_EVIDENCE.** No route currently combines a justified admission gate, supported indispensable inputs, and a bounded future resource plan. `strategy_test_permitted=false`; `trading_authorized=false`. This packet completes the bounded investigation; it is neither a preregistration nor permission to collect, forecast, calibrate, test, or trade.

The decisive gap is not simply a shortage of sessions. Useful dollar effects remain undefined, and no admitted evidence bounds the frozen strategy's session-level paired variance and dependence. Collecting its performance to estimate those quantities would itself encounter the existing pre-test power requirement. Calling that collection a precision pilot does not resolve the dependency.

The three routes were assessed as follows:

| Route | Finding | What would reopen it |
|---|---|---|
| Independent applicable evidence | No qualifying evidence identified in the bounded documentary search. The Kronos paper evaluates a different investment strategy. | Independently produced, exposure-audited evidence applicable to the exact candidate and momentum comparator, supporting effects, costs, joint nuisance uncertainty and transfer to future sessions. |
| Outcome-blind nuisance gate | A possible research direction, not an admitted method. XSMOM's mismatched rank-IC distribution is not Kronos's paired net-dollar distribution. | A reviewed, strategy-specific blinded estimator or conservative bound, with an independently supported dependence/transport envelope and a disclosure firewall. No aligned performance may be exposed to obtain admission. |
| Separately preregistered precision calibration | Relative-variance precision can be calculated under iid normal assumptions, but those assumptions lack support here. Existing policy provides no automatic precision exemption. | A documented policy-compatible admission argument plus supported distributional/dependence assumptions and precision objective. If it is performance calibration, it still needs a pre-test power gate; changing policy requires an explicit separate decision. |

Reconsideration also needs a credential-free TradeStation plan/fee confirmation, execution-cost evidence, allocated capital, operator time value, minimum useful absolute and incremental profit, and an operator-approved future time/cash horizon. Public rates alone cannot establish account economics. The requested two-hour investigation cap and zero purchase budget are established; they do not fund future calibration.

The [economic worksheet](economics.md) keeps personal economics symbolic. The [route analysis](routes.md) describes admission and failure conditions. [Calculations and timeline](calculations.md) retain the existing 3/6/12-month scenarios as conditional capacities, including the hypothetical 20-session calibration reservation; none is powered calibration. [Source notes](source-notes.md), [decision JSON](decision.json), and [review disposition](review.md) preserve the evidence and limitations.

All changes are confined to this new packet. Prior published artifacts, strategy parameters, APIs, CLIs and services remain unchanged. Source/code snapshots are for documentary inspection only. No market data, credentials, local forecasts or session outcomes were accessed; published research was read as literature, not imported as performance observations.

Reproduce offline from the repository root using the existing environment (no installation):

```sh
.venv/bin/python -B docs/reports/kronos-calibration-entry/calculate.py > /tmp/kronos-calibration-calculations.json
.venv/bin/python -B docs/reports/kronos-calibration-entry/verify.py
```

`calculate.py` has no input files, network access or market-data interface: its constants are explicitly synthetic or inherited planning scenarios. The verifier checks arithmetic, synthetic counterexamples, output reproduction, input identities and the packet manifest. `COMPLETE.json` hashes every packet file except itself; Git anchors the manifest. Hashes establish identity, not evidential sufficiency. This is a self-reviewed investigation, not an independent statistical approval.
