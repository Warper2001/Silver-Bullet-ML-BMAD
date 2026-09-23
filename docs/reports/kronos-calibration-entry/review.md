# Review disposition

Review method: same-agent adversarial self-check against the user's four requested risks, source inspection, independent arithmetic formulas and executable synthetic counterexamples. No independent reviewer or external statistical approval is claimed. Findings that require new evidence are retained as blockers, not waived.

| Check / finding | Disposition |
|---|---|
| Circular admission: calibration moments needed to admit the same calibration | Blocking gap explicitly retained for routes B/C. No pilot collection or scoring performed. |
| Precision treated as policy exemption | Rejected. The external-pilot literature supplies methodology only; current policy has not been changed. |
| Favorable covariance assumed / XSMOM moments transferred | Rejected. Paired counterexamples show mismatching can overstate or understate variance. No XSMOM values used. |
| XSMOM firewall rejects whole identity but can retain fixed points | Documented as an inspection caveat relevant to a future design; source unchanged and never executed. |
| Pilot point variance treated as known / confidence confused with power | Corrected in analysis: UCL inflation, joint nuisance coverage and .76/.72 assurance examples are separate from conditional .80 power. |
| Heavy tails or serial dependence hidden behind iid normal precision | Explicit rare-tail and AR(1) counterexamples; chi-square calculations are restricted to assumed iid normal examples. |
| Effects chosen to fit 3/6/12 months | No numeric economic effect adopted. The synthetic $20 distance is identified as an arithmetic fixture. Personal economics remain symbolic. |
| Useful effect conflated with zero-profit null | Separate null, useful effect, planning alternative and future acceptance-rule choice; power distance above a useful boundary must be positive. |
| Public tariff treated as account evidence / costs double counted | Public illustration only; account plan unknown. Broker and exchange clearing distinguished, slippage counted once, dated NFA change retained. |
| Twenty calibration sessions treated as automatically sufficient | Rejected. Prior reservation is preserved solely as conditional capacity; precision objective, gate and freeze delay remain unresolved. |
| Incomplete sessions removed / exposed sessions recycled | Rejected. Unknown outcomes and exposure block unqualified inference; quarantine and revised population accounting required. |
| Scope and permission drift | Both flags false in decision, calculations and manifest; no collection, credential, market-data, local forecast or outcome access, purchases, installs or service commands. |

Verification commands and expected scope are in README.md. `verify.py` validates analytical identities and invalid-input refusals, reproduces calculation-results.json, compares all nine capacity rows against the captured prior ledger, and checks source and artifact hashes. This verifies mechanics and documentation integrity only. It cannot establish the real-data assumptions or independently attest to the process access log.

The access attestation is based on the commands executed in this investigation: explicit documentary reads, official public-document retrieval, fixed synthetic calculations and Git operations. A clean scoped diff and unchanged prior artifact hashes provide supporting integrity checks, not an operating-system audit of every process on the host.

Commit and merge verification is performed after packet finalization: only this directory may differ from the starting commit; existing source snapshots must still match their original bytes; the prior published design manifest's artifact hashes must still verify in the main checkout; then this packet's verifier runs in main. Git history is the post-publication record, so no mutable verification result is inserted into an already hashed packet.
