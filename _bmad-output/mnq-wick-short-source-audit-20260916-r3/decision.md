# MNQ wick-short source audit

**Decision:** SOURCE/LEDGER RECONCILIATION COMPLETE; pre-roll MNQM25 is flagged as contaminated for diagnosis only. This does not adopt an exclusion, change eligibility, rerun calibration, or make a strategy/edge verdict.

The registered external input and both immutable Phase A publication hash chains matched before the source was decoded. Every r2 outcome's ten RTH components mapped to raw rows from its recorded session and exact contract.

Raw session integrity: 576 observed RTH sessions; 515 complete single-contract sessions; 61 incomplete; 40 mixed-contract.

## March 2025 lead-month check

CME's Equity Index roll calendar identifies March 17, 2025 as the lead-month transition. [CME Equity Index roll dates](https://www.cmegroup.com/trading/equity-index/rolldates.html)

Complete single-contract MNQM25 RTH sessions before that date: 8; recorded outcomes: 18; gross recorded dollars: $-549.00.

Affected session IDs:
- 2025-03-04
- 2025-03-06
- 2025-03-07
- 2025-03-10
- 2025-03-11
- 2025-03-12
- 2025-03-13
- 2025-03-14

Mixed or incomplete days were reported separately by the ledger and were never used as substitute bars. Contract labels prove source labeling only; they do not prove execution, front-month truth, or a future edge.

See `report.json` for immutable artifact hashes and the full deterministic evidence.
