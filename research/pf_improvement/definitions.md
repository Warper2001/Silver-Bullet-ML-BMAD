# Portfolio PF shortlist definitions

- Baseline: MIM robustness run `20260912T151842-run-f8608e71fb`, arm A,
  delay 2, one MNQ contract, $2 per point and $2.24 completed round trip.
- Scheduled mark: an existing 10:00--15:30 ET half-hour decision row for which
  the source records a non-zero position. The decision's completed-minute close
  is the current price. A reversal row remains attached to the exiting trade.
- Causal marked net: direction times price change times $2, less the incurred
  $1.12 entry cost. The unincurred exit cost is not charged at an open mark.
- MFE/MAE: running favorable/adverse gross excursion from completed bars known by
  the scheduled mark. The ambiguous catastrophe-stop minute and all post-exit
  prices are excluded.
- Execution gate: `CURRENT_REPAIR_CANDIDATE` requires a recurring current adverse
  economic difference with exact causality, both fills, complete costs and an
  unchanged intended behavior. `NO_REPAIRABLE_MECHANISM` requires complete current
  coverage without such a difference. Any unresolved causal, timing, quote or
  cost evidence yields `INSUFFICIENT_CAUSAL_EVIDENCE`.
- Carry feasibility: metadata and account feasibility only. No carry signal,
  position, return or alternative-strategy P&L is calculated.
- `DISTINCT_HYPOTHESIS_REMAINS`: the scheduled-mark giveback description has
  complete causal coverage and does not duplicate V1, V2 or the exposed neutral
  exit. It is a hypothesis-generation verdict; it does not establish predictive
  value, select a rule or authorize a strategy test.
- Fixed path distributions: all deciles from 0% through 100% are reported for
  every declared group and metric. They are descriptive partitions, never
  candidate thresholds.
