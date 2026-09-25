# GAP-1 N=30 Reading — sealed rule output: SCALE to 2ct

**Date:** 2026-09-25. **Trade 30 closed:** 2026-09-21 (reading and execution are 4 days late).
**Governing documents:** sealed decision rule in `preregistration_gap_fade_panic_open.md` (seal 32da5d5);
reading fixed in advance by `precommitment_gap_fade_n30_decision_20260917.md`.
**This reading executes the sealed rule as written.** It adopts no new threshold and no new rule.

## 1. Result

Ledger: `data/trades.db`, `trader_id='trader-gap-fade'`, `write_mode='realtime'` (GAP-1's
authoritative ledger). Metric: gross PF on 1ct P&L, as sealed.

| | N | Net (gross, 1ct) | PF |
|---|---|---|---|
| **At trade 30 (2026-09-21), the PF of record** | 30 | +$1,013.00 | **1.216** |
| Now (trade 31, 2026-09-24) | 31 | +$1,490.50 | 1.318 |

- **Sealed rule:** PF > 1.20 means **scale to 2ct, continue**. The 30-calendar-day condition cleared
  long ago (first trade 2026-06-25).
- **Trade 30 lost $931** and took PF from about 1.5 down to 1.216. The SCALE margin is thin (0.016).
- **Reconciliation.** `data/gap_fade/trades.csv` has 32 rows over the same window. The extra row is the
  known permanent duplicate of 2026-06-25 (+$1,390.50; memory `gap-fade-ledger-authority`).
  Otherwise the days and per-day P&L match the DB exactly.
- **Costs.** The ledger is gross, as the rule is. Each round trip costs about $1.2 per contract, so over
  about 15 winners and 15 losers the fee-adjusted PF is about 1.21, still above 1.20. This is an
  estimate from the ProjectX per-contract rate; GAP-1's TS SIM commission was not measured.

## 2. How it is read (pre-commitment §"What we will do", unchanged)

1. **The literal output stands:** position size goes to 2ct.
2. **This is logged as "did not fail the STOP bar", not as "edge proven".** Under zero edge the rule
   outputs SCALE 35.5% of the time at N=30 (GAP-1 power gate, 2026-09-13). No document may cite this
   SCALE as evidence of edge.
3. **Data collection continues toward N≈88**, the confirmatory point at full Gate-0 edge, roughly
   2027-03 at the live pace. The sealed N=60 re-evaluation also still applies.
4. **No sizing beyond 2ct** without a fresh pre-registration.

## 3. Execution

- **Change:** `CONTRACTS = 1 → 2` in `src/research/gap_fade_live.py`. This is the only strategy
  constant touched. The log text that hardcoded "1ct" is updated to print the real size.
- **Venue:** unchanged, TS SIM paper (`GAP_FADE_TS_SIM=1`). No real money.
- **Route:** worktree, then merge into `main`, diff, run the unit tests, and restart `trader-gap-fade`.
  The restart falls on Friday 2026-09-25; GAP-1 excludes Fridays, and no position was open (no
  `data/gap_fade/state.json`).
- **What the code already handles:** the modelled P&L (`pnl_pts × $2 × CONTRACTS`), TS SIM order
  quantities, and TS-realised P&L all scale with `CONTRACTS`. Each `trades.db` row records
  `metadata.contracts`.

## 4. Measurement convention after scaling (no new threshold)

The sealed thresholds are defined on 1ct P&L. Dollar P&L from 2ct trades would count double in a
pooled PF, so **every later reading (N=60, N≈88) computes PF on per-contract P&L:
`pnl / metadata.contracts`.** That keeps the sealed rule measuring what it measured at N=30. It is a
unit convention, not a threshold, but it is new and flagged here for Alex's confirmation.

## 5. What this reading does not do

It does not claim an edge, change any other parameter, move GAP-1 off TS SIM, or alter the N=60
rule. It also does not touch the ProjectX promotion gate
(`project_gap_fade_projectx_promotion`), which remains a separate decision.
