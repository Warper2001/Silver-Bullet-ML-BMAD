# GAP-1 Gate-0, re-scored on corrected bars (2026-09-16)

**Pre-registration:** `_bmad-output/preregistration_gap_fade_gate0_rescore.md`, sealing commit `151f1d05499da6f951d0743399c74f9caf76ee32`, committed before the run.
**Holdout access:** logged in `data/sealed_holdout/ACCESS_LOG.md` citing that commit, appended **before** any holdout-period byte was read.
**Re-scores:** the sealed Gate-0 trade list `data/reports/gap_fade_20260625_205328.csv` (N=117, WR 62.4%, PF 1.761, Net $9,878; 2025-01-06 → 2026-06-11).

## Result

| Gate-0 window | N | WR | PF | Net |
|---|---|---|---|---|
| **Sealed (2026-06-25)** | 117 | 62.4% | **1.761** | **$9,878** |
| **Corrected** | 115 | 61.7% | **1.646** | **$8,281** |
| Difference | −2 | −0.7 pt | **−0.115** | **−$1,597 (−16.2%)** |

**The edge survives the correction.** For context — the original Gate-0 rule is not re-applied here, per prereg §3 — the corrected figure still clears the numbers that rule used: PF 1.646 ≥ 1.40 (strong), N=115 ≥ 60, WR 61.7% ≥ 55%, max 3 consecutive losses ≤ 10, worst month −$1,280 (2026-05, unchanged from the sealed run, and the one criterion the sealed run also missed against its −$600 floor).

## Where the $1,597 went

| Segment | Sealed | Corrected | Change |
|---|---|---|---|
| 2025 | 77 / $7,861.00 | 76 / $7,120.50 | −$740.50 |
| 2026 Jan–Feb (was back-month MNQM26) | 8 / $149.00 | 12 / $932.00 | +$783.00 |
| **2026 Mar 1–11 (pre-roll)** | **4 / $1,888.00** | **0 / $0** | **−$1,888.00** |
| 2026 Mar 12 – May 19 (holdout, front month) | 20 / −$1,413.50 | 19 / −$1,165.00 | +$248.50 |
| 2026 May 20 – Jun 11 | 8 / $1,393.50 | 8 / $1,393.50 | unchanged |

**The single biggest item is the pre-roll window.** In the sealed run, 2026-03-01 → 03-11 contributed **+$1,888 from 4 trades** — a fifth of the whole Gate-0 net — priced off the deferred MNQM26 contract while MNQH26 was still the front month. On corrected bars that window produces **no trades at all**: six of its sessions (Mar 1–6) interleave two contracts and are dropped by the sealed rule, and the remaining sessions show no qualifying gap.

**The 2026 Jan–Feb segment improves** (+$783) once it is priced on the real front month instead of the thin deferred contract: 12 trades rather than 8, and seven of the eight surviving sealed trades shift slightly.

**The 2025 segment matches the independent earlier diagnostic exactly** (N=76, $7,120.50) — a cross-check that this harness and `same_contract_prior_close.py` agree.

**The post-holdout tail is byte-identical** (8 trades, +$1,393.50) as expected: those rows are front-month in both versions and untouched by the fix.

## Sessions skipped for want of a same-contract prior close

Per the prereg, a session whose own contract has no prior RTH session in the data is skipped — what live does when a freshly switched symbol has a thin or absent prior session:

| Date | Contract | Note |
|---|---|---|
| 2025-03-03 | MNQM25 | the roll-boundary trade worth +$740.50 in the sealed list |
| 2025-06-02 | MNQU25 | produced no trade in the sealed list either |
| 2026-03-12 | MNQM26 | the prior session (Mar 11) is pure MNQH26, so M26 has no prior close |

## Method and integrity

- **2025:** front-month rebuild by the original dollar-bar writer on the pinned raw extract. Its gate — the unfiltered rebuild reproducing the frozen CSV byte-for-byte (`3f20ec70…`) — passed again on this run, and the rebuild is deterministic (`f1fe5b36…`).
- **2026-01-01 → 03-11:** front-month MNQH26 minutes from the raw JSON, 56,100 of them; the six interleaved sessions (2026-03-01 … 03-06) dropped whole.
- **2026-03-12 → 06-11:** the 2026 CSV's own rows, 89,928 of them, unchanged — MNQM26 is the front month after the roll.
- **Prior close** always from the session's own contract; **no price adjustment**, which the prereg pre-excluded because point adjustment breaks GAP-1's 0.5%-of-prior-close trigger and ratio adjustment rescales its point P&L.
- Strategy parameters, exits and geometry are the sealed ones, run through the strategy's own replay loop.

**Deviation to disclose:** the first execution of this harness assigned each 2025 session's contract by dictionary order rather than by which contract dominates the session, which mis-assigned roll-week sessions (it left 2025-03-03 in and wrongly skipped 2025-09-01). It was fixed to use the same dominance rule that builds the bars, and re-run before any figure was reported. The corrected 2025 segment then matched the independent diagnostic exactly, which is how the bug was caught. No rule, threshold or parameter was changed.

## What this does and does not mean

- **Cite $8,281 / PF 1.646 / N=115 for GAP-1's Gate-0 window**, not the sealed $9,878 / 1.761.
- **It changes no decision.** GAP-1's promotion gate is prospective live N≥30, untouched by this and already UNDERPOWERED (`_bmad-output/diagnostics_gap_fade_power_gate_20260913/`). No parameter, seal or live setting was changed, and no unit was restarted.
- **`SEALED_PARITY_2025` in `gap_fade_live.py` stays pinned** to the frozen artifact (N=77, PF 2.017); it exists to detect code drift, not to carry the corrected figure.
- **A fifth of the sealed Gate-0 net came from 11 days of deferred-contract data.** That is worth remembering the next time a backtest window straddles a roll: the defect was invisible in every summary statistic until the bars were rebuilt.

## Outputs

- `rescore_gate0.py`, `rescore_results.json`, `corrected_gate0_trades.csv`
- this file
