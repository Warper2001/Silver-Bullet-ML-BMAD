# VPOC-1 Backtest Results

Pre-registration: `_bmad-output/preregistration_vpoc1_volume_profile_fade.md`

Data: 126687 RTH 1-min bars, 370 sessions, MNQ 2025-01-01 -> latest available.

## Lookback sweep (dev window 2025, pre-declared grid, locked on max PF)

| N | Trades | EV/trade | PF | WR | be_WR | Worst month avg |
|---|---|---|---|---|---|---|
| 1 | 52 | $-36.37 | 0.625 | 42.3% | 51.6% | $-236.24 |
| 3 | 66 | $-79.04 | 0.449 | 43.9% | 61.8% | $-209.24 |
| 5 | 65 | $-25.62 | 0.765 | 53.8% | 65.6% | $-185.74 |
| 10 | 57 | $-27.21 | 0.805 | 43.9% | 66.0% | $-472.64 |

**Locked N = 10**

## Gate 0 (dev window 2025, locked N)

N=57, EV=$-27.21, PF=0.805, WR=43.9%, be_WR=66.0%, worst_month_avg=$-472.64, avg_RR=0.52

**Gate 0 verdict: FAIL**

## Gate 1 (holdout 2026 YTD, same locked N)

NOT EVALUATED -- Gate 0 did not PASS (verdict: FAIL); holdout not spent per pre-registration.

## Diagnostic: weekly-VWAP confluence filter impact (informational only)

Sessions with a VA-edge touch in a balance session (locked N=10), before the weekly-VWAP filter: 84. After filter (actual trades taken): 57.

## Verdict

**FAIL at Gate 0.** Per the pre-registration this is terminal for VPOC-1 as specified -- no re-sweep, no new lookback grid, no new bin size or Value Area %, no added instrument under this seal.

## Diagnostic note (observational only -- does not reopen the gate)

Checked after the fact to make sure the FAIL is a real finding and not an implementation bug: an initial version of this backtest showed implausible results (PF 53, WR 92%, thousands of trades) traced to a same-session re-entry bug -- the simulator was re-triggering new trades within a session after the first one closed, in violation of the pre-registration's own "first qualifying bar each session only" rule. That bug is fixed (one trade per session, enforced); the corrected numbers above are what actually ran under the frozen spec. Every cell in the pre-declared grid fails (PF 0.449-0.805, all well under the 1.20 gate) -- this is not a marginal or cherry-picked result.

Two structural reasons, both visible in the numbers:

1. **The exit geometry is asymmetric against the trade before a single bar is simulated.** Stop = 1.0x the Value Area's own width, placed beyond the entry edge; target = POC, which sits *inside* the Value Area. Since POC can be at most one VA-width away from the entry edge (and is typically nearer the middle), the best-case reward:risk is close to 1:1, and the realized average (0.52:1 at the locked N) is worse -- meaning the frozen, non-tuned stop rule (a deliberate choice to avoid hand-picking a value) happens to produce a breakeven bar (66.0%) that is unusually high for a mean-reversion setup. This is a property of the *rule as specified*, not a bug -- but it means this specific exit-geometry choice was working against the trade from the start, independent of whether the underlying reversion thesis has any truth to it.
2. **Even the raw win rate falls short.** At 43.9% WR, price failed to fully retrace to POC (before either hitting the wide stop or timing out) in over half of qualifying setups -- consistent with the same underlying finding as HCVWAP v1-v3: MNQ's momentum character reasserts itself often enough that VA-edge excursions, even from a classically-defined "balance" open, do not reliably mean-revert. Only 84 of ~360 eligible sessions (about 23%) produced a VA-edge touch from a balance open at all, and the weekly-VWAP confluence filter cut that further to 57-66 depending on lookback -- the setup is rare as well as unprofitable.

Net: this is a second, independent piece of evidence (distinct construct, distinct data statistic -- volume-at-price rather than a moving average) pointing at the same conclusion as HCVWAP: MNQ mean-reversion off a value-area/VWAP reference does not survive this shop's cost structure at any construction tested so far. A future iteration on this specific thread would need a different exit-geometry rule (e.g., target and stop both derived symmetrically, or a partial target short of full POC) and, per the pre-registration's own open questions, real L3 data to build an accurate profile -- neither is in scope for a re-sweep of this seal.
