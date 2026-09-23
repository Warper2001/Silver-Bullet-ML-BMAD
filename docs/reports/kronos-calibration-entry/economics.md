# TradeStation economic-input worksheet

All dollar amounts are USD. Published schedule observations were retrieved on 2026-09-23. No account, invoice, credential or trade ledger was queried. References and captured source bytes are in [source-notes.md](source-notes.md) and `sources/`. Nothing below adopts a numerical strategy effect or cost for evaluation.

| Component | Documentary observation | Account evidence / treatment |
|---|---|---|
| Broker commission and contract charge | TradeStation's public pricing table labels commission $0 and separately lists micro contract-side charges of $0.50, $0.40, $0.30, $0.25 across ascending monthly volume tiers (0–500, 501–1,000, 1,001–10,000, >10,000). [TS pricing](https://www.tradestation.com/pricing/) | Do not confuse the $0 label with zero broker cost. Actual plan, platform, tier calculation, negotiated terms and promotions unknown. Keep broker charge `b` symbolic. |
| Broker clearing | Same table lists $0.10/side in the first three tiers and $0 at the highest tier. | `c` unknown for this account. Broker clearing is separate from exchange clearing. |
| Exchange execution/clearing | Published nonmember MNQ charge $0.35 per contract-side. [TS exchange fees](https://www.tradestation.com/pricing/exchange-execution-and-clearing-fees/) | `e` conditional on membership/product/date; do not add a second exchange-clearing charge already included here. |
| Regulatory | NFA currently states $0.01/side from July 1, 2026; $0.02/side from July 1, 2027. [NFA assessment FAQ](https://www.nfa.futures.org/faqs/members/nfa-assessment-fees.html) | `r(t)` date-dependent, subject to applicability. The 12-month scenario crosses this announced change. |
| Contract units | MNQ $2/point; outright tick 0.25 point = $0.50. [CME FAQ](https://www.cmegroup.com/articles/faqs/micro-e-mini-equity-index-futures-frequently-asked-questions.html) | Unit conversion supported; not a margin, capital or risk estimate. |
| Slippage/spread/latency | No applicable execution evidence in the inspected packet. SIM uses instant simulated fills. [TS SIM](https://api.tradestation.com/docs/fundamentals/sim-vs-live/) | Keep arm-specific adverse slippage `s` in ticks/filled side, plus timing effects. SIM fills cannot validate it. Spread, impact and delay cannot be assumed independent or counted twice. |
| Operating cost | Public market-data pricing is not an account entitlement record. [TS data pricing](https://www.tradestation.com/pricing/market-data-pricing/) | Data, API/platform, compute, storage, electricity, maintenance and possible service fees `F_j` unknown; document incremental and fully allocated costs separately. Existing access does not prove zero incremental cost. |
| Capital / time / profit | No operator inputs supplied. | Allocated capital `A_j`, opportunity-cost rate `q_j`, operator hours `h_j`, hourly value `w`, minimum useful profit `P_K` and incremental profit `P_D` remain unknown. Neither the $20k aspiration nor available N supplies these inputs. |

Illustration only, conditional on the lowest published volume tier, applicable nonmember fees and the current NFA rate: broker + broker clearing + exchange + regulatory = `0.50+0.10+0.35+0.01 = $0.96/side`, or `$1.92/round trip` before slippage and operating costs. With an explicitly assumed one-tick adverse move on each of two fills, the illustration becomes `$2.92`. From July 2027 the analogous fee-only illustration is `$1.94`, holding other rates fixed. These are schedule arithmetic, not an account quote or adopted simulation costs.

For arm `j` in session `t`, let `G_jt` be gross price PnL at the specified benchmark, `B_jt` filled contract-sides, and `L_jt` the total execution loss versus that benchmark. Define trading-net session PnL

`X_jt = G_jt - sum_over_sides(b+c+e+r(t)) - L_jt`.

If adverse slippage is already embedded in execution prices, it is already in `G_jt` relative to unadjusted fills and must not also be subtracted as `L_jt`. Reversals incur two sides; unchanged targets incur none. On the same predeclared eligible sessions, `K=X_K`, `M=X_M`, `D=K-M`. Different turnover and fill times mean identical fee schedules do not imply identical total costs. Their difference need not cancel. No turnover is measured here.

Choose an operational planning period `H` independently of statistical N and a supported expected eligible session count `S_H`. With fixed costs `F_j(H)`, labor `w*h_j(H)`, and capital opportunity cost `q_j(H)*A_j`, define `O_j(H)` as their sum. Keep one-time preparation/research cost separate and choose its amortization period explicitly. A trading-net useful effect can then be expressed as:

`u_K = [P_K(H)+O_K(H)] / S_H`

`u_D = [P_D(H)+O_K(H)-O_M(H)] / S_H`.

These are accounting expressions, not numbers to tune against sample capacity. Incremental capital costs need an explicit deployment comparison (replace momentum versus add Kronos); no automatic equal-capital cancellation. Taxes and withdrawals are outside this pre-tax evidence decision. Risk and drawdown constraints must be independently specified, not inferred from a positive mean.

| Concept | Kronos | Kronos minus momentum |
|---|---|---|
| Existing zero-profit null | `H0_K: E[K] <= 0` | `H0_D: E[D] <= 0` |
| Economically useful effect | `u_K`, requiring operator costs/profit/capital inputs | `u_D`, requiring incremental deployment economics |
| Planning alternative | Independent plausible `a_K > 0`; evidence must identify its economic relevance | Independent plausible `a_D > 0`; cannot import another strategy's effect |
| Existing statistical requirement | Positive lower bound for mean K using preregistered one-sided alpha .025 | Positive lower bound for mean D using preregistered one-sided alpha .025; both required |
| Eventual useful-performance acceptance | A separately proposed rule could require lower bound `> u_K` | A separately proposed rule could require lower bound `> u_D` |

The last row is an unresolved design choice, not a new sealed acceptance rule. Rejecting zero does not prove useful performance. Power for rejecting zero uses the distance `a_j`; power for rejecting a useful-effect boundary uses `a_j-u_j`, which must be positive and independently justified. Planning with `a_j=u_j` cannot give 90% power to put a lower confidence bound above `u_j`. No numerical planning alternatives are adopted. No income target, risk threshold, or filter is being sealed.
