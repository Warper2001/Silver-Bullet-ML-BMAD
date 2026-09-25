# Vehicle decision memo — ETFTM-1 (2026-09-25)

**Evidence:** `_bmad-output/planning-artifacts/research/domain-vehicle-economics-modest-sharpe-book-2026-09-25/research.md`. It has 28 cited sources: 6 claims verified and 3 unverified in the ledger. The income table comes from `income_math.py` in the same folder.
**Plan:** `/root/.claude/plans/do-d-then-a-imperative-eagle.md`, Checkpoint step.

## What D found

| Candidate | Holds multi-week positions? | Automation | Tax | Verdict |
|---|---|---|---|---|
| Futures prop firms (Topstep, Apex, Alpha, Tradeify) | **No.** Flat daily; Topstep says "No swing trading" | Topstep: **no API in Live Funded** (verified) | 1099 contractor income | **Excluded** |
| Trade The Pool (equities prop) | Yes, on Swing accounts | "Beta", needs approval | 1099 | Poor fit: 30% best-position cap, 14-day inactivity rule |
| **TradeStation IRA** | Yes | **API support unverified** (TS-1) | No current tax | **Best, if TS-1 = yes** |
| **TradeStation taxable cash account** | Yes | API supports Cash accounts | Mostly short-term gains at ordinary rates; wash sales defer losses | **Default, if TS-1 = no** |
| TradeStation margin account | Yes (long/short) | API supports Margin accounts | Same as cash | Not needed for long/flat; 11.75% margin rate |
| Do nothing: buy a managed-futures ETF | n/a | n/a | CTA reports on a 1099 | Last decade Sharpe about 0–0.4; fees 0.75–0.90%; big dispersion between funds |

**Frictions:**
- US ETF commissions are $0.
- The clearing fee is negligible even at worst case (TS-2).
- **Cash earns 0% below $100K**, so the strategy should hold a T-bill ETF instead of idle cash.
- The PDT rule is gone (effective 2026-06-04) and was irrelevant at monthly rebalancing anyway.

**Honest income** (10% volatility, before tax):

| Sharpe | $25K | $50K | Losing years |
|---|---|---|---|
| 0.5 | $1,250/yr | $2,500/yr | about 31% |
| 0.7 | $1,750/yr | $3,500/yr | about 24% |

The $20K/month aspiration needs about $3.4–4.8M at these Sharpes.

**Side finding for the MNQ combine (goal 5):** Topstep Live Funded accounts cannot trade through the API. So the automated MIM-NB/YANK route stops at the Express Funded (XFA) stage. That was already marked "Unavailable" in `account-questions.md`, and is now re-verified verbatim.

## Checkpoint questions for Alex

| # | Question | Recommendation |
|---|---|---|
| 1 | **Account type:** IRA (if TradeStation confirms API support) or taxable cash, either way long/flat with gross exposure ≤ 1.0? Or margin with long/short? | IRA if TS-1 = yes, otherwise cash; long/flat |
| 2 | **Sizing capital:** $25K or $50K for the pre-registration? The other becomes a sensitivity. | $50K: less whole-share rounding error |
| 3 | **Claim:** pooled *timing skill* (the per-asset-intercept test, which answers the Huang et al. critique), or a "robust diversified book", which is really a separate risk-premia pre-registration? | Timing skill |
| 4 | **Income:** do you accept that this is a foundation and research-process project, not an income route at $25–50K? | — |
| 5 | **Go / no-go on A,** given that realized trend performance over the last decade is weak and the likely outcome is UNDERPOWERED or FAIL? | Go: A0 and A1 are cheap, and a clean null closes the question for good |

Also needed before A5 (paper trading), but not before A0–A4: written answers from TradeStation to TS-1 through TS-8 (`research/project_goals/account-questions.md`).

## Alex's answers (2026-09-25)

1. **Account:** IRA if TradeStation confirms API trading in IRAs (TS-1), otherwise taxable cash. **Long/flat, gross exposure ≤ 1.0.**
2. **Sizing capital:** **$50K**, with $25K as a sensitivity.
3. **Claim:** **timing skill**, tested with the pooled per-asset-intercept test.
4. **Income:** acknowledged, implicitly, by choosing Go on the question that stated "a foundation and research-process project, not an income route".
5. **Go** on A: proceed to A0 (data), then A1 (power gate).
