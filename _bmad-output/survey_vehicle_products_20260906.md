# Vehicle product survey — what actually meets the spec

**Date:** 2026-09-06
**Spec derived from:** `result_vehicle_allowance_curve_20260906.md`
**Method:** web search, September 2026. **Prop-firm terms change constantly and marketing pages are unreliable — everything below marked "verify" must be confirmed directly with the firm before any purchase.**

## The spec we're shopping against

| requirement | value | why |
|---|---|---|
| Drawdown allowance | **~$4,000 trailing** or **~$2,500–3,000 static** | knee of the survival curve; past $5,000 buys nothing |
| Daily loss limit | **> $1,000, or none** | MIM-NB's single cat-stop is $1,000 — a $500 DLL is fatal on one trade |
| Instrument / size | MNQ micros, ≥3 contracts | MIM 1ct + YANK 2ct |
| Floor regime | static > EOD-trailing > intraday-trailing | static ≈ worth $1,000–1,500 of extra allowance |

## Finding 1 — a caveat in my own model, and it runs against the current account

**There are three floor regimes, not two.** I had been treating this as static-vs-trailing; the real taxonomy is:

1. **Static** — floor fixed at start, never moves.
2. **EOD trailing** — floor recalculates only at market close, on end-of-day *balance*.
3. **Intraday trailing** — floor ratchets on live equity *including open profit*. Harshest.

**Topstep uses intraday trailing.** That is precisely what killed account 23884932: peak equity $50,955 *including an open YANK position* ratcheted the floor to $48,955 (`project_combine_blown_20260706`).

**But my simulation ratchets the floor once per day on end-of-day balance** (`floor = min(START, max(floor, bal - mll))`, applied after the day loop) — i.e. **my "trailing" column models EOD trailing, not Topstep's intraday rule.** The sealed engine does the same.

Consequence: **the modelled 28.7% blow rate for the current account is optimistic.** Real Topstep is harsher than what I simulated. The trailing column is a fair model of an *EOD-trailing* product — not of the account we're actually on.

Which means simply moving from Topstep to an **EOD-trailing firm at the same $2,000** is already an improvement, before any change in allowance.

## Finding 2 — consistency rules are structurally hostile to this strategy, and vary a lot

MIM-NB is tail-capture: a handful of fat days carry the edge (`project_mim_nb_expectations_reconciled`). Every one of these firms imposes a **consistency rule** capping best-day profit as a share of total.

- Topstep (current): best day < **50%** of profit — already modelled in the MC's pass condition
- TradeDay: **30%** *(verify)*
- Tradeify Select: **40%** *(verify)*
- MyFundedFutures: reportedly traded intraday drawdown away **for** a 30% consistency rule *(verify)*

**A 30% rule is materially worse than 50% for a strategy whose profit arrives in 3 days out of 163.** Shopping purely on drawdown and ignoring consistency could buy more room and then fail on a rule that penalises exactly the profit shape this strategy produces. **Consistency % belongs in the spec alongside allowance.**

## Candidates

| firm | floor regime | allowance (50K-class) | daily loss limit | fit |
|---|---|---|---|---|
| **Topstep** (current) | **intraday** trailing | $2,000 | none | current — worst regime, half-sized |
| **TradeDay** | offers **all three** (EOD / intraday / **static**) variants | static amount **unverified** — one source implies as little as $500 with a $1,500 target | **none, either phase** | structurally the best match *if* the static allowance is ≥$2,500 — **must verify** |
| **Alpha Futures** | **EOD trailing on all plans** | Zero 4% ($1K/$2K/$3K by size); Advanced 3.5% (**$1,750 / $3,500 / $5,250**) | *(verify)* | Advanced mid-tier **$3,500 EOD** lands near the $4,000 trailing target |
| **Tradeify** | **EOD**, locks static after a threshold (Growth 50K locks at $52,100) | $2,000 | **$1,250** | DLL only $250 above a single cat-stop — tight |

## Read

**No single product is a confirmed clean fit yet**, and the two most promising need direct verification:

1. **Alpha Futures Advanced** — $3,500 EOD trailing is the closest *verified* number to the $4,000 trailing target, and EOD is a strictly better regime than what we're on. Need: the account size that carries $3,500, its daily-loss and consistency rules.
2. **TradeDay static** — the only firm found offering a genuine **static** floor *and* no daily loss limit anywhere, which is the ideal combination. But the one static figure surfaced (~$500 on 50K, $1,500 target) would be **fatal** — one cat-stop ends it. Its real static allowance across sizes is the single highest-value thing to confirm.

**The cheapest improvement available needs no new capital and no bigger account:** leaving intraday trailing for an EOD or static floor of the *same* $2,000. On the modelled curve, static at $2,000 versus trailing at $2,000 is 12.1% vs 28.7% blow and 66.1% vs 61.5% pass — and since real Topstep is harsher than my trailing column, the true gain is larger than that.

## What to verify before spending anything

1. TradeDay's actual **static** drawdown by account size (the make-or-break number).
2. Alpha Futures Advanced: which size carries the **$3,500** MLL, plus DLL and consistency %.
3. Consistency % for every shortlisted product — **30% is likely disqualifying for MIM-NB.**
4. Whether MNQ micros and ≥3 concurrent contracts are permitted on the specific plan.
5. Payout/eligibility terms — outside this analysis, but a static floor is worth nothing if the payout rules don't work.

## Sources

- [Static vs Trailing Drawdown 2026 guide — propfirm-reviews.com](https://propfirm-reviews.com/static-vs-trailing-drawdown/)
- [No Daily Drawdown futures challenge list — propfirmmatch.com](https://propfirmmatch.com/futures/prop-firm-lists/challenge-types/no-daily-drawdown)
- [Static drawdown futures challenge list — propfirmmatch.com](https://propfirmmatch.com/futures/prop-firm-lists/challenge-types/static-drawdown)
- [EOD trailing vs intraday drawdown — Alpha Futures](https://alpha-futures.com/resources/eod-trailing-drawdown-vs-intraday-drawdown-2026)
- [Best futures prop firms 2026 — Alpha Futures](https://alpha-futures.com/posts/best-futures-prop-firms-2026-a-traders-guide)
- [Tradeify rules 2026 — tradetanto.com](https://tradetanto.com/learn/tradeify-rules-explained-what-every-trader-should-know)
- [Rules: Trailing Max Drawdowns — Tradeify Help Center](https://help.tradeify.co/en/articles/10495897-rules-trailing-max-drawdowns)
- [TradeDay rules 2026 — proptradingvibes.com](https://www.proptradingvibes.com/blog/tradeday-rules)
- [TradeDay static 50K review — thetraderstack.com](https://www.thetraderstack.com/reviews/tradeday-static-50k)
- [MyFundedFutures intraday drawdown / 30% consistency — financemagnates.com](https://www.financemagnates.com/forex/myfundedfutures-trades-intraday-drawdown-for-a-30-consistency-rule/)
