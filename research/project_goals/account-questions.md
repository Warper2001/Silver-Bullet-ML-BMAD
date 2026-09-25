# Account verification packet — 2026-09-19

Engineering status: provisional configurable model implemented. Evidence status: account-specific terms and invoices pending. No funds transfer or provider communication is authorized by this packet.

Confirm account ID, phase, purchase date, platform, account reset dates and starting balance against a statement. The current read-only account export identifies account 26556101; do not infer a retired account merely from old fills. Obtain dated agreements and itemized invoices for combine subscriptions/resets, activation, API, market data and other operating costs. A published price is a scenario, not an actual expense.

Confirm the 50K floor calculation ($2,000 trailing end-of-day, intraday unrealized breach, floor never decreases), timing of phase changes, profit target, permitted quantity and any scaling restrictions. Confirm the cushion required after payout and whether pending fees affect it. Specify how all accounts aggregate risk and whether automated SIM behavior can continue after combine passage or a Live call-up.

Confirm payout route: Standard five days of at least $150 net, positive net since prior payout with first-payout exception; or Consistency at least three days and largest day no more than 40% of total. Confirm minimum $125, 90% operator split, 50%-of-balance limit, and the account's actual cap ($2,000 Standard/$3,000 Consistency published for 50K before purchase-date/DLL promotions). Verify grandfathering and whether the floor moves to zero on the first payout. Confirm what ends the XFA and how remaining balance is handled at Live call-up.

Published 50K scenarios: Standard $49/month plus $149 activation, No Activation $95/month, resets $49/$95; API published $29/month ($14.50 discount applicability unknown), optional Level2 $38/month, Level1 included. Taxes and invoice applicability are unknown.

The API page currently excludes Live accounts and remote order transmission. This is a deployment constraint requiring a documented resolution, not a research setting to toggle. The Live route in the model is unavailable. Direct self-funded $10,000 is modeled separately; it is not a recommendation to transfer funds. $20,000 monthly remains an aspiration without supported earning capacity.

Sources checked by lead September 19, 2026:
- [API access](https://help.topstep.com/en/articles/11187768-topstepx-api-access)
- [Payout policy](https://help.topstep.com/en/articles/8284233-topstep-payout-policy)
- [Maximum loss limit](https://help.topstep.com/en/articles/8284204-what-is-the-maximum-loss-limit)
- [Live parameters](https://help.topstep.com/en/articles/10657969-live-funded-account-parameters)
- [Pricing](https://help.topstep.com/en/articles/14289835-topstep-pricing-and-payment-questions)

## Response/version register

| Question group | Version checked | Account-specific evidence | Status | Next action |
|---|---|---|---|---|
| Identity / epoch / balance | September 19 broker read-only snapshot | No signed statement | Provisional | Obtain statement and reset history |
| Fees and discounts | September 19 published prices | No invoices | Pending | Record invoice dates, amounts and taxes |
| Payout cap / grandfathering | September 19 public policy | No agreement | Pending | Confirm purchase-date applicable route |
| Floor / cushion / phases | September 19 public policy | No written account ruling | Pending | Confirm first payout and call-up effects |
| Automated route / hosting | September 19 API restrictions | No written exception | Unavailable for Live; hosting unresolved | Obtain written confirmation before deployment decision |

Append responses with date, responder, exact applicable account IDs, source artifact hash, effective date, expiry/recheck date and reviewer. Preserve prior versions. Do not silently replace a published scenario with an asserted actual charge.

Published Combine consistency as checked September 19: target is `max(original profit target, largest profit day / 0.55)`, with at least two trading days. This replaces older public 50% references; confirm applicability to the actual account before treating a modeled transition as passage. [Current Combine consistency](https://help.topstep.com/en/articles/8284208-consistency-at-topstep), [Combine parameters](https://help.topstep.com/en/articles/8284197-trading-combine-parameters).

## TradeStation personal equities account (added 2026-09-25, ETFTM-1 vehicle recon)

Source: `_bmad-output/planning-artifacts/research/domain-vehicle-economics-modest-sharpe-book-2026-09-25/research.md`. Ask TradeStation Client Experience in writing. These answers must exist before any ETFTM-1 paper or live execution (A5).

| # | Question | Why it matters | Status |
|---|---|---|---|
| TS-1 | Can IRA accounts place orders through the v3 API? What `AccountType` do they show? | The IRA is the best vehicle on taxes, but the API spec lists only Cash, Margin, Futures and DVP | Open |
| TS-2 | Does the $0.003/share Tier-1 clearing fee apply to every equity order, or only direct-routed ones? | Cost model | Open |
| TS-3 | Are fractional-share orders supported through the API? | Whole-share rounding at $25–50K | Open |
| TS-4 | What counts as "minimum activity" for the $10/month inactivity fee? | A monthly rebalance may or may not qualify | Open |
| TS-5 | Does any terms-of-service clause restrict automated or API trading in personal accounts? | Allowed use of automation | Open |
| TS-6 | What is the order cutoff for `CLO` (market or limit on close) ETF orders, and how are they routed? | Whether live fills match the backtest | Open |
| TS-7 | When does TradeStation apply FINRA's 2026 intraday-margin framework? | Margin-account rules | Open |
| TS-8 | Is interest paid on cash in IRAs, or on balances under $100K? The pricing page says no. | Cash drag; plan to hold a T-bill ETF instead | Open (confirm) |
