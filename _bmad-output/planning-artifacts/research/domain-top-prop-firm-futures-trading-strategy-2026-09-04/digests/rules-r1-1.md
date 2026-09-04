# Claims

1. Topstep's Trading Combine has exactly one hard drawdown rule: "do not let your account balance hit or go below the Maximum Loss Limit (MLL)." The MLL is a trailing (ratchets up with end-of-day balance, never moves back down) drawdown, sized as account size minus a fixed allowance ($2,000 on $50K, $3,000 on $100K, $4,500 on $150K). Hitting it closes the account permanently with no appeal.
   Source: https://help.topstep.com/en/articles/8284197-trading-combine-parameters
   Publisher: Topstep (official Help Center)
   Publication date: not shown on page (accessed via live fetch)
   Access date: 2026-09-04
   Confidence: high
   Class: primary

2. Topstep's Combine "Consistency Target" requires the best single trading day to stay below 50% of the Profit Target ($1,500 of $50K's target, $3,000 of $100K's, $4,500 of $150K's). Exceeding it does not fail the evaluation — it raises the required profit target to (best day / 0.50) instead.
   Source: https://help.topstep.com/en/articles/8284197-trading-combine-parameters
   Publisher: Topstep (official Help Center)
   Access date: 2026-09-04
   Confidence: high
   Class: primary

3. Topstep's Combine can technically be passed in as few as 2 minimum trading days, though Topstep's own copy frames this negatively ("Big spike days don't build funded traders, consistency does"), signaling the firm's own preference against low-day-count passes even though the rule permits them.
   Source: https://help.topstep.com/en/articles/8284197-trading-combine-parameters
   Publisher: Topstep (official Help Center)
   Access date: 2026-09-04
   Confidence: high
   Class: primary

4. Topstep introduced (effective Feb 5, 2026) a split of its Express Funded Account into a "Standard EFA" (original payout policy) and a "Consistency EFA" that adds a 40% consistency target on payouts — largest single trading day cannot exceed 40% of total net profit in the payout window.
   Source: aggregated from proptradingvibes.com/blog/topstep-trading-combine-rules and other secondary sites returned by search (not independently verified on a Topstep primary page in this session)
   Publisher: PropTradingVibes (secondary/likely-affiliate site) citing Topstep
   Publication date: 2026 (exact date unclear)
   Access date: 2026-09-04
   Confidence: low
   Class: secondary — NOTE: this specific claim was surfaced only via WebSearch AI-summary of aggregator pages, not fetched from Topstep's own site directly; flagged for re-verification.

5. Topstep funded/Express accounts require closing all positions by 3:10 PM CT Monday–Friday (or the product's market close, whichever is first); no overnight or weekend holds are allowed, and holding into rollover is treated as a violation that closes the account.
   Source: WebSearch summary citing tradetanto.com/learn/topstep-rules and h2tfunding.com/topstep-funded-account-rules (secondary sites); not independently fetched from a Topstep primary page in this session — Topstep's own Scaling Plan article (help.topstep.com/en/articles/8284223) was fetched directly and explicitly does NOT mention overnight/weekend rules, so this claim is not primary-confirmed.
   Publisher: multiple secondary/affiliate-style prop-firm-guide sites
   Access date: 2026-09-04
   Confidence: low-medium
   Class: secondary

6. Topstep's Scaling Plan (Express Funded Account) sets maximum position size based on current account balance; max contracts do not increase mid-session — a trader must wait for the next session after crossing a balance threshold to unlock more size. Micro/mini contracts convert at 10:1 generally, with Micro Silver at 5:1 and Micro Bitcoin/Ether capped at mini-equivalent levels. The page was last updated July 16, 2026 and explicitly does not disclose payout split percentages or overnight/weekend rules.
   Source: https://help.topstep.com/en/articles/8284223-what-is-the-scaling-plan
   Publisher: Topstep (official Help Center)
   Publication/update date: 2026-07-16 (as shown on page)
   Access date: 2026-09-04
   Confidence: high
   Class: primary

7. Apex Trader Funding offers two trailing-drawdown types: EOD trailing (updates once at 4:59 PM ET close, more intraday flexibility) and Intraday trailing (tracks peak balance in real time including unrealized gains, locking once it reaches starting balance + $100, i.e., a "Safety Net"). Apex applies no consistency rule during the Evaluation phase; consistency rules (cited elsewhere as 30% or 50% depending on account type/product generation) apply only at PA (Performance Account)/Live payout stage.
   Source: WebSearch AI-summary drawing on apextraderfunding.com/help-center pages (help-center pages themselves returned 403 Forbidden when fetched directly in this session, so this is a secondhand rendering of Apex's own content, not a verified direct read)
   Publisher: Apex Trader Funding (content), relayed via WebSearch/aggregator summarization
   Access date: 2026-09-04
   Confidence: medium (content originates from what appears to be Apex's own help center per URLs, but direct fetch was blocked, so exact wording is unverified)
   Class: primary-sourced-but-unverified (treat as secondary until re-fetched)

8. MyFundedFutures uses different drawdown models per plan: EOD trailing (Pro, Builder, legacy Core), intraday trailing (Rapid), and EOD static (legacy Flex). Consistency rule is 30% for most plans but 50% for Core evaluation and Builder sim-funded. No daily loss limit applies on any MyFundedFutures plan, described as a firm-wide differentiator. Minimum payout is $500, trader must be net +$500 since last payout; the Rapid EOD plan specifically has a $2,100 initial buffer requirement, daily payout cadence, no per-cycle cap, and a 90/10 profit split.
   Source: WebSearch AI-summary citing financemagnates.com/forex/myfundedfutures-trades-intraday-drawdown-for-a-30-consistency-rule/ and proptradingvibes.com/blog/myfundedfutures-rules-overview / myfundedfutures-payout-rules
   Publisher: Finance Magnates (trade press, independent) + PropTradingVibes (secondary/likely-affiliate)
   Access date: 2026-09-04
   Confidence: medium (Finance Magnates is a credible independent fintech trade publication, which corroborates the 30% consistency figure; the granular payout mechanics come from the lower-confidence aggregator)
   Class: secondary

9. Earn2Trade's "Maintain Consistency" rule (Trader Career Path evaluation and Gauntlet Mini only — does not apply to LiveSim/Live accounts) states: "No single trading day can account for 30% or more of your total PnL." Failing it does not end the evaluation; it just requires continuing to trade until that day's share of cumulative profit falls under 30%. Page shows publication date May 7, 2026.
   Source: https://help.earn2trade.com/en/articles/3849975-what-is-the-maintain-consistency-rule
   Publisher: Earn2Trade (official Help Center)
   Publication date: 2026-05-07 (as shown on page)
   Access date: 2026-09-04
   Confidence: high
   Class: primary

10. Bulenox applies a 40% consistency rule at every payout request (no single trading day may exceed 40% of total profit balance); failing it does not violate/close the account, the trader simply must keep trading until the ratio resolves.
    Source: WebSearch AI-summary citing quantvps.com/blog/bulenox-consistency-rule and tradercore.io/blog/bulenox-prop-firm-review-rules-payouts
    Publisher: QuantVPS / TraderCore — both appear to be prop-firm-comparison/review sites with likely affiliate relationships to the firms they review
    Access date: 2026-09-04
    Confidence: low-medium (not independently verified against Bulenox's own site in this session)
    Class: secondary/aggregator — TREAT WITH SKEPTICISM per brief's instruction on comparison sites.

11. Across the sampled firms (Topstep, Apex EOD accounts, Take Profit Trader evaluations excluding PRO, MyFundedFutures on most tiers, Bulenox, Earn2Trade, FundedNext Futures), end-of-day or realized trailing drawdown is the dominant drawdown model as of an April 2026 comparison; intraday/real-time trailing (e.g., Apex's intraday option, MyFundedFutures Rapid, some PRO-tier accounts) is a minority but present model.
    Source: WebSearch AI-summary drawing on multiple aggregator sites (lunefi.com/blog/earn2trade-complete-guide-to-rules-and-payouts and others in the same result set)
    Publisher: aggregator/comparison sites, unnamed original attribution for the "as of April 2026" dating
    Access date: 2026-09-04
    Confidence: low
    Class: aggregator

12. Trailing drawdown structurally punishes trend-following/swing strategies more than scalping because the floor ratchets upward on every new open-trade or closed-trade equity high; a trend trader who lets a profitable trade run and then gives back part of the open profit on a pullback can be stopped out by the ratcheted floor even without a realized losing day. Intraday (real-time) trailing is described as especially severe because it demands taking profits near the peak of a move — the opposite of what trend-following requires (letting winners run through pullbacks). By contrast, static drawdown does not tighten on unrealized gains, so it does not create this same penalty for round-trip/pullback behavior.
    Source: WebSearch AI-summary aggregating multiple content-marketing sites: proptradingvibes.com/blog/what-is-trailing-drawdown, propfirmscompared.com/blog/trailing-drawdown-prop-firms-how-it-works, tradeify.co/post/trailing-drawdown-explained-for-prop-firm-traders, maventrading.com/blog/trailing-drawdown-prop-trading — none of these are academic/quant analyses; all are prop-firm-adjacent content/affiliate sites making essentially the same argument in similar language, suggesting either genuine industry consensus or content-mill repetition of a single popularized framing.
    Publisher: multiple prop-firm content/comparison sites (likely affiliate-monetized — several of these sites list and rank prop firms)
    Access date: 2026-09-04
    Confidence: low (directionally plausible and mechanically explainable from the rules themselves — see claim 1, 7 — but NO firm's own rules page, academic paper, or named-author quant breakdown was found in this session making this argument explicitly; it is inference repeated across content-marketing sites, not a cited primary analysis)
    Class: aggregator (inference, not sourced to a named quant/trader analysis)

13. Consistency rules (claims 2, 8, 9, 10) are explicitly marketed by the firms/commentary as designed to prevent one lucky/concentrated big-winning day from qualifying a trader — i.e., they mechanically punish concentrated-big-winner-day strategies (a single large trend trade or high-R:R outlier win) relative to strategies that produce many small, evenly-distributed winning days, independent of total profit. This is stated directly in Topstep's own combine framing ("Big spike days don't build funded traders, consistency does" — claim 3) and Earn2Trade's own rule description (claim 9), both primary sources, though neither firm explicitly uses the words "trend-following" or "R:R" — the connection to strategy *shape* is the requester's/commentary's inference from the mechanical percentage rule, not the firms' own stated rationale in those exact terms.
    Source: https://help.topstep.com/en/articles/8284197-trading-combine-parameters ; https://help.earn2trade.com/en/articles/3849975-what-is-the-maintain-consistency-rule
    Publisher: Topstep, Earn2Trade (both primary)
    Access date: 2026-09-04
    Confidence: medium (the mechanical fact is primary-sourced; the "punishes concentrated big-winner days" framing is a direct, low-inference reading of the rule but not the firms' own words verbatim)
    Class: primary (for the rule text) / inference layer on top

14. One search specifically for "rule-gaming" tactics (trading small size across many days purely to satisfy day-count/consistency rules, then scaling) did not surface a credible, evidenced trader account, forum thread, or independent analysis — only tangential content: a Benzinga explainer on Apex's consistency rule, a TradingView "Consistency Rule Calculator" script (implying enough demand that traders build tools to manage the rule, which is circumstantial evidence the rule is actively "gamed"/managed rather than organically satisfied), and general prop-firm-guide commentary that the consistency rule exists specifically to catch traders who "trade a large contract one time while trading micros the rest of the time" or "flip contracts, change contract sizes constantly" — i.e., the firms' own stated threat model already assumes this exact gaming pattern is attempted.
    Source: WebSearch results including https://www.tradingview.com/script/ueMUDlgW-Consistency-Rule-Calculator and https://www.benzinga.com/money/apex-consistency-rule-explained
    Publisher: TradingView (user-generated script marketplace), Benzinga (financial media, describing Apex's rule)
    Access date: 2026-09-04
    Confidence: low (existence of a calculator tool and a firm's stated rationale for the rule are circumstantial evidence of gaming behavior, not documented evidence that traders successfully execute this tactic or that it "changes what strategy means")
    Class: secondary/circumstantial

# Leads

- Apex Trader Funding's own help-center pages (apextraderfunding.com/help-center/... and support.apextraderfunding.com) returned HTTP 403 to direct WebFetch in this session — a second round should retry via a different fetch method (e.g., Google cache, a different user agent, or an MCP browser tool) to get Apex's rules verbatim rather than relying on WebSearch's AI-summarized rendering (claim 7).
- The claim that Topstep added a "Consistency EFA" 40% payout rule on Feb 5, 2026 (claim 4) and the overnight/weekend 3:10 PM CT close rule (claim 5) were both surfaced only via secondary/aggregator sites, not Topstep's own site directly. Topstep's own "Live Funded Account Rules" (topstep.com/live-funded-account-rules) and "Express Funded Account Rules" (topstep.com/express-funded-account-rules) pages appeared in search results but were not fetched in this round — high-value primary source to fetch next.
- No academic paper, regulatory filing, or named individual quant's published breakdown was found connecting consistency-rule/trailing-drawdown mechanics to strategy-shape viability (win rate vs. R:R, trend-following vs. mean-reversion). All such connections found in this round come from unnamed, likely-affiliate-monetized content sites repeating similar framing (claim 12) — worth a targeted second-round search for a named trader/quant (e.g., a YouTube quant, a Substack, an arXiv/SSRN paper) making this argument with actual backtested numbers, since none surfaced despite a direct query.
- MyFundedFutures' "no daily loss limit on any plan" (claim 8) is a distinctive and consequential rule if true (it would remove one whole axis — daily loss limits — from the "rules of the game" comparison for that firm) — worth direct primary-source verification on myfundedfutures.com.
- Take Profit Trader and Elite Trader Funding were named in the original question's firm list but were not directly investigated with a dedicated primary-source fetch in this round; only tangential aggregator mentions surfaced (e.g., "Take Profit Trader evaluations (not PRO)" using EOD trailing).
- The reddit-specific query (r/Daytrading, r/FuturesTrading forum discussion, as the brief specifically requested) did not surface any actual Reddit thread in results — worth a more targeted second round using `site:reddit.com` style queries or a dedicated Reddit search, since the brief explicitly names these subreddits as a priority source class and none were retrieved.
- Contradiction to flag: one secondary source (claim 4, "50%" combine consistency vs "40%" EFA payout consistency) suggests Topstep applies different consistency percentages at the evaluation stage (50%, confirmed primary in claim 2) vs. the funded/payout stage (40%, unconfirmed secondary in claim 4) — worth confirming directly against topstep.com/express-funded-account-rules.

# Not found

- Could not directly verify Apex Trader Funding's rules text from Apex's own site (403 Forbidden on two attempted fetches); relied on WebSearch AI-summary of Apex-hosted URLs instead.
- Could not find a primary source (firm page) or a named/credentialed independent analysis explicitly connecting trailing-drawdown/consistency-rule mechanics to win-rate/R:R strategy math with actual numbers or backtests — only unnamed content-site inference (see claim 12, leads).
- Could not find any actual Reddit (r/Daytrading, r/FuturesTrading) or Trade2Win forum thread discussing rule-gaming tactics, despite a direct search — could not establish community-sourced evidence on whether/how traders game day-count or consistency rules, or whether doing so "changes what strategy means" as the brief's question 3 asks.
- Did not obtain primary-source rules pages for Take Profit Trader, Elite Trader Funding, or Bulenox (all three were only covered via WebSearch AI-summaries of secondary/aggregator sites, not direct fetches of the firms' own sites) — daily loss limit specifics and exact scaling-plan/payout-split numbers for these three firms remain unconfirmed at primary-source level.
- Could not confirm exact payout splits (e.g., Topstep's, Apex's) at primary-source level within this round's budget — only MyFundedFutures' 90/10 (Rapid EOD) surfaced, and only via secondary source (claim 8).
