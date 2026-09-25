# Digest r3-verified-edges-r1-1: Who demonstrably has a trading edge, and what kind

Research run 2026-09-25. Budget: 18 tool calls, 7 distinct sources/publishers. Every claim below comes from a source retrieved during this run. Anything else is marked as an unverified belief or listed as a lead.

Source-type labels used here: **peer-reviewed**, **regulatory filing**, **book**, **press**.
Retrieval-mode labels: **fetched** (page or PDF text read directly), **search-summary** (read only through a search engine's synthesized snippet, so weaker).
Class labels: **documented-fact** (the source reports it as an observed fact), **author-estimate** (a model-based estimate by the authors), **self-reported** (a firm's own statement, including statements in filings), **reported-figure** (press reporting of a figure from a non-public document), **inference** (this digest's own synthesis, not a source claim).

---

## Claims

### Q1: Renaissance Technologies, Medallion Fund

**C1.** Medallion's reported annual returns averaged 66% before fees from 1988 to 2018.
- Source: https://jpm.pm-research.com/content/46/4/156 (also https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3504766)
- Author/publisher: Bradford Cornell, "Medallion Fund: The Ultimate Counterexample?", *Journal of Portfolio Management* 46(4), p.156
- Pub date: 2020 (SSRN posting earlier; JPM issue dated 2020)
- Source type: peer-reviewed (a practitioner journal; the underlying return series is not audited, see C4)
- Accessed: 2026-09-25, search-summary. The SSRN page returned 403 and pm-research answered with a redirect that the budget left unfetched.
- Confidence: medium. The figure appears consistently across the abstract listings, but the page text was not read directly.
- Class: documented-fact, with the caveat that it documents *reported* returns

**C2.** $100 invested in Medallion at the start of trading in 1988 would have grown to $398.7 million by 2018, a compound return of 63.3%.
- Source, authors, date and type: same as C1
- Accessed: 2026-09-25, search-summary
- Confidence: medium
- Class: documented-fact (derived from reported returns)
- Note: I could not confirm from retrieved text whether this series is gross or net of fees. The widely cited net figure of about 39% was **not** found in retrieved text.

**C3.** Medallion never had a negative year over the 31 years, including the dot-com crash and the financial crisis. Its market beta and factor loadings were all negative, so Cornell argues its performance cannot be read as a premium for bearing risk. He frames it as a counterexample to market efficiency.
- Source, authors, date and type: same as C1
- Accessed: 2026-09-25, search-summary
- Confidence: medium
- Class: documented-fact (no negative year) plus author-estimate (factor loadings)

**C4.** No regulatory, court or tax filing that discloses Medallion's returns was retrieved this run (see "Looked for, could not find"). The public Medallion record therefore rests on figures reported in a book and a journal article, not on audited public filings.
- Class: inference (from absence in this run's retrievals). This is **not** a claim that no such filing exists.

### Q2: Market makers and HFT

**C5.** Virtu's IPO registration statement says: "we had only one losing trading day during the period depicted, a total of 1,238 trading days" (1 Jan 2009 to 31 Dec 2013). It also says: "we have had only one losing trading day since January 1, 2008," and credits this to "our successful real-time risk management strategy."
- Source: https://www.sec.gov/Archives/edgar/data/1592386/000104746914002070/a2218589zs-1.htm
- Author/publisher: Virtu Financial, Inc., Form S-1, SEC EDGAR
- Pub date: 2014-03 (initial S-1, FY2014 filing)
- Source type: regulatory filing
- Accessed: 2026-09-25, fetched
- Confidence: high that the filing says this. The day count is self-reported, and it sits in a filing that carries securities-law liability.
- Class: self-reported

**C6.** Virtu's revenue and net income: 2013 total revenues $664.5M and net income $182.2M; 2012 $615.6M and $87.6M; 2011 $461.2M and $89.3M. Virtu had 151 employees at 31 Dec 2013.
- Source, publisher, date and type: same as C5 (regulatory filing; the financial statements in an S-1 are audited)
- Accessed: 2026-09-25, fetched
- Confidence: high
- Class: documented-fact

**C7.** Virtu describes its revenue source as "buying and selling large volumes of securities and other financial instruments and earning small amounts of money based on the difference between what buyers are willing to pay and what sellers are willing to accept... 'bid/ask spreads.'" The fetch did not surface per-trade profitability or a percentage of profitable trades.
- Source, publisher, date and type: same as C5 (regulatory filing)
- Accessed: 2026-09-25, fetched
- Confidence: high on the quote. I cannot confirm the absence of per-trade figures, because the fetch may have truncated this very long document.
- Class: self-reported

**C8.** Differences in *relative* latency explain large differences in HFT firms' trading performance. Firms that improved their latency rank through colocation upgrades saw better performance, both through a short-lived information channel and a risk-management channel. Speed helps both market making and cross-market arbitrage.
- Source: https://econpapers.repec.org/RePEc:cup:jfinqa:v:54:y:2019:i:03:p:993-1024_00 (working-paper PDF: https://www.cb.cityu.edu.hk/ef/doc/GRU/HFT%202017/Brogaard_HFT_risk_return_20170825.pdf)
- Authors: Matthew Baron, Jonathan Brogaard, Björn Hagströmer, Andrei Kirilenko, "Risk and Return in High-Frequency Trading", *JFQA* 54(3):993–1024
- Pub date: 2019
- Source type: peer-reviewed
- Accessed: 2026-09-25, fetched (abstract via EconPapers; full text via the working-paper PDF)
- Confidence: high
- Class: documented-fact

**C9.** HFT revenue and volume are concentrated in a few firms. That concentration was "high and non-declining over the five year sample, despite new HFT firm entry and a decline in overall HFT latency". New entrants "are typically slower, earn lower trading revenues, and are more likely to exit." Data: every trade in the 25 largest Swedish stocks, all venues, Jan 2010 to Dec 2014, from Finansinspektionen. An earlier version of the paper found similar results in E-mini S&P 500 futures over 2010–2012.
- Source, authors, date and type: same as C8 (peer-reviewed; quotes taken from the working-paper version, 2017-08-25)
- Accessed: 2026-09-25, fetched
- Confidence: high for the working-paper text. The published version may be worded differently.
- Class: documented-fact

**C10.** Performance is heavily right-skewed across HFT firms. The median HFT earned 6,990 SEK a day in revenue, with an annualized Sharpe of 1.61 and a four-factor alpha of 9%. A firm at the 90th percentile earned 61,354 SEK a day, with a Sharpe of 11.1 and an alpha of 89%.
- Source, authors, date and type: same as C8 (peer-reviewed; working-paper version)
- Accessed: 2026-09-25, fetched
- Confidence: high
- Class: documented-fact

**C11.** Latency-arbitrage races on the London Stock Exchange:
- Frequency: about one race per minute per FTSE 100 symbol.
- Speed: the modal race lasts 5–10 microseconds.
- Volume: races make up about 20% of trading volume.
- Concentration: the top 6 firms take more than 80% of race wins and losses.
- Price impact: races account for about one-third of price impact and effective spread.
- Cost: they act as a tax of about 0.5 bp on trading. Market designs that remove latency arbitrage would cut the cost of liquidity by 17%.
- Scale: about $5 billion a year in global equity markets alone.
- Source: https://www.nber.org/papers/w29011 (QJE: https://academic.oup.com/qje/article/137/1/493/6368348)
- Authors: Matteo Aquilina, Eric Budish, Peter O'Neill, "Quantifying the High-Frequency Trading 'Arms Race'", *QJE* 137(1):493–564
- Pub date: 2022 (NBER WP July 2021)
- Source type: peer-reviewed
- Accessed: 2026-09-25, fetched (NBER abstract page). The OUP page returned only navigation, but a separate search-summary gave the same figures.
- Confidence: high. The race frequency, concentration and dollar size are author measurements and estimates.
- Class: documented-fact (frequency, volume share, concentration) plus author-estimate (0.5 bp tax, 17%, $5B/yr)

### Q3: Citadel Securities, Jane Street, Jump

**C12.** Jane Street has raised money through bond offerings, with cumulative issuance of about US$5.4B since Jan 2024. It launched a deal on 23 Apr 2025 with net proceeds of about US$1.34B. Press figures drawn from the offering disclosures:
- 2024 net trading revenue of about US$20.5B, roughly double the prior year.
- Members' equity of US$29.9B, up 40% year on year.
- About US$6.9M of net trading revenue per employee.
- Q2 2025 net trading revenue of US$10.1B and net profit of US$6.9B.
- Net trading revenue of more than US$24B in the first nine months of 2025.
- Source: https://www.fi-desk.com/jane-street-issues-us1-35-billion-bond-as-market-makers-bulk-up-balance-sheets/ (plus press aggregations surfaced in the same search)
- Publisher: The DESK (fi-desk.com) and others
- Pub date: 2025
- Source type: press, reporting figures from a non-public offering memorandum
- Accessed: 2026-09-25, search-summary. The search engine blended several press sources.
- Confidence: low–medium. Only one publisher lineage, not fetched, and the underlying offering memo is not public.
- Class: reported-figure

**C13.** A regulatory filing exists: Jane Street Capital, LLC, Form X-17A-5 for FY2024, the broker-dealer's annual audited report, on SEC EDGAR at https://www.sec.gov/Archives/edgar/data/1103083/000110308325000002/jscpublic2024.pdf.
- Source type: regulatory filing
- Accessed: 2026-09-25, URL seen in search results only, not fetched
- Confidence: high that it exists. Contents unverified. The public part of an X-17A-5 is usually only a statement of financial condition, not the income statement.
- Class: documented-fact (existence only)

**C14.** This run retrieved no evidence on Citadel Securities or Jump financials. The only source-of-edge evidence for this group is Virtu's own filing (C7: bid/ask spread capture) and the academic papers on speed and latency (C8–C11). Whether payment for order flow or inventory and risk capacity drives these firms' edges went unevidenced this run.
- Class: inference (records the gap)

### Q4: Other edge types, and how they decay

**C15.** A simple distance-based pairs-trading rule on daily US data from 1962 to 2002 produced average annualized excess returns of up to 11% for self-financing portfolios of pairs, and profits typically exceeded conservative estimates of transaction costs.
- Source: https://academic.oup.com/rfs/article-abstract/19/3/797/1646694 (NBER: https://www.nber.org/papers/w7032)
- Authors: Evan Gatev, William N. Goetzmann, K. Geert Rouwenhorst, "Pairs Trading: Performance of a Relative-Value Arbitrage Rule", *RFS* 19(3):797–827
- Pub date: 2006
- Source type: peer-reviewed
- Accessed: 2026-09-25, search-summary
- Confidence: medium–high. This is the standard abstract wording, but the page was not fetched.
- Class: documented-fact (backtest)
- Note: the brief's premise that these returns declined over time was **not** evidenced in retrieved text. See Leads.

**C16.** The S&P 500 index-inclusion effect has largely disappeared:
- Additions: the abnormal return on being added fell from an average of 7.4% in the 1990s to under 1% in the past decade, even though far more assets now track the index.
- Deletions: deletions show the same pattern, averaging only 0.1% over 2010–2020.
- Proposed drivers: more changes are migrations from the S&P MidCap index, and changes have become more predictable. That lets arbitrageurs front-run index demand, buying additions and selling deletions before the announcement, so prices move before the official announcement.
- Source: https://onlinelibrary.wiley.com/doi/10.1111/jofi.13410 (NBER: https://www.nber.org/system/files/working_papers/w30748/w30748.pdf)
- Authors: Robin Greenwood, Marco Sammon, "The Disappearing Index Effect", *Journal of Finance*
- Pub date: 2025 (NBER WP w30748, 2022)
- Source type: peer-reviewed
- Accessed: 2026-09-25, search-summary
- Confidence: medium–high
- Class: documented-fact (magnitudes) plus author-interpretation (drivers)

**C17.** No retrieved evidence this run on insider-trading returns (for example Cohen, Malloy & Pomorski) or on forced-seller and fire-sale flow edges. See "Looked for, could not find".
- Class: inference (records the gap)

### Q5: What the verified edges have in common (synthesis)

**C18.** Every edge evidenced this run whose *holder* is identifiable (Virtu, the fastest HFTs in Baron et al., the top-6 latency-arbitrage firms in ABO) is tied to:
- relative speed, meaning latency rank and colocation (C8, C11);
- very large volume at a small margin per unit: Virtu's bid/ask spread, the 0.5 bp latency tax (C5–C7, C11);
- concentration that does not erode with entry, where slower entrants earn less and leave (C9, C11).
- Class: inference, drawn from C5–C11

**C19.** The edges evidenced for *strategies* rather than firms (pairs trading and index inclusion) were documented in the literature. The one retrieved with a time series, the index effect, was competed away by front-running arbitrageurs, falling from 7.4% to under 1% (C16).
- Class: inference, drawn from C15–C16

**C20.** Medallion is the single documented exception that is neither a speed rent nor a structural flow. Its return record rests on book and journal reporting, not on public audited filings. It is closed to outsiders, so its edge cannot be bought or studied (C1–C4).
- Class: inference. The capacity limit and closure to outsiders are **not evidenced in retrieved text**; see Leads.

**C21.** On whether a small retail or automated trader without colocation or large capital can reach any of these edges:
- Nothing retrieved this run shows it. The mechanisms that are evidenced (relative latency rank, colocation, microsecond races won by 6 firms) exclude such a trader by construction.
- The only structural edge in the retrieved set with a time series (the index effect) decayed to under 1%.
- This is an inference from the absence of evidence, not a proof that no such edge exists.
- Class: inference

---

## Leads (not verified this run; follow up)

- **Cornell (2020) full text.** Get the net-of-fees figure (commonly cited as about 39%), the fee structure (commonly cited as 5% management and 44% performance), and confirmation that Zuckerman's *The Man Who Solved the Market* (2019, book) is the data source. PDF lead: https://www.researchgate.net/publication/338592203_Medallion_Fund_The_Ultimate_Counterexample. Redirected page to fetch: https://pm-research.com/content/iijpormgmt/46/4/156.
- **arXiv 2405.10917**, Shuxin Guo, "Is the annualized compounded return of Medallion over 35%?". A preprint that re-examines Medallion's return figures; not peer-reviewed. https://arxiv.org/pdf/2405.10917
- **Medallion filings.** Renaissance's tax dispute: the 2014 US Senate Permanent Subcommittee on Investigations report on basket options (Deutsche Bank and Barclays) and the IRS settlement reported in 2021. These are candidate regulatory or government sources for trading volume and holding periods. Not searched this run.
- **Medallion capacity and closure.** The claims that the fund is capped at about $10B and that outside investors were expelled around 2005 come from press and book accounts. Not retrieved this run.
- **Jane Street FY2024 X-17A-5** on EDGAR (C13). Fetch it to see what is actually public. Also the Bloomberg piece of 2024-04-17, "Jane Street Scores $10.6 Billion Trading Haul" (press, title only), and a Yahoo/press piece titled "Jane Street Pays $9.38 Billion As Trading Revenue Hits $39.6 Billion" (press, title only, period unverified).
- **Citadel Securities** bond or offering disclosures and its X-17A-5 on EDGAR. Not searched.
- **Decay of pairs-trading returns.** Do & Faff (2010), "Does Simple Pairs Trading Still Work?", *Financial Analysts Journal*, is the usual source for the decline after 2002. Not retrieved.
- **Insider-trading returns.** Cohen, Malloy & Pomorski (2012), "Decoding Inside Information", *Journal of Finance*, on opportunistic versus routine insider trades. Not retrieved.
- **Budish, Cramton & Shim (2015)**, "The High-Frequency Trading Arms Race", *QJE*. Cited inside Baron et al. as the theory behind persistent concentration of latency rents. Not retrieved directly.
- **Virtu 10-K filings.** Later years, for adjusted net trading income per day and the losing-day counts after the IPO. Not retrieved.
- A Substack post, "Market Making Alpha: How Virtu Won 1,237 Days" (press/blog), is secondary only. Do not cite it.

## Looked for, could not find (this run)

- **Cornell (2020) primary text.** SSRN returned HTTP 403 and pm-research returned a 301 redirect that the budget left unfetched. The net-return figure and statements on capacity and closure are unconfirmed.
- **Aquilina, Budish & O'Neill on OUP.** The QJE page returned only a navigation shell. The figures come from the NBER abstract page instead.
- **Virtu per-trade profitability.** The S-1 fetch surfaced no average revenue per trade and no percentage of profitable trades. The fetch may have truncated a very long document.
- **Not searched within the ≤18-call budget:** Citadel Securities and Jump financials; regulatory, court or tax filings for Renaissance; the insider-trading return literature; the forced-seller and fire-sale literature; the decay of the Gatev–Goetzmann–Rouwenhorst pairs returns. These are gaps, **not** negative findings.
