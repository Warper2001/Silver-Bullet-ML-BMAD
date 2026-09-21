# What the 2025 commodity bars establish

Assessment date: 2026-09-07. Population: the completed 2025 TradeStation development pilot only. Decision remains **HOLD-DATA**.

The feed supports a broad, consistently populated multi-contract panel under minimal reported-activity screens. Missing deferred observations and zero-volume contracts exist, but they do not eliminate all candidate pairs on any observed root/timestamp. This supports continuing the data-engineering work; it does not establish a profitable or executable carry strategy.

## Coverage survives the screens

The panel contains 214 contracts, 40,375 bars and 3,007 root/timestamp observations. A candidate pair shares an exact broker timestamp and has broker expirations 90–180 calendar days apart. Observations after broker-expiry date are excluded; there were none.

| Diagnostic | Passing root/timestamp observations |
|---|---:|
| At least one candidate pair | 3,007 / 3,007 |
| At least one pair with positive reported volume and open interest in both legs | 3,007 / 3,007 |
| Same, excluding legs at their third or later consecutive identical close | 3,007 / 3,007 |
| Fixed earliest-expiry pair, positive volume and open interest in both legs | 3,006 / 3,007 |
| Same fixed pair, also applying the repeated-close sensitivity screen | 3,006 / 3,007 |

The fixed diagnostic chooses the earliest-expiring observed contract and its earliest qualifying deferred maturity before applying activity filters. It does not substitute another pair after failure. It omits the protocol's first-notice/last-trade safety buffer because the required calendar evidence is unavailable. It is not an eligible strategy portfolio.

The only fixed-pair activity failure is **ZL on 2025-05-14 at 18:20 UTC**: `BOK25` reported zero volume and open interest of 327, while `BOQ25` reported volume of 16,949 and open interest of 53,496. The nearby broker-expiry date was May 14. Another candidate pair passed the activity screens. This illustrates why expiry/notice safety still matters despite broad pair availability; it does not establish that the protocol would have selected that contract.

Every month's last-observation snapshot has all twelve roots passing even the fixed-pair activity check. These snapshots use different sector timestamps and lack publication-time evidence; they cannot be interpreted as causal rebalance decisions.

## Where the weaknesses are

- **Reported activity:** 1,844 rows have zero volume, and 250 have zero open interest. Copper accounts for 695 zero-volume rows and RBOB gasoline for 514—together about 66% of zero-volume rows. For example, `HGX26` has 161 zero-volume rows out of 251 observed rows. Activity in another pair does not make these contracts executable.
- **Internal missingness:** 52 contract observations are absent within a contract's first-to-last observed span, measured against its root's observed timestamp union. Lean hogs accounts for 18, including six in `LHZ26`; RBOB has eight. January 17 has the largest concentration, with nine missing contract observations across the panel. No root loses every candidate pair as a result.
- **Whole-root absence:** January 9 is the only observed UTC date represented by fewer than twelve roots: all five grain/oilseed roots are absent. They have 250 observations each; the other roots have 251. An exchange calendar is needed to distinguish scheduled closures from missing data. The internal-missingness metric cannot detect an absence shared by every contract in a root.
- **Repeated closes:** Four rows reach a three-observation identical-close run: `CK26` and `CN26` on January 15, `LHK26` on May 29, and `LHV26` on August 21. No longer runs occur under this definition. Missing root-observed timestamps reset the run. Equal prices alone do not prove stale or fabricated data.
- **Contract transitions:** The fixed diagnostic pair changes 5–12 times per root across the year. Exact before/after contract identities are in `transitions.csv`. These are availability-based pair changes, not executed rolls or a turnover estimate.

## What to do with this result

The next useful question is whether the recorded prices and contract dates can support the protocol's settlement and delivery-safety requirements. Another broad 2025 download is not needed to demonstrate minimal activity breadth. Prioritize API-specific settlement lineage, publication/revision evidence and historical first-notice/last-trade calendars. Once those exist, compare the actual calendar-safe pair's coverage against this descriptive baseline.

Do not screen roots by returns, promote these diagnostic pairs into strategy targets, or infer fillability from positive daily volume. The analysis used no returns, carry rankings, P&L, new subscriptions or sealed-holdout access. The research protocol and source artifacts were not changed.

## Evidence and verification

- [Full tables and definitions](reports/commodity-curve-feasibility-2025/report.md)
- [Machine-readable summary and parent hashes](reports/commodity-curve-feasibility-2025/summary.json)
- [Every root/timestamp and fixed pair](reports/commodity-curve-feasibility-2025/daily.csv)
- [Contract-level quality concentrations](reports/commodity-curve-feasibility-2025/contracts.csv)
- [Contract-pair transitions](reports/commodity-curve-feasibility-2025/transitions.csv)
- [Reproduction script](reports/commodity-curve-feasibility-2025/analyze.py)

Offline source verification passed for 3,059 request artifacts and six source reports. An independent pandas self-join reproduced every per-timestamp candidate-pair and activity-pair count. A separate full-root-clock calculation reproduced all contract-level internal-gap and repeated-close-run counts. The output bundle records source SHA256 hashes and the analyzer hash.
