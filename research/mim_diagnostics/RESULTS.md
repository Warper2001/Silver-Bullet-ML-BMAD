# MIM-NB payoff attribution and data feasibility

The unchanged baseline reconciles to **1,323 eligible sessions, 801 trades and $21,889.76 net**. The completed study explains existing payoff and input feasibility; it tests no new strategy or filter and changes no trading behavior.

Primary accounting uses recorded delay-2 fills (the open one full minute after the decision), one MNQ at $2 per point and $2.24 roundtrip cost. Gross payoff is $23,684.00, costs are $1,794.24, and all 544 flat sessions remain in the grid. Every selected contract and source exclusion is preserved.

## Where the payoff comes from

| Original exit label | Trades | Net contribution | Losing-trade dollars | Share of losing dollars |
|---|---:|---:|---:|---:|
| Catastrophe stop | 71 | −$35,331.04 | $35,331.04 | 46.1% |
| EOD close proxy | 723 | $59,824.48 | $38,691.64 | 50.5% |
| Reversal | 7 | −$2,603.68 | $2,603.68 | 3.4% |

EOD contains both winners and losers. The ledger identifies seven true reversals and eleven opposite-direction entries after becoming flat; these are different transitions, and original labels remain available.

Longs contributed $13,589.86 across 411 trades; shorts contributed $8,299.90 across 390. First entries contributed $17,537.04 across 779 trades; the 22 subsequent entries contributed $4,352.72. These exposed historical differences do not justify a directional or re-entry restriction.

Entry time and profit accumulation differ substantially. Trades entering in the 10:00–10:30 bucket contributed $7,363.22 over their full lives, while actual net accrual in that half-hour was −$467.50. Net accrual in 15:00–15:30 was $9,699.24, followed by −$2,552.98 in 15:30–16:00. All thirteen cash-session buckets are shown; no window was selected as a rule.

The previously disclosed best 67 sessions earned $39,552.92, while the rest lost $17,663.16. This is hindsight concentration, not a way to identify future winners.

## Sampled risk

Across 515,970 eligible minute marks, minute-close maximum drawdown was **$3,559.40**, versus $2,437.26 at daily closes. Exposure is bounded by 222,077–222,148 contract-minutes because 71 stops have uncertain intraminute execution times. Excursions separately report close/fill samples and fully held minute ranges, excluding post-exit prices. These measurements do not bound exact intrabar risk or establish capital requirements. Drawdown depth, duration and recovery details are in the report and summary JSON.

## Input feasibility

At **2026-09-12T19:44:22Z**, the inventory found 47 economic-calendar rows, 69 retrospective policy-shock dates, 12 policy windows and 11 forward FOMC dates. The economic calendar contains incorrect dates and lacks original availability provenance. Policy labels are unavailable as causal inputs without contemporaneous evidence. Forward FOMC dates have current official support but require original schedule vintages and revision evidence for historical use.

A recorded scan of 325 CSV headers and 19 parquet filenames found no local Nasdaq options-positioning or leveraged-ETF AUM/leverage archive. Their status is unavailable locally. Price/volume proxies and SPX inventory do not establish Nasdaq closing demand. No dataset was acquired.

The prospective MIM journal contains zero eligible sessions; the FOMC ledger has zero stored events, with its first scheduled event September 16. Collection coverage is reported without efficacy analysis. The report's explicit erratum acknowledges the rejected policy-shock throttle, failed impulse-following study and separately sealed FOMC-fade study, preserving their verdicts and protocols. Future benchmark and mechanism proposals remain specifications with prerequisites and falsifiers.

## Reproduction and verification

- [Standalone HTML report and charts](runs/20260912T194421-run-3346059bd9/report.html), [Markdown report](runs/20260912T194421-run-3346059bd9/report.md), [summary JSON](runs/20260912T194421-run-3346059bd9/summary.json), [feasibility inventory](runs/20260912T194421-run-3346059bd9/feasibility.json).
- [Final input audit](runs/20260912T194418-audit-3a00001098/audit.json) and [run manifest](runs/20260912T194421-run-3346059bd9/manifest.json) bind approved source/data hashes, code and dependency snapshots, definitions, versions and evidence.
- **350 tests passed**: 127 diagnostics tests and 223 existing comparison, lifecycle and robustness tests. All three review passes completed; all actionable findings were resolved, with none deferred. Python undefined-symbol/unused-import checks passed.
- The final run independently verified every minute/trade/event accounting field, exact attribution rows, excursions, scenarios and summary against recorded fills and bars. The eight analytical CSV/JSON outputs were byte-identical to the preceding completed run. Six embedded SVGs and seven HTML tables passed structure checks; the HTML needs no external scripts or styles.
- [Separate CLI verification](runs/20260912-independent-checks/final-verify.log) passed after sealing; [combined test log](runs/20260912-independent-checks/final-tests-packaged.log) records all 350 passing tests.
- Accounting tolerance is $1e-8 with zero relative tolerance. Completion SHA256: `c11063107b175c7ce90250c05b4eae9cd3e5bc766866b6fa3229c157d7e1564e`.

Source and this summary are versioned on the isolated research branch. Large sealed artifacts remain local under ignored `runs/`. See [README](README.md) for audit/run/verify commands.
