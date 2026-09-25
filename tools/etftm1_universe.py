"""ETFTM-1 candidate universe — fixed by rule, never by returns (plan A0.1, 2026-09-25).

Rule: 1–2 flagship US-listed ETFs per asset class, preferring the longest history and deepest
liquidity; the list was written before any ETF price data was fetched for this project.
Excluded by rule (never candidates): leveraged / inverse ETFs, and products with severe roll decay
(e.g. USO, UNG). USO is probed ONLY as a reverse-split test case for the data audit.
Trading-ticker substitutions (e.g. SPY -> SPLG, DBC -> PDBC) are decided in A2, not here.
"""
UNIVERSE: dict[str, list[str]] = {
    "us_equity": ["SPY", "QQQ", "IWM", "MDY"],
    "intl_equity": ["EFA", "EEM", "EWJ"],
    "sectors": ["XLE", "XLK", "XLF", "XLU", "XLV", "XLP", "XLI", "XLY", "XLB"],
    "treasuries": ["SHY", "IEF", "TLT"],
    "credit_inflation": ["LQD", "HYG", "TIP", "EMB"],
    "commodities": ["GLD", "SLV", "DBC", "DBA"],
    "currency": ["UUP", "FXE", "FXY"],
    "real_estate": ["VNQ"],
}
TRADING_ALTERNATES = ["PDBC"]          # probed for history only; substitution decided in A2
AUDIT_ONLY = ["USO"]                   # reverse split 2020-04-28: split-adjustment test
RISK_FREE_YAHOO = "^IRX"               # 13-week T-bill yield (FRED unreachable from this host)

ALL_SYMBOLS = [s for syms in UNIVERSE.values() for s in syms]
ASSET_CLASS = {s: c for c, syms in UNIVERSE.items() for s in syms}
