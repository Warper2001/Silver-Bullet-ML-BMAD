# ETFTM-1 data audit (2026-09-25)

Built by `tools/build_etf_panel.py` (sha `2e68bed2c714`). Data-quality statistics only: no signal or strategy return was computed.

- Development panel: 154,951 rows, 1993-01-29 → 2021-09-30.
- Holdout: 38,781 rows from 2021-10-01, written to `data/sealed_holdout/etf_daily_holdout_20211001_plus.csv` (mode 444).
- Asset classes with ≥ 15 years before the cutoff: 8 (commodities, credit_inflation, currency, intl_equity, real_estate, sectors, treasuries, us_equity).
- Symbols failing a check (monthly TE p95 vs adjclose > 0.2%, or median daily price-return gap vs TradeStation > 0.1%): none.
- **Gate A0: GO**

Total return = Yahoo close + Yahoo dividends (reason in the builder docstring). TE columns compare it with Yahoo adjclose; the TS columns compare daily PRICE returns with TradeStation, an independent source, on non-dividend days.

| Sym | Class | First | Yrs<cutoff | Rows | Dup | ≤0 | >25% days | Unmatched divs | Monthly TE p95 vs adjclose | TS indep. daily diff med / p99 | TS level diff med | Missing in TS | Missing vs SPY cal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SPY | us_equity | 1993-01-29 | 28.7 | 8472 | 0 | 0 | 0 | 0 | 4.65e-05 | 3.4e-08 / 3.5e-03 | 0.0000 | 2 | 0 |
| QQQ | us_equity | 1999-03-10 | 22.6 | 6930 | 0 | 0 | 0 | 0 | 2.47e-05 | 2.7e-05 / 3.9e-03 | 0.0074 | 1 | 0 |
| IWM | us_equity | 2000-05-26 | 21.3 | 6622 | 0 | 0 | 0 | 0 | 5.72e-05 | 3.1e-08 / 1.5e-03 | 0.0000 | 22 | 0 |
| MDY | us_equity | 1995-05-04 | 26.4 | 7901 | 0 | 0 | 0 | 0 | 3.64e-05 | 4.0e-08 / 6.6e-04 | 0.0000 | 949 | 0 |
| EFA | intl_equity | 2001-08-27 | 20.1 | 6307 | 0 | 0 | 0 | 0 | 1.21e-04 | 3.0e-08 / 2.1e-04 | 0.0000 | 1 | 0 |
| EEM | intl_equity | 2003-04-14 | 18.5 | 5901 | 0 | 0 | 0 | 0 | 1.27e-04 | 3.6e-08 / 5.3e-04 | 0.0000 | 4 | 0 |
| EWJ | intl_equity | 1996-03-18 | 25.5 | 7681 | 0 | 0 | 0 | 0 | 7.07e-05 | 2.2e-08 / 5.7e-04 | 0.0000 | 5 | 0 |
| XLE | sectors | 1998-12-22 | 22.8 | 6982 | 0 | 0 | 0 | 0 | 1.45e-04 | 2.1e-04 / 3.0e-03 | 0.0264 | 12 | 0 |
| XLK | sectors | 1998-12-22 | 22.8 | 6982 | 0 | 0 | 0 | 0 | 4.17e-05 | 4.4e-05 / 8.5e-04 | 0.0000 | 34 | 0 |
| XLF | sectors | 1998-12-22 | 22.8 | 6982 | 0 | 0 | 0 | 0 | 8.78e-05 | 6.4e-05 / 8.8e-04 | 0.0003 | 11 | 0 |
| XLU | sectors | 1998-12-22 | 22.8 | 6982 | 0 | 0 | 0 | 0 | 1.04e-04 | 1.2e-04 / 6.7e-04 | 0.0001 | 11 | 0 |
| XLV | sectors | 1998-12-22 | 22.8 | 6982 | 0 | 0 | 0 | 0 | 4.28e-05 | 6.8e-05 / 1.3e-03 | 0.0198 | 12 | 0 |
| XLP | sectors | 1998-12-22 | 22.8 | 6982 | 0 | 0 | 0 | 0 | 7.79e-05 | 2.7e-08 / 4.5e-04 | 0.0000 | 11 | 0 |
| XLI | sectors | 1998-12-22 | 22.8 | 6982 | 0 | 0 | 0 | 0 | 6.69e-05 | 2.7e-08 / 3.5e-04 | 0.0000 | 13 | 0 |
| XLY | sectors | 1998-12-22 | 22.8 | 6982 | 0 | 0 | 0 | 0 | 3.70e-05 | 5.0e-05 / 6.0e-04 | 0.0000 | 11 | 0 |
| XLB | sectors | 1998-12-22 | 22.8 | 6982 | 0 | 0 | 0 | 0 | 8.71e-05 | 1.1e-04 / 8.1e-04 | 0.0001 | 12 | 0 |
| SHY | treasuries | 2002-07-30 | 19.2 | 6079 | 0 | 0 | 0 | 0 | 6.17e-06 | 2.6e-08 / 8.8e-08 | 0.0000 | 1 | 0 |
| IEF | treasuries | 2002-07-30 | 19.2 | 6079 | 0 | 0 | 0 | 0 | 2.85e-05 | 2.3e-08 / 9.3e-05 | 0.0000 | 0 | 0 |
| TLT | treasuries | 2002-07-30 | 19.2 | 6079 | 0 | 0 | 0 | 0 | 6.08e-05 | 2.3e-08 / 7.4e-05 | 0.0000 | 0 | 0 |
| LQD | credit_inflation | 2002-07-30 | 19.2 | 6079 | 0 | 0 | 0 | 0 | 3.74e-05 | 2.0e-08 / 9.6e-08 | 0.0000 | 2 | 0 |
| HYG | credit_inflation | 2007-04-11 | 14.5 | 4897 | 0 | 0 | 0 | 0 | 8.41e-05 | 2.6e-08 / 8.6e-08 | 0.0000 | 0 | 0 |
| TIP | credit_inflation | 2003-12-05 | 17.8 | 5737 | 0 | 0 | 0 | 0 | 3.49e-05 | 2.0e-08 / 6.8e-08 | 0.0000 | 35 | 0 |
| EMB | credit_inflation | 2007-12-19 | 13.8 | 4721 | 0 | 0 | 0 | 0 | 6.02e-05 | 2.1e-08 / 7.4e-08 | 0.0000 | 1 | 0 |
| GLD | commodities | 2004-11-18 | 16.9 | 5497 | 0 | 0 | 0 | 0 | 0.00e+00 | 2.4e-08 / 1.1e-07 | 0.0000 | 0 | 0 |
| SLV | commodities | 2006-04-28 | 15.4 | 5135 | 0 | 0 | 1 | 0 | 0.00e+00 | 2.8e-08 / 5.6e-04 | 0.0000 | 0 | 0 |
| DBC | commodities | 2006-02-06 | 15.6 | 5192 | 0 | 0 | 0 | 0 | 1.77e-07 | 2.4e-08 / 9.3e-08 | 0.0000 | 0 | 0 |
| DBA | commodities | 2007-01-05 | 14.7 | 4962 | 0 | 0 | 0 | 0 | 1.81e-07 | 2.5e-08 / 9.0e-08 | 0.0000 | 0 | 0 |
| UUP | currency | 2007-03-01 | 14.6 | 4925 | 0 | 0 | 0 | 0 | 2.00e-07 | 2.1e-08 / 7.4e-08 | 0.0000 | 2 | 0 |
| FXE | currency | 2005-12-12 | 15.8 | 5229 | 0 | 0 | 0 | 0 | 1.23e-05 | 2.5e-08 / 9.8e-08 | 0.0000 | 0 | 0 |
| FXY | currency | 2007-02-13 | 14.6 | 4936 | 0 | 0 | 0 | 0 | 0.00e+00 | 2.3e-08 / 8.6e-08 | 0.0000 | 0 | 0 |
| VNQ | real_estate | 2004-09-29 | 17.0 | 5533 | 0 | 0 | 0 | 0 | 1.73e-04 | 3.0e-08 / 1.3e-04 | 0.0000 | 0 | 0 |

Days with a total-return move > 25%: {'SLV': ['2026-01-30']}.

Raw files under `data/etf_daily/raw/` contain the holdout period too. They are inputs to this builder only; analysis code must read `panel_dev.csv` (enforced in `research/etf_trend/data.py`, A3).
