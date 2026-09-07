# Pre-registration — COT positioning (E-mini Nasdaq-100)

**Date:** 2026-09-06
**Chosen as the next strategy to test** over the Option 6 Monday near-miss (circular — the hypothesis-generating data is the only data available) and volatility risk premium (data-blocked, needs acquisition + non-combine capital).
**Sealed before any return is computed.**

## Data — verified reachable before sealing

- **Signal:** CFTC Commitments of Traders, futures-only, contract `209742` "NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE" (the E-mini NQ — where institutional positioning actually sits; the micro contract `209747` is too small to be informative). Socrata endpoint `publicreporting.cftc.gov/resource/6dca-aqww.json`. **Verified: 1,420 weekly reports, 1999-06-22 → 2026-09-01.**
- **Price:** MNQ 1-min bars → weekly closes. `mnq_1min_2021_2024_frontmonth.csv` + `mnq_1min_2025.csv` + `mnq_1min_2026_ytd.csv`.
- **Binding constraint is price, not COT:** overlap is **2021-01 → 2026-06, ≈280 weekly observations.** Comparable to the 5 dev years used by TSC-1 and TSMOM-1.

## The look-ahead trap this seal exists to avoid

COT reports carry a **Tuesday** as-of date but are **not published until the following Friday ~15:30 ET.** Aligning a Tuesday-dated signal to that same week's returns is look-ahead bias, and it is the standard way COT studies manufacture false edges.

**Alignment rule, fixed here:** a report dated Tuesday *T* is treated as knowable only from the **following Friday's close**. Forward return is measured from the **Monday after publication** to the Monday after that. Signal is therefore lagged ~10 calendar days behind its as-of date, which is the honest tradeable lag.

## Hypothesis

**Contrarian positioning.** Large speculators (non-commercials) are conventionally the crowded side at extremes; their net length is expected to be **negatively** related to forward returns.

- **Signal:** net non-commercial position = `noncomm_positions_long_all − noncomm_positions_short_all`, z-scored over a trailing **104-week** window (2 years; fixed here, not tuned).
- **Pre-declared sign: NEGATIVE.** A positive correlation is a FAIL, not a "reversed edge" to be re-labelled momentum after the fact.

## Test mechanism before optimising a threshold

Deliberately **not** starting with a threshold rule. Thresholds invite mining, and if no monotonic relationship exists then no threshold can help — the same reasoning that made TSMOM-1's baseline check decisive.

**Primary (Gate 0):** Spearman rank correlation between the z-scored signal at week *t* and the forward weekly MNQ return, over the full overlap.

## Gate 0 — sealed

1. **N ≥ 200** aligned weekly observations.
2. **Sign matches the pre-declared negative direction.** Positive ⇒ FAIL, full stop.
3. **Significance:** p < 0.05 on the Spearman correlation. Single pre-declared test, so no multiple-comparison correction is required — and none may be added afterwards to rescue a near-miss.
4. **Economic check:** mean forward return of the most-net-long decile must be below that of the most-net-short decile by more than **$4.00/week** round-turn cost at 1ct MNQ. A statistically real but economically sub-cost relationship is a FAIL — the standing lesson from S26 and the 1-minute graveyard.
5. **Beat the drift:** the long-side leg must beat the unconditional always-long weekly baseline. MNQ rose over 2021-2026; against a zero benchmark any long-biased rule looks profitable. This is the TSMOM-1 mechanism check.

**Verdict: PASS only if 1–5 all hold.**

## Combine-compatibility screen — applied up front, new this session

`result_consistency_rule_sweep_20260906.md` established that a profit shape concentrated in single fat days collides with prop-firm consistency rules, and that the *formulation* of the rule matters ~18× more than its percentage. So if the primary passes, report **best-week profit as a share of total profit** alongside the P&L. A signal whose gains concentrate in one or two weeks is not combine-deployable regardless of its statistics, and it is better to know that before building a strategy on it than after buying an account.

## Stopping rule

One run. FAIL closes COT positioning — no threshold sweep, no lookback re-tuning, no switching to commercials-as-smart-money as a second bite. Any of those is a new seal on new reasoning. A PASS authorises only a separately pre-registered threshold/trading-rule design, not a deployment.
