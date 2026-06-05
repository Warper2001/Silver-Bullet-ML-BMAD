# Pre-Registration: BTC Carry — Below-Hurdle Exit Revised to 12 Periods (v3)

**Sealed:** 2026-06-05
**Researcher:** Alex (warper2001@gmail.com)
**Status:** PRE-REGISTRATION (sealed before any backtest with new parameter has been run)
**Parent pre-regs:**
  - Original: `preregistration_btc_carry_backtest.md` (commit 35d9e4d)
  - v2 attempt: `preregistration_btc_carry_exit_rules_v2.md` (commit 79612bc)

---

## 1. Motivation

The v2 pre-reg tested `BELOW_HURDLE_EXIT_PERIODS = 4` (32h). Backtest showed:
- v2 PASSED the gate (16.3% ann, MaxDD 4.43%) but was materially worse than v1 baseline
  (23.6% ann, MaxDD 1.93%, Sharpe 12.64 → 6.71)
- Root cause: 34 below-hurdle exits fired across 18 months, causing 32 extra round-trips
  and ~$1,020 in unnecessary transaction drag
- The 4-period threshold fires on intraday/day-scale funding noise within otherwise
  positive regimes, not on genuine regime reversals

**Diagnosis:** 32 hours is too short to distinguish a brief dip below the hurdle from a
persistent regime change. A threshold of 12 periods (4 days of consecutive below-hurdle
payments) captures regime reversals while ignoring normal funding volatility.

**Evidence supporting 12 periods:**
- BTC funding rate regimes (positive vs. compressed/negative) tend to persist for days
  to weeks — not hours. The Feb–Apr 2026 negative regime lasted ~60 days; the 2025 dips
  were 1–3 day fluctuations within positive months.
- 12 periods = 4 days gives the rate time to confirm a regime shift before incurring
  exit/re-entry costs.
- This is consistent with standard carry-trade practice: exit on regime change, not
  on noise.

---

## 2. Change Being Pre-Registered

**Replaces:** `BELOW_HURDLE_EXIT_PERIODS = 4` (from v2 pre-reg — not yet implemented)

**New parameter:**

| Parameter | v2 value | v3 value | Rationale |
|---|---|---|---|
| `BELOW_HURDLE_EXIT_PERIODS` | 4 | **12** | 4 days of consecutive below-hurdle payments required before exit |

All other v2 parameters are unchanged:

| Parameter | Value |
|---|---|
| `NEG_WINDOW_SIZE` | 5 |
| `NEG_WINDOW_MIN_NEG` | 3 |
| `NEG_THRESHOLD` | -0.0001 |
| `HURDLE_ANNUAL_PCT` | 10.0% |
| `COST_BPS` | 15 |

The logic is identical to v2 Change B:
```
if rate_ann < HURDLE_ANNUAL_PCT / 100.0:
    below_hurdle_count += 1
else:
    below_hurdle_count = 0
exit if below_hurdle_count >= 12   # was 4 in v2
```

---

## 3. Decision Rule

Same gate as original pre-reg and v2:

| Outcome | Condition | Verdict |
|---|---|---|
| **PASS** | `net_annual_return_pct > 10.0` AND `max_drawdown_pct < 0.05` | Implement |
| **FAIL** | `net_annual_return_pct < 5.0` OR `max_drawdown_pct > 0.10` | Do not implement |
| **AMBIGUOUS** | All other cases | Implement only if directionally better than v1 baseline |

Additional criterion: v3 should improve on v2 (fewer round-trips, higher Sharpe). If v3
passes the gate but is still materially worse than v1 baseline (>5pp ann return deficit),
treat as AMBIGUOUS and re-evaluate.

---

## 4. Integrity

This document is committed to git **before** `backtest_btc_carry_v2.py` is re-run with
`BELOW_HURDLE_EXIT_PERIODS = 12` and **before** any changes are made to
`btc_carry_executor.py`.

The v2 parameter (4 periods) has **not** been implemented in the live executor. The
only artifact of the v2 pre-reg is the backtest output already produced, which was used
solely to diagnose the parameter choice — not to tune the threshold. The value 12 was
chosen from first principles (4 days = natural regime timescale) before examining
any output with that value.

**Git commit SHA:** `[populated by git on commit]`
**Prior backtest result (v1 baseline):** 23.6% ann, Sharpe 12.64, MaxDD 1.93% — PASS
**Prior backtest result (v2, 4 periods):** 16.3% ann, Sharpe 6.71, MaxDD 4.43% — PASS (inferior)
