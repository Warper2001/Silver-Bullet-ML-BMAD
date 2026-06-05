# Pre-Registration: BTC Carry — Sliding-Window Neg Exit + Below-Hurdle Exit (v2)

**Sealed:** 2026-06-05
**Researcher:** Alex (warper2001@gmail.com)
**Status:** PRE-REGISTRATION (sealed before any backtest with new parameters has been run)
**Parent strategy:** BTC-CARRY (pre-reg: `preregistration_btc_carry_backtest.md`, commit 35d9e4d)

---

## 1. Motivation

Two live paper trades under the original exit rules both produced losses:

| Trade | Entry rate (ann) | Payments | Gross | Net P&L |
|---|---|---|---|---|
| Trade 1 | 15.68% | 3 (all negative, hit NEG_STOP_PERIODS=3) | -$27.66 | -$42.66 |
| Trade 2 | 28.77% | 4+ (ongoing, accrued -$33.44) | -$33.44 | pending |

Two failure modes were identified **before inspecting any backtest output with the proposed
new parameters**:

**Failure A — neg_count reset by noise payments:**
The consecutive-3 exit was gamed by near-zero payments resetting the counter. Trade 2
Payment #3 (+$0.02) reset neg_count from 1 to 0, allowing two more negative payments before
any exit trigger. A sliding window over the last N payments is more robust.

**Failure B — no exit when rate falls and stays below the entry hurdle:**
The code holds an ACTIVE position indefinitely even when `rate_ann < HURDLE_ANNUAL_PCT`.
Trade 2 is currently sitting with `funding=-5.89%ann, hurdle_met=False` with no exit
mechanism. If the rate is below the hurdle at each payment, we are collecting sub-hurdle
(or negative) carry with no trigger to exit.

---

## 2. Changes Being Pre-Registered

### Change A: Sliding-Window Negative Exit

**Replaces:** `NEG_STOP_PERIODS = 3` (consecutive)

**New parameters:**

| Parameter | Value | Rationale |
|---|---|---|
| `NEG_WINDOW_SIZE` | **5** | Rolling window of last 5 payments (40h lookback) |
| `NEG_WINDOW_MIN_NEG` | **3** | Exit if ≥3 of the last 5 payments are below NEG_THRESHOLD |
| `NEG_THRESHOLD` | **-0.0001** | Unchanged from original pre-reg |

**Logic change:**
```
# Old logic (consecutive):
if rate_8h < NEG_THRESHOLD: neg_count += 1
else: neg_count = 0
exit if neg_count >= 3

# New logic (sliding window):
payment_history.append(rate_8h)        # keep only last NEG_WINDOW_SIZE entries
neg_in_window = sum(r < NEG_THRESHOLD for r in payment_history[-NEG_WINDOW_SIZE:])
exit if neg_in_window >= NEG_WINDOW_MIN_NEG
```

State change: replace `neg_count: int` with `payment_history: list[float]` in `CarryState`.
For backward compatibility with the existing state file, `neg_count` is retained as a
derived display field (count of negatives in current window).

### Change B: Below-Hurdle Consecutive Exit

**New parameter:**

| Parameter | Value | Rationale |
|---|---|---|
| `BELOW_HURDLE_EXIT_PERIODS` | **4** | Exit if the rate at each of the last 4 consecutive payment events was below the entry hurdle |

**Logic change (evaluated at each payment, after applying P&L):**
```
if rate_ann < HURDLE_ANNUAL_PCT / 100.0:
    below_hurdle_count += 1
else:
    below_hurdle_count = 0
exit if below_hurdle_count >= BELOW_HURDLE_EXIT_PERIODS
```

This catches the case where the carry rate has persistently reversed below the entry
threshold for 32h (4 × 8h periods) without requiring payments to cross the negative
threshold. The below-hurdle exit fires independently of the sliding-window neg exit;
whichever condition is met first triggers exit.

State change: add `below_hurdle_count: int = 0` to `CarryState`.

---

## 3. Unchanged Parameters

All original sealed parameters from `preregistration_btc_carry_backtest.md` are
unchanged:

| Parameter | Value |
|---|---|
| `HURDLE_ANNUAL_PCT` | 10.0% |
| `COST_BPS` | 15 |
| `NEG_THRESHOLD` | -0.0001 |
| `DEFAULT_NOTIONAL` | $10,000 |

---

## 4. Backtest Methodology

**Script:** `backtest_btc_carry.py` — re-run with new exit logic applied to the same
historical data (`data/kraken/PF_XBTUSD_funding_rate.csv`, Nov 2024 → present).

**Comparison:** Results reported side-by-side vs. the original (consecutive) exit rules.
The new rules should show:
- Fewer or equal completed trades (less churn from bad re-entries after noise resets)
- Equal or lower max drawdown per trade
- Equal or better net annualised return

**Decision rule (same gate as original):**

| Outcome | Condition | Verdict |
|---|---|---|
| **PASS** | `net_annual_return_pct > 10.0` AND `max_drawdown_pct < 0.05` | Implement in executor |
| **FAIL** | `net_annual_return_pct < 5.0` OR `max_drawdown_pct > 0.10` | Revert to original |
| **AMBIGUOUS** | All other cases | Implement if improvement over baseline is directionally positive |

No new data has been examined with these parameters before this document was committed.
The baseline original-rules backtest result (PASS: 23.6% ann, Sharpe 12.64, MaxDD 1.93%)
is the sole prior result already known.

---

## 5. Scope

This pre-registration covers **exit rule changes only**. Entry conditions (`HURDLE_ANNUAL_PCT`,
single-tick entry trigger) are not modified here and remain as sealed in the original pre-reg.
Any future entry-side changes require a separate pre-registration.

---

## 6. Integrity

This document is committed to git **before** `backtest_btc_carry.py` is re-run with the
new parameters and **before** any changes are made to `btc_carry_executor.py`.

**Git commit SHA:** `[populated by git on commit]`
**Referenced prior pre-reg:** `_bmad-output/preregistration_btc_carry_backtest.md` (35d9e4d)
