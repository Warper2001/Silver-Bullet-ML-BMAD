# PL (Platinum) Combine-Fit Gate — VERDICT 2026-07-05: ❌ FAIL

**Follows:** `pl_slippage_verdict_20260705.md` (slippage PASS, commit `895d0b5`) — which
explicitly held that a slippage PASS does NOT clear combine-fit, and combine-fit must be
evaluated **before** finalizing any Gate-1 holdout prereg.
**Question:** can the frozen PL structural edge, at its *only tradeable size* (1 full
50 oz platinum contract — no CME micro platinum exists), survive the Topstep 50K combine's
$2,000 trailing Max-Loss-Limit math?
**Method:** replay the frozen 101-trade PL path (`data/reports/backtest_1year_20260626_025416.csv`,
in-sample — NOT a holdout) through the exact Topstep 50K ratchet (floor $48,000; EOD
ratchet `floor = min(50000, max(floor, EOD_bal − 2000))`; bust if equity touches floor),
gross and net of the measured $34/RT slippage. Script: `pl_combine_fit.py` (tmp).

## Result: FAIL on three independent counts

| Check | Gross | Net @ $34/RT | Limit | |
|---|---|---|---|---|
| **Worst single-trade loss** | −$1,880 | −$1,914 | $1,000 daily halt | ❌ one stop exceeds a full day's loss limit |
| **Trailing-MLL simulation** | **BUST 2025-06-30** (−$1,355 through floor) | **BUST 2025-06-30** (−$2,890 through floor) | $2,000 trailing | ❌ account dies |
| **Max drawdown from HWM** | $3,380 | $4,890 | $2,000 | ❌ 1.7–2.4× the buffer |
| Profit target / consistency | $6,265 but biggest day $4,320 > 50% | $2,831 < $3,000 target | $3,000 + <50% | ❌ |

## Why it can't be salvaged by sizing or a daily breaker

- **No smaller size exists.** Platinum has no CME micro contract; 1 full 50 oz ($50/pt) is
  the *minimum* position. You cannot size down to fit the buffer.
- **One stop ≈ the whole buffer.** The worst single SL is −$1,880 to −$1,914 — **94–96% of
  the entire $2,000 MLL** — from a *fresh* account. One bad trade after any prior drawdown
  is an automatic bust. A daily circuit breaker cannot prevent this: the breaker only fires
  *after* a trade closes, so a single −$1,900 stop is realized in full regardless.
- **The bust is a multi-day drawdown, not one bad day.** June 2025 alone was −$2,560
  (month max-DD $2,590); the trailing floor, ratcheted up from the late-May/early-June
  peak, is touched as that drawdown accumulates. A per-day halt slows but does not stop a
  cumulative drawdown from touching a *trailing* floor.
- Net of measured slippage it is worse on every axis, and net profit ($2,831) no longer
  even reaches the $3,000 target.

## Verdict and consequence

**PL is combine-incompatible for the Topstep 50K account — CLOSED for this account type.**
The slippage gate passed, but combine-fit is the binding constraint and it fails structurally,
not marginally. Per the sealed slippage-verdict protocol, this **blocks writing the Gate-1
holdout prereg**: we do NOT spend the PL sealed holdout
(`data/sealed_holdout/pl_1min_holdout_20260301_plus.csv` — remains UNTOUCHED). Running this
gate first saved that one-shot.

**Scope of the close:** this kills PL for the *$2,000-trailing-MLL 50K combine specifically*.
The structural signal (gross PF 1.344) and the measured PLV26 cost ($34/RT, net PF ~1.141)
are unchanged facts; PL could only ever be revisited on an account whose drawdown buffer is
large relative to a −$1,900 single-trade risk (e.g. a much larger funded/personal account) —
not this program's combine. Given copper's Gate-1 holdout already FAILED (`328cdaf`), the
YANK cross-instrument portability track now has **zero live candidates for the combine**.

Nothing deployed; no holdout touched.
