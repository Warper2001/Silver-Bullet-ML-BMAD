# Pre-Registration: MIM-NB Ops Hardening — Half-Day Skip + Stop-Fill Reconcile Loop

**Generated:** 2026-07-07
**Experiment ID:** mim-nb-ops-hardening
**Type:** Operational parity/hardening only — NO signal-logic change, NO parameter
change, NO holdout access. Closes findings #4 and the ops gaps of the 2026-07-07
halt-and-review (`halt_review_mim_nb_parity_20260707.md`); Winston's minimal
invariant set from the same-day roundtable. Alex's go: "fix the half-day skip and
stop-fill reconciliation loop".

**Status:** SEALED on commit; code lands in the immediately following commit(s).

---

## 1. Half-day (early-close) session skip — sealed-engine parity

**Gap:** the sealed engine skips any session without a 16:00 bar (no trades, no sigma
append, prev_close carries over). Live traded marks on 2026-07-03 (early close) and,
worse, `_new_session` would push a half-day's partial `today_moves` into sigma history
the next morning; a position entered on a half-day would sit unmanaged from the 13:00
halt until the 16:01 safety net.

**Change:** `EARLY_CLOSE_DATES` constant (2026 remainder: 09-07 Labor Day, 11-26
Thanksgiving, 11-27 day-after, 12-24 Christmas Eve) + env extension
`MIM_EARLY_CLOSE_EXTRA=YYYY-MM-DD,...`. In `on_bar`, dates in the set stand down
entirely after bar recording: no session init, no marks, no sigma accumulation,
prev_close untouched — exactly the engine's skip semantics. A wrongly-listed date
costs one skipped session (conservative, matches engine behavior on any incomplete
day). Bars are still archived to `bars_raw.csv`.

## 2. Stop-fill reconciliation loop — broker truth between marks

**Gap:** cat-stop fills are detected only at 30-min marks (07-06: 29-minute blind
window) and the bot books inferred exits without broker corroboration (07-06: booked a
fictional CAT_STOP −$500 for an external flatten that realized −$165).

**Change:** every ~30s within the existing poll loop, while holding with a resting
cat-stop, check `is_order_open(cat_stop_id)`. When the stop is no longer open:

| Broker evidence | Action |
|---|---|
| Our stop order has a fill (Trade/search by orderId) | Book `CAT_STOP` immediately at the sealed stop-level convention; log the REAL fill price+fees as a `FILL` row in orders.csv (slippage evidence); cancel-for-mirror-hygiene. |
| Stop canceled unfilled + an opposite-side closing fill exists | Book `EXTERNAL_CLOSE` at the REAL fill price (broker truth, the 07-06 correction as policy); enter safe-mode for the day (`day_deactivated`); CRITICAL alert. |
| Stop canceled unfilled + no closing fill (position unprotected) | Re-place the protective stop at the same level; CRITICAL alert; if re-place is rejected → flatten per seal. |

Never fabricates an exit the broker doesn't corroborate. Commingling-safe: inspects
only our own order ID and our own fills — never net position. The existing mark-time
detection remains as fallback. Ledger conventions unchanged: trades.csv/trades.db keep
the sealed stop-level booking for genuine cat-stops; EXTERNAL_CLOSE rows carry realized
prices (matching the 07-06 correction precedent).

## 3. Frozen / untouched

Bands, sigma(14d), marks, entry/exit/reversal logic, `CAT_STOP_PTS=250`,
`DLL_GUARD_USD=-1000` (sealed 2026-07-07), dynamic clamp, BUFFER_GATE, EOD flatten,
16:01 safety net, contracts=1, all YAML/S25 configs, YANK.

## 4. Deployment

Code commit(s) immediately after this seal on the same branch; merge to main; restart
`trader-mim-nb` while flat. Unit tests for the reconcile decision routing and the
early-close gate accompany the code (`tests/unit/test_mim_reconcile.py`).

## 5. Integrity

Base: main @ e0e9db2 lineage (post PR #5/#6/#7). `src/research/mim_nb_live.py`
pre-change SHA-256: recorded in the code commit diff itself (tamper-evident via git).
