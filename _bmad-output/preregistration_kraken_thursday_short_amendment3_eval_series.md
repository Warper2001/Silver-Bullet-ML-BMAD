# Pre-Registration Amendment 3: Evaluation Series & Counterfactual No-Stop Ledger

**Status:** PROSPECTIVE — SEALED 2026-07-10
**Amends:** `preregistration_kraken_thursday_short.md` (parent, sealed 2026-06-21),
Amendment 1 (LR gate, 0daf3a6), Amendment 2 (venue CME MBT/MET via TS SIM, cf0e7f8)
**Strategy:** DOW-THU (Day-of-Week Thursday Short)
**Sample state at seal:** N = 2 counted Thursdays (2026-07-02, 2026-07-09), both realized
losses (−$506.55 stop_loss; −$107.80 scheduled_exit). This amendment changes NO trading
behavior and grants NO new decision authority — it clarifies scoring and adds
knowledge-only instrumentation, sealed early precisely so the reconciliation method is
fixed before the sample can tempt anyone.

---

## 1. Motivation (disclosed honestly)

The sealed backtest pool behind the parent decision rule was built from **daily
close-to-close returns with NO intraday stop**. The live system (per Amendment 2's
sealed signal spec) runs a **5% per-leg intraday stop**, which already altered one
outcome (2026-07-02: BTC leg stopped at −341 bps vs −254 bps had it held to 23:05;
ETH leg stopped at −508 bps vs −566 held). Realized fills and the validation pool are
therefore measuring slightly different strategies. Left undeclared, this ambiguity
would contaminate the N≥30 verdict; declared now, it becomes an attribution tool.

## 2. Primary evaluation series (clarification — no change)

The parent decision rule (**PASS if Sharpe > 0.80 after N ≥ 30 prospective
Thursdays**) is scored on:

- **Realized TradeStation SIM fills, INCLUDING the 5% per-leg stop** — i.e., the
  per-Thursday combined (MBT+MET, equal-notional) return actually logged in
  `data/thursday_ts/trades.csv`. This is what Amendment 2 sealed ("the prospective
  sample accumulates from the TradeStation SIM fills"); restated here to remove all
  ambiguity.
- **Costs at evaluation:** deduct **10 bps per leg round-trip** — the identical cost
  convention used in the sealed backtest pool. This is conservative versus real CME
  micro-crypto commissions (SIM charges none; live retail commissions + fees are
  well under 10 bps of ~$6K leg notional).
- Per-Thursday return = mean of the two legs' net returns (equal-weight), matching
  the pool construction.

## 3. New knowledge-only ledger: `data/thursday_ts/counterfactuals.csv`

A NEW hash-chained CSV (existing `trades.csv` / `decisions.csv` schemas and chains
untouched). One row **per leg per counted Thursday**:

`thursday, symbol, qty, entry_px, stop_trigger_px, realized_exit_t, realized_exit_px,
realized_reason, realized_pnl_usd, cf_2305_px, cf_pnl_usd, source, chain`

- `stop_trigger_px` = entry × (1 + STOP_PCT/100) — the intended stop level, so
  realized stop slippage is decomposable.
- `cf_2305_px` / `cf_pnl_usd` = the counterfactual **held-to-23:05 UTC** exit — the
  sealed backtest's exit definition.

**Resolution method (fixed now):**
- `scheduled_exit` (position held to 23:05): counterfactual ≡ realized
  (`cf_2305_px = realized_exit_px`), `source=live`.
- Early exit (`stop_loss`, `emergency_friday`, `shutdown`): the bot resolves
  `cf_2305_px` from the live quote at its **first poll ≥ that Thursday 23:05 UTC**,
  `source=live`. If the bot misses that window (restart/outage), the row is
  backfilled from TradeStation 15-minute bars (the 23:00 UTC bar close),
  `source=backfill`.
- `cf_pnl_usd = (entry_px − cf_2305_px) × qty × 0.1` (micro = 0.1 underlying),
  same arithmetic as realized.

**Pre-declared uses at verdict time (N ≥ 30):**
1. Reconcile realized results to the sealed close-to-close pool (does the edge exist
   under the pool's own exit definition?).
2. Attribute any gate failure between edge decay and stop cost (paired
   realized-vs-counterfactual differences).

**No decision authority.** The counterfactual series cannot pass/fail the strategy,
trigger a config change, or justify removing/altering the stop. Acting on it in any
way requires a future sealed amendment.

## 4. One-time backfill (disclosed)

The two Thursdays elapsed before this seal are backfilled into the ledger at seal
time:
- **2026-07-02** (stop_loss at 13:24 UTC): `cf_2305_px` from TradeStation 15-min bars
  (23:00 UTC bar close) for MBTN26/METN26, `source=backfill`.
- **2026-07-09** (scheduled_exit): counterfactual ≡ realized, `source=backfill`
  (row written post-hoc, values from the chained trades.csv).

These backfilled rows are identified by `source=backfill` and carry no more authority
than the live rows (none).

## 5. Fill-provenance disclosure (known limitation)

- **Entries** are logged at broker `AveragePrice` from the confirmed position
  (quote-mark fallback if the field is absent).
- **Exits** are logged at the quote mark fetched immediately before the market close
  order — NOT read back from the broker fill. On liquid front-month micros this
  differential is small; it is additionally covered by the conservative 10 bps/leg
  evaluation cost. Broker fill readback is a possible future improvement and would
  require its own (logging-only) amendment note.

## 6. Contract-roll note (operational, no strategy content)

MBTN26/METN26 expire 2026-07-31. Roll dry-run executed 2026-07-10 (read-only):
`resolve_front_month` selects N26 for the 2026-07-16 and 07-23 entries and **Q26 for
2026-07-30** (roll_buffer_days=3 → cutoff 08-02 > expiry 07-31), as designed. Q26
quotes verified live (thin pre-roll volume expected to migrate by 07-30; to be
re-checked 07-29). The roll changes contract months only — no strategy parameter.

## 7. Seal information

**Sealed:** 2026-07-10, before the 2026-07-16 Thursday (N=2 of ≥30).
**Branch:** feat/thursday-amendment3.
**Implementation (committed separately, after this seal):** additive counterfactual
ledger in `thursday_short.py` — no change to entry, stop, sizing, exit logic, or
existing log schemas.
