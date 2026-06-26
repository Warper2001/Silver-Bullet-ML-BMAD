# Pre-Registration Amendment 2: Execution Venue — CME MBT/MET via TradeStation SIM

**Status:** PROSPECTIVE — SEALED 2026-06-25
**Amends:** `preregistration_kraken_thursday_short.md` (parent, sealed 2026-06-21)
and `..._amendment1_lr_gate.md` (LR gate, sealed 2026-06-25, 0daf3a6)
**Strategy:** DOW-THU (Day-of-Week Thursday Short)

---

## 1. What changes (and what does not)

**Execution venue changes** from Kraken perps/spot-margin (PF_XBTUSD / PF_ETHUSD)
to **CME Micro Bitcoin (MBT) + Micro Ether (MET) futures on the TradeStation SIM
futures account** (`SIM2797251F`, symbol-isolated from the MNQ bots).

**Unchanged (the edge and its evaluation):**
- The signal: short BTC+ETH at Thursday 00:00 UTC, cover 23:05 UTC, 5% per-leg stop.
- The **regime data source** stays the public BTC reference OHLC used in Amendment 1
  (the LR-channel slope is computed from the same daily BTC closes — NOT from the CME
  futures). The sealed LR-gate (Amendment 1) is unaffected.
- The decision rule: Sharpe > 0.80 after N ≥ 30 prospective Thursdays; the LR-gate
  subset evaluated at N_DOWN ≥ 20.

## 2. Why

The Kraken account is **non-ECP**: Kraken spot margin is reduce-only, so it could
never *open* a short (`EOrder:Reduce only:Non-ECP`). The live bot therefore never
filled a trade and the prospective sample never started. CME micro crypto futures on
TradeStation are US-accessible, allow native shorting, trade **24/7** (verified: the
00:00→23:05 UTC window is clear of the 2-min daily maintenance), and are **cash-settled
to CME spot reference rates** (BRR / ETHUSD_RR), so they track spot closely.

## 3. Proxy-instrument disclosure (honest)

MBT/MET are a **proxy** for the Kraken spot/perp BTC+ETH the edge was measured on:
- **Basis + monthly roll** (last-Friday expiry) introduce small tracking differences.
- The day-of-week effect is a BTC/ETH-market-wide phenomenon, so it should transfer;
  this is therefore also a **cross-venue robustness test**. Realized magnitude may
  differ from the Kraken backtest.
- This is **paper (SIM)** — no real slippage; acceptable for signal validation given a
  daily close-to-close hold.

## 4. Sizing & integrity (sealed)

- **Sizing:** 1 MBT base + notional-matched MET (`n_met = round(price_btc / price_eth)`,
  ≈1:38) → equal-weight BTC/ETH. `THU_MBT_SIZE` env, default 1.
- **Execution integrity:** position state is confirmed against broker truth
  (`get_open_positions`), never assumed; rejected/partial entries leave the bot flat and
  raise an ALARM; startup + per-poll reconciliation; absence alarm. (The failure mode
  that hid the Kraken break cannot recur.)

## 5. Prospective record

The prospective sample (and the LR-gate subset from Amendment 1) now accumulates from
the **TradeStation SIM fills**, logged hash-chained under `data/thursday_ts/`:
- `decisions.csv` — one row per Thursday (ENTERED/REJECTED/SKIPPED) with LR-regime tags.
- `trades.csv` — one row per leg close (entry/exit/qty/ret_bps/pnl + LR tags).

**First counted Thursday:** 2026-07-02 (first `--tssim` session after this seal).

## 6. Implementation reference

- `thursday_short.py` (renamed from kraken_thursday_short.py; Kraken execution removed),
  `src/research/ts_thursday_client.py` — commits e3cda9e (mode), 2db1a25 (logging),
  530173d (rename/cutover) on feat/yank-ml-canary.
- `deploy/systemd/trader-thursday-short.service` — ExecStart `thursday_short.py`,
  `THU_TS_SIM_ACCOUNT=SIM2797251F`; installed + restarted 2026-06-25.

## 7. Seal information

**Sealed:** 2026-06-25 — before the 2026-07-02 first counted Thursday, preserving
prospective validity. **Branch:** feat/yank-ml-canary.
**Cross-asset caveat (inherited):** corr(BTC, ETH) ≈ 0.85 — two legs ≈ one bet.
