# Pre-Registration: YANK Crash-Recovery Hardening (fill-blind reconcile + backfill state clobber)

**Generated:** 2026-07-13
**Experiment ID:** yank-recovery-hardening
**Type:** Ops/safety fix to the live combine bot's crash-recovery path. NO strategy,
signal, sizing, or exit-parameter change. Same family as the MIM ops-hardening seal
(`preregistration_mim_nb_ops_hardening.md`) and the 07-06 mirror stranded-lot fix (PR #10).

**Status:** seals on merge; deploy at next YANK restart after Alex's go.

---

## 1. The three defects (all observed or reachable on 2026-07-13, real money)

**D1 — fill-blind reconcile resumes phantom trades.** Active-trade state is persisted
once, at entry placement, when `sim_tp_order_id`/`sim_sl_order_id` are still null
(ProjectX defers TP/SL to fill). The combine-path recovery classifies
`ACTIVE = entry order no longer open` — it never checks whether the exit brackets
already closed the trade. Observed 07-13: bot was halt-killed 8s after its SL filled;
on restart it resumed a phantom 2ct short. The phantom's only exit path was the
15:10 CT Topstep flatten, which calls `ProjectXClient.close_position_at_market()` —
a raw market order with no position check — i.e. it would have OPENED a real
unprotected 2ct long on a flat account. Mitigated 07-13 by hand-editing the state
file before 15:10 (backup `logs/active_trade_state.json.bak-20260713-phantom`).

**D2 — backfill replay clobbers restored daily risk state.** Startup backfill replays
historical bars through `_detect_and_enter` → `RiskManager.check_and_update`, whose
day-rollover reset fires on the replayed date transitions. Observed 07-13: restored
`daily_pnl = −212.00` was reset to $0.00 seconds after restore ("New trading day
2026-07-12 — resetting daily P&L (was $-212.00)"), disarming the −$300 daily breaker's
carried loss for the rest of the session.

**D3 — resumed trades advance on backfill bars.** `_advance_active_trade(bar)` is not
backfill-gated. A trade resumed at restart is advanced against historical bars:
`bars_held` inflates spuriously (time-stop clock corrupted), and a replayed bar inside
the 15:10–17:00 CT window fires the market flatten against a historical timestamp.
(Live entries cannot exist during backfill, so this only affects resumed trades.)

## 2. Change spec (`src/research/yank_streaming_working.py` only)

- **F1a — persist on fill:** after the entry fill is detected and ProjectX TP/SL are
  placed, re-persist the full trade state so the exit order IDs are on disk.
- **F1b — fill-aware recovery (combine path):** entry order still open → PENDING
  (existing cancel-and-clear behavior unchanged). Entry no longer open:
  - exit IDs in state and **either bracket still open** (or status unknown) → resume
    ACTIVE, and reconstruct `_active_entry_decision` from persisted fields so the
    normal bar-based exit logic manages the resumed trade (today a resumed trade has
    decision=None and NO working exit path except the 15:10 flatten);
  - exit IDs in state and **both brackets confirmed closed** → trade completed during
    downtime: log CRITICAL (PnL not in trades.db — reconcile manually), clear state,
    do NOT resume;
  - **no exit IDs** (crash inside the fill-detect window) → resume ACTIVE, place the
    protective TP/SL now from persisted prices, persist the new IDs.
- **F2 — backfill never mutates live risk state:** `check_and_update(..., is_backfill)`
  returns False without touching `_daily_pnl`/`_daily_halted`/`_last_trading_date`
  during backfill.
- **F3 — backfill never advances a resumed trade:** `_advance_active_trade` returns
  immediately during backfill.

Frozen and unchanged: all entry/exit parameters, ML gate, sizing (2ct), Topstep
flatten window, commingling-safe rule (net position is still never read — recovery
classifies exclusively via this bot's own order IDs), TS SIM mirror, trade logging.

## 3. Verification plan

- New unit tests (`tests/unit/test_yank_recovery.py`): phantom case clears state and
  does not resume; live-bracket case resumes with reconstructed decision; null-ID case
  resumes and re-places protection; backfill leaves restored daily P&L untouched;
  backfill does not advance a resumed trade.
- Existing floor-monitor + MIM reconcile suites must stay green.
- Deploy at next restart; verify startup log shows the new classification line and
  (when flat with clean state) no recovery action.

## 4. Disclosures

- The confirmed-closed branch cannot recover the missed PnL by itself (exit fill price
  attribution requires a fills query); it logs CRITICAL for manual ledger backfill —
  same procedure as the 07-13 trades.db backfill (id 6366).
- D1's root window (state saved before exit IDs exist) is closed by F1a going forward;
  old state files with null IDs remain handled by the F1b null-ID branch.
