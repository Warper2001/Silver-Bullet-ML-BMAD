# Halt-and-Review: MIM-NB Live-vs-Sealed-Engine Parity (2026-07-07)

**Trigger:** sealed deployment prereg (7939eed) halt-and-review line "equity ≤ $48,400"
fired at 2026-07-07 close ($48,326.44 after the day's cat-stop). Alex elected
**review without halting** (recorded here per the seal's log-why requirement); MIM-NB
continues trading during the review.

**Method:** sealed engine (`study_mim_nb_catstop.py`, seal 6957daa) exec'd verbatim and
replayed over spliced bars — `mnq_1min_2026_ytd.csv` (Jan 2 → Jun 10, σ warmup) +
`data/mim_nb/bars_raw.csv` (Jun 11 → Jul 7, the bot's own recorded live bars). Trades
diffed against `data/mim_nb/trades.csv` (S=500 era ≤06-24, S=250 era ≥06-25); broker
truth pulled from ProjectX `Trade/search`/`Order/search` (acct 23884932, read-only).
Replay script: job tmp `mim_parity_replay.py`; outputs `replay_trades_s{250,500}.csv`,
`mark_parity.csv`, `parity_out.txt`.

## Verdict: NOT ON PARITY — the gap is implementation, not (mostly) regime

Sealed engine on the same bars, live era, matching arm eras: **+256 pts gross ≈ +$490/ct
net**. Live ledger: **−$1,657 modeled** (realized account path 49,472 → 48,326 incl. fees).
≈ $2,150 divergence, attributed below in order of severity.

### 1. DLL guard blocks the re-entries the validation counted on (structural, ongoing)

- Sealed spec + engine: after a cat-stop, *"flat; re-entry permitted at any subsequent
  HH:00/HH:30 check."* The authorizing Monte Carlo cut a day only at **−$1,000** (1ct)
  — it modeled continuing after one $500 cat-stop.
- Live: `DLL_GUARD_USD = −500` (set by prereg 30bc6a8, "DLL tracks cat-stop") disables
  entries for the rest of the session after ONE cat-stop. **The deployed combination
  (cat 250 + DLL −500) was never backtested**; the 250-arm results and MC all ran with
  the −$1,000 day cut.
- Realized impact: 2026-07-02 the engine re-entered SHORT at 10:31 after the cat-stop
  and rode the trend day to EOD **+399 pts (+$798)**; engine day = +$284 net vs live
  −$500. (Counter-case 06-24: engine's re-entry short lost −124.25 pts; live avoided it.
  Net effect still ≈ −$550 live-vs-engine, and structurally the guard amputates exactly
  the fat-tail recovery trades the edge lives on.)

### 2. The 2026-07-06 "cat-stop" was NOT a cat-stop — an external flatten closed it

Broker records (ProjectX):
- Bot's resting stop #3228110588 @ 29762.25: **fillVolume 0, CANCELED** at
  17:30:58.473 UTC.
- Order **#3229490103** — market SELL ×1, `customTag: None`, created 17:30:58.450 UTC
  (13:30:58 ET) — closed the position at **29940.5, realized −$165.00**.
- Flatten-then-cancel 23 ms apart = an atomic flatten sequence. **No local process did
  it**: MIM's 13:30 mark ran at 17:30:03 (action NONE), no MIM/YANK/monitor log lines at
  17:30:58, no service restarts (journalctl empty), floor monitor never triggered.
- TS bars never printed below 29776 after entry — 13.75 pts ABOVE the stop. The sealed
  engine holds this trade to EOD (−61.75 pts ≈ −$124).
- **Open question for Alex: was this a manual flatten from the Topstep UI/app
  (Mon 12:30 PM CT)?** If yes — it cost ~$41 vs the engine path and mislabeled the
  ledger. If no — an unknown actor/risk-system closed a position on a real-money
  account and this becomes an ops/security incident for Topstep support.
- Ledger correction needed: trades.csv/trades.db booked CAT_STOP −$500; truth is
  EXTERNAL_FLATTEN −$165. (Combined PF 0.328@10 in the floor monitor is computed off
  the wrong number.)

### 3. Missed engine winners during the June downtime (operational, past)

Engine took 06-15 LONG +109.25 and 06-16 SHORT +384.25 (**+$987/ct** combined); live has
no trades 06-13 → 06-23 (M26→U26 stale-contract incident era). Already-known incident,
now quantified.

### 4. Half-day handling (latent)

The engine never trades early-close sessions (skips any day without a 16:00 bar —
07-03 skipped). Live traded marks 10:00–13:00 on 07-03 (entered nothing, by luck).
Parity fix: skip sessions with early close, or replicate the engine's completeness rule.

### Matched (parity confirmed where it matters least)

06-12, 06-29, 06-30 EOD trades and the 07-02 & 07-07 cat-stop entries/exits match the
engine within entry-fill modeling noise (≤1.5 pts). Today's stop was a genuine broker
fill AT 29553.75 (order #3234901835, −$484 realized).

## Re-assessment of the "cat-stop rate deviation"

Earlier today: 4 cat-stops in 6 trades (250 era) vs backtest 14% pooled / 20% OOS-2026,
p ≈ 0.005/0.017. With 07-06 reclassified as external flatten, genuine cat-stops are
**3 of 6** (06-25≈, 07-02, 07-07): p ≈ 0.04 / 0.10 — elevated, watchable, but no longer
extreme. **The dominant deviation is implementation drift, not market regime.**

## Recommended actions (each config change needs its own prereg + Alex's go)

1. **Alex answers the 07-06 mystery.** If not manual → Topstep support ticket, treat as
   incident, consider halting until explained.
2. **Ledger correction:** reclassify 07-06 in trades.csv/trades.db (EXTERNAL_FLATTEN,
   −$165), recompute floor-monitor PF inputs.
3. **DLL-guard parity prereg:** restore `DLL_GUARD_USD = −1000` (the value the MC
   actually validated; permits the one re-entry). Interaction disclosed: MIM worst-day
   returns to −$1,000, worsening the joint worst-day math (YANK breaker rescale PR #3
   helps; floor monitor remains the backstop).
4. **Half-day skip** parity fix (small code prereg).
5. **Ops hardening:** bot should poll its stop order status each minute — log broker
   stop fills in orders.csv, detect external closes immediately (took 29 min + was
   mispriced by $335), and alarm on any account order it didn't place.
6. PR #3 shadow-ledger draft: 07-06 must be excluded from any cat-stop event ledger;
   noted there before seal.

## Addendum — corrections applied (2026-07-07 evening, Alex's go)

- **Anomaly disposition (Alex):** parked without attribution; escalate to Topstep only
  on recurrence. Tripwire shipped: `tools/account_order_watchdog.py` (read-only
  ProjectX Order/search diff vs local records; #3229490103 pre-acknowledged; alerts to
  `data/combine_joint/order_watchdog_alerts.csv`). Not yet deployed as a service.
- **Ledger corrected** via `tools/mim_ledger_correction_20260706.py` (dry-run verified,
  then --apply): trades.csv 07-06 row → EXTERNAL_FLATTEN, entry 30023.00 (real fill —
  deviates from the mark-model convention of other rows, deliberately broker-true),
  exit 13:30 @ 29940.50, −$165.00; rows 7–8 re-chained (new head `e9781917c4f1a8c7`),
  full chain re-verified from GENESIS; backup at `data/mim_nb/trades.csv.bak-20260707`.
  trades.db id 6343 updated (pnl −165.0, EXTERNAL_FLATTEN, correction note in metadata).
- **Floor monitor PF corrected:** 0.328 → 0.365 @ N=10 on the next poll.
- **trader-mim-nb restarted** while flat (chain head + realized P&L reloaded from
  corrected CSV; startup reconcile FLAT ✓). Restart exposed the recurring `.env`
  landmine (TRADESTATION_CLIENT_ID/SECRET missing again, same as 2026-06-22) — restored
  from `exchange_token_simple.py`. Root cause of the recurrence still unowned: nothing
  prevents .env regressions; consider a boot-time env sanity check.
- **Watch item:** startup logged `BUFFER WARNING −109.50` from the own-ledger fallback
  before the first shared-floor read (shared buffer was $854). Verify at the next 10:00
  ET mark that BUFFER_GATE gates on the shared number.
- DLL parity-reversion prereg drafted: `preregistration_mim_nb_dll_parity_reversion.md`.
