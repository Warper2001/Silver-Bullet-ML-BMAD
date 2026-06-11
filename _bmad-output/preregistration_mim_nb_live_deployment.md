# Pre-Registration: MIM-NB Live Deployment — Topstep 50K Combine Entry

**Generated:** 2026-06-11
**Experiment ID:** mim-nb-live-combine
**Base commit:** 1a67de3 (seal commit SHA recorded by the commit itself)
**Authorizing result:** mim-nb-v2-catstop S-B passed Gate A (pooled net PF 1.514) and Gate B
(combine MC 54.0% pass / 33.3% blow @ 1 contract) — prereg 6957daa, results 1a67de3.
**Decision:** Alex authorized direct combine entry at 1 contract (2026-06-11), with full local
logging for data integrity.
**Status:** SEALED — committed before the live bot code is written.

---

## 1. Strategy (frozen — identical to sealed mim-nb-v2-catstop S-B)

- Instrument: **MNQ front month** (currently MNQU26; mechanical roll rule below), **1 contract**, long and short.
- Noise bands per minute label t (RTH 09:31–16:00 ET, 1-min bars end-labeled):
  σ(t) = mean over prior 14 complete RTH days of |close(d,t)/open_d − 1|;
  UB(t) = O·(1+σ) + max(C_prev − O, 0); LB(t) = O·(1−σ) − max(O − C_prev, 0).
- Checks at HH:00/HH:30 bar completions: entries/reversals 10:00–15:30 (close > UB → long, close < LB → short), wide band-stop 10:00–16:00 (long exits on close < LB; short on close > UB).
- Fill model: market order immediately on check-bar completion (= backtest "next bar open").
- **Catastrophe stop:** resting stop order at entry ∓ 500 points placed immediately after entry fill; re-placed on reversal legs.
- EOD: flatten by market + cancel all orders at the 16:00 ET bar (3:00 PM CT — 8 min before Topstep auto-flatten). Belt-and-braces: any position seen after 16:01 ET → immediate flatten.
- **DLL guard (mirrors the MC's modeled rule):** once realized day P&L ≤ −$1,000, no new entries until next session.
- No other filters, no ML, no size changes. Any change requires a new pre-registration.

## 2. Execution & Account

- Venue: TopstepX via ProjectX Gateway API (existing `src/research/projectx_client.py`, infra prereg 2026-06-07).
- Account: Topstep **50K Trading Combine** — `PROJECTX_ACCOUNT_ID` env var MUST be set explicitly to the combine account in the systemd unit; the bot refuses to start without it.
- Market data: TradeStation REST 1-min bars (unchanged data layer), ≥20-day backfill at startup to seed σ(t).
- Roll rule (mechanical): switch to the next quarterly contract at the start of the session 8 calendar days before front-month expiry. Positions are intraday-only, so no roll risk.

## 3. Data-Integrity Logging (required by Alex)

Append-only artifacts under `data/mim_nb/`, each row carrying a SHA-256 hash chain
(`chain_n = sha256(chain_{n−1} | row)`) making post-hoc edits detectable:

| File | Contents |
|---|---|
| `bars_raw.csv` | every 1-min bar as received from TradeStation (timestamp, OHLCV, received_at) |
| `decisions.csv` | every HH:00/HH:30 evaluation: O, C_prev, σ(t), UB, LB, close, VWAP, position, action taken |
| `orders.csv` | every order event: place/cancel/fill-detect, ProjectX order ID, type, side, price, HTTP outcome |
| `trades.csv` | completed round trips: direction, entry/exit px+time, reason (EOD/STOP/REVERSAL/CAT_STOP/DLL), realized P&L |
| `state.json` | crash-recovery snapshot: position, entry, cat-stop ID, day P&L, chain heads |

Plus `logs/mim_nb_live.log` (operational). The raw bar archive enables exact replay:
the sealed backtest engine run over `bars_raw.csv` must reproduce live decisions.

## 4. Live Decision Rules

- **Combine pass:** balance ≥ $53,000 with best day < 50% of total profit → halt entries, report; funded-account transition is a separate pre-registration.
- **Halt-and-review triggers (any):**
  - account equity ≤ $48,400 ($400 above the initial MLL floor);
  - live slippage per round trip (vs decision-bar reference) averaging > 3× the modeled 1-tick/side over ≥10 trades;
  - replay mismatch: live decisions diverge from the sealed engine replayed on archived bars;
  - 30 completed trades with net PF < 0.70 (≈ below the MC's 5th-percentile path).
- **No discretionary overrides.** Manual intervention = halt the bot first, log why, then act.

## 5. Honest Expectations (from Gate B MC)

54% pass / 33% blow / 13% still-running at 90 days; median 41 trading days (~2 months);
~1 trade/day. A blown combine is a priced, expected outcome at ~33% — it does not
invalidate the strategy unless a halt trigger fires. Combine costs: $49/mo + $149 activation.
