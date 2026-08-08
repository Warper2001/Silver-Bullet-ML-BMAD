# Build Spec: GAP-1 gap-fade → TradeStation live account

**Status:** SPEC ONLY — no code written, no deployment authorized
**Date:** 2026-08-07
**Author:** party-mode round-table (Victor / Amelia / Winston / Mary), at Alex's direction
**Strategy seal:** `_bmad-output/preregistration_gap_fade_panic_open.md` (32da5d5)
**Related:** `_bmad-output/gap1_friday_review_20260704.md` (promotion NO-GO, two blockers)

---

## 0. What this is, and what it is not

**Is:** an execution-venue change. Same signals, same entries, same exits, same trade
population. TradeStation SIM → a funded TradeStation futures account.

**Is not:** a strategy change. Nothing in §*Frozen Parameters* of the seal is touched.
**Is not:** an authorization to deploy. The sealed promotion gate is not yet met (§6).

**Why gap-fade and not YANK or MIM-NB:** MIM-NB is 6 of 40 sessions into Track A parity —
restarting it destroys the only cheap verdict in flight. YANK is sealed mid-drought pending
PR #37. Gap-fade is under neither clock, already TradeStation-native, and has the highest
trade cadence of the three (≈3.0/week vs YANK's 0.97).

---

## 1. Current state — most of this already exists

`src/research/gap_fade_live.py` (995 lines) is **already a TradeStation client**, not a
ProjectX bot with a mirror bolted on. This is the material difference from the YANK/MIM
port costed on 2026-08-07 (~3 weeks); gap-fade does not need a read side built from scratch.

| Capability | State |
|---|---|
| Order submission (`TSSimClient.submit_entry_bracket`, market + OSO bracket) | exists |
| Cancel / close-at-market / cancel-bracket | exists |
| **Order status read** (`fetch_orders_status`, `_BROKERAGE_ORDERS`) | **exists** |
| **Realized-fill logging** (`_log_realized_fills`, `data/gap_fade/fills.csv`) | **exists** — 12 of 18 trades logged |
| Bars already fetched from the **live** host `api.tradestation.com` | exists |
| Crash-safe state (`_save_state` / `_load_state` / `_clear_state`) | exists |
| Hash-chained audit trail (`ChainedCsv`) | exists |
| Contract auto-roll | **absent** — quarterly manual `GAP_FADE_SYMBOL` edit, next ~2026-09-11 → MNQZ26 |

**The July 4th blocker #1 ("realized fills not logged") is largely closed.** Fidelity over
the 12 logged trades:

```
delta (realized − modeled):  mean −$3.38   max |$65.00|   total −$40.50
```

Against a per-trade sd of $615.67 that is noise. Execution is faithful; six trades are
missing a fill record and §4 Phase 1 closes that gap.

---

## 2. The decisive finding — the risk cap does not work, so the venue must change

Blocker #2 from the Friday review: the stop is `2.0 × gap_abs` beyond the RTH open, so risk
scales with the gap and is unbounded by design. Live, at 1 contract:

```
stop risk /ct :  median $1,303   mean $1,414   max $2,781
                 12 of 18 trades > $1,000     3 of 18 > $2,000
```

A $2,000 trailing MLL cannot carry that. The obvious fix is a max-risk-per-trade cap.
**The obvious fix is wrong**, on two independent grounds.

**Empirically** — the trades a cap removes are the ones carrying the profit.

> **Corrected 2026-08-08 (Phase 1).** The first version of this table was computed from
> `data/gap_fade/trades.csv`, which is corrupt (§8c): it double-counts 2026-06-25 and is
> missing 2026-08-06. The figures below are recomputed from `data/trades.db`, which
> reconciles exactly with the CSV on all 17 shared dates and is the surviving record.
> Headline moved from **PF 1.595 / +$1,874.50** to **PF 1.359 / +$1,130.00**.

Reconciled, N=18, risk/ct median $1,294, max $2,781, 2 trades over $2,000:

| Risk cap /ct | Kept | Net (kept) | PF (kept) | Dropped | Net (dropped) |
|---|---|---|---|---|---|
| $1,500 | 14 | +$720 | 1.332 | 4 | +$410 |
| $2,000 | 16 | +$268 | **1.102** | 2 | +$862 |
| $2,500 | 17 | **−$260** | **0.917** | 1 | +$1,390 |
| **none** | **18** | **+$1,130** | **1.359** | — | — |

The conclusion survives the correction but with less margin than first reported. A $2,000
cap — the one the combine forces — takes PF **1.359 → 1.102**, i.e. out of the seal's
*"scale to 2 ct"* band and into *"continue at 1 ct, re-evaluate at N=60"*. A $2,500 cap
takes it **below 1.00**, which is the seal's STOP band.

The relationship is **non-monotone**, not absent: by gap quartile, mean P&L runs
+$171 / +$205 / **−$200** / +$51. The loss is concentrated in the *middle*, not the tail.
A cap does not trim a tail, it amputates the wrong lobe. *(N=18, and the largest single
contributor is one +$1,390 trade — fragile, and stated as fragile.)*

**Procedurally** — the seal forbids it anyway. `preregistration_gap_fade_panic_open.md`:
*"All parameters below are locked at commit time. **No amendments permitted.**"* and
*"No data subsetting."* A risk cap is a parameter amendment and a subset.

**Therefore the conclusion inverts the blocker.** Blocker #2 is not "gap-fade needs a risk
cap." It is **"gap-fade cannot live behind a $2,000 trailing MLL."** The fix is the venue.
A funded TradeStation account has no MLL, no trailing ratchet and no consistency rule, so
the validated trade population survives intact. That — not convenience — is the case for
this build.

---

## 3. Architecture delta

Four hardcoded constants and one behavioural guard.

| # | Site | Change |
|---|---|---|
| 1 | `gap_fade_live.py:148` `TSSimClient._ORDERS_URL` | class constant → instance field, host from env |
| 2 | `gap_fade_live.py:277` `_BROKERAGE_ORDERS` | same |
| 3 | `gap_fade_live.py:86` `TS_SIM_ACCOUNT` | rename to `GAP_FADE_TS_ACCOUNT`; keep the old env name as a fallback so the running unit does not break |
| 4 | service unit | `GAP_FADE_TS_HOST=api.tradestation.com`, live account id |
| 5 | **new** | `GAP_FADE_LIVE=1` explicit opt-in. Absent ⇒ SIM host, regardless of the other vars. Fail closed: an unset/typo'd host must never resolve to live |

**Auth is not a new surface.** The process already holds a `TradeStationAuthV3` token that
successfully calls `api.tradestation.com` for bars (`TS_BARS_BASE`, line 100) *and*
`sim-api.tradestation.com` for orders. Same OAuth credential, two hosts.
**Verification item V1 (§8):** confirm the live host accepts this token for
`orderexecution` scope — bars are a read scope and may not imply order entry.

**Explicitly NOT in this build:**
- `SimScaler` / `InvVolScaler` (`ts_sim_mirror.py`) — equity-driven growth on real money is
  a bug, not a feature. Size stays at the sealed 1 contract.
- The shared account `SIM2797251F`. Live gets its **own** account, no commingling. The
  2026-06-22 contamination incident is the precedent.
- Any change to YANK or MIM-NB.

---

## 4. Phases

| Phase | Work | Est. |
|---|---|---|
| **0** | **V1 auth check** — confirm the token places an order on the live host. Blocks everything | 0.5 d |
| **1** | Close the fill-logging gap: 6 of 18 trades have no `fills.csv` row. Find why (time-stop path? partial? cancel race?) and make a fill record mandatory per trade | 1–2 d |
| **2** | Host/account parameterization (§3 items 1–5) + fail-closed `GAP_FADE_LIVE` gate | 1 d |
| **3** | Tests: fail-closed default, host selection, account isolation, live-flag absent ⇒ SIM. Repo bar: unit tests beside the module | 1.5 d |
| **4** | Contract auto-roll — port the MIM-NB `activeContract` pattern. **Hard-dated: the manual roll is due ~2026-09-11 (MNQU26 → MNQZ26)** and a stale symbol on a live account is the 2026-06-16 incident with real money attached | 1.5 d |
| **5** | **Gate-Minus-One** — one broker-confirmed round trip on the live account before any analysis counts | 1 session |
| **6** | Deployment prereg + seal (§6) | 0.5 d |

**≈6 working days.** Not the ~3 weeks the YANK/MIM port would cost, because the read side
already exists.

Phase 4 is not optional. It is the only phase with a calendar deadline attached, and it
lands *before* the earliest possible deployment date.

---

## 5. Capital

| Item | Amount |
|---|---|
| Worst observed single-trade stop risk (1 ct) | $2,781 |
| Max drawdown from HWM, live sample (1 ct) | $1,560 |
| Suggested risk capital (worst single + 3× observed maxDD) | ≈ $7,500 |
| Margin: day-trade MNQ (position closes 13:00 ET, never held overnight) | ~$50–100 |
| **Recommended account funding** | **$10,000** |

Round-turn cost at TradeStation tier 1: ≈$1.70/contract (commission $0.50/side + clearing
$0.10/side + exchange/regulatory ≈$0.25/side). Against a per-trade sd of $615.67, cost is
not a design consideration for this strategy.

The strategy never holds overnight (13:00 ET time stop), so overnight maintenance margin
(~$2,057) does not bind and weekend gap risk is nil.

---

## 6. Gates — deployment is NOT authorized by this spec

**The sealed promotion gate has not been met.** `preregistration_gap_fade_panic_open.md`,
*OOS / Live Decision Rule*, decision at **N ≥ 30 live trades AND ≥ 30 calendar days** from
first live trade:

- PF > 1.20 → scale to 2 ct, continue
- PF 1.00–1.20 → continue at 1 ct, re-evaluate at N=60
- PF < 1.00 → STOP, archive

**Current: N = 18, PF = 1.359** (reconciled from `data/trades.db` — see §8c; the
`trades.csv` figure of 1.595 is corrupt), first trade 2026-06-25, calendar condition satisfied.
Still inside the *scale to 2 ct* band, but by 0.159 rather than 0.395.
At the observed ≈3.0 trades/week, N=30 lands **≈ early September 2026**.

Note also what the seal's rule actually authorizes at N=30: *"scale to 2 ct, continue"*.
It says nothing about real money or venue. **Moving to a funded account is beyond the
existing seal and requires its own deployment pre-registration**, which must state at
minimum: the venue, the account, the size (1 ct, unchanged), the halt conditions, and the
evidence rule for whether live-TS trades pool with the SIM sample (recommendation: they do
**not** — venue change, clock restarts, same ruling as MIM-NB §1).

**Build now, deploy after N=30 + the deployment prereg.** The build is on the critical path
either way; the auto-roll deadline (~2026-09-11) essentially coincides with N=30.

---

## 7. What this spec does NOT authorize

- Deploying to a funded account (§6).
- Any risk cap, stop-multiplier change, or gap-threshold change (§2 — forbidden by seal).
- Any direction filter. The seal explicitly forbids it, and note the live split has
  **inverted** against the backtest (live: short PF 1.838 N=10, long 1.075 N=8; sealed
  backtest: long 2.247, short 1.440). At N=8/10 that is noise — and it is exactly the
  observation that tempts a split. Do not.
- Position sizing above 1 contract.
- Touching YANK or MIM-NB.
- Sharing an account with any other bot.

---

## 8a. Phase 0 result (executed 2026-08-08, read-only)

**V1 is CLOSED — PASS.** No order was placed; none was needed.

| Check | Result |
|---|---|
| Token audience | `https://api.tradestation.com` — the **live** host, not sim |
| Token scopes | `openid profile MarketData ReadAccount **Trade** offline_access` |
| Live brokerage `GET /accounts` | HTTP 200 |
| Live **futures** accounts | `210MWN27` (Active), `210URF13` (Active) |
| `210MWN27` balance | **$10,539.01** cash/equity, flat, 0 open orders |
| `210URF13` balance | **$0.00**, flat |
| Order entry on live, already proven | **277 historical orders since 2026-01, with fills** — most recent 2026-07-10 |

Order-entry capability on the live host is therefore established by existing history; the
planned order probe is unnecessary and is dropped from Phase 0. Funding also already meets
§5's $10,000 recommendation.

### 8b. NEW BLOCKER — the funded account is not isolated

`210MWN27` is **not an empty account waiting for a bot.** It carries an active discretionary
trading history:

```
symbols : MNQU26 41 | MNQM26 104 | MNQH26 29 | MCLJ26 93 | MGCJ26 6 | CLJ26 4
by month: 2026-03 132 | 2026-05 79 | 2026-06 25 | 2026-07 41   (last 2026-07-10)
```

Micro crude, micro gold, crude, and hand-traded 1-lot MNQ limit+stop brackets. Deploying
gap-fade here violates §3's isolation requirement and creates three concrete failures:

1. **P&L attribution is destroyed.** The sealed decision rule (§6) is a PF computed on
   gap-fade's trades. Commingled fills make that number unrecoverable.
2. **Position-reconcile corruption.** `_load_state` / broker-confirmed state assumes the
   account's MNQ position belongs to the bot. A manual MNQ position would be read as the
   bot's own — the same class of defect as the 2026-06-22 SIM mirror contamination.
3. **Risk interaction.** §5's $10,000 sizing assumes the whole balance backs one strategy.

**Options (Alex's call, blocks Phase 2):**

| | Option | Note |
|---|---|---|
| **A** | Fund `210URF13` (empty, Active) and deploy there | Clean isolation. Unverified whether futures order entry is enabled — it has never traded |
| **B** | Deploy to `210MWN27` and stop trading it manually | Isolation by discipline, not by construction. This program's own history argues against that |
| **C** | Open a third futures account for automation | Cleanest, slowest |

**Recommendation: A**, with a one-lot broker-confirmed round trip on `210URF13` as
Gate-Minus-One (Phase 5), which doubles as the order-entry check for that specific account.

---

## 8c. Phase 1 result (executed 2026-08-08) — the fill gap was the smaller finding

**Blocker #1 is closed as a code defect. `_log_realized_fills` is correct.** It resolves
fills by the **order IDs it submitted** (`ctx["entry_id"]`, `ctx["exit_id"]`), not by
scraping the account, so shared-account contamination cannot corrupt it. The five missing
dates (2026-06-25, 06-29, 07-01, 07-07, 07-08) all **predate the function's existence** —
first fill row is 2026-07-09.

**Those five are permanently unrecoverable and must be marked so, not backfilled.**
`data/gap_fade/fills_backfill_report.csv` is the proof: it reconstructs by *date* from a
**shared** SIM account and therefore picked up other bots' orders — 2026-06-29 shows extra
round trips at 17:00Z and 20:00Z, 2026-07-07 an extra pair at 14:30Z. Backfilling from it
would produce a plausible, internally consistent, false record. Same ruling as the
2026-08-06 sigma corroboration: **corroborate, don't repair; leave the holes visible.**

### The larger finding — the hash-chained audit trail lost a live trade

| Source | 2026-06-25 | 2026-08-06 |
|---|---|---|
| `logs/gap_fade_live.log` | one entry | **full trade**: entry 13:30:01 (#965760604), exit 15:01:01 +$646, realized fills 15:01:07 |
| `data/trades.db` | **1 row** (correct) | **present**, +$646.00 |
| `data/gap_fade/trades.csv` | **2 rows (duplicate)** | **ABSENT** |
| `data/gap_fade/fills.csv` | n/a (predates) | **ABSENT** |
| `data/gap_fade/decisions.csv` | **2 rows (duplicate)** | **ABSENT** |

Two independent defects:

1. **The 06-25 duplicate is the bot's own.** It is present in committed snapshots back to
   `220dc9d`, with two distinct chain hashes — two separate chained appends, not a copy.
   It violates the seal's frozen `Max trades/day: 1`. Root cause not yet found.
2. **The 08-06 loss is a git artifact.** `trades.csv` and `fills.csv` both carry mtime
   `2026-08-06T20:00` yet their content is byte-identical to commit `1910acb`
   ("bot audit trails through 2026-08-04"). The rows appended live at 13:30–15:01Z that day
   were overwritten by tree manipulation during the untracking work, which committed at
   `84bb7b3` **2026-08-06 21:14:30**.

That is the *same* audit-trail-loss mechanism the room diagnosed for MIM-NB on 2026-08-06.
PR #35 removed the cause 74 minutes after it destroyed gap-fade's record — **and nobody
checked gap-fade.** `data/trades.db` survived because it is a database with a UNIQUE
natural key (commit `da78f5a`), not a git-tracked flat file.

**Consequences for the build:**

- `data/trades.db` is the surviving ledger. §2 and §6 are corrected from it.
- **V8 (new):** the 06-25 double-append is an unexplained defect in a bot proposed for real
  money. It must be root-caused before Phase 2. A bot that can log one trade twice can trade
  twice.
- **V9 (new):** the chained CSVs are evidence for a sealed decision rule and they are not
  durable. Either they become the record (and nothing else may write that directory) or
  `trades.db` is declared the record and the CSVs are demoted to diagnostics. Choose before
  deployment, not after.

---

## 8. Open risks / verification items

| | Item |
|---|---|
| ~~**V1**~~ | ~~Live host may require an order-entry scope the current token lacks~~ — **CLOSED PASS, §8a** |
| **V7** | **The funded live account is not isolated (§8b). Blocks Phase 2.** |
| **V8** | **2026-06-25 was logged twice by the bot itself (§8c). Unexplained. Blocks Phase 2.** |
| **V9** | **The hash-chained CSVs are not durable and lost a live trade (§8c). Pick the record of truth before deployment.** |
| ~~**V2**~~ | ~~Six of 18 trades have no realized-fill record~~ — **CLOSED, §8c**: five, all predating the logger; logger is ID-based and correct; the five are unrecoverable by ruling |
| **V3** | §2's cap analysis rests on 3 trades. It argues *against* adding a constraint, which is the safe direction to be wrong in, but it is not strong evidence |
| **V4** | Live direction split has inverted vs the sealed backtest. Descriptive only; recorded so it cannot later be "discovered" |
| **V5** | Manual contract roll due ~2026-09-11. A stale symbol on a funded account is the 2026-06-16 MIM incident with money attached |
| **V6** | `data/trades.db` is the real-money floor monitor's input. Adding a live-TS trader as a writer needs the natural-key idempotency contract (commit `da78f5a`) honoured |
