# Pre-Registration: MIM-NB Sigma Provenance Repair (API-seeded → deterministic)

**Generated:** 2026-07-28
**Experiment ID:** mim-nb-sigma-provenance
**Type:** Parity repair to the sealed spec — NO new strategy, NO parameter tuning, NO
holdout access. Makes σ a pure function of recorded market data.

**Status:** DRAFT — seals on merge to main; deployment only after Alex's explicit go.

---

## 0. Motivation — the live bot's σ is not a function of market data

The sealed engine (`study_mim_nb_catstop.py`, seal 6957daa) computes the noise band from
a 14-session history of `|close(d,mark)/open(d,09:31) − 1|` per RTH minute label, kept in
a `deque(maxlen=14)` appended **only for days that pass a whole-day completeness gate**
(`hms[0] == '09:31'` and `'16:00' in hms`; otherwise `continue` — the day contributes
nothing).

The live bot instead **rebuilds `sigma_hist` from a TradeStation REST fetch**
(`_backfill()`, `barsback=30000`) on **every process start** (line 354) and **every
contract roll** (line 372). Between reseeds it accumulates from the live bar stream.

**Consequence: σ depends on when the process last restarted and what the API returned at
that moment, not on market history alone.** The bot has reseeded **29 times in seven
weeks**.

### Evidence (falsification test, not a P&L observation)

σ was reconstructed from the bot's own hash-chained `data/mim_nb/bars_raw.csv` (spliced
with `mnq_1min_2026_ytd.csv` for pre-live warmup) under four day-acceptance conventions
× every trailing window N = 8…24, and scored against all **400 logged σ values** in
`data/mim_nb/decisions.csv`:

| Convention | Rows matched | **Exact matches (<1e-9)** | mean abs err | max abs err |
|---|---|---|---|---|
| ENGINE (needs 09:31 + 16:00, deque) | 369 | **0** | 0.000546 | 0.002713 |
| OPEN-ONLY (needs 09:31, deque) | 388 | **0** | 0.000449 | 0.002076 |
| OPEN-ONLY + list-trim (live's shape) | 388 | **0** | 0.000449 | 0.002076 |
| ENGINE-accept + list-trim | 369 | **0** | 0.000546 | 0.002713 |

**Zero exact matches under any convention or window.** If live's σ were any deterministic
function of the bars live recorded, one of these would have reproduced it exactly. None
did. Corroborating: bar closes reconcile to the cent (`d_close` mean = max = `0.0` in
`tools/mim_parity_replay.py`), so the divergence is in the σ computation's *provenance*,
not in the data.

### Magnitude

At MNQ ≈ 29,000 and σ ≈ 0.005 the band halfwidth is ≈ 145 pt. Observed band error vs the
sealed engine, over 340 matched decision rows:

| | mean | max |
|---|---|---|
| Upper band | 41.1 pt | 601.0 pt |
| Lower band | 81.6 pt | 928.5 pt |

**The lower band is wrong by more than half its own width on average** — sufficient to
change which marks trigger entries. Over the live era the engine takes 21 trades where
live took 13, on identical bars.

### Secondary defect — silent depth suppression

`on_bar()` (line ~817) returns *before* writing a decision row when a minute label has
fewer than `LOOKBACK_DAYS` observations, so suppressed marks are invisible in
`decisions.csv`. Reseed logs show as few as **229 of 390** labels at full depth. Measured
impact: **29 of 429 decision points (6.8%)**, concentrated in five partial sessions
(2026-06-19, 06-25, 07-03, 07-13, 06-11). Real but **not** the dominant cause.

### Tertiary defect — the seed log misreports its own window

`_backfill()` logs `len(seed_days)` (14 **or** 15) while the append loop only ever uses
`seed_days[-LOOKBACK_DAYS:]` (14). Sixteen `Sigma seeded: 15 days` lines are a logging
inaccuracy, not a 15-day window.

### Justification is parity, not P&L

This defect was found while investigating why live underperformed the sealed engine by
**$1,094.50** over the live era (engine S=250: −$504.00 · engine S=500: −$61.00 · live
actual: −$1,598.50). **That P&L gap is cited as context only and is NOT the justification**
— treating it as such would be outcome-peeking. The justification is that the deployed
system does not compute the quantity the sealed spec defines, and cannot be reproduced
offline by construction.

---

## 1. Change spec (the only diff)

> Sites 1–8 and §1.1 were specified before implementation; **sites 9–10 were found while
> building it** and added pre-seal. §1.2 records that provenance.

**Principle: σ becomes a pure, persisted function of the bot's own hash-chained bar
record. No network call may ever influence it.**

| # | Site (`src/research/mim_nb_live.py`) | Before | After |
|---|---|---|---|
| 1 | `_backfill()` (~430–457) | seeds `sigma_hist` from `_ts_get_bars(barsback=30000)` | seeds from `data/mim_nb/bars_raw.csv` (+ `mnq_1min_2026_ytd.csv` warmup for pre-2026-06-11), engine day-acceptance rule |
| 2 | `_save_state()` (~876) | persists `day/position/entry_px/entry_t/cat_stop_id/day_pnl/prev_close/chains` | **additionally persists `sigma_hist` and its contributing day list** |
| 3 | startup (line 354) | unconditional reseed | seed **only if** `state.json` has no `sigma_hist`; otherwise restore verbatim |
| 4 | contract roll (line 372) | `await self._backfill()` reseeds σ **and** `prev_close` | **σ history is NOT reseeded on roll** (σ is a dimensionless ratio; a roll is not a volatility event). **`prev_close` IS re-derived — see §1.1** |
| 5 | day fold, `_new_session()` (~776) | folds `today_moves` whenever `open_d` is not None | folds **only whole accepted days** (09:31 seen **and** 16:00 seen); partial days contribute nothing, matching the engine's `continue` |
| 6 | depth gate (~817) | silent `return` | emit a `DEPTH_GATE` warning + a `decisions.csv` row with `action=DEPTH_SKIP` so suppression is observable |
| 7 | seed log (~453) | logs `len(seed_days)` | logs the count actually appended |
| 8 | σ reduction (~819) | `sigma = sum(sig) / len(sig)` | `sigma = float(np.mean(np.asarray(sig, dtype=float)))` — **byte-identical to the sealed engine's reduction**, with the window ordered oldest→newest as the engine's `deque` is |
| 9 | σ fold timing — `_new_session()` → `on_bar()` at the `16:00` bar | today's moves fold in at the **start of the next session** | fold at the **`16:00` bar of the day itself**, before any gate can `return`; idempotent via `sigma_days` — see §1.2 |
| 10 | `prev_close` assignment (~1028 → ~817) | set **after** the depth gate, so a depth-starved day never updates it | set at the `16:00` bar **before** the depth gate, for **every** accepted day — see §1.2 |

### 1.1 `prev_close` across a contract roll (amendment 2)

σ is a ratio and is correctly left alone by a roll. **`prev_close` is a price level and is
not.** The gap adjustment `max(prev_close − O, 0)` / `max(O − prev_close, 0)` compares two
different contracts across a roll boundary, injecting a synthetic gap the size of the
calendar spread into the first post-roll session.

**Rule:** on roll, `prev_close` is re-derived as **the new contract's own close of the
most recent completed RTH session** (the 16:00 ET bar), never carried over from the
retiring contract. Both the old and new values and the implied spread are logged at
`WARNING`:

```
ROLL prev_close re-derived: OLD=<sym> <px> -> NEW=<sym> <px> (spread <d> pt)
```

This mirrors what a continuous front-month series does at a roll boundary — the series
switches contract wholesale rather than splicing one contract's close onto another's open.
This is a *level* lookup on a single named session, not a distribution rebuild, so it does
not reintroduce the provenance defect §0 describes; it is nonetheless deterministic given
the bar record and is required to be logged so it can be audited after the fact.

### 1.2 Sites 9 and 10 — added after implementation, before sealing

**Provenance note, recorded deliberately:** sites 1–8 and §1.1 were specified before any
code was written. **Sites 9 and 10 were discovered while implementing them** and are added
here *before* this document seals. This is a pre-seal revision of a draft, not an amendment
to a seal already in force — but it is written down so nobody later reads §1 as a clean
a-priori specification when two of its ten rows were found by building the thing.

Both are consequences of site 1/3, not independent ideas, and **G1–G3 cannot pass without
them.**

**Site 9 — the fold must happen at the 16:00 bar.**
Once σ is *restored* from `state.json` (site 3) rather than rebuilt from an API fetch, the
old fold point becomes lossy. `state.json` is written at 16:00; the fold ran at the *start
of the next session*. So an overnight restart resurrects a σ history that is permanently
missing the final day — and unlike the current code, the repaired code has no API reseed to
silently heal it. Folding at 16:00, before any `return`, closes this. The fold is made
idempotent by checking `sigma_days`, so a mid-16:00 restart cannot double-count a session.

**Site 10 — `prev_close` must not sit behind the depth gate.**
The sealed engine sets `prev_close = closes[-1]` for **every accepted day**, regardless of
whether that day was tradeable. The live bot only set it if the depth gate passed, so a
depth-starved session left `prev_close` stale into the next day. Since
`ub = O*(1+σ) + max(prev_close−O, 0)`, a stale `prev_close` corrupts both bands — **G2
fails without this**, independently of σ.

**Neither site changes any parameter, threshold, or trading rule.** Both move *when* an
existing assignment happens. `CAT_STOP_PTS`, `DLL_GUARD_USD`, `CONTRACTS`, `LOOKBACK_DAYS`,
the band formula, and the check-mark set remain frozen as stated below.

**Frozen and unchanged:** `CAT_STOP_PTS = 250`, `DLL_GUARD_USD = -1000.0`,
`CONTRACTS = 1`, `LOOKBACK_DAYS = 14`, the band formula
`ub = O*(1+σ) + max(prev_close−O,0)` / `lb = O*(1−σ) − max(O−prev_close,0)`, check-mark
set, entry/exit/reversal logic, EOD flatten, cat-stop mechanics, the dynamic DLL clamp,
the per-entry `BUFFER_GATE`, and the shared-floor gate.

**No parameter is tuned. No threshold is chosen. Nothing is fitted.**

---

## 2. Acceptance gate (binary, falsifiable, pre-committed)

The repair is accepted **iff** — replaying `tools/mim_parity_replay.py` over the post-fix
live era:

- **(G1)** every logged decision mark satisfies `|live σ − engine σ| < 1e-9`
  — i.e. **exact** reproduction, not "close." Target: **100% of marks, zero exceptions.**
  *G1 is only meetable because diff site 8 makes live's reduction byte-identical to the
  engine's. Without that site this gate would fail on floating-point summation order
  rather than on the defect under test — a false negative that would wrongly condemn a
  correct repair.*
- **(G2)** `|live UB − engine UB| < 0.01 pt` and `|live LB − engine LB| < 0.01 pt` at
  every mark.
- **(G3)** every session logs the full 13 decision marks, or logs an explicit
  `DEPTH_SKIP` row naming the reason.

**If G1 fails on even one mark, the repair is incomplete and must not be called done.**
G1 is deliberately absolute: σ is either deterministic or it is not.

**Explicitly NOT an acceptance criterion:** post-fix P&L, PF, win rate, or trade count.
Those are the *downstream* question and are governed by a separate future prereg. A repair
that achieves G1–G3 and loses money is still a successful repair.

---

## 3. Disclosures

- **This changes which trades the bot takes.** It is not cosmetic. Post-fix entries will
  differ from what the current binary would have produced.
- **The 13 live trades to date cannot be pooled with post-fix trades.** They were produced
  by a bot whose bands were non-deterministic. Any future "works / adjust / drop" decision
  starts its N at **zero** on the fix date. This is a real, accepted cost of the repair:
  seven weeks of live data becomes diagnostic material, not evidence.
- **Cat-stop shadow ledger interaction** (`preregistration_mim_nb_catstop_shadow_ledger.md`,
  SEALED 2026-07-07, Design B fixed-N=10): that ledger has recorded **zero** events
  (`data/mim_nb/shadow_catstop.csv` does not exist), so **nothing is lost or contaminated**.
  Its clock effectively begins at the first post-fix cat-stop. Noted here so the seal is
  not later read as having been silently re-clocked.
- **No sealed holdout data is accessed or spent** by this prereg.
- **Buffer-gate interaction:** MIM-NB is currently entry-blocked
  (`buffer $499.12 ≤ cat_cost $500.00`) and has been for three sessions. This repair does
  **not** change the buffer gate and does **not** unblock trading. Verifying G1–G3 requires
  the bot to reach decision marks, which it does regardless of the entry block (decision
  rows are logged whether or not an entry fires). **Post-fix trade generation remains
  gated until buffer recovers — a separate decision, not authorized here.**
- **Restart-independence is itself testable:** after the fix, restarting the process
  mid-session must leave σ bit-identical. A restart-parity check is part of deployment
  verification (step 5 below).

---

## 4. Deployment (after merge + explicit go)

1. Merge this seal to `main` **before** touching `src/research/mim_nb_live.py`.
2. Apply the ten-site diff in §1, plus the `prev_close` roll rule in §1.1.
3. Restart `trader-mim-nb` while flat (outside 10:00–16:00 ET, or with no open position).
4. Confirm startup logs `Sigma restored from state (N labels)` rather than
   `Backfilling bars for sigma seed...`.
5. **Restart-parity check:** restart a second time within the same session; assert every
   σ in `decisions.csv` after restart #2 is bit-identical to the pre-restart series.
6. Run `tools/mim_parity_replay.py` after the first full post-fix session and evaluate
   G1–G3.

**Rollback:** revert the diff, delete `sigma_hist` from `state.json`, restart. The bot
falls back to API seeding (the current, defective behavior).

---

## 5. Integrity hashes

| Hash | Value |
|---|---|
| (a) `src/research/mim_nb_live.py` SHA-256 (pre-change) | `7b682b30cea8cdce009ce8f7d5bbac8e94a96ce2a2c3481e14f5bb816557f056` |
| (b) `study_mim_nb_catstop.py` SHA-256 (sealed engine, unchanged) | `210518d63a76374f6b8bb79b8e60a2bb3dd94d3e00e1ac73c6b0a032846cc6a1` |
| (c) Git HEAD at draft time | `0d0870a8a637a6c6a8788bb6d101d71dba5870fe` |
| (d) `bars_raw.csv` schema | `ts_utc,open,high,low,close,volume,received_at,chain` (hash-chained rows) |
| (e) Live-era bar span used in the falsification test | `2026-06-11T05:14:00Z` → `2026-07-28T21:00:00Z` (44,703 bars) |
| (f) Reseed count observed in `logs/mim_nb_live.log` | 29 (58 double-logged lines) |

---

## 6. Named successor — this repair must not sit in a drawer (amendment 3)

This seal deliberately does not unblock MIM-NB's entries, so a flawless repair could pass
G1–G3 and still generate zero trades indefinitely. To prevent that, a successor prereg is
named here with a firm authoring trigger:

| | |
|---|---|
| **Successor** | `preregistration_mim_nb_post_repair_evaluation.md` |
| **Authoring trigger** | Within **5 business days** of G1–G3 all passing |
| **Must specify** | (i) where MIM-NB runs so it can accrue N — combine account vs a venue without a $2,000 trailing MLL; (ii) a dated, prospective **works / adjust / drop** decision rule with a pre-committed N and evaluation date; (iii) how the buffer gate is handled |
| **Must not** | pool pre-fix trades into its N (see §3) |

**If G1–G3 pass and the successor has not been authored within 5 business days, that is
itself a process failure and should be recorded as one.** Splitting the repair from the
deployment question keeps each seal reviewable; it is not licence to leave the second one
unwritten.

---

## 7. What this prereg does NOT authorize

- Any change to `CAT_STOP_PTS`, `DLL_GUARD_USD`, `CONTRACTS`, or the buffer gate.
- Unblocking MIM-NB's entries, moving it off the combine account, or resizing it.
- Any conclusion about whether MIM-NB's edge is real. That question remains
  **unanswerable** until G1–G3 pass and a fresh prospective N accrues under a separate,
  dated decision rule.
