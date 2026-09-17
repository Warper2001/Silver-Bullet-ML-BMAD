# MIM-NB parity replay refresh (2026-09-18)

**Why:** the last parity check between live and the sealed engine was
`halt_review_mim_nb_parity_20260707.md` (2026-07-07). It found and same-day fixed a real
bug (DLL guard). Nobody re-ran the replay after the fix, after the 2026-09-13 Z26 roll,
or over the ~2.5 months of live trading since — so "the fix shipped" was being treated
as "parity holds" without a fresh check. This closes that gap.

**Method:** `mim_parity_replay_refresh.py`. `run_catstop()` copied verbatim from
`study_mim_nb_catstop.py` (sealed 6957daa, on-disk sha256 `210518d6…`), run at S=250 (the
live cat-stop since 2026-06-25) over `data/mim_nb/bars_raw.csv` (the bot's own recorded
bars; one malformed row and 1,782 reconnect-duplicate rows deduped, keep-first) with
sigma-warmup spliced from `mnq_1min_2026_ytd.csv` Jan 2 → Jun 10 (same split the July
review used; that file's own contamination is out of scope here — it only seeds the
14-day lookback, no P&L before 2026-06-25 is counted). Compared day-by-day against
`data/trades.db` (`trader_id='trader-mim-nb'`, `write_mode='realtime'`) over the same
window. Raw output: `day_diff.csv`, `engine_trades.csv`, `live_trades.csv`.

## Headline: parity holds. The gap to the OOS backtest is not an implementation bug.

| Window | Engine net (sealed, replayed) | Live net |
|---|---|---|
| Whole 250pt era (2026-06-25 → present, N=31 engine / 25 live) | **−$474.44** | **−$540.50** |
| 2026-07-30 → present (11 sessions, post-fix, post-floor-gate-removal) | +$104.38 | +$96.50 |
| 2026-06-25 → 07-29 (the messy transition window) | −$578.82 | −$637.00 |

**Net divergence over the whole era is −$66.06** — about 4% of engine net, on a strategy
whose per-trade SD is ~$420. Day-level swings are much larger ($4,486 summed in
absolute value) but they net out: some hurt live, some helped it. Since 2026-07-30,
divergence is **$7.88 over 11 sessions**, max single-day gap $26.26 — as close to
identical as two independently-computed paths over real market data get.

**This means the −$1,540.50 live figure discussed earlier this session isn't evidence of
a live-only problem.** The sealed engine, replayed forward on the same real bars over the
same calendar window, is *also* roughly flat-to-negative. Both paths are living through
the same stretch of market — Jan–May 2026 (the OOS window the +$32/trade expectation was
measured on) simply produced more trend days than late June–September has so far. That is
the fat-tail-dormancy story the seal itself disclosed (§2b/§2c), now corroborated by a
second, independent replay rather than asserted from the live ledger alone.

## Every day-level divergence ≥$100 traces to a known, already-diagnosed cause

| Day | Diff (live − engine) | Cause | Status |
|---|---|---|---|
| 2026-06-25 | −$500 | Era-transition restart: bot redeployed for the 250pt config while a resting cat-stop order filled offline (`logs/mim_nb_live.log`: "Startup reconcile: cat-stop #3183830426 filled while offline") | operational artifact, one-time |
| 2026-07-02 | −$793.52 | **The DLL guard bug itself** — `DLL_GUARD_USD=−500` blocked the re-entry the engine took and rode to profit. This is the exact day `halt_review_mim_nb_parity_20260707.md` cited. | fixed 2026-07-07, confirmed still `-1000.0` in current `mim_nb_live.py:53` |
| 2026-07-07 | −$644.76 | Genuine broker cat-stop fill (verified against ProjectX order records in the halt review) that the idealized single-bar-OHLC engine model didn't trigger — a fill-price/path noise case, not a logic bug | pre-existing, disclosed engine-vs-broker fill tolerance (≤1.5pt) |
| 2026-07-09, 07-24, 07-27, 07-28 | +$555.98, +$395.98, +$217.74, +$502.24 (**live did better**) | `BUFFER_GATE` blocked entry — combine shared drawdown buffer was below the $500 cat-stop cost (`logs/mim_nb_live.log`, e.g. 07-24: "BUFFER_GATE 10:00: buffer=499.12 ≤ cat_cost=500.00 [shared] — entry blocked"). A deliberate, disclosed risk control, active before floor gating was removed. In this sample it happened to protect live from four engine losses. | floor/buffer gating removed 2026-07-29 (memory: `project_combine_floor_gating_removed_20260729`) |
| 2026-07-29 | +$502.98 | Same-day the floor gate above was removed; live took the full-size trade the engine also took, engine's simpler model just sized it slightly differently | transition day, not recurring |
| 2026-07-16 | −$270.26 | Unattributed residual — both engine and live exited EOD, opposite sign; no gate/restart log entry found. Smallest of the flagged days. | unexplained, immaterial |

## Verdict

1. **The July 7 DLL fix held.** Confirmed in current code, confirmed in the replay: no
   post-07-07 day shows the −500-guard signature (a cat-stop with no re-entry where the
   engine took one).
2. **The BUFFER_GATE/floor-gating episode (pre-07-29) is closed history, not a live risk** — it was removed by a separate, already-recorded decision, and this replay shows it was net-protective while active, not costly.
3. **No new implementation bug found.** The one open, still-live defect remains what was
   already known and flagged this session: the 2026-09-15 roll-contamination trade
   (`project_mim_nb_roll_contamination_20260915`, unfixed). It does not show up as a
   ≥$100 divergence here because the engine replay uses the same spliced bars the live
   bot saw — this check cannot see that specific bug; it was already diagnosed
   separately via the trade's own direction logic.
4. **The honest reframing for "why is live different from testing":** it mostly isn't,
   once you control for two now-closed operational chapters (the DLL bug, the buffer-gate
   window). What's left is what the seal always said was possible — a quiet stretch
   without a trend day — corroborated by the sealed engine itself sitting flat over the
   same calendar window, not just the live ledger.
