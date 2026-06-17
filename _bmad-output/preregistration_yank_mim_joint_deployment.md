# Pre-Registration: YANK + MIM-NB Joint Combine Deployment

**Generated:** 2026-06-17
**Experiment ID:** yank-mim-joint-combine-deploy
**Base commit:** (seal commit SHA recorded by the commit itself)
**Authorizing result:** `yank-mim-joint-combine-mc` — joint combine MC (re-seal 75fc1eb, results aca5785) at **MIM 1ct : YANK 2ct** → primary **64.8% pass / 26.2% blow** vs the 54%/33% MIM-only baseline; pass gain is timeout→pass conversion at flat blow.
**Decision:** Alex authorized adding YANK to the live combine at the vol-balanced **1:2** ratio (2026-06-17).
**Status:** SEALED — committed before the YANK→ProjectX execution-port code is written.

---

## 0. What is new vs the MIM-NB-only deployment (prereg 7939eed)

This adds a **second, independently-sealed strategy (YANK) to the same Topstep 50K combine account**. The strategy logic of neither bot changes. What is new and therefore what this document governs:
1. **YANK executes on the combine** (ProjectX), down-sized **5ct → 2ct**, where today it runs only on a separate TradeStation SIM paper account.
2. **One shared trailing floor.** Both bots' fills hit one account; the MLL ratchet is on **combined** equity. Halt triggers and the consistency rule are re-derived on the aggregate, not per-strategy.
3. **Two independent processes, one account.** A coordination/monitor layer is required so neither bot is blind to the joint risk state.
4. **A live correlation monitor** to catch the one thing the backtest can't guarantee: that the ~0.015 daily correlation holds out of sample.

## 1. Integrity Disclosure

- The joint MC that authorizes this (64.8%/26.2% @ 1:2) rests on a **thin joint pool — only 18 days where both strategies traded** — and both instruments are MNQ, so a true tail regime can lift the benign correlation. The 1:2 size was chosen *because* it sits below where blow starts climbing (1:3 → 29.2%) and leaves headroom to trim. **The binding mitigation is the derived distance-to-floor circuit breaker (§5) — calibrated from the MC paths, not asserted — plus the headroom to trim YANK toward 1ct. Correlation is logged as an observe-only diagnostic, never a trigger.**
- Neither strategy's parameters are touched. MIM-NB = sealed mim-nb-v2-catstop S-B (6957daa). YANK = sealed Tier2 SL2/TP8/ml0.50 (138cab1). Adding them is composition of two independently held-out-validated edges, not new fitting.
- No joint live data exists yet. This is sealed before the execution-port code is written.

## 2. Strategy (frozen — by reference to each seal)

**MIM-NB** — unchanged from live (prereg 7939eed): MNQ front month, **1 contract**, noise-band breakout, 500-pt cat-stop, EOD-flat, per-strategy DLL −$1,000. Already live on ProjectX combine 23884932.

**YANK** — sealed Tier2 SL2/TP8/ml_threshold 0.50 (138cab1), **down-sized to 2 contracts** (from 5):
- MNQ front month, bracket entry (limit at FVG midpoint) + TP limit + SL stop; SL = 2×, TP = 8× gap; ML filter threshold **0.50** (must be verified live-effective — see §6 hazard).
- All other YANK parameters identical to the seal. Only `contracts` changes (5 → 2). Any other change requires a new pre-registration.
- YANK retains its **native daily-loss guard (−$750)**, which is *more* conservative than the −$1,000 the MC modeled → live blow risk is bounded at or below the modeled 26.2%.

## 3. Execution & Account

- **Venue:** TopstepX via ProjectX Gateway (`src/research/projectx_client.py`). YANK is ported off TradeStation SIM onto ProjectX using the existing `ProjectXClient.submit_bracket_order` / `place_exit_orders` / `close_position_at_market` / `reconcile_state` (these were written for the Tier2 bracket shape — YANK is Tier2, so the adapter fits).
- **Account:** the single Topstep **50K Trading Combine** (`PROJECTX_ACCOUNT_ID=23884932`). Both systemd units set it explicitly; both bots refuse to start without it.
- **Contract:** both bots use **auto-roll** (resolve broker `activeContract` at startup + session boundary while flat — the mechanism added for MIM-NB on 2026-06-16). YANK inherits the same resolver.
- **Market data:** TradeStation REST 1-min bars (unchanged for both).
- **Standalone YANK paper instance:** the existing `trader-yank` (TradeStation SIM, 5ct) is **stopped** at cutover (one source of truth for the YANK track record on the combine); its history is retained for reference.

## 4. Two-process coordination (the shared-floor problem)

Both bots run as independent systemd services hitting one account. Rules:
- **Per-strategy intraday risk stays per-strategy and as modeled:** each bot owns its own positions, its own cat-stop/bracket, its own DLL. They do not net or hedge each other.
- **Aggregate floor is monitored by a lightweight third process** (`combine_floor_monitor`), polling ProjectX account equity every ≤60s. It does NOT place trades; it enforces the halt triggers in §5 by signaling both bots to halt-and-flatten. The hard MLL floor is enforced by Topstep itself; this monitor is the early-warning layer above it.
- **Combined consistency rule** (Topstep "best day < 50% of total profit") is evaluated by the monitor on aggregate daily P&L, not per bot.
- **Margin:** MIM 1ct + YANK 2ct = 3 MNQ micros max gross; trivially within 50K combine margin. No joint margin constraint binds.

## 5. Live Decision Rules (joint)

- **Combine pass:** **combined** balance ≥ $53,000 with combined best-day < 50% of total profit → monitor halts entries on both bots, report. Funded-account transition is a separate pre-registration.
- **Halt-and-review triggers (any → flatten both, log, stop):**
  - **DISTANCE-TO-FLOOR (derived, replaces the old absolute $48,400):** combined equity ≤ **current trailing floor + $500**. Value derived, not asserted: in the joint 1:2 MC, $500 is where P(eventual blow | start-of-day distance-to-floor) crosses 50% — i.e. the account becomes more likely to blow than to pass (curve: $400→59%, $500-750→45%, $1000→29%; artifact `tools/derive_floor_trigger.py` + `results_floor_trigger.md`). Expressed relative to the *current* ratcheted floor, so it stays valid after the floor moves (the old absolute $48,400 silently stopped meaning "$400 of room" once the floor ratcheted). On trigger → halt-and-review.
  - either strategy's live slippage per round trip averaging > 3× its modeled reference over ≥10 of its own trades;
  - replay mismatch on either bot: live decisions diverge from its sealed engine replayed on archived bars;
  - **combined** net PF < 0.70 after 30 combined completed trades (≈ below the joint MC's low-percentile path).
- **Correlation = OBSERVE-ONLY diagnostic, NOT a trigger.** Compute and log the trailing 20-day realized daily-P&L correlation each day (baseline ~0.015). It is recorded to explain *why* distance-to-floor compresses, never to fire a halt. Rationale (party-mode review 2026-06-17): correlation is a lagging, noisy (sparse YANK series), scale-blind proxy; the account is killed by summed dollars at the floor, which the distance-to-floor trigger measures directly. No mechanical correlation threshold is set here — if one is ever wanted it must be derived by a dedicated single-knob sweep, not asserted. See [[feedback_derive_dont_assert_one_knob]].
- **No discretionary overrides.** Manual intervention = halt the bots first, log why, then act.

## 6. Reproducibility hazard to clear BEFORE cutover (binding)

The YANK `--ml-threshold 0.50` override silently ran **no-ML** on fresh backtest invocation (only the sealed artifact run 181838 correctly applied ml0.50). Before YANK goes on the combine, **verify the live YANK's effective ML threshold is actually 0.50** (not 0.0/disabled) by inspecting a live decision log where the ML gate is exercised. If the live bot is unknowingly running no-ML, the authorizing MC (which used the ml0.50 series) does not describe it → do not cut over until confirmed. See [[project_yank_mim_correlation_portfolio]].

## 7. Data-Integrity Logging

YANK gets the same hash-chained append-only artifacts as MIM-NB, under `data/yank_live/` (`bars_raw`, `decisions`, `orders`, `trades`, `state.json`), enabling exact replay against its sealed engine. The floor monitor logs aggregate equity + the rolling correlation series to `data/combine_joint/monitor.csv` (also hash-chained). `logs/` keeps operational logs per process.

## 8. Honest Expectations (from joint MC @ 1:2)

64.8% pass / 26.2% blow / 9% still-running at 90 days; median ~40 trading days. The cleanest modeled improvement over MIM-only is the **blow tail (33% → ~26%)**; the pass lift comes from the second stream supplying drift to reach target faster, not from added risk. A blown combine remains a priced ~26% outcome — it does not invalidate the approach unless a halt trigger fires. The thin-data correlation caveat is real; the §5 correlation monitor and the headroom-to-trim at 2ct are the mitigations. Combine running cost: $49/mo + $149 activation (unchanged — same single account).

## 9. Out of scope

- **Parallel second combine** (Victor's optionality lever — running a second account for ~79% pass-at-least-one across two). Separate decision, separate fees, not covered here.
- **Income-engine tuning.** Deferred until combine headroom exists, per the standing objective.
- **Any strategy-logic change** to either bot.
