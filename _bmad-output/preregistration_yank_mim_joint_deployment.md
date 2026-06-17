# Pre-Registration: YANK + MIM-NB Joint Combine Deployment

**Generated:** 2026-06-17
**Experiment ID:** yank-mim-joint-combine-deploy
**Base commit:** (seal commit SHA recorded by the commit itself)
**Authorizing result:** `yank-mim-joint-combine-mc` (re-seal 75fc1eb / results aca5785) → ADOPT at **MIM 1ct : YANK 2ct**. **Re-validated under the Topstep-session constraint** (`results_yank_mim_joint_constrained.md` — the GOVERNING figures): primary **61.2% pass / 29.2% blow** vs the 54%/33% MIM-only baseline (the unconstrained run was 64.8%/26.2%, but used a YANK that can't legally trade the combine). 1:2 is the *smallest* size that beats baseline on both axes once YANK obeys the 15:10 CT flatten.
**Decision:** Alex authorized adding YANK to the live combine at the vol-balanced **1:2** ratio (2026-06-17).
**Status:** SEALED (re-seal v2) — re-sealed after the §6 #3 Topstep-session finding: YANK spec now includes the combine-mandated flatten and the authorization uses the constrained MC. Committed before the YANK→ProjectX execution-port code is written.

---

## 0. What is new vs the MIM-NB-only deployment (prereg 7939eed)

This adds a **second, independently-sealed strategy (YANK) to the same Topstep 50K combine account**. The strategy logic of neither bot changes. What is new and therefore what this document governs:
1. **YANK executes on the combine** (ProjectX), down-sized **5ct → 2ct** AND made **Topstep-session-compliant** (flat by 15:10 CT, no new entries 15:08–17:00 CT, no overnight carry across the close), where today it runs unconstrained 24h on a separate TradeStation SIM paper account.
2. **One shared trailing floor.** Both bots' fills hit one account; the MLL ratchet is on **combined** equity. Halt triggers and the consistency rule are re-derived on the aggregate, not per-strategy.
3. **Two independent processes, one account.** A coordination/monitor layer is required so neither bot is blind to the joint risk state.
4. **A derived distance-to-floor circuit breaker** (§5), calibrated from the MC paths — the binding mitigation for the thin-data risk. Realized correlation is logged as an **observe-only diagnostic** (the backtest can't guarantee the ~0.015 holds out of sample, but correlation is not used as a trigger — see §5 rationale).

## 1. Integrity Disclosure

- The authorizing constrained MC (61.2%/29.2% @ 1:2) rests on a **thin joint pool — only 18 days where both strategies traded** — and both instruments are MNQ, so a true tail regime can lift the benign correlation. The 1:2 size sits below where blow hits the gate ceiling (constrained 1:3 → 32.2%, near the 33% bar). **The binding mitigation is the derived distance-to-floor circuit breaker (§5) — calibrated from the MC paths, not asserted. Trimming YANK toward 1ct lowers blow but also loses the pass-edge (constrained 1:1 only matches baseline), so it is a defensive lever, not a free one. Correlation is logged as an observe-only diagnostic, never a trigger.**
- **MIM-NB parameters are untouched** (sealed mim-nb-v2-catstop S-B, 6957daa). **YANK = sealed Tier2 SL2/TP8/ml0.50 (138cab1) with two deployment changes: contracts 5→2, and a Topstep-session flatten (§2) required for venue compliance.** The flatten is not a strategy edit for edge — it is a venue constraint, and its effect is fully modeled in the constrained MC. Composition of two independently held-out-validated edges, not new fitting.
- No joint live data exists yet. This is sealed before the execution-port code is written.

## 2. Strategy (frozen — by reference to each seal)

**MIM-NB** — unchanged from live (prereg 7939eed): MNQ front month, **1 contract**, noise-band breakout, 500-pt cat-stop, EOD-flat, per-strategy DLL −$1,000. Already live on ProjectX combine 23884932.

**YANK** — sealed Tier2 SL2/TP8/ml_threshold 0.50 (138cab1), **down-sized to 2 contracts** (from 5):
- MNQ front month, bracket entry (limit at FVG midpoint) + TP limit + SL stop; SL = 2×, TP = 8× gap; ML filter threshold **0.50** (verified live-effective — see §6).
- **Topstep-session flatten (deployment change, venue compliance):** flatten any open position at **15:10 CT** market, cancel pendings; **no new entries 15:08–17:00 CT**; never carry across the close. Evening/Globex trading (17:00 CT onward) is retained (Topstep permits it). This is the only logic change beyond sizing; its P&L effect is modeled in the constrained MC (`results_yank_mim_joint_constrained.md`).
- Contracts **5 → 2**. All other YANK parameters identical to the seal. Any further change requires a new pre-registration.
- YANK retains its **native daily-loss guard (−$750)**, *more* conservative than the −$1,000 the MC modeled → live blow risk bounded at or below the modeled 29.2%.

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
  - either strategy's live slippage per round trip averaging > 3× its modeled reference over ≥10 of its own trades (inherited verbatim from each strategy's own sealed prereg; not re-derived here);
  - replay mismatch on either bot: live decisions diverge from its sealed engine replayed on archived bars;
  - **combined** net PF < **0.70** after 30 combined completed trades → halt-and-review. **Derived (no longer a placeholder):** in the joint 1:2 MC, running combined PF at the 30-trade checkpoint crosses the 50%-blow break-even right at PF≈0.70 (0.6–0.7 band → 51.7% blow; 0.7–0.8 → 42.7%; 84% of sims reach 30 trades; artifact `tools/derive_pf_trigger.py` + `results_pf_trigger.md`). The threshold **0.70 is empirically calibrated**; the **N = 30** checkpoint is inherited from prereg 7939eed (a separate knob, not re-derived here, per one-knob discipline; see [[feedback_derive_dont_assert_one_knob]]).
- **Correlation = OBSERVE-ONLY diagnostic, NOT a trigger.** Compute and log the trailing 20-day realized daily-P&L correlation each day (baseline ~0.015). It is recorded to explain *why* distance-to-floor compresses, never to fire a halt. Rationale (party-mode review 2026-06-17): correlation is a lagging, noisy (sparse YANK series), scale-blind proxy; the account is killed by summed dollars at the floor, which the distance-to-floor trigger measures directly. No mechanical correlation threshold is set here — if one is ever wanted it must be derived by a dedicated single-knob sweep, not asserted. See [[feedback_derive_dont_assert_one_knob]].
- **No discretionary overrides.** Manual intervention = halt the bots first, log why, then act.

## 6. Pre-cutover verification (binding — all must pass before YANK trades the combine)

1. **ML threshold effective = 0.50.** The YANK `--ml-threshold 0.50` override silently ran **no-ML** on fresh backtest invocation (only sealed run 181838 applied ml0.50). Verify the live YANK's effective ML threshold is actually 0.50 (not 0.0/disabled) by inspecting a live decision log where the ML gate is exercised. The authorizing MC used the ml0.50 series; if live is no-ML, the MC does not describe it → do not cut over. See [[project_yank_mim_correlation_portfolio]].
2. **YANK day-deactivation matches the model.** The joint MC modeled a per-strategy daily-loss deactivation (the strategy stops trading for the day once its DLL is hit). MIM-NB does this (−$1,000). **Confirm the live YANK/Tier2 bot actually deactivates for the day at its −$750 DLL** — not merely logs the number. If YANK has no day-deactivation, the MC's per-strategy DLL is unmodeled and live blow risk can exceed the modeled 26.2% → fix before cutover. (The −$750-vs-modeled-−$1,000 gap is conservative *only if* the deactivation actually fires.)
3. **Topstep session compliance — RESOLVED in design, verify in code.** Finding (2026-06-17, `precutover_verification_results.md`): Topstep permits Globex/evening trading but auto-flattens all positions at 15:10 CT and blocks entries 15:08–17:00 CT; the original 24h YANK violated this. Resolution: the flatten is now in the YANK spec (§2) and its P&L effect is modeled in the constrained authorizing MC (61.2%/29.2%). **Remaining binding check: the LIVE YANK code must actually enforce the 15:10 CT flatten + 15:08 entry cutoff** — verify by replaying a session where a position would otherwise have carried past 15:10 CT and confirming it flattens. Until the live code enforces this, live behavior ≠ the re-validated analysis → do not cut over.

## 7. Data-Integrity Logging

YANK gets the same hash-chained append-only artifacts as MIM-NB, under `data/yank_live/` (`bars_raw`, `decisions`, `orders`, `trades`, `state.json`), enabling exact replay against its sealed engine. The floor monitor logs aggregate equity + the rolling correlation series to `data/combine_joint/monitor.csv` (also hash-chained). `logs/` keeps operational logs per process.

## 8. Honest Expectations (from joint MC @ 1:2)

**Constrained MC (governing): 61.2% pass / 29.2% blow / 9.7% still-running at 90 days.** The modeled improvement over MIM-only solo (54.0%/33.3%) is +7.2pp pass and −4.1pp blow — thinner than the unconstrained 64.8%/26.2% once YANK obeys the 15:10 CT flatten (which removes ~19% of YANK's window edge). A blown combine remains a priced ~29% outcome — it does not invalidate the approach unless a halt trigger fires. The thin-data correlation caveat is real; the binding mitigation is the **derived distance-to-floor circuit breaker (§5)** (correlation is observe-only, never a trigger). Combine running cost: $49/mo + $149 activation (unchanged — same single account).

## 9. Out of scope

- **Parallel second combine** (Victor's optionality lever — running a second account for ~79% pass-at-least-one across two). Separate decision, separate fees, not covered here.
- **Income-engine tuning.** Deferred until combine headroom exists, per the standing objective.
- **Any strategy-logic change** to either bot.
