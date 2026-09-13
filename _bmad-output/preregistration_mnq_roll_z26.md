# Pre-Registration: MNQ quarterly roll MNQU26 → MNQZ26 (trader-gap-fade, trader-yank)

**Generated:** 2026-09-13
**Experiment ID:** mnq-roll-z26
**Type:** Operational only. This is a contract-month roll of the same instrument.
- NO change to signal logic, strategy parameters or decision rules.
- NO holdout access.
- It exists because each unit's contract is set by an `Environment=` line, which AGENTS.md treats as a strategy parameter.

**Why now:** MNQU26 expires on Friday 2026-09-18, and trading in it stops at the 09:30 ET open that day. The quarterly volume roll to MNQZ26 was Thursday 2026-09-10. gap-fade's unit planned its switch for "~2026-09-11 → MNQZ26".

**Status:** SEALED on commit. The unit edits land in later commits, following §4.

---

## 1. The change: exactly one `Environment=` line per unit

| Unit | Line now | Line after | Execution venue |
|---|---|---|---|
| `trader-gap-fade` | `GAP_FADE_SYMBOL=MNQU26` | `GAP_FADE_SYMBOL=MNQZ26` | TS SIM paper `SIM2797251F`, plus internal simulation |
| `trader-yank` | `SYMBOL=MNQU26` | `SYMBOL=MNQZ26` | ProjectX combine acct 26556101 (real), TS SIM mirror |

**Also changed:** the roll comment in `deploy/systemd/trader-gap-fade.service` becomes "Next: ~2026-12-10 → MNQH27". MNQZ26 expires 2026-12-18.

**What each line controls:**
- gap-fade: the TradeStation bar feed and the TS SIM order symbol.
- YANK: the TradeStation bar feed, the ProjectX shadow-data contract, and the ProjectX execution contract (`CON.F.US.MNQ.Z26` via `_to_contract_id`).
  - `SYMBOL_SPECS["MNQZ26"]` already exists in `yank_streaming_working.py`: point value 2.0, tick 0.25. `YANK_CONTRACTS=2` still governs size.

**Not changed:**
- Any other `Environment=` line.
- Any code, including the default `SYMBOL` constants in `gap_fade_live.py` and `mim_nb_live.py`. Both are overridden by the environment, and the healthcheck warns if `GAP_FADE_SYMBOL` is ever missing.
- `strategy_config.yaml`.

## 2. Accrual continuity (the decision this document seals)

A roll changes the contract month, not the instrument or the rule. So:

- **GAP-1:** trades on MNQZ26 count toward the same live N under seal 32da5d5. The rule is unchanged: N≥30 plus 30 days; PF above 1.20, between 1.00 and 1.20, or below 1.00.
- **YANK:** trades on MNQZ26 continue the live record under the current seal named in the header of `strategy_config.yaml`.
- **Ledger queries:** both bots write `symbol` into `data/trades.db` from the configured contract, so rows after the roll say MNQZ26. Accrual queries select by `trader_id` (plus `write_mode = 'realtime'`), never by symbol.
- **VOID sessions:** a session is VOID for accrual if the bot could not trade it because its configured contract had expired or was not tradable at the broker, or because the bot was deliberately held back under §5. VOID sessions are neither wins nor losses and are listed in the verdict docs.

## 3. Why the switch must be a restart while flat (declared risks)

**Gap-fade: spurious gap.** MNQZ26 trades at a carry premium to MNQU26, roughly 1%, which is above the 0.5% gap threshold.
- A prior close taken from U26 compared with an open taken from Z26 would create a false gap-up and a false SHORT fade.
- A restart re-fetches `BARSBACK=3000` bars of the new symbol, so both the prior close and the open come from MNQZ26.
- Nothing about the prior close is kept across restarts: `data/gap_fade/state.json` holds only an open position.

**YANK: mixed indicator history.** Its H1/M15/ATR buffers are rebuilt from the new symbol's history at startup.
- The only thing it persists is `logs/active_trade_state.json`: daily P&L and halt flag for the same account, which is correct to carry over.
- It must be flat, with no working orders, so that no position or bracket is stranded on the old contract.

**YANK: the broker's active contract.** MIM-NB resolves its contract from ProjectX's `activeContract` flag (`/Contract/search`). That was added after the 2026-06-16 incident, where a stale front month silently rejected every entry.
- YANK has no such lookup, so it switches only once ProjectX reports MNQZ26 as active (§4.4).
- MIM-NB's log contains no `AUTOROLL` line since 2026-06-11, and no date-fallback warning. Its latest `/Contract/search` (2026-09-12 02:47 UTC, HTTP 200) therefore still resolved to MNQU26, taken from the broker's own flag.

## 4. Procedure (in order)

### 4.1 Seal
Commit this document (`git add -f`) to `main` and push, before any unit is edited.

### 4.2 Repository change
In a worktree under `.claude/worktrees/`:
1. Edit the two repo unit files as in §1, and nothing else.
2. Commit, fast-forward `main`, verify, push.

### 4.3 gap-fade: switch at once, effective from the 2026-09-14 session

**Preconditions, all required:**
- The time is outside 09:25–13:05 ET, or it is a weekend.
- `data/gap_fade/state.json` is absent. That file exists exactly while a simulated or TS SIM position is open.

**Steps:**
1. Make the same one-line edit in `/etc/systemd/system/trader-gap-fade.service`.
2. Check that `diff` against the repo copy is empty.
3. Run `systemctl daemon-reload`, then `systemctl restart trader-gap-fade`. AGENTS.md permits this restart after a verified merge.

**Deadline:** 2026-09-17 16:00 ET.

### 4.4 YANK: switch at the first flat session boundary after ProjectX flips

**Trigger, either one:**
- MIM-NB logs `AUTOROLL: front month MNQU26 → MNQZ26` or `AUTOROLL startup: … → MNQZ26`.
- A read-only `/Contract/search` for `MNQ` (the same call as `mim_nb_live.resolve_front_month`) returns MNQZ26 with `activeContract: true`.

**Preconditions, all required:**
- The time is between 16:00 ET and 09:00 ET, or it is a weekend.
- `logs/active_trade_state.json` holds no open trade.
- ProjectX `Position/searchOpen` for acct 26556101 is empty.
- No working orders remain for YANK.

**Steps:**
1. Make the one-line edit in the installed `/etc/systemd/system/trader-yank.service`.
2. **Keep the installed-only `TIER2_DEBUG=1` block as it is.** It is out of scope (§6).
3. Check that `diff` against the repo copy shows **only** that block.
4. Run `systemctl daemon-reload`, then `systemctl restart trader-yank`.

## 5. Contingencies (pre-decided)

**C1. ProjectX still shows MNQU26 as active at the 2026-09-17 16:00 ET boundary.**
- YANK stays on MNQU26 and switches at the first later boundary where §4.4 is met.
- Any session it cannot trade in the meantime is VOID (§2).
- Nothing further happens without Alex.

**C2. After a switch, the bot cannot fetch MNQZ26 bars (HTTP error, or no bars at all) during market hours.**
- Revert that unit's line to MNQU26, but only while MNQU26 is still tradable (before 2026-09-18 09:30 ET).
- After that point, the bot's sessions are VOID until fixed, and Alex decides the next step.
- This revert is covered by this document. No new seal is needed.

**C3. ProjectX rejects a YANK order on `CON.F.US.MNQ.Z26` for a contract reason.** Handled the same way as C2.

**Stopping or starting any unit other than the restarts in §4 needs Alex's approval,** as AGENTS.md requires.

## 6. Out of scope (flagged, not changed)

- **`trader-yank`'s installed unit** carries `Environment=TIER2_DEBUG=1`, a temporary diagnostic from 2026-08-21 that isn't in the repo copy. It is left exactly as it is.
- **MIM-NB** auto-rolls from the broker flag (`MIM_NB_AUTOROLL` defaults to on). `MIM_NB_SYMBOL=MNQU26` is only its startup seed. Its roll is observed in V6, not changed.
- **`combine_floor_monitor.py:253`** hardcodes `MNQU26` for its ProjectX client. The monitor's balance and position reads are account-wide, so they are unaffected. The contract is used only by the flatten-on-halt path, which does nothing while `FLOOR_MONITOR_REPORT_ONLY=1`. **It must be fixed before the monitor is ever re-armed.**
- **The GAP-1 decision rule** is underpowered at N=30: the 2026-09-13 power gate found a 35.5% false-scale rate under zero edge. This roll does not address that; it would need its own pre-registration.
- **`recorder-gap-fade-bars`:** drop MNQU26 from `RECORDER_SYMBOLS` after 2026-09-18, and add MNQH27 before about 2026-12-10. The recorder is not a trader, so this is not a strategy parameter.

## 7. Verification (recorded in the roll's verdict note)

| # | Check | Pass condition |
|---|---|---|
| V1 | Installed environment | `systemctl show -p Environment` shows MNQZ26 for both units. gap-fade: installed == repo. YANK: installed vs repo differs only by the `TIER2_DEBUG` block |
| V2 | First log lines after restart | gap-fade requests `barcharts/MNQZ26`. YANK logs `Symbol: MNQZ26` and `px_contract=CON.F.US.MNQ.Z26` |
| V3 | gap-fade's first decision after the switch | the `prior_close` and `rth_open` in `decisions.csv` match the witness file `data/gap_fade/bars/MNQZ26.csv` through `tools/gap_fade_census.py`'s loaders, to the tick. Any disagreement is investigated against `MNQZ26_revisions.csv` |
| V4 | Healthcheck during RTH on 2026-09-14 | `recorder-gap-fade-bars: witnessing MNQZ26 … live` |
| V5 | YANK's first order after the switch, if any | accepted by ProjectX on `CON.F.US.MNQ.Z26`, and mirrored on TS SIM as MNQZ26 |
| V6 | MIM-NB (observation only) | logs its `AUTOROLL … → MNQZ26` line, no later than the boundary that triggers §4.4 |

A failed check does not undo the roll by itself. It is written up, and C2 or C3 applies where they fit.
