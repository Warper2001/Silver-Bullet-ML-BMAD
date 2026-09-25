# MNQU26 → MNQZ26 roll: record (pre-registration `f18b0cd`)

## gap-fade: switched Sun 2026-09-13 17:58 ET (21:58:42 UTC)

**Before the switch:**
- It was a weekend and `data/gap_fade/state.json` was absent, so the bot was flat.
- The pre-roll copy of the installed unit was kept in the session scratch dir.

**Repo change:** `a0b9e9f`, fast-forwarded onto `main` and pushed. It touches only the two unit copies.

| Check | Result |
|---|---|
| V1 | `GAP_FADE_SYMBOL=MNQZ26`. The installed unit is byte-identical to the repo copy. The only edits were the env line and the roll comment. |
| V2 | Startup banner: `GAP-1 PANIC-OPEN FADE — TS SIM SIM2797251F — MNQZ26 ×1 ct`. First fetch: `barcharts/MNQZ26 … barsback=3000` returned 200. No MNQU26 requests since the restart. |
| V3 | **Pending: Monday 2026-09-14 open.** From the witness file (`data/gap_fade/bars/MNQZ26.csv`, census loader), the 2026-09-11 RTH session has 389 bars and last RTH close **29691.75**. Monday's `decisions.csv` row must show `prior_close = 29691.75`. |
| V4 | **Pending: Monday RTH.** Outside RTH the healthcheck already reports `recorder-gap-fade-bars: recording MNQZ26`. |

**The risk in §3, measured.** At the 2026-09-11 16:00 ET bar:

| Contract | Close |
|---|---|
| MNQU26 | 29386.5 |
| MNQZ26 | 29683.0 |
| Spread | **+296.5 pts (1.01%)** |

The spread is larger than the 0.5% gap threshold. Mixing the two contracts would have produced a false gap-up.

## YANK: switched Tue 2026-09-15 16:10 ET (20:10:14 UTC)

**Trigger (§4.4):** MIM-NB logged `AUTOROLL: front month MNQU26 → MNQZ26` at 2026-09-15 13:31:03 UTC. This also satisfies V6. The read-only direct check at 16:05 ET showed MNQZ26 as the only listed and active MNQ contract. Alex approved the switch in session.

**Preconditions,** checked at 16:10:10 ET by `yank_roll_switch.sh`:

| Precondition | Result |
|---|---|
| P1 time | outside 09:00–16:00 ET |
| P2 `logs/active_trade_state.json` | keys are only `daily_halted`, `daily_pnl` and `last_trading_date`, so no open trade |
| P3 ProjectX | `Contract/search` active = `CON.F.US.MNQ.Z26`; `Position/searchOpen` = []; `Order/searchOpen` = [] (acct 26556101; all three HTTP 200) |

- The pre-roll copy of the installed unit was kept in the session scratch dir.
- The edit was the single `SYMBOL` line, followed by `daemon-reload` and `restart trader-yank`.

| Check | Result |
|---|---|
| V1 | `systemctl show` lists `SYMBOL=MNQZ26` and `TIER2_DEBUG=1`. The installed unit differs from the repo copy only by the `TIER2_DEBUG` block. |
| V2 | Startup banner: `Symbol: MNQZ26 \| point_value=2.0 tick=0.25 contracts=2` and `DATA: tradestation (signal) + projectx SHADOW \| px_contract=CON.F.US.MNQ.Z26`. Backfill finished with 2,710 bars from `barcharts/MNQZ26`, and ProjectX auth OK. From 20:11 onward it polls `barcharts/MNQZ26` and `History/retrieveBars` every minute. No ERROR or CRITICAL lines, `NRestarts=0`, and no MNQU26 reference since the restart. |
| V5 | **Pending:** YANK's first order after the switch must be accepted on `CON.F.US.MNQ.Z26` and mirrored on TS SIM as MNQZ26. |

**A false alarm during verification.** The switch script's log excerpt printed five `Traceback` lines. Its timestamp filter compared strings, so it also matched untimestamped lines from anywhere in the 2.5M-line log. The most recent traceback in the log dates from 2026-09-04, so none came from this restart.

## V5 — YANK's first order on MNQZ26 (recorded automatically 2026-09-23 11:30:00 UTC)

Detected by `tools/yank_v5_z26_watch.py`, which reads only YANK's own log and
the live ledger. Verify the contract and the TS SIM mirror before calling V5 passed.

**Log evidence:**

- `2026-09-23 11:23:54,979 | INFO     | 🔔 TIER 2 LIMIT PLACED: SHORT limit=$30984.00 | TP $30878.00 SL $31010.50`

