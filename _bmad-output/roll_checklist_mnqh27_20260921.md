# MNQZ26 -> MNQH27 roll checklist (target ~2026-12-10)

Written 2026-09-21 from what is installed today. **Not a pre-registration.** The previous roll is the template: `roll_z26_verdict.md`
(pre-registration `f18b0cd`), which switched gap-fade on Sun 2026-09-13, YANK on Tue 2026-09-15 and dropped the old recorder symbol on 2026-09-18.
**MNQZ26 expires Fri 2026-12-18** (third Friday of December). The previous quarter's contract, MNQU26, expired 2026-09-18 by the same rule.

## Hardcoded contract symbols today
| component | where | now | at the roll | notes |
|---|---|---|---|---|
| **YANK** | `trader-yank.service` `SYMBOL` | `MNQZ26` | **`MNQH27` by hand** | **No autoroll.** The unit says a missed roll makes orders reject (the 06-16 MIM incident). Highest consequence. Switch only when flat and outside 09:00-16:00 ET. |
| gap-fade | `trader-gap-fade.service` `GAP_FADE_SYMBOL` | `MNQZ26` | `MNQH27` | Do it while flat (a weekend worked last time). The two contracts differ by ~300 pts at the roll, so a session that mixes them fakes a gap. |
| bar recorder | `recorder-gap-fade-bars.service` `RECORDER_SYMBOLS` | `MNQZ26` | **add `MNQH27` before the roll**, drop `MNQZ26` after 12-18 | Changing it is not a strategy-parameter change (the unit says so). |
| MIM-NB | `trader-mim-nb.service` `MIM_NB_SYMBOL` | **`MNQU26` (stale, expired 09-18)** | set to `MNQH27` at the roll | Seed and fallback only: AUTOROLL from the broker's active contract drives the live symbol (sessions show MNQZ26). It is the trigger for the other switches (`AUTOROLL: front month ... -> ...`). |
| combine floor monitor | `src/research/combine_floor_monitor.py:257` | fallback `"MNQZ26"` | change the fallback, or set `MIM_NB_SYMBOL`/`SYMBOL` in its unit | Only used by the flatten path, inert while report-only. The earlier hardcoded `MNQU26` was already fixed in code (env-driven since the 09-18 expiry). |

`btc_combine_streaming.py` (default `MNQM26`) and old research scripts also name quarterly symbols; none is a live MNQ trader.

## Sequence, copied from the Z26 record
1. **Trigger:** MIM-NB logs `AUTOROLL: front month MNQZ26 -> MNQH27`; confirm read-only that ProjectX lists MNQH27 as the only active MNQ contract.
2. **Preconditions before switching YANK** (from the Z26 record): outside 09:00-16:00 ET; `logs/active_trade_state.json` shows no open trade; ProjectX `Position/searchOpen` and `Order/searchOpen` are empty for acct 26556101.
3. Keep a copy of the installed unit before each edit; change only the env line and its roll comment; `daemon-reload`, then restart that one unit.
4. **Rules of engagement:** a restart is allowed after a verified merge; any other start, stop or kill of a trader unit needs Alex's go first. Do not append to a live bot's chained CSV.
5. **Verify** as in the Z26 record: startup banner shows the new symbol; the first data fetch returns 200 for it; no requests for the old symbol; the first RTH session's prior close comes from the same contract.
6. **MIM-NB roll-day tell:** the AUTOROLL mixing bug (U26 open with Z26 marks, spurious LONG, -$355 on 2026-09-15) was fixed and merged 2026-09-17. The tell was `spread +0.00 pt`. Check the first post-roll session in `data/mim_nb/sessions.csv` has `mixed=0`.
7. Drop `MNQZ26` from `RECORDER_SYMBOLS` after 2026-12-18.

Also see `deploy_runbook_20260911.md` for the pre-flight pattern (`git status --porcelain`, record HEAD, keep backups) and its rollback section.
