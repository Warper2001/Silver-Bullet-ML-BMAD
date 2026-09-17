---
title: 'MIM-NB: stop the quarterly roll contaminating its own session'
type: 'bugfix'
created: '2026-09-16'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: '9fcce5faa2563d9585d5f4a7984c640f7d0eae20'
context: ['{project-root}/AGENTS.md']
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** On 2026-09-15 MIM-NB auto-rolled MNQU26 → MNQZ26 inside `on_bar`, then anchored the session on a stale pre-roll bar. `open_d` became 29127.00 (U26) while every later mark was Z26, about 293 points higher, so the bands sat ~293 pts below the mark and `c > ub` was automatic: a forced ENTER_LONG, held to EOD, −$355 on the live combine, when consistent data said no entry at 10:00 and SHORT after. Two further effects: `_prev_close_for_symbol` returned the old contract's close (logged as `spread +0.00 pt`), and all 390 minute-moves were computed as Z26 ÷ U26-open — about 1% each — and folded into 14 sessions of sigma history.

**Approach:** Stamp each bar with the symbol it was *fetched* under, and refuse to let a bar whose stamp differs from the current symbol touch `open_d`, `today_moves` or sigma. Record which contract each session was traded under, so the prior-close lookup can never hand back another contract's price. Both make a rolled session fall through the fail-closed paths that already exist.

## Boundaries & Constraints

**Always:**
- Tag at **fetch** time, never at handle time. The roll runs inside `on_bar`, so by the time a stale bar is handled `self.symbol` is already the new contract — a handle-time tag would mislabel exactly the bar that caused this bug.
- Prefer existing fail-closed paths (`open_d is None` → "stand down today"; `len(today_moves) != FULL_SESSION_BARS` → FOLD REJECT) over new rules.
- Keep `data/mim_nb/bars_raw.csv` append-only and byte-compatible: it is hash-chained, the header is written only at file creation, and `ChainedCsv.append` would emit rows wider than the existing header. Session provenance goes in a **new** file instead.
- A rolled session must be observable in the log: say plainly that bars were dropped, and why.

**Never:**
- No strategy-parameter change: no threshold, band, sigma or cat-stop constant moves. If the fix seems to need one, stop and say so.
- Do not edit `/root/Silver-Bullet-ML-BMAD` directly; work only in the worktree. Do not merge, push, restart any `trader-*` unit, or touch systemd.
- Do not rewrite, migrate or re-chain existing rows of `bars_raw.csv`.
- Do not reconstruct a session's open from a network fetch — the sigma-provenance pre-registration requires bands and sigma to derive from the bot's own record.
- Do not touch `data/mim_nb/state.json` or any already-recorded sigma history (see Decision 1).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| Normal session | Every bar fetched under the current symbol | Unchanged from today: `open_d` set at 09:31, marks evaluated, session folds into sigma | N/A |
| Roll at the session boundary | `_maybe_roll` switches U26→Z26, then a bar fetched under U26 arrives | Bar is recorded but ignored for state: `open_d` stays None → stand down for the session; no entry; fold rejected | Log once, naming both symbols and the dropped bar's timestamp |
| Prior close at a roll | Bar record holds only old-contract sessions for the new symbol | Fall back to the existing broker fetch for the new symbol | Existing "re-derivation FAILED" path when the fetch fails |
| Prior close, contract known | Bar record holds a session recorded under the requested symbol | Use it, as today | N/A |
| Sigma seeding after a roll | History contains a mixed (rolled) session | That session is excluded; all other sessions still seed | If depth < LOOKBACK_DAYS, the existing "entries blocked" warning stands |
| Legacy rows | Sessions recorded before this change, with no symbol on file | Sigma still seeds from them; prior-close lookup does not trust them and fetches instead | N/A |

## Decisions (answered by Alex, 2026-09-16)

1. **Remediation: code fix only.** The ~1%-inflated 2026-09-15 moves are left in place to age out of the 14-session window by about 2026-10-03. Bands stay wider than they should until then, which suppresses entries — the safe direction. Nothing in this spec touches recorded sigma history or `state.json`.
2. **A rolled session stands down.** No entries and no sigma fold, via the fail-closed paths that already exist. About 4 sessions a year go untraded and are VOID for the accrual. Re-anchoring from a network fetch was rejected: the sigma-provenance pre-registration requires bands and sigma to come from the bot's own record.
3. **Scope kept whole** rather than split, so the fix reaches the live bot in one verified merge and one restart.

</frozen-after-approval>

## Code Map

- **Where to work:** every edit and test run happens in the git worktree `/root/Silver-Bullet-ML-BMAD/.claude/worktrees/mim-nb-roll-fix` (branch `fix/mim-nb-roll-contract-contamination`, based on `9fcce5faa2563d9585d5f4a7984c640f7d0eae20`). Every path below is relative to that worktree. The live bots run from `/root/Silver-Bullet-ML-BMAD` itself and a crash restart loads whatever is on disk there, so nothing under that path may be edited. Run tests as `.venv/bin/python` from the worktree (the venv is shared; never install into it).
- `src/research/mim_nb_live.py:516` `_ts_get_bars` -- the single fetch point for both backends (ProjectX shaped and TradeStation REST). Bar dicts carry only `TimeStamp/Open/High/Low/Close/TotalVolume` and no contract field. **Stamp the symbol here.**
- `src/research/mim_nb_live.py:1074` `on_bar` -- handles one bar; calls `_maybe_roll` at the date change (~1090), records to `bars_log` (~1079, before the RTH check), sets `self.open_d = o` (~1096) and `today_moves[hm]` (~1099). The guard belongs between the record and the state updates.
- `src/research/mim_nb_live.py:1236` poll loop -- dedupes strictly by `ts > self.last_bar_ts`, so a bar's timestamp is consumed once. After the guard, the real new-contract 09:31 bar cannot be re-processed; standing down is therefore the only outcome, by design.
- `src/research/mim_nb_live.py:432` `_maybe_roll` / `:357` `_apply_symbol` -- the only writers of `self.symbol`; the roll already clears `open_d`, `today_moves`, `open_d`-derived state.
- `src/research/mim_nb_live.py:467` `_prev_close_for_symbol` -- reads the bar record via `_read_rth_sessions`, with no contract filter. Its broker-fetch fallback is already correct for the new symbol; make the record path reachable only for a session known to be that symbol.
- `src/research/mim_nb_live.py:578` `_read_rth_sessions` / `:700` `_read_recorded_day` -- both `csv.DictReader` by column name; callers at `:475`, `:610` (`_seed_sigma_from_bars`), `:752` (`_catch_up_today`).
- `src/research/mim_nb_live.py:152` `ChainedCsv` + `:177` `bars_log` -- field list is hardcoded and the header is written only when the file is absent. **Do not add a column here.**
- Reuse, do not change: `tools/verify_chain.py:108` derives fields from the file's own header, so a new chained file verifies without edits. `tools/mim_parity_replay.py`, `mim_catstop_shadow_ledger.py`, `bar_parity_probe.py`, `mim_parity_day.py`, `mim_sigma_corroborate.py` all read `bars_raw.csv` by column name and must keep working untouched.
- `tests/unit/test_mim_nb_sigma_provenance.py`, `tests/unit/test_mim_nb_catchup_provenance.py` -- conventions to follow: write a fake `bars_raw.csv` with a hardcoded header and a dummy `chain` value, `monkeypatch.setattr(M, "BARS_RAW_CSV", p)`, build the trader with `object.__new__(MimNbLive)` and set only the needed attributes, then call the target method directly. No network, no credentials.

## Tasks & Acceptance

**Execution:**
- [x] `src/research/mim_nb_live.py` -- in `_ts_get_bars`, stamp every returned bar with the symbol in force at fetch time (a private key on the bar dict) -- so a bar that crossed a roll can be recognised later
- [x] `src/research/mim_nb_live.py` -- in `on_bar`, after recording the bar, drop any bar whose stamp differs from `self.symbol` before it can set `open_d`, `today_moves`, VWAP or sigma; log once per roll naming both symbols -- this is the defect that forced the trade
- [x] `src/research/mim_nb_live.py` -- add an additive, hash-chained session→symbol record (new file under `data/mim_nb/`), written when a session's first RTH bar is handled and marked mixed when a roll lands mid-session -- gives the prior-close lookup something contract-aware to consult
- [x] `src/research/mim_nb_live.py` -- `_prev_close_for_symbol` accepts a recorded session only when that record says the session is the requested symbol; otherwise fall through to the existing broker fetch. Keep the existing log line, and make a `+0.00` spread impossible rather than silent -- a same-price "re-derivation" across contracts is the signature of this bug
- [x] `src/research/mim_nb_live.py` -- `_seed_sigma_from_bars` skips sessions marked mixed, and keeps seeding from sessions of any other contract (a move ratio is contract-agnostic) -- so the fix cannot starve sigma depth and block entries
- [x] `tests/unit/test_mim_nb_roll_contract_guard.py` -- new tests covering every row of the I/O matrix, following the two existing provenance tests' conventions

**Acceptance Criteria:**
- Given the 2026-09-15 shape (09:31 bar stamped U26, later bars stamped Z26, roll in between), when the session is driven, then no entry is taken and the session is not folded into sigma.
- Given a session recorded under a different contract than requested, when the prior close is re-derived, then the recorded value is not returned and the broker fetch is used.
- Given a normal single-contract session, when it is driven, then behaviour is byte-identical to today: same `open_d`, same marks, same fold.
- Given the existing `bars_raw.csv`, when the bot runs, then its header, column count and chain are unchanged, and `tools/` readers keep working.

## Implementation Notes

- **Commit** `8c922fd` on `fix/mim-nb-roll-contract-contamination`, worktree `.claude/worktrees/mim-nb-roll-fix`. 191 lines in `src/research/mim_nb_live.py`, 659 in the new test file.
- **Fetch-time stamp:** `_ts_get_bars` now stamps `_fetch_symbol` on every bar for both backends (the TradeStation branch was restructured into if/else so one stamping loop covers both). `on_bar` drops a bar whose stamp is not the active contract, after it is recorded and before it can touch `open_d`, `today_moves` or VWAP. An unstamped bar (legacy, or a test) is never dropped.
- **Session provenance:** new hash-chained `data/mim_nb/sessions.csv` (`day_et,symbol,first_ts_utc,mixed,detail`). `bars_raw.csv` is untouched — its header is written once at creation, so a new column would have made every row wider than the header and broken chain parsing for all five `tools/` readers. `mixed` is sticky across appended rows, so a restart can only make a session more suspect, never less.
- **Beyond the task list, for the same defect:** `_catch_up_today` now stands down when today's session is recorded as mixed or under another contract. Without it a crash restart (`Restart=on-failure`) would re-anchor from the bar record mid-session and silently undo the guard.
- **Accepted consequence:** the poll loop consumes a bar timestamp once (`ts > last_bar_ts`), so after the stale open bar is dropped the true new-contract 09:31 bar can never be re-processed. A rolled session therefore always stands down — which is Decision 2, reached deliberately rather than as a side effect.
- **Verification:** 52 tests pass (32 new, plus both existing provenance suites unchanged). Full unit suite run separately.
- **Not done here:** the poisoned 2026-09-15 sigma history is left to age out (Decision 1). No merge, no restart.

### Review round (2026-09-17)

- **Patch commit** `67170ae` on top of `8c922fd`. 16 findings patched, 2 deferred, 2 rejected — see the triage log above.
- **Fail-closed gaps closed:** a stood-down session now rolls `prev_close` at 16:00 (the fold still no-ops while `open_d` is None); a failed re-derivation sets `prev_close = None` and lets the existing depth gate stand the session down instead of carrying the retired contract's close; an unreadable provenance record is now `None` ("unreadable") rather than `{}` ("nothing recorded"), and both consumers stand down on it.
- **Record integrity:** `CATCHUP_ROLLED` sets `last_bar_ts` before returning; provenance writes go through a guarded helper that cannot abort bar handling and retries on the next bar; the provenance row is written before the roll and keyed on the bar's own fetch stamp; the sessions chain head joins `_save_state`.
- **Tests:** 44 cases in the new file (was 32), every fix mutation-checked. The ordering test added with a real `_maybe_roll` is the one that matters: moving the guard above the date-change block passed the entire previous suite and fails now.
- **One existing test re-scoped:** `test_mim_nb_1600_closeout_ordering.py` now checks close-out ordering from the CHECK_MARKS gate onward rather than across the whole function. The invariant it encodes (a mark is evaluated against days strictly before today) is intact — the new close-out sits on the `open_d is None` path where no band is ever computed — but it is a sealed-prereg invariant test and is called out here deliberately.

### Verification actually run (deviation from the Verification section)

- **MIM-NB tests:** 335 passed on the fix branch against 289 on a baseline worktree at `9fcce5f` — 46 new tests, no new failures. The 15 failures in `tests/unit/mim_comparison/test_poll_window.py` are pre-existing and identical on the untouched baseline; they assert absolute repo paths and fail in any worktree.
- **Transitive dependants** (gap-fade, SIM mirror, inverse-vol sizing, s26 autoroll, YANK matrix): 74 passed.
- **The full `tests/unit` run was abandoned.** It was killed at 30 and at 50 minutes on this CPU. Instead, every test that can reach `mim_nb_live.py` — 12 files importing it directly, 5 transitively — was run on both branches. Nothing else in the suite imports it.
- **`black --check` / `flake8` at default settings were not achievable:** the file is not clean on the base commit either, and reformatting a live trader file was out of scope. At `--max-line-length=88` the changed file has two fewer findings than the base commit.

### Residual, filed as deferred

A restart *between* sessions across a roll still restores the retired contract's `prev_close`: `state.json` carries no symbol, so `_maybe_roll` sees no change and never re-derives. Verified against the live state file. Pre-existing path, same defect class; the smallest fix is to persist the symbol and distrust `prev_close` on restore when it differs.

## Review Triage Log

Pass 1, 2026-09-17. Three layers reported: blind-hunter (12 findings), edge-case-hunter (11), verification-gap (3 + 5 other). No intent_gap and no bad_spec entry, so no loopback.

| # | Finding | Verdict | Evidence | Route |
|---|---|---|---|---|
| 1 | A stood-down session never rolls `prev_close`, so the next session's gap adjustment uses a two-day-old close | high | Verified: with `open_d is None` the handler returns at `hm != "09:31"`, so the 16:00 bar never reaches `_close_out_session`. The depth-gate path calls it for exactly this reason | patch |
| 2 | `_maybe_roll` carries the retired contract's close forward when re-derivation returns None | high | Verified in the else-branch: `prev_close` is left untouched and only logged. At a real roll the record path can never qualify, so the broker fetch is the only source, and its failure silently keeps cross-contract bands | patch |
| 3 | The `CATCHUP_ROLLED` return leaves `last_bar_ts` unset | high | Verified: the normal catch-up loop sets `self.last_bar_ts = ts_utc`; the new early return precedes it, so the poll loop re-delivers today's bars and re-appends them to the hash-chained `bars_raw.csv` | patch |
| 4 | `sessions_log.append` is unguarded I/O in the hot path, after `bars_log.append` | medium | An OSError propagates out of `on_bar`; the poll loop logs "poll loop error" and never advances `last_bar_ts`, so the same bar is re-appended on every poll — the duplicate-minute damage the fail-closed seed rule exists to catch | patch |
| 5 | New tests append a real row to `data/mim_nb/decisions.csv` (`decisions_log` unpatched) | medium | Mechanism confirmed; blast radius is the worktree's own data directory because `BASE_DIR` derives from `__file__` (the live file was untouched, last written 20:00:03 by the bot). Running the suite from the main checkout would break the live chain | patch |
| 6 | Existing provenance tests now read the live `sessions.csv`; `_catch_up_today` dereferences `self.symbol`, unset in their `_bot()` | medium | Real coupling: neither existing test patches `M.SESSIONS_CSV`. They pass only because their fixture dates predate the file, and `rec is not None` short-circuits the unset attribute | patch |
| 7 | `_read_session_symbols` returns `{}` on a read error, which every caller reads as "unknown, proceed" | medium | Fail-open on a fix whose stated doctrine is fail-closed: a truncated file lets `_catch_up_today` re-anchor and `_seed_sigma_from_bars` seed a contaminated session | patch |
| 8 | A clean `mixed=0` row is appended after the mixed row for the same day | medium | Verified: the stale path sets `_session_contract_mixed` but leaves `_session_contract` None, so the next good bar takes the `is None` branch. Only the collapsing reader saves it; a human or last-row-wins reader sees the day end clean | patch |
| 9 | The sessions chain head is missing from `_save_state`'s `chains` dict | medium | Verified at the `"chains"` literal: bars, decisions, orders and trades only. The new chain is the only one with no restart anchor | patch |
| 10 | The ordering property the whole fix rests on is untested | medium | Pre-verified by the verification-gap layer: every roll-guard test pre-sets `symbol=NEW` and stubs `_maybe_roll`. Moving the guard above the date-change block restores the incident with a green suite | patch |
| 11 | `_backfill` restores sigma from `state.json`, so the mixed-session seed filter never runs live | medium | Pre-verified: `_backfill` returns early when state carries `sigma_hist`, and an existing test asserts that bypass. The behaviour is Decision 1 and accepted; the code comment claims a protection it does not deliver live | patch |
| 12 | The `_read_session_symbols` error branch is untested | medium | Pre-verified: `test_missing_file_is_empty_not_an_error` exits at the `exists()` guard, a different branch. Changing `return {}` to `return out` still passes the whole suite | patch |
| 13 | `_roll_drop_key` is not reset per session, and the drop count is logged at DEBUG while basicConfig is INFO | low | Verified both: a repeat drop with the same symbol pair on a later day logs nothing at INFO, so "every dropped bar is counted" never reaches the log file | patch |
| 14 | Comments overstate coverage: the `+0.00` CRITICAL is exact-equality only, and `_prev_close_for_symbol`'s docstring still says it "prefers the recorded bar file" | low | Verified: a carry-over one tick off logs as a normal roll, and at a real roll the record can never qualify. Direct correction of comment text | patch |
| 15 | A crash between `bars_log.append` and the provenance row leaves the stale bar recorded with no record of its contract | low | Real but a narrow window inside one `on_bar` call; catch-up would then see no record and re-anchor. Writing provenance immediately after the bar is recorded closes it | patch |
| 16 | The poll-loop dedupe that makes the stand-down permanent is asserted only in a docstring | low | Verified: `last_bar_ts = ts` is set after `on_bar` returns, and no test drives it | patch |
| 17 | 2026-09-15 is absent from `sessions.csv`, so a cold re-seed would re-poison sigma from it | medium | Real: absence means "seed it", which inverts the reader's own doctrine for the one session known to be contaminated. The fix is a data write to a live chained file, outside Decision 1's code-only scope | defer |
| 18 | New acceptance rules on sealed functions cite no pre-registration; `data/mim_nb/*` chains are absent from `verify_chain.py` DEFAULT_FILES | medium | Real: neighbouring rules all cite a prereg section. Not a regression for verify_chain, since no mim_nb chain was ever registered. The fix touches methodology docs and tooling registration | defer |
| 19 | The stand-down is unnecessary because a correct re-anchor was available within seconds | false | Refuted by intent rather than by code: Decision 2 chose stand-down deliberately after weighing re-anchoring, which needs the network fetch the sigma-provenance pre-registration forbids | rejected |
| 20 | Early-close sessions get no provenance row, so they look clean to the seed path | false | No reachable harm: early-close sessions have fewer than `FULL_SESSION_BARS` minutes and are already rejected by the fail-closed seed rule before provenance is consulted | rejected |

## Verification

**Commands:**
- `.venv/bin/python -m pytest tests/unit/test_mim_nb_roll_contract_guard.py -v` -- expected: all pass
- `.venv/bin/python -m pytest tests/unit/test_mim_nb_sigma_provenance.py tests/unit/test_mim_nb_catchup_provenance.py -v` -- expected: all pass, unchanged
- `.venv/bin/python -m pytest tests/unit -q` -- expected: no new failures against the same run on the base commit
- `.venv/bin/black --check src/research/mim_nb_live.py; .venv/bin/flake8 src/research/mim_nb_live.py` -- expected: clean
