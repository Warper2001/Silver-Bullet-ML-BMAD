# Pre-Registration: MIM-NB Risk-Mechanics Removal (floor gating OFF)

**Generated:** 2026-07-29
**Experiment ID:** mim-nb-risk-mechanics-removal
**Type:** Owner risk decision. **NOT an experiment, NOT an edge claim, NOT a tuning
change.** No parameter is swept, no threshold is derived, no holdout is touched.

**Status:** DRAFT — seals on merge to main.

**Authorization:** Alex, 2026-07-29, explicit and specific: *"allow for the cat-stop. no
more floor... let MIM-NB blow the account."* Scope confirmed by follow-up: **everything
off** (MIM's internal gates **and** the joint floor monitor), **DLL guard retained at
−$1000**.

---

## 0. Motivation — the bot is refusing trades by $0.88

MIM-NB has not been silent for lack of signals. It produced a qualifying 10:00 ET entry
signal on **four consecutive sessions** and refused all four:

```
2026-07-24 BUFFER_GATE 10:00: buffer=499.12 ≤ cat_cost=500.00 [shared] — entry blocked
2026-07-27 BUFFER_GATE 10:00: buffer=499.12 ≤ cat_cost=500.00 [shared] — entry blocked
2026-07-28 BUFFER_GATE 10:00: buffer=499.12 ≤ cat_cost=500.00 [shared] — entry blocked
2026-07-29 BUFFER_GATE 10:00: buffer=499.12 ≤ cat_cost=500.00 [shared] — entry blocked
```

The gate requires the cat-stop's worst case (`CAT_STOP_PTS 250 × PT_VAL $2 × 1ct = $500`)
to fit inside the remaining Topstep MLL buffer. The shared buffer is **$499.12**. The
shortfall is **88 cents**, and it is structural rather than transient: the buffer only
grows if the account gains, and the bot is the thing that would make it gain.

**The owner elects to remove floor-derived entry gating and accept the loss of the
account as a possible outcome.** This document exists so that the decision, its scope,
and what it invalidates are on the record *before* the change ships — not so that it can
be argued for.

## 0.1 This is not justified by expected value, and no such claim is made

Removing a risk control does not create edge. The honest expectation is that this
**lowers** the probability of the combine surviving, in exchange for the bot being able
to act on its signals at all. A gate that blocks 100% of signals produces a strategy with
no trades, which is neither profitable nor measurable. That — measurability — is the
entire benefit being purchased here, and it is being purchased with survival probability.

---

## 1. Change spec (the only diff)

| # | Site (`src/research/mim_nb_live.py`) | Before | After |
|---|---|---|---|
| 1 | entry gate (~1005) | `if buf <= cat_cost: log BUFFER_GATE; skip entry` | gate **removed**; entry proceeds on band break regardless of buffer. Buffer still computed and logged at every entry mark (`BUFFER_INFO`) so the run stays auditable |
| 2 | dynamic DLL clamp (~875) | `dynamic_dll = -min(abs(DLL_GUARD_USD), max(0.0, buf + cat_cost))` | `-abs(DLL_GUARD_USD)` — the **static** −$1000. The clamp was floor-derived: as buffer fell the daily stop shrank toward $0, so any losing day deactivated the bot. Removing the floor without removing the clamp would leave a second, subtler floor gate in place |
| 3 | `combine-floor-monitor.service` | active; halts **both** bots at floor+$100 and writes a `HALT` flag | **stopped and disabled**. No automated halt for MIM-NB **or YANK** |

**Deliberately retained:**

- `DLL_GUARD_USD = -1000.0` — now the **only** automatic brake, per owner instruction.
- The `HALT` flag check in `initialize()`. Nothing writes the flag once the monitor is
  disabled, so it is inert — but it remains a **manual kill switch**: `touch
  data/combine_joint/HALT` will stop MIM-NB from starting. Removing it would delete the
  last cheap intervention for no benefit.
- Buffer computation and logging (`_remaining_mll_buffer`, `_shared_floor_buffer`). With
  the monitor disabled, `floor_state.json` goes stale and the buffer falls back to MIM's
  own-ledger estimate. It is **observability only** — nothing gates on it.

**Frozen and unchanged:** `CAT_STOP_PTS = 250`, `CONTRACTS = 1`, `LOOKBACK_DAYS = 14`, the
band formula, the check-mark set, entry/exit/reversal logic, EOD flatten, cat-stop
mechanics, and the whole of the sigma-provenance repair sealed at `223634b`/`76b4c8c`.

---

## 2. What this invalidates

**`preregistration_mim_nb_post_repair_evaluation.md` (sealed `89050c4`, ~40 minutes before
this document) evaluates a system whose entry gating this change alters.** Its
works/adjust/drop decision therefore governs a different configuration than the one now
deployed.

**Ruling, pre-committed here:** the post-repair evaluation's trade pool **starts at the
first entry taken under this configuration**. Trades taken before it are not poolable with
trades after it. Sessions 2026-07-24 through 2026-07-29, in which signals fired and were
gate-blocked, are **not** evidence of "no signal" and must not be scored as flat days in
any future evaluation — they are censored observations and are to be reported as such.

This is the second pool-reset in eight days (the first from the sigma repair itself). The
post-repair evaluation's N≥20 clock restarts from zero **again**. That cost is accepted
knowingly.

---

## 3. Verification (operational, not statistical)

There is no hypothesis and therefore no acceptance gate in the falsification sense. What
is checked is only that the code does what this document says:

- **(V1)** At an entry mark with `buf ≤ cat_cost`, a band break produces an **order**, and
  the log shows `BUFFER_INFO` rather than `BUFFER_GATE`.
- **(V2)** After a close leaving `day_pnl > −$1000`, the bot is **not** deactivated
  regardless of buffer; at `day_pnl ≤ −$1000` it **is**.
- **(V3)** `combine-floor-monitor.service` is `inactive` and `disabled`, and no `HALT`
  flag is written on a losing session.

**Explicitly NOT verification criteria:** post-change P&L, PF, win rate, or whether the
account survives. A configuration that trades as specified and then loses the account has
been implemented correctly and is a bad idea; those are separate findings.

---

## 4. Disclosures

- **The account can now be lost.** With no buffer gate and no floor monitor, MIM-NB will
  keep entering while the trailing MLL is approached. The Topstep 50K combine (acct
  23884932) can reach its maximum loss limit and be closed. At the time of writing, real
  equity is **$47,971** against a floor of **$47,472** — **$499 of room, roughly one
  cat-stop.**
- **YANK is also unprotected.** `combine-floor-monitor` was the joint kill path for both
  bots. YANK trades 2ct on the same account and has no internal floor gate of its own.
  A YANK loss can now run to the Topstep MLL without automated intervention. YANK's code
  is unchanged by this diff; its *protection* is not.
- **The two bots share one account and can compound each other.** Nothing coordinates
  them once the joint monitor is off.
- **Reversibility.** Every element is reversible: revert this commit and
  `systemctl enable --now combine-floor-monitor`. Reversal does **not** restore a lost
  account.

---

## 5. Sealing

Sealed on merge to main. The code change ships in the same PR and is deployed immediately
after, per owner instruction — there is no staged rollout and no observation period
between seal and deployment.

Sealed as `c69c86d` (prereg) / `647632c` (code), merged in PR #20, deployed 18:01 UTC
2026-07-29.

---

## 6. Amendment 1 — YANK floor removal (2026-07-29, post-seal)

**Authorization:** Alex, 2026-07-29, after the §1–§5 deployment: *"yank too — no floor on
either."*

### 6.1 Correction to §4

§4 stated *"YANK... has no internal floor gate of its own."* **That was wrong.** YANK
carries its own trailing-drawdown tracker, `RiskManager.check_trailing_dd()`
(`src/research/tier2_streaming_working.py:440`), called on **every bar** from the poll
loop (line 1348). It maintains a Topstep-style intraday trailing floor
(`starting_equity 50000 − topstep_trailing_dd_amount 2000`, rising with each equity peak)
and, on breach, sets `_daily_halted = True` — the same flag the daily circuit breaker
sets, blocking all further entries that day.

So YANK had **two** floor-derived brakes, not zero: the joint monitor (already disabled in
§1 site 3) and this internal tracker. The correction is recorded rather than silently
fixed because §4's disclosure understated what was still protecting the account, and
anyone reading the seal later must not inherit that error.

### 6.2 Change spec

| # | Site (`src/research/tier2_streaming_working.py`) | Before | After |
|---|---|---|---|
| A1 | `check_trailing_dd()` breach branch (~483) | logs `TRAILING_DD_BREACH ... halting`, sets `_daily_halted = True`, persists, returns `True` | **no longer halts.** Logs `TRAILING_DD_BREACH ... NOT halting (floor removal, Amendment 1)` and returns `True` for observability. `_daily_halted` is untouched |

The floor, high-water mark, cushion, and `TRAILING_DD_ALERT` are **all still computed and
logged** — identical treatment to MIM's buffer under §1 site 1. Observability is retained;
only the brake is removed.

### 6.3 Retained on YANK

- **Daily circuit breaker** at `StrategyConfig.max_daily_loss` (−$750),
  `check_and_update()`. This is YANK's analogue of MIM's `DLL_GUARD_USD`, which §1
  retained by explicit instruction; symmetry is assumed rather than re-asked. **It is now
  YANK's only automatic brake.**
- `halt_manually()` — the emergency-stop CLI path (FR22), YANK's manual kill switch.
- `check_consistency()` (FR18/FR19) — a Topstep *consistency* rule affecting position
  size, not a floor. Out of scope.

### 6.4 Verification (operational)

- **(V4)** A breach (`equity ≤ trailing floor`) logs `TRAILING_DD_BREACH` and leaves
  `is_halted` **False**.
- **(V5)** A daily loss ≤ −$750 still halts via `check_and_update`.
- **(V6)** `TRAILING_DD_ALERT` still fires in the thin-cushion band.

### 6.5 Disclosure

**Both bots on account 23884932 now run with no floor-derived brake of any kind** —
neither internal nor external. The account's remaining automatic protections are two
static daily loss caps (MIM −$1000, YANK −$750) that reset each session and do not
reference the Topstep MLL. Nothing prevents the combined equity from reaching the MLL, and
nothing coordinates the two bots' simultaneous exposure.

One existing test is amended by this change: `tests/unit/test_risk_manager.py`
`test_breach_halts_trading` asserted `rm.is_halted is True` after a breach. Its assertion
is inverted to `is False` with a reference to this amendment. **The test is not deleted** —
it now pins the new behaviour, so a silent regression back to halting would still fail.

---

## 7. Amendment 2 — floor monitor restored in REPORT-ONLY mode (2026-08-04)

**Authorization:** Alex, 2026-08-04: *"restore the HWM tracking in report-only mode."*

**This restores no brake.** §1 site 3 stopped and disabled `combine-floor-monitor`, and
§6 removed YANK's internal trailing-DD halt. Both stay removed. Neither bot regains an
automatic floor brake.

### 7.1 What went wrong with observability

§1 retained the buffer computation as "observability only", but the *authoritative* source
of that number was the monitor itself: it alone tracked the account high-water mark, and
`tools/combine_ops_healthcheck.py:191` reads equity/floor from the `floor_state.json` the
monitor writes. Disabling the service froze that file at 18:01 UTC on 2026-07-29.

While the account drifted sideways this was harmless. It stopped being harmless on
**2026-08-04**, when the balance reached **$50,217.86** — **$745.74 above the last recorded
HWM of $49,472.12**. The Topstep trailing floor had ratcheted up and nothing was tracking
it, so every cushion figure produced from the frozen file **overstated the real room**.
A risk number that is stale in the optimistic direction is worse than no number.

### 7.2 Change spec

| # | Site | Before | After |
|---|---|---|---|
| C1 | `combine_floor_monitor.py` config | — | `REPORT_ONLY = os.environ.get("FLOOR_MONITOR_REPORT_ONLY", "1") != "0"` |
| C2 | main loop trigger branch | `if reason and not HALT_FILE.exists(): await do_halt(...)` | when `REPORT_ONLY`, log `TRIGGER (report-only, NOT halting)` and do nothing else; the armed path is unchanged behind the opt-in |
| C3 | startup banner | single ARMED line | REPORT-ONLY runs print a `*** WILL NOT HALT OR FLATTEN ***` warning naming the surviving manual kill switches |
| C4 | `combine-floor-monitor.service` | disabled | re-enabled with `Environment=FLOOR_MONITOR_REPORT_ONLY=1` |

**The default is inverted deliberately.** Halting is now **opt-in**. With opt-out, any
stray `systemctl start` of the old unit would silently re-arm a kill path the owner
explicitly removed; with opt-in, the worst a stray start can do is log.

**Report-only never writes the `HALT` file.** MIM-NB `SystemExit`s at startup when that
file exists, so writing it from a "report-only" mode would be a kill path smuggled in
under a safe-sounding name. Pinned by test.

### 7.3 Effect on the bots — none

Nothing gates on the shared floor any more: §1 removed MIM's `BUFFER_GATE`, and the DLL
clamp is static. Restoring `floor_state.json` only changes which number MIM *prints*
(`[shared]` instead of `[own-ledger]`), and the shared figure is the more accurate of the
two — the own-ledger path double-counts the day's P&L, adding `day_pnl` on top of a
`_realized_pnl` that already includes it. **Display only. No entry, exit or size changes.**

`combine_ops_healthcheck.py` regains a live equity readout and its distance-to-floor
alert, which remains an alert and never an action.

### 7.4 Verification

- **(C-G1)** `REPORT_ONLY` defaults to true; only an explicit `0` arms halting.
- **(C-G2)** On the report-only path the loop neither calls `do_halt` nor references
  `HALT_FILE`.
- **(C-G3)** `do_halt` still exists and is still reachable when armed — a mode switch, not
  a deletion of the kill path.
- **(C-G4)** HWM/floor accounting and trigger *detection* are identical in both modes;
  report-only silences the action, not the detection.
