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
