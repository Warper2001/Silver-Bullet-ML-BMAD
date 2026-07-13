# Halt Review 2026-07-13 — DISTANCE_TO_FLOOR Trigger → Decision: RIDE THE BUFFER

**Generated:** 2026-07-13
**Event ID:** halt-review-20260713-ride-buffer
**Type:** Halt-and-review resolution for the sealed combine floor monitor
(`yank-mim-joint-combine-deploy` derived halt triggers). Owner decision recorded;
one guardrail constant re-armed. NO strategy, sizing, or signal change.

---

## 1. The triggering event (broker-true timeline, UTC)

| Time | Event |
|---|---|
| 14:29:16 | YANK entry fill: SELL 2 MNQU26 @ 29,690.50 (limit #3258987001; ML P=0.576) |
| 15:00:10 | SL stop #3260098229 fills: BUY 2 @ 29,743.50 → **−$212.00** + $1.44 fees |
| 15:00:18 | Monitor trigger: `DISTANCE_TO_FLOOR: equity $48,112 <= floor $47,472 + $750` |
| 15:00:22 | Monitor stopped `trader-mim-nb` + `trader-yank`, wrote HALT flag |

Account state at review: equity **$48,112.00**, trailing floor **$47,472.12**,
buffer **$639.88** — equity $110.12 *below* the floor+$750 halt line. Broker side
clean: flat, zero resting orders (Topstep auto-canceled the sibling TP).
MIM-NB never traded on 07-13 (halt preceded its 10:30 ET window).

Ledger correction applied same day: the killed bot never logged the SL trade;
backfilled into `trades.db` broker-true (id 6366, −$212.0, `backfill: true`).
Combined record since combine start: **N=11, PF 0.341**.

## 2. Owner decision

> **"ride the buffer"** — Alex, 2026-07-13 (this session)

Continue the combine on the remaining ~$640 of buffer; do NOT reset the account.
The distance-to-floor trigger at floor+$750 has served its purpose (it forced this
review); it is now re-armed at a lower line rather than removed.

## 3. Change spec (the only diff)

| Constant (`src/research/combine_floor_monitor.py`) | Before | After |
|---|---|---|
| `HALT_DISTANCE` | `750.0` | `100.0` |

Frozen and unchanged: trailing-floor ratchet, PF trigger (< 0.70 after 30 combined
trades), soft-halt open-position handling, poll cadence, HALT-file mechanism, both
bots' own gates (MIM BUFFER_GATE + dynamic DLL clamp; YANK −$300 daily breaker).

Rationale for $100 (not $0): keeps a final automated tripwire that halts-and-flattens
just above the Topstep hard floor, preserving the early-warning layer's reason to
exist while honoring the ride-the-buffer decision. At $100 the monitor halts at
equity ≤ **$47,572.12** (floor ratchets frozen unless a new HWM is set).

## 4. Disclosures

- A single MIM cat-stop day (−$500) from current equity lands at ~$47,612 — $140
  above the new halt line. A YANK 2ct stop of today's size (−$213) lands at ~$47,899.
  Two average losing days likely end the combine. This is the accepted risk of the
  decision; the alternative (reset) was declined.
- MIM's own BUFFER_GATE (blocks entries when shared buffer ≤ $500 cat cost) becomes
  the binding MIM constraint below $500 of buffer — the monitor line sits beneath it.
- The PF trigger is unaffected but now reads the corrected ledger (N=11, PF 0.341);
  at PF < 0.70 it fires at N ≥ 30 regardless of this change.
- Known open items NOT addressed here (separate fixes): TS SIM mirror sibling-cancel
  gap on halt-kill (stranded SIM long 1 MNQU26 @ 29,478.50 from 07-13, pending
  flatten approval); mirror stranded-lot class previously patched for the
  external-close path only (PR #10).

## 5. Deployment steps (after seal)

1. Edit `HALT_DISTANCE` → `100.0`.
2. `systemctl restart combine-floor-monitor` (picks up the constant).
3. Remove `data/combine_joint/HALT` (both bots refuse startup while it exists).
4. `systemctl start trader-mim-nb trader-yank`.
5. Verify: monitor tick logs no trigger at current equity; both bots log clean
   startup, FLAT reconcile, no orders placed.
