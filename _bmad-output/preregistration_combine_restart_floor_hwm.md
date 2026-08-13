# Pre-Registration: Combine Restart — Trailing-Floor HWM Denomination + Per-Account State Binding

**Generated:** 2026-08-13
**Experiment ID:** combine-restart-floor-hwm
**Type:** Operational risk-monitor correction only — NO signal logic, NO strategy
parameter, NO holdout access, NO change to either bot's entry/exit behavior.
**Status:** SEALED on commit; code lands in the immediately following commit(s).
**Trigger:** Alex, 2026-08-13 — "yes prep 1-4 with the prereg seal", ahead of a new
Topstep 50K combine account starting the week of 2026-08-17.

---

## 0. Why this exists — the finding being corrected

Topstep's dashboard reports account **23884932**: *"This account exceeded max loss
limit on 7/6/26 and remains ineligible for funding."* The account continued trading
for 38 days after that date, and this monitor reported healthy cushion throughout.

Reconstructed from `data/combine_joint/monitor.csv` (138,125 ticks, 2026-06-17 →
2026-08-13):

| When (UTC) | Equity | Monitor floor | True floor | Event |
|---|---|---|---|---|
| 06-22 16:03 | **50,955.56** (peak) | 48,955.56 | 48,955.56 | HWM set by an open YANK short in profit |
| 06-25 16:10 | 48,932.60 | 48,955.56 | 48,955.56 | `DISTANCE_TO_FLOOR` fires at −$22.96. **Correct alarm.** |
| 06-26 04:31 | — | → **47,287.50** | 48,955.56 | Commit `317f3ff` reclassifies the alarm as false; HWM switched `equity` → `bal`; floor reset |
| 07-06 17:31 | **48,811.66** | 47,472.12 | 48,955.56 | **−$143.90 through the true floor.** Monitor reports $1,339.54 of room |
| 07-06 18:00 | — | — | — | Topstep flattens the open MIM-NB long — the ledger's unexplained `EXTERNAL_FLATTEN` −$165 |

**Root cause.** Topstep's trailing Maximum Loss Limit ratchets on **equity including
open position profit**. Commit `317f3ff` (2026-06-26) changed the high-water mark to
track realized balance only, describing that as *"mirrors Topstep's own methodology."*
It does not. The change lowered the tracked floor by **$1,668** and made the monitor
structurally unable to observe the binding limit.

**Why the error survived review.** The commit justified dismissing the 06-25 alarm with
*"despite Topstep showing canTrade=true."* A live probe of `/Account/search` on
2026-08-13 returns `canTrade: true` for 23884932 — 38 days after a confirmed breach.
The flag reports tradeability, never funding eligibility, and was therefore incapable
of falsifying the floor model it was used to validate. The same probe shows two earlier
50K accounts already dead (`10442036` $47,890.54, `11542104` $48,511.82, both
`canTrade: false`); 23884932 was the third.

**Bounding the original error.** Topstep did not fail the account on 06-25 despite the
−$22.96 reading, so their high-water mark sits within ~$23 below our recorded peak
(their floor lies in the interval ($48,811.66, $48,932.60]). The equity-denominated HWM
was accurate to roughly twenty dollars. It was replaced for being wrong.

---

## 1. Change — HWM denomination (item 1)

`src/research/combine_floor_monitor.py`, main loop:

```python
st["hwm"] = max(st["hwm"], bal)        # before  (317f3ff, 2026-06-26)
st["hwm"] = max(st["hwm"], equity)     # after   (restores pre-317f3ff behavior)
```

`equity = bal + upl` is already computed one line above and is already what
`evaluate_triggers()` is tested against. This makes the ratchet and the breach test
use the same denomination, which is the invariant that was broken.

**Accepted cost, stated in advance:** the floor now ratchets on unrealized gains that
may never be realized, exactly as `317f3ff` complained. That is not a defect — it is
Topstep's rule, and the 06-25 alarm it produces is a *true* near-miss, not a false
positive. This change will make the monitor noisier and will sometimes report a
breach-adjacent condition while the account is still tradeable. **That asymmetry is
chosen deliberately: a false alarm costs a review, a missed breach costs the combine.**

## 2. Change — per-account state binding (items 2 + 3)

`floor_state.json` carries `hwm`/`floor` with no record of which account produced them.
A trailing floor is meaningless on a different account, so a reset that reuses the file
starts the new combine with the dead one's numbers ($50,217.86 / $48,217.86 as of this
seal, against a correct fresh $50,000 / $48,000).

`load_state()` gains an `account_id` binding with three paths:

| State on disk | Action |
|---|---|
| `account_id` matches `PROJECTX_ACCOUNT_ID` | Load unchanged (normal) |
| `account_id` absent (legacy file) | Adopt in place, keep `hwm`/`floor`, log INFO. Back-compat: the running monitor's readout is unchanged |
| `account_id` differs | **Archive** to `floor_state.json.acct-<old>`, re-genesis at `hwm=$50,000 / floor=$48,000`, log WARNING |

This removes "remember to delete the state file" from the restart runbook. Swapping
`PROJECTX_ACCOUNT_ID` in the unit is now sufficient and self-correcting.

## 3. Change — combine-start binding (item 4)

`COMBINE_START` is a per-account cutover timestamp with a hardcoded default of
`2026-06-17T00:00:00+00:00`. Left as-is, the new account's `COMBINED_PF` trigger would
be computed over the dead account's 21 trades.

`combine_start` moves **into** the state file, set to genesis time when an account
genesises. The cutover is by definition the moment monitoring of that account began, so
this is self-maintaining. The env var is retained only for the legacy-adopt path, so
the currently running monitor's PF window does not move.

## 4. Deployment runbook (new account, week of 2026-08-17)

1. Merge this branch to `main`.
2. Edit `/etc/systemd/system/combine-floor-monitor.service`: `PROJECTX_ACCOUNT_ID=<new id>`.
3. `systemctl daemon-reload && systemctl restart combine-floor-monitor`.
4. Confirm the log line `ACCOUNT CHANGED <old> -> <new> — floor state re-genesised`
   and that `floor_state.json` reads `hwm 50000.00 / floor 48000.00`.
5. Update `PROJECTX_ACCOUNT_ID` for `trader-mim-nb` / `trader-yank` before starting them.
6. `tools/mim_catstop_shadow_ledger.py:83` hardcodes `23884932` — update if used.

## 5. Explicitly NOT in this change

- **`FLOOR_MONITOR_REPORT_ONLY` stays `1`.** Alex removed floor-derived braking from
  both bots on 2026-07-29 (prereg `mim-nb-risk-mechanics-removal`). Re-arming the brake
  reverses an explicit owner decision and is a separate call, deliberately not bundled
  with a correctness fix. **Consequence, stated plainly: on the new account this monitor
  observes and logs the correct floor but will not act on it.**
- `HALT_DISTANCE` ($100), `PF_THRESHOLD` (0.70), `PF_MIN_TRADES` (30), `TRAIL`
  ($2,000), `START_EQUITY`, `FLOOR_START`, `PASS_TARGET` — all unchanged.
- Every strategy config: S25/YANK YAML, MIM-NB sigma/bands/cat-stop, gap-fade, S26/S27.
- YANK's unresolved gap-ceiling denomination (prereg `yank-gap-ceiling-denomination`,
  `2e811d4`) is a separate sealed item and is not touched here. It remains true that
  YANK has not fired since 2026-07-22, so the new combine is MIM-NB-solo until that is
  resolved.

## 6. Falsifiable prediction (the point of sealing this)

Replaying the recorded 2026-06-17 → 2026-08-13 equity series from `monitor.csv`
through the corrected code must produce:

- tracked HWM **$50,955.56**, floor **$48,955.56**
- first `DISTANCE_TO_FLOOR` trigger on **2026-06-25**
- equity below floor on **2026-07-06**, the date Topstep independently reports

The pre-change code produces floor $47,472.12 and no trigger on either date. This is
committed as an executable regression test against the real recorded series, so the
claim fails loudly if it is wrong.

## 7. Integrity

- Base: `main` @ `554b1674bca44c815f2be02929a129ced1381013`
- `src/research/combine_floor_monitor.py` pre-change SHA-256:
  `b1840dd372913722e63aa4b2ddec07c1b5118bce8ca8754142503a9a132b6614`
- Evidence series: `data/combine_joint/monitor.csv` (untracked live output; the
  regression test pins the specific values above rather than the file)
- Superseded reasoning: commit `317f3ff` (2026-06-26)
