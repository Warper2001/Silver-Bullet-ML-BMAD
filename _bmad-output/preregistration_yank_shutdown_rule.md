# Pre-Registration: YANK-STOP — A Rule Under Which YANK Is Switched Off

**Registered:** 2026-08-28
**Status:** SEALED at commit time. Append-only amendments.

---

## 1. The gap this fills

YANK has traded a live Topstep combine since 2026-06-17. **It has no sealed rule under which
it is ever switched off.**

Its only forward rule (`preregistration_yank_sl2tp8_ml050.md`) reads:

> Forward live stop — Disable ML and re-review if live YANK PF < 0.90 after N ≥ 20 live trades.

That disables a **filter**, not the strategy, and it is not eligible: YANK stands at
**N = 11, PF 3.664**. A strategy with no stopping condition is not a strategy under test;
it is a standing position.

## 2. Why this is written today, at N = 11

Today produced evidence that reflects poorly on YANK: it does not generalise to MES
(`preregistration_yank_viability_mes.md`, t = −2.002), it is t = 0.514 on four unseen MNQ
years, and its headline "27× friction headroom" rests on 11 trades whose 95% CI is
**[−3.73, +32.14] bps** and includes zero.

**None of that is grounds for switching YANK off**, and this document does not do so (§6).
Acting on it would be exactly the move this project's seals exist to prevent: overriding a
pre-commitment because a *different* test, on a *different instrument*, came back badly —
which `preregistration_yank_viability_mes.md` §7.4 explicitly forbids.

The legitimate response is to write the missing rule **now**, while N = 11 and nobody can
see how the next trades land. A stopping rule authored after the losses arrive is not a rule.

## 3. "YANK works" is unreachable — declared, with arithmetic

Per-trade sd from the MNQ 2021–2024 run of this configuration is **≈27.6 bps**. YANK's live
rate is **0.125 trades/day**. Sample required to demonstrate an edge at t = 2.0:

| target | N required | elapsed at 0.125/day |
|---|---|---|
| 1× friction (0.51 bps) | 11,715 | **257 years** |
| 3× at booked cost (1.02 bps) | 2,929 | **64 years** |
| 3× bar (1.53 bps) | 1,302 | **28.5 years** |

**A prospective proof that YANK is economically viable is unobtainable at its own trade
frequency.** This seal therefore does not attempt one. It registers a **futility-and-risk
bound**: conditions under which we stop, accepting that "it works" can never be reached —
the same structure as the MIM-NB post-repair evaluation, which declared "WORKS" unreachable
and tested an answerable question instead.

## 4. What "shutdown" means — and why it costs nothing else

**Shutdown = revert to SIM paper. NOT process termination.**

Verified mechanism: `yank_streaming_working.py:1031-1033` selects the execution backend from
`PROJECTX_ACCOUNT_ID` — set → ProjectX/TopstepX combine, unset → SIM paper. The unit file
records the same: *"Removing these reverts to SIM paper."*

This matters because **three sealed prospective experiments read a log YANK writes**:
`logs/yank_shadow_parity.csv` feeds `yank-compressed-cascade-phase2` (13/30 accrued),
`yank-compressed-cascade-lr-filter`, and `yank-lrc-phase2`. The shadow-parity writer is
documented at `yank_streaming_working.py:1381` as *"Observation only — never touches trade
state"*, so it runs on either backend.

**Reverting to SIM removes the capital risk and preserves all three clocks.** Killing the
process would destroy them. Any execution of §5 uses the revert, never `systemctl stop`.

## 5. Triggers — any ONE fires the revert

Evaluated on `data/trades.db`, `trader_id='trader-yank'`, trades on or after the combine
reset of **2026-08-13** (account 26556101). Current state at seal: **N = 2, +$327.00**.

| # | Trigger | Derivation |
|---|---|---|
| **T1 — risk** | Cumulative live net ≤ **−$1,000** | Half the combine's $2,000 trailing MLL. YANK is one of two strategies on acct 26556101; an equal split is the neutral allocation, chosen before any loss exists. |
| **T2 — futility** | **N ≥ 30** live trades **AND** PF < 1.00 | 30 is ~5 months at 0.125/day. PF < 1.0 after 30 trades means the strategy failed to show positive edge in the only sample that could ever favour it (live, its own instrument, its own era). |
| **T3 — drawdown** | Drawdown from YANK's own live high-water mark ≥ **$1,000** | Catches the win-then-give-it-back path that T1 misses while cumulative P&L is still positive. Same $1,000 basis as T1. |

On any trigger: **unset `PROJECTX_ACCOUNT_ID` and `YANK_CONTRACTS` in
`deploy/systemd/trader-yank.service`, reload, restart.** Record the trigger, the trade
count and the P&L in an amendment to this document. Do not re-derive the threshold that
fired.

## 6. What does NOT trigger a shutdown

Stated explicitly so it cannot be revisited:

1. **The MES result (t = −2.002).** Different instrument, 1.75× the friction, SL/TP never
   calibrated for it. Its own seal §7.4 forbids this reading.
2. **The MNQ 2021–2024 backtest (t = 0.514).** Already-seen evidence, and a null.
3. **The N=11 confidence interval.** It is wide because N is 11. That is an argument for
   more evidence, not for less.
4. **A quiet period.** Low frequency is YANK's known characteristic, not a failure mode.
5. **This document's author disliking the odds.** Only §5 fires.

## 7. Reinstatement

A reverted YANK returns to the combine only via a **new** pre-registration carrying its own
decision rule. There is no automatic reinstatement and no "give it another month".

## 8. This seal does not supersede the ML rule

`preregistration_yank_sl2tp8_ml050.md`'s forward stop (disable ML at N≥20 if PF<0.90)
remains in force and operates on a different lever. Both can fire; neither blocks the other.

## 9. Operational hazard, recorded not fixed

`trader-yank.service` notes: *"YANK has NO autoroll (unlike MIM-NB) — update manually at the
quarterly roll (~2026-09-11 → MNQZ26) or orders will reject."* That is **~2 weeks away**.

A roll failure would produce rejected orders and a silent stall, **not** a P&L loss, so it
would not fire §5 — but it must not be mistaken for strategy failure, nor allowed to mask
the accrual §5 depends on. Flagged for action outside this seal.

## 10. Values fixed at seal time

| Item | Value |
|---|---|
| git HEAD at seal | `efa912d` |
| YANK live since 2026-06-01 | N=11, +$1,843.50, PF 3.664 |
| YANK since combine reset 2026-08-13 | **N=2, +$327.00** |
| YANK's own max drawdown from HWM | −$304.00 |
| Live rate | 0.125 trades/day |
| Account | Topstep 50K combine, 26556101, 2 contracts |
| Shutdown mechanism | unset `PROJECTX_ACCOUNT_ID` → SIM paper (trackers preserved) |
