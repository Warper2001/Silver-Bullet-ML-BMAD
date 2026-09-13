# Pre-Registration: PDM-PHASE2 — Native-Metric Shadow Observability Trial

**Registered:** 2026-09-12
**Status:** SEALED at commit time. Append-only amendments.
**Parents:** `_bmad-output/preregistration_portfolio_decay_monitor_phase1_feasibility.md` (sealed `d223e0a`);
`_bmad-output/feasibility_memo_portfolio_decay_monitor_phase1_20260912.md` (`2b9fe9f`)

---

## 1. What Phase 1 forces this document to say before anything else

Phase 1's feasibility memo closed off the design this initiative started with: a single
win-rate-based decay indicator, shared across the four live strategies. It is not computable
— YANK and MIM-NB have no sealed win-rate baseline at all (they are PF-gated), GAP-1's only
win-rate figure is a Gate-0 in-sample threshold rather than a live-comparable baseline, and
Kraken Thursday-short has zero eligible trades and a Sharpe-based rule in any case.

The memo's closing observation was that a workable indicator would have to be expressed in
**each strategy's own native metric**. Following that thread to its conclusion changes the
shape of Phase 2 more than the original roadmap anticipated:

**Every one of the four live strategies already has its own sealed, native-metric halt
trigger, enforced independently of anything this initiative builds:**

| Strategy | Sealed halt/stop trigger | Source |
|---|---|---|
| YANK | ML disabled if live PF < 0.90 at N≥20; full revert to SIM if N≥30 AND PF < 1.00 | `preregistration_yank_sl2tp8_ml050.md`; `preregistration_yank_shutdown_rule.md` §5 T2 |
| MIM-NB | Halt if 30 completed trades show net PF < 0.70; also slippage >3× modeled, or replay mismatch vs. the sealed engine | `preregistration_mim_nb_live_deployment.md` §4 |
| GAP-1 | STOP (archive) if PF < 1.00 at N ≥ 30 | `preregistration_gap_fade_panic_open.md` |
| Kraken Thursday-short | PASS if Sharpe > 0.80 at N ≥ 30; accrual voided if cumulative absences > 3 | `preregistration_kraken_thursday_short_restart.md` |

There is therefore **no new decision rule left to invent**. A "portfolio decay monitor" that
tried to add a fifth, cross-strategy rule on top of these four would be exactly the kind of
new, hand-set threshold AGENTS.md's monitor policy forbids without its own sweep and citation
— and Phase 1 already showed there's no shared statistical vocabulary to derive one from
honestly.

## 2. What Phase 2 is, given that

**A read-only shadow tool that mirrors each strategy's own already-sealed native metric and
gate, in one place, on a fixed cadence — nothing more.** Its value is not a new statistical
capability; it is removing the need to separately re-derive "how close is each strategy to its
own sealed threshold" from four different documents. This is explicitly the aggregation
convenience named as the unmet job in the parent innovation-strategy document, now correctly
scoped down to what Phase 1 actually supports.

It also carries one operational lesson directly from this program's history: Thursday-short's
24-day gap went undetected because nothing was watching the watcher. Whatever computes this
mirror must itself report a heartbeat, or it can silently fail the same way and nobody would
know until manually checking — which is the exact class of protocol breach Thursday-short's
restart doc already legislates against for its own absence count (§4 there). This document
extends that same discipline to the shadow tool itself (§5.4 below).

## 3. Design — frozen now, one knob (the aggregation itself), no menu

For each of the four strategies, on each run, compute **exactly the statistic and window its
own sealed document already defines** — no new formula, no new window, no new threshold:

| Strategy | Statistic computed | Window | Source data | Gate mirrored |
|---|---|---|---|---|
| YANK | Realized PF | Cumulative live trades (`write_mode='realtime'`, `trader_id='trader-yank'`) | `data/trades.db` | N≥20 → PF<0.90; N≥30 → PF<1.00 |
| MIM-NB | Realized net PF | Cumulative live trades, `trader_id='trader-mim-nb'` | `data/trades.db` | N≥30 → PF<0.70 |
| GAP-1 | Realized PF | Cumulative live trades, `trader_id='trader-gap-fade'` | `data/trades.db` | N≥30 → PF<1.00 |
| Kraken Thursday-short | Realized Sharpe (weekly returns) + absence count | Trades on/after 2026-09-17 only, per the restart-void rule | `data/thursday_ts/trades.csv` | N≥30 → Sharpe>0.80 PASS; >3 absences → void |

`timestamp` parsed `format="ISO8601"` throughout, matching Phase 1's method. Output: one row
per run, per strategy, to a new file — **`logs/portfolio_decay_shadow.csv`** — columns
`{run_at, strategy, n_trades, metric_name, metric_value, gate_n, gate_threshold, trades_to_gate,
distance_to_threshold}`. No existing log, `trades.db`, or live-imported module is written to.

## 4. What this authorizes and what it does not

**Authorizes:** a new, standalone, read-only script (its own file, not touching
`strategy_core.py`, `auth_v3.py`, `models.py`, `trade_db.py`, execution modules, or anything
under `src/ml/`), run manually or via a new systemd timer that only ever appends to
`logs/portfolio_decay_shadow.csv`, for the duration in §5.

**Does not authorize:**
1. Any halt, disable, or config change triggered by this tool. Each strategy's own sealed
   mechanism remains the sole authority; this tool has no write access to `strategy_config.yaml`,
   no unit-file access, and no code path that touches order execution.
2. Any new cross-strategy statistic, composite score, or threshold not already sealed in one
   of the four documents in §1's table. If a genuinely new indicator is wanted later, it is a
   separate initiative with its own pre-registration and its own power gate — not an amendment
   here.
3. Wiring into any alerting channel (Phase 3 of the parent roadmap) — that requires its own
   pre-registration after this trial's gate (§5) is evaluated.
4. Treating a distance-to-threshold number this tool prints as a trading signal. It mirrors;
   it does not decide.

## 5. Shadow trial gate — fixed before the tool runs once

A classical statistical power gate (AGENTS.md's standing "power gate before any new strategy
test") **does not apply here and is deliberately not fabricated**: this tool estimates no new
parameter and tests no new hypothesis about the market. It recomputes four already-sealed,
deterministic formulas against a live-appended ledger. Applying a power calculation to that
would be theater, not diligence. What is gated instead is **reliability and continuity** —
the two ways a mirror tool can fail without anyone noticing.

**Duration:** 30 calendar days from first run, or until any one of the four strategies crosses
its own N-threshold gate in §3 (whichever comes first), capped at 2026-12-31 if neither occurs
first (mirrors the open-ended-accrual handling in the Thursday-short restart doc — INCONCLUSIVE,
not FAIL, if the cap is hit with no gate crossed).

**Pass criteria (all required):**
- **Correctness:** at least once per strategy during the window, the tool's computed
  `metric_value` and `n_trades` are hand-verified against an independent manual query of the
  same source data. Zero discrepancies tolerated — any mismatch is a defect, not a rounding
  note.
- **No boundary error:** when any strategy's `n_trades` crosses its own gate's `gate_n` during
  the window, the tool's row for that run correctly flags it (no off-by-one, no missed run).
- **Continuity:** the tool itself produces a run row on every scheduled cadence with no gap
  exceeding 3 missed runs — the same tolerance Thursday-short's own restart doc uses for its
  absence count, applied here to the monitor watching the monitor. A 4th consecutive missed
  run is a protocol breach, logged as such, and voids this trial's continuity claim (it does
  not silently continue as if nothing happened).

**Fail criteria:** any computation discrepancy found under manual verification, or continuity
broken per above, at any point in the window — recorded as a defect, tool revised, trial does
not restart the clock without a new pre-registration amendment.

**If the window closes on the duration/cap with pass criteria met but no strategy crossed a
gate:** the trial is scored PASS-RELIABILITY / UNTESTED-AT-BOUNDARY — it has shown the mirror
computes correctly and stays alive, but has not yet demonstrated it correctly flags a real
gate crossing. That is an honest, expected outcome given MIM-NB (N=23) is the closest strategy
to its own N=30 gate and may or may not cross it inside 30 days at its observed rate — it is
not grounds for claiming more than was tested.

## 6. What does NOT count as validating this tool

Stated explicitly, matching this program's convention of foreclosing predictable
re-litigation:

1. **A strategy's PF looking bad during the window.** This tool reports; it does not evaluate
   whether a strategy is failing. Only that strategy's own sealed document governs that call.
2. **A quiet 30 days with no gate crossed.** Per §5, that is UNTESTED-AT-BOUNDARY, not a
   pass on the tool's actual purpose — it is not evidence the tool would have flagged a real
   crossing correctly.
3. **Manual convenience alone.** "It's nice to have one file to check" is real value but is
   not what §5 gates — §5 gates whether that one file can be trusted.

## 7. Values fixed at seal time

| Item | Value |
|---|---|
| git HEAD at seal | `2b9fe9fda4fdec7d6503b235c5657085b65e0469` |
| Trial start | first run of the shadow script, to be logged as the first row's `run_at` |
| Trial cap | 2026-12-31 (30-day window or first gate crossing, whichever first) |
| Strategies in scope | YANK (N=5), MIM-NB (N=23, closest to its N=30 gate), GAP-1 (N=26), Kraken Thursday-short (N=0 eligible; restart accrual begins 2026-09-17) |
| Output file | `logs/portfolio_decay_shadow.csv` (new; no existing log or DB touched) |
| Halt authority | None — this tool has no write path to any strategy's execution or config |
