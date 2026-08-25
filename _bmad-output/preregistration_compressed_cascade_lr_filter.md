# Pre-Registration: LR Counter-Trend Regime Filter on the Compressed-Cascade Candidate

**Status:** SEALED — frozen at commit time. No modifications after commit.
**Date:** 2026-08-25
**Authored by:** Claude Code, at Alex's direction (background session)
**Lineage:** Child of `preregistration_yank_compressed_cascade.md` (sealed 445a9ba, PR #50).
Reuses that experiment's Phase 1-passed candidate (M15 sweep / M5 CHoCH / M1 FVG,
`BASELINE_CONFIG` in `yank_compressed_cascade_phase1.py`) unmodified. Does not re-litigate
Phase 1 or restart Phase 2 — this experiment rides on the same trade stream Phase 2 already
produces and asks a narrower question about it.
**Touches:** nothing live. No modification to `strategy_config.yaml`, `strategy_core.py`,
`yank_streaming_working.py`, or the existing `yank_compressed_cascade_phase2_tracker.py` /
`data/yank_compressed_cascade/phase2_trades.csv` ledger. `trader-yank` is not restarted by
this document or anything it authorizes. No orders are placed by anything this document
authorizes — paper/shadow only, same as the parent experiment.

---

## 0. What question this answers

On 2026-08-24, Alex asked whether the compressed-cascade candidate's live YANK LR regime
filter (counter-trend, `fast_len=390, slow_len=1950` — the config actually running in
`src/research/yank_streaming_working.py::LRRegimeFilter`, loaded from
`models/xgboost/lr_regime_config.json`) would have improved its results, and then asked for
a grid search including recent weeks. That grid search (9 combinations of
`fast_len ∈ {195, 390, 585} × slow_len ∈ {975, 1950, 2925}`) was run **exploratorily**, on
data through 2026-08-24 — which includes the live Phase 2 prospective window (sealed
2026-08-19, N=8/30 at the time). Results:

| fast_len | slow_len | n_kept | n_dropped | PF | WR% | Net |
|---|---|---|---|---|---|---|
| 195 | 975 | 347 | 54 | 1.123 | 48.1% | $8,228.25 |
| 195 | 1950 | 336 | 65 | 1.126 | 47.9% | $7,998.50 |
| **195** | **2925** | **337** | **64** | **1.213** | **49.3%** | **$12,952.00** |
| 390 | 975 | 350 | 51 | 1.052 | 47.4% | $3,550.00 |
| 390 | 1950 (live YANK config) | 355 | 46 | 1.137 | 48.7% | $9,073.75 |
| 390 | 2925 | 355 | 46 | 1.065 | 47.6% | $4,408.75 |
| 585 | 975 | 331 | 70 | 1.082 | 48.3% | $5,299.75 |
| 585 | 1950 | 353 | 48 | 1.132 | 49.0% | $8,974.25 |
| 585 | 2925 | 356 | 45 | 1.132 | 49.2% | $8,934.75 |

against a baseline (no LR filter, full history 2025 → 2026-08-24, N=401) of PF=1.152,
WR=48.9%, net=$11,894.75.

**Two things made `fast=195, slow=2925` stand out from the other 8 cells:**
1. It's the only grid point that beats the unfiltered baseline on *both* PF (1.213 > 1.152)
   and total net $ (+$12,952.00 > +$11,894.75) — the 64 trades it drops are net losers in
   aggregate (≈ −$1,057), not a mix that happens to flatter the ratio while bleeding dollars
   (which is what happened at e.g. `390/975`, where dropping 51 trades cut net P&L by
   two-thirds because several of the dropped trades were big winners).
2. **The live YANK LR config (390/1950) does not clear this bar** — filtered PF 1.137 is
   *below* the unfiltered baseline's 1.152. Six of the nine grid cells underperform doing
   nothing at all.

**This experiment exists to find out whether `195/2925` is real or is grid-search noise.**

## 1. The honest origin, and why that matters

`fast=195, slow=2925` was **not derived from first principles** — it is the best of 9 grid
cells, chosen post-hoc, on a window that includes the exact data currently serving as the
parent experiment's own prospective OOS test. This is precisely the failure mode this
project's methodology exists to catch (see `feedback_derive_dont_assert_one_knob` and
`feedback_iteration_loop_pattern` in project memory: no hand-set threshold values without a
sweep-and-freeze step, and no restricting to a favorable subset and calling it validated).

Nine cells is a mild multiple-comparisons problem, not a severe one — but it is not zero.
Nothing about the exploratory backtest in §0 constitutes evidence this configuration will
keep working. The only thing that can answer that is a **genuinely fresh** sample the grid
search never saw.

## 2. The candidate, precisely

**One knob, frozen:** `LRChannelRegimeDetector(fast_len=195, slow_len=2925)`, counter-trend
polarity (block a bearish signal when the regime label is `DOWN`; pass on `UP` or
`SIDEWAYS` — identical semantics to the live `LRRegimeFilter.allows()` in
`yank_streaming_working.py`), applied as an additional gate on top of the unmodified
compressed-cascade candidate (`BASELINE_CONFIG`: M15 sweep, M5 CHoCH, M1 FVG,
`min_gap_atr_ratio=0.426`, `bearish_only=True`, all other `StrategyConfig` fields at current
YANK values — see the parent seal §1).

No further tuning of `fast_len`/`slow_len`/polarity is authorized under this seal. If this
result is ambiguous or fails, the next step is a PIVOT, not a re-grid.

## 3. Implementation — reuses the parent's trade stream, adds no new risk

The parent Phase 2 tracker (`yank_compressed_cascade_phase2_tracker.py`) already runs the
full unmodified cascade daily against `logs/yank_shadow_parity.csv` and logs every
prospective trade (entry_ts ≥ 2026-08-19) to `data/yank_compressed_cascade/phase2_trades.csv`.
This experiment does **not** duplicate that trade generation or touch that ledger. Instead:

- A new script, `yank_compressed_cascade_lr_filter_tracker.py`, runs the identical cascade
  (`_precompute_gates` / `_run_cascade` from `yank_compressed_cascade_phase1.py` — same
  import, zero duplicated logic) against the same shadow-parity bars.
- For every trade with `entry_ts >= SEAL_TS` (this document's seal timestamp, **not** the
  parent's — see §4), it computes the LR regime at entry with `fast=195, slow=2925` and
  records **both** the trade and whether the LR filter would have kept or dropped it, to a
  new ledger: `data/yank_compressed_cascade/lr_filter_trades.csv`
  (fields: `entry_ts, exit_ts, direction, entry_price, exit_price, exit_reason, pnl, lr_kept`).
- Idempotent on `entry_ts`, matching the parent's and this project's established
  double-logging fix (natural-key dedup).
- Runs on the same daily cadence as the parent (`yank-compressed-cascade-phase2.timer`
  pattern: `OnCalendar=*-*-* 06:00:00 UTC`), as its own systemd unit
  (`yank-compressed-cascade-lr-filter.service`/`.timer`) so a bug in one tracker cannot take
  down the other. No orders placed; reads an existing log, writes only to the new ledger.

Because this reuses the parent's exact trade stream, the **fresh prospective trades are
exactly the parent Phase 2 ledger's own trades from this seal date forward** — this
experiment adds a `lr_kept` label to each one rather than generating a separate trade
sample. This means the two experiments' N counts stay comparable and nothing about this
seal can slow-walk or interfere with the parent's own N=30 decision.

## 4. Fresh-window requirement

**Seal timestamp for this experiment: `git log -1 --format=%aI` of this commit** (recorded
in the tracker script as `SEAL_TS`, set at implementation time to match the commit that adds
`yank_compressed_cascade_lr_filter_tracker.py`). Only trades with `entry_ts` on or after this
timestamp count toward the decision rule below. The 401 trades used in the exploratory grid
search (§0) — including the 8 already in the parent's live Phase 2 ledger — are **spent**:
none of them may be reused as "prospective" evidence for this seal, even though several of
them technically postdate the parent's own 2026-08-19 seal. This is stricter than the parent
required of itself, deliberately, to compensate for the multiple-comparisons origin in §1.

## 5. Decision rule

Let *baseline PF* = the parent Phase 2 ledger's own realized PF at the moment this
experiment reaches its own N (i.e., the unfiltered compressed-cascade result over the same
fresh trades this experiment observes — an internally consistent, non-hand-picked
comparator, per the "reuse an existing bar, don't assert a new constant" precedent in the
parent seal §4).

Minimum N before any verdict: **N = 30 fresh trades** logged by the LR-filter tracker
(matching the parent's own N target — no shortcut).

| Condition at N=30 | Verdict |
|---|---|
| Kept-subset PF > baseline PF, **and** kept-subset net $ > baseline net $ over the same 30 trades | PASS — `195/2925` survived a real fresh sample; worth a live-config change proposal |
| Kept-subset PF ≤ baseline PF, **or** kept-subset net $ ≤ baseline net $ | FAIL — treat as grid-search noise, do not re-tune, close this track |

Both PF *and* net $ must improve — a filter that raises PF only by dropping net-positive
trades (as 6 of 9 grid cells did in §0) does not count as a pass under this seal, matching
the origin diagnosis in §0.1 that distinguished `195/2925` from the other 8 cells in the
first place.

## 6. Stopping rule

No re-running, no re-gridding, no subgroup rescue if this fails — same rule as every other
sealed experiment in this project. If PASS, the next step is a proposal to Alex to change
`models/xgboost/lr_regime_config.json` for live YANK (a separate, explicit decision — this
seal does not authorize that change by itself). If FAIL, log the verdict next to this
document and close the track; live YANK's LR config (390/1950) is untouched either way.

## 7. What this does not establish

- Does not validate or invalidate the parent Phase 2 verdict — that decision rule (§5 of the
  parent seal) is untouched and governs independently.
- Does not test whether `195/2925` would have been *derivable* without the grid search — it
  is being given a fresh chance to prove itself, not vindicated retroactively for how it was
  found.
- At the base cascade's observed rate (~401 trades / ~20 months ≈ 1 trade every 1.5 days),
  N=30 fresh trades is likely a multi-month wait, same order as the parent's own Phase 2
  timeline.

---
_Implementation: `yank_compressed_cascade_lr_filter_tracker.py` (repo root), wired to
`yank-compressed-cascade-lr-filter.service`/`.timer`, committed alongside this seal._
