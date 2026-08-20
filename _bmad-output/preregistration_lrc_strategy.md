# Pre-Registration: LRC (Linear-Regression-Channel) Strategy

**Status:** SEALED — frozen at commit time. No modifications after commit.
**Date:** 2026-08-20
**Authored by:** party-mode round-table (Winston / Mary / Amelia / Dr. Quinn / Grumbal), at Alex's direction
**Lineage:** a genuinely new strategy, not a YANK amendment (Alex, 2026-08-19: "this should
be considered a new strategy... start from scratch"). Carries over one structural piece from
the compressed-cascade candidate (`preregistration_yank_compressed_cascade.md`) — the M15
sweep / M5 CHoCH / M1 FVG cascade — but every other parameter, the regime gate itself, and
the optimization were built independently on 2026 data only.
**Touches:** nothing live. No modification to `strategy_config.yaml`, `strategy_core.py`,
any deployed bot's config, or `backtest_engine.py`. No trader restarted.

---

## 0. How this candidate was found

Dev-phase discipline per Alex's explicit direction on 2026-08-19/20: no OOS gate during
exploration, 2026 calendar-year data only, aggressive/iterative. Three grid searches ran in
sequence, each widening around what the previous one found:

| Run | Cells | Best sturdy result |
|---|---|---|
| v1 | 216 | lookback=20/5min/slope, N=172, PF=1.311 |
| v2 | 648 | lookback=20/5min/slope, N=77, PF=1.573 (gap widened); lookback=100/15min/slope, N=39, PF=1.855 (new) |
| v3 | 1,512 | **lookback=100/15min/slope, N=39, PF=1.855 — same cell, same numbers, now rank 1** |

The lookback=100/15min config's exact reproduction across v2→v3 despite the grid widening
around it is the reason this graduated to a pre-registration instead of more grid search — a
grid-search fluke does not typically survive being the target of the *next* widening.

A random-null test (S12 methodology, `yank_lrc_null_test.py`) was then run on this config and
its lookback=150 sibling:

| Config | N | PF | Null median / p90 | Percentile | Verdict |
|---|---|---|---|---|---|
| lookback=100 | 39 | 1.855 | 0.819 / 1.431 | 97th | PATTERNS SURVIVE |
| lookback=150 | 30 | 1.811 | 0.874 / 1.321 | 96th | PATTERNS SURVIVE |

Full detail: `data/reports/lrc_grid_search_20260819*.csv`, `data/reports/lrc_null_test_20260820.txt`.

## 0.1 The honest evidentiary caveat — read before trusting the percentile

This is **not** the same strength of evidence as the compressed cascade's Phase 1 result
(100th percentile). That test evaluated one architecture, pre-specified before any data was
touched. **This candidate's parameters were selected from a 1,512-cell grid search run on the
same 2026 data the null test then ran on.** A random-entry control built from the winning
cell's own calibrated entry rate does not correct for having picked that cell out of 1,512
candidates in the first place — some cell in a fishing expedition this size is expected to
look good by chance alone. The repeated survival across three widened grids (not just one
lucky cell in one grid) is the strongest piece of evidence here, and it is real — but it is
weaker evidence than a pre-specified test, and should be treated that way.

---

## 1. The candidate, precisely

**Base structure** (unchanged from the compressed cascade, itself already null-tested):
H1→M15 sweep, M15→M5 CHoCH, M1 FVG entry. `h1_sweep_lookback=6`, `m15_confirmation=True`.

**New component — linear regression channel**, `src/research/regression_channel.py`: rolling
OLS over `regression_lookback` bars of the resampled `regression_timeframe` series, gating
entries by `slope <= 0` (not in a strong uptrend — consistent with `bearish_only`).

**Primary candidate:**

| Parameter | Value |
|---|---|
| `regression_lookback` | 100 |
| `regression_timeframe` | 15min |
| `gate_mode` | slope (band_k unused in this mode) |
| `sl_multiplier` | 5.0 |
| `tp_multiplier` | 8.0 |
| `min_gap_atr_ratio` | 0.35 |
| `max_gap_atr_ratio` | 0.426 (unchanged from YANK's post-gap-ceiling-fix value) |

**Documented sibling / sensitivity check, not a second primary:** `regression_lookback=150`,
otherwise identical. Included because its near-identical performance (N=30, PF=1.811, 96th
pctile) is part of the evidence that the 100-lookback result isn't an isolated lucky cell —
not because both are being pursued as separate candidates going forward.

All other `StrategyConfig` fields frozen at `yank_lrc_grid_search.py::BASE_CONFIG`:
`entry_pct=0.5, atr_threshold=0.5, max_gap_dollars=60.0, max_hold_bars=60,
max_pending_bars=240, contracts_per_trade=5, max_daily_loss=-750.0,
vol_regime_lookback=120, vol_regime_threshold=0.75, bearish_only=True,
commission_per_roundtrip=4.0, enable_kill_zone_filter=False, tuesday_exclusion=True,
ml_threshold=0.0 (disabled)`.

**Data used to derive this candidate:** `data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv`
merged with `logs/yank_shadow_parity.csv` (TS columns) — full 2026 calendar year through
2026-08-19. **No temporal split was used** — the grid search saw the entire dataset when
selecting this config. This matters for §4 below.

---

## 2. What's explicitly OUT of scope for this seal

- **No ML.** `ml_threshold=0.0`. Any meta-labeling model must be its own pre-registration,
  trained on trades produced by *this locked recipe* — not bundled into further tuning of
  the base recipe itself. Training ML on an unlocked, still-drifting config compounds the
  same overfitting risk the grid search already carries.
- **No further grid search on this recipe.** The parameters in §1 are frozen. A materially
  different idea (a new regime signal, a different base cascade, a different instrument) is
  a new pre-registration, not an amendment to this one.
- **No cherry-picking additional configs from the existing grid CSVs.** The two configs
  tested in §0 are the only ones this seal authorizes treating as candidates.

---

## 3. Hypothesis for the next phase

**H₀:** The primary candidate's live/prospective performance is consistent with the
in-sample result being a survivor of the 1,512-cell search, not a real edge.

**H₁:** The candidate maintains PF meaningfully above 1.0 on genuinely fresh data, consistent
with the repeated-survival evidence in §0 reflecting something real.

---

## 4. Phase 2 — prospective OOS (the only valid next test)

**No temporal holdout exists for this candidate and none can be manufactured after the
fact.** The grid search that selected `lookback=100/15min` already saw every day of 2026
data through 2026-08-19 — carving out, say, July-August as a "holdout" now would be
evaluating the winner against data the selection process already used. This is the same
principle §4 of the compressed-cascade seal was built on, applied to a case where it's even
more binding: there is no unspent slice of history left for this specific candidate.

- **Fresh prospective window only**, starting from this document's seal commit date forward.
- **Threshold: this null test's own p90 (1.431)** — reused, not a new hand-picked number,
  same "derive don't assert" rule the compressed-cascade seal used.
- **Minimum N: 30 trades** before any verdict — this project's standing floor.

### 4.1 Cost disclosure — read before committing to this path

The primary candidate fired **39 times in ~231 calendar days** of 2026 (≈0.17 trades/day).
At that rate, **N=30 fresh prospective trades takes roughly 178 days — about 6 months.** This
is materially slower than the compressed cascade's Phase 2 estimate (8–12 weeks). If Alex
wants a faster read than that, the honest options are:

- **(a)** Accept the ~6-month prospective clock, mirroring how every other validated
  candidate in this project earned its verdict.
- **(b)** Treat §0's grid-stability + null-test result as sufficient evidence to move to a
  smaller paper/shadow allocation now, accruing Phase 2 data through live observation rather
  than a dedicated wait — same mechanism as the compressed cascade's systemd-timer tracker,
  adaptable to this candidate's entry signal.
- **(c)** Look for a genuinely independent dataset (a different instrument, or MNQ data from
  before 2026 the grid never touched) as a faster — but not equivalent — substitute check.

This seal does not pick one of these; it exists so the choice is made explicitly rather than
by default.

---

## 5. Stopping rule

- No re-running the null test with adjusted seeds or calibration.
- No re-deriving `p_enter` after seeing prospective results.
- No subsetting prospective data to a favorable stretch.
- If Phase 2's prospective read disagrees with §0's in-sample read, Phase 2 wins — same
  precedence this project has used every other time historical and live reads conflicted.

---

## Freeze SHA

*(To be filled in by dev agent after `git commit` of this file)*

Commit SHA: `__FILL_AFTER_COMMIT__`

---

## Amendment 1 (2026-08-20 — §4.1's option (c), appended, §0-5 above left intact)

Alex asked whether running 2025 as a check would help, per §4.1 option (c). It does, with
one honesty caveat stated up front: **2025 is genuinely blind for the regression-channel
gate and this SL/TP/gap combo** (the grid search never touched 2025), but **not** a virgin
test of the base M15/M5 cascade underneath it, which was already validated on 2025 in the
compressed-cascade's own Phase 1. This is a partial-independence check, not a full one.

**Result** (`.venv/bin/python`, same `strategy_lrc.py` functions, `mnq_1min_2025.csv`):

| Config | 2026 (selection data) | 2025 (partially independent) |
|---|---|---|
| lookback=100 (primary) | N=39, PF=1.855 | N=30, PF=1.606, WR=50.0%, +$3,313.75 |
| lookback=150 (sibling) | N=30, PF=1.811 | N=34, PF=1.833, WR=52.9%, +$4,930.25 |

Both configs stayed solidly profitable on 2025. **lookback=150 in particular produced
almost the identical PF on both years (1.811 vs 1.833)** — a meaningfully different result
from the raw compressed cascade, whose PF inverted from 1.397 (in-sample) to 0.78 (2026
holdout) under the same kind of check. This is evidence the regression-channel regime gate
is doing real work, not just relocating the same overfitting to a new set of knobs.

**Not a substitute for §4's prospective Phase 2** — 2025 is prior-year data with partial
overlap in what's already been validated, not a fresh forward read, and the base cascade's
prior 2025 exposure means this can't fully clear the multiple-comparisons concern in §0.1.
But it materially raises confidence relative to §0 alone, and is grounds to revisit which
config is "primary": `lookback=150`'s cross-year stability (bigger 2025 N, near-identical
PF) arguably makes it at least co-primary with `lookback=100`, not merely a documented
sibling. Left as an open call for Alex rather than silently promoted — a designation change
this material belongs in a decision, not an amendment's prose.
