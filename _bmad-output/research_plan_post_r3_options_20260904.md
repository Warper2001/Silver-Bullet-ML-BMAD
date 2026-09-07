# Post-R3 Research Plan — Six Options, Prioritized

**Date:** 2026-09-04
**Trigger:** `/goal "plan these options out and validate them"`, following R3's §4 closure and the order-flow ballpark null result (both same day). Context: [[project_ticksim_r3_killed_20260904]], [[project_edge_headroom_screen_20260615]].

**Framing.** The edge-headroom screen already identified the binding constraint: MNQ's move/cost geometry is the *best* on the board (10.8×–78.7× headroom), but realized capture is ~2% of it. The live edges (MIM-NB, YANK, GAP-1) are validated but thin; four separate "find a new N" threads (carry, momentum, prop-firm imports, order-flow/tick-infra) closed FAIL in the last 48 hours. So options that squeeze more capture out of *existing* edges are ranked above options that look for a *new* edge, and cheap/existing-data options are ranked above options needing new data or new capital.

Every option here follows the project's standing Gate 0 pattern (see `option_b_gate0_verdict_20260703.md`, TSC-1, TSMOM-1): a sealed, pre-registered set of criteria — a profitability bar, a beat-random-null comparison, an N floor, and a fat-day/robustness check — decided *before* looking at results, one knob swept at a time, FAIL recorded honestly and closed, never re-swept without a fresh seal.

---

## Priority order

| # | Option | Data | New capital? | Effort | Priority |
|---|---|---|---|---|---|
| 1 | YANK / GAP-1 exit-horizon extension | existing | no | **low** | **1 — do first** |
| 2 | Overnight / Globex hold | existing | no | medium | **2** |
| 3 | Correlation-aware sizing (YANK↔MIM-NB) | existing | no | medium | **3** |
| 4 | Combine-vs-tail-capture vehicle fit | existing (qualitative) | n/a | low | **4** |
| 5 | Volatility risk premium | new (options data) | **yes** | high | 5 |
| 6 | COT positioning / calendar-seasonality | new | no | medium | 6 (low prior) |

---

## Option 1 — YANK / GAP-1 exit-horizon extension

**Hypothesis.** Both strategies exit on a fixed SL/TP multiplier or same-day close; the edge-headroom screen shows MNQ headroom scales 10.8×→78.7× from 5min→240min holds, and MIM-NB (the one strategy that already holds to EOD) is the highest-realized-capture edge live. A longer or trailing exit on YANK/GAP-1 should capture more of the available move without a new signal.

**Data.** 100% existing: `data/processed/mnq_1min_2025.csv` + 2026 YTD, `trades.db` for realized YANK/GAP-1 history, `backtest_gap_fade.py` / `tools/yank_gap_ceiling_backtest.py` and siblings as the harness.

**Method — one knob at a time.**
1. First knob: **hold-time extension** — replace the fixed exit with a longer fixed bar-count hold (sweep a short, pre-declared grid, e.g. 1×/2×/3×/4× current `max_hold_bars`), same entry logic, same instrument, same cost model. This isolates horizon from any other change.
2. Only if that clears Gate 0: a second, separately pre-registered cycle can test a trailing-stop variant (a genuinely different mechanism, not a free extra knob on this seal).

**Gate 0 (to be sealed before running).**
- Net PF > current live PF on the same window, with **N ≥ current live N** (no cherry-picked subwindow).
- Beats the 95th percentile of a random-hold-time null (same entry signals, hold time drawn randomly from the same grid) — guards against "longer holds just harvest MNQ's positive drift," the exact TSMOM-1 mechanism-check trap.
- Fat-day check: ex-top-3-days PF must still be > 1.0 (a longer hold that only works because of 3 outlier days is not a robust exit change).
- Direction-consistency across the swept grid (PF should move monotonically or near-monotonically with horizon, not spike at one arbitrary value — a spike is the one-lucky-cell pattern the order-flow ballpark check just caught).

**Effort:** ~1 day (existing data + existing harness; write the pre-registration, run the grid, write the verdict).

---

## Option 2 — Overnight / Globex hold

**Hypothesis.** From `project_edge_headroom_screen_20260615.md`: "an MNQ event-fade or overnight hold (best geometry, no thin-tick slippage, fired only on scheduled high-vol days, uncorrelated with the intraday trend bet) is testable today on existing data" — flagged three months ago, never run.

**Data.** Existing 1-min MNQ CSVs cover the Globex session; no new purchase needed.

**Method.** This is a *new* strategy shape (no existing live analog), so it needs its own entry-rule design before a knob sweep is possible — e.g., hold RTH-close → next RTH-open on a pre-declared trigger day-type (scheduled high-vol days: FOMC/NFP/CPI, or simply "every day" as the null-comparison baseline). One knob to sweep first: **which day-type filter**, compared against the do-nothing (every day) baseline.

**Gate 0.** Same shape as Option 1: PF/expectancy bar, beat-random-day-selection null, N floor (this will bind harder — scheduled high-vol days are ~34/yr per the earlier event-fade scout's finding — flag this risk up front, don't discover it after building the harness), fat-day robustness check.

**Effort:** ~2–3 days (new entry-rule design + pre-registration + backtest). **Known risk, flagged now:** if gated to scheduled event days only, N may fall short of a usable floor within one seal — the 2026-06-15 event-fade scout already hit this exact wall (~34 events/yr, "fatal validation timeline"). Consider sealing the every-day baseline as the primary test and the event-day filter as a secondary cut, not the other way around.

---

## Option 3 — Correlation-aware position sizing (YANK ↔ MIM-NB)

**Hypothesis.** `project_yank_mim_correlation_portfolio.md` already establishes the two edges are effectively uncorrelated. A sizing scheme that accounts for that (vs. today's fixed 1–2 contracts each) should improve portfolio Sharpe with zero new signal risk — provided it doesn't clip MIM-NB's fat-tail days, which `project_mim_nb_expectations_reconciled.md` documents as 3 tail days out of 163 carrying the edge.

**Data.** Existing: `trades.db` joint YANK+MIM-NB history.

**Method.** One knob: a **volatility-scaled allocation rule** (e.g. inverse-vol weighting recomputed on a fixed schedule) vs. the current fixed sizing, backtested on the same joint history. Explicitly report the fat-tail-day contribution under each scheme — a rule that improves average Sharpe by shaving MIM-NB's size going into its rare big days must be rejected regardless of the aggregate number (the reconciliation memo's central caution).

**Gate 0.** Portfolio Sharpe improvement over fixed sizing on the same trades (no signal change, so this is a pure re-weighting comparison, not a null-beat test); the fat-tail-day check is a **hard constraint**, not a soft one — any rule that reduces expected P&L on the top-3 MIM-NB days is disqualified outright regardless of aggregate Sharpe.

**Effort:** ~1–2 days (no new backtest engine — this is a reweighting of already-realized trade streams).

---

## Option 4 — Combine-vs-tail-capture vehicle fit

**Not a backtest — a strategic review.** MIM-NB's edge is tail-capture (3 fat-tail days of 163). A Topstep combine's MLL and drawdown rules are structurally about capital preservation against exactly the variance a tail-capture strategy needs to survive through. The 2026-07-06 blown combine and the subsequent removal of all floor/buffer gating (`project_combine_floor_gating_removed_20260729.md`) are both symptoms of this tension, not independent incidents.

**Method.** Lay out, concretely: (a) what MLL/drawdown headroom MIM-NB's realized worst-case drawdown actually needs to reach a tail day, using the real trade history; (b) whether the current combine size/rules provide that headroom; (c) alternative vehicles (self-funded, a different prop-firm ruleset, reduced size solely on the combine while running full size self-funded) and their tradeoffs. Deliverable is a decision memo, not a Gate 0 verdict.

**Effort:** ~half a day — this is synthesis of data already in hand, not new computation.

---

## Option 5 — Volatility risk premium

**Status:** parked candidate from the 2026-09-03 low-frequency deep-recon, never tested. Requires options data and **self-funded capital — not combine-eligible** (options aren't a Topstep-combine instrument). Genuinely orthogonal to everything currently live. Held at priority 5 because it requires new data spend and new capital exposure, both larger commitments than options 1–4.

**Effort:** high (new data source, new instrument class, new infra) — scope only after 1–3 resolve.

---

## Option 6 — COT positioning / calendar-seasonality

**Status:** untested but low prior — the 2026-09-03 recon rated both overlay-only/weak, not primary candidates. Do last, and only if research budget remains after 1–4.

---

## Next step

Starting Option 1 now: draft its pre-registration (hold-horizon grid, sealed Gate 0 criteria above) before touching `trades.db` or the 2026 P&L, per house rule.
