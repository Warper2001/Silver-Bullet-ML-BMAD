# Pre-registration — Option 2: MNQ overnight/Globex hold

**Date:** 2026-09-05
**Parent plan:** `_bmad-output/research_plan_post_r3_options_20260904.md`, Option 2.
**Sealed before running:** committed before any P&L number is observed. Design-only for now — the actual run is deferred until compute is free (Option 1b's sweep currently saturates all 4 cores; running this alongside would only slow both down for no benefit).

## Background

`project_edge_headroom_screen_20260615.md` flagged this directly: *"an MNQ event-fade or overnight hold (best geometry, no thin-tick slippage, fired only on scheduled high-vol days, uncorrelated with the intraday trend bet) is testable today on existing data"* — never run. Data check done today: the existing 1-min CSVs (`mnq_1min_2025.csv` / `_2026_ytd.csv`) cover the full 24h session (all UTC hours populated, with the expected dip at the CME daily maintenance window ~22:00–23:00 UTC), so no new purchase is needed — confirmed before drafting the grid below.

This is a **new strategy shape** — no existing live analog — so unlike Options 1a/1b there is no frozen baseline to hold constant. The design choice itself needs to be principled, not arbitrary.

## Hypothesis and its grounding

Equity index futures (and the underlying cash indices) have a well-documented overnight-return anomaly: a disproportionate share of the long-run drift in equity indices accrues during the overnight/closed-market session rather than during regular trading hours (a real, published market-microstructure fact, not something specific to this project's data). MNQ is a levered proxy for that same index. **Primary test: does simply holding long from RTH close to next RTH open capture that drift, net of MNQ's real costs, at a magnitude/consistency that clears Gate 0?** This is deliberately the *simplest* defensible entry rule — always-long, no filter — because it's the one with an independent, external reason to expect a real effect, rather than a filter invented to fit this project's own data.

## Frozen for this seal

- **Entry:** at the last RTH bar's close (15:59 ET), enter long 1 contract.
- **Exit:** at the first RTH bar's open (9:30 ET) the next trading day.
- **Excluded nights:** the night before/after a day inside a CME halt or the 4–5pm CT maintenance window (per seal §2.2 of the (now-closed) tick-infra project — excluded, not modelled, the same convention this project already uses); Friday-to-Monday (weekend) holds are *included* by default (they're a real, tradeable overnight-equivalent gap) but reported as their own row so a Friday-specific effect isn't hidden inside the aggregate.
- **Costs:** $0.72 exch+reg round turn (the same friction figure the tick-infra seal used) + commission, MNQ_PV=$2/pt, 1 contract.
- **The one swept "knob," if the primary clears Gate 0:** a day-type filter — scheduled high-vol days (FOMC/NFP/CPI) only, vs. every day (the primary/baseline). This is Gate 0's secondary question, not run unless the primary passes — running both at once and picking the better one after the fact is exactly the pattern this project's memory calls out as forbidden.

## Data range

Full available history, `mnq_1min_2025.csv` + `mnq_1min_2026_ytd.csv`, truncated at `HOLDOUT_CUTOFF` (2026-03-01) — dev window only, same discipline as Option 1b. No holdout access.

## Gate 0 — sealed before any run

1. **N floor:** every RTH session in the dev window with a valid next-day open qualifies (no discretionary filter on the primary test) — expect N in the several hundreds. Flag if materially short of that; a filter-free MNQ overnight test should never be N-starved the way the event-gated variant risked being.
2. **Profitability bar:** net PF > 1.15 (the project's standing weak-edge floor, per TSC-1/TSMOM-1) after real costs.
3. **Beats a random-direction null:** for each night, flip the sign with p=0.5 (200 draws) — this is the load-bearing check. It directly separates "MNQ genuinely drifts overnight" from "MNQ drifts up generally over this sample and any fixed-direction rule would show a profit" (the exact TSMOM-1 mechanism-check trap — a bull-run sample rewards *any* always-long rule, overnight or not).
4. **Fat-day robustness:** ex-top-5-nights PF must still exceed 1.0 (a small number of index-level shock nights carrying the whole result is a real but different finding — report it, don't let it silently pass as "robust drift").
5. **Friday/weekend nights reported separately**, not pooled silently — weekend gap risk is a different animal from a single weeknight.

**Verdict rule:** PASS only if 1–4 all hold on the primary (every-night, always-long) test.

## Stopping rule

One run on the primary test. PASS → a *separately pre-registered* cycle may test the day-type filter (never both at once). FAIL → record and close, no re-sweep on this seal, consistent with every other closed candidate in this project.

## Deferred: when this runs

Compute-scheduled after Option 1b's sweep completes and frees the box's 4 cores — this is a single vectorized pass over daily bars (GAP-1-speed, seconds), not a `Tier2StreamingTrader` replay, so once it runs it will be fast; the wait is purely about not contending with Option 1b for CPU, not about this test's own cost.
