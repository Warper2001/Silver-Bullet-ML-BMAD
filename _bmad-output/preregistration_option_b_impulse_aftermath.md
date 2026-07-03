# Pre-Registration: Option B — Shock-Agnostic Impulse-Aftermath Scout (Gate 0)

**Date sealed:** 2026-07-03
**Author:** Alex (session run autonomously per Alex's instruction)
**Status:** SEALED before any P&L evaluation. Only the event-COUNT scan (dataset gate) was run before sealing; no simulated trade P&L has been computed at seal time.
**Lineage:** `_bmad-output/innovation-strategy-2026-07-03.md` (Option B), H-B1.

---

## Hypothesis (H-B1)

Anomalous no-scheduled-news impulse bars on MNQ 1-minute data exhibit exploitable drift or reversion at fixed horizons of 30–240 minutes, net of costs, at 60-second polling latency.

## Data

- IS (sweep + selection): `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv`, 2025-01-01 → 2025-12-31 (289,230 bars).
- OOS (single-shot confirmation): `data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv`, 2026-01-01 → 2026-06-11. **This range contains the sealed-holdout window 2026-03-01 → 2026-05-19.** Access follows `data/sealed_holdout/ACCESS_LOG.md` protocol: this document is the pre-registration; its commit SHA is recorded in the access log before the OOS run; the OOS test runs at most once, exactly as specified here, and the result is recorded regardless of outcome.

## Event definition (mechanical, news-free — frozen)

Rolling baseline over prior 120 one-min bars (shifted, min 60 valid): median bar range, median volume. A bar is an impulse event iff:

- range = high−low ≥ K × median_range, K ∈ sweep grid
- |close−open| ≥ 0.6 × range (directional body)
- volume ≥ 3.0 × median_volume
- bar's ET time NOT in excluded slots: 08:30–08:34, 09:30–09:31, 10:00–10:04, 13:00–13:02, 14:00–14:34, 15:59–16:01, 18:00–18:02 (scheduled releases / session artifacts)
- ≥ 60 min since previous accepted event (cooldown)

Direction = sign(close−open) of the impulse bar. No news feed, no post content, no Trump-specific features — by design (non-stationary adversary exclusion, per strategy doc).

## Trade mechanics (frozen)

Entry: market at next bar's open (realistic at 60s polling). Exit: close of first bar ≥ entry + H minutes; trades whose exit bar lands more than H+120 min after entry are dropped (weekend/gap artifacts). 1 contract, $2/pt, **$6 round-trip cost** (program-standard MNQ all-in).

## Sweep grid (frozen; selection on IS only)

K ∈ {4, 6, 8} × side ∈ {follow, fade} × H ∈ {30, 60, 120, 240} min = 24 cells.
Event counts at seal time (dataset gate, IS): K=4: 618, K=6: 235, K=8: 99 → gate PASSED (all K viable at N≥60).

**Selection rule:** among IS cells with N ≥ 60, pick highest net PF (tie-break: larger N). One selected cell; no post-hoc re-selection.

## Gate 0 PASS criteria (IS, all four required)

1. Selected cell net PF ≥ 1.20.
2. Selected PF > 95th percentile of 200 random-entry null samples (same N, same trade directions sequence, same H, entries drawn uniformly from eligible bars; seed 20260703).
3. K-robustness: at least one neighboring K (same side/H) with net PF ≥ 1.00.
4. Fat-day check: excluding the top 3 winning days, IS net total P&L > $0.

If any fails → **Gate 0 FAIL, Option B closed** (or PARKED if only criterion 4 fails, recorded as fat-tail-only edge).

## OOS decision rule (single shot, run only if Gate 0 passes)

Run the selected cell verbatim on 2026-01-01 → 2026-06-11. Report full-window stats plus the 01-01→02-28 (open) and 03-01→06-11 (sealed-window) segments separately.

- **PASS:** net PF ≥ 1.10 AND N ≥ 15 AND ex-top-3-days total > $0 → proceed toward TS SIM paper scout under a fresh deployment prereg.
- **MARGINAL:** net PF 1.00–1.10 → PARK; no deployment; prospective paper logging only.
- **FAIL:** net PF < 1.00 → Option B closed; no re-sweeps on this dataset.

No parameter may change between IS selection and the OOS run. No second OOS attempt regardless of outcome.

## Known-contamination disclosures (honesty section)

- The operator (this session) knows the 2025–2026 macro narrative in detail (Apr 2025 tariff shock, 2026 Iran war) from the innovation-strategy research, and knows from the Option C retrospective that YANK performed *better* on policy-shock days. Mitigation: the grid is small, fixed, and was written down before any P&L computation; the event definition is news-free; selection is by mechanical rule.
- 2026 data (Jan–Jun) was previously opened for YANK's 2026 holdout run (seal 138cab1) — it is not virgin for the *program*, but no impulse-aftermath rule has ever been evaluated on it.
- The Option C shock calendar will NOT be used in selection; it may be used descriptively in the report (e.g., what fraction of events fall on flagged days) AFTER the verdict is fixed.

## Implementation

`tools/option_b_gate0_scout.py` (committed alongside this document; the script implements this spec and was not run in P&L mode before sealing).

## Kill precedents acknowledged

FOMC event-fade PARK (thin N — avoided here, dataset gate passed), S26 net-cost death (+$1.09/ct gross vs $6 costs — costs are in the objective here), MIM-NB fat-day fragility (criterion 4 / OOS ex-top-3 check), iteration-loop pattern (single sealed sweep, no re-entry).
