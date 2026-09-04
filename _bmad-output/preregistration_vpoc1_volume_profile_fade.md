# Pre-Registration: VPOC-1 — Volume-Profile Value-Area Fade on MNQ

**Generated:** 2026-09-04
**Experiment ID:** vpoc-1
**Status:** SEALED — no backtest code exists at time of this document; `tools/backtest_vpoc1.py` is written AFTER this commit.
**Origin:** deep-recon `domain-top-prop-firm-futures-trading-strategy-2026-09-04` (commit `1e9d563`), conditional pick #1 (VWAP-anchored "auction market theory"), narrowed after discovering substantial overlap with this shop's already-killed HCVWAP v1–v3 line (see §1).

---

## 1. Why this is not a re-test of HCVWAP (already dead)

`preregistration_hcvwap.md`, `_v2.md`, and `_v3_longonly_oos.md` (2026-06-09, commit `e1e153f` for v3) tested a VWAP 2σ-band fade with a 4-condition confirmation stack, including a 15-min EMA9/EMA21-spread-vs-ATR "ranging" filter — functionally the same "only fade in balance, not in trend" idea the prop-firm research's "auction market theory" candidate proposes. **v3 (the clean OOS test) FAILED**: WR=30.0% vs. required 30.2%, PF=0.847. v2's own scope note declared: *"VWAP mean reversion is not viable for the Topstep $50K combine on MNQ/ES at any bar resolution or entry architecture tested."*

VPOC-1 tests a mechanically different construct, not a re-parameterization of HCVWAP:

| Element | HCVWAP (dead) | VPOC-1 (this test) |
|---|---|---|
| Reference level | Session VWAP (volume-weighted **running mean**) | Volume Point of Control — the price with the **most actual traded volume** in a lookback window (a distributional statistic, not a mean) |
| Band/edge | Statistical σ-band around the mean | Value Area (VAH/VAL) — the actual price range containing 70% of traded volume |
| Regime filter | 15-min EMA9/21 spread vs. ATR (a trend-strength proxy) | Classic Market-Profile "Open Type": does today's session open **inside or outside** the prior period's real traded value area |
| Target | VWAP centerline (moving) | POC (static for the lookback window) |
| Timeframe | Session-only | Adds a **weekly**-anchored VWAP as a frozen multi-timeframe confluence filter (not present in any HCVWAP version) |

If VPOC-1 also fails, that is a materially different and stronger conclusion than "HCVWAP failed" — it means the *volume-distributional* version of this idea fails too, not just the moving-average version.

## 2. Hypothesis

**H₁ (alternative):** When a trading session opens inside the value area established by the prior N sessions' real volume-at-price distribution (a "balance" open, per classic auction market theory / Market Profile), price that subsequently reaches the edge of that value area (VAH or VAL) has positive expectancy reverting toward the prior window's Point of Control, net of realistic MNQ costs, on 1-min RTH bars.

**H₀ (null):** The volume-distributional (POC/Value-Area) formulation fails for the same underlying reason HCVWAP failed — MNQ's persistent momentum regime dominates any mean-reversion construct at any resolution tested by this shop.

## 3. Data (already on hand — no new purchase)

- `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv` — dev window, full year, 24h Globex bars (`timestamp, open, high, low, close, volume, notional`).
- `data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv` — holdout, 2026-01-01 through last available bar (2026-06-11 as of this seal).
- **Dev window:** 2025-01-01 – 2025-12-31.
- **Holdout (untouched until the single Gate 1 evaluation run):** 2026-01-01 – latest available (~2026-06-11, ~5.5 months — a real limitation, flagged, not concealed).
- **Disclosed simplification — no L3/tick data used.** Alex noted mid-session that L3 order-book data can be brought in for deeper analysis. This first pass deliberately does not use it: a real volume profile is best built from trade-level price/volume, and this test instead allocates each 1-min bar's entire volume to one price bin at the bar's typical price `(H+L+C)/3` — the same simplification this shop already uses for VWAP construction. This is a real approximation of a true volume profile, not the thing itself. If VPOC-1 clears Gate 0, rebuilding the profile from L3 data is the natural next step before Gate 1 — logged as an open question, not done in this seal, to keep this test cheap and fast.
- **RTH-only.** Both the profile construction and the trading window use 09:30–16:00 ET bars only (weekdays), matching HCVWAP's own convention. Overnight/Globex volume is excluded from the profile — a disclosed simplification (see Open Questions), not a claim that overnight volume is uninformative.

## 4. Signal construction (frozen)

### 4a. Volume profile, POC, and Value Area

For each RTH session day, and for a lookback window of the `N` most recently **completed** RTH sessions (current session excluded — no lookahead):

1. Bin all 1-min bars' typical price `(H+L+C)/3` into fixed **5.0-index-point bins** (a round, practitioner-standard granularity for a >20,000-point instrument — not tuned to any backtest outcome; not swept).
2. Sum each bin's bar volume across the `N`-session window → the volume profile.
3. **POC** = the bin with maximum summed volume.
4. **Value Area** (standard 70% convention — fixed, not swept): starting from the POC bin, repeatedly add whichever adjacent bin (above or below the current VA range) has more volume, until cumulative included volume ≥ 70% of the window's total profile volume. VAH = top edge of the highest included bin; VAL = bottom edge of the lowest included bin.

### 4b. Regime classification ("Open Type")

For each session day, using the lookback-window profile computed from the *prior* `N` sessions:

- **Balance** if that session's first RTH bar's open price falls within `[VAL, VAH]`.
- **Discovery** otherwise — **no trade is taken that session under this signal.**

### 4c. Weekly VWAP (multi-timeframe confluence filter, frozen, not swept)

- Computed on **all** available bars (including overnight/Globex — weekly VWAP is conventionally a full-session statistic), reset each Monday.
- `weekly_vwap(t) = cumsum(typical_price × volume) / cumsum(volume)` since the most recent Monday 00:00 UTC.
- Used only as a binary directional gate at entry (see 4d) — never as a target, never swept.

### 4d. Entry (within "balance" sessions only)

- **Long:** RTH 1-min bar's `low ≤ VAL` **AND** that bar's `close ≤ weekly_vwap` at that bar → enter long at bar close.
- **Short:** RTH 1-min bar's `high ≥ VAH` **AND** that bar's `close ≥ weekly_vwap` at that bar → enter short at bar close.
- One trade at a time; no new entry while a trade is open; first qualifying bar each session only.

### 4e. Exit (frozen)

| Element | Rule |
|---|---|
| Target | The lookback window's POC (the same POC used for regime classification) |
| Stop | Fixed at `1.0 × (VAH − VAL)` of the lookback window, placed beyond the entry-side VA edge (e.g., long stop = `VAL − (VAH−VAL)`) — a **measured**, disclosed quantity derived from the profile's own width, not hand-tuned to backtest performance |
| Hold max | Force-close at 15:55 ET (session end), matching HCVWAP convention |
| Skip guard | Skip the session if `VAH − VAL` is degenerate (≤ 1 bin) or POC coincides with a VA edge |

### 4f. Costs and sizing

- **Cost:** $2.24/round-turn (this shop's established live MNQ cost model, reused verbatim from `preregistration_event_fade_prospective.md`).
- **Point value:** $2.00/point (MNQ).
- **Size:** 1 contract, fixed (matches this shop's Gate-0 convention).

## 5. The one swept knob ("one knob at a time")

**Lookback window `N`** (sessions), swept over the pre-declared grid **{1, 3, 5, 10}** on the dev window only. All other quantities in §4 are fixed/derived and disclosed as such — bin size, Value Area %, stop multiple, and the weekly-VWAP confluence rule are not tuned. The value of `N` that maximizes **dev-window profit factor** is locked and used, unchanged, for the single Gate 1 holdout evaluation.

## 6. Decision rule (frozen, two-gate)

**Gate 0 (dev window, 2025, in-sample), locked `N`:**

| Criterion | Gate |
|---|---|
| EV (avg net P&L/trade) | > $0 |
| Profit factor | ≥ 1.20 |
| WR vs. breakeven | ≥ realized breakeven WR + 5pp (`be_wr = 1/(avg_realized_R:R + 1)`, computed from the actual trade log — R:R varies per trade since both stop and target are profile-derived, not fixed points) |
| N (trade count) | ≥ 20 — below this, INCONCLUSIVE (too rare to evaluate), not FAIL |
| Worst-month avg P&L | ≥ −$50/trade (variance guard, informational unless breached badly) |

A Gate 0 FAIL (sufficient N, criteria not met) is terminal for this pre-registration: no re-sweep, no new lookback grid, no new bin size or Value Area %, no added instrument. A materially different construct (e.g., L3-based profile, RTH+overnight profile, cross-sectional across instruments) needs its own new pre-registration.

**Gate 1 (holdout, 2026-01-01 → latest available, spent once), same locked `N`:**

| Criterion | Gate |
|---|---|
| Profit factor | ≥ 1.10 (relaxed — short, ~5.5-month holdout) |
| WR vs. breakeven | ≥ be_wr + 3pp (relaxed) |
| N | ≥ 10 — below this, **INCONCLUSIVE**, not FAIL; wait for more live/backfilled data |

## 7. What this run will NOT do

- Will not use L3/tick data to build the profile in this first pass (see §3) — a disclosed approximation, not the real thing.
- Will not include overnight/Globex volume in the profile or trading window.
- Will not test any instrument other than MNQ.
- Will not vol-scale position size.
- Will not sweep bin size, Value Area %, or the stop multiple — only `N` is swept.
- Will not combine with, or attempt to rescue, the HCVWAP session-VWAP/σ-band/EMA-ranging construct — that mechanism is superseded, not reused.
- Will not re-litigate a Gate 0 FAIL by trying a different bin size or VA% after the fact.

## 8. Open questions logged in advance (not resolved by this test)

- Bin size (5.0 pts) and Value Area % (70%) are practitioner-standard conventions, not derived from or swept against this shop's own data — a PASS here is provisional on confirming these aren't accidentally load-bearing (a future robustness check, not a re-sweep of this seal).
- RTH-only profile construction excludes real overnight volume for a near-24h-traded instrument — may materially understate or misplace the true POC/VA.
- Allocating a whole bar's volume to one bin at its typical price is a coarse approximation of a real volume profile; L3 data (available per Alex's note) would let the profile reflect actual intrabar trade distribution — the natural next step if this clears Gate 0.
- The weekly-VWAP confluence filter is a binary gate, not swept or weighted; whether it helps, hurts, or is irrelevant is only visible as a post-hoc diagnostic (with-filter vs. without-filter trade counts/PF), reported for transparency but never used to re-tune the sealed rule.
- The short (~5.5 month) holdout window means Gate 1, even on a clean PASS, is a directional check only, not a statistical confirmation — same caveat this shop applies to every short-holdout test (TSC-1, HCVWAP v3).
