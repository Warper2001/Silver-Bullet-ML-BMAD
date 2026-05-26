# Research Queue: News Calendar Filter (S28 Candidate)

**Added:** 2026-05-25
**Status:** QUEUED — blocked until S25 reaches N≥20 live trades; runs after S27 (IFVG)
**Priority:** Medium — moderate implementation friction, uses public data

---

## Motivation

High-impact macro releases (FOMC, CPI, NFP, EIA, ISM) cause sharp, short-lived
volatility spikes that frequently invalidate FVG setups within the signal bar window.
These events are predictable (published calendars) and the market's reaction window is
narrow (typically ±15–30 minutes). A pre/post-event blackout could eliminate a subset
of low-quality setups without sacrificing signal frequency on clean days.

The volatility regime filter (H1 ATR > 75th pct → block) catches the **aftermath** but
not the pre-announcement window. A news filter acts upstream: if a high-impact event is
scheduled within the next N bars, skip FVG entry entirely for that session window.

Relevant to the time-stop problem: if limit orders are placed just before an event and
price runs hard through the zone (filling the IFVG but not the primary), the news
blackout would prevent the entry — avoiding a bad fill at the worst time.

---

## Concept

```
1. At each bar, check: is there a high-impact event scheduled within ±T minutes?
   Sources: ForexFactory economic calendar, FRED, CME event calendar (all public)

2. If yes → skip _detect_and_enter() for that bar (same as Tuesday exclusion logic)

3. Post-event cool-down: optionally block for T_post minutes after release
   (price needs time to establish a new structure; early entry often caught in the spike)

4. Calendar updates: fetch once per session start or use a static pre-loaded file
   for backtest; live system polls nightly
```

---

## Hypothesis (pre-registration candidate)

> Adding a ±30-minute blackout around high-impact (3-star) macro events improves
> Profit Factor vs S25 baseline by reducing low-quality fills around volatility spikes,
> without reducing trade count by more than 25%.

**H₀ (null):** News-filtered PF ≤ S25 baseline PF
**H₁ (alternative):** News-filtered PF > S25 baseline PF + 0.05

---

## Implementation Sketch

1. **New utility:** `news_calendar.py` — loads event schedule from CSV/JSON;
   exposes `is_blackout(timestamp, window_minutes=30) -> bool`.

2. **Static backtest data:** Download 2025 economic calendar (ForexFactory or FRED)
   as a CSV of `(datetime, impact_level, event_name)`. High-impact = 3-star or
   "High" tier events only.

3. **Trader integration:** One guard in `_detect_and_enter()` — same pattern as the
   Tuesday exclusion check.

4. **Config parameters:** `enable_news_filter: bool = False`,
   `news_blackout_minutes: int = 30` in `StrategyConfig`.

5. **Parity gate:** With `enable_news_filter=False`, backtest output byte-identical.

---

## Pre-Registration Requirements

Per methodology (AR6–AR8):
1. Commit planning doc (done)
2. When ready to test: run `prereg_seal.py` with exact event list and blackout window
   BEFORE running backtest
3. Must reference S25 (or S27 if IFVG is active) as comparison baseline

---

## Sequencing Constraint

Runs after S27 (IFVG fallback) is evaluated. Do not begin pre-registration until
S25 reaches N≥20 and S27 decision is made.

---

## Open Questions

1. **Event universe:** 3-star ForexFactory events only? Or broader (2-star+)?
   Wider net reduces risk of catastrophic fill; narrower net preserves frequency.

2. **Blackout symmetry:** Should pre-event and post-event windows be equal?
   Post-event spike resolution typically faster than pre-event uncertainty window.

3. **Interaction with volatility filter:** May be redundant for the biggest events
   (ATR spikes catch them anyway). Worth measuring marginal contribution.

4. **Data source for live system:** Public calendar APIs have latency/reliability
   risk. Nightly batch download is safer than real-time polling.
