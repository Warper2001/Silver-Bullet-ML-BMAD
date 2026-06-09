# Pre-Registration: MES ORB Reversion v2 — Wide-ORB + Opposite ORB Level Target

**Generated:** 2026-06-09
**Experiment ID:** mes-orb-reversion-v2
**Pre-registration commit:** (populate after `git commit`)
**Supersedes:** mes-orb-reversion-v1 (commit cfceffb)
**Status:** SEALED — study_mes_orb_reversion_v2.py does not yet exist at time of this document

---

## 1. Why v1 Failed and What v2 Changes

### v1 geometric analysis (from Gate 0 results + diagnostic run)

| Element | v1 | v2 | Impact |
|---|---|---|---|
| ORB min size | 5 pts | **20 pts** | Wider ORB → target naturally further from entry |
| ORB max size | 30 pts | 40 pts | Allow slightly larger ORB days |
| Target | VWAP centerline | **opp_orb_level** | VWAP is median 3.2 pts from entry → N=0 |
| Result | N=0 (primary) | — | Gate 0 FAIL trivially |

**The VWAP target failure (geometry, not edge):**

During the AM session (09:45–11:30 ET), session VWAP is still early in its accumulation and
sits median 3.2 pts from the ORB-area entry price. The min R/R filter (1.5 × stop = 15 pts
for stop=10) eliminated every single signal. 62% direction sanity (VWAP) vs 96% direction
sanity (opp_orb_level). The v1 primary was architecturally wrong — the target type, not the
entry signal.

**The narrow-ORB stop problem:**

Median ES ORB = 15 pts. ES 5-min intrabar H-L range ≈ 8–10 pts. A 10-pt stop on a 15-pt ORB
day is inside the typical single-bar swing. v1 grid confirmed: WR rises monotonically with
stop width (23.8% → 35.5% → 46.7%) as stop clears intrabar noise. But wider stops reduce N
below the 20-trade minimum.

**The fix:** Require ORB ≥ 20 pts. On a 20-pt ORB day:
- opp_orb target = ~17–20 pts from entry (entry is a few pts inside the ORB_HIGH/LOW)
- With stop=10, R/R ≈ 1.7–2.0:1 — passes min_rr_mult=1.5 without filtering signals
- The 10-pt stop represents 50% of the ORB, leaving room to breathe
- Wider ORB days indicate higher opening volatility → rejection at extremes tends to be
  more forceful (institutional participation at the extremes is clearer)

**Disclosure of what was observed in v1 (and what was NOT):**

Observed in v1:
- ORB size distribution: median=15 pts, p75=21.2 pts, p95=33.6 pts
- Signal geometry (VWAP/opp_orb distances, direction sanity rates)
- Grid WR/PF trend: stop=12/opp_orb → WR=46.7%, PF=1.23, N=15

NOT observed before this pre-registration:
- WR or P&L of opp_orb_level strategy with ORB_MIN=20 filter
- Signal count or frequency with ORB_MIN=20 filter
- Any trade outcome with the v2 specification

The ORB_MIN_SIZE=20 value is derived from the ORB size distribution (p75=21.2), not from
observing P&L outcomes. This is a geometry correction, not a result-driven optimization.

---

## 2. Hypothesis

### H₁ (alternative)

An ORB false-breakout rejection fade on 5-min MES bars — **restricted to days when the
opening range is ≥ 20 pts** — with opposite ORB level as target, 10-pt stop, and volume
confirmation, produces positive expectancy in-sample (2025-05-01 → 2026-02-28):

- **EV > $0** per trade net of commission
- **PF ≥ 1.20**
- **WR ≥ avg_be_wr + 5pp**
- **N ≥ 20**
- **Median stop ≤ $150/contract** (10 pts × $5 = $50 ✓)
- **Worst-month avg ≥ −$50**

### H₀ (null)

Even on high-volatility ORB days (≥ 20 pts), ES mean reversion from ORB extremes fails. The
signal quality at the ORB boundary does not depend on the ORB size — false breakouts continue
at the same rate regardless of the opening range magnitude. Or: N is so small (<20) that no
reliable conclusion can be drawn at all.

---

## 3. Signal Definition (Frozen)

### 3a. Opening Range Filter (key change from v1)

- ORB period: same as v1 (09:30–09:44 ET, 3 × 5-min bars)
- **`orb_min_size_pts: 20.0`** — skip days with ORB < 20 pts
- `orb_max_size_pts: 40.0` — skip extreme gap days (v1 was 30; raised to 40 to avoid
  cutting off valid wide-ORB days near the 30-pt boundary)

### 3b. Entry Signal (unchanged from v1)

```
SHORT: bar.high >= ORB_HIGH  AND  bar.close < ORB_HIGH
LONG:  bar.low  <= ORB_LOW   AND  bar.close > ORB_LOW
```

Direction sanity check: skip if opp_orb_level target is on wrong side of entry.

### 3c. Confirmation Filters (unchanged from v1)

1. Time window: 09:45–11:30 ET
2. Volume: bar volume > 1.5× 20-bar rolling mean

### 3d. Trade Rules

| Element | Rule |
|---|---|
| Stop | Fixed 10 pts from entry (primary) |
| Target | **Opposite ORB level**: ORB_LOW for short, ORB_HIGH for long |
| Min R/R | 1.5× (naturally satisfied: ORB≥20 pts → opp target ≥ ~17 pts for most) |
| Hold max | 12 × 5-min = 60 min |
| Session close | 15:55 ET |
| One trade at a time | No concurrent positions |

---

## 4. Gate 0 Thresholds (same as v1)

| Criterion | Gate |
|---|---|
| EV per trade | > $0 |
| Profit factor | ≥ 1.20 |
| WR vs breakeven | ≥ avg_be_wr + 5pp |
| Median stop | ≤ $150/contract |
| N | ≥ 20 |
| Worst-month avg | ≥ −$50 |

**Sensitivity grid:**
- `stop_pts` ∈ {8, 10, 12}
- `orb_min_size` ∈ {10, 15, 20, 25} — shows v1→v2 progression
- Primary: stop=10, orb_min=20

---

## 5. Scope

- If v2 passes Gate 0 → combine-math path simulation → Gate 2 pre-registration (OOS holdout)
- If v2 fails Gate 0 → MES ORB Reversion declared exhausted; wait for S25 validation
  (decision rule: N≥20 AND 60 days from 2026-05-24 deployment → ~2026-07-23)
- GC CPI prospective test (Event 1: June 11) continues independently
- S25 (`tier2_streaming_working.py`, account 23884932) continues unchanged
- Sealed holdout ≥2026-03-01 stays sealed

---

## 6. Integrity Seal

| Item | Value |
|---|---|
| mes_orb_reversion_v2_config.yaml | (SHA-256 at commit) |
| Git HEAD at pre-registration | (populate after `git commit`) |
| study_mes_orb_reversion_v2.py | NOT YET WRITTEN |
