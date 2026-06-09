# Pre-Registration: MES ORB Reversion — 5-min False-Breakout Fade

**Generated:** 2026-06-09
**Experiment ID:** mes-orb-reversion-v1
**Pre-registration commit:** (populate after `git commit`)
**Instrument pivot:** MNQ/ES search declared exhausted after 12 families (commit `fb8d094`)
**Status:** SEALED — study_mes_orb_reversion.py does not yet exist at time of this document

---

## 1. Why This Study Exists

### The instrument diagnosis

Twelve strategy families tested on MNQ, zero viable. All 12 failures share one structural cause:
MNQ is a momentum-heavy, tech-concentrated instrument. Intraday VWAP extensions on MNQ
continue with 84% probability (confirmed in-house). No mean-reversion strategy can survive that
base rate regardless of bar resolution, confirmation stack, stop geometry, or target.

Web research and academic literature consistently describe ES/MES differently: the S&P 500
index futures exhibit documented intraday mean reversion, with studies showing statistical
mean-reversion edge at 6× round-trip transaction costs. Practitioner backtests report:
- ES 5-min ORB: 72.17% WR, PF=1.623, 115 trades over 6 months
- MES 15-min mean reversion: 62%+ WR with combine-compatible stops (5 pts = $25/contract)
- $26,000+ profit on a single standard ES contract from mean reversion in 2025-2026

**This is the first study targeting MES as the PRIMARY instrument.**

### Why ORB Reversion specifically

Prior ORB work (SORM, ORBM-1/2/3) tested ORB CONTINUATION on MNQ (momentum-with). Those failed
because the stop had to sit inside the ORB, inside the noise band of the retest. The stop
geometry was incompatible with the combine.

ORB REVERSION is the structurally opposite trade:
- Setup: price breaks the opening range, but the breakout FAILS and price reverses back inside
- Entry: the false-breakout bar at the ORB boundary (rejection candle — same pattern as HCVWAP v2)
- Stop: outside the rejection wick (no longer inside the retest noise band)
- Target: VWAP centerline (same dynamic target as HCVWAP v2)

This has never been tested in this research program. External evidence supports it specifically
on ES/MES.

### Connection to HCVWAP v2

HCVWAP v2 (commit `fb8d094`) validated the false-breakout rejection candle pattern as a real
entry signal on 5-min bars: the LONG side on MNQ showed WR=38.3%, PF=1.87. The architecture
(5-min rejection candle + VWAP centerline target) was the correct design. The failure was that
MNQ short side (WR=17.2%) destroyed the combined result due to MNQ's directional bias.

This study takes the same proven entry architecture (5-min false-breakout rejection) and applies
it to the ORB boundary on MES — an instrument where BOTH directions should work.

---

## 2. Hypothesis

### H₁ (alternative)

A mean-reversion fade of the opening range false breakout on 5-min MES bars — filtered by
volume confirmation within the 09:45–11:30 ET time window, with VWAP centerline target and
10-pt stop — produces positive expectancy in-sample (2025-05-01 → 2026-02-28):

- **EV > $0** per trade net of commission
- **PF ≥ 1.20**
- **WR ≥ weighted-average breakeven + 5pp** (avg_be_wr at ~3:1 R/R ≈ 25%; gate ≈ 30%)
- **N ≥ 20** on the filtered population
- **Median stop ≤ $150/contract** (10 pts × $5 MES = $50 ✓)
- **Worst-month avg ≥ −$50**

### H₀ (null)

The MES ORB reversion has no edge. Either: (a) ORB false breakouts on ES are rare enough that
N < 20 after all filters, or (b) ES mean reversion also fails post-2025 due to macro/vol regime,
or (c) the VWAP target geometry doesn't work on MES as it didn't on MNQ short side.

---

## 3. Signal Definition (Frozen)

### 3a. Opening Range (5-min bars)

- RTH bars resampled from 1-min: 09:30 → 15:55 ET
- **ORB period**: bars starting at 09:30, 09:35, 09:40 (3 × 5-min = 15 min)
- `ORB_HIGH = max(bar.high)` across those 3 bars
- `ORB_LOW  = min(bar.low)`  across those 3 bars
- `ORB_SIZE = ORB_HIGH − ORB_LOW`
- **Day skipped** if ORB_SIZE < 5 pts (degenerate flat open) or > 30 pts (gap/extreme vol)

### 3b. False-Breakout Rejection Entry (frozen)

```
SHORT signal (ORB high rejection):
  bar.high >= ORB_HIGH   (price PIERCED the ORB high intrabar)
  AND bar.close < ORB_HIGH  (close RETURNED inside the opening range)

LONG signal (ORB low rejection):
  bar.low  <= ORB_LOW    (price PIERCED the ORB low intrabar)
  AND bar.close > ORB_LOW   (close RETURNED inside the opening range)
```

**Entry**: close of the rejection bar.

**Direction sanity check**: skip if VWAP target is on the wrong side of entry
(VWAP ≥ entry for SHORT, or VWAP ≤ entry for LONG). Same fix applied in HCVWAP v2.

### 3c. Confirmation Filters

1. **Time window**: 09:45–11:30 ET only (post-ORB, AM session)
2. **Volume spike**: 5-min bar volume > 1.5× rolling 20-bar mean

No HTF ranging filter (the ORB boundary is a stronger structural anchor than a σ band; keeping
the filter count low to preserve N).

### 3d. Trade Rules

| Element | Rule |
|---|---|
| Stop | Fixed 10 pts from entry (primary) |
| Target | Session VWAP price at entry time (dynamic R/R) |
| Min R/R | Skip if VWAP distance < 1.5 × 10 = 15 pts |
| Hold max | 12 × 5-min bars = 60 min |
| Session close | 15:55 ET force-close |
| One trade at a time | No concurrent positions |

---

## 4. Gate 0 Thresholds

| Criterion | Gate | Notes |
|---|---|---|
| EV per trade | > $0 | Primary edge gate |
| Profit factor | ≥ 1.20 | |
| WR vs breakeven | ≥ avg_be_wr + 5pp | avg_be_wr from realized per-trade R/R |
| Median stop | ≤ $150/contract | 10 pts × $5 = $50 MES ✓ |
| N (filtered) | ≥ 20 | |
| Worst-month avg | ≥ −$50/trade | Variance guard |

**Sensitivity grid** (informational only, primary spec frozen above):
- `stop_pts` ∈ {8, 10, 12}
- `target` ∈ {vwap_centerline, opp_orb_level}
- Primary: stop=10, target=vwap_centerline

---

## 5. Data Observation Disclosure

**Clean pre-registration.** No backtest of MES ORB reversion has been run at time of this document.

Prior observations disclosed:
- HCVWAP v2 long side (MNQ, N=60): WR=38.3%, PF=1.87 — confirms false-breakout rejection
  candle pattern has real edge on 5-min bars. This is NOT a reason to pre-select direction;
  both long and short are tested here.
- External practitioner data (web research 2026-06-09): 72% WR on ES 5-min ORB strategies,
  62%+ WR on MES 15-min mean reversion. These are NOT from our data; cited only to justify
  the instrument pivot. Architecture is derived from this research, not optimized on our data.

---

## 6. Scope and Declaration

- This is the **first strategy targeting MES as primary instrument** in this combine search.
- If this study fails Gate 0, the next step is to assess whether S25 (if it validates N≥20 + 60
  days by 2026-07-23) can serve as the combine strategy directly.
- GC CPI prospective test continues independently (Event 1: June 11).
- S25 (`tier2_streaming_working.py`, account 23884932) continues unchanged.
- Sealed holdout ≥2026-03-01 stays sealed.

---

## 7. Integrity Seal

| Item | Value |
|---|---|
| mes_orb_reversion_config.yaml | (SHA-256 computed by git at commit time) |
| Git HEAD at pre-registration | (populate after `git commit`) |
| study_mes_orb_reversion.py | NOT YET WRITTEN at time of this document |

`study_mes_orb_reversion.py` is written AFTER this pre-registration commit — tamper-evident.
