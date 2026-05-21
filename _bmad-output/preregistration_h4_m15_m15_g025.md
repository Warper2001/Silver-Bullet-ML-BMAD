# Pre-Registration: S18 H4·M15·M15·g0.25 Holdout Test
**Registered:** 2026-05-21
**Authored by:** Alex (warper2001@gmail.com)
**Status:** ACTIVE — frozen at commit time. No amendments after the commit SHA is used to access the sealed holdout.

---

## Purpose

Rounds 1–3 of the Program C Phase 2a multi-timeframe factorial DOE identified the following architecture
as the strongest pre-cutoff candidate:

> **H4 liquidity sweep → M15 CHoCH confirmation → M15 FVG entry, with MIN_GAP_ATR_RATIO = 0.25**

On 2025 pre-cutoff data (289,230 bars), this produced PF = 1.6221 with N = 41 trades (IR = 10.39).
The gap filter tightening (0.15 → 0.25) was the dominant lever found across all three DOE rounds.
H4 sweep was the second-largest factor. M15 entry was confirmed as superior to M1 and M5.

This test answers the critical question:

> **Does the H4·M15·M15·g0.25 architecture produce a profit factor above the S12 random-entry baseline
> (PF > 1.1350) on the sealed holdout?**

The result determines whether to advance to Phase 2 ML meta-labeling on this architecture or pivot.

---

## Architecture Changes vs Phase 1 (S12)

Three changes from the Phase 1 frozen set. All other parameters are identical.

| Parameter | Phase 1 (S12) | S18 Value | Rationale |
|---|---|---|---|
| Sweep TF | H1 (1-hour) | **H4 (4-hour)** | H4 sweep was dominant factor in R1/R2 (spread 0.07–0.26 avg PF) |
| Confirm TF | None | **M15 CHoCH** | M15 CHoCH added consistent +0.05–0.10 PF across all gap levels in R3 |
| Entry TF | M1 (1-min) | **M15 (15-min)** | Entry TF was single largest factor (spread 0.13 in R1) |
| `MIN_GAP_ATR_RATIO` | 0.15 | **0.25** | Largest single lever: +0.51 PF lift (1.12→1.62) at cost of ~27% fewer trades |
| All other parameters | See table below | Unchanged | Frozen from Phase 1 pre-registration |

---

## Frozen Parameters

All parameters below are locked at commit time. No changes permitted after the SHA is used to access the holdout.

### Risk / Exit (unchanged from Phase 1)
| Parameter | Frozen Value |
|---|---|
| `SL_MULTIPLIER` | 5.0 |
| `TP_MULTIPLIER` | 6.0 |
| `ENTRY_PCT` | 0.5 (FVG midpoint) |
| `MAX_HOLD` | 12 M15 bars = 180 minutes |
| `MAX_PENDING` | 16 M15 bars = 240 minutes |

### Filters
| Parameter | Frozen Value |
|---|---|
| Direction | Bearish only |
| Tuesday | Blocked |
| Volatility regime | Block when H1 ATR percentile > 0.75 over 120-bar rolling H1 window |
| `MIN_GAP_ATR_RATIO` | **0.25** (H1 ATR relative) |
| `ATR_THRESHOLD` | 0.5 × M15 ATR (entry-TF ATR) |
| `MAX_GAP_DOLLARS` | $60.00 |
| ML filter | Disabled |
| LR regime filter | Disabled |

### Multi-Timeframe Cascade
| Layer | Parameter | Frozen Value |
|---|---|---|
| Sweep | TF | H4 (4-hour bars) |
| Sweep | Detection | 2-bar symmetric swing high radius; last H4 bar high > swing AND close < swing |
| Sweep | Source cap | H1_BAR_CAP = 3000 bars (prevents all-time-high creep) |
| Sweep | Window | Expires 24 hours after sweep bar timestamp (6 × sweep_tf_hours) |
| Confirm | TF | M15 (15-min bars) |
| Confirm | CHoCH definition | First M15 bar that closes below most recent M15 swing low by ≥ 0.3 × M15 ATR |
| Confirm | Swing detection | 2-bar symmetric radius, must be ≥ 2 bars old (confirmed at idx − RADIUS) |
| Entry | TF | M15 (15-min bars) |
| Entry | FVG definition | Bearish 3-bar: `c1.low > c3.high AND c2.close < c2.open` |
| Entry | Entry price | Midpoint of FVG gap: `c3.high + gap × 0.5`, snapped to 0.25 tick |

---

## Important Caveat: Expected Trade Count

On 2025 pre-cutoff data (289,230 bars, full year), S17 recorded N = 41 trades.
The sealed holdout covers 2026-03-01 to 2026-05-19 (75,081 bars, approximately 26% of 2025 volume).

**Expected holdout trade count: ≈ 10–15 trades.**

This is a low-sample test. The pre-committed decision rule below accounts for this:
- PF estimates from fewer than 15 trades carry high variance and should be treated as directional signals, not
  conclusive evidence.
- A `no_edge` verdict with N < 15 should not by itself trigger an architectural PIVOT without a secondary test.

---

## Test S18: Hypothesis and Decision Rule

### Hypothesis
> The H4·M15·M15·g0.25 strategy on the sealed holdout will produce PF > 1.1350
> (the 90th-percentile random-entry PF established by S12 on the same holdout window).

### Baseline Reference (from S12 — do not re-run)
- **S12 random baseline p90: 1.1350** ← primary threshold
- **S12 real strategy PF: 1.2154** (96 trades, H1 + M1 + g0.15) ← secondary reference
- S12 baseline used same bar-eligibility rules (Tuesday block, market hours, vol regime). Directly comparable.

### S18 Decision Rule (pre-committed, no exceptions)

| Condition | Verdict | Direction |
|---|---|---|
| N < 10 | `insufficient_sample` | Log result as directional only. No architectural decision. Run a secondary test with g0.20 (wider filter) before concluding. |
| N ≥ 10 AND PF ≤ 1.1350 | `no_edge` | H4·M15·M15 architecture has no holdout edge at g0.25. Evaluate whether g0.20 (PF=1.32, N=49 in-sample) deserves its own pre-registration, or PIVOT to a different architecture. |
| N ≥ 10 AND PF > 1.1350 AND PF ≤ 1.6221 | `edge_confirmed` | Holdout PF beats S12 random baseline. Proceed to Phase 2 ML meta-labeling on this architecture. |
| N ≥ 10 AND PF > 1.6221 | `edge_exceeds_insample` | Holdout PF exceeds the pre-cutoff PF. Unusually strong result — proceed to Phase 2 ML with high confidence, but note that in-sample PF = 1.6221 was the expected upper bound. |

### What Is Not Pre-Committed
- Adjusting `MIN_GAP_ATR_RATIO` after observing S18 results
- Re-running with a different date range after observing outcomes
- Interpreting `no_edge` as permission to test g0.30 or g0.35 on the holdout without a new pre-registration
- Changing the confirm TF after observing results
- Using S18 results to motivate any change to the Phase 1 frozen params (SL/TP/entry/vol regime)

---

## Implementation Reference

The holdout test script is `s18_h4_m15_m15_holdout.py`.
Run: `.venv/bin/python s18_h4_m15_m15_holdout.py --preregistration <SHA_OF_THIS_COMMIT>`

The script will:
1. Verify the SHA points to a commit containing this file
2. Append an entry to `data/sealed_holdout/ACCESS_LOG.md`
3. Load `data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv`
4. Run the H4·M15·M15·g0.25 cascade using the frozen parameters above
5. Apply the S18 decision rule and write the verdict to the ACCESS_LOG
6. Write a report to `data/reports/s18_<timestamp>.txt`

---

## Acknowledgement

By committing this document, the author pre-commits to all decision rules above.
Any deviation constitutes a methodology violation and must be disclosed in `data/sealed_holdout/ACCESS_LOG.md`.

*This document is intentionally difficult to amend — that is its purpose.*
