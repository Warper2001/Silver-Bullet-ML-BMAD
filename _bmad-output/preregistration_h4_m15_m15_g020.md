# Pre-Registration: S19 H4·M15·M15·g0.20 Holdout Test
**Registered:** 2026-05-21
**Authored by:** Alex (warper2001@gmail.com)
**Status:** ACTIVE — frozen at commit time. No amendments after the commit SHA is used to access the sealed holdout.

---

## Purpose

S18 (H4·M15·M15·g0.25) returned verdict `insufficient_sample` (N=2, PF=1.3121) on the sealed holdout.
Per the pre-committed S18 decision rule:

> *N < 10: Log result as directional only. No architectural decision. Run a secondary test with g0.20
> (wider filter) before concluding.*

This test is the prescribed secondary test. It widens the gap filter from 0.25 → 0.20 to obtain a
sufficient trade count. The architecture, cascade logic, and all other frozen parameters are unchanged.

> **H4 liquidity sweep → M15 CHoCH confirmation → M15 FVG entry, with MIN_GAP_ATR_RATIO = 0.20**

On 2025 pre-cutoff data (289,230 bars), H4·M15·M15·g0.20 produced PF = 1.3226 with N = 49 trades (IR = 9.26).

This test answers:

> **Does the H4·M15·M15·g0.20 architecture produce a profit factor above the S12 random-entry baseline
> (PF > 1.1350) on the sealed holdout?**

---

## Motivation: Why g0.20, Not a Different Architecture

The S18 `insufficient_sample` verdict does NOT indicate the architecture has no edge — it indicates the
g0.25 filter is too restrictive for the 2026-03-01 to 2026-05-19 holdout window to generate a testable
sample. g0.20 was the pre-specified fallback in the S18 decision rule (N=49 in-sample, ~26% more trades
than g0.25). Testing a completely different architecture at this point would constitute p-hacking.

---

## Architecture (unchanged from S18)

| Parameter | Phase 1 (S12) | S19 Value | Rationale |
|---|---|---|---|
| Sweep TF | H1 (1-hour) | **H4 (4-hour)** | Dominant factor in R1/R2 |
| Confirm TF | None | **M15 CHoCH** | Consistent +PF across all gap levels in R3 |
| Entry TF | M1 (1-min) | **M15 (15-min)** | Single largest factor in R1 |
| `MIN_GAP_ATR_RATIO` | 0.15 | **0.20** | Wider than S18 (0.25); prescribed by S18 decision rule |
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
| `MIN_GAP_ATR_RATIO` | **0.20** (H1 ATR relative) |
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

On 2025 pre-cutoff data (289,230 bars, full year), H4·M15·M15·g0.20 recorded N = 49 trades.
The sealed holdout covers 2026-03-01 to 2026-05-19 (75,081 bars, approximately 26% of 2025 volume).

**Expected holdout trade count: ≈ 12–13 trades.**

This is still a low-sample test. PF estimates from 12–13 trades carry high variance; treat any verdict
as a directional signal rather than conclusive evidence.

---

## Test S19: Hypothesis and Decision Rule

### Hypothesis
> The H4·M15·M15·g0.20 strategy on the sealed holdout will produce PF > 1.1350
> (the 90th-percentile random-entry PF established by S12 on the same holdout window).

### Baseline Reference (from S12 — do not re-run)
- **S12 random baseline p90: 1.1350** ← primary threshold
- **S12 real strategy PF: 1.2154** (96 trades, H1 + M1 + g0.15) ← secondary reference
- S12 baseline used same bar-eligibility rules (Tuesday block, market hours, vol regime). Directly comparable.

### S19 Decision Rule (pre-committed, no exceptions)

| Condition | Verdict | Direction |
|---|---|---|
| N < 10 | `insufficient_sample` | g0.20 is also too sparse. This is an architectural concern — the H4·M15·M15 cascade generates too few trades on the holdout window. Consider pivoting to a different architecture or extending the holdout window in a new pre-registration. |
| N ≥ 10 AND PF ≤ 1.1350 | `no_edge` | H4·M15·M15 architecture has no holdout edge at g0.20. Combined with S18 `insufficient_sample` at g0.25, this is a PIVOT signal. |
| N ≥ 10 AND PF > 1.1350 AND PF ≤ 1.3226 | `edge_confirmed` | Holdout PF beats S12 random baseline. Proceed to Phase 2 ML meta-labeling on this architecture at g0.20. |
| N ≥ 10 AND PF > 1.3226 | `edge_exceeds_insample` | Holdout PF exceeds the pre-cutoff PF (1.3226). Unusually strong result — proceed to Phase 2 ML with high confidence. |

### What Is Not Pre-Committed
- Adjusting `MIN_GAP_ATR_RATIO` after observing S19 results
- Re-running with a different date range after observing outcomes
- Interpreting `no_edge` as permission to test g0.15 on the holdout without a new pre-registration
- Changing the confirm TF after observing results
- Using S19 results to motivate any change to the Phase 1 frozen params (SL/TP/entry/vol regime)

---

## Implementation Reference

The holdout test script is `s19_h4_m15_m15_g020_holdout.py`.
Run: `.venv/bin/python s19_h4_m15_m15_g020_holdout.py --preregistration <SHA_OF_THIS_COMMIT>`

The script will:
1. Verify the SHA points to a commit containing this file
2. Append an entry to `data/sealed_holdout/ACCESS_LOG.md`
3. Load `data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv`
4. Run the H4·M15·M15·g0.20 cascade using the frozen parameters above
5. Apply the S19 decision rule and write the verdict to the ACCESS_LOG
6. Write a report to `data/reports/s19_<timestamp>.txt`

---

## Acknowledgement

By committing this document, the author pre-commits to all decision rules above.
Any deviation constitutes a methodology violation and must be disclosed in `data/sealed_holdout/ACCESS_LOG.md`.

*This document is intentionally difficult to amend — that is its purpose.*
