# Pre-Registration: S22 H1·M15·M1·g0.25 Holdout Test
**Registered:** 2026-05-21
**Authored by:** Alex (warper2001@gmail.com)
**Status:** ACTIVE — frozen at commit time. No amendments after the commit SHA is used to access the sealed holdout.

---

## Purpose

Program C Phase 2b DOE (S21) screened 8 H1+M1 combinations on 2025 pre-cutoff data.
The key finding:

> **H1+M1 without M15 CHoCH confirmation has PF < 1.0 at all gap levels.**
> **Adding M15 CHoCH + g0.25 filter rescues the signal: PF=1.1656, N=109.**

The M15 CHoCH acts as a structural filter — it confirms that the market has broken
below a swing low after the H1 sweep, before looking for an M1 FVG entry.

This test answers:

> **Does H1·M15·M1·g0.25 produce PF > 1.1350 (S12 p90 random baseline) on the sealed holdout?**

### Phase 2b Context

| Script | Architecture | Result |
|---|---|---|
| S18 | H4·M15·M15·g0.25 | insufficient_sample (N=2) |
| S19 | H4·M15·M15·g0.20 | insufficient_sample (N=5) |
| S20 | H1·{NoConf,M15}·M15·g{0.15-0.25} | PF 0.98–1.14, weak edge |
| S21 | H1·{NoConf,M15}·M1·g{0.15-0.30} | NoConf PF<1.0 everywhere; M15+g0.25 PF=1.1656 |
| **S22** | **H1·M15·M1·g0.25** | **← this test** |

---

## Architecture

| Parameter | Phase 1 (S12) | S22 Value | Rationale |
|---|---|---|---|
| Sweep TF | H1 (1-hour) | **H1 (1-hour)** | Unchanged — proven load-bearing (S14) |
| Confirm TF | None | **M15 CHoCH** | S21: M15 confirm required for positive edge with M1 entry |
| Entry TF | M1 (1-min) | **M1 (1-min)** | Unchanged — M1 FVG gives ~109 in-sample trades |
| `MIN_GAP_ATR_RATIO` | 0.15 | **0.25** | Dose-response peak: g0.25 maximises PF with sufficient N |
| All other parameters | See table | Unchanged | Frozen from Phase 1 |

---

## Frozen Parameters

All parameters below are locked at commit time. No changes permitted after the SHA is used to access the holdout.

### Risk / Exit (unchanged from Phase 1)
| Parameter | Frozen Value |
|---|---|
| `SL_MULTIPLIER` | 5.0 |
| `TP_MULTIPLIER` | 6.0 |
| `ENTRY_PCT` | 0.5 (FVG midpoint) |
| `MAX_HOLD` | 60 M1 bars = 60 minutes |
| `MAX_PENDING` | 240 M1 bars = 240 minutes |

### Filters
| Parameter | Frozen Value |
|---|---|
| Direction | Bearish only |
| Tuesday | Blocked |
| Volatility regime | Block when H1 ATR percentile > 0.75 over 120-bar rolling H1 window |
| `MIN_GAP_ATR_RATIO` | **0.25** (H1 ATR relative) |
| `ATR_THRESHOLD` | 0.5 × M1 ATR (entry-TF ATR) |
| `MAX_GAP_DOLLARS` | $60.00 |
| ML filter | Disabled |
| LR regime filter | Disabled |

### Multi-Timeframe Cascade
| Layer | Parameter | Frozen Value |
|---|---|---|
| Sweep | TF | H1 (1-hour bars) |
| Sweep | Detection | 2-bar symmetric swing high radius; last H1 bar high > swing AND close < swing |
| Sweep | Source cap | H1_BAR_CAP = 3000 bars |
| Sweep | Window | Expires 6 hours after sweep bar timestamp (6 × sweep_tf_hours) |
| Confirm | TF | M15 (15-min bars) |
| Confirm | CHoCH definition | First M15 bar that closes below most recent M15 swing low by ≥ 0.3 × M15 ATR |
| Confirm | Swing detection | 2-bar symmetric radius, must be ≥ 2 bars old |
| Entry | TF | M1 (1-min bars) |
| Entry | FVG definition | Bearish 3-bar: `c1.low > c3.high AND c2.close < c2.open` |
| Entry | Entry price | Midpoint of FVG gap: `c3.high + gap × 0.5`, snapped to 0.25 tick |

---

## Expected Trade Count

On 2025 pre-cutoff data (289,230 bars, full year), S21 recorded N = 109 trades for H1·M15·M1·g0.25.
The sealed holdout covers 2026-03-01 to 2026-05-19 (75,081 bars, approximately 26% of 2025 volume).

**Expected holdout trade count: ≈ 28 trades.**

This is a viable test sample. The N=10 minimum is expected to be comfortably exceeded.

---

## Test S22: Hypothesis and Decision Rule

### Hypothesis
> The H1·M15·M1·g0.25 strategy on the sealed holdout will produce PF > 1.1350
> (the 90th-percentile random-entry PF established by S12 on the same holdout window).

### Baseline Reference (from S12 — do not re-run)
- **S12 random baseline p90: 1.1350** ← primary threshold
- **S12 real strategy PF: 1.2154** (96 trades, H1·NoConf·M1·g0.15) ← secondary reference
- S12 baseline used same bar-eligibility rules (Tuesday block, market hours, vol regime). Directly comparable.

### S22 Decision Rule (pre-committed, no exceptions)

| Condition | Verdict | Direction |
|---|---|---|
| N < 10 | `insufficient_sample` | Cascade filters are too aggressive for this holdout window. Re-evaluate architecture. |
| N ≥ 10 AND PF ≤ 1.1350 | `no_edge` | H1·M15·M1 has no holdout edge at g0.25. The in-sample PF=1.1656 did not generalise. PIVOT. |
| N ≥ 10 AND PF > 1.1350 AND PF ≤ 1.1656 | `edge_confirmed` | Holdout PF beats S12 random baseline. Proceed to Phase 2 ML meta-labeling. |
| N ≥ 10 AND PF > 1.1656 | `edge_exceeds_insample` | Holdout PF exceeds pre-cutoff PF. Proceed to Phase 2 ML with high confidence. |

### What Is Not Pre-Committed
- Adjusting `MIN_GAP_ATR_RATIO` after observing S22 results
- Re-running with a different date range after observing outcomes
- Removing the M15 CHoCH confirm after observing results
- Testing g0.20 or g0.30 on the holdout without a new pre-registration
- Using S22 results to modify the Phase 1 frozen params (SL/TP/entry/vol regime)

---

## Implementation Reference

The holdout test script is `s22_h1_m15_m1_g025_holdout.py`.
Run: `.venv/bin/python s22_h1_m15_m1_g025_holdout.py --preregistration <SHA_OF_THIS_COMMIT>`

The script will:
1. Verify the SHA points to a commit containing this file
2. Append an entry to `data/sealed_holdout/ACCESS_LOG.md`
3. Load `data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv`
4. Run the H1·M15·M1·g0.25 cascade using the frozen parameters above
5. Apply the S22 decision rule and write the verdict to the ACCESS_LOG
6. Write a report to `data/reports/s22_<timestamp>.txt`

---

## Acknowledgement

By committing this document, the author pre-commits to all decision rules above.
Any deviation constitutes a methodology violation and must be disclosed in `data/sealed_holdout/ACCESS_LOG.md`.

*This document is intentionally difficult to amend — that is its purpose.*
