# Pre-Registration: S14 Unfiltered FVG Control Test
**Registered:** 2026-05-20
**Authored by:** Alex (warper2001@gmail.com)
**Status:** ACTIVE — frozen at commit time. No amendments after the commit SHA is used to access the sealed holdout.

---

## Purpose

S12 and S13 (Phase 1) validated the H1-sweep + FVG filtered strategy and returned `design_phase2_ml_test`.
Before committing to Phase 2 ML meta-labeling, this test answers a prerequisite question:

> **Does removing the H1 sweep requirement improve or destroy the FVG edge on the sealed holdout?**

In-sample analysis showed that the unfiltered FVG baseline (no H1 sweep) produces 754 trades/year at PF 1.240
vs the filtered strategy's 191 trades/year at PF 1.033. However, that comparison was made on data already used
to develop and evaluate the filtered strategy. This test resolves the question cleanly on untouched holdout data.

The result determines Phase 2 direction:
- If unfiltered FVG edge exists and exceeds the filtered strategy → **pivot to high-frequency variant** (new program)
- If unfiltered FVG edge exists but does not exceed filtered → **stay the course, proceed to Phase 2 ML**
- If unfiltered FVG has no edge above random → **Phase 2 ML is the only viable path**

---

## What Changes vs Phase 1 (S12/S13)

**One change only:** The H1 liquidity sweep requirement is removed. All other parameters are identical to the Phase 1 frozen set.

| Parameter | Phase 1 Value | S14 Value |
|---|---|---|
| H1 sweep required | Yes (active sweep within 6 H1 bars) | **No — removed** |
| All other parameters | (see table below) | Unchanged |

---

## Frozen Parameters

All parameters below match `preregistration_phase1.md` exactly and are locked:

### Risk / Exit
| Parameter | Frozen Value |
|---|---|
| `SL_MULTIPLIER` | 5.0 |
| `TP_MULTIPLIER` | 6.0 |
| `ENTRY_PCT` | 0.5 (FVG midpoint) |
| `MAX_HOLD_BARS` | 60 bars |
| `MAX_PENDING_BARS` | 240 bars |

### Filters (all identical to Phase 1 except sweep)
| Parameter | Frozen Value |
|---|---|
| Direction | Bearish only |
| Tuesday | Blocked |
| Volatility regime | Block when H1 ATR percentile > 0.75 over 120-bar rolling window |
| `MIN_GAP_ATR_RATIO` | 0.15 |
| `ATR_THRESHOLD` | 0.5 |
| `MAX_GAP_DOLLARS` | $60.00 |
| H1 sweep requirement | **DISABLED — this is the only change from Phase 1** |
| ML filter | Disabled |
| LR regime filter | Disabled |

### FVG Detection (bearish)
Bearish 3-bar FVG: `c1.low > c3.high` and `c2.close < c2.open`, gap snapped to 0.25-tick, entry at FVG midpoint.

---

## Test S14: Unfiltered FVG on Holdout

### Hypothesis
> The unfiltered bearish FVG strategy (H1 sweep removed) on the sealed holdout will produce PF > 1.1350
> (the 90th-percentile random-entry PF established by S12), AND will produce more trades than S12's
> 96 trades on the same holdout window, confirming the FVG pattern has edge independent of the H1 sweep.

### Baseline Reference (from S12 — do not re-run)
S12 already established the random-entry baseline on this holdout:
- S12 random baseline p10: recorded in `data/reports/s12_20260520_153708.txt`
- S12 random baseline median: recorded in `data/reports/s12_20260520_153708.txt`
- **S12 random baseline p90: 1.1350** ← primary comparison threshold for S14
- **S12 real strategy PF: 1.2154** (filtered, 96 trades) ← secondary comparison

The S12 random baseline used the same bar-eligibility rules (Tuesday block, market hours, vol regime) as S14 will use. No sweep requirement was applied to random-entry bars in S12. The baseline is directly comparable.

### Exact Test Procedure

1. Append ACCESS_LOG with SHA and accessor label `"s14 script"` before reading any holdout bars.
2. Load `data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv`.
3. Run bearish FVG strategy with:
   - H1 structure still computed (for vol regime gate and H1 ATR percentile tracking)
   - H1 sweep detection: **skipped** (no sweep required before FVG entry)
   - All other filters applied identically to S12 real strategy
4. Record PF and trade count.
5. Compare to thresholds below.

### S14 Decision Rule (pre-committed)

| Condition | Verdict | Phase 2 Direction |
|---|---|---|
| S14 PF ≤ 1.1350 (≤ S12 p90 random) | **no_unfiltered_edge** | The FVG pattern has no edge without the sweep. Proceed to Phase 2 ML on the filtered 191-trade set. |
| S14 PF > 1.1350 AND S14 PF ≤ 1.2154 (≤ S12 filtered PF) | **edge_exists_stay_filtered** | Unfiltered FVG beats random, but the filtered strategy is still superior per-trade. Proceed to Phase 2 ML — the sweep adds value even if it reduces frequency. |
| S14 PF > 1.2154 (> S12 filtered PF) AND trade count ≥ 80 | **high_frequency_pivot_approved** | Unfiltered FVG beats the filtered strategy on holdout. Pivot: design a new program around the unfiltered FVG base with fresh pre-registration. Do not apply Phase 2 ML to the filtered set. |

**Minimum trade count for `high_frequency_pivot_approved`:** 80 trades. If PF > 1.2154 but trade count < 80, verdict is `edge_exists_stay_filtered` (insufficient sample to claim frequency advantage).

---

## What Is Not Pre-Committed (and therefore not permitted)

- Adjusting any frozen parameter after observing S14 results
- Re-running with a different date range after seeing S14 outcomes
- Interpreting `edge_exists_stay_filtered` as permission to pivot to high-frequency
- Running S14 again with modified FVG detection rules after seeing the first result
- Using S14 results to motivate any change to the Phase 1 pre-registered parameters

---

## Acknowledgement

By committing this document, the author pre-commits to all decision rules above. Any deviation constitutes
a methodology violation and must be disclosed in `data/sealed_holdout/ACCESS_LOG.md`.

*This document is intentionally difficult to amend — that is its purpose.*
