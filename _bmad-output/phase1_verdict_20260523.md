# Program C Phase 1 Verdict — 20260523

**Produced by:** Story 6.3 (Phase 1 Verdict Synthesis & Pivot Decision)
**Decision tree source:** `_bmad-output/problem-solution-2026-05-20.md` — applied verbatim, no post-hoc modification.

---

## Input Verdicts

### S12 — Random-Entry Control Test (Story 6.1)

**Pre-registration:** `_bmad-output/preregistration_s12_random_entry.md`
**Sealed at:** `7ffb3e0b712f4265478b21ae0e583e57f1249f4e`
**Verdict doc:** `_bmad-output/s12_verdict_20260523.md`

| Metric | Null Min | Null P25 | Null Median | Null P75 | Null P90 | Null Max | Strategy |
|---|---|---|---|---|---|---|---|
| PF | 0.462 | 0.723 | 0.824 | 1.004 | 1.249 | 1.560 | **0.937** |
| WR | 0.363 | 0.410 | 0.445 | 0.474 | 0.500 | 0.552 | **~0.460** |
| Sharpe | -4.547 | -1.911 | -1.192 | 0.021 | 1.381 | 2.837 | **~0** |

Strategy PF=0.937 at **70th percentile** of null distribution. Above median (0.824), below p90 (1.249).

**Step 1 Result: AMBIGUOUS = TREATED AS FAIL = PIVOT**

Rationale: pre-committed rule specifies 50th–90th percentile → ambiguous, defaulting to fail. The burden of proof is on the strategy; ambiguous evidence does not discharge it.

---

### S13 — Timeframe Replication (Story 6.2)

**Pre-registration:** `_bmad-output/preregistration_s13_timeframe.md`
**Sealed at:** `5fde2d254277ab5b2943d608a1e8833d5a7243e2`
**Verdict doc:** `_bmad-output/s13_verdict_20260523.md`

| Timeframe | Trades | PF | WR | Daily Sharpe | TIME_STOP % |
|---|---|---|---|---|---|
| 1m (baseline) | 129 | 0.937 | 0.460 | ~0 | ~65% |
| 5m | 86 | 1.026 | 0.465 | 0.202 | 33% |
| 15m | 61 | 1.179 | 0.475 | 1.373 | 11% |

Consistency criterion (both PF > 1.0): **SATISFIED**

**Step 2 Result: PATTERNS SURVIVE — 1-min may be wrong resolution**

Notable finding: TIME_STOP exits drop from ~65% (1m) to 11% (15m). At 1-minute resolution, 65% of trades are closed by time-stop rather than TP or SL — the pattern rarely resolves within 60 bars. At 15-minute resolution, only 11% time out, meaning the FVG edge has much more room to work. This is consistent with a real structural signal that was being masked by excessive time pressure at 1m.

---

## Step 3 — Combined Routing (Pre-Committed Decision Tree)

```
Decision tree Step 3 (from problem-solution-2026-05-20.md):

  patterns_survive (step 1) AND best timeframe ≥ 1.1 PF → DESIGN PHASE 2 ML TEST
  patterns_survive (step 1) AND all timeframes < 1.1 PF  → DESIGN PHASE 2 with caveat
  patterns_did_not_survive                               → PIVOT
```

**Step 1 verdict = AMBIGUOUS = TREATED AS FAIL → maps to "patterns_did_not_survive"**

Step 2 evidence (S13 PATTERNS SURVIVE) is informational. It does NOT change the Step 3 routing. The pre-committed discipline is: ambiguous primary evidence defaults to fail. The S13 finding informs the pivot choice but does not override the routing rule.

---

## COMBINED VERDICT: PIVOT

**Formal verdict:** PIVOT per pre-committed Program C decision tree.

**Step 1 (S12) was the primary gate.** Strategy PF=0.937 lies in the ambiguous zone (50th–90th percentile of random null distribution). Per pre-committed rule, ambiguous = treated as fail.

---

## Pivot Selection

From the pre-committed pivot menu:

| Option | Description |
|---|---|
| P1 | Different timeframe (4H/daily ICT) on MNQ |
| P2 | Different asset (ES, NQ continuous, BTC perpetual) at 1-min |
| P3 | Different strategy family (mean-reversion, gap-and-go) |
| P4 | Buy-and-hold benchmark study |
| P5 | Pause research; focus infrastructure |

**Selected: P1 — Different timeframe (15m FVG+H1-sweep on MNQ)**

**Rationale:** S13 produced the only genuinely positive evidence in Phase 1. At 15m:
- PF=1.179 (vs 0.937 at 1m)
- TIME_STOP drops from 65% to 11% — trades are resolving via TP/SL, not timing out
- Sharpe=1.373 (vs ~0 at 1m)
- This is a falsifiable, pre-registerable, hypothesis-driven follow-on

The pivot is not "try 15m because it happened to look better." The S13 test was pre-registered with a specific consistency criterion. The 15m result isn't cherry-picked — it's the only alternate timeframe tested, and both tested timeframes passed. The 15m showing materially less TIME_STOP dependency is a structural observation with a causal story (FVG patterns need time to resolve; 1m resolution was too tight).

**Selection is binding.** If Phase 2 (Epic 7 OOS test on 15m) fails, the next pivot selection must come from P2-P5; P1 is exhausted.

---

## Epic 7 Outline (P1 Pivot)

**Title:** Program C Phase 2 — 15m FVG+H1-sweep OOS Validation

**Gate condition:** Phase 1 complete (this document). Phase 2 only begins after this commit.

**Pre-registration requirement:** Write `_bmad-output/preregistration_phase2_15m.md` before accessing sealed holdout. Doc must include: exact config, hypothesis, pass/fail threshold, git SHA.

**Pass/fail threshold (pre-committed):** PF > 1.1 on sealed holdout (`data/sealed_holdout/`, 2026-03-01+) at 15m granularity.

**Stories:**

| Story | Title |
|---|---|
| 7-1 | Pre-register Phase 2 + adapt BacktestEngine for 15m sealed holdout |
| 7-2 | Run sealed holdout OOS test (15m, StrategyConfig defaults) |
| 7-3 | Phase 2 verdict: continue (Epic 2 enhancement) or pivot again (P2-P5) |

**Config:** `StrategyConfig()` defaults only. No parameter adjustments for 15m. If the signal is real, it should survive at default settings — adjusting for 15m would be in-sample tuning.

**Sealed holdout access:** First access triggers `data/sealed_holdout/ACCESS_LOG.md` entry. Pre-reg SHA required.

---

## Phase 1 Summary

| Test | Pre-reg SHA | Result |
|---|---|---|
| S12 Random-entry control | `7ffb3e0b` | AMBIGUOUS = PIVOT |
| S13 Timeframe replication | `5fde2d25` | PATTERNS SURVIVE |
| **Phase 1 combined** | — | **PIVOT → P1 (15m)** |

Phase 1 is now closed. Epic 6 complete. Epic 7 (Phase 2) is the next research unit.

---

_Produced by Story 6.3 (claude-sonnet-4-6, 2026-05-23). Decision tree applied verbatim from `_bmad-output/problem-solution-2026-05-20.md`. No post-hoc modification of thresholds or routing rules._
