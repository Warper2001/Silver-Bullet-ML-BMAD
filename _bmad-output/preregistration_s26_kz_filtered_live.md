# Pre-Registration: S26 Kill-Zone Filtered Live Paper Trading — H1·M15·M1·g0.25·KZ
**Registered:** 2026-05-21
**Authored by:** Alex (warper2001@gmail.com)
**Status:** ACTIVE — frozen at commit time.

---

## Purpose

S25 is live paper trading H1·M15·M1·g0.25 with no time-of-day filter (commit 69972c3). Kill-zone analysis on the S23 in-sample dataset (109 trades, 2025) revealed that 59% of trades fall outside ICT kill zones and produce PF=0.845 (net losing), dragging the overall PF from 1.6273 to 1.1656.

S26 is a **prospective subgroup analysis of S25 live trades**: the same `logs/tier2_filter_log.csv` log is parsed at evaluation close and filtered by the pre-committed rule below. No code changes to Tier2StreamingTrader are required. S25 runs unchanged.

This pre-registration seals the filter definition before any S25 live trade timestamps are examined. Any analysis of live results prior to this commit is a protocol violation and must be disclosed.

---

## S26 Filter Definition (pre-committed, exact)

```python
KZ_HOURS    = {10, 11, 14}   # 10:00–12:00 ET (NY AM) + 14:00–15:00 ET (NY PM)
BLOCKED_DOW = {0, 1}         # Monday (0) + Tuesday (1, already blocked in S25)

def is_s26_eligible(entry_ts_utc) -> bool:
    """True if this S25 trade counts toward the S26 evaluation."""
    import pytz
    ET   = pytz.timezone("America/New_York")
    et   = entry_ts_utc.astimezone(ET)
    return et.hour in KZ_HOURS and et.weekday() not in BLOCKED_DOW
```

A trade is S26-eligible if and only if its `entry_ts` falls within a kill-zone hour AND is not on Monday or Tuesday.

---

## Architecture (identical to S25 — no changes to Tier2StreamingTrader)

| Parameter | Value | Source |
|---|---|---|
| Sweep TF | H1 (1-hour) | S22 frozen |
| Confirm TF | M15 CHoCH | S22 frozen |
| Entry TF | M1 FVG | S22 frozen |
| `MIN_GAP_ATR_RATIO` | 0.25 | S22 frozen |
| `SL_MULTIPLIER` | 5.0 | Phase 1 frozen |
| `TP_MULTIPLIER` | 6.0 | Phase 1 frozen |
| `ENTRY_PCT` | 0.5 (FVG midpoint) | Phase 1 frozen |
| `MAX_HOLD_BARS` | 60 M1 bars | Phase 1 frozen |
| `MAX_PENDING_BARS` | 240 M1 bars | Phase 1 frozen |
| `VOL_REGIME_LOOKBACK` | 120 H1 bars | Phase 1 frozen |
| `VOL_REGIME_THRESHOLD` | 0.75 | Phase 1 frozen |
| Direction | Bearish only | Phase 1 frozen |
| Tuesday | Blocked (S25) | Phase 1 frozen |
| ML filter | Disabled | S24 verdict |
| **Kill-zone filter** | **10:00–12:00 ET + 14:00–15:00 ET** | **S26 pre-registration** |
| **Monday block** | **Blocked** | **S26 pre-registration** |

The kill-zone and Monday-block filters are applied **at evaluation time** to the S25 log — they are not enforced by the live Tier2StreamingTrader system.

---

## In-Sample Baseline (S23 labeled data, 2025 pre-cutoff)

Validated by `s26_kz_validate.py` (run 2026-05-21):

| Metric | All S25 trades | S26 eligible | S26 excluded |
|---|---|---|---|
| N | 109 | **50** | 59 |
| Profit Factor | 1.1656 | **1.6273** | 0.8448 |
| Win Rate (TP hit) | 20.2% | **28.0%** | 13.6% |
| Win Rate (pnl > 0) | 47.7% | **50.0%** | 45.8% |
| Annualized Sharpe | ~0.65 | **1.42** | -0.52 |
| Annual P&L (5 contracts) | +$5,242 | **+$8,142** | -$2,900 |
| Exit: TP / TIME / SL | 22 / 64 / 23 | **14 / 25 / 11** | 8 / 39 / 12 |

Kill-zone hour breakdown (S26 eligible, 2025 in-sample):

| Window | N | PF |
|---|---|---|
| 10:00–11:00 ET (NY AM) | 26 | 1.8711 |
| 11:00–12:00 ET (NY AM ext.) | 12 | 1.7710 |
| 14:00–15:00 ET (NY PM) | 12 | 1.2431 |

Day-of-week breakdown (S26 eligible):

| Day | N | PF |
|---|---|---|
| Wednesday | 20 | 1.2428 |
| Thursday | 15 | 2.0085 |
| Friday | 15 | 1.9050 |

---

## Hypothesis

> H1·M15·M1·g0.25 filtered to kill-zone entries (10:00–12:00 ET + 14:00–15:00 ET, Mon+Tue blocked) produces profit factor > 1.1350 (S12 p90 random baseline) over a minimum of 20 live paper trades on MNQ.

---

## Evaluation Window

- **Start:** Date of this pre-registration commit (S26 filter seals; S25 live log accumulates)
- **Minimum close:** N_live_filtered ≥ 20 AND 90 calendar days elapsed (whichever is later)
- **Maximum:** 180 calendar days from start date

The 90-day minimum (vs S25's 60-day) reflects lower live frequency: at the conservative rate of 0.12 filtered trades/day, N=20 requires ~169 calendar days. The 180-day maximum is the hard cap.

---

## S26 Decision Rule (pre-committed, no exceptions)

Evaluated after the evaluation window closes on the subset of S25 live trades matching `is_s26_eligible()`:

| Condition | Verdict | Action |
|---|---|---|
| N_live_filtered < 20 after 180 days | `insufficient_live_sample` | Kill-zone filtered strategy fires too rarely live. Re-evaluate: extend kill-zone windows OR add bullish direction. |
| N_live_filtered ≥ 20 AND PF ≤ 1.0 | `live_no_edge` | Kill-zone filter did not rescue the edge in live trading. **STOP. PIVOT to ORB or mean-reversion.** |
| N_live_filtered ≥ 20 AND 1.0 < PF ≤ 1.1350 | `live_marginal` | Edge present but below S12 random baseline. Extend evaluation by 30 days before deciding. |
| N_live_filtered ≥ 20 AND 1.1350 < PF ≤ 1.6273 | `live_edge_confirmed` | Live PF beats S12 random baseline. **Deploy kill-zone filter in Tier2StreamingTrader. Promote to full position size.** |
| N_live_filtered ≥ 20 AND PF > 1.6273 | `live_exceeds_insample` | Live PF exceeds 2025 in-sample filtered PF. **Deploy with high confidence. Begin S27 planning (frequency expansion).** |

---

## Evaluation Procedure

At evaluation close, compute S26 PF as follows:

```python
import pandas as pd

log = pd.read_csv("logs/tier2_filter_log.csv", parse_dates=["entry_ts"])
log["s26"] = log["entry_ts"].apply(is_s26_eligible)   # function defined above
s26_trades = log[log["s26"]]

gross_profit = s26_trades.loc[s26_trades["pnl"] > 0, "pnl"].sum()
gross_loss   = abs(s26_trades.loc[s26_trades["pnl"] < 0, "pnl"].sum())
live_pf      = gross_profit / gross_loss
n_live       = len(s26_trades)
```

`pnl` is the realized P&L from TradeStation SIM execution prices (not theoretical backtest prices).

---

## What Is Not Pre-Committed

- Changing the kill-zone hours after observing live `entry_ts` distribution
- Adding Wednesday to the block list after observing live DOW results
- Using S26 live PF to select a different filter and redeploy without a new pre-registration
- Claiming "S26 confirmed the edge" after fewer than 20 filtered trades
- Modifying S25 Tier2StreamingTrader parameters to force trades into kill zones

---

## Monitoring

Live trades log to `logs/tier2_filter_log.csv`. S26-eligible trades can be counted at any time using `is_s26_eligible()` without violating the pre-registration, provided no parameter changes are made in response to interim results before the evaluation window closes.

Historical reference:
- S22 holdout (H1·M15·M1·g0.25, unfiltered): PF=1.2742, N=15
- S21 in-sample (H1·M15·M1·g0.25, unfiltered): PF=1.1656, N=109
- S26 in-sample (H1·M15·M1·g0.25, KZ+Mon-blocked): PF=1.6273, N=50

---

## Implementation Reference

No Tier2StreamingTrader code changes required for S26 data collection.

For deployment of the kill-zone filter in live trading (only after `live_edge_confirmed` or `live_exceeds_insample` verdict):
1. Add `KZ_HOURS = {10, 11, 14}` constant to `src/research/tier2_streaming_working.py`
2. In `_detect_and_enter()`: gate entry on `is_s26_eligible(bar.timestamp)`
3. Pre-register the deployment as S27 before activating

---

## Acknowledgement

By committing this document, the author pre-commits to all filter definitions and decision rules above. Any deviation must be disclosed in `data/sealed_holdout/ACCESS_LOG.md`.

Validation script: `s26_kz_validate.py` (run 2026-05-21, output in `data/reports/s26_kz_validate_20260521_155159.txt`)

*This document is intentionally difficult to amend — that is its purpose.*
