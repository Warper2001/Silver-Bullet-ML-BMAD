# Pre-Registration: S26v2 Directional Flow Continuity — H1·M15·M1·g0.25·DFC
**Registered:** 2026-05-29
**Authored by:** Alex (warper2001@gmail.com)
**Status:** ACTIVE — frozen at commit time.

---

## Purpose

S25 is live paper trading H1·M15·M1·g0.25 with no time-of-day filter (commit 69972c3). S26 pre-registered a kill-zone subgroup analysis (SHA a97b21c) using hours {10, 11, 14} + Monday blocked.

Post-registration diagnostic work (1-year backtest, May 2025–May 2026) revealed that S26 generates approximately 0.06 eligible trades/day — requiring 332–553 days to reach N=20. Additionally, the original mechanism stated for S26 ("kill zones are more volatile, therefore produce better signals") was found to be incomplete: S25 already contains two volatility-calibrated filters (ATR regime gate + `min_gap_atr_ratio=0.25`), and the highest-volatility session hour (09:00–10:00 ET) was excluded from S26 despite having positive performance.

This pre-registration defines S26v2, a redesigned subgroup hypothesis grounded in a more specific mechanism: **Directional Flow Continuity**. S26v2 is a prospective subgroup analysis of S25 live trades, identical in architecture to S26 — no code changes to Tier2StreamingTrader are required.

This document is committed **before any targeted S26v2-parameter backtest is run.** The diagnostic data described above motivated the theoretical question; it did not select the parameters. The specific window boundaries, day filter, and decision rule below are derived from market microstructure theory only.

---

## Mechanism Under Test: Directional Flow Continuity

S25 enters bearish FVGs after an H1 liquidity sweep + M15 CHoCH confirmation. The probability of bearish follow-through after entry depends on whether a sustained institutional seller is present to carry price toward the target.

**The Directional Flow Continuity hypothesis states:**

> Bearish FVG trades entered during hours when institutional order flow tends to be *directional* (continuation-mode) have higher probability of follow-through than trades entered during hours when order flow is *mean-reverting or contested* (price-discovery or digestion mode). The H1 sweep and M15 CHoCH conditions confirm structural setup quality; the time-of-day filter gates on the execution environment — specifically, whether the dominant flow at entry time is likely to reinforce or oppose the bearish thesis.

Two session windows are identified on first-principles grounds as candidates for directional flow environments in MNQ futures:

**Window A — Opening Range (09:00–11:00 ET) — Null Candidate:**
The NY open is the highest-participation hour of the RTH session. Directional institutions establish or unwind positions at the open. However, this window also contains post-event price discovery after overnight and pre-market moves, competing order flow as participants establish the opening range, and elevated mean-reversion risk as the initial directional spike is frequently faded. Window A is included in S26v2 as a **null candidate**: if the Directional Flow Continuity mechanism is correct, Window A should *underperform* Window B, because opening-range volatility is predominantly contested rather than directional.

**Window B — Afternoon Institutional Session (13:00–15:00 ET) — Primary Hypothesis:**
The afternoon NY session (post-lunch) begins after the midday low-participation period. By 13:00 ET, overnight gaps have been resolved, the opening range is established, and institutional participants (macro funds, systematic CTA desks, options market-makers managing delta ahead of close) are executing their remaining daily position targets. This environment is theoretically more likely to produce *directional* follow-through: a bearish impulse confirmed by CHoCH at 13:30–14:30 ET is being executed by participants who have the same directional commitment, not contested by participants still sorting out price discovery. Window B is the **primary hypothesis window**.

**The falsifiable prediction:** Window B PF > Window A PF > or ≈ S25 full-session PF. If Window A PF > Window B PF, the Directional Flow Continuity mechanism is rejected and neither window should be used as a live filter.

---

## S26v2 Filter Definition (pre-committed, exact)

```python
# S26v2 filter — Directional Flow Continuity
# Window A: 09:00–11:00 ET  (opening range, null candidate)
# Window B: 13:00–15:00 ET  (afternoon institutional session, primary hypothesis)
# Day filter: Tuesday blocked only (inherited from S25 live system)
# Monday is NO LONGER blocked (S26 blocked Monday; S26v2 removes this restriction
#   — the CHoCH confirmation filter provides sufficient directional quality control
#   and Monday's sweep/CHoCH setups are theoretically no different from other days)

WINDOW_A_HOURS = {9, 10}    # 09:00–11:00 ET
WINDOW_B_HOURS = {13, 14}   # 13:00–15:00 ET
DFC_HOURS      = WINDOW_A_HOURS | WINDOW_B_HOURS  # all S26v2 eligible hours

BLOCKED_DOW = {1}  # Tuesday (1). S25 live system blocks Tuesday at code level.

def is_s26v2_eligible(entry_ts_utc) -> bool:
    """True if this S25 trade counts toward the S26v2 evaluation."""
    import pytz
    ET  = pytz.timezone("America/New_York")
    et  = entry_ts_utc.astimezone(ET)
    return et.hour in DFC_HOURS and et.weekday() not in BLOCKED_DOW

def s26v2_window(entry_ts_utc) -> str:
    """Returns 'A', 'B', or 'none' for eligible trades."""
    import pytz
    ET  = pytz.timezone("America/New_York")
    et  = entry_ts_utc.astimezone(ET)
    if et.weekday() in BLOCKED_DOW:
        return "none"
    if et.hour in WINDOW_A_HOURS:
        return "A"
    if et.hour in WINDOW_B_HOURS:
        return "B"
    return "none"
```

A trade is S26v2-eligible if and only if its `entry_ts` falls in a DFC hour AND is not on Tuesday.

---

## Architecture (identical to S25 — no changes to Tier2StreamingTrader)

| Parameter | Value | Source |
|---|---|---|
| Sweep TF | H1 (1-hour) | S22 frozen |
| Confirm TF | M15 CHoCH | S22 frozen |
| Entry TF | M1 FVG | S22 frozen |
| `MIN_GAP_ATR_RATIO` | 0.25 | S25 frozen |
| `SL_MULTIPLIER` | 5.0 | Phase 1 frozen |
| `TP_MULTIPLIER` | 6.0 | Phase 1 frozen |
| `ENTRY_PCT` | 0.5 (FVG midpoint) | Phase 1 frozen |
| `MAX_HOLD_BARS` | 60 M1 bars | Phase 1 frozen |
| `MAX_PENDING_BARS` | 240 M1 bars | Phase 1 frozen |
| `VOL_REGIME_LOOKBACK` | 120 H1 bars | Phase 1 frozen |
| `VOL_REGIME_THRESHOLD` | 0.75 | Phase 1 frozen |
| Direction | Bearish only | Phase 1 frozen |
| Tuesday | Blocked (S25 live) | Phase 1 frozen |
| ML filter | Disabled | S24 verdict |
| **DFC window A** | **09:00–11:00 ET (hours {9, 10})** | **S26v2 pre-registration** |
| **DFC window B** | **13:00–15:00 ET (hours {13, 14})** | **S26v2 pre-registration** |
| **Monday block** | **REMOVED** (vs S26) | **S26v2 pre-registration** |

The DFC windows and Monday unblock are applied **at evaluation time** to the S25 log — not enforced by the live system.

---

## Contamination Disclosure

An exploratory per-hour backtest analysis of May 2025–May 2026 data was conducted prior to this pre-registration and motivated the theoretical question. The specific per-hour profit factor numbers from that analysis are **not cited here** and are **not used to select the window boundaries or day filter**. The window definitions (09:00–11:00 and 13:00–15:00) are derived from market microstructure theory as stated above.

This document is committed before any targeted S26v2-parameter backtest is run. Any backtest of the S26v2 parameters conducted after this commit SHA is a legitimate prospective validation. Any backtest conducted before this commit — including the exploratory hour analysis — is prior context only and cannot be used as evidence for or against S26v2.

---

## Hypothesis

> H1·M15·M1·g0.25 filtered to Directional Flow Continuity windows (Window B: 13:00–15:00 ET primary; Window A: 09:00–11:00 ET null candidate; Tuesday blocked) produces:
>
> 1. **Primary:** Window B (13:00–15:00 ET) PF > 1.20 over N≥20 live trades
> 2. **Directional:** Window B PF > Window A PF (mechanism test)
> 3. **Comparative:** Window B PF > S25 full-session PF at evaluation close

If condition 2 fails (Window A PF ≥ Window B PF), the Directional Flow Continuity mechanism is rejected regardless of Window B absolute PF.

---

## Evaluation Window

- **Start:** Date of this pre-registration commit (S26v2 filter seals; S25 live log accumulates)
- **Minimum close:** N_Window_B_live ≥ 20 AND 60 calendar days elapsed (whichever is later)
- **Maximum:** 180 calendar days from start date

Expected live frequency (Window B only, bearish direction): estimated 0.06–0.12 trades/day, yielding N=20 in approximately 167–333 days at conservative rates. The 180-day maximum is a hard cap; if N_Window_B < 20 after 180 days, verdict is `insufficient_sample`.

---

## S26v2 Decision Rule (pre-committed, no exceptions)

Evaluated after the evaluation window closes on S25 live trades matching `is_s26v2_eligible()`, subdivided by `s26v2_window()`:

| Condition | Verdict | Action |
|---|---|---|
| N_B < 20 after 180 days | `insufficient_sample` | Window B fires too rarely. Either (a) discard S26v2 and investigate frequency further, or (b) pre-register S26v3 with broader windows. |
| Window A PF ≥ Window B PF | `mechanism_rejected` | Directional Flow Continuity theory falsified — opening-range hour matches or beats afternoon. Do not deploy either window as a live filter. Record as evidence against time-of-day filtering for this strategy. |
| N_B ≥ 20 AND Window B PF ≤ 1.0 | `no_edge` | No edge in afternoon window. **STOP. Do not deploy.** |
| N_B ≥ 20 AND 1.0 < Window B PF ≤ 1.20 AND Window B PF > Window A PF | `marginal` | Edge present but below pre-committed threshold. Extend evaluation by 30 days before deciding. |
| N_B ≥ 20 AND Window B PF > 1.20 AND Window B PF > Window A PF AND Window B PF > S25 baseline PF | `confirmed` | **Deploy Window B (13:00–15:00 ET) filter in Tier2StreamingTrader. Pre-register deployment as S27.** |
| N_B ≥ 20 AND Window B PF > 1.50 AND above conditions met | `strong_confirmed` | **Deploy with high confidence. Begin Window A separate study pre-registration.** |

---

## Evaluation Procedure

At evaluation close:

```python
import pandas as pd
import pytz

log = pd.read_csv("logs/tier2_filter_log.csv", parse_dates=["entry_ts"])

log["s26v2_window"] = log["entry_ts"].apply(s26v2_window)

window_b = log[log["s26v2_window"] == "B"]
window_a = log[log["s26v2_window"] == "A"]

def pf(trades):
    gp = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gl = abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
    return gp / gl if gl > 0 else float("inf")

pf_b    = pf(window_b)
pf_a    = pf(window_a)
n_b     = len(window_b)
n_a     = len(window_a)

# S25 full-session baseline at evaluation time
pf_s25  = pf(log)

print(f"Window B (13:00-15:00): N={n_b}, PF={pf_b:.4f}")
print(f"Window A (09:00-11:00): N={n_a}, PF={pf_a:.4f}")
print(f"S25 full-session:       N={len(log)}, PF={pf_s25:.4f}")
print(f"Mechanism test (B > A): {'PASS' if pf_b > pf_a else 'FAIL'}")
```

---

## What Is Not Pre-Committed

- Changing window boundaries after observing live `entry_ts` distribution
- Adding additional day exclusions (e.g., Friday) based on live DOW results
- Splitting Window B into sub-windows (e.g., 13:00–14:00 vs 14:00–15:00) after seeing live data
- Claiming "S26v2 confirmed" after fewer than 20 Window B trades
- Using S26v2 live PF to retroactively justify different parameters without a new pre-registration
- Evaluating S26v2 before S25 has reached its own N≥20 decision gate

---

## Relationship to S26 and Research Queue

S26 (SHA a97b21c) remains active and unchanged. S26v2 runs in parallel on the same S25 live log — it is a different filter definition applied to the same data stream. Both studies are evaluated independently.

The broader research queue (S27 IFVG, S28 news calendar filter, S29 ES/MNQ divergence) remains blocked until S25 reaches N≥20 and its primary verdict is recorded.

---

## Monitoring

Live trades log to `logs/tier2_filter_log.csv`. S26v2-eligible trades can be counted at any time using `is_s26v2_eligible()` without violating the pre-registration, provided no parameter changes are made in response to interim results before the evaluation window closes.

---

## Prior Context (Not Used for Parameter Selection)

The following observations motivated the *theoretical question* but did not select the parameters:

- S26 exploratory backtest showed 0.06 DFC-eligible trades/day — infeasible evaluation timeline
- The original S26 volatility mechanism was found incomplete given S25's existing ATR filters
- The Directional Flow Continuity hypothesis was developed as an alternative theoretical explanation

Specific per-hour performance numbers from the exploratory analysis are intentionally omitted from this document.

---

## Acknowledgement

By committing this document, the author pre-commits to all filter definitions and decision rules above. Any deviation must be disclosed in `data/sealed_holdout/ACCESS_LOG.md`.

*This document is intentionally difficult to amend — that is its purpose.*

---

## Integrity Hashes

| Hash | Value |
|---|---|
| (a) strategy_config.yaml SHA-256 | `58c7aa652a5a6f1794d88cc2d065e5b1c73fc9fb1f29c1ad0f5ffb8738808569` |
| (b) strategy_core.py SHA-256 | `31010a51cbb93105e9f42f84ee83c560ed08a2bca2843edc561394805529be06` |
| (c) Git HEAD at seal time | `4c63a9f7de23cd053d183c12a4ee23cd52a9b020` |

*Hash (a): SHA-256 of `strategy_config.yaml` file bytes at seal time.*
*Hash (b): SHA-256 of `src/research/strategy_core.py` source bytes at seal time.*
*Hash (c): `git rev-parse HEAD` at seal time — commit this document to make it tamper-evident.*
