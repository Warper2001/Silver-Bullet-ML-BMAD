# Pre-Registration: ES/MNQ Stat Arb — Short-Only Combine Strategy

**Generated:** 2026-06-09
**Experiment ID:** stat-arb-short-v1
**Pre-registration commit:** (populate after `git commit`)

---

## ⚠️ Data-Observation Disclosure

This strategy was designed and filtered AFTER running diagnostic studies on in-sample data
(2025-05-01 → 2026-02-28). The following findings directly informed parameter choices:

| Finding | Source Study | Impact on Strategy |
|---|---|---|
| 5-bar cumulative ES/MNQ divergence has positive EV at 1:1 R/R before direction split | `study_es_mnq_stat_arb.py` | Confirms the spread-reversion signal exists |
| Strong directional asymmetry: Long WR=47.9% (−$8.01), Short WR=57.4% (+$5.84) at THRESH=20 | `study_stat_arb_large_div.py` | Direction filter — short only |
| Short-only Gate 0 PASS: WR=58.0%, PF=1.27, freq=2.96/d, worst-mo=38.5% | `study_stat_arb_short_only.py` | Confirms feasibility of direction-filtered strategy |
| THRESH=20 at STOP=1.0× is positive EV; STOP=2.0× is negative EV across all thresholds | `study_stat_arb_short_only.py` | Stop multiplier locked at 1.0× |

**Structural justification for short-only direction:**
When MNQ outperforms ES, the driver is typically Nasdaq-specific (tech/AI catalysts,
single-stock moves propagating to the index). These outperformance episodes partially
revert as order flow normalises and the divergence gets arbitraged. When MNQ
underperforms ES, the driver is often ES-specific macro strength (defensive rotation,
energy, value) — MNQ genuinely does not participate, so there is no reversion signal
to fade.

**Consequence:** The in-sample backtest is confirmatory, NOT discovery.
The direction was selected after observing the in-sample asymmetry.
A strong in-sample result is expected and does NOT constitute independent validation.
**The OOS holdout (≥ 2026-03-01) is the primary validity test.**

---

## Hypothesis

**H₁ (alternative):** ES/MNQ stat arb short-only — shorting MNQ when its 5-bar
cumulative return divergence from beta-predicted ES exceeds +20 pts, with a stop at
1× divergence and target at 1× divergence recovery — generates positive expectancy
on MNQ 1-minute RTH bars with WR ≥ 56%, PF ≥ 1.20, frequency ≥ 1.0/day, and
worst-month WR ≥ 35% in the 2025-05-01 → 2026-02-28 in-sample period; and retains
PF ≥ 1.10 OOS.

**H₀ (null):** Fading MNQ outperformance of ES at a fixed divergence threshold has
no edge (PF ≤ 1.0 OOS); the in-sample short-direction edge is a noise artifact of
the exploration that produced the direction-selection decision.

---

## Strategy Logic (Frozen)

| Element | Rule |
|---|---|
| Instruments | MNQ (entry/exit) + ES (signal construction only) |
| Beta estimation | Rolling 60-bar OLS: β = Cov(ΔMNQ, ΔES) / Var(ΔES), clipped [0, 10], forward-filled |
| Divergence | 5-bar cumulative: div = Σ₅ΔMNQ − β × Σ₅ΔES |
| Direction | **SHORT ONLY** — enter when div > +20 pts (MNQ outperformed ES) |
| Entry | Close of the triggering bar (MNQ close price) |
| TP | Entry − divergence (MNQ gives back exactly 1× the divergence) |
| Stop | Entry + 1.0 × divergence (divergence widens further against us) |
| Stop cap | Skip trade if stop distance > $150/contract; enforce at entry |
| Hold max | 30 bars (~30 min); forced market-close if neither TP nor stop hit |
| Session close | 15:55 ET: force-close all open positions at close price |
| Trade sequencing | One trade at a time; no new entry while trade is active |
| RTH only | 09:30–15:55 ET; no overnight or pre-market |
| Daily halt | Halt new entries if session P&L ≤ −$300 for the day |

---

## Go / No-Go Decision Rules (Pre-committed, Immutable After Seal)

### Gate 1 — In-Sample Full Backtest (2025-05-01 → 2026-02-28)

Gate 0 has already passed on the primary spec. Gate 1 runs the same simulation with
the full combine-accounting overlay (trailing DD path, qualifying day count, daily halt).

| Criterion | Minimum | Action if below |
|---|---|---|
| Win rate | ≥ 56% | STOP — no OOS access |
| Avg net P&L/trade | > $0 | STOP |
| Profit factor | ≥ 1.20 | STOP |
| Max trailing DD (in-sample) | ≤ $1,500 | STOP |
| Frequency | ≥ 1.0 setups/day | STOP |
| N trades | ≥ 80 | STOP |
| Worst-month WR | ≥ 35% | STOP |
| Qualifying sessions / last 20 | ≥ 6 | WARNING |
| Largest single day as % of total P&L | ≤ 50% | WARNING |

### Gate 2 — OOS Holdout (≥ 2026-03-01, one-shot, requires Gate 1 pass)

| Criterion | Minimum |
|---|---|
| OOS WR | ≥ 53% |
| OOS avg net P&L/trade | > $0 |
| OOS profit factor | ≥ 1.10 |
| OOS PF retention vs in-sample | ≥ 75% |
| N OOS trades | ≥ 20 |

**OOS stopping rule (live, if deployed):** Halt combine trading if PF < 1.05
after first 25 OOS trades.

Accessing the OOS holdout before Gate 1 passes voids this pre-registration.

---

## Strategy Parameters Snapshot

```yaml
# ES/MNQ Stat Arb — Short-Only Combine Strategy Config
# Pre-registered 2026-06-09
# DO NOT MODIFY after the pre-registration commit SHA is recorded.
# Any parameter change requires a new pre-registration cycle.

strategy: es_mnq_stat_arb_short_only
version: v1

# ── data ranges ───────────────────────────────────────────────────────────────
in_sample_start: "2025-05-01"   # limited by ES data availability
in_sample_end:   "2026-02-28"
oos_start:       "2026-03-01"   # sealed holdout — DO NOT ACCESS before Gate 1 pass

# ── data files ────────────────────────────────────────────────────────────────
mnq_paths:
  - "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
  - "data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv"
es_path:  "data/processed/dollar_bars/1_minute/es_1min_2025_2026.csv"
oos_path: "data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv"

# ── signal construction ───────────────────────────────────────────────────────
beta_window:   60   # rolling bars for OLS beta estimation (MNQ_chg ~ ES_chg)
spread_window:  5   # bars over which cumulative divergence is summed

# ── entry filter ──────────────────────────────────────────────────────────────
direction:        short        # ONLY short: fade MNQ outperformance of ES
threshold_pts:    20.0         # enter when 5-bar cumulative div > +20 pts (MNQ overperformed)
stop_cap_usd:    150.0         # skip trade if stop > $150/contract

# ── trade geometry ────────────────────────────────────────────────────────────
stop_mult:        1.0          # stop at 1× divergence beyond entry (dir = short)
tp_mult:          1.0          # TP: MNQ recovers 1× divergence back to entry
hold_max_bars:   30            # forced exit after 30 bars (~30 min) if neither TP/stop hit

# ── session timing ────────────────────────────────────────────────────────────
rth_start:        "09:30"      # ET: RTH open
session_close:    "15:55"      # ET: force-close all positions at or after this bar

# ── instrument economics ──────────────────────────────────────────────────────
mnq_point_value:  2.0          # $/point per contract
commission_rt:    4.80         # $/contract round-trip (entry + exit)

# ── combine risk management ───────────────────────────────────────────────────
combine_trailing_dd:     2000.0   # $2k trailing DD limit (Topstep $50K combine)
combine_profit_target:   3000.0   # $3k profit target
qualifying_day_min:       150.0   # $/day minimum for a qualifying session
daily_loss_halt:         -300.0   # halt new entries if session P&L ≤ -$300
                                   # (~7 max-loss trades at $40/contract)
```

---

## In-Sample and Holdout Data Ranges

- **Development data (in-sample):** 2025-05-01 → 2026-02-28
  - MNQ: `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv`
  - MNQ: `data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv`
  - ES:  `data/processed/dollar_bars/1_minute/es_1min_2025_2026.csv`
- **Sealed holdout (DO NOT ACCESS until Gate 1 passes):** 2026-03-01 → present
  - `data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv`

Note: ES data is required for signal construction only; MNQ is the traded instrument.
OOS holdout currently contains MNQ only. ES OOS data must be fetched (TradeStation)
before running the OOS backtest.

---

## Integrity Hashes

| Hash | File | Value |
|---|---|---|
| (a) stat_arb_short_config.yaml SHA-256 | `stat_arb_short_config.yaml` | `403c4c521f2ea100d8d66006e1f76206c7dff45eaaf4cf506dd0c9d068cf673b` |
| (b) study_stat_arb_short_only.py SHA-256 | `study_stat_arb_short_only.py` | `efc10cc3d2e0749fe97178b87a3527db94f72d2470e32d3c039ca54f51ba176b` |
| (c) Git HEAD at seal time | — | `44cc6192c7f18b284ec9e4d5ade81430cf9bc900` |

*Hash (a): Proves strategy parameters (threshold, stop, TP, session rules) unchanged.*
*Hash (b): Proves Gate 0 simulation logic (beta estimation, divergence calc, P&L formula) unchanged.*
*Hash (c): Commit this document before any full backtest to prove pre-reg preceded data access.*

---

## Scope Constraint

This pre-registration covers **backtest-validation only**.
No live ProjectX trader, combine account setup, or position-size decision is made
until Gate 2 passes.
S25 (tier2_streaming_working.py) continues running unchanged on Topstep account 23884932.

## Combine Strategy Search Context

This is the **first Gate 0 PASS** after eight failed strategy families:
ORB (4), VWAP Reversion (2), PBC (2), ES/MNQ Stat Arb all-directions, Vol Compression
(1-min + 15-min), Volume Profile POC Fade, Lunch-Window Oscillation.
The direction-asymmetry discovery (long fails, short passes) is the novel structural
finding that separates this candidate from the prior stat arb attempt.
