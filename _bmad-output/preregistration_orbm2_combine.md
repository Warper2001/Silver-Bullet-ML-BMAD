# Pre-Registration: ORBM-2 Combine Strategy

**Generated:** 2026-06-08
**Experiment ID:** orbm2-combine-v1
**Pre-registration commit:** (populate after `git commit`)

---

## ⚠️ Data-Observation Disclosure

This strategy was designed AFTER running Phase A diagnostic studies on in-sample data
(2025-01-01 → 2026-02-28). The following findings from in-sample exploration directly
informed the ORBM-2 parameter choices:

| Finding | Source | Impact on ORBM-2 |
|---|---|---|
| 74.2% ORB continuation rate | `study_orb_control_window.py` | Enter WITH extension (not against it) |
| +17.4 ppt ORB-specific excess over control | `study_orb_control_window.py` | Confirms ORB context is load-bearing |
| Opposite-boundary stop: 81% setups skipped, median $295/contract | `study_orb_continuation_target.py` | Stop at ORB boundary (0.25×ORB_size from entry) instead |
| NSR=44% with ORBM-1 v2; SHORT WR=56%, LONG WR=29% | `study_orb_noise_stop_rate.py` | Symmetric entry with stop at boundary; lower threshold for frequency |
| 0.25× threshold expected frequency: ~1.5–2.5/session | Design inference | extension_threshold: 0.25 |

**Consequence:** The in-sample backtest is confirmatory, not discovery. A strong
in-sample result is expected and does NOT constitute independent validation.
**The OOS holdout (≥ 2026-03-01) is the primary validity test.**

---

## Hypothesis

**H₁ (alternative):** ORBM-2 — entering IN THE DIRECTION of ORB extensions at
0.25×ORB_size threshold, stopping at the ORB boundary, and taking profit at 1.5R —
generates positive expectancy on MNQ 1-minute bars with frequency ≥ 1.0 setups/day,
WR ≥ 55%, PF ≥ 1.40, and MaxDD ≤ $1,500 in the 2025-01-01 → 2026-02-28
in-sample period; and retains PF ≥ 1.30 OOS.

**H₀ (null):** Continuation trades at a tight ORB-boundary stop have no edge
(PF ≤ 1.0 OOS); the in-sample result is a noise artifact of the explored threshold.

---

## Strategy Logic (Frozen)

| Element | Rule |
|---|---|
| ORB window | 09:30–09:44 ET (bars starting before 09:45) |
| Extension threshold | Close ≥ 0.25 × ORB_size beyond boundary |
| Extension window | 09:45–10:45 ET (no new entries after) |
| Entry direction | LONG for upward extension; SHORT for downward extension |
| Entry price | Extension bar's close (market order at next bar open equivalent) |
| Stop | ORB boundary ± 1 tick (structural invalidation) |
| Stop distance | ≈ 0.25 × ORB_size from entry |
| Stop cap | Skip trade if stop > 75 pts ($150/contract) |
| Take profit | 1.5R from entry in continuation direction (single target) |
| Hard close | 11:30 ET — all positions closed at market |
| Max trades/session | 1 (first qualifying extension only) |
| Daily loss halt | Halt new signals if session P&L ≤ -$200 |
| Daily profit halt | Halt new signals if session P&L ≥ +$750 |
| ORB minimum size | Skip sessions with ORB < 5 pts (no real range) |
| Sizing — small stop | stop_pts < 50 → 2 contracts |
| Sizing — large stop | stop_pts 50–75 → 1 contract |
| Sizing — skip | stop_pts > 75 → no trade |

---

## Go / No-Go Decision Rules (Pre-committed, Immutable After Seal)

### Gate 1 — In-Sample Backtest (2025-01-01 → 2026-02-28)

| Criterion | Minimum | Target | Action if below minimum |
|---|---|---|---|
| Win rate | ≥ 55% | ≥ 60% | STOP — no OOS access |
| Profit factor | ≥ 1.40 | ≥ 1.80 | STOP |
| Max backtest drawdown | ≤ $1,500 | ≤ $1,000 | STOP |
| Frequency | ≥ 1.0 setups/day | ≥ 1.5/day | STOP |
| N trades | ≥ 80 | ≥ 120 | STOP |
| Qualifying sessions / 20 | ≥ 6 | ≥ 8 | STOP |
| Largest day as % of total P&L | ≤ 40% | ≤ 30% | WARNING |
| Per-trade Sharpe | ≥ 0.20 | ≥ 0.30 | WARNING |

### Gate 2 — OOS Holdout (≥ 2026-03-01, one-shot, requires Gate 1 pass)

| Criterion | Minimum | Target |
|---|---|---|
| OOS profit factor | ≥ 1.30 | ≥ 1.50 |
| OOS PF retention vs in-sample | ≥ 75% | ≥ 85% |
| OOS win rate | ≥ 52% | ≥ 56% |
| N OOS trades | ≥ 20 | — |

**OOS stopping rule (live, if deployed):** Halt combine account trading if PF < 1.10
after first 25 OOS trades.

---

## Strategy Parameters Snapshot

```yaml
# ORBM-2 (ORB Breakout Momentum v2) — Topstep Combine Strategy
# Pre-registration: frozen before any backtest run. Do not edit after commit.
# Created: 2026-06-08
#
# KEY CHANGE FROM ORBM-1: Enter WITH the extension (not against it).
# The 74.2% ORB continuation finding is the signal. ORBM-1 faded it.
# ORBM-2 runs with it.

# ── Timing (Eastern Time) ──────────────────────────────────────────────────────
orb_start_et: "09:30"         # ORB build window start
orb_end_et: "09:45"           # ORB build window end (bars starting before 09:45)
extension_start_et: "09:45"   # Earliest bar for extension detection
extension_end_et: "10:45"     # No new extensions after this time
hard_close_et: "11:30"        # All open positions closed by this time

# ── Signal ─────────────────────────────────────────────────────────────────────
extension_threshold: 0.25     # Extension requires close ≥ 0.25 × orb_size beyond boundary
                              # (lowered from 0.5 to increase frequency; ~80-90% of sessions)

# ── Entry ──────────────────────────────────────────────────────────────────────
# Enter IN THE DIRECTION of the extension (LONG for upward, SHORT for downward)
# Entry at the extension bar's close (market order)

# ── Stop ───────────────────────────────────────────────────────────────────────
# Stop at the ORB boundary ± 1 tick (structural invalidation level)
# If price re-enters the range, the breakout thesis is dead.
# Stop distance ≈ 0.25 × orb_size from entry
stop_cap_pts: 75              # Skip trade if stop distance > 75 pts ($150/contract)

# ── Take Profit ────────────────────────────────────────────────────────────────
tp_r_multiple: 1.5            # TP at 1.5R in the continuation direction (single target)

# ── Position Sizing ────────────────────────────────────────────────────────────
contracts_small_stop: 2       # stop_pts < 50 → 2 contracts (stop < $100/contract)
contracts_large_stop: 1       # stop_pts 50–75 → 1 contract ($100–$150/contract)
# skip if stop_pts > 75 (stop > $150/contract)

# ── Session Management ─────────────────────────────────────────────────────────
max_trades_per_session: 1     # One trade per RTH session maximum (first extension only)
daily_loss_limit_usd: -200    # Halt new signals if session P&L ≤ -$200
daily_profit_halt_usd: 750    # Halt new signals once session P&L ≥ +$750 (consistency cap buffer)

# ── Quality Filter ─────────────────────────────────────────────────────────────
orb_min_size_points: 5.0      # Skip sessions with ORB < 5 points (no real range formed)

# ── Market Constants ───────────────────────────────────────────────────────────
point_value_usd: 2.0          # USD per MNQ index point
tick_size: 0.25               # MNQ minimum price increment
commission_per_contract_rt: 0.40  # $0.40/contract/roundtrip
```

---

## In-Sample and Holdout Data Ranges

- **Development data (in-sample):** 2025-01-01 → 2026-02-28 (UTC)
- **Data files:**
  - `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv`
  - `data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv` (filtered to < 2026-03-01)
- **Sealed holdout (DO NOT TOUCH until Gate 1 passes):** 2026-03-01 → present

Accessing holdout before Gate 1 passes voids this pre-registration.

---

## Integrity Hashes

| Hash | Path | Value |
|---|---|---|
| (a) orbm2_config.yaml SHA-256 | `orbm2_config.yaml` | `8dc2a487a5c2fa4d890cf74cacd7646a7b81bd1946cea3c0af75b22b67036afb` |
| (b) sorm_core.py SHA-256 | `src/research/sorm_core.py` | `9861c0c9580fdb5820c71913f35e4fe02ab2be4820d5ec027b45a8622290b191` |
| (c) Git HEAD at seal time | — | `ffb60f50843d299754282e4abb26f5514227fd26` |

*Hash (a): Proves config parameters unchanged between pre-registration and backtest run.*
*Hash (b): Proves shared signal logic (ORB build, extension detect) unchanged.*
*Hash (c): Commit this document first; then `git rev-parse HEAD` in the backtest script confirms pre-reg preceded data access.*

---

## Scope Constraint

This pre-registration covers the **backtest-validation phase only**.
No live ProjectX trader or combine account trading is built until Gate 2 passes.
S25 (tier2_streaming_working.py) continues running unchanged on Topstep account 23884932.
