# Pre-Registration: SORM Combine Strategy v1

**Generated:** 2026-06-08
**Experiment ID:** sorm-combine-v1

---

## Hypothesis

**H₁ (alternative):** The Session Open Range Mean Reversion strategy generates
positive expectancy on MNQ 1-minute bars: extensions beyond the 09:30–09:44 ET
opening range by ≥ 50% of ORB size (09:45–10:45 ET window) revert to the ORB
midpoint in >55% of cases, and a filtered trade (RSI 30–70, stop ≤ $200/contract)
produces PF ≥ 1.40 with MaxDD ≤ $2,500 out-of-sample.

**H₀ (null):** Extension events do not revert to the ORB midpoint at a rate
above chance; the strategy has no edge (PF ≤ 1.0 OOS, reversion rate ≤ 50%).

---

## Go / No-Go Decision Rules (Pre-committed, Immutable After Seal)

### Gate 0 — Reversion Rate Study (run FIRST, before full backtest)

| Criterion | Threshold | Action |
|---|---|---|
| Reversion rate (extensions → orb_mid) | ≥ 55% | Proceed to full backtest |
| Reversion rate | 50–54% | Reconsider parameters before proceeding |
| Reversion rate | < 50% | STOP — no live code built |

### Gate 1 — In-Sample Backtest

| Criterion | Threshold |
|---|---|
| Win rate | ≥ 55% |
| Profit factor (in-sample) | ≥ 1.40 |
| Max drawdown | ≤ $2,500 (target ≤ $1,500) |
| Frequency | 1.0–3.5 trades/day |
| Risk per trade | $100–$200 avg |
| Minimum trades | N ≥ 150 |

### Gate 2 — OOS Holdout (one-shot, ≥ 2026-03-01)

| Criterion | Threshold |
|---|---|
| OOS profit factor | ≥ 1.40 |
| OOS PF retention vs in-sample | ≥ 80% of in-sample PF |
| Stopping rule | Halt live if PF < 1.10 after first 30 OOS trades |

---

## Strategy Parameters Snapshot

```yaml
# SORM (Session Open Range Mean Reversion) v1 — Topstep Combine Strategy
# Pre-registration: frozen before any backtest run. Do not edit after commit.
# Created: 2026-06-08

# ── Timing (all Eastern Time) ──────────────────────────────────────────────────
orb_start_et: "09:30"         # ORB build window start
orb_end_et: "09:45"           # ORB build window end (last bar starting at 09:44)
extension_start_et: "09:45"   # Earliest bar allowed for extension detection
extension_end_et: "10:45"     # No new extensions detected after this time
hard_close_et: "11:30"        # All open positions closed by this time (time-stop)

# ── Entry Conditions ───────────────────────────────────────────────────────────
extension_threshold: 0.5      # Extension requires close ≥ 0.5 × orb_size beyond boundary
rsi_period: 14                # RSI lookback period
rsi_low: 30                   # RSI below this: skip (momentum too extreme, no fade)
rsi_high: 70                  # RSI above this: skip (momentum too extreme, no fade)
rsi_direction_lookback: 3     # Bars to measure RSI direction (must point toward mid-range)

# ── Risk Parameters ────────────────────────────────────────────────────────────
stop_skip_threshold_usd: 200  # Skip trade if stop distance > $200/contract
stop_small_threshold_usd: 100 # Stop < $100/contract → 2 contracts; $100–$200 → 1 contract
contracts_small_stop: 2       # Contracts when stop_dist_usd < stop_small_threshold_usd
contracts_large_stop: 1       # Contracts when stop_dist_usd $100–$200
daily_loss_limit_usd: -300    # Halt rest of session if session P&L ≤ -$300

# ── Exit Parameters ────────────────────────────────────────────────────────────
tp1_fraction: 0.60            # Fraction of position closed at TP1 (orb_mid)
# TP2 target: opposite ORB boundary (orb_low for bear fade, orb_high for bull fade)
max_trades_per_session: 1     # One trade per RTH session maximum

# ── Minimum ORB Size ───────────────────────────────────────────────────────────
orb_min_size_points: 2.0      # Skip sessions with ORB size < 2 index points (stale)

# ── Market Constants ───────────────────────────────────────────────────────────
point_value_usd: 2.0          # USD per MNQ index point
tick_size: 0.25               # MNQ minimum price increment
commission_per_contract_rt: 0.40  # $0.40/contract/roundtrip (both sides)
```

---

## In-Sample Data Range

- **Development data:** 2025-01-01 → 2026-02-28 (UTC)
- **Data files:**
  - `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv`
  - `data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv` (filtered to < 2026-03-01)
- **Sealed holdout (DO NOT TOUCH until Gate 1 passes):** 2026-03-01 → present

---

## Integrity Hashes

| Hash | Path | Value |
|---|---|---|
| (a) sorm_config.yaml SHA-256 | `sorm_config.yaml` | `7dae4806adf9dcceae80f97fe8a1c8798205435a7c14548aad8558a79cd4e458` |
| (b) sorm_core.py SHA-256 | `src/research/sorm_core.py` | `9861c0c9580fdb5820c71913f35e4fe02ab2be4820d5ec027b45a8622290b191` |
| (c) Git HEAD commit | — | `1ff646d9998d4f6698cfe0ea349ea86d6dc258db` |

*Hash (a): SHA-256 of `sorm_config.yaml` file bytes — proves parameters unchanged.*
*Hash (b): SHA-256 of `src/research/sorm_core.py` source bytes — proves signal logic unchanged.*
*Hash (c): `git rev-parse HEAD` at seal time — commit this document to make it tamper-evident.*

---

## Scope Constraint

This pre-registration covers the **backtest-validation phase only**.
No live ProjectX trader is built until Gate 2 passes.
S25 (tier2_streaming_working.py) continues running unchanged on Topstep account 23884932.
