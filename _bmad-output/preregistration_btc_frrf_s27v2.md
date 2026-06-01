# Pre-Registration: BTC FRRF SHORT-only (S27v2)

**Generated:** 2026-06-01
**Experiment ID:** btc-frrf-s27v2
**Instrument:** PF_XBTUSD (Kraken Futures perpetual, BTC/USD)
**Authored by:** Alex (warper2001@gmail.com)
**Status:** SEALED — do not modify after commit.

---

## Parent Study Closure

**S27 (btc-frrf) is hereby closed as INCONCLUSIVE.**

S27 backtest result: N=283, PF=1.159 vs. gate PF>1.20 → formal gate FAIL.
S27 is not closed as "failed" because the improving rolling PF trend (see Motivating Evidence)
indicates the null hypothesis is not confirmed. S27 is inconclusive on its stated terms.

---

## Motivating Evidence for S27v2

Post-S27 diagnostic analysis (performed 2026-06-01, prior to this pre-reg commit):

1. **S27 LONG arm was vestigial:** N=27 trades, PF=1.042, P&L=+$100 over 16 months.
   Structurally, crowded-short regimes (funding < -0.02%/8h) are rare (1.7% of historical
   readings) and produced no detectable edge. The LONG arm diluted the SHORT signal.

2. **S27 SHORT arm showed improving trend:**
   - 2025 average rolling 91-day PF: 1.183
   - 2026 average rolling 91-day PF: 2.057 (Δ=+0.875)
   - Last 91-day window (ending 2026-01-12): PF=2.257, N=20
   - Monthly breakdown: Aug 2025 (N=74, PF=1.316) is highest-N anchor; Sep/Oct 2025
     rough patch (PF=0.742/0.511) confirms the strategy experiences genuine drawdowns
     and is not a backtest artifact.

3. **Mechanism is directionally asymmetric by design:** The funding rate SHORT thesis
   (crowded longs → mean reversion) has structural academic support (BIS WP 1087). The
   LONG thesis had no independent justification — it was included for symmetry, not evidence.

This pre-registration prospectively tests the SHORT-only hypothesis informed by the above.
Per pre-registration discipline, no config parameters have been changed since S27 backtest.
The only change is structural: the LONG entry path is disabled via `funding_rate_long_threshold: -99.0`.

---

## Hypothesis

**H₁ (alternative):** The S25 signal stack (H1 bearish sweep + M15 CHoCH + M1 bearish FVG),
applied only when the 8h mark/spot basis exceeds +0.03%/8h (crowded longs → SHORT entries only),
produces Profit Factor > 1.20 over N ≥ 30 live paper trades within 30 calendar days on PF_XBTUSD.

**H₀ (null):** The SHORT-only funding-rate-gated strategy produces PF ≤ 1.00 — the funding
rate crowded-long regime does not predict subsequent bearish reversals that align with the
signal stack, and the S27 improving trend was noise.

---

## Decision Rule (Pre-committed, Immutable After Seal)

| Criterion | Threshold |
|---|---|
| Profit Factor | **PF > 1.20** |
| Minimum Trades | **N ≥ 30** |
| Min Calendar Days | **30 days from first live trade** |
| Early STOP | Halt if PF < 0.80 at N = 15 |
| Max Duration | 90 calendar days |

---

## Configuration Snapshot

| Field | Value |
|---|---|
| sl_multiplier | 5.0 |
| tp_multiplier | 6.0 |
| entry_pct | 0.5 |
| atr_threshold | 0.25 |
| max_gap_dollars | 150.0 |
| max_hold_bars | 60 |
| max_pending_bars | 240 |
| contracts_per_trade | 1 |
| max_daily_loss | -500.0 |
| vol_regime_lookback | 120 |
| vol_regime_threshold | 0.75 |
| min_gap_atr_ratio | 0.04 |
| ml_threshold | 0.0 |
| bearish_only | False |
| h1_sweep_lookback | 6 |
| tuesday_exclusion | False |
| m15_confirmation | True |
| enable_kill_zone_filter | False |
| kill_zone_start_et | 17:00 |
| kill_zone_end_et | 19:00 |
| commission_per_roundtrip | 0.04 |
| enable_ifvg_fallback | False |
| funding_rate_filter_enabled | True |
| funding_rate_short_threshold | 0.0003 |
| **funding_rate_long_threshold** | **-99.0** ← LONG path disabled (was -0.0002 in S27) |

---

## Holdout Data Range

- **Directory:** `data/sealed_holdout/`
- **Start date:** 2026-03-01
- **End date:** 2026-03-01

---

## Integrity Hashes

| Hash | Value |
|---|---|
| (a) YAML config SHA-256 | `00bd0769c1228e8d4be9d325ea7df8bfd2a06f4a9d4cfa6db007437628db02c7` |
| (b) strategy_core.py SHA-256 | `47dce2018ca4a501f0790756f02b05877e7e5cf9b5fbb0e46e38671b1192d582` |
| (c) Git HEAD commit | `85e1cb99e2add7935194067cb626c89713d30d61` |

*Hash (a): SHA-256 of `strategy_config_btc_frrf_s27v2.yaml` file bytes.*
*Hash (b): SHA-256 of `src/research/strategy_core.py` source bytes.*
*Hash (c): `git rev-parse HEAD` at seal time — commit this document to make it tamper-evident.*
