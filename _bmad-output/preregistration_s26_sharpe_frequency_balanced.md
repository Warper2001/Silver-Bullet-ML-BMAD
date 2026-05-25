# Pre-Registration: s26-sharpe-frequency-balanced

**Generated:** 2026-05-25
**Experiment ID:** s26-sharpe-frequency-balanced

---

## Hypothesis

<!-- Fill in H₁ and H₀ after running prereg_seal.py, BEFORE committing -->

**H₁ (alternative):**

**H₀ (null):**

---

## Decision Rule (Pre-committed, Immutable After Seal)

| Criterion | Threshold |
|---|---|
| Profit Factor | PF ≥ 2.0 |
| Sharpe Ratio | Sharpe ≥ 1.5 |
| Max Drawdown | ≤ 10% |
| Minimum Trades | N ≥ 200 |
| Stopping Rule | Halt if PF < 1.1 after first 100 OOS trades |

---

## Configuration Snapshot

| Field | Value |
|---|---|
| sl_multiplier | 6.0 |
| tp_multiplier | 8.0 |
| entry_pct | 0.5 |
| atr_threshold | 0.5 |
| max_gap_dollars | 60.0 |
| max_hold_bars | 60 |
| max_pending_bars | 240 |
| contracts_per_trade | 5 |
| max_daily_loss | -750.0 |
| vol_regime_lookback | 120 |
| vol_regime_threshold | 0.75 |
| min_gap_atr_ratio | 0.35 |
| ml_threshold | 0.0 |
| bearish_only | False |
| h1_sweep_lookback | 6 |
| kill_zone_start_et | 09:30 |
| kill_zone_end_et | 11:00 |
| commission_per_roundtrip | 4.0 |
| enable_kill_zone_filter | False |
| m15_confirmation | False |
| tuesday_exclusion | False |

---

## Holdout Data Range

- **Directory:** `data/sealed_holdout/`
- **Start date:** 2026-03-01
- **End date:** 2026-03-01

---

## Integrity Hashes

| Hash | Value |
|---|---|
| (a) YAML config SHA-256 | `e70b4895337afdda67c6ae117903004922550d7033bac0300b168ea182e184d2` |
| (b) strategy_core.py SHA-256 | `df5153e52901268c71e8576c6c74c56d33e934566b6867da9a887f6cd42d1d46` |
| (c) Git HEAD commit | `86c843887f4b6b4b627901d09521a0ed8fcbd842` |

*Hash (a): SHA-256 of `strategy_config.yaml` file bytes.*
*Hash (b): SHA-256 of `/root/Silver-Bullet-ML-BMAD/src/research/strategy_core.py` source bytes.*
*Hash (c): `git rev-parse HEAD` at seal time — commit this document to make it tamper-evident.*
