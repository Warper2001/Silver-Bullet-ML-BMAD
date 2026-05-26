# Pre-Registration: s26-crypto-live-deployment

**Generated:** 2026-05-26
**Experiment ID:** s26-crypto-live-deployment

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
| sl_multiplier | 5.0 |
| tp_multiplier | 6.0 |
| entry_pct | 0.5 |
| atr_threshold | 0.5 |
| max_gap_dollars | 60.0 |
| max_hold_bars | 60 |
| max_pending_bars | 240 |
| contracts_per_trade | 1 |
| max_daily_loss | -50.0 |
| vol_regime_lookback | 120 |
| vol_regime_threshold | 0.75 |
| min_gap_atr_ratio | 0.35 |
| ml_threshold | 0.0 |
| bearish_only | False |
| h1_sweep_lookback | 6 |
| kill_zone_start_et | 04:00 |
| kill_zone_end_et | 11:00 |
| commission_per_roundtrip | 0.04 |
| enable_kill_zone_filter | False |
| m15_confirmation | True |
| tuesday_exclusion | False |
| enable_ifvg_fallback | False |

---

## Holdout Data Range

- **Directory:** `data/sealed_holdout/`
- **Start date:** 2026-03-01
- **End date:** 2026-03-01

---

## Integrity Hashes

| Hash | Value |
|---|---|
| (a) YAML config SHA-256 | `16d285fb82270b99f3d83fc27c31abe731f037775de9655c5e969d75cde388db` |
| (b) strategy_core.py SHA-256 | `31010a51cbb93105e9f42f84ee83c560ed08a2bca2843edc561394805529be06` |
| (c) Git HEAD commit | `6895b3cfab0511c8b071025dbba824419eca6f02` |

*Hash (a): SHA-256 of `strategy_config_kraken_s26.yaml` file bytes.*
*Hash (b): SHA-256 of `/root/Silver-Bullet-ML-BMAD/src/research/strategy_core.py` source bytes.*
*Hash (c): `git rev-parse HEAD` at seal time — commit this document to make it tamper-evident.*
