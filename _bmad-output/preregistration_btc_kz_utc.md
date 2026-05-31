# Pre-Registration: BTC Kill Zone — H1·M15·M1·g0.04 in 21:00–23:00 UTC Window

**Generated:** 2026-05-31
**Experiment ID:** btc-kz-utc
**Instrument:** PF_XBTUSD (Kraken Futures perpetual, BTC/USD)
**Authored by:** Alex (warper2001@gmail.com)
**Status:** SEALED — do not modify after commit.

---

## Background

The S25 bearish FVG signal stack (H1 sweep + M15 CHoCH + M1 FVG) was backtested
on BTC 1-min data (2025-01-01 → 2026-05-10) across all hours and showed no edge
(PF 0.71–0.88 across 600+ trades). Academic research identifies 21:00–23:00 UTC
as the highest-return window in BTC (post-US close / pre-Asia handoff) with a
seasonality strategy producing 40.64% annualized, Calmar 1.79 when trading only
this window. This pre-registration tests whether concentrating the S25 signal stack
in this documented high-return window recovers or creates a BTC edge.

BTC-specific calibration applied vs MNQ S25:
- `atr_threshold: 0.25` (BTC 1-min ATR ~100–150pts, FVGs ~30–50pts)
- `min_gap_atr_ratio: 0.04` (BTC H1 ATR ~700pts; min gap = 28pts)
- `max_gap_dollars: 150.0` (= 300pt cap after POINT_VALUE_USD×2.0 multiplication)
- `bearish_only: true` — pure SHORT, no Golden Flip inversion
- Kill zone: `kill_zone_start_et="17:00"`, `kill_zone_end_et="19:00"` (America/New_York)
  → 21:00–23:00 UTC in EDT; 22:00–00:00 UTC in EST

---

## Hypothesis

**H₁ (alternative):** The S25 signal stack (H1 bearish liquidity sweep within last 6
H1 bars + M15 CHoCH + M1 bearish FVG ≥ 0.04×H1 ATR), applied only during the
17:00–19:00 ET kill zone window on PF_XBTUSD Kraken Futures perpetual, produces
Profit Factor > 1.10 over N ≥ 30 live paper trades within 90 calendar days.

**H₀ (null):** The strategy produces PF ≤ 1.00 in the kill zone window — the
seasonal BTC edge does not interact with the S25 signal stack in a positive way,
and directional FVG selection in this window provides no advantage over random entry.

---

## Decision Rule (Pre-committed, Immutable After Seal)

| Criterion | Threshold |
|---|---|
| Profit Factor | **PF > 1.10** |
| Minimum Trades | **N ≥ 30** |
| Minimum Calendar Days | **60 days from first live trade** |
| Early STOP rule | Halt if PF < 0.80 at N = 15 (clear non-edge) |
| Max study duration | 120 calendar days from deployment |

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
| bearish_only | True |
| h1_sweep_lookback | 6 |
| kill_zone_start_et | 17:00 |
| kill_zone_end_et | 19:00 |
| commission_per_roundtrip | 0.04 |
| enable_kill_zone_filter | True |
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
| (a) YAML config SHA-256 | `ba5d0e6d5ab3d907fcad8c772502fa3cebb2cc6847e007b69b2df3d523e711a2` |
| (b) strategy_core.py SHA-256 | `31010a51cbb93105e9f42f84ee83c560ed08a2bca2843edc561394805529be06` |
| (c) Git HEAD commit | `717c85c50a562fa96a8dac56f045cd0ed265f270` |

*Hash (a): SHA-256 of `strategy_config_btc_kz_utc.yaml` file bytes.*
*Hash (b): SHA-256 of `/root/Silver-Bullet-ML-BMAD/src/research/strategy_core.py` source bytes.*
*Hash (c): `git rev-parse HEAD` at seal time — commit this document to make it tamper-evident.*
