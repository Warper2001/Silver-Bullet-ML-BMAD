# Pre-Registration: BTC Funding Rate Regime Filter (FRRF / S27)

**Generated:** 2026-05-31
**Experiment ID:** btc-frrf
**Instrument:** PF_XBTUSD (Kraken Futures perpetual, BTC/USD)
**Authored by:** Alex (warper2001@gmail.com)
**Status:** SEALED — do not modify after commit.

---

## Background

Prior BTC experiments on the S25 signal stack showed no edge (PF 0.71–0.88 all-hours;
PF 0.63 in 21:00–23:00 UTC kill zone). The kill zone failed because the documented
bullish seasonality in that window conflicts with bearish-only entries.

This pre-registration tests whether gating the SAME signal stack (H1 sweep + M15 CHoCH
+ M1 FVG) with a funding rate regime filter produces edge. Funding rate is the defining
"quant crypto standard" signal — unique to perpetuals markets — with academic support:
BIS Working Paper 1087 (Sharpe ~12.8 for BTC carry), ScienceDirect funding rate study
(15–35% annualized delta-neutral). When one side is overcrowded, the funding rate goes
extreme and mean reversion tends to follow.

Funding rate data: mark/spot basis from Kraken 1h charts, averaged per 8h window.
`data/kraken/PF_XBTUSD_funding_rate.csv` — 1,731 rows, 2024-11-01 → 2026-05-31.
Historical regime: 22.4% SHORT bias, 1.7% LONG bias, 75.9% NEUTRAL.

---

## Hypothesis

**H₁ (alternative):** The S25 signal stack (H1 bearish/bullish sweep + M15 CHoCH for
SHORT path + M1 bearish/bullish FVG), applied only when the 8h mark/spot basis exceeds
±threshold — SHORT when basis > +0.03%/8h (crowded longs), LONG when basis < -0.02%/8h
(crowded shorts) — produces Profit Factor > 1.20 over N ≥ 30 live paper trades within
90 calendar days on PF_XBTUSD.

**H₀ (null):** The strategy produces PF ≤ 1.00 — the funding rate regime does not
predict subsequent reversals that align with the signal stack, and crowded-regime
gating provides no advantage over random entry.

---

## Decision Rule (Pre-committed, Immutable After Seal)

| Criterion | Threshold |
|---|---|
| Profit Factor | **PF > 1.20** |
| Minimum Trades | **N ≥ 30** |
| Min Calendar Days | **60 days from first live trade** |
| Early STOP | Halt if PF < 0.80 at N = 15 |
| Max Duration | 120 calendar days |

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
| kill_zone_start_et | 17:00 |
| kill_zone_end_et | 19:00 |
| commission_per_roundtrip | 0.04 |
| enable_kill_zone_filter | False |
| m15_confirmation | True |
| tuesday_exclusion | False |
| enable_ifvg_fallback | False |
| funding_rate_filter_enabled | True |
| funding_rate_short_threshold | 0.0003 |
| funding_rate_long_threshold | -0.0002 |

---

## Holdout Data Range

- **Directory:** `data/sealed_holdout/`
- **Start date:** 2026-03-01
- **End date:** 2026-03-01

---

## Integrity Hashes

| Hash | Value |
|---|---|
| (a) YAML config SHA-256 | `f6c8cbc960996363ad2b3726d07e5b0f8da8ac580574bb0df5cb956344609d44` |
| (b) strategy_core.py SHA-256 | `9fdf2f28afba95f39aad756ed51aaf26ead82317a4c945975b463a90143d4889` |
| (c) Git HEAD commit | `31669a7331f2b288ae3bf24c26f0a8371395480f` |

*Hash (a): SHA-256 of `strategy_config_btc_frrf.yaml` file bytes.*
*Hash (b): SHA-256 of `/root/Silver-Bullet-ML-BMAD/src/research/strategy_core.py` source bytes.*
*Hash (c): `git rev-parse HEAD` at seal time — commit this document to make it tamper-evident.*
