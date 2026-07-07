# Pre-Registration: YANK Daily Circuit-Breaker Rescale (5ct relic → 2ct-derived)

**Generated:** 2026-07-06
**Experiment ID:** yank-daily-breaker-2ct-rescale
**Type:** Risk-parameter rescale only — NO signal-logic change, NO new experiment.
No holdout data is accessed or spent by this change.

**Status:** DRAFT — awaiting Alex's explicit go. Becomes SEALED when this document
is committed to main-line history; the YAML edit happens only AFTER the seal commit
(Epic-8 order: seal → YAML edit → restart `trader-yank`).

---

## 0. Motivation

The 2026-07-06 concurrency check (mechanics roundtable, A2-step1) found YANK's
`max_daily_loss = -750.0` is a relic of the 5-contract era. YANK has run at
**2 contracts** since joint deployment (systemd `YANK_CONTRACTS=2` overrides the
YAML's `contracts_per_trade: 5`), but the daily breaker was never rescaled.

Consequences at −$750:
- Joint worst-day allowance = MIM-NB DLL −$500 + YANK −$750 = **−$1,250**, which
  breaches the floor-monitor halt line (floor + $750) by **$494** from the
  2026-07-02 equity position (room-to-halt $755.76).
- YANK alone can consume the entire remaining room-to-halt before its own
  breaker fires.

## 1. Config change (the only diff)

| Field (`strategy_config.yaml`) | Before | After |
|---|---|---|
| `max_daily_loss` | `-750.0` | `-300.0` |

All other YAML fields, all S25-frozen parameters, YANK signal logic, contract
count (2), and MIM-NB constants are **unchanged**.

## 2. Derivation (derive-don't-assert)

The new value is not hand-picked; it preserves the original design ratio at the
current contract count:

- Worst-case single trade: `max_gap_dollars` $60 caps the gap at 30 MNQ pts;
  `sl_multiplier` 2.0 puts the stop 2×gap from entry → max loss **$120/contract**.
- Original design (5ct): worst single trade = 5 × $120 = $600; breaker −$750
  → implicit ratio **1.25× worst single trade**.
- Current deployment (2ct): worst single trade = 2 × $120 = **$240**;
  1.25 × $240 = **$300** → `max_daily_loss = -300.0`.

Equivalently: −$750 × (2ct/5ct) = −$300. Both derivations agree.

Secondary check (joint worst-day): MIM −$500 + YANK −$300 = −$800 (was −$1,250).
This does not by itself guarantee the halt line can't be hit on a bad day when
equity already sits near floor + $750 — the external floor monitor remains the
backstop. The structural fix (per-entry shared-floor buffer gate in YANK, parity
with MIM's BUFFER_GATE) is tracked separately and is NOT part of this change.

## 3. Deployment procedure (after seal commit + Alex's go)

1. Edit `strategy_config.yaml`: `max_daily_loss: -300.0` (one line).
2. `systemctl restart trader-yank` (YAML is read at startup).
3. Verify in `logs/`: breaker threshold logged as −300 on first bar of next session.

Rollback: revert the YAML line, restart. No state migration.

## 4. Non-goals / freeze reaffirmation

- No change to `min_gap_atr_ratio`, CHoCH params, SL/TP multipliers, ML threshold,
  or any S25/sealed YANK config (SL2/TP8/ml0.50 stays per seal 138cab1 lineage).
- No change to MIM-NB (`CAT_STOP_PTS=250`, `DLL_GUARD_USD=-500` stay per 30bc6a8).
- Breaker semantics unchanged: realized daily PnL ≤ threshold → no NEW entries for
  the rest of the session (existing `DailyLossCircuitBreaker`, no code edit).

---

## Configuration Snapshot

| Field | Value |
|---|---|
| sl_multiplier | 2.0 |
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
| min_gap_atr_ratio | 0.25 |
| ml_threshold | 0.5 |
| bearish_only | True |
| h1_sweep_lookback | 6 |
| kill_zone_start_et | 09:30 |
| kill_zone_end_et | 11:00 |
| commission_per_roundtrip | 4.0 |
| enable_kill_zone_filter | True |
| m15_confirmation | True |
| tuesday_exclusion | True |
| enable_ifvg_fallback | False |
| funding_rate_filter_enabled | False |
| funding_rate_short_threshold | 0.03 |
| funding_rate_long_threshold | -0.02 |
| enable_breakeven_stop | False |
| breakeven_trigger_r | 2.0 |
| enable_trailing_stop | False |
| trailing_stop_mult | 1.5 |

---

## Holdout Data Range

- **Directory:** `data/sealed_holdout/`
- **Start date:** 2026-03-01
- **End date:** 2026-03-01

---

## Integrity Hashes

| Hash | Value |
|---|---|
| (a) YAML config SHA-256 | `faef1c740ed753449796dc948cce15f41722e0b8b60dbb99df3d82e37d1d8b52` |
| (b) strategy_core.py SHA-256 | `4824e46a35228afd4b600ce8ac631e0dc8e43d8b195274848303ec4b68c23598` |
| (c) Git HEAD commit | `22a9ea4cfefef9bac25c5781156dad9a48a85232` |

*Hash (a): SHA-256 of `strategy_config.yaml` file bytes.*
*Hash (b): SHA-256 of `/root/Silver-Bullet-ML-BMAD/.claude/worktrees/prereg-risk-mechanics/src/research/strategy_core.py` source bytes.*
*Hash (c): `git rev-parse HEAD` at seal time — commit this document to make it tamper-evident.*
