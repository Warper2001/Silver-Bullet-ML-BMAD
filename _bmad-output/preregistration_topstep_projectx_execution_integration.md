# Pre-Registration: Topstep / ProjectX Execution Integration

**Registered:** 2026-06-07
**Authored by:** Alex (warper2001@gmail.com)
**Status:** ACTIVE — frozen at commit time.

---

## Purpose

This is an **infrastructure pre-registration**, not a strategy experiment. No strategy
parameters are changing. The sole purpose is to document the execution-layer swap
**before any code is written**, so the SHA of the current strategy state is sealed
against the change.

**What is changing:** The order execution adapter is being replaced.

| Component | Before | After |
|---|---|---|
| Execution adapter | `TradeStationClient` (SIM) | `ProjectXClient` (Topstep funded account) |
| Execution venue | TradeStation SIM | TopstepX via ProjectX REST API |
| Auth | `TradeStationAuthV3` / `.access_token` | ProjectX API key / JWT (24h) |
| Market data source | TradeStation REST (1-min bars) | **Unchanged** — TradeStation REST polling |
| Strategy logic | S25 frozen | **Unchanged** |
| All StrategyConfig fields | S25 frozen | **Unchanged** |

---

## Strategy Isolation (Immutable)

The strategy config snapshot below is **frozen**. The execution integration work
MUST NOT alter any of these values as a side effect. If a strategy parameter change
is needed, it requires a separate pre-registration.

**S25 decision rule remains in force.** Live trades on the Topstep account count
toward the S25 N≥20 gate under the same PF threshold (> 1.1350).

---

## Scope of Permitted Changes

The following changes are pre-approved under this registration:

1. **Create** `src/research/projectx_auth.py`
   - `ProjectXAuth` class: API key → JWT, 24h token caching, auto-refresh
   - Credentials stored in `.projectx_api_key` file (never committed)

2. **Create** `src/research/projectx_client.py`
   - `ProjectXClient` class with identical public interface to `TradeStationClient`
   - Methods: `submit_bracket_order`, `cancel_order`, `close_position_at_market`,
     `reconcile_state`, `cancel_all_pending_orders`
   - REST target: `https://api.topstepx.com`

3. **Modify** `src/research/tier2_streaming_working.py` — **execution path only**
   - Add `USE_PROJECTX` env-var gate in `initialize()`
   - Swap `self._ts_client` assignment: `TradeStationClient` → `ProjectXClient` when gate is set
   - No changes to strategy logic, filters, signal detection, or exit rules

4. **Market data source**: remains `https://api.tradestation.com/v3/marketdata/barcharts/{symbol}` — no change.

---

## What Is NOT Permitted Under This Registration

- Changing any `StrategyConfig` field (sl_multiplier, tp_multiplier, min_gap_atr_ratio, etc.)
- Changing entry/exit logic, filter logic, or CHoCH detection
- Changing market data source (Phase 2 data migration requires its own registration)
- Adjusting position size (`contracts_per_trade`) for Topstep account sizing without a new pre-registration
- Accessing `data/sealed_holdout/` during this work

---

## Risk Notes

- This routes orders to a **live funded Topstep account** — execution risk is real.
- Topstep daily drawdown limits apply independently of `max_daily_loss=-$750` in StrategyConfig.
- Verify ProjectX symbol name for MNQ before going live (may differ from `MNQM26`).
- Run parallel SIM → ProjectX comparison (same signals, both clients) before cutting over.

---

## Configuration Snapshot (S25 frozen — no changes)

| Field | Value |
|---|---|
| sl_multiplier | 5.0 |
| tp_multiplier | 6.0 |
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
| ml_threshold | 0.0 |
| bearish_only | True |
| h1_sweep_lookback | 6 |
| kill_zone_start_et | 09:30 |
| kill_zone_end_et | 11:00 |
| commission_per_roundtrip | 4.0 |
| enable_kill_zone_filter | False |
| m15_confirmation | True |
| tuesday_exclusion | True |
| enable_ifvg_fallback | False |
| funding_rate_filter_enabled | False |
| funding_rate_short_threshold | 0.03 |
| funding_rate_long_threshold | -0.02 |

---

## Holdout Data Range

- **Directory:** `data/sealed_holdout/`
- **Start date:** 2026-03-01
- **End date:** 2026-03-01

---

## Integrity Hashes

| Hash | Value |
|---|---|
| (a) YAML config SHA-256 | `58c7aa652a5a6f1794d88cc2d065e5b1c73fc9fb1f29c1ad0f5ffb8738808569` |
| (b) strategy_core.py SHA-256 | `47dce2018ca4a501f0790756f02b05877e7e5cf9b5fbb0e46e38671b1192d582` |
| (c) Git HEAD commit | `55c905c93d11a071284ce308bf8bce09df503e62` |

*Hash (a): SHA-256 of `strategy_config.yaml` file bytes.*
*Hash (b): SHA-256 of `src/research/strategy_core.py` source bytes.*
*Hash (c): `git rev-parse HEAD` at seal time — commit this document to make it tamper-evident.*

---

## Acknowledgement

By committing this document, the author pre-commits that all code changes under this
registration are limited to the execution adapter swap described above. Any strategy
parameter change requires a new pre-registration with a new git SHA.

*This document is intentionally difficult to amend — that is its purpose.*
