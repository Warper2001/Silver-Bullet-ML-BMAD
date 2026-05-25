# Story 8.3: Multi-Instrument Support

Status: done

## Story

As Alex (the researcher),
I want `Tier2StreamingTrader` to accept a `--symbol` argument so three instances can run simultaneously on MNQ, MES, and M2K,
so that live trade accumulation increases to ~18-25 trades/month (3× from single-instrument ~6/month).

## Background

**Why this matters:** The S25 decision rule requires N≥20 trades AND 60 days before a verdict. At ~6 MNQ trades/month, that's 3-4 months just to reach N=20. Adding MES and M2K (same strategy, different point values) triples the accumulation rate to ~5-8 live trades/week — observable within 1-2 weeks per config change.

**Per-symbol specs** (point values and tick sizes differ):
- `MNQM26`: point_value=2.0, tick_size=0.25, contracts=5 (current defaults)
- `MESM26`: point_value=5.0, tick_size=0.25, contracts=2
- `M2KM26`: point_value=5.0, tick_size=0.10, contracts=2

**Running three instances (after this story):**
```bash
SYMBOL=MNQM26 PYTHONPATH=. .venv/bin/python src/research/tier2_streaming_working.py
SYMBOL=MESM26 PYTHONPATH=. .venv/bin/python src/research/tier2_streaming_working.py
SYMBOL=M2KM26 PYTHONPATH=. .venv/bin/python src/research/tier2_streaming_working.py
```

Or with CLI arg: `python src/research/tier2_streaming_working.py --symbol MESM26`

**Key constraint:** The strategy parameters (entry/exit logic, filters) are identical across all instruments — only the market data feed URL, point value, tick size, and contract count differ. `strategy_core.py` must NOT be changed (AR1 purity).

## Acceptance Criteria

1. `SYMBOL_SPECS` dict defined at module level in `tier2_streaming_working.py` maps each symbol to `{point_value, tick_size, contracts}` for MNQM26, MESM26, M2KM26.
2. `Tier2StreamingTrader.__init__` accepts `symbol: str = "MNQM26"` parameter. Raises `ValueError` for unknown symbols. Stores `self._symbol`, `self._point_value`, `self._tick_size`, `self._contracts` from the spec.
3. `self._bars_base_url` built from `self._symbol` at init (not module-level constant).
4. All `SYMBOL` references in order payloads (`_submit_bracket_order`, `_submit_close_order`) use `self._symbol`.
5. `_close_active_trade` uses `self._point_value` for P&L (not `strategy_core.POINT_VALUE_USD`).
6. `_snap_tick` uses `self._tick_size` (change from `@staticmethod` that uses `strategy_core.TICK_SIZE` to instance method).
7. Order quantity uses `self._contracts` (not `cfg.contracts_per_trade`) in `_submit_bracket_order` and `_submit_close_order`.
8. `_log_trade()` method writes completed trades to `logs/tier2_trade_log.csv` with columns: `timestamp, instrument, direction, entry_price, exit_price, exit_reason, bars_held, pnl_usd`. Called from `_close_active_trade`.
9. `main()` reads symbol from `--symbol` CLI arg (falling back to `SYMBOL` env var, falling back to `"MNQM26"`).
10. All existing tests pass — `python -m pytest tests/ -q` (excluding pre-existing collection errors).

## Tasks / Subtasks

- [x] Task 1: Add `SYMBOL_SPECS` and update module-level constants (AC: #1, #3)
  - [x] Added `SYMBOL_SPECS` dict with MNQM26, MESM26, M2KM26 entries
  - [x] Removed module-level `SYMBOL = "MNQM26"` and `BARS_BASE_URL` (now instance variables)

- [x] Task 2: Update `Tier2StreamingTrader.__init__` (AC: #2, #3)
  - [x] Added `symbol: str = "MNQM26"` parameter with `ValueError` on unknown symbols
  - [x] Set `self._symbol`, `self._point_value`, `self._tick_size`, `self._contracts` from SYMBOL_SPECS
  - [x] Set `self._bars_base_url` built from `self._symbol` at init
  - [x] Updated `initialize()` log to show symbol + point_value + tick + contracts

- [x] Task 3: Replace `SYMBOL` in order payloads (AC: #4)
  - [x] `_submit_bracket_order`: all three `"Symbol": SYMBOL` → `"Symbol": self._symbol`
  - [x] `_submit_close_order`: `"Symbol": SYMBOL` → `"Symbol": self._symbol`

- [x] Task 4: Replace `BARS_BASE_URL` in `_poll_and_process` (AC: #3)
  - [x] `f"{BARS_BASE_URL}&..."` → `f"{self._bars_base_url}&..."`

- [x] Task 5: Replace P&L constant and tick snap (AC: #5, #6)
  - [x] `_close_active_trade`: `strategy_core.POINT_VALUE_USD * cfg.contracts_per_trade` → `self._point_value * self._contracts`
  - [x] `_snap_tick`: changed from `@staticmethod` to instance method using `self._tick_size`

- [x] Task 6: Replace contract quantity in order submissions (AC: #7)
  - [x] `_submit_bracket_order`: `str(cfg.contracts_per_trade)` → `str(self._contracts)`
  - [x] `_submit_close_order`: same change

- [x] Task 7: Add `_log_trade()` and call from `_close_active_trade` (AC: #8)
  - [x] Added `_log_trade()` instance method writing to `logs/tier2_trade_log.csv`
  - [x] Columns: `timestamp, instrument, direction, entry_price, exit_price, exit_reason, bars_held, pnl_usd`
  - [x] Called at end of `_close_active_trade` after P&L computed

- [x] Task 8: Update `main()` (AC: #9)
  - [x] Added `argparse` with `--symbol` flag defaulting to `$SYMBOL` env var or `"MNQM26"`
  - [x] `Tier2StreamingTrader(symbol=args.symbol)` passed through

- [x] Task 9: Run full test suite (AC: #10)
  - [x] All smoke tests pass: MNQM26/MESM26/M2KM26 specs, unknown symbol ValueError, bars_url, snap_tick
  - [x] 219 tests pass; no regressions

## Dev Notes

### Files to Modify

- `src/research/tier2_streaming_working.py` — all changes in this story

### Current Module-Level Constants (Lines 62–67)

```python
SYMBOL = "MNQM26"
BAR_INTERVAL = "1"
BAR_UNIT = "Minute"
BARS_BASE_URL = (f"https://api.tradestation.com/v3/marketdata/barcharts/{SYMBOL}"
                 f"?interval={BAR_INTERVAL}&unit={BAR_UNIT}")
HISTORY_HOURS = 48
POLL_INTERVAL_SECONDS = 60
```

After changes:
```python
# BAR_INTERVAL and BAR_UNIT remain as module-level constants (not symbol-specific)
BAR_INTERVAL = "1"
BAR_UNIT = "Minute"
HISTORY_HOURS = 48
POLL_INTERVAL_SECONDS = 60

SYMBOL_SPECS = {
    "MNQM26": {"point_value": 2.0,  "tick_size": 0.25, "contracts": 5},
    "MESM26": {"point_value": 5.0,  "tick_size": 0.25, "contracts": 2},
    "M2KM26": {"point_value": 5.0,  "tick_size": 0.10, "contracts": 2},
}
```

### `_snap_tick` — Change from Static to Instance Method

Current (line 1042–1045):
```python
@staticmethod
def _snap_tick(price: float) -> float:
    """Round price to nearest MNQ tick (0.25). Avoids float artifacts."""
    return round(round(price / strategy_core.TICK_SIZE) * strategy_core.TICK_SIZE, 10)
```

New:
```python
def _snap_tick(self, price: float) -> float:
    """Round price to nearest instrument tick. Avoids float artifacts."""
    return round(round(price / self._tick_size) * self._tick_size, 10)
```

Call sites: line 1067 `self._snap_tick(entry_dec.entry_price)` and 1068/1069 — already instance method calls, no change needed.

### P&L Calculation — Line 751

Current:
```python
pnl = ((price - t.entry_price) if t.direction == "LONG" else (t.entry_price - price)) * strategy_core.POINT_VALUE_USD * cfg.contracts_per_trade - cfg.commission_per_roundtrip
```

New:
```python
pnl = ((price - t.entry_price) if t.direction == "LONG" else (t.entry_price - price)) * self._point_value * self._contracts - cfg.commission_per_roundtrip
```

Note: `commission_per_roundtrip` stays from StrategyConfig (it's per-roundtrip, not per-contract).

### `_log_trade()` Implementation

```python
def _log_trade(self, entry_time, exit_time, direction: str, entry_price: float,
               exit_price: float, exit_reason: str, bars_held: int, pnl_usd: float) -> None:
    import csv as _csv
    log_path = Path(__file__).parent.parent.parent / "logs/tier2_trade_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp":   str(exit_time),
        "instrument":  self._symbol,
        "direction":   direction,
        "entry_price": round(entry_price, 4),
        "exit_price":  round(exit_price, 4),
        "exit_reason": exit_reason,
        "bars_held":   bars_held,
        "pnl_usd":     round(pnl_usd, 2),
    }
    write_header = not log_path.exists()
    try:
        with log_path.open("a", newline="") as _f:
            _w = _csv.DictWriter(_f, fieldnames=list(row.keys()))
            if write_header:
                _w.writeheader()
            _w.writerow(row)
    except Exception as _e:
        logger.warning(f"Trade log write failed: {_e}")
```

### `main()` Update

Current (lines 1117-1123):
```python
async def main():
    trader = Tier2StreamingTrader()
    await trader.initialize()
    await trader.start_streaming()

if __name__ == "__main__":
    asyncio.run(main())
```

New:
```python
async def main():
    import argparse as _argparse
    parser = _argparse.ArgumentParser(description="Tier 2 FVG Paper Trader")
    parser.add_argument(
        "--symbol",
        default=os.environ.get("SYMBOL", "MNQM26"),
        help="Futures symbol to trade (default: MNQM26 or $SYMBOL env var)",
    )
    args = parser.parse_args()
    trader = Tier2StreamingTrader(symbol=args.symbol)
    await trader.initialize()
    await trader.start_streaming()

if __name__ == "__main__":
    asyncio.run(main())
```

### Strategy Core Unchanged

`strategy_core.POINT_VALUE_USD` and `strategy_core.TICK_SIZE` are NOT removed — they remain as constants (may be used by backtest scripts or other code). Only `Tier2StreamingTrader` stops using them for its own P&L calculation.

### Testing Notes

There are no automated tests for `Tier2StreamingTrader` (it's validated by backtesting). Use smoke tests:

```bash
# Verify SYMBOL_SPECS and import
PYTHONPATH=. .venv/bin/python -c "
from src.research.tier2_streaming_working import Tier2StreamingTrader, SYMBOL_SPECS
print('SYMBOL_SPECS:', SYMBOL_SPECS)
# Verify MNQ default
t = Tier2StreamingTrader()
assert t._symbol == 'MNQM26'
assert t._point_value == 2.0
assert t._tick_size == 0.25
assert t._contracts == 5
print(f'MNQM26: pv={t._point_value} tick={t._tick_size} contracts={t._contracts} PASS')
# Verify MES
t2 = Tier2StreamingTrader(symbol='MESM26')
assert t2._point_value == 5.0
assert t2._tick_size == 0.25
assert t2._contracts == 2
print(f'MESM26: pv={t2._point_value} tick={t2._tick_size} contracts={t2._contracts} PASS')
# Verify M2K
t3 = Tier2StreamingTrader(symbol='M2KM26')
assert t3._point_value == 5.0
assert t3._tick_size == 0.10
assert t3._contracts == 2
print(f'M2KM26: pv={t3._point_value} tick={t3._tick_size} contracts={t3._contracts} PASS')
# Verify unknown symbol raises
try:
    Tier2StreamingTrader(symbol='BOGUS')
    assert False, 'should have raised'
except ValueError as e:
    print(f'Unknown symbol raises ValueError: {e} PASS')
"

# Verify snap_tick uses instance tick_size
PYTHONPATH=. .venv/bin/python -c "
from src.research.tier2_streaming_working import Tier2StreamingTrader
t = Tier2StreamingTrader(symbol='M2KM26')  # tick=0.10
snapped = t._snap_tick(17234.13)  # should round to nearest 0.10
print(f'M2K snap(17234.13) = {snapped}')
assert abs(snapped - 0.10 * round(17234.13 / 0.10)) < 1e-6, f'got {snapped}'
print('PASS')
"
```

### What NOT to Change

- `TIER2_CONFIG` label string
- `strategy_core.py` (AR1 purity — absolutely no changes)
- `StrategyConfig` (parameters stay as-is; `contracts_per_trade` in StrategyConfig is not used for order submission in the multi-instrument path — `self._contracts` takes precedence)
- Any existing test files
- The CHoCH state machine or entry/exit logic

### References

- `src/research/tier2_streaming_working.py` line 62–67: module-level constants to change
- Line 104–126: `_build_strategy_config()` (no change needed in this story)
- Line 340–342: `Tier2StreamingTrader.__init__` signature
- Line 433: `logger.info(f"Symbol: {SYMBOL}")` → change to `self._symbol`
- Line 481: `BARS_BASE_URL` → `self._bars_base_url`
- Line 751: P&L calc using `strategy_core.POINT_VALUE_USD`
- Lines 951–984: `_submit_bracket_order` with hardcoded `SYMBOL`
- Lines 1019–1040: `_submit_close_order` with hardcoded `SYMBOL`
- Lines 1042–1045: `_snap_tick` static method

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

### Completion Notes List

- `SYMBOL_SPECS` dict added with MNQM26/MESM26/M2KM26; module-level `SYMBOL` and `BARS_BASE_URL` constants removed
- `Tier2StreamingTrader.__init__` now accepts `symbol: str = "MNQM26"`; raises `ValueError` on unknown symbols; stores `_symbol`, `_point_value`, `_tick_size`, `_contracts`, `_bars_base_url`
- All hardcoded `SYMBOL` refs in order payloads replaced with `self._symbol`
- P&L calc uses `self._point_value * self._contracts` instead of `strategy_core.POINT_VALUE_USD * cfg.contracts_per_trade`
- `_snap_tick` converted from `@staticmethod` to instance method using `self._tick_size` (M2K tick=0.10 verified)
- `_log_trade()` added: writes `logs/tier2_trade_log.csv` with `instrument` column per trade
- `main()` accepts `--symbol` CLI arg (falls back to `$SYMBOL` env var, then `MNQM26`)
- All 219 tests pass; 0 regressions

### Review Findings

- [x] [Review][Patch] `_active_entry_decision` not cleared when pending order expires [tier2_streaming_working.py:734] — `self.active_trade = None` is set but `self._active_entry_decision` is not; stale `EntryDecision` from the expired trade persists and will be used as the TP/SL reference for the NEXT fill
- [x] [Review][Patch] `_vol_regime_high` set to `False` on `ValueError` [tier2_streaming_working.py:603] — safe-fail should BLOCK trading (`True`), not pass through; `ValueError` from `volatility_regime_filter` currently silently opens the volatility gate
- [x] [Review][Patch] Missing blank line between `_close_active_trade` body and `def _log_trade` [tier2_streaming_working.py:776-777] — PEP 8 violation; methods need one blank line separator inside a class
- [x] [Review][Defer] Crash recovery half-implemented: `StatePersistence.save_state()` in `_enter_trade` but `load_state()` never called in `initialize()` [tier2_streaming_working.py:153] — Epic 4 Story 4-2 work; dangling bracket orders on crash, deferred
- [x] [Review][Defer] `detect_fvg` uses hardcoded `POINT_VALUE_USD=2.0` for dollar-ceiling gate; MES ($5/pt) gaps up to 2.5× more expensive than MNQ [strategy_core.py:359] — pre-existing in AR1-protected strategy_core; deferred
- [x] [Review][Defer] `commission_per_roundtrip=4.0` not scaled per instrument; MES (2 contracts) incorrectly charged MNQ 5-contract rate [strategy_core.py:98] — explicitly noted in Story 8-3 dev notes as "per-roundtrip, not per-contract"; deferred
- [x] [Review][Defer] CSV header TOCTOU race: two instances starting simultaneously can both write headers [tier2_streaming_working.py:802] — SIM context, low operational risk; deferred
- [x] [Review][Defer] `_bar_processing_times` list grows unboundedly for long-running sessions [tier2_streaming_working.py:438] — fix is `deque(maxlen=N)`; deferred
- [x] [Review][Defer] Symbol expiry codes `MNQM26/MESM26/M2KM26` require code change after each quarterly rollover — operational concern; deferred
- [x] [Review][Defer] `_log_trade` records `exit_time` as `timestamp`; `entry_time` parameter accepted but silently dropped [tier2_streaming_working.py:779,793] — spec says "timestamp" column without specifying entry vs exit; deferred

### File List

- `src/research/tier2_streaming_working.py` (updated throughout)
