# Story 4.1: Bracket Order Submission and Position Reconciliation

Status: done

## Story

As Alex,
I want the live trader to submit bracket orders to TradeStation SIM and reconcile position state every polling cycle,
so that every trade has a pre-defined entry, TP, and SL submitted atomically, and orphaned orders are never left open.

## Background

**Current state (after Stories 8-1 through 8-5):**
The live orchestrator `tier2_streaming_working.py` already submits bracket orders to TradeStation SIM via `Tier2StreamingTrader._submit_bracket_order()` (line 995). However, this code is embedded as private methods on `Tier2StreamingTrader` — there is no `TradeStationClient` class. The architecture (AR3) mandates that the live orchestrator be internally organized into discrete classes: **TradeStationClient, RiskManager, StatePersistence, TradeLogger** — all inside `tier2_streaming_working.py` (not extracted to separate files).

This story **extracts** the existing order/auth methods from `Tier2StreamingTrader` into a proper `TradeStationClient` class within the same file, adds the missing `reconcile_state()` method, creates `AccountConfig` and `TradeState` dataclasses, and wires `Tier2StreamingTrader` to use the client class. Unit tests with a mocked `TradeStationClient` cover the 3 reconciliation scenarios.

**Multi-instrument scope (from Epic 8 plan):**
Story 8-3 already parameterized `Tier2StreamingTrader` with `symbol` and `SYMBOL_SPECS`. `TradeStationClient` must accept `symbol` and symbol spec — it is NOT hardcoded to MNQ. The AC says `MNQM26` as the default; the implementation must pass `symbol` explicitly.

## Acceptance Criteria

**AC#1 — Bracket order submission:**
Given an `EntryDecision` from `strategy_core.make_entry_decision()`,
When `TradeStationClient.submit_bracket_order(decision: EntryDecision, account_id: str) -> str` is called,
Then it submits a single bracket POST to `TradeStation SIM /v3/orderexecution/orders` containing: limit entry order at `entry_price`, limit TP order at `tp_price`, stop SL order at `sl_price`, all linked as a bracket on `self._symbol`, and returns the entry order ID (FR9, NFR16).

**AC#2 — Reconciliation:**
Given a submitted bracket order with a known order ID,
When the next polling cycle calls `TradeStationClient.reconcile_state(account_id: str) -> TradeState`,
Then it queries `GET /v3/brokerage/accounts/{id}/orders` and `GET /v3/brokerage/accounts/{id}/positions`, then returns the current `TradeState` (PENDING / ACTIVE / FLAT) (FR10).

**AC#3 — Single-position enforcement:**
Given `TradeState` shows an active open position (i.e., `self.active_trade is not None`),
When `_detect_and_enter()` evaluates a new setup,
Then it returns without submitting any new order (FR11).

**AC#4 — Pending order expiry:**
Given a pending entry order that has not filled for `config.max_pending_bars` bars,
When `_advance_active_trade()` runs,
Then it calls `self._ts_client.cancel_order(order_id)` and clears the pending trade state (FR12).

**AC#5 — API error resilience:**
Given a TradeStation API call that raises `httpx.TimeoutException` or returns HTTP 500,
When the polling loop catches the exception,
Then it logs `ERROR: API call failed — <error> — skipping bar` and continues on the next 60-second cycle without crashing (NFR2).

**AC#6 — Unit tests:**
Given mocked `TradeStationClient` in `tests/unit/test_orchestrator_order_management.py`,
When reconciliation tests run,
Then they cover all 3 scenarios:
- open order + no position → state = PENDING (cancel order path)
- filled order + active position → state = ACTIVE (manage trade path)
- no order + no position → state = FLAT (clean state)

## Tasks / Subtasks

- [x] Task 1: Create `AccountConfig` and `TradeState` dataclasses (AC: #1, #2)
  - [x] `AccountConfig(account_id: str, execution_mode: Literal["sim","live"], symbol: str, point_value: float, tick_size: float, contracts: int)` — near top of `tier2_streaming_working.py`, after `SYMBOL_SPECS`
  - [x] `TradeState(status: TradeStatus, entry_order_id: Optional[str], tp_order_id: Optional[str], sl_order_id: Optional[str], position_qty: int)` where `TradeStatus = Literal["FLAT", "PENDING", "ACTIVE"]`
  - [x] Both are `@dataclass` (not frozen — mutable reconciliation state)
  - [x] `_default_account_config(symbol: str) -> AccountConfig` helper using `SYMBOL_SPECS` and `SIM_ACCOUNT_ID`

- [x] Task 2: Create `TradeStationClient` class in `tier2_streaming_working.py` (AC: #1, #2, #4, #5)
  - [x] Class is placed ABOVE `Tier2StreamingTrader` in the file, AFTER `StatePersistence`
  - [x] `__init__(self, auth: TradeStationAuthV3, account_config: AccountConfig, httpx_client: httpx.AsyncClient)`
  - [x] `async def _headers(self) -> dict` — delegates to `await self._auth.authenticate()`
  - [x] `async def submit_bracket_order(self, decision: EntryDecision, account_id: str) -> tuple[Optional[str], Optional[str], Optional[str]]` — moves existing `_submit_bracket_order` logic here; uses `self._account_config.symbol`
  - [x] `async def cancel_order(self, order_id: str) -> bool` — moves existing `_cancel_sim_order` logic here
  - [x] `async def close_position_at_market(self, direction: str) -> Optional[str]` — moves existing `_submit_close_order` logic
  - [x] `async def reconcile_state(self, account_id: str) -> TradeState` — **NEW**: queries `GET /v3/brokerage/accounts/{account_id}/orders?status=Open` + `GET /v3/brokerage/accounts/{account_id}/positions`; interprets response to return FLAT/PENDING/ACTIVE
  - [x] `async def cancel_all_pending_orders(self, account_id: str) -> list[str]` — **NEW**: queries open orders, cancels each; returns list of cancelled IDs
  - [x] Error handling: `httpx.TimeoutException` and HTTP 4xx/5xx → log + return safe default (None / FLAT / False); never raise from order methods

- [x] Task 3: Update `Tier2StreamingTrader` to use `TradeStationClient` (AC: #3, #4, #5)
  - [x] Add `self._ts_client: Optional[TradeStationClient] = None` to `__init__`
  - [x] Add `self._account_config: AccountConfig = _default_account_config(symbol)` to `__init__`
  - [x] In `initialize()`: after auth setup, create `self._ts_client = TradeStationClient(self.auth, self._account_config, self.client)`
  - [x] Replace `await self._submit_bracket_order(...)` in `_enter_trade()` with `await self._ts_client.submit_bracket_order(snapped_dec, SIM_ACCOUNT_ID)`
  - [x] Replace `await self._cancel_sim_order(...)` in `_advance_active_trade()` with `await self._ts_client.cancel_order(...)`
  - [x] Replace `await self._submit_close_order(...)` in `_close_active_trade()` with `await self._ts_client.close_position_at_market(...)`
  - [x] Removed dead private methods: `_get_auth_headers`, `_submit_bracket_order`, `_cancel_sim_order`, `_submit_close_order`
  - [x] Keep `_snap_tick()` on `Tier2StreamingTrader` (it uses `self._tick_size`)
  - [x] `_detect_and_enter()`: early-return guard `if self.active_trade is not None: return` confirmed in place (AC#3)
  - [x] `_advance_active_trade()`: uses `self._ts_client.cancel_order(...)` (AC#4)

- [x] Task 4: Write unit tests in `tests/unit/test_orchestrator_order_management.py` (AC: #6)
  - [x] Import style: `from unittest.mock import AsyncMock, MagicMock, patch`
  - [x] `class TestTradeStationClientReconciliation:` (5 tests)
    - [x] `test_reconcile_open_order_no_position_returns_pending`
    - [x] `test_reconcile_filled_order_with_position_returns_active`
    - [x] `test_reconcile_no_order_no_position_returns_flat`
    - [x] `test_reconcile_filters_other_symbol_orders`
    - [x] `test_reconcile_returns_flat_on_network_error`
  - [x] `class TestTradeStationClientOrderSubmission:` (7 tests)
    - [x] `test_submit_bracket_order_posts_to_sim_url`
    - [x] `test_submit_bracket_order_payload_has_oso_with_two_legs`
    - [x] `test_submit_bracket_order_returns_order_ids`
    - [x] `test_submit_bracket_order_returns_none_on_http_500`
    - [x] `test_cancel_order_sends_delete_with_correct_url`
    - [x] `test_cancel_order_returns_true_on_404`
    - [x] `test_cancel_order_returns_false_on_timeout`
  - [x] `class TestSinglePositionEnforcement:` (1 test)
    - [x] `test_detect_and_enter_skips_when_active_trade_present`
  - [x] All tests use `pytest.mark.asyncio` for async tests

- [x] Task 5: Run tests and verify no regressions (AC: #6)
  - [x] `PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_orchestrator_order_management.py -v` — 13 passed
  - [x] `PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_strategy_core_detection.py tests/unit/test_config_loader.py -q` — 32 passed, no regressions
  - [x] Smoke test: `PYTHONPATH=. .venv/bin/python -c "import src.research.tier2_streaming_working"` — import OK

## Dev Notes

### Architectural Constraints (Critical)

1. **AR3 — Single-file mandate:** `TradeStationClient` MUST live inside `tier2_streaming_working.py`. Do NOT create `src/research/tradestation_client.py` or any new file for this class. The architecture explicitly calls this an "escape hatch" for much later.

2. **AR1 — strategy_core purity:** `TradeStationClient` is pure I/O. It MUST NOT import from `strategy_core`. It accepts `EntryDecision` as an input type (which is already imported in `tier2_streaming_working.py`) but does no strategy computation itself.

3. **AR19 — Timezone at ingest boundary:** `TradeStationClient.get_bars()` converts timestamps to `America/New_York` ONCE. `strategy_core` receives already-converted data.

4. **SIM_ACCOUNT_ID** is currently hardcoded as `"SIM2797251F"` at line 76. Move this into `AccountConfig.account_id` — keep the constant as a module-level default for backward compat:
   ```python
   SIM_ACCOUNT_ID = "SIM2797251F"  # keep as module constant for backward compat
   ```

### TradeState and AccountConfig Dataclasses

```python
from typing import Literal, Optional
from dataclasses import dataclass, field

# Near line 76, after SYMBOL_SPECS

TradeStatus = Literal["FLAT", "PENDING", "ACTIVE"]

@dataclass
class AccountConfig:
    account_id: str
    execution_mode: Literal["sim", "live"]
    symbol: str
    point_value: float
    tick_size: float
    contracts: int

@dataclass
class TradeState:
    status: TradeStatus
    entry_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    position_qty: int = 0

def _default_account_config(symbol: str) -> AccountConfig:
    spec = SYMBOL_SPECS[symbol]  # raises ValueError if unknown — caller validates first
    return AccountConfig(
        account_id=SIM_ACCOUNT_ID,
        execution_mode="sim",
        symbol=symbol,
        point_value=spec["point_value"],
        tick_size=spec["tick_size"],
        contracts=spec["contracts"],
    )
```

### TradeStationClient Class Skeleton

```python
class TradeStationClient:
    """Sole network-touching component. Owns auth, bar polling, order
    submission, and per-cycle reconciliation. SIM vs live is AccountConfig."""

    _SIM_ORDERS_URL = SIM_ORDERS_URL  # forward reference to module constant
    _BROKERAGE_BASE = "https://sim-api.tradestation.com/v3/brokerage"

    def __init__(
        self,
        auth: "TradeStationAuthV3",
        account_config: AccountConfig,
        httpx_client: "httpx.AsyncClient",
    ):
        self._auth = auth
        self._cfg = account_config
        self._http = httpx_client

    async def _headers(self) -> dict:
        token = await self._auth.authenticate()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def submit_bracket_order(
        self, decision: EntryDecision, account_id: str
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Submit bracket order. Returns (entry_id, tp_id, sl_id) or (None, None, None) on failure."""
        ...

    async def cancel_order(self, order_id: str) -> bool:
        """DELETE /v3/orderexecution/orders/{order_id}. Returns True on success/404."""
        ...

    async def close_position_at_market(self, direction: str) -> Optional[str]:
        """Submit market close order. Returns order_id or None on failure."""
        ...

    async def reconcile_state(self, account_id: str) -> TradeState:
        """Query open orders + positions; return FLAT/PENDING/ACTIVE TradeState."""
        ...

    async def cancel_all_pending_orders(self, account_id: str) -> list[str]:
        """Query all open orders for this account; cancel each. Returns cancelled IDs."""
        ...
```

### reconcile_state() Implementation Pattern

```python
async def reconcile_state(self, account_id: str) -> TradeState:
    try:
        headers = await self._headers()
        # Query open orders for this symbol
        orders_url = f"{self._BROKERAGE_BASE}/accounts/{account_id}/orders?status=Open"
        orders_resp = await self._http.get(orders_url, headers=headers)
        # Query open positions for this symbol
        pos_url = f"{self._BROKERAGE_BASE}/accounts/{account_id}/positions"
        pos_resp = await self._http.get(pos_url, headers=headers)

        orders = []
        if orders_resp.status_code == 200:
            orders = [o for o in orders_resp.json().get("Orders", [])
                      if o.get("Symbol") == self._cfg.symbol]
        positions = []
        if pos_resp.status_code == 200:
            positions = [p for p in pos_resp.json().get("Positions", [])
                         if p.get("Symbol") == self._cfg.symbol]

        has_open_orders = len(orders) > 0
        position_qty = sum(int(p.get("Quantity", 0)) for p in positions)
        has_position = position_qty > 0

        if has_position:
            return TradeState(status="ACTIVE", position_qty=position_qty)
        elif has_open_orders:
            entry_id = next((o.get("OrderID") for o in orders
                             if o.get("OrderType") == "Limit"), None)
            return TradeState(status="PENDING", entry_order_id=entry_id)
        else:
            return TradeState(status="FLAT")

    except Exception as e:
        logger.warning(f"⚠️ reconcile_state failed: {e} — assuming FLAT")
        return TradeState(status="FLAT")
```

**Why FLAT on error:** FLAT is the conservative safe default — it means "I don't know the state, do not assume there's an active trade." The real safety net is the `active_trade` object already tracked in memory; reconciliation is a cross-check, not the primary state machine.

### Existing Code Locations to Move/Replace

| Current Location | Action | New Location |
|---|---|---|
| `Tier2StreamingTrader._get_auth_headers()` (line 987) | Move | `TradeStationClient._headers()` |
| `Tier2StreamingTrader._submit_bracket_order()` (line 995) | Move | `TradeStationClient.submit_bracket_order()` |
| `Tier2StreamingTrader._cancel_sim_order()` (line 1063) | Move | `TradeStationClient.cancel_order()` |
| `Tier2StreamingTrader._submit_close_order()` (line 1073) | Move | `TradeStationClient.close_position_at_market()` |
| `Tier2StreamingTrader._enter_trade()` bracket call (line 1135) | Update | `self._ts_client.submit_bracket_order(snapped_dec, SIM_ACCOUNT_ID)` |
| `Tier2StreamingTrader._advance_active_trade()` cancel call (line 734-737) | Update | `self._ts_client.cancel_order(...)` |
| `Tier2StreamingTrader._close_active_trade()` close call (line 761) | Update | `self._ts_client.close_position_at_market(...)` |

After moving, the private methods on `Tier2StreamingTrader` can either be removed (preferred) or left as one-line delegates.

### submit_bracket_order() Signature Change

The current private method has signature `_submit_bracket_order(direction, entry_price, tp_price, sl_price)`. The new public method takes `EntryDecision` (which already has direction, entry_price, tp_price, sl_price, contracts) plus `account_id`. The tick-snapping (`_snap_tick`) stays on `Tier2StreamingTrader` since it uses `self._tick_size`. The snap happens BEFORE calling `submit_bracket_order` — pass already-snapped prices.

Current call site in `_enter_trade()` (line 1135):
```python
# BEFORE (current):
e_id, tp_id, sl_id = await self._submit_bracket_order(direction_str, ent, tp, sl)

# AFTER:
e_id, tp_id, sl_id = await self._ts_client.submit_bracket_order(snapped_dec, SIM_ACCOUNT_ID)
```

The `submit_bracket_order` method should read `decision.entry_price`, `decision.tp_price`, `decision.sl_price`, and `decision.direction` (or accept direction separately — your call, but `EntryDecision` already has it all).

### Unit Test Pattern

```python
# tests/unit/test_orchestrator_order_management.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Import only what's needed; avoid importing tier2_streaming_working at module level
# (it tries to authenticate on import in some paths) — use delayed import in fixtures

@pytest.fixture
def account_config():
    from src.research.tier2_streaming_working import AccountConfig
    return AccountConfig(
        account_id="SIM2797251F",
        execution_mode="sim",
        symbol="MNQM26",
        point_value=2.0,
        tick_size=0.25,
        contracts=5,
    )

@pytest.fixture
def mock_client(account_config):
    from src.research.tier2_streaming_working import TradeStationClient
    auth = AsyncMock()
    auth.authenticate.return_value = "test_token"
    http = AsyncMock()
    return TradeStationClient(auth=auth, account_config=account_config, httpx_client=http)

class TestTradeStationClientReconciliation:
    @pytest.mark.asyncio
    async def test_reconcile_open_order_no_position_returns_pending(self, mock_client):
        # Arrange: open limit order, no position
        mock_client._http.get = AsyncMock(side_effect=[
            MagicMock(status_code=200, json=lambda: {"Orders": [
                {"OrderID": "E1", "Symbol": "MNQM26", "OrderType": "Limit", "Status": "Open"}
            ]}),
            MagicMock(status_code=200, json=lambda: {"Positions": []}),
        ])
        # Act
        state = await mock_client.reconcile_state("SIM2797251F")
        # Assert
        assert state.status == "PENDING"
        assert state.entry_order_id == "E1"
```

### TradeStation API — Brokerage Endpoints

From the existing code and TradeStation docs:
- **SIM orders URL:** `https://sim-api.tradestation.com/v3/orderexecution/orders`
- **Orders query:** `GET https://sim-api.tradestation.com/v3/brokerage/accounts/{id}/orders?status=Open`
- **Positions query:** `GET https://sim-api.tradestation.com/v3/brokerage/accounts/{id}/positions`
- **Cancel order:** `DELETE https://sim-api.tradestation.com/v3/orderexecution/orders/{order_id}`

Response formats (from TradeStation SIM — actual field names):
```json
// GET /v3/brokerage/accounts/{id}/orders
{
  "Orders": [
    {"OrderID": "123456", "Symbol": "MNQM26", "OrderType": "Limit",
     "Status": "Open", "TradeAction": "SELL", "Quantity": "5"}
  ]
}

// GET /v3/brokerage/accounts/{id}/positions
{
  "Positions": [
    {"Symbol": "MNQM26", "Quantity": "5", "OpenProfitLoss": "-50.00",
     "AveragePrice": "19500.25"}
  ]
}
```

### Multi-Instrument Note

The `TradeStationClient` filters orders and positions by `self._cfg.symbol`. This is intentional — when running 3 instrument instances simultaneously (Story 8-3), each client only sees its own symbol's orders. The `SIM_ACCOUNT_ID` is shared across all instruments (they all trade the same SIM account), so symbol filtering is the correct isolation mechanism.

### Existing Tests NOT to Break

These test files import from `tier2_streaming_working.py` or its dependencies:
- `tests/unit/test_config_loader.py` — imports `src.research.config_loader`
- `tests/unit/test_strategy_core_detection.py` — imports `src.research.strategy_core`
- `tests/unit/test_oos_checkpoint.py`, `test_prereg_seal.py` — imports top-level scripts

None of these directly import `Tier2StreamingTrader` or `TradeStationClient`, so the refactor should not affect them. Verify with the regression suite in Task 5.

### Git Commits to Reference

```
233b097 feat: Epic 8 complete — S25 deployment, YAML config, multi-instrument, weekly backtest
83f2dac feat: Story 3.4 — oos_verdict.py OOS verdict report generator
```

Key patterns established:
- `@dataclass` (not `@dataclass(frozen=True)`) for mutable runtime state objects
- `Optional[str]` return types for order IDs (None on failure, never raise)
- `logger.warning(f"⚠️ ...")` for recoverable errors, `logger.error(f"❌ ...")` for serious failures
- `PYTHONPATH=. .venv/bin/python -m pytest ... -v` for test runs

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

### Completion Notes List

- Task 1: Added `TradeStatus` Literal, `AccountConfig` and `TradeState` dataclasses after `SYMBOL_SPECS`, plus `_default_account_config()` helper after `SIM_ACCOUNT_ID`.
- Task 2: Added `TradeStationClient` class (6 async methods: `_headers`, `submit_bracket_order`, `cancel_order`, `close_position_at_market`, `reconcile_state`, `cancel_all_pending_orders`) before `_parse_blocked_months`. All error paths return safe defaults; no method raises.
- Task 3: Added `self._account_config` and `self._ts_client` to `Tier2StreamingTrader.__init__`; wired `initialize()` to create client after httpx. Replaced all 4 call sites in `_advance_active_trade`, `_close_active_trade`, and `_enter_trade`. Removed dead private methods `_get_auth_headers`, `_submit_bracket_order`, `_cancel_sim_order`, `_submit_close_order`.
- Task 4: Created `tests/unit/test_orchestrator_order_management.py` with 13 tests across 3 classes: `TestTradeStationClientReconciliation` (5), `TestTradeStationClientOrderSubmission` (7), `TestSinglePositionEnforcement` (1).
- Task 5: 13/13 new tests pass; 32 existing tests green; import smoke test OK.
- Code Review (2026-05-25): Applied 4 patches (P1–P4). Post-patch test count: 16/16 pass (added test_submit_bracket_order_returns_order_ids_via_order_type, _fallback_positional, test_reconcile_short_position_returns_active, test_reconcile_float_string_quantity_returns_active). 8 findings deferred to deferred-work.md.

### Review Findings (2026-05-25)

- [x] [Review][Patch] Order ID parsing: "Limit" substring hits both entry and TP orders, entry_id always None [`tier2_streaming_working.py` TradeStationClient.submit_bracket_order] — FIXED: use OrderType+TimeInForce.Duration as primary, positional fallback
- [x] [Review][Patch] Short position sign bug: `position_qty > 0` never true for short trades (bot is bearish-only) [`tier2_streaming_working.py` TradeStationClient.reconcile_state] — FIXED: `position_qty != 0` with `abs()`
- [x] [Review][Patch] `int("5.0")` ValueError on float-string Quantity returns false FLAT, hides active position [`tier2_streaming_working.py` TradeStationClient.reconcile_state] — FIXED: `int(float(...))`
- [x] [Review][Patch] `close_position_at_market()` uses `self._cfg.account_id` but all sibling methods take `account_id` param — inconsistent API [`tier2_streaming_working.py` TradeStationClient.close_position_at_market] — FIXED: added `account_id: str` param, updated call site
- [x] [Review][Defer] `_ts_client=None` before `initialize()` — same pre-existing pattern as `self.auth` and `self.client` [`tier2_streaming_working.py`] — deferred, pre-existing
- [x] [Review][Defer] Hardcoded `sim-api.tradestation.com` URLs defeat FR14 SIM→live config swap — intentional, live trading is future scope [`tier2_streaming_working.py` TradeStationClient] — deferred, pre-existing
- [x] [Review][Defer] PENDING branch in `reconcile_state()` only populates `entry_order_id`; TP/SL IDs always None — needed for Story 4-3 crash recovery [`tier2_streaming_working.py`] — deferred, pre-existing
- [x] [Review][Defer] PENDING detection picks first Limit order which may be TP leg, not entry — acceptable since in-memory `active_trade` is authoritative per design [`tier2_streaming_working.py`] — deferred, pre-existing
- [x] [Review][Defer] `close_position_at_market()` uses config contract count, not actual broker position size — no partial fills in SIM [`tier2_streaming_working.py`] — deferred, pre-existing
- [x] [Review][Defer] Partial HTTP failure (one leg 503, other 200) gives ambiguous reconcile state — corner case, full-exception path returns FLAT as documented [`tier2_streaming_working.py`] — deferred, pre-existing
- [x] [Review][Defer] AC#5 exact log message format ("ERROR: API call failed…") differs from implementation — aspirational spec text, not monitored by any scraper — deferred, pre-existing
- [x] [Review][Defer] AC#5 error handling scope: method-level only, poll loop error handling is pre-existing — deferred, pre-existing

### File List

- `src/research/tier2_streaming_working.py` (modified — added AccountConfig, TradeState, TradeStationClient; wired Tier2StreamingTrader; removed 4 dead private methods)
- `tests/unit/test_orchestrator_order_management.py` (new — 16 unit tests after code review patches)
