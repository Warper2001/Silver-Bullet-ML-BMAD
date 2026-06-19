"""TradeStation SIM order mirror + position-scaling testbed for the combine traders.

Both YANK (``yank_streaming_working.py``) and MIM-NB (``mim_nb_live.py``) execute
on a REAL Topstep 50K combine account via ``ProjectXClient``. This module lets
them ALSO place a copy of every order on a TradeStation SIM account, without
touching strategy logic and — critically — without any ability to delay, block,
or crash the authoritative combine path.

PURPOSE (Alex, 2026-06-18): the SIM account is a dress-rehearsal to validate the
strategy AND the position-SCALING logic before real money. Roadmap: pass the
combine -> repoint this SIM strategy at a TradeStation 10K LIVE account -> after
+$5K profit, scale position size up as margin allows. So the mirror grows its OWN
margin-aware, equity-driven contract size on SIM (see :class:`SimScaler`), diverging
from the combine's fixed size on purpose — that divergence IS the scaling test.

Design (party-mode roundtable, 2026-06-18; Winston / Amelia / John / Dr. Quinn):

* **Universal seam.** Every order both bots place funnels through
  ``ProjectXClient._place_order(payload)`` (``submit_bracket_order``,
  ``place_exit_orders`` and ``close_position_at_market`` all call it internally;
  MIM-NB calls it directly), and every cancel through ``cancel_order``.
  :class:`MirrorProjectXClient` subclasses ``ProjectXClient`` and overrides ONLY
  those two methods. All reads (``is_order_open`` / ``reconcile_state`` /
  ``net_position`` / ``account_balance``) are inherited untouched and stay 100%
  ProjectX-authoritative.

* **Strict subordination — the mirror can never harm the combine** (Dr. Quinn's
  failure axes): the authoritative ``await super()._place_order`` returns first;
  the mirror action is handed to a bounded queue via a non-awaiting ``put_nowait``
  and processed by a SEPARATE worker task (Axis 1); the worker wraps every TS call
  in a total exception firewall (Axis 2); the mirror never feeds a value back to
  the strategy (Axis 3); it owns its own ``httpx.AsyncClient`` (Axis 5); the queue
  is bounded, drop-oldest on overflow (Axis 6).

* **Scaling (SIM side only).** :class:`SimScaler` sizes each SIM order by current
  SIM-account equity via fractional Kelly, gated by a +$5K high-water unlock latch
  and capped by margin. A periodic equity poll (own task, firewalled) feeds it.
  Combine orders are NEVER resized — scaling lives entirely on the mirror.

Enabled per-service behind an env var (default OFF):
``YANK_MIRROR_TS_SIM=1`` / ``MIM_MIRROR_TS_SIM=1``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from src.research.projectx_client import ProjectXClient

logger = logging.getLogger(__name__)

# TradeStation SIM endpoints / account (mirrors the constants in the trader modules).
SIM_ACCOUNT_ID = "SIM2797251F"
SIM_ORDERS_URL = "https://sim-api.tradestation.com/v3/orderexecution/orders"
SIM_ORDER_BASE = "https://sim-api.tradestation.com/v3/orderexecution/orders"
SIM_BROKERAGE_BASE = "https://sim-api.tradestation.com/v3/brokerage"

# ProjectX order-type / side enums (from ProjectXClient).
_PX_TYPE_LIMIT = 1
_PX_TYPE_MARKET = 2
_PX_TYPE_STOP = 4
_PX_SIDE_BUY = 0
_PX_SIDE_SELL = 1


def _contract_to_ts_symbol(contract_id: str) -> str:
    """Convert ProjectX contract id back to a TradeStation symbol.

    'CON.F.US.MNQ.U26' -> 'MNQU26'. Derives the symbol from the order payload so
    the mirror automatically tracks the front-month contract roll.
    """
    parts = contract_id.split(".")
    return f"{parts[-2]}{parts[-1]}"


def px_payload_to_ts_order(payload: dict, sim_account_id: str,
                           qty_override: Optional[int] = None) -> Optional[dict]:
    """Translate a ProjectX ``/Order/place`` payload into a single-leg TS SIM order.

    ``qty_override`` (when set) replaces the combine's contract count with the
    SIM-scaled size. Returns ``None`` for an unrecognised order type (the leg is
    then skipped — the mirror silently drops what it cannot represent).
    """
    ptype = payload.get("type")
    if ptype == _PX_TYPE_LIMIT:
        order_type, duration = "Limit", "GTC"
    elif ptype == _PX_TYPE_MARKET:
        order_type, duration = "Market", "DAY"
    elif ptype == _PX_TYPE_STOP:
        order_type, duration = "StopMarket", "GTC"
    else:
        return None

    qty = qty_override if qty_override is not None else payload["size"]
    action = "BUY" if payload.get("side") == _PX_SIDE_BUY else "SELL"
    ts_order: dict = {
        "AccountID": sim_account_id,
        "Symbol": _contract_to_ts_symbol(payload["contractId"]),
        "Quantity": str(int(qty)),
        "OrderType": order_type,
        "TradeAction": action,
        "TimeInForce": {"Duration": duration},
        "Route": "Intelligent",
    }
    if order_type == "Limit":
        ts_order["LimitPrice"] = str(payload["limitPrice"])
    elif order_type == "StopMarket":
        ts_order["StopPrice"] = str(payload["stopPrice"])
    return ts_order


class SimScaler:
    """Equity-driven, margin-aware fractional-Kelly position sizer for the SIM mirror.

    Sizing rule (Alex's choices, 2026-06-18):

    * **+$5K high-water unlock latch.** Trade ``base_contracts`` until realised
      profit (equity - start_equity) first reaches ``unlock_profit``; once that
      high-water event happens the latch stays open (rolling high-water — we never
      relock just because of a later drawdown).
    * **Fractional Kelly, symmetric.** After unlock,
      ``contracts = kelly_fraction * edge_mean * usable_equity / edge_std**2``
      where edge_mean/edge_std are the strategy's per-contract, per-trade P&L mean
      and std (dollars). Because size is proportional to current equity, it scales
      DOWN automatically on drawdown — the symmetric behaviour requested.
    * **Margin cap.** ``contracts <= margin_buffer * buying_power / margin_per_contract``.
    * **Hard ceiling** ``max_contracts`` and floor ``base_contracts``.

    Sizing is TTL-cached so every leg of one trade (entry + TP/SL + close, placed
    seconds apart) shares a single contract count — no unbalanced SIM brackets.

    NOTE: ``edge_mean``/``edge_std`` defaults are documented PLACEHOLDERS pending
    calibration from the OOS backtests; ``equity_fraction`` < 1 accounts for two
    strategies sharing one SIM account (prefer a separate SIM account per strategy).
    """

    def __init__(
        self,
        strategy: str,
        base_contracts: int = 1,
        unlock_profit: float = 5000.0,
        kelly_fraction: float = 0.5,
        edge_mean: float = 30.0,     # PLACEHOLDER $/contract/trade — confirm from OOS
        edge_std: float = 400.0,     # PLACEHOLDER $/contract/trade — confirm from OOS
        margin_per_contract: float = 200.0,  # PLACEHOLDER MNQ day-margin — confirm
        margin_buffer: float = 0.8,
        max_contracts: int = 50,
        equity_fraction: float = 0.5,  # shared SIM account → allocate half per strategy
        ttl_seconds: float = 120.0,
        state_path: Optional[Path] = None,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self.strategy = strategy
        self.base_contracts = base_contracts
        self.unlock_profit = unlock_profit
        self.kelly_fraction = kelly_fraction
        self.edge_mean = edge_mean
        self.edge_std = edge_std
        self.margin_per_contract = margin_per_contract
        self.margin_buffer = margin_buffer
        self.max_contracts = max_contracts
        self.equity_fraction = equity_fraction
        self.ttl_seconds = ttl_seconds
        self._log = log or logger
        self._state_path = state_path

        self.equity: Optional[float] = None
        self.buying_power: Optional[float] = None
        self.start_equity: Optional[float] = None
        self.hwm: Optional[float] = None
        self.unlocked: bool = False
        self._cached_contracts: Optional[int] = None
        self._cached_at: float = 0.0
        self._load_state()

    # -- persistence ---------------------------------------------------------

    def _load_state(self) -> None:
        if self._state_path and self._state_path.exists():
            try:
                d = json.loads(self._state_path.read_text())
                self.start_equity = d.get("start_equity")
                self.hwm = d.get("hwm")
                self.unlocked = bool(d.get("unlocked", False))
            except Exception as exc:  # noqa: BLE001
                self._log.warning("SimScaler[%s] state load failed: %s", self.strategy, exc)

    def _save_state(self) -> None:
        if not self._state_path:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps({
                "start_equity": self.start_equity,
                "hwm": self.hwm,
                "unlocked": self.unlocked,
            }))
        except Exception as exc:  # noqa: BLE001
            self._log.warning("SimScaler[%s] state save failed: %s", self.strategy, exc)

    # -- inputs --------------------------------------------------------------

    def update_equity(self, equity: float, buying_power: Optional[float] = None) -> None:
        """Feed a fresh SIM-account equity reading (from the periodic poll)."""
        if equity is None:
            return
        self.equity = equity
        self.buying_power = buying_power
        if self.start_equity is None:
            self.start_equity = equity
        self.hwm = equity if self.hwm is None else max(self.hwm, equity)
        if not self.unlocked and (equity - self.start_equity) >= self.unlock_profit:
            self.unlocked = True
            self._log.info("SimScaler[%s] UNLOCKED — equity %.2f reached start %.2f + $%.0f",
                           self.strategy, equity, self.start_equity, self.unlock_profit)
        self._cached_contracts = None  # force recompute on next sizing
        self._save_state()

    # -- output --------------------------------------------------------------

    def _compute_contracts(self) -> int:
        if self.equity is None or self.start_equity is None:
            return self.base_contracts
        if not self.unlocked:
            return self.base_contracts  # building the $5K cushion at base size
        if self.edge_std <= 0:
            return self.base_contracts
        usable = self.equity * self.equity_fraction
        raw = self.kelly_fraction * self.edge_mean * usable / (self.edge_std ** 2)
        contracts = int(math.floor(raw))
        if self.buying_power is not None and self.margin_per_contract > 0:
            cap = int(math.floor(self.margin_buffer * self.buying_power / self.margin_per_contract))
            contracts = min(contracts, cap)
        contracts = max(self.base_contracts, contracts)
        contracts = min(contracts, self.max_contracts)
        return contracts

    def target_contracts(self) -> int:
        """SIM contract count for the current trade (TTL-cached so a trade's legs match)."""
        now = time.monotonic()
        if self._cached_contracts is not None and (now - self._cached_at) < self.ttl_seconds:
            return self._cached_contracts
        contracts = self._compute_contracts()
        self._cached_contracts = contracts
        self._cached_at = now
        return contracts


class InvVolScaler:
    """Fixed-size scaler for the inverse-vol allocation PAPER-TRACK (SIM only).

    Unlike :class:`SimScaler` (which grows size by equity to dress-rehearse the
    eventual live scaling), this holds a CONSTANT per-bot contract count so the
    SIM account runs the two bots at a candidate inverse-vol allocation while the
    real combine keeps its live size. Comparing realized giveback-from-HWM
    between the two answers the open question (party-mode 2026-06-19; Quinn vs
    Mary, John's ruling): does trimming YANK's weight actually reduce giveback?

    First experiment (Alex, 2026-06-19): YANK 1ct / MIM 1ct in SIM (treatment)
    vs the live combine's YANK 2ct / MIM 1ct (control) — a sign-stable one-notch
    YANK trim, minimal assumptions, no dependence on the N=13 correlation lift.

    Duck-types the interface :class:`TSSimMirror` uses: ``strategy``,
    ``target_contracts()`` and ``update_equity()``. Selected via ``SIM_INVVOL=1``.
    """

    def __init__(self, strategy: str, contracts: int = 1,
                 log: Optional[logging.Logger] = None) -> None:
        self.strategy = strategy
        self.contracts = int(contracts)
        self._log = log or logger
        self.equity: Optional[float] = None  # recorded for the equity log only

    def update_equity(self, equity: float, buying_power: Optional[float] = None) -> None:
        """Record the latest SIM equity (for the equity-curve log). Size is fixed,
        so the reading never changes the contract count."""
        if equity is not None:
            self.equity = equity

    def target_contracts(self) -> int:
        return self.contracts


class TSSimMirror:
    """Best-effort TradeStation SIM order mirror running on its own worker task.

    Submit/cancel intents are *enqueued* (never awaited) by the authoritative
    coroutine; a single background task drains the queue FIFO and talks to TS.
    Every TS call is wrapped in a total exception firewall. An optional
    :class:`SimScaler` resizes SIM orders independently of the combine.
    """

    def __init__(
        self,
        ts_auth,
        sim_account_id: str = SIM_ACCOUNT_ID,
        orders_url: str = SIM_ORDERS_URL,
        order_base_url: str = SIM_ORDER_BASE,
        maxsize: int = 256,
        http: Optional[httpx.AsyncClient] = None,
        scaler=None,
        equity_poll_interval: float = 30.0,
        equity_log_path: Optional[Path] = None,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self._auth = ts_auth
        self._sim_account = sim_account_id
        self._orders_url = orders_url
        self._order_base = order_base_url.rstrip("/")
        self._log = log or logger
        self._queue: "asyncio.Queue[tuple]" = asyncio.Queue(maxsize=maxsize)
        self._idmap: dict[str, str] = {}  # px_order_id -> ts_order_id
        self._http = http
        self._owns_http = http is None
        self._task: Optional[asyncio.Task] = None
        self._scaler = scaler
        self._equity_poll_interval = equity_poll_interval
        self._equity_log_path = equity_log_path  # SIM equity-curve CSV (giveback evidence)
        self._poll_task: Optional[asyncio.Task] = None
        self.dropped = 0  # count of queue-overflow drops (observability)

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=15.0)
        if self._task is None:
            self._task = asyncio.create_task(self._run())
        if self._scaler is not None and self._poll_task is None:
            self._poll_task = asyncio.create_task(self._equity_poll_loop())
        self._log.info("TS SIM mirror started (account %s, queue=%d, scaling=%s)",
                       self._sim_account, self._queue.maxsize,
                       "ON" if self._scaler else "off")

    async def stop(self) -> None:
        for task in (self._task, self._poll_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
        self._task = self._poll_task = None
        if self._owns_http and self._http is not None:
            try:
                await self._http.aclose()
            except Exception:  # noqa: BLE001
                pass
            self._http = None

    # -- producer side (called from the authoritative coroutine) -------------
    # These MUST be non-blocking and MUST NOT raise. They only enqueue.

    def submit(self, px_order_id, payload: dict) -> None:
        """Enqueue a 'place' mirror of a ProjectX order. Non-blocking, never raises."""
        self._offer(("place", str(px_order_id), dict(payload)))

    def cancel(self, px_order_id) -> None:
        """Enqueue a 'cancel' mirror of a ProjectX cancel. Non-blocking, never raises."""
        self._offer(("cancel", str(px_order_id), None))

    def _offer(self, item: tuple) -> None:
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            # Drop-oldest (Axis 6): discard one item, then enqueue the new one.
            try:
                self._queue.get_nowait()
                self.dropped += 1
            except Exception:  # noqa: BLE001
                pass
            try:
                self._queue.put_nowait(item)
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001 — producer must never raise into the combine path
            pass

    # -- consumer side (the isolated worker task) ----------------------------

    async def _run(self) -> None:
        while True:
            action, px_oid, payload = await self._queue.get()
            try:
                if action == "place":
                    await self._do_place(px_oid, payload)
                elif action == "cancel":
                    await self._do_cancel(px_oid)
            except Exception as exc:  # noqa: BLE001 — total firewall (Axis 2)
                self._log.warning("TS SIM mirror %s failed (px #%s): %s", action, px_oid, exc)
            finally:
                self._queue.task_done()

    async def _headers(self) -> dict:
        token = await self._auth.authenticate()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _do_place(self, px_oid: str, payload: dict) -> None:
        qty_override = self._scaler.target_contracts() if self._scaler else None
        ts_order = px_payload_to_ts_order(payload, self._sim_account, qty_override)
        if ts_order is None:
            self._log.debug("TS SIM mirror: skipping unmappable order type %s", payload.get("type"))
            return
        headers = await self._headers()
        resp = await self._http.post(self._orders_url, headers=headers, json=ts_order)
        if resp.status_code not in (200, 201):
            self._log.warning("TS SIM mirror place HTTP %s: %s", resp.status_code, resp.text[:160])
            return
        orders = resp.json().get("Orders", [])
        ts_oid = orders[0].get("OrderID") if orders else None
        if ts_oid:
            self._idmap[px_oid] = ts_oid
            self._log.info("TS SIM mirror: px #%s -> ts #%s (%s %s x%s)",
                           px_oid, ts_oid, ts_order["TradeAction"], ts_order["OrderType"],
                           ts_order["Quantity"])
        else:
            self._log.warning("TS SIM mirror place: no OrderID in response for px #%s", px_oid)

    async def _do_cancel(self, px_oid: str) -> None:
        ts_oid = self._idmap.pop(px_oid, None)
        if ts_oid is None:
            return  # no TS twin (place failed / never mirrored) — nothing to cancel
        headers = await self._headers()
        url = f"{self._order_base}/{ts_oid}"
        resp = await self._http.delete(url, headers=headers)
        if resp.status_code not in (200, 204, 404):
            self._log.warning("TS SIM mirror cancel HTTP %s for ts #%s", resp.status_code, ts_oid)

    # -- equity poll (own isolated task; feeds the scaler) -------------------

    async def _equity_poll_loop(self) -> None:
        while True:
            try:
                await self._poll_balances()
            except Exception as exc:  # noqa: BLE001 — firewall; scaler keeps last reading
                self._log.warning("TS SIM mirror equity poll failed: %s", exc)
            await asyncio.sleep(self._equity_poll_interval)

    async def _poll_balances(self) -> None:
        headers = await self._headers()
        url = f"{SIM_BROKERAGE_BASE}/accounts/{self._sim_account}/balances"
        resp = await self._http.get(url, headers=headers)
        if resp.status_code != 200:
            return
        balances = resp.json().get("Balances", [])
        if not balances:
            return
        b = balances[0]
        equity = _to_float(b.get("Equity")) or _to_float(b.get("CashBalance"))
        bp = _to_float(b.get("BuyingPower"))
        if equity is not None and self._scaler is not None:
            self._scaler.update_equity(equity, bp)
        if equity is not None:
            self._log_equity(equity, bp)

    def _log_equity(self, equity: float, bp: Optional[float]) -> None:
        """Append one SIM equity-curve row (for offline giveback-from-HWM analysis).

        No-op unless ``equity_log_path`` was set. Firewalled: a logging failure
        never propagates into the poll loop. NOTE: both bots mirror to the same
        SIM account, so this curve is the SHARED-account (portfolio) equity — which
        is exactly the series whose giveback we want for the inverse-vol test."""
        p = self._equity_log_path
        if not p:
            return
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            new = not p.exists()
            ct = self._scaler.target_contracts() if self._scaler else ""
            with open(p, "a") as f:
                if new:
                    f.write("ts_utc,equity,buying_power,contracts\n")
                f.write(f"{datetime.now(timezone.utc).isoformat()},{equity},"
                        f"{bp if bp is not None else ''},{ct}\n")
        except Exception as exc:  # noqa: BLE001 — never break the poll loop
            self._log.warning("TS SIM equity log failed: %s", exc)


def _to_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class MirrorProjectXClient(ProjectXClient):
    """ProjectXClient that ALSO best-effort mirrors every order to TradeStation SIM.

    Overrides only the two universal order-mutation seams. The authoritative
    ProjectX result is always returned verbatim; the mirror is enqueued
    fire-and-forget and can never delay, block, or fail the combine call.
    """

    def __init__(self, *args, ts_mirror: TSSimMirror, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ts_mirror = ts_mirror

    async def _place_order(self, payload: dict) -> Optional[int]:
        px_oid = await super()._place_order(payload)  # AUTHORITATIVE — returned unchanged
        if px_oid is not None:
            try:
                self._ts_mirror.submit(px_oid, payload)
            except Exception:  # noqa: BLE001 — never let the mirror touch the combine path
                pass
        return px_oid

    async def cancel_order(self, order_id) -> bool:
        ok = await super().cancel_order(order_id)  # AUTHORITATIVE
        try:
            self._ts_mirror.cancel(order_id)
        except Exception:  # noqa: BLE001
            pass
        return ok
