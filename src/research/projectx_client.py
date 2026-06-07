"""ProjectX execution adapter for TopstepX — drop-in replacement for TradeStationClient.

Public interface is identical to TradeStationClient so tier2_streaming_working.py
requires only a one-line swap.  One extra method, place_exit_orders(), is added to
support ProjectX's sequential bracket model (entry first, TP/SL after fill).

Account ID:
    ProjectX uses integer account IDs.  Set PROJECTX_ACCOUNT_ID env var to your
    account ID (e.g. export PROJECTX_ACCOUNT_ID=11542104).  Falls back to the
    known default — always override explicitly for safety.

Symbol mapping:
    TradeStation uses 'MNQM26'; ProjectX uses 'CON.F.US.MNQ.M26'.
    _to_contract_id() converts automatically.

Order IDs:
    ProjectX returns integers; tier2 stores them as Optional[str].
    All conversions happen here — callers see only strings.
"""

import logging
import os
from typing import TYPE_CHECKING, Optional

import httpx

if TYPE_CHECKING:
    from src.research.projectx_auth import ProjectXAuth

from src.research.strategy_core import Direction, EntryDecision

logger = logging.getLogger(__name__)

# ProjectX order-type and side enums
_TYPE_LIMIT = 1
_TYPE_MARKET = 2
_TYPE_STOP = 4
_SIDE_BUY = 0   # Bid
_SIDE_SELL = 1  # Ask

_BASE_URL = "https://api.topstepx.com/api"


def _to_contract_id(ts_symbol: str) -> str:
    """Convert TradeStation symbol ('MNQM26') → ProjectX contract ID ('CON.F.US.MNQ.M26')."""
    root = ts_symbol[:3]       # 'MNQ'
    expiry = ts_symbol[3:]     # 'M26'
    return f"CON.F.US.{root}.{expiry}"


class ProjectXClient:
    """Execution adapter for TopstepX via ProjectX Gateway API.

    Mirrors TradeStationClient's public interface:
        submit_bracket_order  — places entry limit only; TP/SL deferred to fill
        cancel_order          — cancels any open order by string ID
        close_position_at_market — market order to flatten position
        reconcile_state       — returns TradeState (FLAT / PENDING / ACTIVE)
        cancel_all_pending_orders — cancels all open orders for this contract

    Extra method (ProjectX-only):
        place_exit_orders     — places TP limit + SL stop after entry fills;
                                called by Tier2StreamingTrader._advance_active_trade()
    """

    def __init__(
        self,
        auth: "ProjectXAuth",
        account_config,          # AccountConfig — imported lazily to avoid circular
        httpx_client: httpx.AsyncClient,
        projectx_account_id: Optional[int] = None,
    ) -> None:
        self._auth = auth
        self._cfg = account_config
        self._http = httpx_client
        self._contract_id = _to_contract_id(account_config.symbol)

        env_id = os.environ.get("PROJECTX_ACCOUNT_ID", "")
        if projectx_account_id is not None:
            self._account_id = projectx_account_id
        elif env_id:
            self._account_id = int(env_id)
        else:
            # Known default for this Topstep account — set PROJECTX_ACCOUNT_ID to override
            self._account_id = 11542104
            logger.warning(
                "PROJECTX_ACCOUNT_ID env var not set — using default 11542104. "
                "Set it explicitly: export PROJECTX_ACCOUNT_ID=11542104"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _headers(self) -> dict:
        token = await self._auth.authenticate()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _place_order(self, payload: dict) -> Optional[int]:
        """POST /Order/place — returns orderId (int) or None on failure."""
        try:
            headers = await self._headers()
            resp = await self._http.post(f"{_BASE_URL}/Order/place", json=payload, headers=headers)
            if resp.status_code not in (200, 201):
                logger.warning(f"⚠️ ProjectX /Order/place HTTP {resp.status_code}")
                return None
            data = resp.json()
            if not data.get("success"):
                logger.warning(f"⚠️ ProjectX order rejected: {data.get('errorMessage')}")
                return None
            return int(data["orderId"])
        except Exception as exc:
            logger.warning(f"⚠️ ProjectX /Order/place exception: {exc}")
            return None

    # ------------------------------------------------------------------
    # Public interface (mirrors TradeStationClient)
    # ------------------------------------------------------------------

    async def submit_bracket_order(
        self, decision: "EntryDecision", account_id: str
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Place entry limit order only.  TP/SL are deferred until fill is detected.

        Returns (entry_id, None, None) — tier2 calls place_exit_orders() when
        pending_entry transitions to False to place the protective orders.

        Returns (None, None, None) on any failure.
        """
        is_short = decision.direction == Direction.BEARISH
        entry_side = _SIDE_SELL if is_short else _SIDE_BUY
        payload = {
            "accountId": self._account_id,
            "contractId": self._contract_id,
            "type": _TYPE_LIMIT,
            "side": entry_side,
            "size": decision.contracts,
            "limitPrice": float(decision.entry_price),
        }
        oid = await self._place_order(payload)
        if oid is None:
            return None, None, None
        entry_id = str(oid)
        logger.info(f"✓ ProjectX entry limit #{entry_id} @ {decision.entry_price:.2f} (TP/SL deferred)")
        return entry_id, None, None

    async def place_exit_orders(
        self, decision: "EntryDecision", account_id: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Place TP limit + SL stop after entry fills.

        Called by Tier2StreamingTrader._advance_active_trade() on the bar where
        pending_entry transitions False (price touches FVG midpoint).

        Returns (tp_id, sl_id) — either or both may be None on failure.
        An unprotected position after failure will be caught by the time-stop
        which issues a market close.
        """
        is_short = decision.direction == Direction.BEARISH
        exit_side = _SIDE_BUY if is_short else _SIDE_SELL

        # Take-profit limit
        tp_id: Optional[str] = None
        oid = await self._place_order({
            "accountId": self._account_id,
            "contractId": self._contract_id,
            "type": _TYPE_LIMIT,
            "side": exit_side,
            "size": decision.contracts,
            "limitPrice": float(decision.tp_price),
        })
        if oid is not None:
            tp_id = str(oid)
            logger.info(f"✓ ProjectX TP limit #{tp_id} @ {decision.tp_price:.2f}")
        else:
            logger.error("⚠️ ProjectX TP limit order FAILED — position partially unprotected")

        # Stop-loss stop
        sl_id: Optional[str] = None
        oid = await self._place_order({
            "accountId": self._account_id,
            "contractId": self._contract_id,
            "type": _TYPE_STOP,
            "side": exit_side,
            "size": decision.contracts,
            "stopPrice": float(decision.sl_price),
        })
        if oid is not None:
            sl_id = str(oid)
            logger.info(f"✓ ProjectX SL stop #{sl_id} @ {decision.sl_price:.2f}")
        else:
            logger.error("⚠️ ProjectX SL stop order FAILED — position partially unprotected")

        return tp_id, sl_id

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by string ID.  Returns True on success or 'already gone'."""
        try:
            headers = await self._headers()
            resp = await self._http.post(
                f"{_BASE_URL}/Order/cancel",
                json={"accountId": self._account_id, "orderId": int(order_id)},
                headers=headers,
            )
            ok = resp.status_code in (200, 201, 204)
            if not ok:
                logger.warning(f"⚠️ ProjectX cancel #{order_id} HTTP {resp.status_code}")
            return ok
        except Exception as exc:
            logger.warning(f"⚠️ ProjectX cancel #{order_id} exception: {exc}")
            return False

    async def close_position_at_market(
        self, direction: str, account_id: str, contracts: Optional[int] = None
    ) -> Optional[str]:
        """Submit a market order to flatten the open position.  Returns order ID or None."""
        close_side = _SIDE_BUY if direction == "SHORT" else _SIDE_SELL
        qty = contracts if contracts is not None else self._cfg.contracts
        oid = await self._place_order({
            "accountId": self._account_id,
            "contractId": self._contract_id,
            "type": _TYPE_MARKET,
            "side": close_side,
            "size": qty,
        })
        if oid is not None:
            logger.info(f"✓ ProjectX market close #{oid}")
            return str(oid)
        return None

    async def reconcile_state(self, account_id: str):
        """Query ProjectX for open orders + positions; return TradeState.

        Conservative safe default: FLAT on any error.
        """
        # Import here to avoid circular — TradeState is defined in tier2
        from src.research.tier2_streaming_working import TradeState

        try:
            headers = await self._headers()
            orders_resp, pos_resp = await _parallel_requests(
                self._http,
                headers,
                f"{_BASE_URL}/Order/searchOpen",
                f"{_BASE_URL}/Position/searchOpen",
                {"accountId": self._account_id, "contractId": self._contract_id},
            )

            position_qty = 0
            if pos_resp.status_code == 200:
                position_qty = sum(
                    int(p.get("size", 0))
                    for p in pos_resp.json().get("positions", [])
                )

            if position_qty != 0:
                return TradeState(status="ACTIVE", position_qty=position_qty)

            symbol_orders = []
            if orders_resp.status_code == 200:
                symbol_orders = [
                    o for o in orders_resp.json().get("orders", [])
                    if o.get("contractId") == self._contract_id
                ]

            if symbol_orders:
                # Identify the pending entry: first open limit order
                entry_id = next(
                    (str(o["id"]) for o in symbol_orders if o.get("type") == _TYPE_LIMIT),
                    None,
                )
                return TradeState(status="PENDING", entry_order_id=entry_id)

            return TradeState(status="FLAT")

        except Exception as exc:
            logger.warning(f"⚠️ ProjectX reconcile_state failed: {exc} — assuming FLAT")
            return TradeState(status="FLAT")

    async def cancel_all_pending_orders(self, account_id: str) -> list:
        """Cancel all open orders for this contract.  Returns list of cancelled IDs."""
        cancelled: list = []
        try:
            headers = await self._headers()
            resp = await self._http.post(
                f"{_BASE_URL}/Order/searchOpen",
                json={"accountId": self._account_id, "contractId": self._contract_id},
                headers=headers,
            )
            if resp.status_code != 200:
                return cancelled
            orders = [
                o for o in resp.json().get("orders", [])
                if o.get("contractId") == self._contract_id
            ]
            for order in orders:
                oid = str(order.get("id", ""))
                if oid and await self.cancel_order(oid):
                    cancelled.append(oid)
        except Exception as exc:
            logger.warning(f"⚠️ cancel_all_pending_orders failed: {exc}")
        return cancelled


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

async def _parallel_requests(
    client: httpx.AsyncClient,
    headers: dict,
    url_a: str,
    url_b: str,
    body: dict,
):
    """Fire two POST requests concurrently and return both responses."""
    import asyncio
    resp_a, resp_b = await asyncio.gather(
        client.post(url_a, json=body, headers=headers),
        client.post(url_b, json=body, headers=headers),
        return_exceptions=False,
    )
    return resp_a, resp_b
