"""TradeStation SIM futures execution client for the Thursday-short (MBT/MET).

Phase 1 of buildplan_thursday_short_ts_sim_mbt_met.md.

Symbol-isolated on the shared futures SIM account (crypto roots MBT/MET are
distinct from the MNQ that YANK/MIM trade, so position/PnL attribution is exact).
All read methods are side-effect-free; ``place_order`` is the ONLY state-changing
call. Position state must be confirmed via ``get_open_positions`` (broker truth),
never assumed from an order ack — see G1/G2 in the build plan.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

from src.data.auth_v3 import TradeStationAuthV3

logger = logging.getLogger(__name__)

MD_BASE = "https://api.tradestation.com/v3/marketdata"
SIM_ORDERS_URL = "https://sim-api.tradestation.com/v3/orderexecution/orders"
SIM_BROKERAGE = "https://sim-api.tradestation.com/v3/brokerage"
MONTH_CODES = "FGHJKMNQUVXZ"          # 1..12 -> CME month letters
CRYPTO_ROOTS = ("MBT", "MET")         # the only symbols this client ever touches


class TradeStationThursdayClient:
    """Thin async client: front-month resolution, quotes, positions, market orders."""

    def __init__(self, auth: TradeStationAuthV3, account_id: str,
                 http: httpx.AsyncClient, roll_buffer_days: int = 3):
        self._auth = auth
        self._account = account_id
        self._http = http
        self._roll_buffer = roll_buffer_days
        self._front: dict[str, str] = {}
        self._front_date: Optional[str] = None

    async def _headers(self) -> dict:
        token = await self._auth.authenticate()
        return {"Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json"}

    # ---- read-only: market data -------------------------------------------

    async def quotes(self, symbols: list[str]) -> dict:
        h = await self._headers()
        r = await self._http.get(f"{MD_BASE}/quotes/" + ",".join(symbols), headers=h)
        r.raise_for_status()
        return {q["Symbol"]: q for q in r.json().get("Quotes", [])}

    async def last_price(self, symbol: str) -> Optional[float]:
        q = (await self.quotes([symbol])).get(symbol, {})
        v = q.get("Last")
        return float(v) if v not in (None, "") else None

    async def resolve_front_month(self, now: Optional[datetime] = None) -> dict[str, str]:
        """Active MBTxNN/METxNN, rolling ``roll_buffer_days`` before expiry.

        Returns e.g. {'MBT': 'MBTN26', 'MET': 'METN26'}. Cached per UTC date.
        Picks the earliest monthly contract whose ExpirationDate is more than
        ``roll_buffer`` days out, so a near-expiry front (where liquidity has
        already migrated) is skipped — avoids trading a contract that expires
        inside the Thursday hold.
        """
        now = now or datetime.now(timezone.utc)
        key = now.strftime("%Y-%m-%d")
        if self._front_date == key and self._front:
            return self._front
        cutoff = now + timedelta(days=self._roll_buffer)
        out: dict[str, str] = {}
        for root in CRYPTO_ROOTS:
            cands = []
            for i in range(5):
                mm = (now.month - 1 + i) % 12 + 1
                yy = now.year + (now.month - 1 + i) // 12
                cands.append(f"{root}{MONTH_CODES[mm - 1]}{yy % 100:02d}")
            chosen = None
            for s in cands:
                h = await self._headers()
                rd = await self._http.get(f"{MD_BASE}/symbols/{s}", headers=h)
                for sd in rd.json().get("Symbols", []):
                    exp = sd.get("ExpirationDate")
                    if not exp:
                        continue
                    expdt = datetime.fromisoformat(exp.replace("Z", "+00:00"))
                    if expdt > cutoff:
                        chosen = s
                        break
                if chosen:
                    break
            out[root] = chosen or cands[0]
        self._front, self._front_date = out, key
        return out

    # ---- read-only: broker truth ------------------------------------------

    async def get_open_positions(self) -> list[dict]:
        """Crypto-only (MBT/MET) open positions — symbol-isolated from MNQ."""
        h = await self._headers()
        r = await self._http.get(
            f"{SIM_BROKERAGE}/accounts/{self._account}/positions", headers=h)
        r.raise_for_status()
        return [p for p in r.json().get("Positions", [])
                if any((p.get("Symbol") or "").startswith(root) for root in CRYPTO_ROOTS)]

    # ---- state-changing: the ONLY mutating call ---------------------------

    async def place_order(self, symbol: str, action: str, quantity: int) -> Optional[str]:
        """Market order. action in {'BUY','SELL'} (futures are symmetric: SELL opens
        a short, BUY closes it). Returns the TS order id, or None on rejection.
        Caller MUST confirm the resulting position via get_open_positions().
        """
        assert action in ("BUY", "SELL"), action
        assert any(symbol.startswith(r) for r in CRYPTO_ROOTS), f"refusing non-crypto symbol {symbol}"
        payload = {
            "AccountID": self._account,
            "Symbol": symbol,
            "Quantity": str(quantity),
            "OrderType": "Market",
            "TradeAction": action,
            "TimeInForce": {"Duration": "DAY"},
            "Route": "Intelligent",
        }
        h = await self._headers()
        r = await self._http.post(SIM_ORDERS_URL, headers=h, json=payload)
        if r.status_code not in (200, 201):
            logger.warning("TS Thursday order FAILED HTTP %s: %s", r.status_code, r.text[:200])
            return None
        orders = r.json().get("Orders", [])
        oid = orders[0].get("OrderID") if orders else None
        logger.info("TS Thursday order: %s %s x%s -> #%s", action, symbol, quantity, oid)
        return oid
