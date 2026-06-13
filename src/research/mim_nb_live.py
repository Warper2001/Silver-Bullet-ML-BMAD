#!/usr/bin/env python3
"""
MIM-Noise-Bands live combine bot — Topstep 50K via ProjectX.

Strategy: sealed mim-nb-v2-catstop S-B spec (prereg 6957daa, deployment prereg 7939eed).
  - Noise bands sigma(t) = 14-day mean |close(d,t)/open_d - 1| per RTH minute label
  - UB = O*(1+sigma) + max(Cprev-O,0); LB = O*(1-sigma) - max(O-Cprev,0)
  - Entry/reversal checks at HH:00/HH:30 completions 10:00-15:30 ET (market order)
  - Wide band-stop checks 10:00-16:00 (long exits close<LB, short exits close>UB)
  - Catastrophe stop: resting stop at entry -/+ 500 pts, placed after entry
  - EOD flatten at the 16:00 ET bar; DLL guard: day P&L <= -$1,000 -> no entries
  - 1 contract, MNQ front month

Data: TradeStation REST 1-min bars (poll). Execution: ProjectX (TopstepX).
Integrity: append-only CSVs under data/mim_nb/ with SHA-256 hash chaining.

PROJECTX_ACCOUNT_ID must be set in the environment — the bot refuses to start without it.
"""
import asyncio
import csv
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytz

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.auth_v3 import TradeStationAuthV3
from src.research.projectx_auth import ProjectXAuth
from src.research.projectx_client import (
    ProjectXClient, _to_contract_id, _BASE_URL,
    _TYPE_MARKET, _TYPE_STOP, _SIDE_BUY, _SIDE_SELL,
)

# ----------------------------------------------------------------------
# Frozen configuration (deployment prereg 7939eed — do not edit without a new seal)
# ----------------------------------------------------------------------
SYMBOL = os.environ.get("MIM_NB_SYMBOL", "MNQU26")   # front month; mechanical roll rule
CONTRACTS = 1
CAT_STOP_PTS = 500.0
LOOKBACK_DAYS = 14
DLL_GUARD_USD = -1000.0
PT_VAL = 2.0
ET = pytz.timezone("America/New_York")

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "mim_nb"
LOG_DIR = BASE_DIR / "logs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "mim_nb_live.log"), logging.StreamHandler()],
)
logger = logging.getLogger("mim_nb")

BARS_URL = f"https://api.tradestation.com/v3/marketdata/barcharts/{SYMBOL}?interval=1&unit=Minute"

ENTRY_MARKS = {f"{h:02d}:{m}" for h in range(10, 16) for m in ("00", "30")} - {"16:00"}
CHECK_MARKS = ENTRY_MARKS | {"16:00"}


# ----------------------------------------------------------------------
# Hash-chained append-only CSV logger
# ----------------------------------------------------------------------
class ChainedCsv:
    def __init__(self, path: Path, fields: list):
        self.path = path
        self.fields = fields + ["chain"]
        self.head = "GENESIS"
        if path.exists():
            try:
                with open(path) as f:
                    for row in csv.DictReader(f):
                        self.head = row.get("chain", self.head)
            except Exception:
                logger.warning("chain reload failed for %s — restarting chain", path.name)
        else:
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()

    def append(self, row: dict):
        payload = "|".join(str(row.get(k, "")) for k in self.fields if k != "chain")
        self.head = hashlib.sha256((self.head + "|" + payload).encode()).hexdigest()[:16]
        row = dict(row)
        row["chain"] = self.head
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore").writerow(row)


bars_log = ChainedCsv(DATA_DIR / "bars_raw.csv",
                      ["ts_utc", "open", "high", "low", "close", "volume", "received_at"])
decisions_log = ChainedCsv(DATA_DIR / "decisions.csv",
                           ["ts_et", "mark", "open_d", "prev_close", "sigma", "ub", "lb",
                            "close", "vwap", "position", "action", "detail"])
orders_log = ChainedCsv(DATA_DIR / "orders.csv",
                        ["ts_utc", "event", "order_id", "otype", "side", "size",
                         "price", "outcome", "detail"])
trades_log = ChainedCsv(DATA_DIR / "trades.csv",
                        ["day", "dir", "entry_t", "entry_px", "exit_t", "exit_px",
                         "reason", "pnl_pts", "pnl_usd", "day_pnl_usd"])


class MimNbLive:
    def __init__(self):
        acct = os.environ.get("PROJECTX_ACCOUNT_ID", "")
        if not acct:
            raise SystemExit("PROJECTX_ACCOUNT_ID is not set — refusing to start (deployment prereg 7939eed)")
        self.account_id = int(acct)
        self.http = None
        self.ts_auth = None
        self.px_auth = None
        self.px = None
        self.contract_id = _to_contract_id(SYMBOL)

        # strategy state
        self.sigma_hist = {}          # "HH:MM" -> list of |close/open - 1| (deque-like, max 14)
        self.prev_close = None        # prior session close
        self.day = None               # current ET session date
        self.open_d = None            # today's 09:31-bar open
        self.cum_pv = 0.0
        self.cum_v = 0.0
        self.today_moves = {}         # mark -> |close/open-1| accumulated for today
        self.position = 0
        self.entry_px = 0.0
        self.entry_t = None
        self.cat_stop_id = None
        self.day_pnl = 0.0
        self.day_deactivated = False
        self.last_bar_ts = None

    # ------------------------------------------------------------------
    async def initialize(self):
        logger.info("=" * 70)
        logger.info("MIM-NB LIVE — %s — %d contract — cat stop %.0f pts — account %s",
                    SYMBOL, CONTRACTS, CAT_STOP_PTS, self.account_id)
        logger.info("Sealed spec 6957daa / deployment 7939eed")
        logger.info("=" * 70)
        self.http = httpx.AsyncClient(timeout=30)
        self.ts_auth = TradeStationAuthV3.from_file(".access_token")
        await self.ts_auth.authenticate()
        self.px_auth = ProjectXAuth.from_file(".projectx_api_key")

        class _Cfg:
            symbol = SYMBOL
        self.px = ProjectXClient(self.px_auth, _Cfg(), self.http,
                                 projectx_account_id=self.account_id)
        await self._backfill()
        await self._reconcile_startup()
        await self._catch_up_today()

    async def _ts_token(self) -> str:
        """Token source of truth is the .access_token file — a sibling bot's
        auto-refresh loop rewrites it every ~minute. Re-read per request so we
        never serve a stale in-memory token (root cause of the 2026-06-11 401s)."""
        tok_path = BASE_DIR / ".access_token"
        age = datetime.now(timezone.utc).timestamp() - tok_path.stat().st_mtime
        if age > 1800:
            logger.error("ALERT: .access_token is %.0f min stale — refresher down? "
                         "Falling back to own refresh.", age / 60)
            return await self.ts_auth.authenticate()
        return tok_path.read_text().strip()

    async def _ts_get_bars(self, barsback=1500):
        token = await self._ts_token()
        url = f"{BARS_URL}&barsback={barsback}"
        r = await self.http.get(url, headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        return r.json().get("Bars", [])

    @staticmethod
    def _bar_et(bar):
        ts = datetime.fromisoformat(bar["TimeStamp"].replace("Z", "+00:00"))
        return ts.astimezone(ET)

    async def _backfill(self):
        """Seed sigma history from the last >=14 complete RTH sessions."""
        logger.info("Backfilling bars for sigma seed...")
        bars = await self._ts_get_bars(barsback=30000)
        sessions = {}
        for b in bars:
            et = self._bar_et(b)
            hm = et.strftime("%H:%M")
            if "09:31" <= hm <= "16:00":
                sessions.setdefault(et.date(), []).append((hm, float(b["Open"]), float(b["Close"])))
        days = sorted(sessions)
        complete = [d for d in days if sessions[d][0][0] == "09:31" and sessions[d][-1][0] == "16:00"]
        seed_days = complete[-(LOOKBACK_DAYS + 1):]
        today_et = datetime.now(ET).date()
        seed_days = [d for d in seed_days if d < today_et][-LOOKBACK_DAYS - 1:]
        for d in seed_days[-LOOKBACK_DAYS:]:
            o = sessions[d][0][1]
            for hm, _, c in sessions[d]:
                self.sigma_hist.setdefault(hm, []).append(abs(c / o - 1.0))
                self.sigma_hist[hm] = self.sigma_hist[hm][-LOOKBACK_DAYS:]
        if seed_days:
            last = seed_days[-1]
            self.prev_close = sessions[last][-1][2]
        n_ok = sum(1 for v in self.sigma_hist.values() if len(v) >= LOOKBACK_DAYS)
        logger.info("Sigma seeded: %d days, %d minute-labels at full depth, prev_close=%.2f",
                    len(seed_days), n_ok, self.prev_close or float("nan"))
        if len(seed_days) < LOOKBACK_DAYS:
            logger.warning("Fewer than %d complete sessions in backfill — entries blocked until depth reached",
                           LOOKBACK_DAYS)

    async def _reconcile_startup(self):
        st = await self.px.reconcile_state(str(self.account_id))
        if st.status != "FLAT":
            qty = getattr(st, "position_qty", 0) or 0
            logger.warning("Startup reconcile: account not flat (%s qty=%s) — flattening per seal",
                           st.status, qty)
            await self.px.cancel_all_pending_orders(str(self.account_id))
            if qty != 0:
                await self.px.close_position_at_market(
                    "LONG" if qty > 0 else "SHORT", str(self.account_id), contracts=abs(qty))
            orders_log.append({"ts_utc": datetime.now(timezone.utc).isoformat(),
                               "event": "FLATTEN", "order_id": "", "otype": _TYPE_MARKET,
                               "side": "", "size": abs(qty), "price": "",
                               "outcome": "SENT", "detail": "STARTUP_RECONCILE"})
        else:
            logger.info("Startup reconcile: FLAT ✓")

    async def _catch_up_today(self):
        """If started/restarted mid-session, rebuild today's intraday state
        (open anchor, VWAP, sigma moves) from completed bars WITHOUT acting on
        missed checks — missed signals are skipped, never back-traded."""
        now_et = datetime.now(ET)
        if not ("09:31" <= now_et.strftime("%H:%M") <= "16:00"):
            return
        bars = await self._ts_get_bars(barsback=1500)
        todays = [b for b in bars
                  if self._bar_et(b).date() == now_et.date()
                  and "09:31" <= self._bar_et(b).strftime("%H:%M") <= "16:00"]
        if not todays:
            return
        first_hm = self._bar_et(todays[0]).strftime("%H:%M")
        if first_hm != "09:31":
            logger.warning("Catch-up: today's 09:31 bar unavailable (first=%s) — standing down today",
                           first_hm)
            return
        self._new_session(now_et.date())
        skipped_checks = 0
        for b in todays:
            et = self._bar_et(b)
            hm = et.strftime("%H:%M")
            o, c = float(b["Open"]), float(b["Close"])
            v = float(b.get("TotalVolume", 0) or 0)
            if self.open_d is None:
                self.open_d = o
            self.cum_pv += c * v
            self.cum_v += v
            self.today_moves[hm] = abs(c / self.open_d - 1.0)
            if hm in CHECK_MARKS:
                skipped_checks += 1
            self.last_bar_ts = b["TimeStamp"]
        logger.warning("Catch-up complete: open_d=%.2f, %d bars rebuilt, %d check marks "
                       "MISSED (not back-traded, per seal fill model). Live from next check.",
                       self.open_d, len(todays), skipped_checks)

    # ------------------------------------------------------------------
    # Order helpers (market / stop via ProjectX)
    # ------------------------------------------------------------------
    async def _order(self, otype, side, price=None):
        payload = {"accountId": self.account_id, "contractId": self.contract_id,
                   "type": otype, "side": side, "size": CONTRACTS}
        if otype == _TYPE_STOP:
            payload["stopPrice"] = float(price)
        oid = await self.px._place_order(payload)
        orders_log.append({"ts_utc": datetime.now(timezone.utc).isoformat(),
                           "event": "PLACE", "order_id": oid or "FAIL",
                           "otype": otype, "side": side, "size": CONTRACTS,
                           "price": price or "", "outcome": "OK" if oid else "REJECTED",
                           "detail": ""})
        if oid is not None and otype == _TYPE_MARKET:
            asyncio.get_event_loop().create_task(self._log_fill(oid))
        return oid

    async def _log_fill(self, order_id):
        """Best-effort: fetch the venue fill price for a market order and log it
        (slippage evidence per deployment prereg halt trigger #2)."""
        await asyncio.sleep(3)
        try:
            headers = await self.px._headers()
            since = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
            r = await self.http.post(f"{_BASE_URL}/Trade/search",
                                     json={"accountId": self.account_id,
                                           "startTimestamp": since}, headers=headers)
            if r.status_code == 200 and r.json().get("trades"):
                t = r.json()["trades"][0]
                orders_log.append({"ts_utc": datetime.now(timezone.utc).isoformat(),
                                   "event": "FILL", "order_id": order_id,
                                   "otype": _TYPE_MARKET, "side": t.get("side"),
                                   "size": t.get("size"), "price": t.get("price"),
                                   "outcome": "OK",
                                   "detail": f"fees={t.get('fees')}"})
                logger.info("FILL order #%s @ %s (fees %s)", order_id,
                            t.get("price"), t.get("fees"))
        except Exception as exc:
            logger.warning("fill logging failed for #%s: %s", order_id, exc)

    async def _cancel_cat_stop(self):
        if self.cat_stop_id:
            ok = await self.px.cancel_order(str(self.cat_stop_id))
            orders_log.append({"ts_utc": datetime.now(timezone.utc).isoformat(),
                               "event": "CANCEL", "order_id": self.cat_stop_id,
                               "otype": _TYPE_STOP, "side": "", "size": CONTRACTS,
                               "price": "", "outcome": "OK" if ok else "FAIL", "detail": "cat stop"})
            self.cat_stop_id = None

    async def _enter(self, direction, ref_px, mark):
        side = _SIDE_BUY if direction == 1 else _SIDE_SELL
        oid = await self._order(_TYPE_MARKET, side)
        if oid is None:
            logger.error("ENTRY market order rejected at %s", mark)
            return False
        self.position = direction
        self.entry_px = ref_px      # decision-bar close = slippage reference
        self.entry_t = mark
        stop_px = ref_px - CAT_STOP_PTS if direction == 1 else ref_px + CAT_STOP_PTS
        sid = await self._order(_TYPE_STOP, _SIDE_SELL if direction == 1 else _SIDE_BUY, stop_px)
        self.cat_stop_id = sid
        if sid is None:
            logger.error("CAT STOP rejected — flattening unprotected position per seal")
            await self._flatten("CAT_STOP_REJECTED")
            return False
        logger.info("ENTER %s @~%.2f (mark %s) cat-stop %.2f #%s",
                    "LONG" if direction == 1 else "SHORT", ref_px, mark, stop_px, sid)
        return True

    async def _exit(self, ref_px, mark, reason):
        await self._cancel_cat_stop()
        side = _SIDE_SELL if self.position == 1 else _SIDE_BUY
        oid = await self._order(_TYPE_MARKET, side)
        if oid is None:
            logger.error("EXIT market order rejected — will retry next loop")
            return False
        self._record_trade(ref_px, mark, reason)
        return True

    async def _flatten(self, reason):
        await self.px.cancel_all_pending_orders(str(self.account_id))
        self.cat_stop_id = None
        if self.position != 0:
            await self.px.close_position_at_market(
                "LONG" if self.position == 1 else "SHORT",
                str(self.account_id), contracts=CONTRACTS)
        orders_log.append({"ts_utc": datetime.now(timezone.utc).isoformat(),
                           "event": "FLATTEN", "order_id": "", "otype": _TYPE_MARKET,
                           "side": "", "size": CONTRACTS, "price": "",
                           "outcome": "SENT", "detail": reason})
        if self.position != 0:
            self._record_trade(self.prev_ref_price(), datetime.now(ET).strftime("%H:%M"), reason)

    def prev_ref_price(self):
        return self.entry_px  # conservative reference when no bar close available

    def _record_trade(self, exit_px, exit_t, reason):
        pnl_pts = self.position * (exit_px - self.entry_px)
        pnl_usd = pnl_pts * PT_VAL * CONTRACTS
        
        # Log to DB
        from src.monitoring.trade_db import TradeDatabase
        db = TradeDatabase()
        db.log_trade(
            trader_id='trader-mim-nb',
            timestamp=datetime.now(timezone.utc).isoformat(),
            pnl=round(pnl_usd, 2),
            direction='L' if self.position == 1 else 'S',
            entry_price=self.entry_px,
            exit_price=exit_px,
            exit_reason=reason,
            metadata={'pnl_pts': pnl_pts}
        )

        self.day_pnl += pnl_usd
        trades_log.append({"day": str(self.day), "dir": self.position,
                           "entry_t": self.entry_t, "entry_px": f"{self.entry_px:.2f}",
                           "exit_t": exit_t, "exit_px": f"{exit_px:.2f}", "reason": reason,
                           "pnl_pts": f"{pnl_pts:+.2f}", "pnl_usd": f"{pnl_usd:+.2f}",
                           "day_pnl_usd": f"{self.day_pnl:+.2f}"})
        logger.info("TRADE CLOSED %s %s→%s %+.2f pts ($%+.2f) day P&L %+.2f [%s]",
                    "L" if self.position == 1 else "S", self.entry_t, exit_t,
                    pnl_pts, pnl_usd, self.day_pnl, reason)
        self.position = 0
        if self.day_pnl <= DLL_GUARD_USD and not self.day_deactivated:
            self.day_deactivated = True
            logger.warning("DLL GUARD: day P&L $%.2f ≤ -$1,000 — entries disabled until next session",
                           self.day_pnl)

    # ------------------------------------------------------------------
    # Bar processing — mirrors the sealed backtest engine
    # ------------------------------------------------------------------
    def _new_session(self, d):
        if self.day is not None and self.open_d is not None:
            for hm, mv in self.today_moves.items():
                self.sigma_hist.setdefault(hm, []).append(mv)
                self.sigma_hist[hm] = self.sigma_hist[hm][-LOOKBACK_DAYS:]
        self.day = d
        self.open_d = None
        self.cum_pv = 0.0
        self.cum_v = 0.0
        self.today_moves = {}
        self.day_pnl = 0.0
        self.day_deactivated = False
        logger.info("=== New session %s ===", d)

    async def on_bar(self, bar):
        et = self._bar_et(bar)
        hm = et.strftime("%H:%M")
        o, h, l, c, v = (float(bar["Open"]), float(bar["High"]), float(bar["Low"]),
                         float(bar["Close"]), float(bar.get("TotalVolume", 0) or 0))
        bars_log.append({"ts_utc": bar["TimeStamp"], "open": o, "high": h, "low": l,
                         "close": c, "volume": v,
                         "received_at": datetime.now(timezone.utc).isoformat()})
        if not ("09:31" <= hm <= "16:00"):
            return
        if et.date() != self.day:
            self._new_session(et.date())
        if self.open_d is None:
            if hm != "09:31":
                return  # joined mid-session without the open — stand down today
            self.open_d = o
        self.cum_pv += c * v
        self.cum_v += v
        self.today_moves[hm] = abs(c / self.open_d - 1.0)

        if hm not in CHECK_MARKS:
            return
        sig = self.sigma_hist.get(hm, [])
        if len(sig) < LOOKBACK_DAYS or self.prev_close is None:
            return
        sigma = sum(sig) / len(sig)
        gap_dn = max(self.prev_close - self.open_d, 0.0)
        gap_up = max(self.open_d - self.prev_close, 0.0)
        ub = self.open_d * (1 + sigma) + gap_dn
        lb = self.open_d * (1 - sigma) - gap_up
        vwap = self.cum_pv / self.cum_v if self.cum_v else c
        action, detail = "NONE", ""

        if hm == "16:00":
            if self.position != 0:
                await self._exit(c, hm, "EOD")
                action = "EOD_EXIT"
        else:
            live_pos = await self._verify_position()
            if self.position != 0 and live_pos == 0:
                # catastrophe stop filled since last check
                stop_px = self.entry_px - CAT_STOP_PTS if self.position == 1 else self.entry_px + CAT_STOP_PTS
                self._record_trade(stop_px, hm, "CAT_STOP")
                await self.px.cancel_all_pending_orders(str(self.account_id))
                self.cat_stop_id = None
                action = "CAT_STOP_DETECTED"
            elif self.position == 1 and c < lb:
                await self._exit(c, hm, "STOP")
                action = "BAND_STOP_EXIT"
            elif self.position == -1 and c > ub:
                await self._exit(c, hm, "STOP")
                action = "BAND_STOP_EXIT"
            if not self.day_deactivated and hm in ENTRY_MARKS:
                if c > ub and self.position != 1:
                    if self.position == -1:
                        await self._exit(c, hm, "REVERSAL")
                    if await self._enter(1, c, hm):
                        action = (action + "+" if action != "NONE" else "") + "ENTER_LONG"
                elif c < lb and self.position != -1:
                    if self.position == 1:
                        await self._exit(c, hm, "REVERSAL")
                    if await self._enter(-1, c, hm):
                        action = (action + "+" if action != "NONE" else "") + "ENTER_SHORT"

        decisions_log.append({"ts_et": et.isoformat(), "mark": hm,
                              "open_d": f"{self.open_d:.2f}",
                              "prev_close": f"{self.prev_close:.2f}",
                              "sigma": f"{sigma:.6f}", "ub": f"{ub:.2f}", "lb": f"{lb:.2f}",
                              "close": f"{c:.2f}", "vwap": f"{vwap:.2f}",
                              "position": self.position, "action": action, "detail": detail})

        if hm == "16:00":
            self.prev_close = c
            self._save_state()

    async def _verify_position(self):
        try:
            headers = await self.px._headers()
            r = await self.http.post(f"{_BASE_URL}/Position/searchOpen",
                                     json={"accountId": self.account_id,
                                           "contractId": self.contract_id}, headers=headers)
            if r.status_code == 200:
                return sum(int(p.get("size", 0)) for p in r.json().get("positions", []))
        except Exception as exc:
            logger.warning("position verify failed: %s", exc)
        return self.position  # fall back to local belief on error

    def _save_state(self):
        (DATA_DIR / "state.json").write_text(json.dumps({
            "day": str(self.day), "position": self.position, "entry_px": self.entry_px,
            "entry_t": self.entry_t, "cat_stop_id": self.cat_stop_id,
            "day_pnl": self.day_pnl, "prev_close": self.prev_close,
            "chains": {"bars": bars_log.head, "decisions": decisions_log.head,
                       "orders": orders_log.head, "trades": trades_log.head},
            "saved_at": datetime.now(timezone.utc).isoformat()}, indent=2))

    # ------------------------------------------------------------------
    async def run(self):
        await self.initialize()
        logger.info("Polling loop started (10s)")
        while True:
            try:
                now_et = datetime.now(ET)
                # safety net: never hold past 16:01 ET
                if now_et.strftime("%H:%M") >= "16:01" and self.position != 0:
                    logger.warning("SAFETY: position open after 16:01 ET — flattening")
                    await self._flatten("SAFETY_1601")
                bars = await self._ts_get_bars(barsback=60)
                for b in bars:
                    ts = b["TimeStamp"]
                    if self.last_bar_ts is None or ts > self.last_bar_ts:
                        # only completed bars: skip the still-forming latest bar
                        bar_et = self._bar_et(b)
                        if (datetime.now(timezone.utc) -
                                bar_et.astimezone(timezone.utc)) >= timedelta(seconds=2):
                            await self.on_bar(b)
                            self.last_bar_ts = ts
            except Exception as exc:
                logger.error("poll loop error: %s", exc)
            # 2s cadence in the first 20s of each minute (catch the completed bar
            # the moment TradeStation publishes it, ~6-9s past the minute);
            # 10s otherwise. Bounds check-to-order latency at the API publish lag.
            await asyncio.sleep(2 if datetime.now(timezone.utc).second < 20 else 10)


if __name__ == "__main__":
    asyncio.run(MimNbLive().run())
