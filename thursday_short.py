#!/usr/bin/env python3
"""
Thursday Short — BTC + ETH day-of-week strategy on TradeStation SIM.

Empirical finding: BTC and ETH return significantly negative on Thursdays
(UTC 00:00-23:59). See _bmad-output/preregistration_kraken_thursday_short.md
and amendment 2 (venue: CME micro futures via TradeStation).

Strategy:
  - Short BTC (MBT) and ETH (MET) micro futures at Thursday 00:00 UTC
  - Close at Thursday 23:05 UTC
  - Stop: if a leg moves >5% against entry, close immediately
  - Sizing: 1 MBT + notional-matched MET (equal-weight)

Execution: CME Micro Bitcoin (MBT) / Micro Ether (MET) futures on the TradeStation
SIM futures account, symbol-isolated from the MNQ bots. Position state is confirmed
against broker truth (never assumed) — see G1/G2/G3 integrity gates below.

Usage:
  .venv/bin/python thursday_short.py            # paper, TradeStation SIM (MBT/MET)

Decision rule: PASS if Sharpe > 0.80 after N >= 30 prospective Thursdays.
"""
import asyncio
import csv
import hashlib
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Strategy configuration
# ──────────────────────────────────────────────
STOP_PCT       = float(os.environ.get("THU_STOP_PCT", "5.0"))         # % against a leg
TS_SIM_ACCOUNT = os.environ.get("THU_TS_SIM_ACCOUNT", "SIM2797251F")  # futures SIM account
MBT_SIZE       = int(os.environ.get("THU_MBT_SIZE", "1"))             # BTC-leg contracts; MET matches notional
ALARM_FILE     = Path("data/thursday_ts/ALARM")

ENTRY_HOUR  = 0      # 00:00 UTC Thursday
EXIT_HOUR   = 23
EXIT_MINUTE = 5      # 23:05 UTC


def fetch_btc_lr_slopes():
    """BTC LR-channel slope (bps/day) for the regime tag — read-only, firewalled.

    Returns (slope_len20, slope_len40) from completed daily BTC closes through the
    prior day (the regime known at the Thursday 00:00 UTC entry), or (None, None)
    on any failure. The data source is a free public BTC reference feed — it is the
    SEALED regime source per pre-registration amendment 1 (do not change without a
    new amendment). Instrumentation only; does NOT influence trading.
    """
    try:
        import numpy as np
        from src.research.lr_channel import compute_lr_channel
        with httpx.Client(timeout=10.0) as c:
            r = c.get("https://api.kraken.com/0/public/OHLC",   # public market data; sealed regime source
                      params={"pair": "XBTUSD", "interval": 1440})
            r.raise_for_status()
            body = r.json()
        if body.get("error"):
            raise RuntimeError(str(body["error"]))
        result = body["result"]
        key = next(k for k in result if k != "last")
        closes = np.array([float(c[4]) for c in result[key][:-1]], dtype=float)
        if len(closes) < 45:
            return None, None
        last = closes[-1]
        s20 = float(compute_lr_channel(closes, 20).slope[-1] / last * 1e4)
        s40 = float(compute_lr_channel(closes, 40).slope[-1] / last * 1e4)
        return s20, s40
    except Exception as e:  # firewall — regime tag must never disrupt trading
        logger.warning("LR-slope regime tag failed (logging only): %s", e)
        return None, None


class ChainedCsv:
    """Tamper-evident hash-chained CSV (local copy of the MIM-NB pattern)."""

    def __init__(self, path, fields):
        self.path = path
        self.fields = list(fields) + ["chain"]
        self.head = "GENESIS"
        if path.exists():
            try:
                with open(path) as f:
                    for row in csv.DictReader(f):
                        self.head = row.get("chain", self.head)
            except Exception:
                logger.warning("chain reload failed for %s — restarting chain", path.name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()

    def append(self, row):
        payload = "|".join(str(row.get(k, "")) for k in self.fields if k != "chain")
        self.head = hashlib.sha256((self.head + "|" + payload).encode()).hexdigest()[:16]
        row = dict(row); row["chain"] = self.head
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore").writerow(row)


class ThursdayShortTrader:
    """Weekly Thursday short on BTC (MBT) + ETH (MET) via TradeStation SIM."""

    def __init__(self) -> None:
        self.running       = False
        self.btc_entry: Optional[float] = None   # MBT leg entry mark
        self.eth_entry: Optional[float] = None   # MET leg entry mark
        self.position_open = False
        self.current_thursday: Optional[str] = None
        self.total_pnl_usd = 0.0
        self.n_trades = 0
        self.n_wins   = 0
        # LR-regime tag cache (computed once per Thursday at entry)
        self._lr_cache_date: Optional[str] = None
        self._lr20: Optional[float] = None
        self._lr40: Optional[float] = None
        # per-Thursday execution state
        self.ts_mbt_sym: Optional[str] = None
        self.ts_met_sym: Optional[str] = None
        self.ts_n_mbt = 0
        self.ts_n_met = 0
        self.ts_entry_attempted_date: Optional[str] = None
        self.ts_entry_time = ""
        self._trades_log = None       # ChainedCsv data/thursday_ts/trades.csv
        self._decisions_log = None    # ChainedCsv data/thursday_ts/decisions.csv
        self._cf_log = None           # ChainedCsv data/thursday_ts/counterfactuals.csv (Amendment 3)
        self._pending_cf: Optional[list] = None       # early-exit legs awaiting held-to-23:05 mark
        self._pending_cf_thursday: Optional[str] = None

    # ── Integrity helpers ─────────────────────────────────────────────────

    def _alarm(self, reason: str) -> None:
        """Hard integrity alarm: write the ALARM flag + ERROR. Startup refuses to
        run while the flag exists (forces human review after any breach)."""
        try:
            ALARM_FILE.parent.mkdir(parents=True, exist_ok=True)
            ALARM_FILE.write_text(f"{datetime.now(timezone.utc).isoformat()} {reason}\n")
        except Exception as e:
            logger.error(f"failed to write ALARM flag: {e}")
        logger.error(f"🚨 ALARM: {reason}")

    @staticmethod
    def _is_short(p: dict) -> bool:
        return str(p.get("LongShort", "")).lower().startswith("s")

    async def _confirm_positions(self, client, want_syms, timeout: int = 25) -> list:
        """Poll broker positions until all want_syms are present (or timeout)."""
        import time
        t0 = time.time()
        while time.time() - t0 < timeout:
            pos = await client.get_open_positions()
            if all(any(p.get("Symbol") == s for p in pos) for s in want_syms):
                return pos
            await asyncio.sleep(2)
        return await client.get_open_positions()

    def _check_stop(self, btc_price: float, eth_price: float) -> bool:
        if self.btc_entry is not None:
            pct_move = (btc_price - self.btc_entry) / self.btc_entry * 100
            if pct_move > STOP_PCT:
                logger.warning(f"BTC STOP HIT: entry={self.btc_entry:.2f} current={btc_price:.2f} +{pct_move:.2f}%")
                return True
        if self.eth_entry is not None:
            pct_move = (eth_price - self.eth_entry) / self.eth_entry * 100
            if pct_move > STOP_PCT:
                logger.warning(f"ETH STOP HIT: entry={self.eth_entry:.2f} current={eth_price:.2f} +{pct_move:.2f}%")
                return True
        return False

    def _clear_position(self) -> None:
        self.position_open = False
        self.btc_entry = self.eth_entry = None
        self.ts_mbt_sym = self.ts_met_sym = None
        self.ts_n_mbt = self.ts_n_met = 0
        wr = 100 * self.n_wins / self.n_trades if self.n_trades > 0 else 0
        logger.info(f"Session P&L: ${self.total_pnl_usd:+.2f} | Trades: {self.n_trades} | WR: {wr:.0f}%")

    def _log_decision(self, action: str, detail: str = "", mbt=None, met=None,
                      pb=None, pe=None, n_mbt=0, n_met=0) -> None:
        """One row per Thursday entry decision (ENTERED/REJECTED/SKIPPED/...),
        with the LR-regime tags — feeds the prospective N count + LR-gate subset."""
        if self._decisions_log is None:
            return
        self._decisions_log.append({
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "thursday": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "mbt_sym": mbt or "", "met_sym": met or "",
            "mark_btc": pb or "", "mark_eth": pe or "",
            "n_mbt": n_mbt, "n_met": n_met,
            "lr_slope20_bpd": "" if self._lr20 is None else round(self._lr20, 3),
            "lr_slope40_bpd": "" if self._lr40 is None else round(self._lr40, 3),
            "action": action, "detail": detail,
        })

    def _write_cf(self, row: dict) -> None:
        """Amendment 3 counterfactual ledger — knowledge-only, firewalled."""
        if self._cf_log is None:
            return
        try:
            self._cf_log.append(row)
        except Exception as e:
            logger.warning(f"counterfactual log write failed: {e}")

    # ── Entry / exit ──────────────────────────────────────────────────────

    async def _enter(self, client) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.ts_entry_attempted_date = today  # G3: record attempt regardless of outcome
        self._lr20, self._lr40 = fetch_btc_lr_slopes()  # regime tag at entry
        self._lr_cache_date = today
        logger.info("=" * 60)
        logger.info("THURSDAY SHORT — ENTERING SHORT POSITIONS")

        pre = await client.get_open_positions()           # G1 pre-trade reconcile
        if pre:
            self._alarm(f"unexpected open position before entry: {[p.get('Symbol') for p in pre]}")
            self._log_decision("SKIPPED_NOT_FLAT", str([p.get('Symbol') for p in pre]))
            return

        front = await client.resolve_front_month()
        mbt, met = front["MBT"], front["MET"]
        pb = await client.last_price(mbt)
        pe = await client.last_price(met)
        if not pb or not pe:
            self._alarm(f"no marks for {mbt}/{met}")
            self._log_decision("NO_MARKS", f"{mbt}/{met}", mbt, met)
            return
        n_mbt = MBT_SIZE
        n_met = max(1, round(n_mbt * pb / pe))            # equal-notional (~1:38)
        logger.info(f"  {n_mbt} {mbt} (~${n_mbt*pb*0.1:,.0f}) + {n_met} {met} (~${n_met*pe*0.1:,.0f}) short")

        o1 = await client.place_order(mbt, "SELL", n_mbt)
        o2 = await client.place_order(met, "SELL", n_met)

        pos = await self._confirm_positions(client, [mbt, met])   # G1: broker truth
        held = {p.get("Symbol"): p for p in pos}
        ok = (mbt in held and met in held and self._is_short(held[mbt]) and self._is_short(held[met]))
        if not ok:
            self._alarm(f"entry NOT confirmed (mbt#{o1} met#{o2}); broker shows {[p.get('Symbol') for p in pos]}")
            for p in pos:                                  # flatten any partial leg
                act = "BUY" if self._is_short(p) else "SELL"
                await client.place_order(p["Symbol"], act, abs(int(float(p["Quantity"]))))
            self._log_decision("REJECTED", f"mbt#{o1} met#{o2}", mbt, met, pb, pe, n_mbt, n_met)
            return                                         # position_open stays False (anti-phantom)

        self.btc_entry = float(held[mbt].get("AveragePrice") or pb)
        self.eth_entry = float(held[met].get("AveragePrice") or pe)
        self.ts_mbt_sym, self.ts_met_sym = mbt, met
        self.ts_n_mbt, self.ts_n_met = n_mbt, n_met
        self.ts_entry_time = datetime.now(timezone.utc).strftime("%H:%M")
        self.position_open = True
        self.current_thursday = today
        self._log_decision("ENTERED", "", mbt, met, self.btc_entry, self.eth_entry, n_mbt, n_met)
        logger.info(f"✓ SHORT confirmed | {mbt} @ {self.btc_entry} | {met} @ {self.eth_entry}")

    async def _exit(self, reason: str, client) -> None:
        import time
        logger.info(f"THURSDAY SHORT — EXITING ({reason})")
        exit_t = datetime.now(timezone.utc).strftime("%H:%M")
        cf_pending = []   # Amendment 3: legs whose held-to-23:05 counterfactual resolves later
        for sym, entry, qty, label in [
            (self.ts_mbt_sym, self.btc_entry, self.ts_n_mbt, "MBT"),
            (self.ts_met_sym, self.eth_entry, self.ts_n_met, "MET"),
        ]:
            if not sym or entry is None:
                continue
            try:
                exit_price = await client.last_price(sym)
                await client.place_order(sym, "BUY", qty)            # close short
                ret_bps = (entry - exit_price) / entry * 10_000
                pnl = (entry - exit_price) * qty * 0.1               # micro = 0.1 underlying
                self.total_pnl_usd += pnl
                self.n_trades += 1
                if ret_bps > 0:
                    self.n_wins += 1
                logger.info(f"{label} {sym} EXIT: {entry:.2f}→{exit_price:.2f} {ret_bps:+.1f}bps ${pnl:+.2f}")
                if self._trades_log is not None:
                    self._trades_log.append({
                        "thursday": self.current_thursday or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        "symbol": sym, "dir": "short", "entry_t": self.ts_entry_time, "entry_px": entry,
                        "exit_t": exit_t, "exit_px": exit_price, "qty": qty,
                        "ret_bps": round(ret_bps, 2), "pnl_usd": round(pnl, 2), "reason": reason,
                        "lr_slope20_bpd": "" if self._lr20 is None else round(self._lr20, 3),
                        "lr_slope40_bpd": "" if self._lr40 is None else round(self._lr40, 3),
                    })
                cf_row = {
                    "thursday": self.current_thursday or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "symbol": sym, "qty": qty, "entry_px": entry,
                    "stop_trigger_px": round(entry * (1 + STOP_PCT / 100), 2),
                    "realized_exit_t": exit_t, "realized_exit_px": exit_price,
                    "realized_reason": reason, "realized_pnl_usd": round(pnl, 2),
                }
                if reason == "scheduled_exit":   # held to 23:05 -> counterfactual == realized
                    self._write_cf(dict(cf_row, cf_2305_px=exit_price,
                                        cf_pnl_usd=round(pnl, 2), source="live"))
                else:                            # early exit -> resolve at first poll >= 23:05
                    cf_pending.append(cf_row)
            except Exception as e:
                logger.error(f"{label} exit FAILED: {e}")
        if cf_pending:
            self._pending_cf = cf_pending
            self._pending_cf_thursday = cf_pending[0]["thursday"]
        pos = await client.get_open_positions()                     # confirm flat
        t0 = time.time()
        while pos and time.time() - t0 < 25:
            await asyncio.sleep(2); pos = await client.get_open_positions()
        if pos:
            self._alarm(f"NOT flat after exit: {[p.get('Symbol') for p in pos]}")
        self._clear_position()

    # ── Main loop ─────────────────────────────────────────────────────────

    async def _run(self) -> None:
        from src.data.auth_v3 import TradeStationAuthV3
        from src.research.ts_thursday_client import TradeStationThursdayClient
        if ALARM_FILE.exists():
            raise SystemExit(f"ALARM flag present ({ALARM_FILE}); review and remove to resume.")
        auth = TradeStationAuthV3.from_file(".access_token")
        await auth.authenticate()
        await auth.start_auto_refresh()                              # keep token fresh (d4c0c39)
        async with httpx.AsyncClient(timeout=30) as http:
            client = TradeStationThursdayClient(auth, TS_SIM_ACCOUNT, http)
            _tdir = ALARM_FILE.parent
            self._trades_log = ChainedCsv(_tdir / "trades.csv",
                ["thursday", "symbol", "dir", "entry_t", "entry_px", "exit_t", "exit_px",
                 "qty", "ret_bps", "pnl_usd", "reason", "lr_slope20_bpd", "lr_slope40_bpd"])
            self._decisions_log = ChainedCsv(_tdir / "decisions.csv",
                ["ts_utc", "thursday", "mbt_sym", "met_sym", "mark_btc", "mark_eth",
                 "n_mbt", "n_met", "lr_slope20_bpd", "lr_slope40_bpd", "action", "detail"])
            self._cf_log = ChainedCsv(_tdir / "counterfactuals.csv",
                ["thursday", "symbol", "qty", "entry_px", "stop_trigger_px",
                 "realized_exit_t", "realized_exit_px", "realized_reason",
                 "realized_pnl_usd", "cf_2305_px", "cf_pnl_usd", "source"])
            orphan = await client.get_open_positions()              # G2 startup reconcile
            if orphan:
                logger.warning(f"orphan position(s) at startup: {[p.get('Symbol') for p in orphan]} — flattening")
                for p in orphan:
                    act = "BUY" if self._is_short(p) else "SELL"
                    await client.place_order(p["Symbol"], act, abs(int(float(p["Quantity"]))))
                self._alarm("orphan position at startup (flattened)")
            # don't false-alarm the G3 absence check for a Thursday we started too late to trade
            _n = datetime.now(timezone.utc)
            if _n.weekday() == 3 and (_n.hour, _n.minute) > (ENTRY_HOUR, 5):
                self.ts_entry_attempted_date = _n.strftime("%Y-%m-%d")
            while self.running:
                try:
                    now = datetime.now(timezone.utc)
                    dow = now.weekday(); hour, minute = now.hour, now.minute
                    today = now.strftime("%Y-%m-%d")
                    day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

                    # Amendment 3: resolve early-exit counterfactuals at the first
                    # poll >= that Thursday 23:05 UTC (knowledge-only, firewalled)
                    if self._pending_cf and (today > self._pending_cf_thursday
                            or (today == self._pending_cf_thursday
                                and (hour, minute) >= (EXIT_HOUR, EXIT_MINUTE))):
                        try:
                            for leg in self._pending_cf:
                                px = await client.last_price(leg["symbol"])
                                if px is None:
                                    raise RuntimeError(f"no cf mark for {leg['symbol']}")
                                cf_pnl = (leg["entry_px"] - px) * leg["qty"] * 0.1
                                self._write_cf(dict(leg, cf_2305_px=px,
                                                    cf_pnl_usd=round(cf_pnl, 2), source="live"))
                            self._pending_cf = None
                        except Exception as e:  # firewall — cf must never disrupt trading
                            logger.warning(f"counterfactual resolution failed (retry next poll): {e}")

                    if (dow == 3 and hour == ENTRY_HOUR and minute < 5
                            and not self.position_open and self.current_thursday != today):
                        await self._enter(client)
                    elif (self.position_open and dow == 3
                          and hour == EXIT_HOUR and minute >= EXIT_MINUTE):
                        await self._exit("scheduled_exit", client)
                    elif self.position_open and dow == 4:
                        logger.warning("Open position on Friday — emergency close")
                        await self._exit("emergency_friday", client)
                    elif self.position_open:
                        pos = await client.get_open_positions()     # G2 per-poll reconcile
                        held = {p.get("Symbol") for p in pos}
                        if not ({self.ts_mbt_sym, self.ts_met_sym} <= held):
                            self._alarm(f"local/broker divergence: local {self.ts_mbt_sym}/{self.ts_met_sym}, broker {sorted(held)}")
                        bp = await client.last_price(self.ts_mbt_sym)
                        ep = await client.last_price(self.ts_met_sym)
                        if bp and ep and self._check_stop(bp, ep):
                            await self._exit("stop_loss", client)
                        else:
                            bbps = (self.btc_entry - bp)/self.btc_entry*10_000 if (self.btc_entry and bp) else 0
                            ebps = (self.eth_entry - ep)/self.eth_entry*10_000 if (self.eth_entry and ep) else 0
                            logger.info(f"{self.ts_mbt_sym} {bbps:+.1f}bps ({bp}) {self.ts_met_sym} {ebps:+.1f}bps ({ep}) | {day_names[dow]} {now.strftime('%H:%M UTC')}")
                    else:
                        if (dow == 3 and (hour, minute) > (ENTRY_HOUR, 5)
                                and self.ts_entry_attempted_date != today
                                and self.current_thursday != today):
                            self._alarm(f"no entry attempted on Thursday {today}")    # G3 absence
                            self.ts_entry_attempted_date = today                       # alarm once
                        logger.info(f"Waiting [{day_names[dow]} {now.strftime('%H:%M UTC')}] — next trade: next Thursday 00:00 UTC")

                    await asyncio.sleep(60 if self.position_open else 300)
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.error(f"Loop error: {exc}", exc_info=True)
                    await asyncio.sleep(30)
            if self.position_open:
                logger.warning("Shutdown with open position — attempting close")
                await self._exit("shutdown", client)

    async def run(self) -> None:
        logger.info("=" * 60)
        logger.info("THURSDAY SHORT TRADER STARTING [TradeStation SIM — MBT/MET]")
        logger.info(f"  Account: {TS_SIM_ACCOUNT} | base size: {MBT_SIZE} MBT (+ notional-matched MET)")
        logger.info(f"  Stop: {STOP_PCT}% | Entry: Thu 00:00 UTC | Exit: Thu 23:05 UTC")
        logger.info("=" * 60)
        self.running = True
        await self._run()

    def stop(self) -> None:
        logger.info("Thursday Short Trader — stopping")
        self.running = False


async def main() -> None:
    trader = ThursdayShortTrader()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, trader.stop)
    try:
        await trader.run()
    except Exception as exc:
        logger.error(f"Fatal: {exc}", exc_info=True)
    finally:
        trader.stop()


if __name__ == "__main__":
    asyncio.run(main())
