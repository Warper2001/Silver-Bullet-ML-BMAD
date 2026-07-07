#!/usr/bin/env python3
"""
MIM-Noise-Bands live combine bot — Topstep 50K via ProjectX.

Strategy: sealed mim-nb-v2-catstop S-B spec (prereg 6957daa, deployment prereg 7939eed).
  - Noise bands sigma(t) = 14-day mean |close(d,t)/open_d - 1| per RTH minute label
  - UB = O*(1+sigma) + max(Cprev-O,0); LB = O*(1-sigma) - max(O-Cprev,0)
  - Entry/reversal checks at HH:00/HH:30 completions 10:00-15:30 ET (market order)
  - Wide band-stop checks 10:00-16:00 (long exits close<LB, short exits close>UB)
  - Catastrophe stop: resting stop at entry -/+ 250 pts, placed after entry
  - EOD flatten at the 16:00 ET bar; DLL guard: day P&L <= -$500 -> no entries
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
import signal
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
from src.research.projectx_bars import fetch_px_ts_shaped, ProjectXBarFetchError
from src.research.shadow_parity import ShadowParityLogger, bars_by_minute

# ----------------------------------------------------------------------
# Frozen configuration (catstop-250 prereg 30bc6a8 — do not edit without a new seal)
# ----------------------------------------------------------------------
SYMBOL = os.environ.get("MIM_NB_SYMBOL", "MNQU26")   # startup seed / fallback when autoroll disabled
AUTOROLL = os.environ.get("MIM_NB_AUTOROLL", "1") != "0"  # set 0 to pin MIM_NB_SYMBOL (manual roll)
CONTRACTS = 1
CAT_STOP_PTS = 250.0   # was 500 — halved to bring max loss to 25% of $2k trailing DD
LOOKBACK_DAYS = 14
DLL_GUARD_USD = -1000.0  # MC-authorized day cut (seal: preregistration_mim_nb_dll_parity_reversion.md); permits one post-cat-stop re-entry
PT_VAL = 2.0
ET = pytz.timezone("America/New_York")

# Combine account math (Topstep 50K EOD trailing drawdown)
COMBINE_START_BALANCE = 50_000.0
MLL_DD = 2_000.0  # Topstep 50K maximum loss limit (trailing, EOD)

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "mim_nb"
# Authoritative shared floor written every tick by combine_floor_monitor. Its
# balance/equity reflect the REAL combined account (incl. YANK), and its floor is
# the recalibrated Topstep trailing floor — single source of truth for the buffer
# gate. MIM falls back to its own realized ledger only if this state is stale/absent.
FLOOR_STATE_FILE = BASE_DIR / "data" / "combine_joint" / "floor_state.json"
FLOOR_STATE_MAX_AGE_S = 300.0  # 10 monitor ticks; older => assume monitor down, fall back
LOG_DIR = BASE_DIR / "logs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "mim_nb_live.log"), logging.StreamHandler()],
)
logger = logging.getLogger("mim_nb")

def _bars_url(symbol: str) -> str:
    return f"https://api.tradestation.com/v3/marketdata/barcharts/{symbol}?interval=1&unit=Minute"


def _contract_id_to_symbol(contract_id: str) -> str:
    """'CON.F.US.MNQ.U26' -> 'MNQU26' (inverse of projectx_client._to_contract_id)."""
    parts = contract_id.split(".")
    return f"{parts[3]}{parts[4]}"


async def resolve_front_month(http, px_auth, root: str = "MNQ") -> str:
    """Return the active front-month TradeStation symbol (e.g. 'MNQU26').

    Source of truth is ProjectX's own `activeContract` flag via /Contract/search —
    i.e. the exact contract the broker will accept orders for. Falls back to the
    date-based FuturesSymbolGenerator (third-Friday math) only if the API call
    fails, so a broker outage never blocks the bot. Added after the 2026-06-16
    incident where a stale front month silently rejected every entry.
    """
    try:
        token = await px_auth.authenticate()
        headers = {"Authorization": f"Bearer {token}",
                   "Content-Type": "application/json", "Accept": "application/json"}
        r = await http.post(f"{_BASE_URL}/Contract/search",
                            json={"searchText": root, "live": False}, headers=headers)
        if r.status_code == 200:
            active = [c for c in r.json().get("contracts", [])
                      if c.get("activeContract")
                      and str(c.get("id", "")).split(".")[3:4] == [root]]
            if active:
                return _contract_id_to_symbol(active[0]["id"])
            logger.warning("Contract/search: no active %s contract in response — date fallback", root)
        else:
            logger.warning("Contract/search HTTP %s — date fallback", r.status_code)
    except Exception as exc:
        logger.warning("Contract/search failed (%s) — date fallback", exc)
    from src.data.futures_symbols import FuturesSymbolGenerator
    return FuturesSymbolGenerator()._find_current_contract().symbol


ENTRY_MARKS = {f"{h:02d}:{m}" for h in range(10, 16) for m in ("00", "30")} - {"16:00"}
CHECK_MARKS = ENTRY_MARKS | {"16:00"}

# Early-close sessions — sealed-engine parity: the engine skips any session without
# a 16:00 bar (no marks, no sigma append, prev_close carries over). CME equity-futures
# early closes for 2026 remainder; extend via MIM_EARLY_CLOSE_EXTRA=YYYY-MM-DD,...
# (seal: preregistration_mim_nb_ops_hardening.md)
EARLY_CLOSE_DATES = {"2026-09-07", "2026-11-26", "2026-11-27", "2026-12-24"} | {
    d.strip() for d in os.environ.get("MIM_EARLY_CLOSE_EXTRA", "").split(",") if d.strip()}

RECONCILE_INTERVAL_S = 30.0  # broker-truth stop-fill check cadence between marks


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
        self._ts_sim_mirror = None  # TSSimMirror when MIM_MIRROR_TS_SIM=1
        self._cfg = None
        self.symbol = SYMBOL
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
        self._running = True
        self._early_close_logged = None   # date already announced as stand-down
        self._last_reconcile_mono = 0.0   # monotonic ts of last stop-fill reconcile

        # Combine balance tracking for buffer-aware risk gates
        self._realized_pnl = 0.0       # cumulative realized P&L across all sessions
        self._mll_eod_hwm = COMBINE_START_BALANCE  # EOD high-water mark for MLL floor
        self._buffer_source = "own-ledger"  # set by _remaining_mll_buffer: "shared" | "own-ledger"

        # Data backend (Stage-1 shadow migration to ProjectX). Default = TradeStation REST.
        self._data_source = os.environ.get("MIM_NB_DATA_SOURCE", "tradestation")
        self._data_shadow = (os.environ.get("MIM_NB_DATA_SHADOW", "0") == "1"
                             and self._data_source == "tradestation")
        self._data_px_live = os.environ.get("MIM_NB_DATA_PX_LIVE", "0") == "1"
        self._shadow_logger = None

    # ------------------------------------------------------------------
    # Combine buffer tracking (risk gate support)
    # ------------------------------------------------------------------
    def _init_combine_balance(self):
        """Replay trades.csv to compute cumulative realized P&L and EOD MLL floor.
        Called once at startup so the buffer gate has accurate state from day one."""
        path = DATA_DIR / "trades.csv"
        if not path.exists():
            return
        try:
            running = COMBINE_START_BALANCE
            hwm = COMBINE_START_BALANCE
            with open(path) as f:
                for row in csv.DictReader(f):
                    pnl = float(row.get("pnl_usd", "0").replace("+", ""))
                    running += pnl
                    hwm = max(hwm, running)
            self._realized_pnl = running - COMBINE_START_BALANCE
            self._mll_eod_hwm = hwm
            mll_floor = hwm - MLL_DD
            buf = running - mll_floor
            cat_cost = CAT_STOP_PTS * PT_VAL * CONTRACTS
            logger.info("COMBINE BALANCE: realized=%+.2f balance=%.2f hwm=%.2f "
                        "mll_floor=%.2f buffer=%.2f cat_cost=%.2f",
                        self._realized_pnl, running, hwm, mll_floor, buf, cat_cost)
            if buf <= cat_cost:
                logger.warning("BUFFER WARNING: %.2f ≤ cat-stop cost %.2f — "
                               "entries will be blocked until buffer recovers",
                               buf, cat_cost)
        except Exception as exc:
            logger.warning("combine balance init failed: %s — buffer gate disabled", exc)

    def _shared_floor_buffer(self):
        """Remaining MLL buffer from the floor monitor's authoritative state, using
        the REAL combined equity (incl. YANK) and the recalibrated trailing floor.
        Returns None if the shared state is missing/stale/malformed so the caller
        falls back to MIM's own conservative ledger (never trade on stale risk data)."""
        try:
            st = json.loads(FLOOR_STATE_FILE.read_text())
            floor = float(st["floor"])
            equity = float(st["equity"])
            age = (datetime.now(timezone.utc)
                   - datetime.fromisoformat(st["ts_utc"])).total_seconds()
            if age > FLOOR_STATE_MAX_AGE_S:
                return None
            return equity - floor
        except Exception:
            return None

    def _remaining_mll_buffer(self) -> float:
        """Remaining Topstep MLL buffer = current estimated equity − MLL floor.
        Prefers the floor monitor's shared state (real combined account incl. YANK,
        recalibrated floor); falls back to MIM's own realized ledger if that state
        is unavailable/stale."""
        shared = self._shared_floor_buffer()
        if shared is not None:
            self._buffer_source = "shared"
            return shared
        self._buffer_source = "own-ledger"
        balance = COMBINE_START_BALANCE + self._realized_pnl + self.day_pnl
        mll_floor = self._mll_eod_hwm - MLL_DD
        return balance - mll_floor

    # ------------------------------------------------------------------
    def _apply_symbol(self, sym: str):
        """Point every contract reference (bars URL, ProjectX orders, position
        verify) at `sym`. Single place that mutates the active contract."""
        self.symbol = sym
        self.contract_id = _to_contract_id(sym)
        if self.px is not None:
            self.px._contract_id = self.contract_id
        if self._cfg is not None:
            self._cfg.symbol = sym

    async def initialize(self):
        _halt = BASE_DIR / "data" / "combine_joint" / "HALT"
        if _halt.exists():
            raise SystemExit(f"HALT flag present ({_halt}) — combine floor monitor halted trading; "
                             "remove the flag after review to resume.")
        self.http = httpx.AsyncClient(timeout=30)
        self.ts_auth = TradeStationAuthV3.from_file(".access_token")
        await self.ts_auth.authenticate()
        # Keep the TS token fresh so the SIM mirror does not 401 on stale
        # tokens (TS access tokens expire ~20 min). Matches YANK:957.
        await self.ts_auth.start_auto_refresh()
        self.px_auth = ProjectXAuth.from_file(".projectx_api_key")

        # Auto-roll: resolve the broker's active front month at startup so a
        # restart always trades the contract that will actually accept orders.
        if AUTOROLL:
            sym = await resolve_front_month(self.http, self.px_auth)
            if sym != self.symbol:
                logger.info("AUTOROLL startup: %s → %s (broker active contract)", self.symbol, sym)
            self._apply_symbol(sym)
        else:
            logger.info("AUTOROLL disabled (MIM_NB_AUTOROLL=0) — pinned to %s", self.symbol)

        logger.info("=" * 70)
        logger.info("MIM-NB LIVE — %s — %d contract — cat stop %.0f pts — account %s",
                    self.symbol, CONTRACTS, CAT_STOP_PTS, self.account_id)
        logger.info("Sealed spec 6957daa / deployment 7939eed")
        logger.info("=" * 70)

        self._cfg = type("_Cfg", (), {"symbol": self.symbol})()
        # Optional best-effort TradeStation SIM order mirror (default OFF). Cannot
        # delay/block/crash the authoritative ProjectX combine path — see ts_sim_mirror.
        if os.environ.get("MIM_MIRROR_TS_SIM", "0") == "1":
            from src.research.ts_sim_mirror import (TSSimMirror, MirrorProjectXClient,
                                                    SimScaler, InvVolScaler)
            _sim_dir = BASE_DIR / "data" / "ts_sim_mirror"
            if os.environ.get("SIM_INVVOL", "0") == "1":
                # Inverse-vol allocation paper-track: MIM held at 1ct in SIM (same
                # as the live combine) alongside YANK trimmed to 1ct — see InvVolScaler.
                _scaler = InvVolScaler("MIM-NB", contracts=1, log=logger)
                _eq_log = _sim_dir / "mim_invvol_equity.csv"
            else:
                _scaler = SimScaler("MIM-NB", base_contracts=CONTRACTS,
                                    state_path=_sim_dir / "mim_nb_scaler.json", log=logger)
                _eq_log = None
            self._ts_sim_mirror = TSSimMirror(self.ts_auth, scaler=_scaler,
                                              equity_log_path=_eq_log, log=logger)
            await self._ts_sim_mirror.start()
            self.px = MirrorProjectXClient(self.px_auth, self._cfg, self.http,
                                           projectx_account_id=self.account_id,
                                           ts_mirror=self._ts_sim_mirror)
            logger.info("TS SIM MIRROR: ENABLED — combine orders also copied to SIM (best-effort)")
        else:
            self.px = ProjectXClient(self.px_auth, self._cfg, self.http,
                                     projectx_account_id=self.account_id)
        if self._data_shadow:
            self._shadow_logger = ShadowParityLogger(DATA_DIR / "shadow_parity.csv")
        logger.info("DATA: %s (signal)%s | px_contract=%s live=%s", self._data_source,
                    " + projectx SHADOW" if self._data_shadow else "",
                    self.contract_id, self._data_px_live)
        self._init_combine_balance()
        await self._backfill()
        await self._reconcile_startup()
        await self._catch_up_today()

    async def _maybe_roll(self):
        """At a session boundary, roll to the new front month if the broker's
        active contract changed. Only runs while FLAT (MIM-NB is flat overnight),
        so it never strands a live position. Re-seeds sigma/prev_close from the
        new contract's price series. Fixes the 2026-06-16 stale-contract incident
        without needing a manual restart each quarter."""
        if not AUTOROLL or self.position != 0:
            return
        sym = await resolve_front_month(self.http, self.px_auth)
        if sym == self.symbol:
            return
        logger.warning("AUTOROLL: front month %s → %s — reseeding on new contract",
                       self.symbol, sym)
        self._apply_symbol(sym)
        await self._backfill()              # reseed sigma_hist + prev_close on new contract
        self.open_d = None                  # backfill is authoritative; skip _new_session re-fold
        self.today_moves = {}
        orders_log.append({"ts_utc": datetime.now(timezone.utc).isoformat(),
                           "event": "ROLL", "order_id": "", "otype": "", "side": "",
                           "size": "", "price": "", "outcome": sym, "detail": "AUTOROLL"})

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
        # Data backend: ProjectX bars (TS-shaped, +1-min-aligned, roll-tracked via
        # self.contract_id) when MIM_NB_DATA_SOURCE=projectx; else TradeStation REST.
        if self._data_source == "projectx":
            return await fetch_px_ts_shaped(
                self.http, self.px_auth, self.contract_id,
                now_utc=datetime.now(timezone.utc), live=self._data_px_live, barsback=barsback)
        token = await self._ts_token()
        url = f"{_bars_url(self.symbol)}&barsback={barsback}"
        r = await self.http.get(url, headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        return r.json().get("Bars", [])

    async def _run_shadow_parity(self, ts_bars):
        """Stage-1 shadow: fetch ProjectX bars in parallel and log TS-vs-PX parity to
        data/mim_nb/shadow_parity.csv. Observation only — never touches trade state.
        All failures swallowed (must not affect the live MIM-NB signal path)."""
        try:
            import time as _t
            now_utc = datetime.now(timezone.utc)
            t0 = _t.perf_counter()
            error = ""
            try:
                px = await fetch_px_ts_shaped(self.http, self.px_auth, self.contract_id,
                                              now_utc=now_utc, live=self._data_px_live, barsback=60)
            except ProjectXBarFetchError as e:
                px, error = [], str(e)
            fetch_ms = (_t.perf_counter() - t0) * 1000.0
            self._shadow_logger.log_poll(bars_by_minute(ts_bars), bars_by_minute(px),
                                         now_utc, fetch_ms, error)
        except Exception as e:
            logger.warning("shadow parity logging failed (non-fatal): %s", e)

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

    def _load_persisted_position(self):
        """Read MIM-NB's OWN last-known state from its hash-chained state.json.
        Required for commingling-safe recovery: MIM-NB shares account+contract with
        YANK, which net into ONE position, so we recover from our own file — never
        from net account position."""
        p = DATA_DIR / "state.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception as exc:
            logger.warning("state.json read failed (%s) — treating as flat", exc)
            return None

    async def _reconcile_startup(self):
        """Commingling-safe startup recovery. NEVER reads net position or flattens/
        cancels-all (that would clobber YANK, which shares this account+contract).
        Recovers our position from our own state file and verifies our own cat-stop
        order by ID."""
        st = self._load_persisted_position()
        believed = int(st.get("position", 0)) if st else 0
        if believed == 0:
            logger.info("Startup reconcile: own state FLAT — no account action ✓")
            return
        # We believed we were holding — restore our own state.
        self.position = believed
        self.entry_px = float(st.get("entry_px", 0.0))
        self.entry_t = st.get("entry_t")
        self.day = st.get("day")
        cat_id = st.get("cat_stop_id")
        self.cat_stop_id = cat_id
        open_flag = await self.px.is_order_open(cat_id) if cat_id is not None else None
        if open_flag is True:
            logger.warning("Startup reconcile: RESUMED %s position (entry %.2f, cat-stop #%s live)",
                           "LONG" if believed == 1 else "SHORT", self.entry_px, cat_id)
        elif open_flag is False:
            # cat-stop gone while we were offline → it filled. Book the stop exit.
            stop_px = self.entry_px - CAT_STOP_PTS if believed == 1 else self.entry_px + CAT_STOP_PTS
            self._record_trade(stop_px, datetime.now(ET).strftime("%H:%M"), "CAT_STOP_OFFLINE")
            await self.px.cancel_orders([cat_id])
            self.cat_stop_id = None
            logger.warning("Startup reconcile: cat-stop #%s filled while offline — booked CAT_STOP @ %.2f",
                           cat_id, stop_px)
        else:
            logger.warning("Startup reconcile: cat-stop status UNKNOWN (id=%s) — resuming holding; "
                           "next check-mark verifies", cat_id)

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

    async def _search_trades(self, hours_back: float = 8.0) -> list:
        """Our account's broker-side fills in the recent window (truth source)."""
        headers = await self.px._headers()
        since = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
        r = await self.http.post(f"{_BASE_URL}/Trade/search",
                                 json={"accountId": self.account_id,
                                       "startTimestamp": since}, headers=headers)
        if r.status_code != 200:
            return []
        return r.json().get("trades", []) or []

    async def _find_fill_for(self, order_id) -> dict | None:
        for t in await self._search_trades():
            if t.get("orderId") == order_id:
                return t
        return None

    async def _find_closing_fill(self) -> dict | None:
        """Most recent fill that would close our current position and isn't one of
        our known orders — evidence of an external close (07-06 pattern)."""
        closing_side = _SIDE_SELL if self.position == 1 else _SIDE_BUY
        ours = {self.cat_stop_id}
        candidates = [t for t in await self._search_trades()
                      if t.get("side") == closing_side and t.get("size") == CONTRACTS
                      and t.get("orderId") not in ours
                      and t.get("profitAndLoss") is not None]  # P&L set = position-closing fill
        if not candidates:
            return None
        return max(candidates, key=lambda t: t.get("creationTimestamp", ""))

    async def _reconcile_stop_fill(self):
        """Broker-truth reconcile between marks (seal: preregistration_mim_nb_ops_hardening.md).
        Detects our cat-stop filling — or being externally canceled/closed — within
        ~RECONCILE_INTERVAL_S instead of at the next 30-min mark, and never books an
        exit the broker doesn't corroborate. Commingling-safe: inspects only our own
        order ID and our own fills, never net position."""
        if self.position == 0 or self.cat_stop_id is None:
            return
        open_flag = await self.px.is_order_open(self.cat_stop_id)
        if open_flag is not False:
            return  # True = still resting; None = unknown, retry next cycle
        old_id = self.cat_stop_id
        now_hm = datetime.now(ET).strftime("%H:%M")
        stop_px = self.entry_px - CAT_STOP_PTS if self.position == 1 else self.entry_px + CAT_STOP_PTS
        fill = await self._find_fill_for(old_id)
        if fill is not None:
            orders_log.append({"ts_utc": datetime.now(timezone.utc).isoformat(),
                               "event": "FILL", "order_id": old_id, "otype": _TYPE_STOP,
                               "side": fill.get("side"), "size": fill.get("size"),
                               "price": fill.get("price"), "outcome": "OK",
                               "detail": f"fees={fill.get('fees')} pnl={fill.get('profitAndLoss')}"})
            logger.warning("CAT-STOP FILLED @ %s (broker, order #%s) — booking at stop level "
                           "%.2f per sealed convention", fill.get("price"), old_id, stop_px)
            self._record_trade(stop_px, now_hm, "CAT_STOP")
            await self.px.cancel_orders([old_id])  # mirror hygiene (TS-SIM stop may still rest)
            self.cat_stop_id = None
            return
        close_fill = await self._find_closing_fill()
        if close_fill is not None:
            px_real = float(close_fill["price"])
            logger.critical("EXTERNAL CLOSE: our stop #%s canceled UNFILLED and closing fill "
                            "@ %.2f exists (order #%s) — booking EXTERNAL_CLOSE at broker "
                            "truth; safe-mode for the day", old_id, px_real,
                            close_fill.get("orderId"))
            self._record_trade(px_real, now_hm, "EXTERNAL_CLOSE")
            await self.px.cancel_orders([old_id])
            self.cat_stop_id = None
            self.day_deactivated = True
            return
        logger.critical("CAT-STOP #%s gone (no fill, no closing fill) — position UNPROTECTED, "
                        "re-placing protective stop @ %.2f", old_id, stop_px)
        sid = await self._order(_TYPE_STOP,
                                _SIDE_SELL if self.position == 1 else _SIDE_BUY, stop_px)
        self.cat_stop_id = sid
        if sid is None:
            logger.error("protective stop re-place REJECTED — flattening per seal")
            await self._flatten("STOP_REPLACE_FAILED")

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
        self._save_state()  # persist holding state so an intraday restart recovers it (commingling-safe)
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
        await self.px.cancel_orders([self.cat_stop_id])  # commingling-safe: only our own resting order
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
        self._realized_pnl += pnl_usd
        self._mll_eod_hwm = max(self._mll_eod_hwm,
                                COMBINE_START_BALANCE + self._realized_pnl)
        trades_log.append({"day": str(self.day), "dir": self.position,
                           "entry_t": self.entry_t, "entry_px": f"{self.entry_px:.2f}",
                           "exit_t": exit_t, "exit_px": f"{exit_px:.2f}", "reason": reason,
                           "pnl_pts": f"{pnl_pts:+.2f}", "pnl_usd": f"{pnl_usd:+.2f}",
                           "day_pnl_usd": f"{self.day_pnl:+.2f}"})
        buf = self._remaining_mll_buffer()
        logger.info("TRADE CLOSED %s %s→%s %+.2f pts ($%+.2f) day P&L %+.2f buffer %.2f [%s]",
                    "L" if self.position == 1 else "S", self.entry_t, exit_t,
                    pnl_pts, pnl_usd, self.day_pnl, buf, reason)
        self.position = 0
        self._save_state()  # persist flat state immediately after close (commingling-safe recovery)
        # Dynamic DLL: allow losing at most (buffer - cat_stop_cost) today, up to static cap
        cat_cost = CAT_STOP_PTS * PT_VAL * CONTRACTS
        dynamic_dll = -min(abs(DLL_GUARD_USD), max(0.0, buf + cat_cost))
        if self.day_pnl <= dynamic_dll and not self.day_deactivated:
            self.day_deactivated = True
            logger.warning("DLL GUARD: day P&L $%.2f ≤ %.2f (dynamic) buffer=%.2f — "
                           "entries disabled until next session",
                           self.day_pnl, dynamic_dll, buf)

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
        if str(et.date()) in EARLY_CLOSE_DATES:
            if self._early_close_logged != et.date():
                self._early_close_logged = et.date()
                logger.warning("EARLY-CLOSE session %s — standing down for engine parity "
                               "(no marks, no sigma update, prev_close carries over)", et.date())
            return
        if et.date() != self.day:
            await self._maybe_roll()       # quarterly contract roll, while flat
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
            # Commingling-safe: the account nets MIM+YANK, so detect our OWN cat-stop
            # fill by whether its order ID is still open — never by net position.
            cat_filled = self.cat_stop_id is not None and (await self.px.is_order_open(self.cat_stop_id) is False)
            if self.position != 0 and cat_filled:
                # catastrophe stop filled since last check
                stop_px = self.entry_px - CAT_STOP_PTS if self.position == 1 else self.entry_px + CAT_STOP_PTS
                self._record_trade(stop_px, hm, "CAT_STOP")
                await self.px.cancel_orders([self.cat_stop_id])
                self.cat_stop_id = None
                action = "CAT_STOP_DETECTED"
            elif self.position == 1 and c < lb:
                await self._exit(c, hm, "STOP")
                action = "BAND_STOP_EXIT"
            elif self.position == -1 and c > ub:
                await self._exit(c, hm, "STOP")
                action = "BAND_STOP_EXIT"
            if not self.day_deactivated and hm in ENTRY_MARKS:
                buf = self._remaining_mll_buffer()
                cat_cost = CAT_STOP_PTS * PT_VAL * CONTRACTS
                if buf <= cat_cost:
                    logger.warning("BUFFER_GATE %s: buffer=%.2f ≤ cat_cost=%.2f [%s] — entry blocked",
                                   hm, buf, cat_cost, self._buffer_source)
                elif c > ub and self.position != 1:
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

    def _save_state(self):
        (DATA_DIR / "state.json").write_text(json.dumps({
            "day": str(self.day), "position": self.position, "entry_px": self.entry_px,
            "entry_t": self.entry_t, "cat_stop_id": self.cat_stop_id,
            "day_pnl": self.day_pnl, "prev_close": self.prev_close,
            "chains": {"bars": bars_log.head, "decisions": decisions_log.head,
                       "orders": orders_log.head, "trades": trades_log.head},
            "saved_at": datetime.now(timezone.utc).isoformat()}, indent=2))

    # ------------------------------------------------------------------
    async def _shutdown(self, sig_name: str):
        """Graceful shutdown: flatten any open position before exiting.
        Registered on SIGTERM/SIGINT so `systemctl stop` closes live positions
        rather than leaving them orphaned until the cat-stop fires."""
        if not self._running:
            return
        self._running = False
        logger.warning("SHUTDOWN (%s) — flattening if held", sig_name)
        if self.position != 0:
            try:
                await self._flatten(f"SHUTDOWN_{sig_name}")
            except Exception as exc:
                logger.error("flatten on shutdown failed: %s", exc)
        logger.info("SHUTDOWN complete")

    async def run(self):
        await self.initialize()
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig, lambda s=sig: asyncio.ensure_future(self._shutdown(s.name)))
        logger.info("Polling loop started (10s)")
        while self._running:
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
                if self._data_shadow:
                    await self._run_shadow_parity(bars)
                mono = asyncio.get_event_loop().time()
                if mono - self._last_reconcile_mono >= RECONCILE_INTERVAL_S:
                    self._last_reconcile_mono = mono
                    await self._reconcile_stop_fill()
            except Exception as exc:
                logger.error("poll loop error: %s", exc)
            # 2s cadence in the first 20s of each minute (catch the completed bar
            # the moment TradeStation publishes it, ~6-9s past the minute);
            # 10s otherwise. Bounds check-to-order latency at the API publish lag.
            if self._running:
                await asyncio.sleep(2 if datetime.now(timezone.utc).second < 20 else 10)
        logger.info("Poll loop exited cleanly")


if __name__ == "__main__":
    asyncio.run(MimNbLive().run())
