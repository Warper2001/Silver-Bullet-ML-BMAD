"""combine_floor_monitor.py — combined-account floor monitor for the YANK + MIM-NB
joint Topstep 50K combine (sealed deployment prereg yank-mim-joint-combine-deploy).

Does NOT trade. Polls combined account equity and enforces the DERIVED halt triggers:
  - distance-to-floor: equity <= trailing_floor + $750  (updated 2026-06-28, was $500)
  - combined PF < 0.70 after 30 combined trades         (results_pf_trigger.md)
  - correlation: OBSERVE-ONLY (logged, never triggers)

On a trigger it flattens the whole account (intentional: halting everything),
stops both trader services, and drops a HALT flag. Equity = ProjectX account
balance + open-position MtM; the trailing floor is tracked locally mirroring
Topstep's ratchet (the hard floor is enforced by Topstep itself — this is the
early-warning layer).
"""
import asyncio
import csv
import hashlib
import json
import logging
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.research.projectx_auth import ProjectXAuth
from src.research.projectx_client import ProjectXClient

# ---- config (from the sealed deployment prereg) ----
ACCOUNT_ID = os.environ.get("PROJECTX_ACCOUNT_ID", "")
START_EQUITY = 50_000.0
FLOOR_START = 48_000.0
TRAIL = 2_000.0
HALT_DISTANCE = 750.0          # updated 2026-06-28: max_single_trade_loss × 1.5 = $500 × 1.5
PF_THRESHOLD = 0.70            # DERIVED — results_pf_trigger.md
PF_MIN_TRADES = 30
PASS_TARGET = 53_000.0
POLL_SEC = int(os.environ.get("MONITOR_POLL_SEC", "30"))
TRADER_IDS = ("trader-mim-nb", "trader-yank")
COMBINE_START = os.environ.get("COMBINE_START", "2026-06-17T00:00:00+00:00")

BASE = Path(__file__).parent.parent.parent
DATA = BASE / "data" / "combine_joint"
DATA.mkdir(parents=True, exist_ok=True)
HALT_FILE = DATA / "HALT"
STATE_FILE = DATA / "floor_state.json"
LOG_CSV = DATA / "monitor.csv"
DB_PATH = BASE / "data" / "trades.db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.FileHandler(BASE / "logs" / "combine_floor_monitor.log"),
                              logging.StreamHandler()])
logger = logging.getLogger("floor_monitor")


# ---------------------------------------------------------------- pure logic
def update_floor(prev_floor: float, hwm_equity: float) -> float:
    """Topstep trailing ratchet: floor rises with the high-water mark, capped at start."""
    return min(START_EQUITY, max(prev_floor, hwm_equity - TRAIL))


def evaluate_triggers(equity: float, floor: float, combined_pf, n_trades: int):
    """Return a halt reason string, or None. Distance-to-floor first (the binding one)."""
    if equity <= floor + HALT_DISTANCE:
        return (f"DISTANCE_TO_FLOOR: equity ${equity:,.0f} <= floor ${floor:,.0f} + ${HALT_DISTANCE:.0f} "
                f"(only ${equity - floor:,.0f} of room)")
    if n_trades >= PF_MIN_TRADES and combined_pf is not None and combined_pf < PF_THRESHOLD:
        return f"COMBINED_PF: {combined_pf:.2f} < {PF_THRESHOLD} after {n_trades} combined trades"
    return None


def combined_pf_and_count(db_path, since_iso):
    """Combined net PF and trade count across both bots since the combine start.
    Returns (pf_or_None, n). pf=None when there are no losses yet (undefined)."""
    try:
        con = sqlite3.connect(str(db_path))
        rows = con.execute(
            "SELECT pnl FROM trades WHERE trader_id IN (?,?) AND timestamp >= ? AND pnl IS NOT NULL",
            (*TRADER_IDS, since_iso),
        ).fetchall()
        con.close()
    except Exception as exc:
        logger.warning("combined_pf read failed: %s", exc)
        return None, 0
    pnls = [r[0] for r in rows]
    gp = sum(p for p in pnls if p > 0)
    gl = -sum(p for p in pnls if p < 0)
    pf = (gp / gl) if gl > 0 else None
    return pf, len(pnls)


# ---------------------------------------------------------------- state + log
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"hwm": START_EQUITY, "floor": FLOOR_START, "chain": "GENESIS"}


def save_state(st: dict):
    STATE_FILE.write_text(json.dumps(st, indent=2))


def log_tick(st, equity, floor, pf, n, reason):
    new = not LOG_CSV.exists()
    fields = ["ts_utc", "equity", "floor", "distance", "combined_pf", "n_trades", "reason", "chain"]
    payload = f"{equity}|{floor}|{equity-floor}|{pf}|{n}|{reason or ''}"
    st["chain"] = hashlib.sha256((st.get("chain", "GENESIS") + "|" + payload).encode()).hexdigest()[:16]
    with open(LOG_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({"ts_utc": datetime.now(timezone.utc).isoformat(), "equity": f"{equity:.2f}",
                    "floor": f"{floor:.2f}", "distance": f"{equity-floor:.2f}",
                    "combined_pf": f"{pf:.3f}" if pf is not None else "", "n_trades": n,
                    "reason": reason or "", "chain": st["chain"]})


# ---------------------------------------------------------------- halt action
_HALT_POLL_SEC = 5   # seconds between position checks during soft-halt wait
_HALT_POLL_MAX = 24  # 24 × 5s = 2 minutes before force-flatten timeout

async def do_halt(px, reason):
    logger.error("🛑 HALT TRIGGERED: %s", reason)
    HALT_FILE.write_text(json.dumps({"reason": reason, "ts": datetime.now(timezone.utc).isoformat()}, indent=2))
    try:
        size, _ = await px.net_position(ACCOUNT_ID)
        if size != 0:
            # Soft-halt: leave bracket orders intact so the open position stays
            # protected. Wait for the bot to close its own trade (TP/SL/time-stop).
            # Cancelling brackets before the position closes leaves a naked position
            # with no stop — that is the failure mode we are fixing here.
            logger.error(
                "SOFT HALT: open position %d ct — waiting up to %ds for natural close "
                "(brackets intact, services still running). New entries blocked by HALT file.",
                size, _HALT_POLL_SEC * _HALT_POLL_MAX,
            )
            for _ in range(_HALT_POLL_MAX):
                await asyncio.sleep(_HALT_POLL_SEC)
                try:
                    size, _ = await px.net_position(ACCOUNT_ID)
                except Exception as exc:
                    logger.warning("position poll error during soft-halt wait: %s", exc)
                    continue
                if size == 0:
                    logger.error("Position closed naturally — proceeding to stop services.")
                    break
            else:
                # 2-minute timeout: force-flatten at market
                logger.error(
                    "Position still open after %ds — force-flattening at market.",
                    _HALT_POLL_SEC * _HALT_POLL_MAX,
                )
                try:
                    await px.cancel_all_pending_orders(str(ACCOUNT_ID))
                    size, _ = await px.net_position(ACCOUNT_ID)
                    if size != 0:
                        await px.close_position_at_market(
                            "LONG" if size > 0 else "SHORT",
                            str(ACCOUNT_ID), contracts=abs(size),
                        )
                        logger.error("Force-flattened %d ct at market.", size)
                        await asyncio.sleep(5)
                except Exception as exc:
                    logger.error("Force-flatten error: %s", exc)
        else:
            # Already flat — cancel any stale pending orders immediately.
            await px.cancel_all_pending_orders(str(ACCOUNT_ID))
    except Exception as exc:
        logger.error("HALT position-check error (stopping services anyway): %s", exc)
    subprocess.run(["systemctl", "stop", "trader-mim-nb", "trader-yank"], check=False)
    logger.error("Stopped trader-mim-nb + trader-yank. HALT-and-REVIEW: human action required.")


# ---------------------------------------------------------------- main loop
async def main():
    if not ACCOUNT_ID:
        raise SystemExit("PROJECTX_ACCOUNT_ID not set — refusing to start")
    logger.info("Combine floor monitor — acct %s | halt at floor+$%.0f, PF<%.2f@%d | poll %ds",
                ACCOUNT_ID, HALT_DISTANCE, PF_THRESHOLD, PF_MIN_TRADES, POLL_SEC)
    auth = ProjectXAuth.from_file(".projectx_api_key")
    http = httpx.AsyncClient(timeout=30)
    cfg = type("_Cfg", (), {"symbol": "MNQU26", "contracts": 1})()
    px = ProjectXClient(auth, cfg, http, projectx_account_id=int(ACCOUNT_ID))
    st = load_state()
    while True:
        try:
            bal = await px.account_balance(ACCOUNT_ID)
            if bal is None:
                logger.warning("balance unavailable — skipping tick")
                await asyncio.sleep(POLL_SEC)
                continue
            size, upl = await px.net_position(ACCOUNT_ID)
            equity = bal + upl
            # HWM tracks realized balance only (mirrors Topstep's methodology).
            # Tracking equity (bal+unrealized) overstates HWM when open positions
            # are profitable, permanently ratcheting the floor from gains never realized.
            st["hwm"] = max(st["hwm"], bal)
            st["floor"] = update_floor(st["floor"], st["hwm"])
            pf, n = combined_pf_and_count(DB_PATH, COMBINE_START)
            reason = evaluate_triggers(equity, st["floor"], pf, n)
            log_tick(st, equity, st["floor"], pf, n, reason)
            # Publish the real combined balance/equity so trader buffer gates can
            # consume the authoritative shared floor (single source of truth incl.
            # both bots) instead of each recomputing it from its own ledger.
            st["balance"] = round(bal, 2)
            st["equity"] = round(equity, 2)
            st["ts_utc"] = datetime.now(timezone.utc).isoformat()
            save_state(st)
            if equity >= PASS_TARGET:
                logger.info("✅ PASS target reached: equity $%.0f >= $%.0f — confirm consistency rule, halt entries",
                            equity, PASS_TARGET)
            if reason and not HALT_FILE.exists():
                await do_halt(px, reason)
        except Exception as exc:
            logger.error("monitor loop error: %s", exc)
        await asyncio.sleep(POLL_SEC)


if __name__ == "__main__":
    asyncio.run(main())
