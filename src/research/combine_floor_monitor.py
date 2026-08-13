"""combine_floor_monitor.py — combined-account floor monitor for the YANK + MIM-NB
joint Topstep 50K combine (sealed deployment prereg yank-mim-joint-combine-deploy).

Does NOT trade. Polls combined account equity and enforces the DERIVED halt triggers:
  - distance-to-floor: equity <= trailing_floor + $100  (updated 2026-07-13, was $750)
  - combined PF < 0.70 after 30 combined trades         (results_pf_trigger.md)
  - correlation: OBSERVE-ONLY (logged, never triggers)

On a trigger it flattens the whole account (intentional: halting everything),
stops both trader services, and drops a HALT flag. Equity = ProjectX account
balance + open-position MtM; the trailing floor is tracked locally mirroring
Topstep's ratchet (the hard floor is enforced by Topstep itself — this is the
early-warning layer).

The floor state is bound to the account that produced it (see load_state): a
trailing floor is meaningless on a different account, and inheriting one across
a combine reset is how the 2026-07-06 breach stayed invisible.
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
HALT_DISTANCE = 100.0          # updated 2026-07-13: ride-the-buffer halt-review decision (halt_review_20260713_ride_buffer.md); was 750 (2026-06-28)
PF_THRESHOLD = 0.70            # DERIVED — results_pf_trigger.md
PF_MIN_TRADES = 30
PASS_TARGET = 53_000.0
POLL_SEC = int(os.environ.get("MONITOR_POLL_SEC", "30"))
# Report-only is the DEFAULT, and halting is now the opt-in.
#
# Alex removed floor-derived braking from both bots on 2026-07-29 (prereg
# mim-nb-risk-mechanics-removal §1 site 3 + Amendment 1) and this service was stopped and
# disabled. That also blinded the account readout — nothing tracked the high-water mark,
# so once the balance made new highs on 08-04 the recorded floor was stale and the
# reported cushion overstated. This mode restores the tracking WITHOUT restoring the brake.
#
# The default is inverted deliberately: with halting opt-out, any stray `systemctl start`
# of the old unit would re-arm a kill path the owner explicitly removed. With halting
# opt-in, the worst a stray start can do is log.
REPORT_ONLY = os.environ.get("FLOOR_MONITOR_REPORT_ONLY", "1") != "0"
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
def genesis_state(combine_start: str) -> dict:
    """A fresh combine: floor at the account's starting limit, PF window opens now."""
    return {"account_id": str(ACCOUNT_ID), "combine_start": combine_start,
            "hwm": START_EQUITY, "floor": FLOOR_START, "chain": "GENESIS"}


def load_state() -> dict:
    """Load the floor state, bound to the account it was built for.

    A trailing floor only means something on the account that produced it. Reusing
    one across a combine reset starts the new account with the dead one's numbers —
    so an account change re-genesises rather than inherits. Sealed prereg
    combine-restart-floor-hwm §2.
    """
    if not STATE_FILE.exists():
        return genesis_state(datetime.now(timezone.utc).isoformat())
    try:
        st = json.loads(STATE_FILE.read_text())
    except Exception:
        logger.warning("floor_state unreadable — genesis for acct %s", ACCOUNT_ID)
        return genesis_state(datetime.now(timezone.utc).isoformat())

    bound = st.get("account_id")
    if bound is None:
        # Legacy file, written before the binding existed. Adopt in place: whatever
        # floor it carries was built for the account running right now. Keeping the
        # env COMBINE_START here means the running monitor's PF window does not move.
        st["account_id"] = str(ACCOUNT_ID)
        st.setdefault("combine_start", COMBINE_START)
        logger.info("floor_state adopted by acct %s (legacy, unbound) — hwm $%.2f floor $%.2f",
                    ACCOUNT_ID, st.get("hwm", START_EQUITY), st.get("floor", FLOOR_START))
        return st

    if str(bound) != str(ACCOUNT_ID):
        archive = STATE_FILE.with_name(f"{STATE_FILE.name}.acct-{bound}")
        try:
            STATE_FILE.replace(archive)
        except Exception as exc:
            logger.warning("could not archive prior floor_state: %s", exc)
        fresh = genesis_state(datetime.now(timezone.utc).isoformat())
        logger.warning("ACCOUNT CHANGED %s -> %s — floor state re-genesised "
                       "(hwm $%.2f, floor $%.2f, combine_start %s); prior state archived to %s",
                       bound, ACCOUNT_ID, fresh["hwm"], fresh["floor"],
                       fresh["combine_start"], archive.name)
        return fresh
    return st


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
    if REPORT_ONLY:
        logger.warning("Combine floor monitor — acct %s | *** REPORT-ONLY: WILL NOT HALT "
                       "OR FLATTEN *** | tracks HWM/floor, publishes %s, logs triggers | "
                       "poll %ds", ACCOUNT_ID, STATE_FILE.name, POLL_SEC)
        logger.warning("REPORT-ONLY: neither bot has an automatic floor brake. Manual kill "
                       "switches remain: touch %s (MIM) / emergency-stop CLI (YANK).",
                       HALT_FILE)
    else:
        logger.info("Combine floor monitor — acct %s | ARMED: halt at floor+$%.0f, "
                    "PF<%.2f@%d | poll %ds",
                    ACCOUNT_ID, HALT_DISTANCE, PF_THRESHOLD, PF_MIN_TRADES, POLL_SEC)
    auth = ProjectXAuth.from_file(".projectx_api_key")
    http = httpx.AsyncClient(timeout=30)
    cfg = type("_Cfg", (), {"symbol": "MNQU26", "contracts": 1})()
    px = ProjectXClient(auth, cfg, http, projectx_account_id=int(ACCOUNT_ID))
    st = load_state()
    logger.info("floor state: acct %s | hwm $%.2f | floor $%.2f | PF window from %s",
                st.get("account_id"), st.get("hwm", START_EQUITY),
                st.get("floor", FLOOR_START), st.get("combine_start", COMBINE_START))
    while True:
        try:
            bal = await px.account_balance(ACCOUNT_ID)
            if bal is None:
                logger.warning("balance unavailable — skipping tick")
                await asyncio.sleep(POLL_SEC)
                continue
            size, upl = await px.net_position(ACCOUNT_ID)
            equity = bal + upl
            # HWM tracks EQUITY (balance + open MtM), because that is what Topstep's
            # trailing MLL ratchets on. 317f3ff (2026-06-26) switched this to realized
            # balance on the belief that it "mirrors Topstep's methodology"; it does not.
            # That cost $955.56 of floor here, and the hand-written floor_state reset
            # shipped alongside it cost $712.50 more — which is why acct 23884932
            # breached on 2026-07-06 with this monitor still reporting $1,339 of room.
            #
            # Yes, this ratchets the floor on unrealized gains that may never be realized.
            # That is Topstep's rule, not a modelling choice. The ratchet and the breach
            # test below must use one denomination — splitting them is the actual bug.
            # Sealed prereg combine-restart-floor-hwm §1.
            st["hwm"] = max(st["hwm"], equity)
            st["floor"] = update_floor(st["floor"], st["hwm"])
            pf, n = combined_pf_and_count(DB_PATH, st.get("combine_start", COMBINE_START))
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
            if reason and REPORT_ONLY:
                # Log loudly, change nothing. No HALT file is written — MIM-NB SystemExits
                # at startup if one exists, so writing it here would be a kill path
                # smuggled in through a "report-only" mode.
                logger.warning("TRIGGER (report-only, NOT halting): %s", reason)
            elif reason and not HALT_FILE.exists():
                await do_halt(px, reason)
        except Exception as exc:
            logger.error("monitor loop error: %s", exc)
        await asyncio.sleep(POLL_SEC)


if __name__ == "__main__":
    asyncio.run(main())
