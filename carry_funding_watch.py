#!/usr/bin/env python3
"""
carry_funding_watch.py
======================
Arms the BTC cash-and-carry sleeve to ANNOUNCE itself. The executor
(btc_carry_executor.py, paper mode) already monitors Kraken funding 24/7 and
auto-enters when annualised funding holds > the +10% hurdle for 3 consecutive
8h periods. But that transition would otherwise be buried in a verbose poll log.

This watcher (run hourly by carry-funding-watch.timer) reads the executor's
state.json and emits a LOUD, deduplicated alert to logs/carry_normalization_alert.log
the moment funding starts normalising (above_hurdle_count climbs) or a carry
entry actually fires (status -> ACTIVE). Idempotent per transition via
logs/carry_watch_state.json so it never spams.

It changes nothing live and places no orders — it only surfaces the regime change
so Alex can make the paper->live (real-money) decision. Going live separately
requires Kraken API credentials in .env and the executor's --live flag.
"""
import json, re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/Silver-Bullet-ML-BMAD")
STATE = ROOT / "data/carry_executor_state.json"
WATCH = ROOT / "logs/carry_watch_state.json"
ALERT = ROOT / "logs/carry_normalization_alert.log"
EXEC_LOG = ROOT / "logs/btc_carry_executor.log"
CONFIRM_PERIODS = 3

def now(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def alert(msg):
    line = f"[{now()}] {msg}"
    print(line)
    with open(ALERT, "a") as f:
        f.write(line + "\n")

def latest_funding():
    """best-effort: last 'funding=-XX.XX%ann' from the executor log."""
    try:
        tail = EXEC_LOG.read_text(errors="replace").splitlines()[-50:]
        for ln in reversed(tail):
            m = re.search(r"funding=(-?[0-9.]+)%ann", ln)
            if m: return float(m.group(1))
    except Exception:
        pass
    return None

def main():
    if not STATE.exists():
        return  # executor not initialised; nothing to watch
    st = json.loads(STATE.read_text())
    status = st.get("status", "FLAT")
    above = int(st.get("above_hurdle_count", 0))
    n_trades = int(st.get("n_trades", 0))

    prev = {"status": "FLAT", "above": -1, "n_trades": n_trades}
    if WATCH.exists():
        try: prev.update(json.loads(WATCH.read_text()))
        except Exception: pass

    fund = latest_funding()
    fstr = f" (funding={fund:+.2f}%ann)" if fund is not None else ""

    # 1) entry fired: status flipped FLAT -> ACTIVE (or n_trades incremented)
    if status == "ACTIVE" and prev["status"] != "ACTIVE":
        alert(f"🟢 BTC-CARRY ENTERED (paper){fstr} — funding normalised and the 3x8h "
              f"confirmation completed. DECISION: review and decide on going LIVE "
              f"(needs Kraken creds in .env + executor --live). entry_ann="
              f"{st.get('entry_funding_ann',0)*100:.2f}%")
    # 2) normalisation onset / progress while still FLAT (count climbing toward entry)
    elif status == "FLAT" and above > prev["above"] and above >= 1:
        eta = "ENTRY IMMINENT next 8h period" if above >= CONFIRM_PERIODS - 1 else "watching"
        alert(f"🟡 BTC funding normalising{fstr}: {above}/{CONFIRM_PERIODS} consecutive "
              f"8h readings > +10% hurdle — {eta}.")
    # 3) informational: exited back to FLAT
    elif status == "FLAT" and prev["status"] == "ACTIVE":
        alert(f"⚪ BTC-CARRY exited back to FLAT{fstr}. Sleeve dormant again, still armed.")

    WATCH.write_text(json.dumps(
        {"status": status, "above": above, "n_trades": n_trades, "checked": now()}, indent=2))

if __name__ == "__main__":
    main()
