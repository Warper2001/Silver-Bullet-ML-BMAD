"""Emergency stop CLI for the Tier2StreamingTrader paper-trading system.

Usage:
    python -m src.cli.emergency_stop [--force]

Actions taken:
  1. Cancel all pending SIM orders via TradeStation API.
  2. If an active trade is found in persisted state, close position at market
     and write a MANUAL exit record to the trade log.
  3. Persist daily_halted=True via RiskManager so the system stays halted
     even after a restart.

--force: persist the halt even if the API calls fail; exit code 1 on API error.
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone

import httpx

from src.research.tier2_streaming_working import (
    RiskManager,
    SIM_ACCOUNT_ID,
    StatePersistence,
    TradeLogger,
    TradeRecord,
    TradeStationAuthV3,
    TradeStationClient,
    _default_account_config,
)


async def _run_emergency_stop(force: bool = False) -> int:
    """Execute the emergency stop sequence. Returns exit code (0=success, 1=API error)."""
    auth = TradeStationAuthV3.from_file(".access_token")
    await auth.authenticate()

    account_config = _default_account_config("MNQM26")
    exit_code = 0

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        ts_client = TradeStationClient(auth, account_config, http_client)

        try:
            cancelled = await ts_client.cancel_all_pending_orders(SIM_ACCOUNT_ID)
            print(f"Cancelled {len(cancelled)} pending order(s): {cancelled}")
        except Exception as exc:
            print(f"WARNING: cancel_all_pending_orders failed: {exc}", file=sys.stderr)
            exit_code = 1
            if not force:
                # Still persist the halt before returning
                rm = RiskManager()
                rm.halt_manually()
                return exit_code

        # Check for an active trade in persisted state
        state = StatePersistence.load_state()
        if state and state.get("direction") and state.get("entry_price") is not None:
            direction = state["direction"]
            try:
                await ts_client.close_position_at_market(direction, SIM_ACCOUNT_ID)

                # Log the manual exit
                entry_time_str = state.get("entry_time", "")
                try:
                    entry_time = datetime.fromisoformat(entry_time_str) if entry_time_str else datetime.now(timezone.utc)
                except (ValueError, TypeError):
                    entry_time = datetime.now(timezone.utc)

                trade_logger = TradeLogger()
                trade_logger.append_trade(TradeRecord(
                    timestamp_entry=entry_time,
                    timestamp_exit=datetime.now(timezone.utc),
                    direction=direction,
                    entry_price=float(state.get("entry_price", 0.0)),
                    exit_price=0.0,
                    tp_price=float(state.get("tp_price", 0.0)),
                    sl_price=float(state.get("sl_price", 0.0)),
                    gap_size=float(state.get("gap_size", 0.0)),
                    pnl_usd=0.0,
                    exit_reason="MANUAL",
                    h1_sweep_bars_ago=int(state.get("h1_sweep_bars_ago", 0)),
                    m15_confirmed=bool(state.get("m15_confirmed", False)),
                    kill_zone_active=bool(state.get("kill_zone_active", False)),
                    vol_regime_pct=float(state.get("vol_regime_pct", 0.0)),
                    contracts=int(state.get("contracts", 5)),
                ))
            except Exception as exc:
                print(f"WARNING: close_position_at_market failed: {exc}", file=sys.stderr)
                exit_code = 1

    rm = RiskManager()
    rm.halt_manually()

    print("EMERGENCY STOP COMPLETE")
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description="Emergency stop for Tier2StreamingTrader")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Persist halt even if API calls fail (exit code 1 on failure)",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_run_emergency_stop(force=args.force)))


if __name__ == "__main__":
    main()
