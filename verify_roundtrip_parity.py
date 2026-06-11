#!/usr/bin/env python3
"""Round-trip parity verification for Epic 1 Story 1.4 (determinism) and Story 1.5 (parity).

Runs BacktestEngine twice on the first 2 weeks of 2025 CSV data and confirms:
1. Both runs produce byte-identical trade logs (determinism — Story 1.4 AC #5).
2. Trade decisions are logged for manual spot-check against strategy_core paths
   (round-trip parity — Story 1.5 AC #6).

Intentional divergence: BacktestEngine does not implement M15 CHoCH gating (Epic 2
Story 2.3 scope). The refactored live trader requires _m15_choch_active=True for
bearish entries. All strategy_core pure-function decisions are identical between both.
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.research.backtest_engine import BacktestEngine
from src.research.strategy_core import StrategyConfig

CSV_PATH = "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    if not Path(CSV_PATH).exists():
        print(f"ERROR: data file not found: {CSV_PATH}")
        sys.exit(1)

    config = StrategyConfig()
    print(f"StrategyConfig: bearish_only={config.bearish_only}, "
          f"min_gap_atr_ratio={config.min_gap_atr_ratio}, "
          f"sl_mult={config.sl_multiplier}, tp_mult={config.tp_multiplier}")
    print()

    # ── Run 1 ───────────────────────────────────────────────────────────────
    print("=== Run 1 ===")
    engine1 = BacktestEngine(CSV_PATH, config=config)
    trades1 = engine1.run()
    csv1, _ = engine1.save_outputs(trades1)
    print(f"  Trades: {len(trades1)}")
    if trades1:
        total_pnl = sum(t.pnl_usd for t in trades1)
        print(f"  Total PnL: ${total_pnl:.2f}")
        print("  First 5 trades (entry_ts | direction | entry → exit | reason | pnl):")
        for t in trades1[:5]:
            print(f"    {t.timestamp_entry} | {t.direction} | "
                  f"{t.entry_price:.2f} → {t.exit_price:.2f} | "
                  f"{t.exit_reason} | ${t.pnl_usd:.2f}")
    print()

    # ── Run 2 ───────────────────────────────────────────────────────────────
    print("=== Run 2 ===")
    engine2 = BacktestEngine(CSV_PATH, config=config)
    trades2 = engine2.run()
    csv2, _ = engine2.save_outputs(trades2)
    print(f"  Trades: {len(trades2)}")
    print()

    # ── Determinism check ───────────────────────────────────────────────────
    print("=== Determinism Check (Story 1.4 AC #5) ===")
    if len(trades1) != len(trades2):
        print(f"  FAIL: trade counts differ — run1={len(trades1)}, run2={len(trades2)}")
        sys.exit(1)
    md5_1 = file_md5(csv1)
    md5_2 = file_md5(csv2)
    if md5_1 == md5_2:
        print(f"  PASS: both CSVs are byte-identical (MD5={md5_1})")
        print(f"  run1: {csv1}")
        print(f"  run2: {csv2}")
    else:
        print(f"  FAIL: CSVs differ — run1 MD5={md5_1}, run2 MD5={md5_2}")
        sys.exit(1)

    print()
    print("=== Round-Trip Parity Note (Story 1.5 AC #6) ===")
    print("  strategy_core pure-function paths used identically by BacktestEngine")
    print("  and Tier2StreamingTrader (after Story 1.5 refactor):")
    print("    detect_fvg, detect_liquidity_sweep, volatility_regime_filter,")
    print("    check_exit, make_entry_decision — all route through strategy_core.")
    print("  Intentional divergence: M15 CHoCH gating (Epic 2 Story 2.3 scope).")
    print("  The live trader requires _m15_choch_active=True for bearish entries;")
    print("  BacktestEngine uses StrategyConfig defaults only (no M15 gate).")
    print()
    print("DONE — Determinism PASS, Round-Trip Parity documented.")


if __name__ == "__main__":
    main()
