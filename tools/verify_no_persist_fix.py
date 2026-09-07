"""Verify the TradeLogger(persist=False) fix: a backtest replay must write
neither a trades.db row nor a logs/tier2_trade_log.csv append.

Checks both real target paths before and after a short replay, and asserts
they are byte-for-byte unchanged. Run unbuffered (-u) so progress is visible.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

WT = "/root/Silver-Bullet-ML-BMAD/.claude/worktrees/post-r3-options-research"
sys.path.insert(0, WT)

MAIN_CSV = Path("/root/Silver-Bullet-ML-BMAD/logs/tier2_trade_log.csv")
WT_CSV = Path(WT) / "logs/tier2_trade_log.csv"
CWD_DB = Path.cwd() / "data/trades.db"
MAIN_DB = Path("/root/Silver-Bullet-ML-BMAD/data/trades.db")


def snap(p: Path) -> tuple:
    if not p.exists():
        return ("absent", None, None)
    b = p.read_bytes()
    return ("present", len(b), hashlib.sha256(b).hexdigest()[:16])


def main() -> None:
    targets = {
        "main logs/tier2_trade_log.csv": MAIN_CSV,
        "worktree logs/tier2_trade_log.csv": WT_CSV,
        "CWD data/trades.db": CWD_DB,
        "main data/trades.db": MAIN_DB,
    }
    before = {k: snap(p) for k, p in targets.items()}
    print("BEFORE:")
    for k, v in before.items():
        print(f"  {k}: {v}")

    import backtest_tier2_1year_validation as bt

    print(f"\nbacktest module: {bt.__file__}")
    import src.research.tier2_streaming_working as t2

    print(f"trader module  : {t2.__file__}")
    print(f"TradeLogger has persist flag: {'persist' in t2.TradeLogger.__init__.__code__.co_varnames}")

    path = "/root/Silver-Bullet-ML-BMAD/" + str(bt.CSV_2025)
    bars = bt.load_bars(
        path,
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2025, 3, 1, tzinfo=timezone.utc),
    )
    print(f"\nloaded {len(bars)} bars; running replay...", flush=True)
    t0 = time.monotonic()
    trades = asyncio.run(
        bt.run_backtest(bars, ml_threshold=0.50, config_overrides={"max_hold_bars": 60})
    )
    print(
        f"replay done: {len(trades)} trades, "
        f"PF={bt.profit_factor([t.pnl for t in trades]):.3f}, {time.monotonic()-t0:.0f}s"
    )

    after = {k: snap(p) for k, p in targets.items()}
    print("\nAFTER:")
    for k, v in after.items():
        print(f"  {k}: {v}")

    print("\nRESULT:")
    ok = True
    for k in targets:
        same = before[k] == after[k]
        ok &= same
        print(f"  [{'UNCHANGED' if same else 'CHANGED!!'}] {k}")
    print(f"\nVERDICT: {'PASS -- replay wrote nothing' if ok else 'FAIL -- a real path was written'}")
    print(f"(replay still produced {len(trades)} trades, so the fix did not suppress the run itself)")


if __name__ == "__main__":
    main()
