"""§A8.2 Part A coverage probe — 'N legs / M missed' against the real sources.

Replays the CLI's Part A reconstruction (mim-nb ``orders.csv`` + yank
``projectx_fills.json``, with ``stop_out_exit_ts`` populated from ProjectX for the
fired otype-4 stop-out exit), splits every trade into per-fill legs exactly as
``gate_cli`` does, and routes each leg to a ``gate_windows.json`` window with the
same ``_window_of`` predicate the gate uses. No DBN reads — this only checks that
every scored leg's stamp span lands inside exactly one purchased window.

Usage (from the worktree root, PYTHONPATH=.):
    .venv/bin/python tools/ticksim_part_a_coverage_probe.py \
        --orders-csv /root/Silver-Bullet-ML-BMAD/data/mim_nb/orders.csv \
        --projectx-fills /root/Silver-Bullet-ML-BMAD/data/mim_nb/projectx_fills.json \
        --windows /root/Silver-Bullet-ML-BMAD/_bmad-output/parity/gate_windows.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.ticksim.cli import _reconstruct_part_a_trades
from src.ticksim.parity.gate_cli import GateCliError, WindowSpec, _window_of
from src.ticksim.parity.part_a import split_legs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders-csv", required=True)
    ap.add_argument("--projectx-fills", required=True)
    ap.add_argument("--windows", required=True)
    args = ap.parse_args()

    raw = json.loads(Path(args.windows).read_text())
    windows = {
        key: WindowSpec(
            lo_ns=int(e["lo_ns"]),
            hi_ns=int(e["hi_ns"]),
            degraded_days=tuple(e.get("degraded_days", [])),
        )
        for key, e in raw.items()
    }

    trades, provenance = _reconstruct_part_a_trades(
        Path(args.orders_csv), Path(args.projectx_fills)
    )
    legs = [leg for t in trades for leg in split_legs(t)]

    missed: list[str] = []
    routed: list[tuple[str, str]] = []
    for leg in legs:
        try:
            routed.append((leg.trade_id, _window_of(leg, windows)))
        except GateCliError as exc:
            missed.append(f"{leg.trade_id}: {exc}")

    print(f"reconstructed trades : {len(trades)}")
    print(f"scored legs          : {len(legs)}")
    print(f"missed (no window)   : {len(missed)}")
    for m in missed:
        print(f"  - {m}")
    print()
    print(f"RESULT: {len(legs)} legs / {len(missed)} missed")
    print()
    if provenance:
        print(provenance)
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
