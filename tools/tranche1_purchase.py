"""Tranche 1 purchase driver for the §A8.2 parity gate (prereg Amendment 8/9, Mode C).

Derives the 28 merged ±90-min MBO windows from `data/trades.db`, prices them with
Databento's **free** `metadata.get_cost` probe, and only then — behind an explicit
confirmation flag — downloads them.

The window list is NOT a judgement call: it is the deterministic output of the
pre-registered Mode C derivation (guide §"Mode C — targeted windows"). Do not
hand-edit it. A different range is a seal deviation requiring an amendment.

Usage
-----
    # 1. price it (free, bills nothing)
    PYTHONPATH=. .venv/bin/python tools/tranche1_purchase.py --probe

    # 2. write the manifest for the gate (free)
    PYTHONPATH=. .venv/bin/python tools/tranche1_purchase.py --manifest

    # 3. buy it -- ONLY after reading the probe total
    PYTHONPATH=. .venv/bin/python tools/tranche1_purchase.py \
        --download --confirm-spend 22.50 --max-spend 30.00

Safety
------
Every `timeseries.get_range` call **bills immediately** (credit first, then card)
and cannot be undone. Therefore:

* `--download` refuses to run without `--confirm-spend`, and refuses if the live
  probe total differs from the confirmed figure by more than `--tolerance`.
* `--max-spend` is a hard ceiling; exceeding it aborts before any billing call.
* Already-downloaded windows are skipped, so a re-run never double-bills.
* `end` is clamped to the 2026-08-28 22:30 UTC data-availability cutoff.
* Set a spend limit on the Databento account too -- Tranche 2 is a ~$1,200
  fat-finger risk.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

DATASET = "GLBX.MDP3"
SCHEMA = "mbo"
STYPE_IN = "raw_symbol"

TRADES_DB = Path("data/trades.db")
OUT_DIR = Path("data/tick/mnq_mbo_parity")
MANIFEST = OUT_DIR / "windows_manifest.json"

# Hard data-availability cutoff (guide §Mode A note). Do not set `end` later.
AVAIL_END = dt.datetime(2026, 8, 28, 22, 30, tzinfo=dt.timezone.utc)
# M6 -> U6 volume roll band. A window starting before this is M6, after it U6;
# a window *inside* it needs BOTH contracts pulled (the derivation currently
# produces none, but the check stays so a future re-derivation cannot slip).
ROLL_LO = dt.datetime(2026, 6, 12, tzinfo=dt.timezone.utc)
ROLL_HI = dt.datetime(2026, 6, 19, tzinfo=dt.timezone.utc)
# Databento-flagged degraded sessions (guide §data quality). Recorded, never
# dropped -- spine AD-13 / prereg §A9.3.
DEGRADED_DAYS = {"2026-05-24", "2026-07-30"}

PARITY_FILL_QUERY = """
    select timestamp from trades
    where exit_price != 0 and exit_reason not in ('PENDING')
      and ( trader_id = 'trader-mim-nb'
            or (trader_id = 'trader-yank' and timestamp >= '2026-06-17') )
    order by timestamp
"""


@dataclass(frozen=True)
class Window:
    index: int
    start: str
    end: str
    symbol: str
    minutes: float
    n_fills: int
    degraded_days: list[str]
    needs_both_contracts: bool

    @property
    def filename(self) -> str:
        return f"win{self.index:03d}_{self.symbol}.mbo.dbn.zst"


def derive_windows(db_path: Path) -> list[Window]:
    """The pre-registered Mode C derivation: parity fills -> ±90 min -> merge."""
    if not db_path.is_file():
        sys.exit(f"error: no trades DB at {db_path}")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        stamps = [
            dt.datetime.fromisoformat(row[0].replace("Z", "+00:00"))
            for row in con.execute(PARITY_FILL_QUERY)
        ]
    finally:
        con.close()

    merged: list[list] = []
    for moment in stamps:
        lo = moment - dt.timedelta(minutes=90)
        hi = moment + dt.timedelta(minutes=90)
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
            merged[-1][2] += 1
        else:
            merged.append([lo, hi, 1])

    windows: list[Window] = []
    for i, (lo, hi, n) in enumerate(merged):
        hi = min(hi, AVAIL_END)
        degraded = sorted(
            {
                day
                for day in (lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d"))
                if day in DEGRADED_DAYS
            }
        )
        windows.append(
            Window(
                index=i,
                start=lo.isoformat(),
                end=hi.isoformat(),
                symbol="MNQM6" if lo < ROLL_LO else "MNQU6",
                minutes=round((hi - lo).total_seconds() / 60, 1),
                n_fills=n,
                degraded_days=degraded,
                needs_both_contracts=ROLL_LO <= lo < ROLL_HI,
            )
        )
    return windows


def _client():
    try:
        import databento as db
    except ImportError:
        sys.exit("error: `databento` not installed -- .venv/bin/pip install databento")
    try:
        return db.Historical()
    except Exception as exc:  # noqa: BLE001 -- surfaced verbatim to the operator
        sys.exit(
            f"error: cannot construct Historical client ({exc}). Set DATABENTO_API_KEY."
        )


def probe(windows: list[Window]) -> float:
    """Free `metadata.get_cost` over every window. Bills nothing."""
    client = _client()
    total = 0.0
    print(f"{'#':>3} {'start (UTC)':<17} {'min':>6} {'sym':<7} {'cost':>8}")
    print("-" * 50)
    for win in windows:
        cost = client.metadata.get_cost(
            dataset=DATASET,
            symbols=[win.symbol],
            stype_in=STYPE_IN,
            schema=SCHEMA,
            start=win.start,
            end=win.end,
        )
        total += float(cost)
        stamp = win.start.replace("T", " ")[:16]
        print(
            f"{win.index:>3} {stamp:<17} {win.minutes:>6.0f} {win.symbol:<7} ${float(cost):>7.2f}"
        )
    print("-" * 50)
    print(f"TOTAL for {len(windows)} windows: ${total:.2f}")
    return total


def download(
    windows: list[Window], *, confirmed: float, tolerance: float, ceiling: float
) -> None:
    """Pull every window. EVERY CALL BILLS. Guarded by a fresh probe."""
    live = probe(windows)
    if live > ceiling:
        sys.exit(f"ABORT: live total ${live:.2f} exceeds --max-spend ${ceiling:.2f}")
    if abs(live - confirmed) > tolerance:
        sys.exit(
            f"ABORT: live total ${live:.2f} differs from --confirm-spend "
            f"${confirmed:.2f} by more than ${tolerance:.2f}. Re-probe and re-confirm."
        )

    client = _client()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nconfirmed ${live:.2f} -- downloading to {OUT_DIR}/\n")
    for win in windows:
        target = OUT_DIR / win.filename
        if target.exists():
            print(f"  [{win.index:>3}] skip (already present): {target.name}")
            continue
        print(
            f"  [{win.index:>3}] pulling {win.symbol} {win.start} .. {win.end}",
            flush=True,
        )
        data = client.timeseries.get_range(
            dataset=DATASET,
            symbols=[win.symbol],
            stype_in=STYPE_IN,
            schema=SCHEMA,
            start=win.start,
            end=win.end,
        )
        data.to_file(target)
        print(f"        -> {target} ({target.stat().st_size / 1e6:.0f} MB)")
    print("\ndone. Next: --manifest, then build the gate's --windows JSON.")


def write_manifest(windows: list[Window]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "derivation": "prereg Amendment 8/9 Mode C -- parity fills +/-90min, merged",
        "dataset": DATASET,
        "schema": SCHEMA,
        "availability_cutoff_utc": AVAIL_END.isoformat(),
        "n_windows": len(windows),
        "total_minutes": round(sum(w.minutes for w in windows), 1),
        "windows": [asdict(w) | {"filename": w.filename} for w in windows],
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST} ({len(windows)} windows)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--probe", action="store_true", help="free cost probe; bills nothing"
    )
    parser.add_argument(
        "--manifest", action="store_true", help="write the window manifest JSON"
    )
    parser.add_argument(
        "--download", action="store_true", help="BILLS. Pull every window."
    )
    parser.add_argument(
        "--confirm-spend",
        type=float,
        metavar="USD",
        help="required with --download: the probe total you accept",
    )
    parser.add_argument(
        "--max-spend",
        type=float,
        default=40.0,
        metavar="USD",
        help="hard ceiling; abort before billing if exceeded (default: 40)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2.0,
        metavar="USD",
        help="allowed drift between --confirm-spend and the live probe (default: 2)",
    )
    parser.add_argument("--trades-db", default=str(TRADES_DB), metavar="PATH")
    args = parser.parse_args(argv)

    # Validate the billing guards BEFORE touching the DB or the network, so a
    # missing --confirm-spend fails loudly at the usage layer rather than behind
    # an unrelated error.
    if not (args.probe or args.manifest or args.download):
        parser.error("pick one of --probe / --manifest / --download")
    if args.download and args.confirm_spend is None:
        parser.error("--download requires --confirm-spend (run --probe first)")

    windows = derive_windows(Path(args.trades_db))
    both = [w.index for w in windows if w.needs_both_contracts]
    degraded = [w.index for w in windows if w.degraded_days]
    print(
        f"derived {len(windows)} windows, "
        f"{sum(w.minutes for w in windows) / 60:.1f} h of tape, "
        f"{sum(w.n_fills for w in windows)} parity fills"
    )
    if degraded:
        print(f"  degraded-day windows (record, never drop -- AD-13): {degraded}")
    if both:
        print(f"  !! windows inside the M6->U6 roll band need BOTH contracts: {both}")
    print()

    if args.manifest:
        write_manifest(windows)
    if args.probe and not args.download:
        probe(windows)
    if args.download:
        download(
            windows,
            confirmed=args.confirm_spend,
            tolerance=args.tolerance,
            ceiling=args.max_spend,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
