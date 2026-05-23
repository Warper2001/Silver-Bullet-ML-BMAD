#!/usr/bin/env python3
"""protect_holdout.py — OS-level write protection for data/sealed_holdout/.

Usage:
    python protect_holdout.py --init    # Apply chmod 444 to all CSVs, log to ACCESS_LOG
    python protect_holdout.py --verify  # Check all CSVs are 444; exit 0 pass, 1 fail
"""

import argparse
import os
import re
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

HOLDOUT_DIR = Path("data/sealed_holdout")
ACCESS_LOG = HOLDOUT_DIR / "ACCESS_LOG.md"
HOLDOUT_CUTOFF = "2026-03-01"


def _extract_date(csv_path: Path) -> str | None:
    """Extract YYYY-MM-DD from filename like mnq_1min_holdout_20260301_plus.csv."""
    m = re.search(r"(\d{4})(\d{2})(\d{2})", csv_path.stem)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def verify(holdout_dir: Path) -> int:
    """Return 0 if all CSVs are chmod 444 and dated >= HOLDOUT_CUTOFF, 1 otherwise."""
    csvs = sorted(holdout_dir.glob("*.csv"))
    if not csvs:
        print(f"VERIFY FAIL — no CSV files found in {holdout_dir}")
        return 1

    # Date validation first
    for csv_path in csvs:
        date_str = _extract_date(csv_path)
        if date_str and date_str < HOLDOUT_CUTOFF:
            print(f"VERIFY FAIL — {csv_path.name} predates cutoff {HOLDOUT_CUTOFF} (found {date_str})")
            return 1

    # Permission check
    offenders = []
    for csv_path in csvs:
        mode = stat.S_IMODE(os.stat(csv_path).st_mode)
        if mode != 0o444:
            offenders.append((csv_path.name, oct(mode)))

    if offenders:
        for name, mode in offenders:
            print(f"VERIFY FAIL — {name} is writable (mode {mode})")
        return 1

    print(f"VERIFY PASS — all {len(csvs)} file(s) protected (chmod 444)")
    return 0


def init(holdout_dir: Path) -> int:
    """Apply chmod 444 to all CSVs; log to ACCESS_LOG.md; return 0."""
    holdout_dir.mkdir(parents=True, exist_ok=True)
    access_log = holdout_dir / "ACCESS_LOG.md"
    csvs = sorted(holdout_dir.glob("*.csv"))
    protected = []
    already = []
    for csv_path in csvs:
        mode = stat.S_IMODE(os.stat(csv_path).st_mode)
        if mode != 0o444:
            os.chmod(csv_path, 0o444)
            protected.append(csv_path.name)
        else:
            already.append(csv_path.name)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = (
        f"\n## Init — {timestamp}\n\n"
        f"- Protected: {protected if protected else 'none (all already 444)'}\n"
        f"- Already protected: {already}\n"
    )
    with open(access_log, "a") as f:
        f.write(entry)

    print(f"INIT PASS — {len(csvs)} CSV(s) protected, ACCESS_LOG updated")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Sealed holdout protection utility")
    parser.add_argument("--init", action="store_true", help="Apply chmod 444 to all CSVs")
    parser.add_argument("--verify", action="store_true", help="Check all CSVs are 444")
    args = parser.parse_args()

    if args.init:
        sys.exit(init(HOLDOUT_DIR))
    elif args.verify:
        sys.exit(verify(HOLDOUT_DIR))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
