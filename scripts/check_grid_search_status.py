#!/usr/bin/env python3
"""Check grid search status and report results."""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def check_process():
    """Check if grid search is still running."""
    try:
        result = subprocess.run(
            ["ps", "-p", "552041"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except:
        return False

def get_log_tail():
    """Get last 50 lines of log."""
    log_file = Path("/tmp/grid_search_full.log")
    if not log_file.exists():
        return ["Log file not found"]

    with open(log_file, 'r') as f:
        lines = f.readlines()

    return lines[-50:]

def main():
    print("=" * 80)
    print("GRID SEARCH STATUS CHECK")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check process
    running = check_process()

    if running:
        print("✅ Status: RUNNING")
        try:
            result = subprocess.run(
                ["ps", "-p", "552041", "-o", "etime,%cpu,%mem"],
                capture_output=True,
                text=True
            )
            print(result.stdout)
        except:
            pass

        print("\nLatest Log:")
        print("-" * 40)
        for line in get_log_tail()[-10:]:
            print(line.strip())

        print("\n🔄 Still running... check back later")
        return 0

    else:
        print("❌ Status: COMPLETED or STOPPED")
        print("\nFinal Log:")
        print("-" * 40)

        log_lines = get_log_tail()
        in_results = False
        for line in log_lines:
            if "TOP 10" in line:
                in_results = True
            if in_results:
                print(line.strip())

        # Check for output files
        csv_path = Path("data/reports/exit_parameter_optimization_1min.csv")
        md_path = Path("data/reports/exit_parameter_optimization_1min.md")

        print("\n📊 Output Files:")
        print("-" * 40)
        if csv_path.exists():
            print(f"✅ CSV: {csv_path} ({csv_path.stat().st_size:,} bytes)")
        else:
            print(f"❌ CSV not found: {csv_path}")

        if md_path.exists():
            print(f"✅ Report: {md_path} ({md_path.stat().st_size:,} bytes)")
        else:
            print(f"❌ Report not found: {md_path}")

        print("\n✅ Grid search complete!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
