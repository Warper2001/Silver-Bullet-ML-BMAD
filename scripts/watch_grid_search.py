#!/usr/bin/env python3
"""Watch grid search progress and report results."""

import time
import subprocess
from pathlib import Path

log_file = Path("/tmp/grid_search_full.log")

print("🔍 Grid Search Monitor")
print("=" * 80)
print()

while True:
    # Check if process is running
    try:
        result = subprocess.run(
            ["ps", "-p", "552041"],
            capture_output=True,
            text=True
        )
        running = result.returncode == 0
    except:
        running = False

    if not running:
        print("❌ Process completed or stopped")
        break

    # Get latest progress
    if log_file.exists():
        with open(log_file, 'r') as f:
            lines = f.readlines()

        # Find latest progress line
        for line in reversed(lines):
            if "Progress:" in line:
                print(f"\r🔄 {line.strip()}", end='', flush=True)
                break
        else:
            print("\r⏳ Initializing feature generation...", end='', flush=True)

    time.sleep(10)  # Check every 10 seconds

print()
print("\n✅ Grid search complete!")
print("📊 Results:")
print("   CSV: data/reports/exit_parameter_optimization_1min.csv")
print("   Report: data/reports/exit_parameter_optimization_1min.md")
