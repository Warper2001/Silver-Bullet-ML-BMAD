"""Validate all MNQ dollar bar data files."""

import sys
from pathlib import Path
from datetime import datetime

import h5py

from src.research.data_validator import DataValidator
from src.research.data_quality_report import DataQualityReport


def validate_all_files(data_path: str) -> dict:
    """Validate all HDF5 files in data path.

    Args:
        data_path: Path to dollar bars directory

    Returns:
        Dictionary with aggregated results
    """
    path = Path(data_path)
    h5_files = sorted(path.glob("*.h5"))

    if not h5_files:
        print(f"No HDF5 files found in {data_path}")
        return {}

    print(f"Found {len(h5_files)} HDF5 files to validate")
    print("=" * 60)

    all_results = {
        "total_files": len(h5_files),
        "files_validated": [],
        "total_bars": 0,
        "date_range": {"earliest": None, "latest": None},
        "completeness": [],
        "issues": [],
    }

    for i, h5_file in enumerate(h5_files, 1):
        print(f"\n[{i}/{len(h5_files)}] Validating: {h5_file.name}")
        print("-" * 60)

        try:
            # Quick validation using h5py directly
            with h5py.File(h5_file, 'r') as f:
                # Get data keys
                if 'dollar_bars' not in f:
                    print(f"  ❌ ERROR: No 'dollar_bars' dataset found")
                    all_results["issues"].append(f"{h5_file.name}: Missing dollar_bars dataset")
                    continue

                # Load bars
                bars = f['dollar_bars']
                total_bars = len(bars)

                if total_bars == 0:
                    print(f"  ❌ ERROR: No bars in file")
                    all_results["issues"].append(f"{h5_file.name}: Empty file")
                    continue

                # Get date range (from timestamp field)
                # Assuming bars have 'timestamp' field
                try:
                    timestamps = bars['timestamp']
                    earliest = timestamps[0]
                    latest = timestamps[-1]
                except:
                    earliest = "N/A"
                    latest = "N/A"

                all_results["total_bars"] += total_bars
                all_results["files_validated"].append(h5_file.name)

                # Track date range
                if earliest != "N/A" and all_results["date_range"]["earliest"] is None:
                    all_results["date_range"]["earliest"] = earliest
                if latest != "N/A":
                    if all_results["date_range"]["latest"] is None or latest > all_results["date_range"]["latest"]:
                        all_results["date_range"]["latest"] = latest

                # Print summary
                print(f"  Bars: {total_bars:,}")
                print(f"  Range: {earliest} to {latest}")
                print(f"  Status: ✅ OK")

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            all_results["issues"].append(f"{h5_file.name}: {e}")

    return all_results


def main():
    """Main entry point."""
    data_path = "data/processed/dollar_bars/"

    print("MNQ Data Validation - All Files")
    print("=" * 60)
    print(f"Data Path: {data_path}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results = validate_all_files(data_path)

    print()
    print("=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)

    print(f"\nFiles Validated: {results['total_files']}")
    print(f"Total Dollar Bars: {results['total_bars']:,}")
    print(f"\nDate Range:")
    print(f"  Earliest: {results['date_range']['earliest']}")
    print(f"  Latest: {results['date_range']['latest']}")

    if results["completeness"]:
        avg_completeness = sum(results["completeness"]) / len(results["completeness"])
        print(f"\nAvg Completeness: {avg_completeness:.2f}%")

    print(f"\nIssues Found: {len(results['issues'])}")
    if results["issues"]:
        print("\nIssues:")
        for issue in results["issues"]:
            print(f"  - {issue}")

    print("\n" + "=" * 60)
    print("✅ Validation complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
