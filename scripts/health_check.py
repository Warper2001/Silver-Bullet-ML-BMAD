#!/usr/bin/env python3
"""
System Health Check Script

This script performs automated health checks on the paper trading system
to verify all components are functioning correctly.

Usage:
    python scripts/health_check.py

Output:
    Health check report with pass/fail status for each component
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import psutil
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv


class SystemHealthCheck:
    """Health check tool for paper trading system"""

    def __init__(self):
        self.checks = []
        self.warnings = []
        self.errors = []

    def log_check(self, component: str, status: str, message: str):
        """Log a health check result"""
        self.checks.append({
            "component": component,
            "status": status,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })

        emoji = "✅" if status == "PASS" else "⚠️" if status == "WARNING" else "❌"
        print(f"{emoji} {component}: {status}")
        print(f"   {message}")
        print()

    def check_process_running(self):
        """Check if paper trading process is running"""
        print("=" * 60)
        print("1. Process Status")
        print("=" * 60)

        try:
            # Look for python process running start_paper_trading.py
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and any('start_paper_trading.py' in arg for arg in cmdline):
                        self.log_check(
                            "Paper Trading Process",
                            "PASS",
                            f"Process running (PID: {proc.info['pid']}, Memory: {proc.memory_info().rss / 1024 / 1024:.1f} MB)"
                        )
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            self.log_check(
                "Paper Trading Process",
                "FAIL",
                "Paper trading process is not running"
            )
            return False

        except Exception as e:
            self.log_check(
                "Paper Trading Process",
                "ERROR",
                f"Failed to check process: {str(e)}"
            )
            return False

    def check_websocket_connection(self):
        """Check WebSocket connection status from logs"""
        print("=" * 60)
        print("2. WebSocket Connection")
        print("=" * 60)

        log_file = Path("logs/paper_trading.log")

        if not log_file.exists():
            self.log_check(
                "WebSocket Connection",
                "WARNING",
                "Log file not found - system may not have started yet"
            )
            return False

        try:
            # Check last 20 lines for WebSocket status
            result = subprocess.run(
                ['tail', '-20', str(log_file)],
                capture_output=True,
                text=True
            )

            if 'WebSocket connected' in result.stdout:
                # Check if messages are being received
                if 'messages_received' in result.stdout:
                    # Extract message count
                    for line in result.stdout.split('\n'):
                        if 'messages_received' in line:
                            self.log_check(
                                "WebSocket Connection",
                                "PASS",
                                f"Connected and receiving data: {line.strip()}"
                            )
                            return True

                self.log_check(
                    "WebSocket Connection",
                    "PASS",
                    "Connected (no recent messages - market may be closed)"
                )
                return True
            else:
                self.log_check(
                    "WebSocket Connection",
                    "WARNING",
                    "WebSocket connection status unclear - check logs manually"
                )
                return False

        except Exception as e:
            self.log_check(
                "WebSocket Connection",
                "ERROR",
                f"Failed to check WebSocket: {str(e)}"
            )
            return False

    def check_dashboard_accessible(self):
        """Check if Streamlit dashboard is accessible"""
        print("=" * 60)
        print("3. Dashboard Accessibility")
        print("=" * 60)

        try:
            # Check if streamlit process is running
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and any('streamlit' in arg for arg in cmdline):
                        self.log_check(
                            "Dashboard",
                            "PASS",
                            f"Dashboard running (PID: {proc.info['pid']}, URL: http://localhost:8501)"
                        )
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            self.log_check(
                "Dashboard",
                "WARNING",
                "Dashboard process not found"
            )
            return False

        except Exception as e:
            self.log_check(
                "Dashboard",
                "ERROR",
                f"Failed to check dashboard: {str(e)}"
            )
            return False

    def check_log_files(self):
        """Check if log files are being written"""
        print("=" * 60)
        print("4. Log Files")
        print("=" * 60)

        log_files = {
            "Main Log": "logs/paper_trading.log",
            "Daily Loss": "logs/daily_loss.csv",
            "Drawdown": "logs/drawdown.csv",
            "Per-Trade Risk": "logs/per_trade_risk.csv"
        }

        all_exist = True

        for name, path in log_files.items():
            log_path = Path(path)

            if log_path.exists():
                size = log_path.stat().st_size
                mtime = datetime.fromtimestamp(log_path.stat().st_mtime)

                # Check if recently modified (within 1 hour)
                age = (datetime.now() - mtime).total_seconds() / 60

                if age < 60:
                    self.log_check(
                        f"Log File: {name}",
                        "PASS",
                        f"Exists ({size:,} bytes, modified {age:.0f} minutes ago)"
                    )
                else:
                    self.log_check(
                        f"Log File: {name}",
                        "WARNING",
                        f"Exists but not recently updated (modified {age:.0f} minutes ago)"
                    )
                    all_exist = False
            else:
                self.log_check(
                    f"Log File: {name}",
                    "FAIL",
                    f"File not found: {path}"
                )
                all_exist = False

        return all_exist

    def check_risk_management(self):
        """Check risk management CSV files for recent activity"""
        print("=" * 60)
        print("5. Risk Management")
        print("=" * 60)

        try:
            # Check daily loss CSV
            daily_loss_file = Path("logs/daily_loss.csv")

            if daily_loss_file.exists():
                with open(daily_loss_file, 'r') as f:
                    lines = f.readlines()

                if len(lines) > 1:  # Has data beyond header
                    last_line = lines[-1].strip()
                    self.log_check(
                        "Risk Management",
                        "PASS",
                        f"Risk tracking active. Last entry: {last_line}"
                    )
                    return True
                else:
                    self.log_check(
                        "Risk Management",
                        "WARNING",
                        "Risk CSV files exist but no data yet"
                    )
                    return False
            else:
                self.log_check(
                    "Risk Management",
                    "FAIL",
                    "Risk CSV files not found"
                )
                return False

        except Exception as e:
            self.log_check(
                "Risk Management",
                "ERROR",
                f"Failed to check risk management: {str(e)}"
            )
            return False

    def check_models(self):
        """Check if ML models are present"""
        print("=" * 60)
        print("6. ML Models")
        print("=" * 60)

        model_dir = Path("models/xgboost/5_minute")

        required_files = [
            "model.joblib",
            "preprocessor.pkl",
            "metadata.json"
        ]

        all_present = True

        for filename in required_files:
            model_path = model_dir / filename

            if model_path.exists():
                size = model_path.stat().st_size
                self.log_check(
                    f"Model File: {filename}",
                    "PASS",
                    f"Present ({size:,} bytes)"
                )
            else:
                self.log_check(
                    f"Model File: {filename}",
                    "FAIL",
                    f"Not found: {model_path}"
                )
                all_present = False

        return all_present

    def check_configuration(self):
        """Check configuration file"""
        print("=" * 60)
        print("7. Configuration")
        print("=" * 60)

        config_file = Path("config.yaml")

        if config_file.exists():
            self.log_check(
                "Configuration File",
                "PASS",
                f"config.yaml exists ({config_file.stat().st_size:,} bytes)"
            )
            return True
        else:
            self.log_check(
                "Configuration File",
                "FAIL",
                "config.yaml not found"
            )
            return False

    def check_environment(self):
        """Check environment variables"""
        print("=" * 60)
        print("8. Environment Variables")
        print("=" * 60)

        load_dotenv()

        required_vars = [
            "TRADESTATION_APP_ID",
            "TRADESTATION_APP_SECRET",
            "TRADESTATION_REFRESH_TOKEN",
            "TRADESTATION_ENVIRONMENT"
        ]

        all_set = True

        for var in required_vars:
            value = os.getenv(var)
            if value:
                # Mask sensitive values
                if "SECRET" in var or "TOKEN" in var:
                    display_value = f"{value[:10]}...{value[-4:]}"
                else:
                    display_value = value
                self.log_check(
                    f"Env Var: {var}",
                    "PASS",
                    f"Set to: {display_value}"
                )
            else:
                self.log_check(
                    f"Env Var: {var}",
                    "FAIL",
                    "Not set"
                )
                all_set = False

        return all_set

    def check_disk_space(self):
        """Check disk space"""
        print("=" * 60)
        print("9. Disk Space")
        print("=" * 60)

        try:
            stat = psutil.disk_usage('.')
            used_percent = stat.percent
            free_gb = stat.free / (1024**3)

            if used_percent < 80:
                self.log_check(
                    "Disk Space",
                    "PASS",
                    f"{free_gb:.1f} GB free ({used_percent:.1f}% used)"
                )
                return True
            elif used_percent < 90:
                self.log_check(
                    "Disk Space",
                    "WARNING",
                    f"{free_gb:.1f} GB free ({used_percent:.1f}% used) - Consider cleanup"
                )
                return True
            else:
                self.log_check(
                    "Disk Space",
                    "FAIL",
                    f"{free_gb:.1f} GB free ({used_percent:.1f}% used) - Cleanup required"
                )
                return False

        except Exception as e:
            self.log_check(
                "Disk Space",
                "ERROR",
                f"Failed to check disk space: {str(e)}"
            )
            return False

    def run_all_checks(self):
        """Run all health checks"""
        print("\n" + "=" * 60)
        print("PAPER TRADING SYSTEM HEALTH CHECK")
        print("=" * 60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Run all checks
        self.check_process_running()
        self.check_websocket_connection()
        self.check_dashboard_accessible()
        self.check_log_files()
        self.check_risk_management()
        self.check_models()
        self.check_configuration()
        self.check_environment()
        self.check_disk_space()

        return self.generate_report()

    def generate_report(self):
        """Generate health check report"""
        print("\n" + "=" * 60)
        print("HEALTH CHECK REPORT")
        print("=" * 60)

        passed = sum(1 for c in self.checks if c["status"] == "PASS")
        warnings = sum(1 for c in self.checks if c["status"] == "WARNING")
        failed = sum(1 for c in self.checks if c["status"] in ["FAIL", "ERROR"])
        total = len(self.checks)

        print(f"\nTotal Checks: {total}")
        print(f"✅ Passed: {passed}")
        print(f"⚠️  Warnings: {warnings}")
        print(f"❌ Failed: {failed}")
        print(f"Success Rate: {passed/total*100:.1f}%")

        if failed == 0:
            print("\n🎉 ALL CRITICAL CHECKS PASSED!")
            print("\nThe system is healthy. You can:")
            print("  - Monitor dashboard: http://localhost:8501")
            print("  - Follow logs: tail -f logs/paper_trading.log")
            print("  - Check status: ./deploy_paper_trading.sh status")
        else:
            print("\n⚠️  SOME CHECKS FAILED")
            print("\nFailed components:")
            for check in self.checks:
                if check["status"] in ["FAIL", "ERROR"]:
                    print(f"  ❌ {check['component']}: {check['message']}")

        if warnings > 0:
            print("\nWarnings (non-critical):")
            for check in self.checks:
                if check["status"] == "WARNING":
                    print(f"  ⚠️  {check['component']}: {check['message']}")

        print("\n" + "=" * 60)
        print(f"Health check completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        return failed == 0


def main():
    """Main entry point"""
    health_check = SystemHealthCheck()
    success = health_check.run_all_checks()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
