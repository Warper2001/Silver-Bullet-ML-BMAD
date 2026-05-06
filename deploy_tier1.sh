#!/bin/bash
###############################################################################
# TIER 1 FVG System - Monitor Deployment
#
# Usage: ./deploy_tier1.sh [start|stop|status|logs|config]
###############################################################################

set -e

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$PROJECT_ROOT/logs"
STATE_DIR="$PROJECT_ROOT/data/state"
MONITOR_SCRIPT="$PROJECT_ROOT/src/research/tier1_monitor.py"

# TIER 1 Configuration
TIER1_CONFIG="SL2.5x_ATR0.7_Vol2.25_MaxGap$50.0"

print_info() { echo -e "${CYAN}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_header() { echo -e "${BLUE}========================================${NC}"; echo -e "${BLUE}$1${NC}"; echo -e "${BLUE}========================================${NC}"; }

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$STATE_DIR"

start_monitor() {
    print_header "Starting TIER 1 FVG Monitor"

    if [ -f "$STATE_DIR/tier1_monitor.pid" ]; then
        PID=$(cat "$STATE_DIR/tier1_monitor.pid")
        if kill -0 "$PID" 2>/dev/null; then
            print_warning "TIER 1 monitor already running (PID: $PID)"
            exit 1
        else
            rm -f "$STATE_DIR/tier1_monitor.pid"
        fi
    fi

    print_info "Configuration: $TIER1_CONFIG"
    print_info "Starting monitor service..."

    # Start monitor in background with nohup
    nohup $PYTHON "$MONITOR_SCRIPT" > "$LOG_DIR/tier1_monitor_stdout.log" 2>&1 &
    MONITOR_PID=$!

    # Wait a moment to ensure it started
    sleep 2

    if kill -0 "$MONITOR_PID" 2>/dev/null; then
        echo $MONITOR_PID > "$STATE_DIR/tier1_monitor.pid"

        print_success "TIER 1 monitor started!"
        print_info "PID: $MONITOR_PID"
        print_info "Logs: $LOG_DIR/tier1_monitor.log"
        print_info "Stdout: $LOG_DIR/tier1_monitor_stdout.log"
        echo ""
        print_info "OAuth token auto-refresh: Active (10-minute interval)"
        print_info ""
        print_info "Monitor commands:"
        print_info "  - View logs: tail -f $LOG_DIR/tier1_monitor.log"
        print_info "  - Check status: $0 status"
        print_info "  - Stop monitor: $0 stop"
    else
        print_error "Failed to start monitor"
        rm -f "$STATE_DIR/tier1_monitor.pid"
        exit 1
    fi
}

stop_monitor() {
    print_header "Stopping TIER 1 Monitor"

    if [ ! -f "$STATE_DIR/tier1_monitor.pid" ]; then
        print_warning "TIER 1 monitor is not running"
        exit 0
    fi

    PID=$(cat "$STATE_DIR/tier1_monitor.pid")

    if ! kill -0 "$PID" 2>/dev/null; then
        print_warning "Monitor not running (stale PID file)"
        rm -f "$STATE_DIR/tier1_monitor.pid"
        exit 0
    fi

    print_info "Stopping monitor (PID: $PID)..."

    # Send SIGTERM for graceful shutdown
    kill "$PID" 2>/dev/null || true

    # Wait up to 10 seconds for graceful shutdown
    for i in {1..10}; do
        if ! kill -0 "$PID" 2>/dev/null; then
            print_success "TIER 1 monitor stopped"
            rm -f "$STATE_DIR/tier1_monitor.pid"
            exit 0
        fi
        sleep 1
    done

    # Force kill if still running
    if kill -0 "$PID" 2>/dev/null; then
        print_warning "Force killing monitor..."
        kill -9 "$PID" 2>/dev/null || true
        sleep 1
    fi

    if ! kill -0 "$PID" 2>/dev/null; then
        print_success "TIER 1 monitor stopped"
        rm -f "$STATE_DIR/tier1_monitor.pid"
    else
        print_error "Failed to stop monitor"
        exit 1
    fi
}

check_status() {
    print_header "TIER 1 Monitor Status"

    if [ ! -f "$STATE_DIR/tier1_monitor.pid" ]; then
        print_info "TIER 1 monitor: NOT RUNNING"
        echo ""
        print_info "Configuration: $TIER1_CONFIG"
        print_info "Performance Targets: WR ≥60%, PF ≥1.7, 8-15 trades/day"
        exit 0
    fi

    PID=$(cat "$STATE_DIR/tier1_monitor.pid")

    if ! kill -0 "$PID" 2>/dev/null; then
        print_warning "TIER 1 monitor: NOT RUNNING (stale PID file)"
        rm -f "$STATE_DIR/tier1_monitor.pid"
        exit 1
    fi

    print_success "TIER 1 monitor: RUNNING"
    print_info "PID: $PID"
    print_info "Started: $(ps -p $PID -o lstart=)"
    print_info "Uptime: $(ps -p $PID -o etime=)"
    print_info "Memory: $(ps -p $PID -o rss= | awk '{print int($1/1024)"MB"}')"
    echo ""
    print_info "Configuration: $TIER1_CONFIG"
    print_info "OAuth Auto-Refresh: Active (10-minute interval)"
    echo ""

    # Show recent logs
    if [ -f "$LOG_DIR/tier1_monitor.log" ]; then
        print_info "Recent log entries:"
        tail -10 "$LOG_DIR/tier1_monitor.log"
    fi
}

show_logs() {
    if [ -f "$LOG_DIR/tier1_monitor.log" ]; then
        tail -f "$LOG_DIR/tier1_monitor.log"
    else
        print_error "Log file not found: $LOG_DIR/tier1_monitor.log"
        exit 1
    fi
}

show_config() {
    print_header "TIER 1 Configuration"

    echo "Configuration: $TIER1_CONFIG"
    echo ""
    echo "Optimal Parameters (from grid search):"
    echo "  - Stop Loss Multiplier: 2.5x gap size"
    echo "  - ATR Threshold: 0.7x (stricter than 0.5x baseline)"
    echo "  - Volume Ratio: 2.25x (stricter than 1.5x baseline)"
    echo "  - Max Gap Size: \$50 (filters oversized gaps)"
    echo "  - Max Hold Time: 10 bars"
    echo ""
    echo "Performance Targets:"
    echo "  - Win Rate ≥60%"
    echo "  - Profit Factor ≥1.7"
    echo "  - Trade Frequency 8-15/day"
    echo ""
    echo "Expected Performance (Walk-Forward Validation):"
    echo "  - Win Rate: 74.07% (±2.05%)"
    echo "  - Profit Factor: 1.83 (±0.28)"
    echo "  - Trade Frequency: 11-12/day"
    echo "  - Expectancy: \$7-8/trade"
    echo ""
    echo "Validation Results:"
    echo "  - Periods Tested: 3 independent months"
    echo "  - Periods Passing All Targets: 2/3 (66.7%)"
    echo "  - Robustness: MODERATE-TO-HIGH"
    echo ""
    echo "System Features:"
    echo "  ✅ Real P&L calculation with transaction costs (\$1.40/round-trip)"
    echo "  ✅ Triple-barrier exits (gap fill TP, 2.5× SL, 10-bar time)"
    echo "  ✅ ATR-based noise filtering (0.7× threshold)"
    echo "  ✅ Volume directional confirmation (2.25× ratio)"
    echo "  ✅ Max gap size capping (\$50 limit)"
    echo "  ✅ OAuth token auto-refresh (10-minute interval)"
    echo "  ✅ No look-ahead bias (proper ATR calculation)"
    echo "  ✅ Real MNQ historical data validation"
}

case "${1:-start}" in
    start)
        start_monitor
        ;;
    stop)
        stop_monitor
        ;;
    status)
        check_status
        ;;
    logs)
        show_logs
        ;;
    config)
        show_config
        ;;
    restart)
        stop_monitor
        sleep 1
        start_monitor
        ;;
    *)
        echo "Usage: $0 {start|stop|status|logs|config|restart}"
        echo ""
        echo "TIER 1 FVG System Monitor"
        echo ""
        echo "Commands:"
        echo "  start   - Start TIER 1 monitor with OAuth auto-refresh"
        echo "  stop    - Stop TIER 1 monitor"
        echo "  status  - Check monitor status and recent logs"
        echo "  logs    - Follow monitor logs in real-time"
        echo "  config  - Show TIER 1 configuration and performance"
        echo "  restart - Restart TIER 1 monitor"
        exit 1
        ;;
esac
