#!/bin/bash
###############################################################################
# TIER 1 FVG Real-Time Paper Trading Deployment
#
# Live paper trading with real MNQ data via TradeStation REST API.
# Configuration: SL2.5x_ATR0.7_Vol2.25_MaxGap$50.0
#
# Usage: ./deploy_tier1_realtime.sh [start|stop|status|logs|config]
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
TRADING_SCRIPT="$PROJECT_ROOT/src/research/tier1_paper_rest.py"

# TIER 1 Configuration (EXACT SAME AS BACKTEST)
TIER1_CONFIG="SL2.5x_ATR0.7_Vol2.25_MaxGap$50.0"
SL_MULTIPLIER=2.5
ATR_THRESHOLD=0.7
VOLUME_RATIO_THRESHOLD=2.25
MAX_GAP_DOLLARS=50.0

print_info() { echo -e "${CYAN}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_header() { echo -e "${BLUE}========================================${NC}"; echo -e "${BLUE}$1${NC}"; echo -e "${BLUE}========================================${NC}"; }
print_tier1() { echo -e "${GREEN}[TIER 1]${NC} $1"; }

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$STATE_DIR"

start_paper_trading() {
    print_header "Starting TIER 1 Real-Time Paper Trading"

    if [ -f "$STATE_DIR/tier1_paper_trading.pid" ]; then
        PID=$(cat "$STATE_DIR/tier1_paper_trading.pid")
        if kill -0 "$PID" 2>/dev/null; then
            print_warning "Paper trading already running (PID: $PID)"
            exit 1
        else
            rm -f "$STATE_DIR/tier1_paper_trading.pid"
        fi
    fi

    print_tier1 "Configuration: $TIER1_CONFIG"
    print_tier1 "Stop Loss: ${SL_MULTIPLIER}x gap size"
    print_tier1 "ATR Threshold: ${ATR_THRESHOLD}x"
    print_tier1 "Volume Ratio: ${VOLUME_RATIO_THRESHOLD}x"
    print_tier1 "Max Gap Size: \$${MAX_GAP_DOLLARS}"

    echo ""
    print_info "Starting real-time paper trading system..."
    print_info "Connecting to TradeStation WebSocket..."

    # Start paper trading system
    nohup $PYTHON "$TRADING_SCRIPT" > "$LOG_DIR/tier1_paper_trading_stdout.log" 2>&1 &
    TRADING_PID=$!

    # Wait to ensure it started
    sleep 3

    if kill -0 "$TRADING_PID" 2>/dev/null; then
        echo $TRADING_PID > "$STATE_DIR/tier1_paper_trading.pid"

        print_success "TIER 1 paper trading started!"
        print_info "PID: $TRADING_PID"
        print_info "Logs: $LOG_DIR/tier1_paper_trading.log"
        print_info "Stdout: $LOG_DIR/tier1_paper_trading_stdout.log"
        echo ""
        print_tier1 "System Features:"
        print_tier1 "  ✅ Real-time MNQ data via TradeStation WebSocket"
        print_tier1 "  ✅ Dollar bar transformation (\$50M threshold)"
        print_tier1 "  ✅ FVG detection with TIER 1 filters"
        print_tier1 "  ✅ Realistic P&L calculation (\$1.40/round-trip)"
        print_tier1 "  ✅ Triple-barrier exits (gap fill TP, 2.5× SL, 10-bar time)"
        print_tier1 "  ✅ OAuth auto-refresh (10-minute interval)"
        echo ""
        print_info "Performance Targets:"
        print_info "  - Win Rate ≥60% (Expected: 74%)"
        print_info "  - Profit Factor ≥1.7 (Expected: 1.83)"
        print_info "  - Trade Frequency 8-15/day (Expected: 11-12)"
        echo ""
        print_info "Commands:"
        print_info "  - View logs: $0 logs"
        print_info "  - Check status: $0 status"
        print_info "  - Stop system: $0 stop"
    else
        print_error "Failed to start paper trading system"
        rm -f "$STATE_DIR/tier1_paper_trading.pid"
        exit 1
    fi
}

stop_paper_trading() {
    print_header "Stopping TIER 1 Paper Trading"

    if [ ! -f "$STATE_DIR/tier1_paper_trading.pid" ]; then
        print_warning "Paper trading system is not running"
        exit 0
    fi

    PID=$(cat "$STATE_DIR/tier1_paper_trading.pid")

    if ! kill -0 "$PID" 2>/dev/null; then
        print_warning "System not running (stale PID file)"
        rm -f "$STATE_DIR/tier1_paper_trading.pid"
        exit 0
    fi

    print_info "Stopping paper trading system (PID: $PID)..."

    # Send SIGTERM for graceful shutdown
    kill "$PID" 2>/dev/null || true

    # Wait up to 15 seconds for graceful shutdown (to generate final report)
    for i in {1..15}; do
        if ! kill -0 "$PID" 2>/dev/null; then
            print_success "Paper trading system stopped"
            rm -f "$STATE_DIR/tier1_paper_trading.pid"

            # Show final performance report
            if [ -f "$LOG_DIR/tier1_paper_trading.log" ]; then
                echo ""
                print_info "Final Performance Report:"
                grep -A 20 "FINAL PERFORMANCE REPORT" "$LOG_DIR/tier1_paper_trading.log" | tail -20
            fi

            exit 0
        fi
        sleep 1
    done

    # Force kill if still running
    if kill -0 "$PID" 2>/dev/null; then
        print_warning "Force killing system..."
        kill -9 "$PID" 2>/dev/null || true
        sleep 1
    fi

    if ! kill -0 "$PID" 2>/dev/null; then
        print_success "Paper trading system stopped"
        rm -f "$STATE_DIR/tier1_paper_trading.pid"
    else
        print_error "Failed to stop system"
        exit 1
    fi
}

check_status() {
    print_header "TIER 1 Paper Trading Status"

    if [ ! -f "$STATE_DIR/tier1_paper_trading.pid" ]; then
        print_info "Paper trading system: NOT RUNNING"
        echo ""
        print_info "Configuration: $TIER1_CONFIG"
        exit 0
    fi

    PID=$(cat "$STATE_DIR/tier1_paper_trading.pid")

    if ! kill -0 "$PID" 2>/dev/null; then
        print_warning "Paper trading system: NOT RUNNING (stale PID file)"
        rm -f "$STATE_DIR/tier1_paper_trading.pid"
        exit 1
    fi

    print_success "Paper trading system: RUNNING"
    print_info "PID: $PID"
    print_info "Started: $(ps -p $PID -o lstart=)"
    print_info "Uptime: $(ps -p $PID -o etime=)"
    print_info "Memory: $(ps -p $PID -o rss= | awk '{print int($1/1024)"MB"}')"
    print_info "CPU: $(ps -p $PID -o %cpu=)%"
    echo ""
    print_info "Configuration: $TIER1_CONFIG"
    print_info "Performance Targets: WR ≥60%, PF ≥1.7, 8-15 trades/day"
    echo ""

    # Show recent activity
    if [ -f "$LOG_DIR/tier1_paper_trading.log" ]; then
        print_info "Recent Activity:"
        echo ""
        echo "Trades:"
        grep "PAPER TRADE" "$LOG_DIR/tier1_paper_trading.log" | tail -5
        echo ""
        echo "System Status:"
        grep "SYSTEM STATUS" "$LOG_DIR/tier1_paper_trading.log" | tail -1
    fi
}

show_logs() {
    if [ -f "$LOG_DIR/tier1_paper_trading.log" ]; then
        tail -f "$LOG_DIR/tier1_paper_trading.log"
    else
        print_error "Log file not found: $LOG_DIR/tier1_paper_trading.log"
        exit 1
    fi
}

show_config() {
    print_header "TIER 1 Paper Trading Configuration"

    echo "Configuration: $TIER1_CONFIG"
    echo ""
    echo "Strategy Parameters (VALIDATED):"
    echo "  - Stop Loss Multiplier: ${SL_MULTIPLIER}x gap size"
    echo "  - ATR Threshold: ${ATR_THRESHOLD}x (stricter than 0.5x baseline)"
    echo "  - Volume Ratio: ${VOLUME_RATIO_THRESHOLD}x (stricter than 1.5x baseline)"
    echo "  - Max Gap Size: \$${MAX_GAP_DOLLARS} (filters oversized gaps)"
    echo "  - Max Hold Time: 10 bars"
    echo ""
    echo "Execution Parameters:"
    echo "  - Entry: Gap boundary (bottom for long, top for short)"
    echo "  - Take Profit: Gap fill (opposite boundary)"
    echo "  - Stop Loss: ${SL_MULTIPLIER}x gap size against position"
    echo "  - Time Exit: Max 10 bars"
    echo "  - Transaction Cost: \$1.40 per round-trip"
    echo ""
    echo "Performance Targets (from walk-forward validation):"
    echo "  - Win Rate: ≥60% (Expected: 74.07% ±2.05%)"
    echo "  - Profit Factor: ≥1.7 (Expected: 1.83 ±0.28)"
    echo "  - Trade Frequency: 8-15/day (Expected: 11-12/day)"
    echo "  - Expectancy: \$7-8/trade"
    echo ""
    echo "Data & Infrastructure:"
    echo "  ✅ Real MNQ data via TradeStation WebSocket"
    echo "  ✅ Dollar bar transformation (\$50M threshold)"
    echo "  ✅ ATR calculation (14-period EWM)"
    echo "  ✅ Volume directional ratio (20-period rolling)"
    echo "  ✅ OAuth token auto-refresh (10-minute interval)"
    echo "  ✅ Realistic P&L calculation (not fake binary)"
    echo ""
    echo "Validation Quality:"
    echo "  ✅ Real MNQ historical data (368MB, 795K bars)"
    echo "  ✅ Walk-forward validation (3 independent periods)"
    echo "  ✅ No look-ahead bias (proper ATR calculation)"
    echo "  ✅ 2/3 periods passed all targets (66.7% robustness)"
    echo "  ✅ No evidence of significant overfitting"
}

case "${1:-start}" in
    start)
        start_paper_trading
        ;;
    stop)
        stop_paper_trading
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
        stop_paper_trading
        sleep 2
        start_paper_trading
        ;;
    *)
        echo "Usage: $0 {start|stop|status|logs|config|restart}"
        echo ""
        echo "TIER 1 FVG Real-Time Paper Trading"
        echo ""
        echo "Commands:"
        echo "  start   - Start real-time paper trading with live MNQ data"
        echo "  stop    - Stop paper trading (generates final report)"
        echo "  status  - Check system status and recent activity"
        echo "  logs    - Follow paper trading logs in real-time"
        echo "  config  - Show TIER 1 configuration and validation results"
        echo "  restart - Restart paper trading system"
        echo ""
        echo "Configuration: $TIER1_CONFIG"
        echo "Performance Targets: WR ≥60%, PF ≥1.7, 8-15 trades/day"
        exit 1
        ;;
esac
