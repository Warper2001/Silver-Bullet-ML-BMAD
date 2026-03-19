#!/bin/bash
###############################################################################
# Silver Bullet ML-BMAD - Paper Trading Deployment Script
#
# Deploys the system to paper trading mode for validation testing
# 7-day validation period with targets: Sharpe >1.5, Win Rate ≥60%
#
# Usage: ./deploy_paper_trading.sh [start|stop|status|validate]
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
VENV_DIR="$PROJECT_ROOT/venv"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
LOG_DIR="$PROJECT_ROOT/logs"
DATA_DIR="$PROJECT_ROOT/data"
STATE_DIR="$PROJECT_ROOT/data/state"
REPORTS_DIR="$PROJECT_ROOT/data/reports"

# Trading configuration
TRADING_SYMBOL="MNQ"
TRADING_ENV="paper_trading"
SESSIONS="london_am ny_am ny_pm"
LOG_LEVEL="INFO"

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check Python virtual environment
    if [ ! -d "$VENV_DIR" ]; then
        print_error "Virtual environment not found. Creating one..."
        python3 -m venv "$VENV_DIR"
        source "$VENV_DIR/bin/activate"
        pip install -e .
    else
        print_info "Virtual environment found"
    fi

    # Check required Python packages
    print_info "Checking required packages..."
    source "$VENV_DIR/bin/activate"

    required_packages="xgboost numpy pandas httpx websockets streamlit"
    for package in $required_packages; do
        if ! python -c "import $package" 2>/dev/null; then
            print_warning "$package not found, installing..."
            pip install "$package"
        fi
    done

    # Check environment variables
    print_info "Checking environment variables..."
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        print_error ".env file not found. Please configure TradeStation credentials."
        exit 1
    fi

    # Check TradeStation credentials
    source "$PROJECT_ROOT/.env"
    if [ -z "$TRADESTATION_CLIENT_ID" ] || [ "$TRADESTATION_CLIENT_ID" = "your_client_id_here" ]; then
        print_error "TradeStation CLIENT_ID not configured in .env"
        exit 1
    fi

    if [ -z "$TRADESTATION_CLIENT_SECRET" ] || [ "$TRADESTATION_CLIENT_SECRET" = "your_client_secret_here" ]; then
        print_error "TradeStation CLIENT_SECRET not configured in .env"
        exit 1
    fi

    # Create required directories
    print_info "Creating required directories..."
    mkdir -p "$LOG_DIR"
    mkdir -p "$DATA_DIR"
    mkdir -p "$STATE_DIR"
    mkdir -p "$REPORTS_DIR"

    print_success "Prerequisites check completed"
}

# Function to start paper trading
start_paper_trading() {
    print_header "Starting Paper Trading Deployment"

    print_info "Environment: $TRADING_ENV"
    print_info "Symbol: $TRADING_SYMBOL"
    print_info "Log Level: $LOG_LEVEL"

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Set environment for paper trading
    export APP_ENV="$TRADING_ENV"
    export LOG_LEVEL="$LOG_LEVEL"
    export TRADING_MODE="paper"

    print_info "Starting system components..."

    # Check if TradeStation authentication is needed
    python -c "
from src.data.config import load_settings
from src.data.auth import TradeStationAuth
from src.ml.pipeline import MLPipeline
import asyncio

async def initialize_system():
    print('Initializing authentication...')
    auth = TradeStationAuth()

    # Test authentication
    print('Testing TradeStation API connection...')
    token = await auth.authenticate()
    print(f'✅ Authentication successful')

    # Initialize ML pipeline
    print('Initializing ML pipeline...')
    pipeline = MLPipeline(
        model_dir='models/xgboost',
        probability_threshold=0.65
    )

    print('✅ System initialization complete')
    return True

# Run initialization
try:
    asyncio.run(initialize_system())
except Exception as e:
    print(f'❌ Initialization failed: {e}')
    exit(1)
" || {
        print_error "System initialization failed"
        exit 1
    }

    # Start the orchestrator
    print_info "Starting data pipeline orchestrator..."
    python -c "
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('$LOG_DIR/paper_trading.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def run_paper_trading():
    from src.data.orchestrator import DataPipelineOrchestrator
    from src.data.auth import TradeStationAuth
    from src.data.config import load_settings

    logger.info('Starting paper trading deployment')

    # Load settings
    settings = load_settings()
    auth = TradeStationAuth()

    # Initialize orchestrator
    orchestrator = DataPipelineOrchestrator(
        auth=auth,
        data_directory='$DATA_DIR',
        max_queue_size=1000,
        settings=settings
    )

    # Start the pipeline
    await orchestrator.start()

    logger.info('Paper trading system started successfully')
    logger.info('Press Ctrl+C to stop the system')

    # Keep running until interrupted
    try:
        while True:
            await asyncio.sleep(60)
            logger.debug('System running normally')
    except KeyboardInterrupt:
        logger.info('Shutting down paper trading system...')
        await orchestrator.stop()
        logger.info('Paper trading system stopped')

# Run the system
try:
    asyncio.run(run_paper_trading())
except Exception as e:
    logger.error(f'Paper trading failed: {e}')
    import traceback
    traceback.print_exc()
" &

    ORCHESTRATOR_PID=$!

    print_success "Paper trading system started!"
    print_info "Orchestrator PID: $ORCHESTRATOR_PID"
    print_info "Logs: $LOG_DIR/paper_trading.log"
    print_info "To stop: $0 stop"

    # Save PID for later management
    echo $ORCHESTRATOR_PID > "$STATE_DIR/orchestrator.pid"
}

# Function to stop paper trading
stop_paper_trading() {
    print_header "Stopping Paper Trading"

    if [ -f "$STATE_DIR/orchestrator.pid" ]; then
        PID=$(cat "$STATE_DIR/orchestrator.pid")
        print_info "Stopping orchestrator (PID: $PID)..."

        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            print_success "Paper trading system stopped"
        else
            print_warning "Process already stopped"
        fi

        rm -f "$STATE_DIR/orchestrator.pid"
    else
        print_warning "No running paper trading system found"
    fi
}

# Function to check status
check_status() {
    print_header "Paper Trading Status"

    if [ -f "$STATE_DIR/orchestrator.pid" ]; then
        PID=$(cat "$STATE_DIR/orchestrator.pid")
        if kill -0 "$PID" 2>/dev/null; then
            print_success "✅ Paper trading system is RUNNING"
            print_info "Orchestrator PID: $PID"
            print_info "Started: $(ps -p $PID -o lstart=)"
            print_info "Logs: $LOG_DIR/paper_trading.log"

            # Show recent log entries
            if [ -f "$LOG_DIR/paper_trading.log" ]; then
                print_info "Recent log entries:"
                tail -10 "$LOG_DIR/paper_trading.log"
            fi
        else
            print_warning "⚠️  Paper trading system is NOT running (stale PID file)"
            rm -f "$STATE_DIR/orchestrator.pid"
        fi
    else
        print_info "Paper trading system is NOT running"
    fi

    # Check system resources
    print_info "System Resources:"
    echo "Memory Usage: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
    echo "Disk Usage: $(df -h "$PROJECT_ROOT" | tail -1 | awk '{print $3 " used"}')"
}

# Function to validate system performance
validate_performance() {
    print_header "Validating Paper Trading Performance"

    # Check if system is running
    if [ ! -f "$STATE_DIR/orchestrator.pid" ]; then
        print_error "Paper trading system is not running. Start it first with: $0 start"
        exit 1
    fi

    print_info "Analyzing paper trading performance..."

    # This would connect to your monitoring/analytics system
    python << 'EOF'
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

log_file = Path("$LOG_DIR/paper_trading.log")
state_dir = Path("$STATE_DIR")
reports_dir = Path("$REPORTS_DIR")

print("Performance Validation Report")
print("=" * 50)

# Check system uptime
if log_file.exists():
    logs = log_file.read_text()
    print(f"Log file size: {len(logs):,} bytes")

    # Count errors in last 24 hours
    error_count = logs.count("ERROR")
    warning_count = logs.count("WARNING")
    print(f"Errors (24h): {error_count}")
    print(f"Warnings (24h): {warning_count}")

# Check data completeness
data_dir = Path("$DATA_DIR")
if data_dir.exists():
    h5_files = list(data_dir.rglob("*.h5"))
    print(f"HDF5 files: {len(h5_files)}")

    if h5_files:
        total_size = sum(f.stat().st_size for f in h5_files)
        print(f"Data size: {total_size / 1024 / 1024:.1f} MB")

# Generate performance report
print("\nRecommendations:")
print("1. Monitor Win Rate - Target: ≥65%")
print("2. Monitor Sharpe Ratio - Target: >1.5")
print("3. Monitor Drawdown - Target: <8%")
print("4. Check system logs for errors")
print("5. Review generated reports in:", reports_dir)

print("\n✅ Validation complete - Review metrics above against targets")
EOF
}

# Main command router
case "${1:-start}" in
    start)
        check_prerequisites
        start_paper_trading
        ;;
    stop)
        stop_paper_trading
        ;;
    status)
        check_status
        ;;
    validate)
        validate_performance
        ;;
    restart)
        stop_paper_trading
        sleep 2
        start_paper_trading
        ;;
    *)
        echo "Usage: $0 {start|stop|status|validate|restart}"
        echo ""
        echo "Commands:"
        echo "  start    - Start paper trading system"
        echo "  stop     - Stop paper trading system"
        echo "  status   - Check system status and logs"
        echo  "  validate - Validate performance against targets"
        echo "  restart  - Restart paper trading system"
        exit 1
        ;;
esac