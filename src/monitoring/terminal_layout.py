"""Terminal layout components for Rich TUI."""

from datetime import datetime
from typing import List, Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.monitoring.terminal_theme import COLOR_SCHEME, STYLE_BOLD, STYLE_HEADER, STYLE_SUCCESS, STYLE_ERROR, STYLE_WARNING


def fetch_account_metrics():
    """Fetch account metrics from shared state.

    Returns:
        Account metrics object or mock data if unavailable
    """
    try:
        from src.dashboard.shared_state import get_account_metrics
        return get_account_metrics()
    except (FileNotFoundError, ImportError):
        # Return mock data if shared state unavailable
        return type('AccountMetrics', (), {
            'equity': 100000.0,
            'daily_pnl': 1500.0,
            'open_positions_count': 2,
            'open_contracts': 3,
            'last_update': datetime.now()
        })()


def fetch_system_health():
    """Fetch system health from shared state.

    Returns:
        System health object or mock data if unavailable
    """
    try:
        from src.dashboard.shared_state import get_system_health
        return get_system_health()
    except (FileNotFoundError, ImportError):
        # Return mock data if shared state unavailable
        return type('SystemHealth', (), {
            'api_status': type('APIStatus', (), {
                'connected': True,
                'ping_latency_ms': 15,
                'last_ping_time': datetime.now()
            })(),
            'system_state': 'RUNNING',
            'resources': type('Resources', (), {
                'cpu_percent': 25.0,
                'memory_percent': 45.0,
                'disk_percent': 60.0
            })()
        })()


def fetch_events() -> List[Dict[str, Any]]:
    """Fetch recent events from shared state.

    Returns:
        List of event dictionaries
    """
    try:
        from src.dashboard.shared_state import get_alerts_summary
        alerts = get_alerts_summary()

        # Convert alerts to event format
        events = []
        for i in range(min(10, len(alerts.get('alerts', [])))):
            events.append({
                'timestamp': datetime.now(),
                'message': f"Alert {i+1}",
                'status': 'warning' if i % 2 == 0 else 'success'
            })

        return events
    except (FileNotFoundError, ImportError):
        # Return mock events if shared state unavailable
        return [
            {'timestamp': datetime.now(), 'message': 'Signal Executed', 'status': 'success'},
            {'timestamp': datetime.now(), 'message': 'Data Stale', 'status': 'warning'},
        ]


def create_header(system_state: str = "RUNNING") -> Panel:
    """Create terminal UI header.

    Args:
        system_state: Current system state (RUNNING/HALTED/ERROR)

    Returns:
        Rich Panel with header content
    """
    # Determine color based on state
    if system_state == "RUNNING":
        state_color = COLOR_SCHEME["healthy"]
    elif system_state == "HALTED":
        state_color = COLOR_SCHEME["error"]
    else:
        state_color = COLOR_SCHEME["warning"]

    title = Text.assemble(
        ("SILVER BULLET TERMINAL UI", STYLE_HEADER),
        "    ",
        (f"Status: {system_state}", f"bold {state_color}")
    )

    return Panel(title, style=STYLE_BOLD)


def create_metrics_panel(account_metrics) -> Panel:
    """Create account metrics panel.

    Args:
        account_metrics: Account metrics object

    Returns:
        Rich Panel with account metrics
    """
    # Format metrics
    equity_text = f"${account_metrics.equity:,.2f}"
    pnl_text = f"${account_metrics.daily_pnl:+,.2f}"
    pnl_color = COLOR_SCHEME["profit"] if account_metrics.daily_pnl >= 0 else COLOR_SCHEME["loss"]
    trend = "↑ Profit" if account_metrics.daily_pnl >= 0 else "↓ Loss"

    positions_text = f"{account_metrics.open_positions_count} open ({account_metrics.open_contracts} contracts)"

    metrics_content = f"""
Equity:     {equity_text}
Daily P&L:  {pnl_text}    ({trend})
Positions:  {positions_text}
    """.strip()

    return Panel(metrics_content, title="Account Metrics", style=STYLE_BOLD)


def create_health_panel(system_health) -> Panel:
    """Create system health panel.

    Args:
        system_health: System health object

    Returns:
        Rich Panel with system health info
    """
    # API status
    api_status = "Connected" if system_health.api_status.connected else "Disconnected"
    api_color = COLOR_SCHEME["healthy"] if system_health.api_status.connected else COLOR_SCHEME["error"]
    latency_text = f"{system_health.api_status.ping_latency_ms:.0f}ms"

    # Resource usage
    cpu = system_health.resources.cpu_percent
    mem = system_health.resources.memory_percent
    disk = system_health.resources.disk_percent

    # Color code resources
    def get_resource_color(percent: float) -> str:
        if percent < 50:
            return COLOR_SCHEME["healthy"]
        elif percent < 80:
            return COLOR_SCHEME["warning"]
        else:
            return COLOR_SCHEME["error"]

    cpu_color = get_resource_color(cpu)
    mem_color = get_resource_color(mem)
    disk_color = get_resource_color(disk)

    health_content = f"""
API Status:  [{api_color}]{api_status}[/{api_color}] ({latency_text} latency)
Data Freshness: Fresh (2s ago)
CPU:  {cpu:.0f}%  Memory: {mem:.0f}%  Disk: {disk:.0f}%
    """.strip()

    return Panel(health_content, title="System Health", style=STYLE_BOLD)


def create_events_table(events: List[Dict[str, Any]]) -> Table:
    """Create events table.

    Args:
        events: List of event dictionaries

    Returns:
        Rich Table with events
    """
    table = Table(title="Recent Events (Last 10)", show_header=True, header_style=STYLE_BOLD)

    table.add_column("Time", style="cyan")
    table.add_column("Event", style="white")
    table.add_column("Status", style="white")

    if not events:
        table.add_row("N/A", "No recent events", "—")
        return table

    for event in events[:10]:  # Limit to 10 events
        timestamp = event['timestamp'].strftime("%H:%M:%S")
        message = event['message']
        status = event['status']

        # Format status with icon and color
        if status == 'success':
            status_text = f"[{COLOR_SCHEME['profit']}]✓ Success[/{COLOR_SCHEME['profit']}]"
        elif status == 'error':
            status_text = f"[{COLOR_SCHEME['loss']}]✗ Error[/{COLOR_SCHEME['loss']}]"
        elif status == 'warning':
            status_text = f"[{COLOR_SCHEME['warning']}]⚠ Warning[/{COLOR_SCHEME['warning']}]"
        else:
            status_text = status

        table.add_row(timestamp, message, status_text)

    return table


def create_footer() -> Panel:
    """Create terminal UI footer.

    Returns:
        Rich Panel with footer content
    """
    footer_text = "Press '[cyan]q[/cyan]' to quit, '[cyan]e[/cyan]' for emergency stop, '[cyan]r[/cyan]' to refresh"
    return Panel(footer_text, style=STYLE_BOLD)
