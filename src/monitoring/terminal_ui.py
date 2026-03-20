"""Main Rich Terminal UI application."""

from rich.console import Console
from rich.live import Live
from rich.layout import Layout

from src.monitoring.terminal_layout import (
    fetch_account_metrics,
    fetch_system_health,
    fetch_events,
    create_header,
    create_metrics_panel,
    create_health_panel,
    create_events_table,
    create_footer,
)
from src.monitoring.terminal_events import should_quit, should_refresh, should_emergency_stop

# Auto-refresh settings
REFRESH_INTERVAL = 2.0  # 2 seconds
AUTO_REFRESH = True


def create_layout() -> Layout:
    """Create terminal UI layout.

    Returns:
        Rich Layout with all components
    """
    layout = Layout()

    # Split into sections
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="metrics", size=7),
        Layout(name="health", size=7),
        Layout(name="events", size=15),
        Layout(name="footer", size=3),
    )

    return layout


def update_layout(layout: Layout) -> None:
    """Update layout with current data.

    Args:
        layout: Rich Layout to update
    """
    # Fetch data
    account_metrics = fetch_account_metrics()
    system_health = fetch_system_health()
    events = fetch_events()

    # Update components
    system_state = getattr(system_health, 'system_state', 'RUNNING')

    layout["header"].update(create_header(system_state))
    layout["metrics"].update(create_metrics_panel(account_metrics))
    layout["health"].update(create_health_panel(system_health))
    layout["events"].update(create_events_table(events))
    layout["footer"].update(create_footer())


def run_terminal_ui():
    """Run the terminal UI application."""
    console = Console()

    # Print startup message
    console.print("[bold cyan]Starting Silver Bullet Terminal UI...[/bold cyan]")
    console.print("[dim]Press 'q' to quit, 'e' for emergency stop, 'r' to refresh[/dim]")
    console.print()

    # Create initial layout
    layout = create_layout()
    update_layout(layout)

    # Run live display
    try:
        with Live(layout, refresh_per_second=1/REFRESH_INTERVAL, console=console) as live:
            while True:
                # Update layout with current data
                update_layout(live.renderable)

                # Check for keyboard input (simplified - in real implementation would use proper input handling)
                # For now, just run continuously until Ctrl+C
                import time
                time.sleep(REFRESH_INTERVAL)

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Terminal UI stopped.[/bold yellow]")


def main():
    """Main entry point for terminal UI."""
    run_terminal_ui()


if __name__ == "__main__":
    main()
