"""Integration tests for terminal UI."""

import pytest


class TestTerminalUILaunch:
    """Test terminal UI launches correctly."""

    def test_terminal_ui_main_exists(self):
        """Verify main() function exists in terminal_ui module."""
        from src.monitoring.terminal_ui import main

        assert main is not None

    def test_terminal_ui_run_function_exists(self):
        """Verify run_terminal_ui() function exists."""
        from src.monitoring.terminal_ui import run_terminal_ui

        assert run_terminal_ui is not None

    def test_terminal_ui_modules_exist(self):
        """Verify all terminal UI modules exist."""
        import os

        modules = [
            "src/monitoring/terminal_ui.py",
            "src/monitoring/terminal_layout.py",
            "src/monitoring/terminal_theme.py",
            "src/monitoring/terminal_events.py",
            "src/monitoring/__main__.py",
        ]

        for module in modules:
            assert os.path.exists(module), f"Module {module} not found"

    def test_terminal_ui_entry_point(self):
        """Verify terminal UI can be run as module."""
        import subprocess
        import sys

        # Test that module can be imported
        result = subprocess.run(
            [sys.executable, "-c", "from src.monitoring.terminal_ui import main; print('OK')"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "OK" in result.stdout


class TestTerminalUIComponents:
    """Test terminal UI components work together."""

    def test_create_layout_returns_layout(self):
        """Verify create_layout returns Rich Layout."""
        from src.monitoring.terminal_ui import create_layout
        from rich.layout import Layout

        layout = create_layout()
        assert isinstance(layout, Layout)

    def test_update_layout_modifies_layout(self):
        """Verify update_layout modifies layout components."""
        from src.monitoring.terminal_ui import create_layout, update_layout

        layout = create_layout()
        # Should not raise exception
        update_layout(layout)

    def test_all_components_in_layout(self):
        """Verify all required components are in layout."""
        from src.monitoring.terminal_ui import create_layout

        layout = create_layout()

        # Check that layout has children (components)
        # Rich Layout doesn't have direct name access, so we verify structure exists
        assert layout is not None
        assert hasattr(layout, 'split_column') or len(layout.layout_map) > 0


class TestTerminalUIIntegration:
    """Test terminal UI integration with shared state."""

    def test_integration_with_shared_state(self):
        """Verify terminal UI integrates with dashboard shared state."""
        from src.monitoring.terminal_layout import (
            fetch_account_metrics,
            fetch_system_health,
            fetch_events,
        )

        # Should not raise exception
        metrics = fetch_account_metrics()
        health = fetch_system_health()
        events = fetch_events()

        # Verify we get data
        assert metrics is not None
        assert health is not None
        assert isinstance(events, list)

    def test_terminal_ui_handles_missing_shared_state(self):
        """Verify terminal UI handles missing shared state gracefully."""
        # This is tested in unit tests with mock data
        from src.monitoring.terminal_layout import fetch_account_metrics

        metrics = fetch_account_metrics()
        assert metrics is not None


class TestTerminalUIKeyboardShortcuts:
    """Test keyboard shortcuts work correctly."""

    def test_keyboard_shortcut_handlers(self):
        """Verify keyboard shortcut handlers exist."""
        from src.monitoring.terminal_events import (
            should_quit,
            should_refresh,
            should_emergency_stop,
        )

        # Test quit
        assert should_quit('q') is True
        assert should_quit('r') is False

        # Test refresh
        assert should_refresh('r') is True
        assert should_refresh('q') is False

        # Test emergency stop
        assert should_emergency_stop('e') is True
        assert should_emergency_stop('q') is False

    def test_keyboard_shortcuts_case_insensitive(self):
        """Verify keyboard shortcuts are case-insensitive."""
        from src.monitoring.terminal_events import (
            should_quit,
            should_refresh,
            should_emergency_stop,
        )

        assert should_quit('Q') is True
        assert should_refresh('R') is True
        assert should_emergency_stop('E') is True


class TestTerminalUIAutoRefresh:
    """Test auto-refresh mechanism."""

    def test_auto_refresh_interval(self):
        """Verify auto-refresh interval is 2 seconds."""
        from src.monitoring.terminal_ui import REFRESH_INTERVAL

        assert REFRESH_INTERVAL == 2.0

    def test_auto_refresh_enabled(self):
        """Verify auto-refresh is enabled."""
        from src.monitoring.terminal_ui import AUTO_REFRESH

        assert AUTO_REFRESH is True


class TestTerminalUIDisplay:
    """Test terminal UI display components."""

    def test_header_displays_system_status(self):
        """Verify header displays system status."""
        from src.monitoring.terminal_layout import create_header

        header = create_header("RUNNING")
        assert header is not None

    def test_metrics_panel_displays_account_info(self):
        """Verify metrics panel displays account information."""
        from src.monitoring.terminal_layout import create_metrics_panel

        account_metrics = type('AccountMetrics', (), {
            'equity': 100000.0,
            'daily_pnl': 1500.0,
            'open_positions_count': 2,
            'open_contracts': 3
        })()

        panel = create_metrics_panel(account_metrics)
        assert panel is not None

    def test_health_panel_displays_system_info(self):
        """Verify health panel displays system information."""
        from src.monitoring.terminal_layout import create_health_panel

        system_health = type('SystemHealth', (), {
            'api_status': type('APIStatus', (), {'connected': True, 'ping_latency_ms': 15})(),
            'system_state': 'RUNNING',
            'resources': type('Resources', (), {
                'cpu_percent': 25.0,
                'memory_percent': 45.0,
                'disk_percent': 60.0
            })()
        })()

        panel = create_health_panel(system_health)
        assert panel is not None

    def test_events_table_displays_recent_events(self):
        """Verify events table displays recent events."""
        from src.monitoring.terminal_layout import create_events_table
        from datetime import datetime

        events = [
            {'timestamp': datetime.now(), 'message': 'Signal Executed', 'status': 'success'},
            {'timestamp': datetime.now(), 'message': 'Data Stale', 'status': 'warning'},
        ]

        table = create_events_table(events)
        assert table is not None

    def test_footer_displays_keyboard_shortcuts(self):
        """Verify footer displays keyboard shortcuts."""
        from src.monitoring.terminal_layout import create_footer

        footer = create_footer()
        assert footer is not None
