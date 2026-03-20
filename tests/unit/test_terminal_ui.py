"""Unit tests for terminal UI components."""

import pytest
from datetime import datetime


class TestTerminalTheme:
    """Test terminal theme and color schemes."""

    def test_color_scheme_defined(self):
        """Verify color scheme is defined with green, red, yellow."""
        from src.monitoring.terminal_theme import COLOR_SCHEME

        # Check that color scheme has the required keys
        assert "profit" in COLOR_SCHEME
        assert "loss" in COLOR_SCHEME
        assert "healthy" in COLOR_SCHEME
        assert "error" in COLOR_SCHEME
        assert "warning" in COLOR_SCHEME
        assert "info" in COLOR_SCHEME

    def test_profit_color(self):
        """Verify profit uses green color."""
        from src.monitoring.terminal_theme import COLOR_SCHEME

        assert COLOR_SCHEME["profit"] == "green"
        assert COLOR_SCHEME["healthy"] == "green"

    def test_loss_color(self):
        """Verify loss uses red color."""
        from src.monitoring.terminal_theme import COLOR_SCHEME

        assert COLOR_SCHEME["loss"] == "red"
        assert COLOR_SCHEME["error"] == "red"

    def test_warning_color(self):
        """Verify warning uses yellow color."""
        from src.monitoring.terminal_theme import COLOR_SCHEME

        assert COLOR_SCHEME["warning"] == "yellow"
        assert COLOR_SCHEME["stale"] == "yellow"


class TestTerminalLayout:
    """Test terminal layout components."""

    def test_create_header_component(self):
        """Verify header component can be created."""
        from src.monitoring.terminal_layout import create_header

        header = create_header("RUNNING")
        assert header is not None

    def test_create_metrics_panel(self):
        """Verify metrics panel can be created."""
        from src.monitoring.terminal_layout import create_metrics_panel

        # Mock account metrics
        account_metrics = type('AccountMetrics', (), {
            'equity': 100000.0,
            'daily_pnl': 1500.0,
            'open_positions_count': 2,
            'open_contracts': 3
        })()

        panel = create_metrics_panel(account_metrics)
        assert panel is not None

    def test_create_health_panel(self):
        """Verify health panel can be created."""
        from src.monitoring.terminal_layout import create_health_panel

        # Mock system health
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

    def test_create_events_table(self):
        """Verify events table can be created."""
        from src.monitoring.terminal_layout import create_events_table

        # Mock events
        events = [
            {'timestamp': datetime.now(), 'message': 'Signal Executed', 'status': 'success'},
            {'timestamp': datetime.now(), 'message': 'Data Stale', 'status': 'warning'},
        ]

        table = create_events_table(events)
        assert table is not None

    def test_create_footer(self):
        """Verify footer component can be created."""
        from src.monitoring.terminal_layout import create_footer

        footer = create_footer()
        assert footer is not None


class TestDataFetching:
    """Test data fetching from shared state."""

    def test_fetch_account_metrics(self):
        """Verify fetch_account_metrics returns valid data."""
        from src.monitoring.terminal_layout import fetch_account_metrics

        metrics = fetch_account_metrics()
        assert metrics is not None

    def test_fetch_system_health(self):
        """Verify fetch_system_health returns valid data."""
        from src.monitoring.terminal_layout import fetch_system_health

        health = fetch_system_health()
        assert health is not None

    def test_fetch_events(self):
        """Verify fetch_events returns list of events."""
        from src.monitoring.terminal_layout import fetch_events

        events = fetch_events()
        assert isinstance(events, list)

    def test_missing_shared_state_handled(self):
        """Verify missing shared state is handled gracefully."""
        from src.monitoring.terminal_layout import fetch_account_metrics

        # Should return mock data if shared state unavailable
        metrics = fetch_account_metrics()
        assert metrics is not None


class TestKeyboardShortcuts:
    """Test keyboard shortcuts functionality."""

    def test_quit_shortcut_defined(self):
        """Verify 'q' shortcut is defined for quit."""
        from src.monitoring.terminal_events import KEYBOARD_SHORTCUTS

        assert 'q' in KEYBOARD_SHORTCUTS
        assert KEYBOARD_SHORTCUTS['q'] == 'quit'

    def test_refresh_shortcut_defined(self):
        """Verify 'r' shortcut is defined for refresh."""
        from src.monitoring.terminal_events import KEYBOARD_SHORTCUTS

        assert 'r' in KEYBOARD_SHORTCUTS
        assert KEYBOARD_SHORTCUTS['r'] == 'refresh'

    def test_emergency_stop_shortcut_defined(self):
        """Verify 'e' shortcut is defined for emergency stop."""
        from src.monitoring.terminal_events import KEYBOARD_SHORTCUTS

        assert 'e' in KEYBOARD_SHORTCUTS
        assert KEYBOARD_SHORTCUTS['e'] == 'emergency_stop'


class TestAutoRefresh:
    """Test auto-refresh mechanism."""

    def test_refresh_interval_defined(self):
        """Verify refresh interval is set to 2 seconds."""
        from src.monitoring.terminal_ui import REFRESH_INTERVAL

        assert REFRESH_INTERVAL == 2.0

    def test_auto_refresh_enabled(self):
        """Verify auto-refresh is enabled by default."""
        from src.monitoring.terminal_ui import AUTO_REFRESH

        assert AUTO_REFRESH is True
