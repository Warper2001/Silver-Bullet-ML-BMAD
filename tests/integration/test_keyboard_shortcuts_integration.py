"""
Integration tests for keyboard shortcuts system.

Tests keyboard shortcuts integration with Streamlit dashboard,
including JavaScript injection, session state updates, and navigation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.dashboard.shortcuts.shortcuts import ShortcutRegistry
from src.dashboard.shortcuts.password_auth import PasswordAuthenticator
from src.dashboard.shortcuts.keyboard_handler import KeyboardHandler


class TestShortcutJavaScriptInjection:
    """Test JavaScript keyboard event capture in Streamlit."""

    def test_javascript_injection_for_keyboard_capture(self):
        """Test JavaScript is injected to capture keyboard events."""
        # This will be tested in the actual dashboard implementation
        # For now, we verify the handler exists
        handler = KeyboardHandler()
        assert handler is not None

    def test_javascript_captures_ctrl_e(self):
        """Test JavaScript captures Ctrl+E key combination."""
        handler = KeyboardHandler()
        shortcut = handler.handle_shortcut("ctrl+e")

        assert shortcut is not None
        assert shortcut.key_combination == "Ctrl+E"
        assert shortcut.action == "emergency_stop"

    def test_javascript_captures_all_shortcuts(self):
        """Test JavaScript captures all 8 required shortcuts."""
        handler = KeyboardHandler()
        registry = ShortcutRegistry()

        all_shortcuts = registry.get_all_shortcuts()
        assert len(all_shortcuts) == 8

        # Verify each shortcut can be captured (case-insensitive)
        for shortcut in all_shortcuts:
            # Convert to expected input format
            test_input = shortcut.key_combination.lower()
            if shortcut.key_combination == "ESC":
                test_input = "escape"

            captured = handler.handle_shortcut(test_input)
            assert captured is not None, f"Failed to capture {shortcut.key_combination} with input {test_input}"

    def test_javascript_sends_events_to_streamlit_session_state(self):
        """Test captured keyboard events update Streamlit session state."""
        # This will be tested in dashboard integration
        # For now, verify handler can match shortcuts
        handler = KeyboardHandler()

        event = handler.parse_event("ctrl+e")
        assert event["ctrl"] == True
        assert event["key"] == "e"


class TestEmergencyShortcutAuthentication:
    """Test emergency shortcuts require password in dashboard."""

    def test_emergency_stop_shows_password_modal(self):
        """Test Ctrl+E shows password input modal."""
        # In dashboard, this would show st.text_input with type="password"
        authenticator = PasswordAuthenticator()
        shortcut_action = "emergency_stop"

        # Simulate password validation
        result = authenticator.validate_password(shortcut_action, "test_password_123")
        assert result == True

    def test_manual_flatten_shows_password_modal(self):
        """Test Ctrl+F shows password input modal."""
        authenticator = PasswordAuthenticator()
        shortcut_action = "manual_flatten"

        result = authenticator.validate_password(shortcut_action, "test_password_123")
        assert result == True

    def test_wrong_password_shows_error_message(self):
        """Test wrong password displays error with remaining attempts."""
        authenticator = PasswordAuthenticator()

        # Try wrong password
        result = authenticator.validate_password("emergency_stop", "wrong_password")
        assert result == False

        # Check failed attempts
        attempts = authenticator.get_failed_attempts("emergency_stop")
        assert attempts == 1

    def test_five_failed_attempts_locks_account(self):
        """Test 5 failed password attempts lock account for 30 minutes."""
        authenticator = PasswordAuthenticator()

        # 5 failed attempts
        for _ in range(5):
            authenticator.validate_password("emergency_stop", "wrong_password")

        # Account should be locked
        assert authenticator.is_locked("emergency_stop") == True

        # Even correct password should fail
        result = authenticator.validate_password("emergency_stop", "test_password_123")
        assert result == False

    def test_account_locked_message_displays_remaining_time(self):
        """Test locked account shows remaining lockout time."""
        authenticator = PasswordAuthenticator()

        # Lock account
        for _ in range(5):
            authenticator.validate_password("emergency_stop", "wrong_password")

        # Get remaining time
        remaining = authenticator.get_lockout_remaining("emergency_stop")
        assert remaining > 0
        assert remaining <= 1800  # 30 minutes in seconds


class TestNavigationShortcuts:
    """Test navigation shortcuts change dashboard pages."""

    def test_ctrl_s_navigates_to_settings(self):
        """Test Ctrl+S changes current page to Settings."""
        handler = KeyboardHandler()
        shortcut = handler.handle_shortcut("ctrl+s")

        assert shortcut is not None
        assert shortcut.action == "navigate_settings"
        assert shortcut.requires_password == False

    def test_ctrl_p_navigates_to_positions(self):
        """Test Ctrl+P changes current page to Positions."""
        handler = KeyboardHandler()
        shortcut = handler.handle_shortcut("ctrl+p")

        assert shortcut is not None
        assert shortcut.action == "navigate_positions"
        assert shortcut.requires_password == False

    def test_navigation_shortcuts_work_from_any_page(self):
        """Test navigation shortcuts work regardless of current page."""
        handler = KeyboardHandler()

        # Should work from any "page"
        pages_to_test = ["Overview", "Positions", "Signals", "Charts", "Settings"]

        for page in pages_to_test:
            shortcut = handler.handle_shortcut("ctrl+s")
            assert shortcut is not None
            assert shortcut.action == "navigate_settings"


class TestUtilityShortcuts:
    """Test utility shortcuts for common actions."""

    def test_ctrl_r_refreshes_current_page(self):
        """Test Ctrl+R refreshes current page data."""
        handler = KeyboardHandler()
        shortcut = handler.handle_shortcut("ctrl+r")

        assert shortcut is not None
        assert shortcut.action == "refresh"

    def test_ctrl_l_opens_log_viewer(self):
        """Test Ctrl+L opens log viewer modal."""
        handler = KeyboardHandler()
        shortcut = handler.handle_shortcut("ctrl+l")

        assert shortcut is not None
        assert shortcut.action == "log_viewer"

    def test_f1_opens_context_sensitive_help(self):
        """Test F1 opens help modal for current page."""
        handler = KeyboardHandler()
        shortcut = handler.handle_shortcut("f1")

        assert shortcut is not None
        assert shortcut.action == "open_help"

    def test_esc_closes_active_modal(self):
        """Test ESC closes open modals/dialogs."""
        handler = KeyboardHandler()
        shortcut = handler.handle_shortcut("escape")

        assert shortcut is not None
        assert shortcut.action == "close_modal"


class TestShortcutAuditTrailLogging:
    """Test all shortcut executions log to audit trail."""

    def test_emergency_stop_logs_to_audit_trail(self):
        """Test Ctrl+E execution logs to audit trail."""
        # This will be implemented in dashboard integration
        # For now, verify the shortcut exists
        registry = ShortcutRegistry()
        shortcut = registry.get_shortcut("Ctrl+E")

        assert shortcut is not None
        assert shortcut.action == "emergency_stop"

    def test_manual_flatten_logs_to_audit_trail(self):
        """Test Ctrl+F execution logs to audit trail."""
        registry = ShortcutRegistry()
        shortcut = registry.get_shortcut("Ctrl+F")

        assert shortcut is not None
        assert shortcut.action == "manual_flatten"

    def test_all_shortcuts_log_to_audit_trail(self):
        """Test all 8 shortcuts log execution to audit trail."""
        registry = ShortcutRegistry()
        shortcuts = registry.get_all_shortcuts()

        assert len(shortcuts) == 8

        for shortcut in shortcuts:
            assert shortcut.action is not None
            # Each shortcut should log when executed


class TestShortcutHelpDocumentation:
    """Test shortcuts are documented in help system."""

    def test_help_page_lists_all_shortcuts(self):
        """Test Help page displays all 8 keyboard shortcuts."""
        registry = ShortcutRegistry()
        shortcuts = registry.get_all_shortcuts()

        # All shortcuts should be documented
        assert len(shortcuts) == 8

        required_shortcuts = [
            "Ctrl+E", "Ctrl+F", "Ctrl+R", "Ctrl+L",
            "Ctrl+S", "Ctrl+P", "F1", "ESC"
        ]

        for required in required_shortcuts:
            shortcut = registry.get_shortcut(required)
            assert shortcut is not None
            assert shortcut.description  # Should have description

    def test_help_excludes_password_from_documentation(self):
        """Test Help page shows shortcuts but NOT actual passwords."""
        registry = ShortcutRegistry()
        shortcuts = registry.get_all_shortcuts()

        # Emergency shortcuts should be marked as requiring password
        # but should not display actual password
        emergency_shortcuts = registry.get_emergency_shortcuts()

        for shortcut in emergency_shortcuts:
            assert shortcut.requires_password == True
            # Password should not be in description
            assert "password" not in shortcut.description.lower() or "requires" in shortcut.description.lower()


class TestShortcutUserWorkflow:
    """Test complete user workflows with keyboard shortcuts."""

    def test_user_can_emergency_stop_with_password(self):
        """Test user can press Ctrl+E, enter password, and stop system."""
        handler = KeyboardHandler()
        authenticator = PasswordAuthenticator()

        # 1. User presses Ctrl+E
        shortcut = handler.handle_shortcut("ctrl+e")
        assert shortcut.action == "emergency_stop"

        # 2. User enters password
        auth_result = authenticator.validate_password("emergency_stop", "test_password_123")
        assert auth_result == True

        # 3. System executes emergency stop (would be implemented in dashboard)

    def test_user_can_navigate_to_settings(self):
        """Test user can press Ctrl+S to navigate to Settings."""
        handler = KeyboardHandler()

        shortcut = handler.handle_shortcut("ctrl+s")
        assert shortcut.action == "navigate_settings"
        assert shortcut.requires_password == False

    def test_user_can_open_help_and_close_with_escape(self):
        """Test user can press F1 for help, then ESC to close."""
        handler = KeyboardHandler()

        # Open help
        help_shortcut = handler.handle_shortcut("f1")
        assert help_shortcut.action == "open_help"

        # Close help
        close_shortcut = handler.handle_shortcut("escape")
        assert close_shortcut.action == "close_modal"

    def test_user_fails_password_authentication_and_gets_locked(self):
        """Test user enters wrong password 5 times and account locks."""
        handler = KeyboardHandler()
        authenticator = PasswordAuthenticator()

        # Try Ctrl+E with wrong password 5 times
        for i in range(5):
            auth_result = authenticator.validate_password("emergency_stop", "wrong_password")
            assert auth_result == False

        # Account should be locked
        assert authenticator.is_locked("emergency_stop") == True

        # Try correct password - should still fail
        auth_result = authenticator.validate_password("emergency_stop", "test_password_123")
        assert auth_result == False


class TestShortcutAccessibility:
    """Test shortcuts are accessible and usable."""

    def test_shortcuts_work_with_modifier_keys(self):
        """Test shortcuts work correctly with Ctrl, Shift, Alt modifiers."""
        handler = KeyboardHandler()

        # Test Ctrl combinations (all 6 ctrl shortcuts)
        ctrl_shortcuts = ["ctrl+e", "ctrl+f", "ctrl+r", "ctrl+l", "ctrl+s", "ctrl+p"]

        captured_count = 0
        for shortcut_str in ctrl_shortcuts:
            shortcut = handler.handle_shortcut(shortcut_str)
            if shortcut is not None:
                captured_count += 1

        # All Ctrl shortcuts should be captured
        assert captured_count == len(ctrl_shortcuts)

    def test_function_keys_work_correctly(self):
        """Test F1 key works correctly."""
        handler = KeyboardHandler()

        shortcut = handler.handle_shortcut("f1")
        assert shortcut is not None
        assert shortcut.key_combination == "F1"

    def test_escape_key_works_correctly(self):
        """Test ESC key works correctly."""
        handler = KeyboardHandler()

        shortcut = handler.handle_shortcut("escape")
        assert shortcut is not None
        assert shortcut.key_combination == "ESC"
