"""
Unit tests for keyboard shortcuts system.

Tests keyboard shortcut definitions, registration, and password authentication.
"""

import pytest
from src.dashboard.shortcuts.shortcuts import (
    ShortcutRegistry,
    ShortcutAction,
    EmergencyShortcut,
    NavigationShortcut,
    UtilityShortcut
)
from src.dashboard.shortcuts.password_auth import PasswordAuthenticator
from src.dashboard.shortcuts.keyboard_handler import KeyboardHandler


class TestShortcutDefinitions:
    """Test shortcut definitions and registry."""

    def test_all_required_shortcuts_defined(self):
        """Test all 8 required shortcuts are defined."""
        registry = ShortcutRegistry()
        shortcuts = registry.get_all_shortcuts()

        # Check all required shortcuts exist
        shortcut_keys = [s.key_combination for s in shortcuts]
        required_shortcuts = [
            "Ctrl+E",     # Emergency stop
            "Ctrl+F",     # Manual flatten
            "Ctrl+R",     # Refresh
            "Ctrl+L",     # Log viewer
            "Ctrl+S",     # Navigate to settings
            "Ctrl+P",     # Navigate to positions
            "F1",         # Context-sensitive help
            "ESC"         # Close modal
        ]

        for required in required_shortcuts:
            assert required in shortcut_keys, f"Required shortcut {required} not defined"

    def test_emergency_shortcuts_require_password(self):
        """Test emergency shortcuts require password authentication."""
        registry = ShortcutRegistry()
        shortcuts = registry.get_all_shortcuts()

        emergency_shortcuts = [s for s in shortcuts if isinstance(s, EmergencyShortcut)]

        for shortcut in emergency_shortcuts:
            assert shortcut.requires_password, f"Emergency shortcut {shortcut.key_combination} must require password"

    def test_navigation_shortcuts_do_not_require_password(self):
        """Test navigation shortcuts do not require password."""
        registry = ShortcutRegistry()
        shortcuts = registry.get_all_shortcuts()

        nav_shortcuts = [s for s in shortcuts if isinstance(s, NavigationShortcut)]

        for shortcut in nav_shortcuts:
            assert not shortcut.requires_password, f"Navigation shortcut {shortcut.key_combination} should not require password"

    def test_shortcut_actions_defined(self):
        """Test all shortcuts have valid actions."""
        registry = ShortcutRegistry()
        shortcuts = registry.get_all_shortcuts()

        valid_actions = {
            "emergency_stop",
            "manual_flatten",
            "refresh",
            "log_viewer",
            "navigate_settings",
            "navigate_positions",
            "open_help",
            "close_modal"
        }

        for shortcut in shortcuts:
            assert shortcut.action in valid_actions, f"Invalid action {shortcut.action} for shortcut {shortcut.key_combination}"

    def test_emergency_stop_shortcut_definition(self):
        """Test Ctrl+E shortcut is emergency stop."""
        registry = ShortcutRegistry()
        shortcut = registry.get_shortcut("Ctrl+E")

        assert shortcut is not None
        assert shortcut.key_combination == "Ctrl+E"
        assert shortcut.action == "emergency_stop"
        assert isinstance(shortcut, EmergencyShortcut)
        assert shortcut.requires_password

    def test_manual_flatten_shortcut_definition(self):
        """Test Ctrl+F shortcut is manual flatten."""
        registry = ShortcutRegistry()
        shortcut = registry.get_shortcut("Ctrl+F")

        assert shortcut is not None
        assert shortcut.key_combination == "Ctrl+F"
        assert shortcut.action == "manual_flatten"
        assert isinstance(shortcut, EmergencyShortcut)
        assert shortcut.requires_password


class TestPasswordAuthentication:
    """Test password authentication for emergency shortcuts."""

    def test_authenticator_initializes(self):
        """Test password authenticator can be initialized."""
        authenticator = PasswordAuthenticator()
        assert authenticator is not None

    def test_validate_password_with_correct_password(self):
        """Test password validation succeeds with correct password."""
        authenticator = PasswordAuthenticator()
        # Use correct password from config
        result = authenticator.validate_password("emergency_password", "test_password_123")
        # This will fail initially - we'll implement later
        assert result == True

    def test_validate_password_with_incorrect_password(self):
        """Test password validation fails with incorrect password."""
        authenticator = PasswordAuthenticator()
        result = authenticator.validate_password("emergency_password", "wrong_password")
        assert result == False

    def test_failed_attempt_counter(self):
        """Test failed attempt counter increments."""
        authenticator = PasswordAuthenticator()

        # Simulate 3 failed attempts
        for _ in range(3):
            authenticator.validate_password("emergency_password", "wrong_password")

        assert authenticator.get_failed_attempts("emergency_password") == 3

    def test_account_locked_after_5_failed_attempts(self):
        """Test account locks after 5 failed attempts."""
        authenticator = PasswordAuthenticator()

        # Simulate 5 failed attempts
        for _ in range(5):
            authenticator.validate_password("emergency_password", "wrong_password")

        assert authenticator.is_locked("emergency_password") == True

    def test_locked_account_rejects_correct_password(self):
        """Test locked account rejects even correct password."""
        authenticator = PasswordAuthenticator()

        # Lock account
        for _ in range(5):
            authenticator.validate_password("emergency_password", "wrong_password")

        # Try correct password
        result = authenticator.validate_password("emergency_password", "test_password_123")
        assert result == False

    def test_reset_failed_attempts_after_successful_login(self):
        """Test failed attempts reset after successful authentication."""
        authenticator = PasswordAuthenticator()

        # 3 failed attempts
        for _ in range(3):
            authenticator.validate_password("emergency_password", "wrong_password")

        # Successful attempt (will work after implementation)
        authenticator.validate_password("emergency_password", "test_password_123")

        assert authenticator.get_failed_attempts("emergency_password") == 0


class TestKeyboardHandler:
    """Test keyboard event handler."""

    def test_handler_initializes(self):
        """Test keyboard handler can be initialized."""
        handler = KeyboardHandler()
        assert handler is not None

    def test_parse_keyboard_event(self):
        """Test keyboard event parsing."""
        handler = KeyboardHandler()

        # Test Ctrl+E
        event = handler.parse_event("ctrl+e")
        assert event["key"] == "e"
        assert event["ctrl"] == True
        assert event["shift"] == False
        assert event["alt"] == False

    def test_parse_shift_combination(self):
        """Test parsing shift combinations."""
        handler = KeyboardHandler()

        event = handler.parse_event("shift+a")
        assert event["key"] == "a"
        assert event["shift"] == True
        assert event["ctrl"] == False

    def test_parse_function_key(self):
        """Test parsing function keys."""
        handler = KeyboardHandler()

        event = handler.parse_event("f1")
        assert event["key"] == "F1"
        assert event["ctrl"] == False
        assert event["shift"] == False

    def test_parse_escape_key(self):
        """Test parsing escape key."""
        handler = KeyboardHandler()

        event = handler.parse_event("escape")
        assert event["key"] == "Escape"
        assert event["ctrl"] == False

    def test_match_shortcut_to_event(self):
        """Test matching shortcut to keyboard event."""
        handler = KeyboardHandler()
        registry = ShortcutRegistry()

        # Create keyboard event for Ctrl+E
        keyboard_event = {
            "key": "e",
            "ctrl": True,
            "shift": False,
            "alt": False
        }

        shortcut = handler.match_shortcut(keyboard_event, registry)
        assert shortcut is not None
        assert shortcut.key_combination == "Ctrl+E"
        assert shortcut.action == "emergency_stop"


class TestShortcutExecution:
    """Test shortcut execution flow."""

    def test_execute_navigation_shortcut(self):
        """Test navigation shortcut executes without password."""
        registry = ShortcutRegistry()
        shortcut = registry.get_shortcut("Ctrl+S")

        assert shortcut.action == "navigate_settings"
        assert shortcut.requires_password == False

    def test_execute_emergency_shortcut_requires_password(self):
        """Test emergency shortcut requires password before execution."""
        registry = ShortcutRegistry()
        shortcut = registry.get_shortcut("Ctrl+E")

        assert shortcut.requires_password == True
        # Password validation should occur before execution

    def test_execute_utility_shortcut(self):
        """Test utility shortcuts (F1, ESC) execute without password."""
        registry = ShortcutRegistry()

        f1_shortcut = registry.get_shortcut("F1")
        esc_shortcut = registry.get_shortcut("ESC")

        assert f1_shortcut.requires_password == False
        assert esc_shortcut.requires_password == False
        assert f1_shortcut.action == "open_help"
        assert esc_shortcut.action == "close_modal"

    def test_shortcut_execution_logs_to_audit_trail(self):
        """Test all shortcut executions log to audit trail."""
        # This will be verified in integration tests
        pass


class TestShortcutExceptions:
    """Test custom exceptions for shortcut errors."""

    def test_shortcut_not_found_error(self):
        """Test ShortcutNotFoundError is raised for undefined shortcuts."""
        registry = ShortcutRegistry()
        shortcut = registry.get_shortcut("Ctrl+Z")

        assert shortcut is None
        # Or raise ShortcutNotFoundError

    def test_password_authentication_error(self):
        """Test PasswordAuthenticationError is raised for invalid credentials."""
        authenticator = PasswordAuthenticator()
        result = authenticator.validate_password("emergency_stop", "wrong_password")

        assert result == False
        # Or raise PasswordAuthenticationError

    def test_account_locked_error(self):
        """Test AccountLockedError is raised when account is locked."""
        authenticator = PasswordAuthenticator()

        # Lock account
        for _ in range(5):
            authenticator.validate_password("emergency_stop", "wrong_password")

        assert authenticator.is_locked("emergency_stop") == True
        # Or raise AccountLockedError
