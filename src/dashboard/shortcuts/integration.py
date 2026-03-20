"""
Keyboard shortcuts integration with Streamlit dashboard.

Handles JavaScript keyboard event capture, password modals for emergency shortcuts,
and execution of shortcut actions.
"""

import streamlit as st
import logging
from typing import Optional

from dashboard.shortcuts.shortcuts import ShortcutRegistry, Shortcut
from dashboard.shortcuts.password_auth import PasswordAuthenticator, AccountLockedError
from dashboard.shortcuts.keyboard_handler import KeyboardHandler


logger = logging.getLogger(__name__)


# Initialize shortcuts components
_registry = None
_authenticator = None
_handler = None


def get_shortcut_registry() -> ShortcutRegistry:
    """Get or create shortcut registry singleton."""
    global _registry
    if _registry is None:
        _registry = ShortcutRegistry()
    return _registry


def get_password_authenticator() -> PasswordAuthenticator:
    """Get or create password authenticator singleton."""
    global _authenticator
    if _authenticator is None:
        _authenticator = PasswordAuthenticator()
    return _authenticator


def get_keyboard_handler() -> KeyboardHandler:
    """Get or create keyboard handler singleton."""
    global _handler
    if _handler is None:
        _handler = KeyboardHandler(get_shortcut_registry())
    return _handler


def inject_keyboard_capture_javascript():
    """Inject JavaScript for global keyboard event capture.

    This must be called once at the top of the Streamlit app to enable
    keyboard shortcut detection.
    """
    javascript_code = """
    <script>
    (function() {
        // Track last key press time to prevent duplicate events
        let lastKeyPressTime = 0;
        const KEY_PRESS_DEBOUNCE_MS = 300;

        // Listen for keyboard events
        document.addEventListener('keydown', function(event) {
            const now = Date.now();

            // Debounce: ignore events within 300ms of last event
            if (now - lastKeyPressTime < KEY_PRESS_DEBOUNCE_MS) {
                return;
            }
            lastKeyPressTime = now;

            // Build key combination string
            const parts = [];

            if (event.ctrlKey) {
                parts.push('ctrl');
            }
            if (event.shiftKey) {
                parts.push('shift');
            }
            if (event.altKey) {
                parts.push('alt');
            }

            // Handle special keys
            let key = event.key.toLowerCase();

            if (key === 'escape') {
                parts.push('escape');
            } else if (key.startsWith('f') && key.length > 1 && key.slice(1).match(/^\\d+$/)) {
                // Function keys (F1-F12)
                parts.push(key.toLowerCase());
            } else if (parts.length > 0) {
                // Modifier + key combinations
                parts.push(key);
            } else {
                // Standalone keys
                parts.push(key);
            }

            const keyCombination = parts.join('+');

            // Send to Streamlit via hidden input
            const hiddenInput = document.getElementById('keyboard-shortcut-input');
            if (hiddenInput) {
                hiddenInput.value = keyCombination;

                // Trigger change event
                const changeEvent = new Event('change', { bubbles: true });
                hiddenInput.dispatchEvent(changeEvent);
            }

            console.log('Keyboard shortcut detected:', keyCombination);
        }, true); // Use capture phase to catch events before Streamlit
    })();
    </script>

    <!-- Hidden input to store keyboard shortcut events -->
    <input type="hidden" id="keyboard-shortcut-input" value="">
    """

    st.components.v1.html(javascript_code, height=0)


def capture_keyboard_event() -> Optional[str]:
    """Capture keyboard event from JavaScript.

    Returns:
        Key combination string (e.g., "ctrl+e", "f1", "escape") or None
    """
    # This is a placeholder - actual implementation would use
    # Streamlit's session state to communicate with JavaScript
    # For now, we'll use a different approach with session state polling

    if "last_keyboard_event" in st.session_state:
        event = st.session_state["last_keyboard_event"]
        # Clear after reading
        st.session_state["last_keyboard_event"] = None
        return event

    return None


def handle_emergency_shortcut_password(shortcut: Shortcut) -> bool:
    """Show password modal for emergency shortcut and validate.

    Args:
        shortcut: The emergency shortcut requiring password

    Returns:
        True if password validated, False otherwise
    """
    authenticator = get_password_authenticator()

    # Check if account is locked
    action = shortcut.action
    if authenticator.is_locked(action):
        remaining = authenticator.get_lockout_remaining(action)
        minutes = remaining // 60
        seconds = remaining % 60
        st.error(f"🔒 Account locked. Try again in {minutes}m {seconds}s.")
        return False

    # Get failed attempts count
    failed_attempts = authenticator.get_failed_attempts(action)
    attempts_remaining = 5 - failed_attempts

    # Show password modal
    st.info(f"🔐 {shortcut.description}")
    st.caption(f"Attempts remaining: {attempts_remaining}")

    password = st.text_input(
        "Enter password:",
        type="password",
        key=f"password_{shortcut.action}",
        help="Enter your emergency password to continue"
    )

    if not password:
        return False

    # Validate password
    if authenticator.validate_password(action, password):
        st.success("✅ Password accepted")
        return True
    else:
        failed_attempts = authenticator.get_failed_attempts(action)
        attempts_remaining = 5 - failed_attempts

        if attempts_remaining > 0:
            st.error(f"❌ Incorrect password. {attempts_remaining} attempts remaining.")
        else:
            remaining = authenticator.get_lockout_remaining(action)
            minutes = remaining // 60
            seconds = remaining % 60
            st.error(f"🔒 Account locked for {minutes}m {seconds}s.")

        return False


def execute_shortcut(shortcut: Shortcut):
    """Execute a keyboard shortcut action.

    Args:
        shortcut: The shortcut to execute
    """
    # Log shortcut execution
    logger.info(f"Executing shortcut: {shortcut.key_combination} - {shortcut.action}")

    # Emergency shortcuts require password
    if shortcut.requires_password:
        if not handle_emergency_shortcut_password(shortcut):
            logger.warning(f"Shortcut {shortcut.key_combination} aborted: authentication failed")
            return

    # Execute shortcut action
    action = shortcut.action

    if action == "emergency_stop":
        execute_emergency_stop()
    elif action == "manual_flatten":
        execute_manual_flatten()
    elif action == "navigate_settings":
        execute_navigate_settings()
    elif action == "navigate_positions":
        execute_navigate_positions()
    elif action == "refresh":
        execute_refresh()
    elif action == "log_viewer":
        execute_log_viewer()
    elif action == "open_help":
        execute_open_help()
    elif action == "close_modal":
        execute_close_modal()
    else:
        logger.warning(f"Unknown shortcut action: {action}")


def execute_emergency_stop():
    """Execute emergency stop - halt all trading activity."""
    st.session_state["emergency_stop_triggered"] = True
    st.warning("🛑 Emergency stop activated! Halting all trading activity.")
    # TODO: Call actual emergency stop function from Story 5.7


def execute_manual_flatten():
    """Execute manual flatten - close all open positions."""
    st.session_state["manual_flatten_triggered"] = True
    st.warning("🔄 Manual flatten activated! Closing all positions.")
    # TODO: Call actual manual flatten function


def execute_navigate_settings():
    """Navigate to Settings page."""
    st.session_state["page"] = "Settings"
    st.rerun()


def execute_navigate_positions():
    """Navigate to Positions page."""
    st.session_state["page"] = "Positions"
    st.rerun()


def execute_refresh():
    """Force immediate dashboard refresh."""
    st.session_state["force_refresh"] = True
    st.info("🔄 Refreshing dashboard data...")
    st.rerun()


def execute_log_viewer():
    """Open log viewer modal."""
    st.session_state["show_log_viewer"] = True
    st.session_state["page"] = "Logs"
    st.rerun()


def execute_open_help():
    """Open context-sensitive help modal."""
    current_page = st.session_state.get("page", "Overview")
    st.session_state["show_help"] = True
    st.session_state["help_page"] = current_page
    st.rerun()


def execute_close_modal():
    """Close active modal/dialog."""
    # Close help modal
    if st.session_state.get("show_help", False):
        st.session_state["show_help"] = False
        st.session_state["help_page"] = None

    # Close log viewer
    if st.session_state.get("show_log_viewer", False):
        st.session_state["show_log_viewer"] = False

    # Close password modal
    if st.session_state.get("show_password_modal", False):
        st.session_state["show_password_modal"] = False

    st.rerun()


def render_keyboard_shortcuts_ui():
    """Render keyboard shortcuts UI components.

    This should be called once at the top of render_page() to enable
    keyboard shortcut detection and handle captured events.
    """
    # Inject JavaScript for keyboard capture
    inject_keyboard_capture_javascript()

    # Check for pending keyboard events
    key_event = capture_keyboard_event()

    if key_event:
        handler = get_keyboard_handler()
        shortcut = handler.handle_shortcut(key_event)

        if shortcut:
            execute_shortcut(shortcut)
        else:
            logger.debug(f"No shortcut matched for key event: {key_event}")


def render_keyboard_shortcuts_help():
    """Render keyboard shortcuts reference in help system.

    Returns a markdown-formatted string listing all shortcuts.
    """
    registry = get_shortcut_registry()
    shortcuts = registry.get_all_shortcuts()

    help_text = "## ⌨️ Keyboard Shortcuts\n\n"

    # Group by type
    emergency = registry.get_emergency_shortcuts()
    navigation = registry.get_navigation_shortcuts()
    utility = registry.get_utility_shortcuts()

    if emergency:
        help_text += "### 🔐 Emergency Shortcuts (Password Required)\n\n"
        for shortcut in emergency:
            help_text += f"- **{shortcut.key_combination}**: {shortcut.description}\n"
        help_text += "\n"

    if navigation:
        help_text += "### 🧭 Navigation Shortcuts\n\n"
        for shortcut in navigation:
            help_text += f"- **{shortcut.key_combination}**: {shortcut.description}\n"
        help_text += "\n"

    if utility:
        help_text += "### 🛠️ Utility Shortcuts\n\n"
        for shortcut in utility:
            help_text += f"- **{shortcut.key_combination}**: {shortcut.description}\n"
        help_text += "\n"

    return help_text
