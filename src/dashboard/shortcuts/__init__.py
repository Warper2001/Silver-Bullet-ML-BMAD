"""
Keyboard shortcuts system for BMAD dashboard.

Provides keyboard shortcut definitions, registration, password authentication,
and event handling for emergency, navigation, and utility shortcuts.
"""

from src.dashboard.shortcuts.shortcuts import (
    Shortcut,
    EmergencyShortcut,
    NavigationShortcut,
    UtilityShortcut,
    ShortcutRegistry,
    ShortcutType,
    ShortcutAction
)
from src.dashboard.shortcuts.password_auth import (
    PasswordAuthenticator,
    SilverBulletError,
    ShortcutAuthenticationError,
    AccountLockedError
)
from src.dashboard.shortcuts.keyboard_handler import (
    KeyboardHandler,
    ShortcutNotDefinedError
)

__all__ = [
    # Shortcuts
    "Shortcut",
    "EmergencyShortcut",
    "NavigationShortcut",
    "UtilityShortcut",
    "ShortcutRegistry",
    "ShortcutType",
    "ShortcutAction",
    # Authentication
    "PasswordAuthenticator",
    "SilverBulletError",
    "ShortcutAuthenticationError",
    "AccountLockedError",
    # Handler
    "KeyboardHandler",
    "ShortcutNotDefinedError"
]
