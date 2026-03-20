"""
Keyboard shortcuts definitions and registry.

Defines all keyboard shortcuts for the BMAD dashboard including emergency,
navigation, and utility shortcuts.
"""

from dataclasses import dataclass
from typing import List, Optional, Literal
from enum import Enum


class ShortcutType(Enum):
    """Types of shortcuts."""
    EMERGENCY = "emergency"
    NAVIGATION = "navigation"
    UTILITY = "utility"


ShortcutAction = Literal[
    "emergency_stop",
    "manual_flatten",
    "refresh",
    "log_viewer",
    "navigate_settings",
    "navigate_positions",
    "open_help",
    "close_modal"
]


@dataclass
class Shortcut:
    """Base shortcut definition."""
    key_combination: str
    action: ShortcutAction
    description: str
    requires_password: bool = False
    shortcut_type: ShortcutType = ShortcutType.UTILITY

    def __str__(self) -> str:
        """Return string representation."""
        password_req = " [PASSWORD]" if self.requires_password else ""
        return f"{self.key_combination}: {self.description}{password_req}"


@dataclass
class EmergencyShortcut(Shortcut):
    """Emergency shortcut requiring password authentication."""
    shortcut_type: ShortcutType = ShortcutType.EMERGENCY
    requires_password: bool = True


@dataclass
class NavigationShortcut(Shortcut):
    """Navigation shortcut for quick page access."""
    shortcut_type: ShortcutType = ShortcutType.NAVIGATION
    requires_password: bool = False


@dataclass
class UtilityShortcut(Shortcut):
    """Utility shortcut for common actions."""
    shortcut_type: ShortcutType = ShortcutType.UTILITY
    requires_password: bool = False


class ShortcutRegistry:
    """Registry for all keyboard shortcuts."""

    def __init__(self):
        """Initialize shortcut registry with all shortcuts."""
        self._shortcuts: dict[str, Shortcut] = {}
        self._register_all_shortcuts()

    def _register_all_shortcuts(self):
        """Register all required shortcuts."""
        # Emergency shortcuts
        self.register(EmergencyShortcut(
            key_combination="Ctrl+E",
            action="emergency_stop",
            description="Emergency stop - halt all trading activity"
        ))

        self.register(EmergencyShortcut(
            key_combination="Ctrl+F",
            action="manual_flatten",
            description="Manual flatten - close all open positions"
        ))

        # Navigation shortcuts
        self.register(NavigationShortcut(
            key_combination="Ctrl+S",
            action="navigate_settings",
            description="Navigate to Settings page"
        ))

        self.register(NavigationShortcut(
            key_combination="Ctrl+P",
            action="navigate_positions",
            description="Navigate to Positions page"
        ))

        # Utility shortcuts
        self.register(UtilityShortcut(
            key_combination="Ctrl+R",
            action="refresh",
            description="Refresh current page data"
        ))

        self.register(UtilityShortcut(
            key_combination="Ctrl+L",
            action="log_viewer",
            description="Open log viewer"
        ))

        self.register(UtilityShortcut(
            key_combination="F1",
            action="open_help",
            description="Open context-sensitive help"
        ))

        self.register(UtilityShortcut(
            key_combination="ESC",
            action="close_modal",
            description="Close active modal/dialog"
        ))

    def register(self, shortcut: Shortcut) -> None:
        """Register a shortcut in the registry."""
        self._shortcuts[shortcut.key_combination] = shortcut

    def get_shortcut(self, key_combination: str) -> Optional[Shortcut]:
        """Get shortcut by key combination."""
        return self._shortcuts.get(key_combination)

    def get_all_shortcuts(self) -> List[Shortcut]:
        """Get all registered shortcuts."""
        return list(self._shortcuts.values())

    def get_emergency_shortcuts(self) -> List[EmergencyShortcut]:
        """Get all emergency shortcuts."""
        return [s for s in self._shortcuts.values() if isinstance(s, EmergencyShortcut)]

    def get_navigation_shortcuts(self) -> List[NavigationShortcut]:
        """Get all navigation shortcuts."""
        return [s for s in self._shortcuts.values() if isinstance(s, NavigationShortcut)]

    def get_utility_shortcuts(self) -> List[UtilityShortcut]:
        """Get all utility shortcuts."""
        return [s for s in self._shortcuts.values() if isinstance(s, UtilityShortcut)]
