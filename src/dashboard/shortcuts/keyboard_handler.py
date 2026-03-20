"""
Keyboard event handler for shortcut detection.

Handles parsing keyboard events and matching them to registered shortcuts.
"""

import logging
from typing import Optional, Dict, Any
from src.dashboard.shortcuts.shortcuts import Shortcut, ShortcutRegistry


logger = logging.getLogger(__name__)


class KeyboardHandler:
    """Handler for keyboard events and shortcut matching."""

    def __init__(self, registry: Optional[ShortcutRegistry] = None):
        """Initialize keyboard handler.

        Args:
            registry: ShortcutRegistry instance (creates new one if None)
        """
        self.registry = registry or ShortcutRegistry()

    def parse_event(self, key_combination: str) -> Dict[str, Any]:
        """Parse keyboard event from key combination string.

        Args:
            key_combination: Key combination string (e.g., "ctrl+e", "shift+a", "f1")

        Returns:
            Dictionary with parsed event data:
            {
                "key": str,           # The key (e.g., "e", "a", "F1", "Escape")
                "ctrl": bool,         # True if Ctrl pressed
                "shift": bool,        # True if Shift pressed
                "alt": bool           # True if Alt pressed
            }
        """
        # Normalize to lowercase
        key_combination = key_combination.lower().strip()

        # Parse modifiers
        parts = key_combination.split("+")
        modifiers = set(parts[:-1])
        key = parts[-1]

        # Parse function keys (only standalone F1-F12, not ctrl+f, etc.)
        if key.startswith("f") and len(key) > 1 and key[1:].isdigit():
            key = key.upper()

        # Parse escape
        if key in ["escape", "esc"]:
            key = "Escape"

        # Parse event
        event = {
            "key": key,
            "ctrl": "ctrl" in modifiers or "control" in modifiers,
            "shift": "shift" in modifiers,
            "alt": "alt" in modifiers
        }

        logger.debug(f"Parsed keyboard event: {event}")
        return event

    def match_shortcut(self, keyboard_event: Dict[str, Any], registry: Optional[ShortcutRegistry] = None) -> Optional[Shortcut]:
        """Match keyboard event to a registered shortcut.

        Args:
            keyboard_event: Parsed keyboard event from parse_event()
            registry: ShortcutRegistry (uses self.registry if None)

        Returns:
            Matching Shortcut or None if no match
        """
        registry = registry or self.registry

        # Build key combination string from event
        key_parts = []
        if keyboard_event["ctrl"]:
            key_parts.append("Ctrl")
        if keyboard_event["shift"]:
            key_parts.append("Shift")
        if keyboard_event["alt"]:
            key_parts.append("Alt")

        key_parts.append(keyboard_event["key"].upper() if keyboard_event["key"] != "Escape" else "ESC")
        key_combination = "+".join(key_parts)

        # Handle special cases
        if keyboard_event["key"] == "Escape":
            key_combination = "ESC"
        elif keyboard_event["key"].startswith("F"):
            key_combination = keyboard_event["key"].upper()

        # Lookup shortcut
        shortcut = registry.get_shortcut(key_combination)

        if shortcut:
            logger.info(f"Matched keyboard event to shortcut: {shortcut.key_combination}")

        return shortcut

    def handle_shortcut(self, key_combination: str) -> Optional[Shortcut]:
        """Handle shortcut from key combination string.

        Args:
            key_combination: Key combination string (e.g., "ctrl+e")

        Returns:
            Matching Shortcut or None if no match
        """
        event = self.parse_event(key_combination)
        return self.match_shortcut(event)


class ShortcutNotDefinedError(Exception):
    """Raised when attempting to use undefined shortcut."""

    pass
