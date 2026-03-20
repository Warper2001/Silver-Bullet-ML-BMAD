"""Keyboard event handling for terminal UI."""

# Keyboard shortcuts mapping
KEYBOARD_SHORTCUTS = {
    'q': 'quit',
    'r': 'refresh',
    'e': 'emergency_stop',
}


def handle_keypress(key: str) -> str:
    """Handle keypress and return action.

    Args:
        key: Key pressed by user

    Returns:
        Action string ('quit', 'refresh', 'emergency_stop', or 'none')
    """
    return KEYBOARD_SHORTCUTS.get(key.lower(), 'none')


def should_quit(key: str) -> bool:
    """Check if key should quit application.

    Args:
        key: Key pressed by user

    Returns:
        True if should quit, False otherwise
    """
    return key.lower() == 'q'


def should_refresh(key: str) -> bool:
    """Check if key should trigger refresh.

    Args:
        key: Key pressed by user

    Returns:
        True if should refresh, False otherwise
    """
    return key.lower() == 'r'


def should_emergency_stop(key: str) -> bool:
    """Check if key should trigger emergency stop.

    Args:
        key: Key pressed by user

    Returns:
        True if should emergency stop, False otherwise
    """
    return key.lower() == 'e'
