"""Terminal theme and color scheme definitions for Rich TUI."""

from rich.console import Console
from rich.style import Style

# Color scheme for terminal UI
COLOR_SCHEME = {
    "profit": "green",
    "loss": "red",
    "healthy": "green",
    "error": "red",
    "warning": "yellow",
    "stale": "yellow",
    "info": "blue",
    "neutral": "white",
}

# Text styles
STYLE_BOLD = Style(bold=True)
STYLE_HEADER = Style(bold=True, color="cyan")
STYLE_SUCCESS = Style(color="green", bold=True)
STYLE_ERROR = Style(color="red", bold=True)
STYLE_WARNING = Style(color="yellow", bold=True)

# Console instance for terminal UI
console = Console()
