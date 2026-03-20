"""Search functionality for help content."""

from src.dashboard.help.help_content import (
    OVERVIEW_HELP,
    POSITIONS_HELP,
    SIGNALS_HELP,
    CHARTS_HELP,
    SETTINGS_HELP,
)
from src.dashboard.help.glossary import GLOSSARY


def search_help(query: str) -> list:
    """Search help content for matching topics.

    Args:
        query: Search query string

    Returns:
        List of tuples (page, section, preview) matching the query
    """
    if not query or len(query) < 2:
        return []

    query_lower = query.lower()
    results = []

    # Search in page help content
    help_pages = {
        "Overview": OVERVIEW_HELP,
        "Positions": POSITIONS_HELP,
        "Signals": SIGNALS_HELP,
        "Charts": CHARTS_HELP,
        "Settings": SETTINGS_HELP,
    }

    for page, content in help_pages.items():
        # Split content into sections
        sections = content.split("##")

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # Extract section title (first line)
            lines = section.split("\n")
            title = lines[0].strip() if lines else ""

            # Check if query matches title or content
            if query_lower in title.lower() or query_lower in section.lower():
                # Get preview (first 100 chars)
                preview = section.replace("\n", " ")[:100] + "..."

                results.append({
                    "page": page,
                    "section": title,
                    "preview": preview,
                })

    # Search in glossary
    for term, definition in GLOSSARY.items():
        if query_lower in term.lower() or query_lower in definition.lower():
            # Clean up definition (remove markdown)
            clean_def = definition.replace("**", "").replace("\n", " ").strip()
            preview = clean_def[:100] + "..."

            results.append({
                "page": "Glossary",
                "section": term,
                "preview": preview,
            })

    return results


def search_glossary_only(query: str) -> list:
    """Search only glossary for matching terms.

    Args:
        query: Search query string

    Returns:
        List of matching glossary terms
    """
    if not query or len(query) < 2:
        return []

    query_lower = query.lower()
    results = []

    for term, definition in GLOSSARY.items():
        if query_lower in term.lower() or query_lower in definition.lower():
            results.append({
                "term": term,
                "definition": definition.replace("**", "").strip(),
            })

    return results
