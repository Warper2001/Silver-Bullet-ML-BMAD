"""Help modal component for displaying context-sensitive help."""

import streamlit as st

from src.dashboard.help.help_content import get_help_content, HELP_TABLE_OF_CONTENTS
from src.dashboard.help.glossary import GLOSSARY


def show_help_modal(page: str = "Overview") -> None:
    """Display help modal for specified page.

    Args:
        page: Current page name for context-sensitive help
    """
    # Modal title
    st.title(f"📖 Help - {page} Page")

    # Table of contents
    st.markdown("---")
    st.subheader("📑 Table of Contents")

    # Create columns for TOC layout
    cols = st.columns(3)
    for i, (name, description) in enumerate(HELP_TABLE_OF_CONTENTS):
        with cols[i % 3]:
            if st.button(f"**{name}**", key=f"toc_{name}", use_container_width=True):
                st.session_state["help_page"] = name
                st.rerun()

    # Show page-specific help content
    st.markdown("---")
    st.subheader(f"📄 {page} Page Help")

    help_content = get_help_content(page)
    st.markdown(help_content)

    # Glossary section
    st.markdown("---")
    st.subheader("📚 Glossary of Trading Terms")

    # Glossary search
    search_query = st.text_input("🔍 Search glossary:", key="glossary_search")

    if search_query and len(search_query) >= 2:
        from src.dashboard.help.help_search import search_glossary_only

        matching_terms = search_glossary_only(search_query)

        if matching_terms:
            st.write(f"**Found {len(matching_terms)} matching terms:**")

            for term in matching_terms:
                with st.expander(f"**{term['term']}**"):
                    st.markdown(term["definition"])
        else:
            st.info("No matching terms found.")
    else:
        # Show all glossary terms (expandable)
        st.write("**Click a term to see its definition:**")

        # Group terms alphabetically
        sorted_terms = sorted(GLOSSARY.keys())

        # Show in columns
        glossary_cols = st.columns(2)
        for i, term in enumerate(sorted_terms):
            with glossary_cols[i % 2]:
                with st.expander(f"**{term}**"):
                    st.markdown(GLOSSARY[term])

    # Close button
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("✖️ Close Help", type="primary", key="close_help"):
            st.session_state["show_help"] = False
            st.rerun()


def show_help_button() -> None:
    """Display help button in sidebar."""
    if st.button("📖 Help", key="help_button"):
        st.session_state["show_help"] = True
        st.session_state["help_page"] = st.session_state.get("page", "Overview")
        st.rerun()


def show_tooltip(metric: str) -> None:
    """Display tooltip info for a metric.

    Args:
        metric: Metric name to show tooltip for
    """
    from src.dashboard.help.tooltips import get_tooltip

    tooltip_text = get_tooltip(metric)
    if tooltip_text:
        st.caption(f"ℹ️ {tooltip_text}")
    else:
        st.caption("")  # Empty caption to maintain spacing
