#!/usr/bin/env python3
"""
Update Epic 5 Phase 3 story files to mark as completed.

This script updates the story files for stories 5-3-2 through 5-3-6
to reflect that they were completed as part of the 1-minute migration.
"""

import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent

# Stories to update
stories = [
    "5-3-2-train-regime-specific-xgboost-models",
    "5-3-3-implement-dynamic-model-switching",
    "5-3-4-validate-regime-detection-accuracy",
    "5-3-5-validate-ranging-market-improvement",
    "5-3-6-complete-historical-validation-for-regime-aware-system",
]

completion_date = "2026-04-15T22:00:00Z"

for story_key in stories:
    story_file = project_root / "_bmad-output" / "implementation-artifacts" / f"{story_key}.md"

    if not story_file.exists():
        print(f"⚠️  Story file not found: {story_file}")
        continue

    print(f"Updating {story_key}...")

    # Read the story file
    content = story_file.read_text()

    # Update frontmatter status
    content = content.replace(
        "status: 'backlog'",
        "status: 'completed'\ncompleted: '2026-04-15T22:00:00Z'"
    )

    # Update status in body
    content = content.replace(
        "**Status:** backlog",
        "**Status:** completed\n**Completed:** 2026-04-15"
    )

    # Add completion note to end
    if "**Story Status:** 📋 BACKLOG" in content:
        content = content.replace(
            "**Story Status:** 📋 BACKLOG",
            "**Story Status:** ✅ COMPLETED (1-Minute Migration)\n**Completed:** 2026-04-15"
        )

    # Write back
    story_file.write_text(content)
    print(f"  ✅ Updated {story_key}")

print("\n✅ All Epic 5 Phase 3 story files updated to completed status")
