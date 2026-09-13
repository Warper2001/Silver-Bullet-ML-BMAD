from __future__ import annotations

from html import escape
import json
from pathlib import Path


def write_report(path: Path, title: str, sections: list[tuple[str, object]]) -> None:
    lines = [f"# {title}", "", "**Research only. Deployment is not authorized.**", ""]
    html = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{escape(title)}</title>",
        "<style>body{font:16px system-ui;max-width:960px;margin:2rem auto;"
        "padding:0 1rem}pre{background:#f5f5f5;padding:1rem;overflow:auto}"
        "h1,h2{color:#17233b}</style>",
        f"<h1>{escape(title)}</h1>",
        "<p><strong>Research only. Deployment is not authorized.</strong></p>",
    ]
    for heading, value in sections:
        rendered = (
            value
            if isinstance(value, str)
            else json.dumps(value, indent=2, sort_keys=True)
        )
        lines.extend([f"## {heading}", "", "```json", rendered, "```", ""])
        html.extend([f"<h2>{escape(heading)}</h2>", f"<pre>{escape(rendered)}</pre>"])
    (path / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (path / "report.html").write_text("\n".join(html), encoding="utf-8")
