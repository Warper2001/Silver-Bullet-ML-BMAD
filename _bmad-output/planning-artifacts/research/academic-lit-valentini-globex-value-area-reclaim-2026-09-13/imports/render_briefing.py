"""Render the complete research markdown offline, without changing its claims."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re
import subprocess

from markdown_it import MarkdownIt

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[3]
REPORT = RUN / 'research.md'
text = REPORT.read_text()
source_result = subprocess.run(
    ['uv', 'run', str(ROOT / '.agents/skills/bmad-deep-recon/scripts/recon_kit.py'),
     'escape-sources', str(REPORT)], capture_output=True, text=True, check=True)
sources = json.loads(source_result.stdout)
if sources['invalid_urls']:
    raise ValueError(sources['invalid_urls'])
(RUN / 'escaped-sources.json').write_text(source_result.stdout)
md = MarkdownIt('commonmark', {'html': False}).enable('table')
body = re.sub(r'^---\n.*?\n---\n', '', text, count=1, flags=re.S)
parts = re.split(r'^## (.+)$', body, flags=re.M)
sections = []
nav = []
for index in range(1, len(parts), 2):
    title, content = parts[index], parts[index + 1]
    sid = 'section-' + str(index)
    nav.append(f'<a href="#{sid}">{html.escape(title)}</a>')
    if title == 'Source appendix':
        intro = content.split('\n|', 1)[0]
        rendered = md.render(intro) + sources['html']
        section = f'<details id="{sid}"><summary>{html.escape(title)}</summary>{rendered}</details>'
    else:
        rendered = md.render(content)
        rendered = re.sub(r'\[(\d+)\]', r'<a class="citation" href="#src-\1">[\1]</a>', rendered)
        section = f'<section id="{sid}"><h2>{html.escape(title)}</h2>{rendered}</section>'
    sections.append(section)
css = '''
:root{color-scheme:light dark;--bg:#f5f6f8;--card:#fff;--fg:#18212c;--muted:#526072;--line:#d7dde5;--accent:#174b73}
@media(prefers-color-scheme:dark){:root{--bg:#101820;--card:#19242e;--fg:#e8edf3;--muted:#a9b8c7;--line:#394653;--accent:#99cbef}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.65 system-ui,sans-serif}
header,main,footer{max-width:1050px;margin:auto;padding:24px}h1{font-size:2rem;line-height:1.2}h2{line-height:1.3}a{color:var(--accent)}
section,details{background:var(--card);padding:22px;margin:20px 0;border:1px solid var(--line);border-radius:8px;overflow:auto}
nav{display:flex;gap:14px;overflow:auto;position:sticky;top:0;background:var(--card);padding:12px;border-bottom:1px solid var(--line);z-index:1;font-size:.85rem}
nav a{white-space:nowrap}table{border-collapse:collapse;width:100%;font-size:.86rem}td,th{padding:10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:var(--bg)}
code{font-size:.85em;overflow-wrap:anywhere}summary{cursor:pointer;font-size:1.3rem;font-weight:700}.meta{color:var(--muted);font-size:.9rem}.badge{display:inline-block;padding:3px 9px;border-radius:5px;border:1px solid var(--line);margin:3px;font-size:.8rem}
.warning{border-color:#b67b28;font-weight:700}.citation{font-size:.9em}footer{font-size:.8rem;color:var(--muted)}
@media(max-width:650px){header,main,footer{padding:12px}section,details{padding:14px}h1{font-size:1.5rem}}
'''
header = '''<header><div class="meta">Academic literature · Standard scope · Normal verification · 2026-09-13</div>
<h1>MNQ Globex value-area reclaim</h1><p>Decision: whether to backtest or prospectively study the specified long-only setup.</p>
<span class="badge warning">DATA_UNSUITABLE</span><span class="badge warning">POWER_UNDETERMINED</span>
<p class="meta">Verified refers to independently corroborated methodological claims. Medium or unverified claims are qualified in the text and source table. No strategy profitability is established.</p></header>'''
page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>MNQ Globex value-area reclaim research</title><style>' + css + '</style>' + header
page += '<main>' + sections[0] + '<nav aria-label="Report sections">' + ''.join(nav) + '</nav>' + ''.join(sections[1:]) + '</main>'
page += '<footer>Complete presentation of research.md · SHA-256 ' + hashlib.sha256(REPORT.read_bytes()).hexdigest() + ' · No external assets or network requests.</footer></html>'
(RUN / 'research-briefing.html').write_text(page)
print(RUN / 'research-briefing.html')
