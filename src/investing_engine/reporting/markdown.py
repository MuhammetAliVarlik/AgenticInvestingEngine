"""A deliberately tiny, safe Markdown-to-HTML converter for model output.

Model output is untrusted, so general-purpose Markdown libraries (which pass
raw HTML through) are not used. Everything is HTML-escaped first; only
headings, bold, italics, bullet lists and paragraphs are then recognised.
Links and images are rendered as plain text, so the PDF engine never sees a
URL it could fetch.
"""

from __future__ import annotations

import html
import re

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])")
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")


def _inline(text: str) -> str:
    text = _LINK.sub(r"\1", text)
    text = html.escape(text, quote=True)
    text = _BOLD.sub(r"<strong>\1</strong>", text)
    return _ITALIC.sub(r"<em>\1</em>", text)


def to_html(markdown: str) -> str:
    blocks: list[str] = []
    paragraph: list[str] = []
    items: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(f"<p>{'<br>'.join(paragraph)}</p>")
            paragraph.clear()
        if items:
            blocks.append("<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")
            items.clear()

    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            flush()
        elif heading := re.match(r"^(#{1,4})\s+(.*)$", line):
            flush()
            level = min(len(heading.group(1)) + 1, 4)
            blocks.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
        elif bullet := re.match(r"^[-*•]\s+(.*)$", line):
            if paragraph:
                flush()
            items.append(_inline(bullet.group(1)))
        else:
            if items:
                flush()
            paragraph.append(_inline(line))
    flush()
    return "\n".join(blocks)
