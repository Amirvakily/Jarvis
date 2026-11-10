"""Dependency-free Markdown rendering and the front matter drafts are stored with.

The renderer covers the everyday subset: headings, paragraphs, emphasis, code,
links, images, single-level lists, quotes and rules. All text is escaped, raw
HTML is shown as text, and link or image targets are emitted only for a short
list of schemes, so model-written Markdown cannot inject markup or script.
"""

from __future__ import annotations

import html
import re
import unicodedata

MAX_SLUG_LENGTH = 80
SAFE_URL_PREFIXES = ("http://", "https://", "mailto:", "/", "#")
FRONT_MATTER_LIST_KEYS = frozenset({"tags"})

_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})\s*([\w+#.-]*)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)(?:\s+#+)?\s*$")
_HR = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
_QUOTE = re.compile(r"^\s{0,3}>\s?(.*)$")
_UL = re.compile(r"^\s*[-*+]\s+(.*)$")
_OL = re.compile(r"^\s*\d+\.\s+(.*)$")
_BLOCK_STARTERS = (_FENCE, _HEADING, _HR, _QUOTE, _UL, _OL)
_URL = r"(?:[^()\s]|\([^()\s]*\))+"
_INLINE = re.compile(
    r"`(?P<code>[^`]+)`"
    rf"|!\[(?P<alt>[^\]]*)\]\((?P<src>{_URL})\)"
    rf"|\[(?P<text>[^\]]+)\]\((?P<href>{_URL})\)"
    r"|\*\*(?P<bold>.+?)\*\*"
    r"|__(?P<bold_u>.+?)__"
    r"|\*(?P<em>[^\s*][^*]*?)\*"
    r"|(?<!\w)_(?P<em_u>[^\s_][^_]*?)_(?!\w)"
)

_STYLE = """
:root { color-scheme: light dark; --fg: #1d1d1f; --bg: #fbfbfa; --muted: #6b6b6b;
  --accent: #2a5db0; --code: #efefec; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --fg: #e8e8e6; --bg: #161618; --muted: #9a9a9a;
    --accent: #7aa7ff; --code: #26262a; }
}
:root[data-theme="dark"] { --fg: #e8e8e6; --bg: #161618; --muted: #9a9a9a;
  --accent: #7aa7ff; --code: #26262a; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 1.05rem/1.7 Georgia, "Times New Roman", serif; }
main { max-width: 42rem; margin: 0 auto; padding: 2rem 1rem; }
h1, h2, h3, h4, h5, h6 { line-height: 1.25; font-family: system-ui, sans-serif; }
a { color: var(--accent); }
img { max-width: 100%; height: auto; }
pre { background: var(--code); padding: 1rem; overflow-x: auto; border-radius: 6px; }
code { font-family: ui-monospace, monospace; font-size: 0.92em; }
blockquote { margin: 1rem 0; padding-left: 1rem; border-left: 3px solid var(--muted);
  color: var(--muted); }
hr { border: 0; border-top: 1px solid var(--muted); margin: 2rem 0; }
"""


def slugify(title: str) -> str:
    """Lowercase ASCII letters and digits joined by single hyphens; never empty."""
    text = unicodedata.normalize("NFKD", str(title)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:MAX_SLUG_LENGTH].strip("-")
    return text or "untitled"


def to_html(markdown: str) -> str:
    """Render Markdown to an HTML fragment with no page wrapper."""
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4).split("\n")
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        fence = _FENCE.match(line)
        if fence:
            marker = fence.group(1)
            code: list[str] = []
            i += 1
            while i < len(lines) and not _closes_fence(lines[i], marker):
                code.append(lines[i])
                i += 1
            i += 1
            source = html.escape("\n".join(code))
            lang = fence.group(2)
            cls = f' class="language-{lang}"' if lang else ""
            blocks.append(f"<pre><code{cls}>{source}</code></pre>")
            continue
        heading = _HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            blocks.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            i += 1
            continue
        if _HR.match(line):
            blocks.append("<hr>")
            i += 1
            continue
        if _QUOTE.match(line):
            quoted: list[str] = []
            while i < len(lines) and (match := _QUOTE.match(lines[i])):
                quoted.append(match.group(1))
                i += 1
            inner = to_html("\n".join(quoted))
            blocks.append(f"<blockquote>\n{inner}\n</blockquote>")
            continue
        if _UL.match(line) or _OL.match(line):
            pattern, tag = (_UL, "ul") if _UL.match(line) else (_OL, "ol")
            items: list[str] = []
            while i < len(lines) and (match := pattern.match(lines[i])):
                items.append(f"<li>{_inline(match.group(1))}</li>")
                i += 1
            blocks.append(f"<{tag}>\n" + "\n".join(items) + f"\n</{tag}>")
            continue
        para: list[str] = []
        while i < len(lines) and lines[i].strip() and not _starts_block(lines[i]):
            para.append(lines[i].strip())
            i += 1
        blocks.append(f"<p>{_inline(' '.join(para))}</p>")
    return "\n".join(blocks)


def render_page(title: str, body_html: str, summary: str = "") -> str:
    """Wrap a rendered fragment in a complete, minimal, standalone HTML5 page."""
    name = html.escape(title.strip() or "Untitled")
    description = ""
    if summary.strip():
        description = f'<meta name="description" content="{html.escape(summary.strip())}">\n'
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{name}</title>\n{description}"
        f"<style>{_STYLE}</style>\n"
        "</head>\n<body>\n<main>\n<article>\n"
        f"{body_html}\n"
        "</article>\n</main>\n</body>\n</html>\n"
    )


def split_front_matter(text: str) -> tuple[dict, str]:
    """Split a ``---`` delimited ``key: value`` header from the body.

    Keys listed in ``FRONT_MATTER_LIST_KEYS`` (such as ``tags``) become lists
    split on commas. Text without a complete header returns ``({}, text)``.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text
    end = next((n for n in range(1, len(lines)) if lines[n].strip() == "---"), None)
    if end is None:
        return {}, text
    meta: dict = {}
    for raw in lines[1:end]:
        key, sep, value = raw.partition(":")
        key = key.strip()
        if not sep or not key:
            continue
        value = value.strip()
        if key in FRONT_MATTER_LIST_KEYS:
            meta[key] = [item.strip() for item in value.split(",") if item.strip()]
        else:
            meta[key] = value
    body = "\n".join(lines[end + 1 :])
    if body.startswith("\n"):
        body = body[1:]
    return meta, body


def join_front_matter(meta: dict, body: str) -> str:
    """Write the header produced by :func:`split_front_matter` above ``body``."""
    if not meta:
        return body
    lines = ["---"]
    for key, value in meta.items():
        if isinstance(value, (list, tuple)):
            text = ", ".join(" ".join(str(item).split()) for item in value)
        else:
            text = " ".join(str(value).split())
        lines.append(f"{key}: {text}" if text else f"{key}:")
    return "\n".join(lines) + "\n---\n\n" + body


def _closes_fence(line: str, marker: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) == {marker[0]} and len(stripped) >= len(marker)


def _starts_block(line: str) -> bool:
    return any(pattern.match(line) for pattern in _BLOCK_STARTERS)


def _safe_url(url: str) -> bool:
    return url.lower().startswith(SAFE_URL_PREFIXES)


def _inline(text: str) -> str:
    out: list[str] = []
    pos = 0
    for match in _INLINE.finditer(text):
        out.append(html.escape(text[pos : match.start()]))
        out.append(_inline_token(match))
        pos = match.end()
    out.append(html.escape(text[pos:]))
    return "".join(out)


def _inline_token(match: re.Match[str]) -> str:
    if match.group("code") is not None:
        return f"<code>{html.escape(match.group('code'))}</code>"
    if match.group("src") is not None:
        src, alt = match.group("src"), match.group("alt")
        if _safe_url(src):
            return f'<img src="{html.escape(src)}" alt="{html.escape(alt)}">'
        return html.escape(alt)
    if match.group("href") is not None:
        text = _inline(match.group("text"))
        href = match.group("href")
        if _safe_url(href):
            return f'<a href="{html.escape(href)}" rel="noopener noreferrer">{text}</a>'
        return text
    if match.group("bold") is not None:
        return f"<strong>{_inline(match.group('bold'))}</strong>"
    if match.group("bold_u") is not None:
        return f"<strong>{_inline(match.group('bold_u'))}</strong>"
    inner = match.group("em") if match.group("em") is not None else match.group("em_u")
    return f"<em>{_inline(inner)}</em>"
