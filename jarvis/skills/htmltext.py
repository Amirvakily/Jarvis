"""Stdlib-only HTML helpers: readable text, structured extraction, simple selectors."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urljoin

_VOID = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
    "source", "track", "wbr",
}  # fmt: skip
_SKIP = {"script", "style", "noscript", "svg", "template", "head"}
_BLOCK = {
    "address", "article", "aside", "blockquote", "br", "dd", "details", "div", "dl", "dt",
    "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6",
    "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section", "summary", "table",
    "tr", "ul",
}  # fmt: skip
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_SELECTOR = re.compile(r"^([a-zA-Z][a-zA-Z0-9]*)?(?:([.#])([\w-]+))?$")


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tidy(raw: str) -> str:
    lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in raw.split("\n"))
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._in_title = False
        self._skip = 0
        self._focus = 0
        self.all: list[str] = []
        self.focused: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "title":
            self._in_title = True
        if tag in _VOID:
            if tag in _BLOCK:
                self._emit("\n")
            return
        if tag in _SKIP:
            self._skip += 1
            return
        if tag in ("main", "article"):
            self._focus += 1
        if tag in _BLOCK:
            self._emit("\n")

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        if tag in _BLOCK:
            self._emit("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in _VOID:
            return
        if tag in _SKIP:
            self._skip = max(0, self._skip - 1)
            return
        if tag in ("main", "article"):
            self._focus = max(0, self._focus - 1)
        if tag in _BLOCK:
            self._emit("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        if not self._skip and not self._in_title:
            self._emit(data)

    def _emit(self, text: str) -> None:
        if self._skip:
            return
        self.all.append(text)
        if self._focus:
            self.focused.append(text)


def extract_text(html: str) -> tuple[str, str]:
    """Return (title, readable text), preferring <main>/<article> content."""
    parser = _TextParser()
    parser.feed(html)
    parser.close()
    focused = _tidy("".join(parser.focused))
    text = focused or _tidy("".join(parser.all))
    return _squash(parser.title), text


class _DocParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base = base_url
        self.links: list[dict] = []
        self.headings: list[dict] = []
        self.tables: list[list[list[str]]] = []
        self.images: list[dict] = []
        self.meta: dict[str, str] = {}
        self.forms: list[dict] = []
        self.scripts: list[str] = []
        self._link: dict | None = None
        self._heading: tuple[int, list[str]] | None = None
        self._table_stack: list[list[list[str]]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def _abs(self, value: str | None) -> str:
        return urljoin(self.base, value.strip()) if value and value.strip() else ""

    def handle_starttag(self, tag: str, attrs: list) -> None:
        a = {k: (v or "") for k, v in attrs}
        if tag == "a" and a.get("href"):
            self._link = {"text": [], "url": self._abs(a["href"])}
        elif tag in _HEADINGS:
            self._heading = (int(tag[1]), [])
        elif tag == "img":
            self.images.append({"src": self._abs(a.get("src")), "alt": a.get("alt", "").strip()})
        elif tag == "meta":
            key = a.get("name") or a.get("property") or a.get("http-equiv")
            if key and "content" in a:
                self.meta[key] = a["content"].strip()
        elif tag == "form":
            self.forms.append(
                {"action": self._abs(a.get("action")) or self.base,
                 "method": (a.get("method") or "get").lower()}
            )  # fmt: skip
        elif tag == "script" and a.get("src"):
            self.scripts.append(self._abs(a["src"]))
        elif tag == "table":
            self._table_stack.append([])
        elif tag == "tr" and self._table_stack:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._link is not None:
            self.links.append(
                {"text": _squash("".join(self._link["text"])), "url": self._link["url"]}
            )
            self._link = None
        elif tag in _HEADINGS and self._heading is not None:
            level, parts = self._heading
            text = _squash("".join(parts))
            if text:
                self.headings.append({"level": level, "text": text})
            self._heading = None
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(_squash("".join(self._cell)))
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table_stack:
            if self._row:
                self._table_stack[-1].append(self._row)
            self._row = None
        elif tag == "table" and self._table_stack:
            rows = self._table_stack.pop()
            if rows:
                self.tables.append(rows)

    def handle_data(self, data: str) -> None:
        if self._link is not None:
            self._link["text"].append(data)
        if self._heading is not None:
            self._heading[1].append(data)
        if self._cell is not None:
            self._cell.append(data)


def parse_document(html: str, base_url: str) -> dict:
    """Structured view of a page; every URL is made absolute against ``base_url``."""
    parser = _DocParser(base_url)
    parser.feed(html)
    parser.close()
    return {
        "links": parser.links,
        "headings": parser.headings,
        "tables": parser.tables,
        "images": parser.images,
        "meta": parser.meta,
        "forms": parser.forms,
        "scripts": parser.scripts,
    }


class _SelectParser(HTMLParser):
    def __init__(self, tag: str | None, kind: str | None, value: str | None) -> None:
        super().__init__(convert_charrefs=True)
        self.tag, self.kind, self.value = tag, kind, value
        self.stack: list[tuple[str, list[str] | None]] = []
        self.captures: list[list[str]] = []
        self.results: list[list[str]] = []
        self._skip = 0

    def _matches(self, tag: str, attrs: dict[str, str]) -> bool:
        if self.tag and tag != self.tag:
            return False
        if self.kind == ".":
            return self.value in attrs.get("class", "").split()
        if self.kind == "#":
            return attrs.get("id") == self.value
        return True

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in _VOID:
            return
        a = {k: (v or "") for k, v in attrs}
        buf: list[str] | None = None
        if self._matches(tag, a):
            buf = []
            self.captures.append(buf)
            self.results.append(buf)
        self.stack.append((tag, buf))
        if tag in _SKIP:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID:
            return
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                while len(self.stack) > i:
                    name, buf = self.stack.pop()
                    if name in _SKIP:
                        self._skip = max(0, self._skip - 1)
                    if buf is not None:
                        self.captures.remove(buf)
                break

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        for buf in self.captures:
            buf.append(data)


def select(html: str, selector: str) -> list[str]:
    """Text of elements matching ``tag``, ``.class``, ``#id``, ``tag.class`` or ``tag#id``."""
    match = _SELECTOR.match(selector.strip())
    if not match or not (match.group(1) or match.group(2)):
        raise ValueError(
            f"Unsupported selector '{selector}'. Use tag, .class, #id, tag.class or tag#id."
        )
    tag = match.group(1).lower() if match.group(1) else None
    parser = _SelectParser(tag, match.group(2), match.group(3))
    parser.feed(html)
    parser.close()
    return [text for text in (_squash("".join(buf)) for buf in parser.results) if text]
