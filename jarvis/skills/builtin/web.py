"""Web skills: fetch a page, scrape structured data, search the web."""

from __future__ import annotations

from collections.abc import Callable
from html.parser import HTMLParser
from typing import Any, Literal
from urllib.parse import parse_qs, quote_plus, urljoin, urlsplit

from jarvis._http import httpx
from jarvis.skills.base import SkillContext, SkillError, skill
from jarvis.skills.htmltext import extract_text, parse_document, select
from jarvis.skills.util import check_url, http_client

MAX_BODY = 2_000_000
MAX_REDIRECTS = 5
_REDIRECT_CODES = {301, 302, 303, 307, 308}
_TEXT_SUBTYPES = ("json", "xml", "javascript", "yaml", "csv")


async def safe_get(
    ctx: SkillContext,
    url: str,
    *,
    allow_private: bool | None = None,
    max_bytes: int = MAX_BODY,
    follow: Callable[[str], bool] | None = None,
) -> tuple[str, int, httpx.Headers, bytes]:
    """GET ``url`` with every hop validated by ``check_url``.

    Redirects are followed manually (up to 5). If ``follow`` is given and returns
    False for a redirect target, the redirect response itself is returned. The body
    is streamed and cut off at ``max_bytes``. Returns (final_url, status, headers, body).
    """
    current = await check_url(ctx, url, allow_private=allow_private)
    async with http_client(ctx, follow_redirects=False) as client:
        for _ in range(MAX_REDIRECTS + 1):
            try:
                async with client.stream("GET", current) as response:
                    body = bytearray()
                    if response.status_code not in _REDIRECT_CODES:
                        async for chunk in response.aiter_bytes():
                            body.extend(chunk)
                            if len(body) >= max_bytes:
                                del body[max_bytes:]
                                break
                    status, headers = response.status_code, response.headers
            except httpx.HTTPError as exc:
                raise SkillError(f"Could not reach {current}: {exc or type(exc).__name__}") from exc
            location = headers.get("location")
            if status not in _REDIRECT_CODES or not location:
                return current, status, headers, bytes(body)
            target = urljoin(current, location)
            if follow is not None and not follow(target):
                return current, status, headers, b""
            current = await check_url(ctx, target, allow_private=allow_private)
    raise SkillError(f"Too many redirects fetching {url}.")


def _decode(headers: httpx.Headers, body: bytes) -> str:
    charset = "utf-8"
    for part in headers.get("content-type", "").split(";")[1:]:
        key, _, value = part.partition("=")
        if key.strip().lower() == "charset" and value.strip():
            charset = value.strip().strip("\"'")
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def _kind(content_type: str) -> str:
    base = content_type.split(";")[0].strip().lower()
    if base in ("text/html", "application/xhtml+xml"):
        return "html"
    if not base or base.startswith("text/"):
        return "text"
    if any(s in base for s in _TEXT_SUBTYPES):
        return "text"
    return "binary"


async def _load_html(ctx: SkillContext, url: str) -> tuple[str, str]:
    final, status, headers, body = await safe_get(ctx, url)
    if status >= 400:
        raise SkillError(f"{final} returned HTTP {status}.")
    if _kind(headers.get("content-type", "")) == "binary":
        raise SkillError(f"{final} is not a web page ({headers.get('content-type')}).")
    return final, _decode(headers, body)


@skill(tags=["web"])
async def web_fetch(url: str, max_chars: int = 6000, ctx: SkillContext = None) -> dict:
    """Download a web page and return its readable text.

    Use it to read an article, documentation page or API response at a known URL.

    Args:
        url: The http(s) URL to fetch.
        max_chars: Maximum number of characters of text to return.
    """
    final, status, headers, body = await safe_get(ctx, url)
    content_type = headers.get("content-type", "")
    kind = _kind(content_type)
    if kind == "binary":
        raise SkillError(f"{final} is binary content ({content_type}); I can only read text.")
    raw = _decode(headers, body)
    title = ""
    if kind == "html":
        title, text = extract_text(raw)
    else:
        text = raw.strip()
    limit = max(1, int(max_chars))
    truncated = len(text) > limit or len(body) >= MAX_BODY
    return {
        "url": final,
        "status": status,
        "content_type": content_type.split(";")[0].strip(),
        "title": title,
        "text": text[:limit],
        "truncated": truncated,
    }


@skill(tags=["web"])
async def web_scrape(
    url: str,
    extract: Literal["all", "links", "headings", "tables", "images", "meta"] = "all",
    selector: str | None = None,
    limit: int = 50,
    ctx: SkillContext = None,
) -> dict[str, Any]:
    """Extract structured data (links, headings, tables, images, metadata) from a web page.

    Use it when you need specific parts of a page rather than its prose, or text
    from elements matching a simple CSS selector.

    Args:
        url: The http(s) URL of the page.
        extract: Which kind of data to return; "all" returns every kind.
        selector: Optional simple selector (tag, .class, #id, tag.class, tag#id); returns matching text.
        limit: Maximum number of items returned per list.
    """
    final, html = await _load_html(ctx, url)
    cap = max(1, int(limit))
    if selector:
        try:
            matches = select(html, selector)
        except ValueError as exc:
            raise SkillError(str(exc)) from exc
        return {"url": final, "selector": selector, "matches": matches[:cap]}
    doc = parse_document(html, final)
    if extract != "all":
        doc = {extract: doc[extract]}
    result: dict[str, Any] = {"url": final}
    for key, value in doc.items():
        result[key] = list(value.items())[:cap] if key == "meta" else value[:cap]
        if key == "meta":
            result[key] = dict(result[key])
    return result


class _DDGParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._title: list[str] | None = None
        self._snippet: tuple[str, int, list[str]] | None = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        a = {k: (v or "") for k, v in attrs}
        classes = a.get("class", "").split()
        if tag == "a" and "result__a" in classes:
            self.results.append({"title": "", "url": a.get("href", ""), "snippet": ""})
            self._title = []
        elif "result__snippet" in classes and self.results and self._snippet is None:
            self._snippet = (tag, 1, [])
        elif self._snippet and tag == self._snippet[0]:
            self._snippet = (tag, self._snippet[1] + 1, self._snippet[2])

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._title is not None:
            self.results[-1]["title"] = " ".join("".join(self._title).split())
            self._title = None
        if self._snippet and tag == self._snippet[0]:
            name, depth, parts = self._snippet
            if depth <= 1:
                self.results[-1]["snippet"] = " ".join("".join(parts).split())
                self._snippet = None
            else:
                self._snippet = (name, depth - 1, parts)

    def handle_data(self, data: str) -> None:
        if self._title is not None:
            self._title.append(data)
        if self._snippet:
            self._snippet[2].append(data)


def _real_url(href: str) -> str:
    absolute = urljoin("https://duckduckgo.com/", href)
    parts = urlsplit(absolute)
    if parts.hostname and parts.hostname.endswith("duckduckgo.com") and parts.path == "/l/":
        target = parse_qs(parts.query).get("uddg")
        return target[0] if target else ""
    return absolute


@skill(tags=["web"])
async def web_search(query: str, max_results: int = 5, ctx: SkillContext = None) -> list[dict]:
    """Search the web (DuckDuckGo) and return titles, URLs and snippets.

    Use it to find current information or to discover a URL; follow up with
    web_fetch to read a result.

    Args:
        query: The search query.
        max_results: Maximum number of results to return.
    """
    if not query.strip():
        raise SkillError("The search query is empty.")
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query.strip())}"
    try:
        final, status, headers, body = await safe_get(ctx, url)
    except SkillError as exc:
        raise SkillError(f"Web search is unavailable right now ({exc}).") from exc
    if status >= 400:
        raise SkillError(f"Web search is unavailable right now (HTTP {status}).")
    parser = _DDGParser()
    parser.feed(_decode(headers, body))
    results = []
    for item in parser.results:
        target = _real_url(item["url"])
        if target.startswith(("http://", "https://")) and item["title"]:
            results.append({"title": item["title"], "url": target, "snippet": item["snippet"]})
    if not results:
        raise SkillError("Web search is unavailable or returned no results.")
    return results[: max(1, int(max_results))]
