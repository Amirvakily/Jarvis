from __future__ import annotations

import pytest

from jarvis._http import httpx
from jarvis.skills.base import SkillError
from jarvis.skills.builtin.web import web_fetch, web_scrape, web_search
from jarvis.skills.htmltext import extract_text, select

PAGE = """<html><head><title> Hello  Page </title><style>p{}</style>
<script>var secret = 1;</script></head>
<body><nav>menu</nav><main><h1>Heading</h1><p>First   para.</p><p>Second.</p>
<a href="/about">About us</a> <a href="https://x.test/y">Ext</a>
<img src="img/a.png" alt="pic">
<div class="note">Note one</div><div class="note big">Note two</div><span id="u">Uniq</span>
<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table></main>
<footer>foot</footer></body></html>"""

DDG = """<html><body>
<div class="result"><h2><a class="result__a"
 href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fpage%3Fa%3D1&amp;rut=abc">
 Example <b>Org</b></a></h2>
<a class="result__snippet" href="x">A <b>great</b> snippet.</a></div>
<div class="result"><a class="result__a" href="https://direct.test/">Direct</a>
<div class="result__snippet">Plain snippet</div></div>
</body></html>"""


def _ctx_with(ctx, handler):
    ctx.services["http_transport"] = httpx.MockTransport(handler)
    return ctx


def html_handler(request):
    return httpx.Response(200, text=PAGE, headers={"content-type": "text/html; charset=utf-8"})


async def test_fetch_html_extracts_text(ctx):
    _ctx_with(ctx, html_handler)
    result = await web_fetch("https://site.test/", ctx=ctx)
    assert result["title"] == "Hello Page"
    assert result["status"] == 200
    assert "var secret" not in result["text"]
    assert "p{}" not in result["text"]
    assert "Heading" in result["text"] and "First para." in result["text"]
    assert "menu" not in result["text"]  # <main> preferred
    assert result["truncated"] is False


def test_extract_text_without_main():
    title, text = extract_text("<title>T</title><body><p>a</p><p>b</p><script>x</script>")
    assert title == "T" and text == "a\n\nb"


async def test_fetch_json_and_binary(ctx):
    def handler(request):
        if request.url.path == "/data":
            return httpx.Response(200, json={"a": 1})
        return httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"})

    _ctx_with(ctx, handler)
    data = await web_fetch("https://site.test/data", ctx=ctx)
    assert '"a"' in data["text"] and data["content_type"] == "application/json"
    with pytest.raises(SkillError, match="binary"):
        await web_fetch("https://site.test/img", ctx=ctx)


async def test_fetch_truncates(ctx):
    _ctx_with(
        ctx,
        lambda r: httpx.Response(200, text="x" * 500, headers={"content-type": "text/plain"}),
    )
    result = await web_fetch("https://site.test/", max_chars=100, ctx=ctx)
    assert len(result["text"]) == 100 and result["truncated"] is True


async def test_fetch_caps_body_size(ctx):
    big = b"a" * 2_500_000
    _ctx_with(
        ctx, lambda r: httpx.Response(200, content=big, headers={"content-type": "text/plain"})
    )
    result = await web_fetch("https://site.test/", max_chars=10_000_000, ctx=ctx)
    assert len(result["text"]) == 2_000_000 and result["truncated"] is True


@pytest.mark.parametrize("target", ["http://127.0.0.1/", "http://169.254.169.254/latest"])
async def test_redirect_to_private_is_refused(ctx, target):
    def handler(request):
        if request.url.host == "public.test":
            return httpx.Response(302, headers={"location": target})
        raise AssertionError("private host must never be requested")

    _ctx_with(ctx, handler)
    with pytest.raises(SkillError, match="private"):
        await web_fetch("https://public.test/", ctx=ctx)


async def test_redirect_followed_and_loop_limited(ctx):
    def handler(request):
        if request.url.path == "/loop":
            return httpx.Response(302, headers={"location": "/loop"})
        if request.url.path == "/start":
            return httpx.Response(301, headers={"location": "/end"})
        return httpx.Response(200, text="done", headers={"content-type": "text/plain"})

    _ctx_with(ctx, handler)
    result = await web_fetch("https://site.test/start", ctx=ctx)
    assert result["url"] == "https://site.test/end" and result["text"] == "done"
    with pytest.raises(SkillError, match="redirects"):
        await web_fetch("https://site.test/loop", ctx=ctx)


async def test_localhost_blocked_unless_allowed(ctx):
    _ctx_with(ctx, html_handler)
    with pytest.raises(SkillError, match="private"):
        await web_fetch("http://localhost/page", ctx=ctx)
    ctx.config.set("web.allow_private_hosts", True)
    result = await web_fetch("http://localhost/page", ctx=ctx)
    assert result["status"] == 200


async def test_scrape_links_absolute(ctx):
    _ctx_with(ctx, html_handler)
    result = await web_scrape("https://site.test/dir/page", extract="links", ctx=ctx)
    assert result["links"] == [
        {"text": "About us", "url": "https://site.test/about"},
        {"text": "Ext", "url": "https://x.test/y"},
    ]
    assert "headings" not in result


async def test_scrape_all_and_limit(ctx):
    _ctx_with(ctx, html_handler)
    result = await web_scrape("https://site.test/dir/page", limit=1, ctx=ctx)
    assert len(result["links"]) == 1
    assert result["headings"] == [{"level": 1, "text": "Heading"}]
    assert result["tables"] == [[["A", "B"], ["1", "2"]]]
    assert result["images"] == [{"src": "https://site.test/dir/img/a.png", "alt": "pic"}]


async def test_scrape_selector(ctx):
    _ctx_with(ctx, html_handler)
    result = await web_scrape("https://site.test/", selector=".note", ctx=ctx)
    assert result["matches"] == ["Note one", "Note two"]
    result = await web_scrape("https://site.test/", selector="span#u", ctx=ctx)
    assert result["matches"] == ["Uniq"]
    with pytest.raises(SkillError):
        await web_scrape("https://site.test/", selector="div > p", ctx=ctx)


def test_select_rejects_complex():
    with pytest.raises(ValueError):
        select(PAGE, "div p")
    assert select(PAGE, "h1") == ["Heading"]


async def test_search_parses_results(ctx):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, text=DDG, headers={"content-type": "text/html"})

    _ctx_with(ctx, handler)
    results = await web_search("jarvis ai", ctx=ctx)
    assert "html.duckduckgo.com/html/" in seen["url"] and "q=jarvis+ai" in seen["url"]
    assert results[0] == {
        "title": "Example Org",
        "url": "https://example.org/page?a=1",
        "snippet": "A great snippet.",
    }
    assert results[1]["url"] == "https://direct.test/"
    assert results[1]["snippet"] == "Plain snippet"
    assert len(await web_search("jarvis", max_results=1, ctx=ctx)) == 1


async def test_search_unavailable(ctx):
    _ctx_with(ctx, lambda r: httpx.Response(200, text="<html></html>"))
    with pytest.raises(SkillError, match="unavailable"):
        await web_search("nothing", ctx=ctx)
    _ctx_with(ctx, lambda r: httpx.Response(503, text="no"))
    with pytest.raises(SkillError, match="unavailable"):
        await web_search("nothing", ctx=ctx)
