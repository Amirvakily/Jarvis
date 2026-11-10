"""Markdown rendering, slugs, front matter and the draft-to-publish skills."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import pytest

from jarvis._http import httpx
from jarvis.skills.base import SkillError, build_schema
from jarvis.skills.builtin.content import (
    format_article,
    list_drafts,
    publish_article,
    read_draft,
    save_draft,
)
from jarvis.skills.markdown import (
    join_front_matter,
    render_page,
    slugify,
    split_front_matter,
    to_html,
)

LONG_BODY = (
    "## Intro\n\n"
    + "\n\n".join([" ".join(["word"] * 100)] * 4)
    + "\n\n## Details\n\n"
    + " ".join(["more"] * 40)
)


# -- markdown ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("# One", "<h1>One</h1>"),
        ("### Three", "<h3>Three</h3>"),
        ("###### Six", "<h6>Six</h6>"),
        ("## Closed ##", "<h2>Closed</h2>"),
        ("#hashtag", "<p>#hashtag</p>"),
        ("---", "<hr>"),
        ("- - -", "<hr>"),
    ],
)
def test_block_constructs(source, expected):
    assert to_html(source) == expected


def test_paragraphs_join_lines_and_split_on_blank_lines():
    assert to_html("a\nb\n\nc") == "<p>a b</p>\n<p>c</p>"


def test_inline_emphasis_and_code():
    out = to_html("Hello **bold** and *it* and `code < x`.")
    assert out == "<p>Hello <strong>bold</strong> and <em>it</em> and <code>code &lt; x</code>.</p>"


def test_fenced_code_with_and_without_language():
    assert (
        to_html("```python\nx = 1 < 2\n```")
        == '<pre><code class="language-python">x = 1 &lt; 2</code></pre>'
    )
    assert to_html("```\n<b>raw</b>\n```") == "<pre><code>&lt;b&gt;raw&lt;/b&gt;</code></pre>"


def test_fenced_code_ignores_markdown_inside():
    assert "<strong>" not in to_html("```\n**not bold**\n```")


def test_links_and_images():
    out = to_html("[site](https://example.com) ![pic](/img/a.png)")
    assert '<a href="https://example.com" rel="noopener noreferrer">site</a>' in out
    assert '<img src="/img/a.png" alt="pic">' in out


def test_unsafe_link_keeps_text_without_anchor():
    out = to_html("[bad](javascript:alert(1))")
    assert out == "<p>bad</p>"


def test_unsafe_image_keeps_alt_text_only():
    out = to_html("![x](data:text/html,hi)")
    assert "<img" not in out and "data:" not in out and "x" in out


def test_raw_html_is_escaped_not_passed_through():
    out = to_html("<script>alert(1)</script>")
    assert "<script>" not in out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out


def test_text_is_escaped():
    assert to_html('Tom & "Jerry"') == "<p>Tom &amp; &quot;Jerry&quot;</p>"


def test_unordered_and_ordered_lists():
    assert to_html("- a\n- b") == "<ul>\n<li>a</li>\n<li>b</li>\n</ul>"
    assert to_html("1. x\n2. y") == "<ol>\n<li>x</li>\n<li>y</li>\n</ol>"


def test_blockquote():
    assert (
        to_html("> quoted **x**") == "<blockquote>\n<p>quoted <strong>x</strong></p>\n</blockquote>"
    )


def test_render_page_escapes_title_and_summary():
    page = render_page("A <b>& B", "<p>body</p>", summary='say "hi" <now>')
    assert page.startswith("<!doctype html>")
    assert "<title>A &lt;b&gt;&amp; B</title>" in page
    assert 'content="say &quot;hi&quot; &lt;now&gt;"' in page
    assert "<p>body</p>" in page


# -- slugs and front matter --------------------------------------------------


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Hello, World!", "hello-world"),
        ("  Crème Brûlée  ", "creme-brulee"),
        ("../etc/passwd", "etc-passwd"),
        ("!!!", "untitled"),
        ("", "untitled"),
        ("Ünïcode ✓ only", "unicode-only"),
    ],
)
def test_slugify(title, expected):
    assert slugify(title) == expected


def test_slugify_limits_length_and_stays_safe():
    slug = slugify("word " * 100)
    assert len(slug) <= 80
    assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug)


def test_front_matter_round_trip():
    meta = {"title": "Ready: set", "tags": ["ai", "voice"], "summary": "Short"}
    body = "\n# Ready: set\n\nHello\n"
    text = join_front_matter(meta, body)
    assert split_front_matter(text) == (meta, body)


def test_text_without_front_matter_is_returned_unchanged():
    assert split_front_matter("# Plain\n") == ({}, "# Plain\n")


# -- draft workflow ----------------------------------------------------------


async def test_save_list_read_format_publish_local(ctx):
    saved = save_draft(
        title="My First Post",
        body_markdown=LONG_BODY,
        tags=["ai", "voice"],
        ctx=ctx,
    )
    assert saved["slug"] == "my-first-post"
    assert saved["path"] == "drafts/my-first-post.md"
    assert saved["words"] > 300

    rows = list_drafts(ctx=ctx)
    assert [row["slug"] for row in rows] == ["my-first-post"]
    assert rows[0]["status"] == "draft"

    draft = read_draft("my-first-post", ctx=ctx)
    assert draft["title"] == "My First Post"
    assert draft["tags"] == ["ai", "voice"]
    assert "## Intro" in draft["body_markdown"]

    formatted = format_article("my-first-post", ctx=ctx)
    assert formatted["headings"][0] == {"level": 1, "text": "My First Post"}
    assert {"level": 2, "text": "Intro"} in formatted["headings"]
    assert formatted["warnings"] == []
    assert formatted["reading_minutes"] == -(-formatted["words"] // 200)
    summary = read_draft("my-first-post", ctx=ctx)["summary"]
    assert summary.startswith("word word") and len(summary) <= 200
    assert read_draft("my-first-post", ctx=ctx)["body_markdown"].startswith("# My First Post\n")

    first = await publish_article(slug="my-first-post", target="local", ctx=ctx)
    assert first["target"] == "local"
    folder = Path(ctx.config.get("publish.output_dir"))
    assert Path(first["location"]) == folder / "my-first-post.html"
    assert (folder / "my-first-post.md").exists()
    assert (folder / "my-first-post.html").read_text(encoding="utf-8").startswith("<!doctype")
    index = json.loads((folder / "index.json").read_text(encoding="utf-8"))
    assert [entry["slug"] for entry in index] == ["my-first-post"]
    assert index[0]["tags"] == ["ai", "voice"]
    assert read_draft("my-first-post", ctx=ctx)["status"] == "published"
    assert not (folder / "my-first-post.md").read_text(encoding="utf-8").startswith("---")

    await publish_article(slug="my-first-post", target="local", ctx=ctx)
    index = json.loads((folder / "index.json").read_text(encoding="utf-8"))
    assert len(index) == 1


def test_format_demotes_extra_h1_and_collapses_blank_lines(ctx):
    save_draft(
        title="Two Titles",
        body_markdown="# Old heading\n\n\n\n\nText here.\n\n# Two Titles\n\n```\n# not a heading\n```",
        ctx=ctx,
    )
    format_article("two-titles", ctx=ctx)
    body = read_draft("two-titles", ctx=ctx)["body_markdown"]
    assert body.startswith("# Two Titles\n")
    assert "## Old heading" in body
    assert "\n\n\n\n" not in body
    assert "# not a heading" in body


def test_format_warns_about_short_flat_article_with_long_paragraph(ctx):
    save_draft(title="Short Note", body_markdown=" ".join(["word"] * 200), ctx=ctx)
    warnings = format_article("short-note", ctx=ctx)["warnings"]
    assert any("Only" in w and "words" in w for w in warnings)
    assert any("No subheadings" in w for w in warnings)
    assert any("paragraph of 200 words" in w for w in warnings)


def test_save_draft_overwrites_same_slug(ctx):
    save_draft(title="Same Title", body_markdown="first version", ctx=ctx)
    save_draft(title="Same Title", body_markdown="second version", ctx=ctx)
    assert [row["slug"] for row in list_drafts(ctx=ctx)] == ["same-title"]
    assert "second version" in read_draft("same-title", ctx=ctx)["body_markdown"]


def test_list_drafts_empty_without_folder(ctx):
    assert list_drafts(ctx=ctx) == []


@pytest.mark.parametrize("bad", ["../x", "Sub/Dir", "Has Space", "", "a/../../b"])
def test_invalid_slug_rejected(ctx, bad):
    with pytest.raises(SkillError, match="not a valid draft slug"):
        read_draft(bad, ctx=ctx)


async def test_invalid_slug_rejected_by_publish(ctx):
    with pytest.raises(SkillError, match="not a valid draft slug"):
        await publish_article(slug="../x", ctx=ctx)


def test_unknown_slug_lists_available(ctx):
    save_draft(title="Exists", body_markdown="text", ctx=ctx)
    with pytest.raises(SkillError, match="Available drafts: exists"):
        read_draft("missing", ctx=ctx)


async def test_webhook_publish_posts_json(ctx):
    save_draft(title="Hook Post", body_markdown="## Part\n\nSome **text**.", tags=["x"], ctx=ctx)
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True})

    ctx.config.set("publish.webhook_url", "https://hooks.example.test/jarvis")
    ctx.services["http_transport"] = httpx.MockTransport(handler)
    result = await publish_article(slug="hook-post", target="webhook", ctx=ctx)

    assert seen["url"] == "https://hooks.example.test/jarvis"
    assert seen["body"]["title"] == "Hook Post"
    assert seen["body"]["slug"] == "hook-post"
    assert seen["body"]["tags"] == ["x"]
    assert "**text**" in seen["body"]["markdown"]
    assert "<h2>Part</h2>" in seen["body"]["html"]
    assert result["location"] == "https://hooks.example.test/jarvis"
    assert read_draft("hook-post", ctx=ctx)["status"] == "published"


async def test_webhook_not_configured_raises(ctx):
    save_draft(title="No Hook", body_markdown="text", ctx=ctx)
    with pytest.raises(SkillError, match="publish.webhook_url"):
        await publish_article(slug="no-hook", target="webhook", ctx=ctx)


async def test_webhook_http_error_raises_with_status(ctx):
    save_draft(title="Failing Hook", body_markdown="text", ctx=ctx)
    ctx.config.set("publish.webhook_url", "https://hooks.example.test/fail")
    ctx.services["http_transport"] = httpx.MockTransport(lambda r: httpx.Response(500))
    with pytest.raises(SkillError, match="HTTP 500"):
        await publish_article(slug="failing-hook", target="webhook", ctx=ctx)
    assert read_draft("failing-hook", ctx=ctx)["status"] == "draft"


async def test_wordpress_missing_credentials_raises(ctx, monkeypatch):
    save_draft(title="Blog Post", body_markdown="text", ctx=ctx)
    monkeypatch.delenv("WORDPRESS_APP_PASSWORD", raising=False)
    ctx.config.set("publish.wordpress_url", "https://blog.example.test")
    with pytest.raises(SkillError) as info:
        await publish_article(slug="blog-post", target="wordpress", ctx=ctx)
    message = str(info.value)
    assert "publish.wordpress_user" in message
    assert "WORDPRESS_APP_PASSWORD" in message
    assert "publish.wordpress_url" not in message


async def test_wordpress_success_creates_post(ctx, monkeypatch):
    save_draft(title="Blog Post", body_markdown="## Heading\n\nBody.", summary="Sum", ctx=ctx)
    monkeypatch.setenv("WORDPRESS_APP_PASSWORD", "app-pass")
    ctx.config.set("publish.wordpress_url", "https://blog.example.test/")
    ctx.config.set("publish.wordpress_user", "editor")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": 7, "link": "https://blog.example.test/?p=7"})

    ctx.services["http_transport"] = httpx.MockTransport(handler)
    result = await publish_article(slug="blog-post", target="wordpress", ctx=ctx)

    assert seen["url"] == "https://blog.example.test/wp-json/wp/v2/posts"
    expected_auth = base64.b64encode(b"editor:app-pass").decode()
    assert seen["auth"] == f"Basic {expected_auth}"
    assert seen["body"]["title"] == "Blog Post"
    assert seen["body"]["status"] == "publish"
    assert seen["body"]["excerpt"] == "Sum"
    assert "<h2>Heading</h2>" in seen["body"]["content"]
    assert result == {
        "slug": "blog-post",
        "target": "wordpress",
        "location": "https://blog.example.test/?p=7",
    }


def test_publish_schema_exposes_targets_and_hides_ctx():
    _, parameters, wants_context = build_schema(publish_article)
    assert parameters["properties"]["target"]["enum"] == ["local", "webhook", "wordpress"]
    assert "ctx" not in parameters["properties"]
    assert wants_context
