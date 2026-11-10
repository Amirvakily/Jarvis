"""Writing skills: save, review, tidy and publish Markdown article drafts.

Drafts live in ``<workspace>/drafts`` as Markdown files with a small front matter
header. Publishing writes to a local folder, posts JSON to a webhook, or creates
a WordPress post.
"""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from jarvis._http import httpx
from jarvis.skills.base import SkillContext, SkillError, skill
from jarvis.skills.markdown import (
    join_front_matter,
    render_page,
    slugify,
    split_front_matter,
    to_html,
)
from jarvis.skills.util import http_client

DRAFTS_DIR = "drafts"
SUMMARY_LIMIT = 200
MIN_WORDS = 300
MAX_PARAGRAPH_WORDS = 150
WORDS_PER_MINUTE = 200

_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)(?:\s+#+)?\s*$")
_H1 = re.compile(r"^#\s")
_STRUCTURAL = re.compile(r"^\s*(#{1,6}\s|[-*+]\s|\d+\.\s|>|\||([-*_]\s*){3,}$)")
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MARKUP = re.compile(r"[*_`]+")


# -- helpers -----------------------------------------------------------------


def _word_count(text: str) -> int:
    return len(_WORD.findall(text))


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def _clean_tags(tags: list[str] | None) -> list[str]:
    cleaned: list[str] = []
    for tag in tags or []:
        text = " ".join(str(tag).replace(",", " ").split())
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _check_slug(slug: str) -> str:
    if not isinstance(slug, str) or not slug or slugify(slug) != slug:
        raise SkillError(
            f"'{slug}' is not a valid draft slug. Use the slug from list_drafts "
            "(lowercase letters, digits and hyphens only)."
        )
    return slug


def _available(ctx: SkillContext) -> list[str]:
    folder = ctx.workspace / DRAFTS_DIR
    if not folder.is_dir():
        return []
    return sorted(path.stem for path in folder.glob("*.md"))


def _load(ctx: SkillContext, slug: str) -> tuple[dict, str, Path]:
    _check_slug(slug)
    path = ctx.workspace / DRAFTS_DIR / f"{slug}.md"
    if not path.is_file():
        available = ", ".join(_available(ctx)) or "none yet"
        raise SkillError(f"No draft with slug '{slug}'. Available drafts: {available}.")
    meta, body = split_front_matter(path.read_text(encoding="utf-8"))
    return meta, body, path


def _write(path: Path, meta: dict, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(join_front_matter(meta, body), encoding="utf-8")


def _code_flags(lines: list[str]) -> list[bool]:
    """True for every line that sits inside a fenced code block (fences included)."""
    flags: list[bool] = []
    fence: str | None = None
    for line in lines:
        match = _FENCE.match(line)
        if fence is None:
            flags.append(bool(match))
            if match:
                fence = match.group(1)
        else:
            flags.append(True)
            if match and line.strip() == match.group(1) and len(match.group(1)) >= len(fence):
                fence = None
    return flags


def _heading_text(line: str) -> str:
    match = _HEADING.match(line)
    return match.group(2) if match else ""


def _collapse_blank_runs(lines: list[str], flags: list[bool]) -> list[str]:
    """Reduce every run of three or more blank lines (outside code) to one."""
    out: list[str] = []
    run: list[str] = []
    for line, code in zip(lines, flags, strict=True):
        if line == "" and not code:
            run.append(line)
            continue
        out.extend(run[:1] if len(run) >= 3 else run)
        run = []
        out.append(line)
    out.extend(run[:1] if len(run) >= 3 else run)
    return out


def _normalise(title: str, body: str) -> str:
    """Trim whitespace, make the title the only H1, and return the new body."""
    lines = [line.rstrip() for line in body.split("\n")]
    flags = _code_flags(lines)
    h1_lines = [i for i, line in enumerate(lines) if not flags[i] and _H1.match(line)]
    wanted = _norm(title)
    keep = next((i for i in h1_lines if _norm(_heading_text(lines[i])) == wanted), None)
    for i in h1_lines:
        if i != keep:
            lines[i] = "#" + lines[i]
    heading = f"# {title}"
    if keep is not None:
        heading = lines.pop(keep)
        flags.pop(keep)
    lines = _collapse_blank_runs(lines, flags)
    rest = "\n".join(lines).strip("\n")
    return (f"{heading}\n\n{rest}" if rest else heading) + "\n"


def _headings(lines: list[str]) -> list[dict]:
    flags = _code_flags(lines)
    found: list[dict] = []
    for line, code in zip(lines, flags, strict=True):
        match = _HEADING.match(line)
        if match and not code:
            found.append({"level": len(match.group(1)), "text": match.group(2)})
    return found


def _paragraphs(lines: list[str]) -> list[str]:
    flags = _code_flags(lines)
    paragraphs: list[str] = []
    current: list[str] = []
    for line, code in zip(lines, flags, strict=True):
        if not code and line.strip() and not _STRUCTURAL.match(line):
            current.append(line.strip())
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def _summary_from(paragraph: str) -> str:
    plain = " ".join(_MARKUP.sub("", _LINK.sub(r"\1", paragraph)).split())
    if len(plain) <= SUMMARY_LIMIT:
        return plain
    cut = plain[:SUMMARY_LIMIT]
    return cut[: cut.rfind(" ")] if " " in cut else cut


def _read_index(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SkillError(f"{path} is not valid JSON; fix or remove it and try again.") from exc
    if not isinstance(entries, list):
        raise SkillError(f"{path} must contain a JSON list.")
    return entries


def _publish_local(ctx: SkillContext, slug: str, meta: dict, body: str) -> str:
    configured = str(ctx.config.get("publish.output_dir", "") or "").strip()
    if not configured:
        raise SkillError("publish.output_dir is not set, so there is nowhere to write.")
    out_dir = Path(configured).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    title = str(meta.get("title") or slug)
    summary = str(meta.get("summary", ""))
    html_path = out_dir / f"{slug}.html"
    (out_dir / f"{slug}.md").write_text(body, encoding="utf-8")
    html_path.write_text(render_page(title, to_html(body), summary), encoding="utf-8")

    index_path = out_dir / "index.json"
    entry = {
        "slug": slug,
        "title": title,
        "date": str(meta.get("date", "")),
        "summary": summary,
        "tags": list(meta.get("tags") or []),
    }
    entries = [e for e in _read_index(index_path) if e.get("slug") != slug] + [entry]
    entries.sort(key=lambda e: str(e.get("date", "")), reverse=True)
    index_path.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return str(html_path)


async def _publish_webhook(ctx: SkillContext, slug: str, meta: dict, body: str) -> str:
    url = str(ctx.config.get("publish.webhook_url", "") or "").strip()
    if not url:
        raise SkillError("No webhook is configured. Set publish.webhook_url first.")
    payload = {
        "title": str(meta.get("title") or slug),
        "slug": slug,
        "summary": str(meta.get("summary", "")),
        "tags": list(meta.get("tags") or []),
        "markdown": body,
        "html": to_html(body),
    }
    async with http_client(ctx) as client:
        try:
            response = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise SkillError(f"The webhook request failed: {exc}.") from exc
    if not 200 <= response.status_code < 300:
        raise SkillError(f"The webhook returned HTTP {response.status_code}.")
    return url


async def _publish_wordpress(ctx: SkillContext, slug: str, meta: dict, body: str) -> str:
    base = str(ctx.config.get("publish.wordpress_url", "") or "").strip().rstrip("/")
    user = str(ctx.config.get("publish.wordpress_user", "") or "").strip()
    env_name = str(ctx.config.get("publish.wordpress_password_env", "") or "").strip()
    password = ctx.config.secret(env_name) if env_name else None
    missing = []
    if not base:
        missing.append("publish.wordpress_url")
    if not user:
        missing.append("publish.wordpress_user")
    if not password:
        missing.append(f"the application password in the ${env_name} variable")
    if missing:
        raise SkillError("WordPress publishing is missing: " + ", ".join(missing) + ".")

    payload = {
        "title": str(meta.get("title") or slug),
        "content": to_html(body),
        "excerpt": str(meta.get("summary", "")),
        "status": "publish",
    }
    async with http_client(ctx) as client:
        try:
            response = await client.post(
                f"{base}/wp-json/wp/v2/posts",
                json=payload,
                auth=httpx.BasicAuth(user, password),
            )
        except httpx.HTTPError as exc:
            raise SkillError(f"The WordPress request failed: {exc}.") from exc
    if not 200 <= response.status_code < 300:
        raise SkillError(f"WordPress returned HTTP {response.status_code}.")
    try:
        data = response.json()
    except ValueError:
        data = {}
    link = data.get("link") if isinstance(data, dict) else None
    return str(link or "")


# -- skills ------------------------------------------------------------------


@skill(tags=["content"])
def save_draft(
    title: str,
    body_markdown: str,
    tags: list[str] | None = None,
    summary: str = "",
    *,
    ctx: SkillContext,
) -> dict:
    """Save an article draft as Markdown in the workspace. A draft with the same title is replaced.

    Use this to store a new article, or a revised version, before formatting or publishing it.

    Args:
        title: The article title. The draft's file name is derived from it.
        body_markdown: The article body in Markdown.
        tags: Optional short topic tags.
        summary: Optional one-sentence summary. If left empty, formatting fills it in.
    """
    clean_title = " ".join(str(title).split())
    if not clean_title:
        raise SkillError("A draft needs a title.")
    if not body_markdown.strip():
        raise SkillError("The draft body is empty.")
    slug = slugify(clean_title)
    body = body_markdown.strip("\n").rstrip() + "\n"
    meta = {
        "title": clean_title,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "tags": _clean_tags(tags),
        "summary": " ".join((summary or "").split()),
        "status": "draft",
    }
    path = ctx.workspace / DRAFTS_DIR / f"{slug}.md"
    _write(path, meta, body)
    return {
        "slug": slug,
        "path": path.relative_to(ctx.workspace).as_posix(),
        "words": _word_count(body),
    }


@skill(tags=["content"])
def list_drafts(*, ctx: SkillContext) -> list[dict]:
    """List saved article drafts, newest first.

    Use this to see which drafts exist before reading, formatting or publishing one.
    """
    folder = ctx.workspace / DRAFTS_DIR
    if not folder.is_dir():
        return []
    rows = []
    for path in folder.glob("*.md"):
        meta, body = split_front_matter(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "slug": path.stem,
                "title": str(meta.get("title") or path.stem),
                "status": str(meta.get("status") or "draft"),
                "date": str(meta.get("date", "")),
                "words": _word_count(body),
            }
        )
    return sorted(rows, key=lambda row: row["date"], reverse=True)


@skill(tags=["content"])
def read_draft(slug: str, *, ctx: SkillContext) -> dict:
    """Read a saved draft's metadata and Markdown body.

    Use this to review a draft's text before editing or publishing it.

    Args:
        slug: The draft's slug, as listed by list_drafts.
    """
    meta, body, _ = _load(ctx, slug)
    tags = meta.get("tags") or []
    return {
        "slug": slug,
        "title": str(meta.get("title") or slug),
        "tags": list(tags) if isinstance(tags, list) else [str(tags)],
        "summary": str(meta.get("summary", "")),
        "status": str(meta.get("status", "draft")),
        "body_markdown": body,
    }


@skill(tags=["content"])
def format_article(slug: str, *, ctx: SkillContext) -> dict:
    """Tidy a draft in place and check its readability.

    Makes the title the only H1, demotes other H1s to H2, trims whitespace, fills in a
    missing summary, and returns warnings about length, subheadings and long paragraphs.
    Use this after writing or revising a draft and before publishing it.

    Args:
        slug: The draft's slug, as listed by list_drafts.
    """
    meta, body, path = _load(ctx, slug)
    title = str(meta.get("title") or slug)
    text = _normalise(title, body)
    lines = text.split("\n")
    paragraphs = _paragraphs(lines)
    headings = _headings(lines)
    words = _word_count(text)

    warnings: list[str] = []
    if words < MIN_WORDS:
        warnings.append(f"Only {words} words; aim for at least {MIN_WORDS}.")
    if not any(heading["level"] >= 2 for heading in headings):
        warnings.append("No subheadings; add H2 sections so readers can skim.")
    for paragraph in paragraphs:
        count = _word_count(paragraph)
        if count > MAX_PARAGRAPH_WORDS:
            warnings.append(
                f"A paragraph of {count} words starts '{paragraph[:40]}...'; consider splitting it."
            )

    if not str(meta.get("summary", "")).strip() and paragraphs:
        meta["summary"] = _summary_from(paragraphs[0])
    _write(path, meta, text)
    return {
        "slug": slug,
        "words": words,
        "reading_minutes": math.ceil(words / WORDS_PER_MINUTE),
        "headings": headings,
        "warnings": warnings,
    }


@skill(tags=["content"])
async def publish_article(
    slug: str,
    target: Literal["local", "webhook", "wordpress"] = "local",
    *,
    ctx: SkillContext,
) -> dict:
    """Publish a draft locally, to a webhook, or to WordPress, and mark it published.

    Use this once the user has approved the article. Local writes Markdown, HTML and an
    index.json into the output folder. Webhook posts JSON to the configured URL. WordPress
    creates a published post.

    Args:
        slug: The draft's slug, as listed by list_drafts.
        target: Where to publish: local, webhook or wordpress.
    """
    meta, body, path = _load(ctx, slug)
    if target == "local":
        location = _publish_local(ctx, slug, meta, body)
    elif target == "webhook":
        location = await _publish_webhook(ctx, slug, meta, body)
    else:
        location = await _publish_wordpress(ctx, slug, meta, body)

    meta["status"] = "published"
    meta["published_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    meta["published_to"] = target
    _write(path, meta, body)
    return {"slug": slug, "target": target, "location": location}
