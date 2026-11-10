from __future__ import annotations

import pytest

from jarvis._http import httpx
from jarvis.skills.base import SkillError
from jarvis.skills.builtin.security import security_scan

HARDENED = {
    "content-type": "text/html",
    "strict-transport-security": "max-age=31536000",
    "content-security-policy": "default-src 'self'; frame-ancestors 'none'",
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "permissions-policy": "camera=()",
}
SECRET = "DB_PASSWORD=hunter2-supersecret"


def make_site(headers, body="<html><body>hi</body></html>", extra=None, cookies=()):
    extra = extra or {}

    def handler(request):
        if request.url.scheme == "http" and request.url.path == "/":
            return httpx.Response(301, headers={"location": f"https://{request.url.host}/"})
        if request.url.path in extra:
            return extra[request.url.path]
        if request.url.path == "/":
            h = httpx.Headers(headers)
            resp = httpx.Response(200, text=body, headers=h)
            return resp
        return httpx.Response(404, text="nope")

    return handler


def scan_ctx(ctx, handler, targets=("example.com",)):
    ctx.config.set("security.authorized_targets", list(targets))
    ctx.services["http_transport"] = httpx.MockTransport(handler)
    return ctx


def ids(result):
    return {f["id"] for f in result["findings"]}


async def test_unauthorised_host_refused(ctx):
    ctx.services["http_transport"] = httpx.MockTransport(make_site(HARDENED))
    with pytest.raises(SkillError, match="authorized_targets"):
        await security_scan("https://example.com", ctx=ctx)


async def test_exact_and_wildcard_targets(ctx):
    scan_ctx(ctx, make_site(HARDENED), targets=["*.example.com"])
    result = await security_scan("https://www.example.com", ctx=ctx)
    assert result["target"] == "https://www.example.com"
    with pytest.raises(SkillError):
        await security_scan("https://example.com", ctx=ctx)
    with pytest.raises(SkillError):
        await security_scan("https://evilexample.com", ctx=ctx)
    scan_ctx(ctx, make_site(HARDENED), targets=["example.com"])
    assert (await security_scan("example.com", ctx=ctx))["grade"] == "A"


async def test_loopback_always_allowed(ctx):
    ctx.services["http_transport"] = httpx.MockTransport(
        lambda r: httpx.Response(200, text="hi", headers={"content-type": "text/html"})
    )
    result = await security_scan("http://localhost:8080/", ctx=ctx)
    assert "no-https" in ids(result)


async def test_bare_site_findings(ctx):
    scan_ctx(ctx, make_site({"content-type": "text/html", "server": "nginx/1.18.0"}))
    result = await security_scan("https://example.com", ctx=ctx)
    found = ids(result)
    assert {
        "missing-hsts", "missing-csp", "missing-nosniff", "missing-clickjacking",
        "missing-referrer-policy", "missing-permissions-policy", "version-disclosure",
        "no-security-txt",
    } <= found  # fmt: skip
    assert result["grade"] in ("D", "F")
    assert result["note"].startswith("Passive checks only")
    severities = [f["severity"] for f in result["findings"]]
    order = ["high", "medium", "low", "info"]
    assert severities == sorted(severities, key=order.index)
    assert sum(result["summary"].values()) == len(result["findings"])


async def test_hardened_beats_bare(ctx):
    scan_ctx(ctx, make_site(HARDENED, extra={"/.well-known/security.txt": httpx.Response(200)}))
    good = await security_scan("https://example.com", ctx=ctx)
    scan_ctx(ctx, make_site({"content-type": "text/html"}))
    bare = await security_scan("https://example.com", ctx=ctx)
    assert good["grade"] == "A" and bare["grade"] > good["grade"]
    assert "security-txt" in ids(good)
    assert good["summary"]["high"] == 0 and good["summary"]["medium"] == 0


async def test_cookies_cors_and_mixed_content(ctx):
    def handler(request):
        if request.url.scheme == "http":
            return httpx.Response(200, text="plain")
        if request.url.path != "/":
            return httpx.Response(404)
        return httpx.Response(
            200,
            headers=[
                ("content-type", "text/html"),
                ("set-cookie", "sid=1; Path=/"),
                ("set-cookie", "ok=1; Secure; HttpOnly; SameSite=Lax"),
                ("access-control-allow-origin", "*"),
                ("access-control-allow-credentials", "true"),
                ("x-powered-by", "PHP/7.4"),
            ],
            text='<script src="http://cdn.test/a.js"></script><form action="http://x.test/">',
        )

    scan_ctx(ctx, handler)
    result = await security_scan("https://example.com", ctx=ctx)
    found = ids(result)
    assert {"cookie-no-secure", "cookie-no-httponly", "cookie-no-samesite"} <= found
    assert {"cors-wildcard-credentials", "mixed-content", "no-http-redirect"} <= found
    cookie = next(f for f in result["findings"] if f["id"] == "cookie-no-secure")
    assert (
        "sid" in cookie["detail"]
        and "ok=" not in cookie["detail"]
        and "Cookies sid " in cookie["detail"]
    )
    mixed = next(f for f in result["findings"] if f["id"] == "mixed-content")
    assert "cdn.test" in mixed["detail"]


async def test_exposed_git_and_server_status(ctx):
    extra = {
        "/.git/HEAD": httpx.Response(200, text="ref: refs/heads/main\n"),
        "/server-status": httpx.Response(200, text="<h1>Apache Server Status for x</h1>"),
    }
    scan_ctx(ctx, make_site(HARDENED, extra=extra))
    result = await security_scan("https://example.com", ctx=ctx)
    assert {"exposed-git", "exposed-server-status"} <= ids(result)
    assert result["summary"]["high"] >= 1


async def test_env_not_leaked_and_html_ignored(ctx):
    scan_ctx(
        ctx,
        make_site(
            HARDENED,
            extra={
                "/.env": httpx.Response(200, text=SECRET, headers={"content-type": "text/plain"})
            },
        ),
    )
    result = await security_scan("https://example.com", ctx=ctx)
    assert "exposed-env" in ids(result)
    assert "hunter2" not in repr(result) and "DB_PASSWORD" not in repr(result)

    html_404 = httpx.Response(
        200, text="<html>KEY=VALUE</html>", headers={"content-type": "text/html"}
    )
    scan_ctx(ctx, make_site(HARDENED, extra={"/.env": html_404, "/.git/HEAD": html_404}))
    result = await security_scan("https://example.com", ctx=ctx)
    assert not ({"exposed-env", "exposed-git"} & ids(result))


async def test_offsite_redirect_not_followed(ctx):
    def handler(request):
        if request.url.host == "example.com" and request.url.scheme == "https":
            return httpx.Response(302, headers={"location": "https://other.test/"})
        if request.url.host == "other.test":
            raise AssertionError("must not contact unauthorised host")
        return httpx.Response(404)

    scan_ctx(ctx, handler)
    result = await security_scan("https://example.com", ctx=ctx)
    assert "redirect-offsite" in ids(result)
