"""Passive website security posture check for sites the user is authorised to assess."""

from __future__ import annotations

import asyncio
import re
import socket
import ssl
import time
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import urlsplit

from jarvis.skills.base import SkillContext, SkillError, skill
from jarvis.skills.builtin.web import safe_get
from jarvis.skills.util import check_url

_LOOPBACK = {"localhost", "127.0.0.1", "::1"}
_REDIRECTS = {301, 302, 303, 307, 308}
_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}
_WEIGHT = {"high": 10, "medium": 4, "low": 1, "info": 0}
_ENV_LINE = re.compile(r"^(export\s+)?[A-Za-z_][A-Za-z0-9_.]*\s*=")
_VERSIONED = re.compile(r"\d+\.\d+|/\d")


def _is_authorized(host: str, patterns: list[str]) -> bool:
    host = host.lower().rstrip(".")
    if host in _LOOPBACK:
        return True
    for raw in patterns:
        pattern = str(raw).strip().lower().rstrip(".")
        if pattern.startswith("*."):
            if host.endswith(pattern[1:]) and host != pattern[2:]:
                return True
        elif pattern and host == pattern:
            return True
    return False


def _finding(fid: str, severity: str, title: str, detail: str, recommendation: str = "") -> dict:
    return {
        "id": fid,
        "severity": severity,
        "title": title,
        "detail": detail,
        "recommendation": recommendation,
    }


class _ResourceParser(HTMLParser):
    """Collects subresource and form URLs that matter for mixed-content checks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.insecure: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        a = {k: v or "" for k, v in attrs}
        key = {"script": "src", "img": "src", "iframe": "src", "link": "href", "form": "action"}
        attr = key.get(tag)
        if attr and a.get(attr, "").lower().startswith("http://"):
            self.insecure.append(f"{tag} {a[attr][:80]}")


def _probe_tls(host: str, port: int) -> dict:
    context = ssl.create_default_context()
    with socket.create_connection((host, port), timeout=8) as sock:
        with context.wrap_socket(sock, server_hostname=host) as tls:
            cert = tls.getpeercert() or {}
            return {"version": tls.version(), "not_after": cert.get("notAfter")}


async def _tls_findings(host: str, port: int) -> list[dict]:
    try:
        info = await asyncio.to_thread(_probe_tls, host, port)
    except ssl.SSLCertVerificationError as exc:
        return [
            _finding(
                "tls-cert-invalid",
                "high",
                "TLS certificate failed validation",
                str(exc.verify_message or exc),
                "Install a certificate from a trusted CA that matches the hostname.",
            )  # fmt: skip
        ]
    except (OSError, ssl.SSLError) as exc:
        return [_finding("tls-probe-failed", "info", "TLS probe failed", str(exc))]
    out: list[dict] = []
    version = info["version"] or "unknown"
    if version in ("TLSv1", "TLSv1.1", "SSLv3"):
        out.append(
            _finding(
                "tls-old-protocol",
                "medium",
                "Outdated TLS protocol negotiated",
                version,
                "Disable TLS 1.0/1.1 and require TLS 1.2 or newer.",
            )  # fmt: skip
        )
    if info["not_after"]:
        days = int((ssl.cert_time_to_seconds(info["not_after"]) - time.time()) // 86400)
        if days < 0:
            out.append(
                _finding("tls-cert-expired", "high", "TLS certificate has expired",
                         f"Expired {-days} days ago.", "Renew the certificate.")
            )  # fmt: skip
        elif days < 14:
            out.append(
                _finding("tls-cert-expiring", "high", "TLS certificate expires very soon",
                         f"{days} days left.", "Renew the certificate now.")
            )  # fmt: skip
        elif days < 30:
            out.append(
                _finding("tls-cert-expiring", "medium", "TLS certificate expires soon",
                         f"{days} days left.", "Renew the certificate and automate renewal.")
            )  # fmt: skip
        else:
            out.append(
                _finding("tls-ok", "info", "TLS certificate and protocol",
                         f"{version}, certificate valid for {days} more days.")
            )  # fmt: skip
    return out


def _header_findings(headers, https: bool) -> list[dict]:
    out: list[dict] = []
    csp = headers.get("content-security-policy", "")
    if https and not headers.get("strict-transport-security"):
        out.append(
            _finding("missing-hsts", "medium", "Strict-Transport-Security header missing",
                     "Browsers are not told to insist on HTTPS.",
                     "Send 'Strict-Transport-Security: max-age=31536000; includeSubDomains'.")
        )  # fmt: skip
    if not csp:
        out.append(
            _finding("missing-csp", "medium", "Content-Security-Policy header missing",
                     "No policy restricts where scripts and other content may load from.",
                     "Define a Content-Security-Policy, starting with default-src 'self'.")
        )  # fmt: skip
    if headers.get("x-content-type-options", "").strip().lower() != "nosniff":
        out.append(
            _finding("missing-nosniff", "low", "X-Content-Type-Options: nosniff missing",
                     "Browsers may MIME-sniff responses.",
                     "Send 'X-Content-Type-Options: nosniff'.")
        )  # fmt: skip
    if not headers.get("x-frame-options") and "frame-ancestors" not in csp.lower():
        out.append(
            _finding("missing-clickjacking", "medium", "No clickjacking protection",
                     "Neither X-Frame-Options nor CSP frame-ancestors is set.",
                     "Send 'X-Frame-Options: DENY' or CSP 'frame-ancestors none'.")
        )  # fmt: skip
    if not headers.get("referrer-policy"):
        out.append(
            _finding("missing-referrer-policy", "low", "Referrer-Policy header missing",
                     "Full URLs may leak to other sites via the Referer header.",
                     "Send 'Referrer-Policy: strict-origin-when-cross-origin' or stricter.")
        )  # fmt: skip
    if not headers.get("permissions-policy"):
        out.append(
            _finding("missing-permissions-policy", "low", "Permissions-Policy header missing",
                     "Powerful browser features are not explicitly restricted.",
                     "Send a Permissions-Policy disabling features the site does not use.")
        )  # fmt: skip
    return out


def _cookie_findings(headers) -> list[dict]:
    missing: dict[str, list[str]] = {"secure": [], "httponly": [], "samesite": []}
    for cookie in headers.get_list("set-cookie"):
        parts = [p.strip() for p in cookie.split(";")]
        name = parts[0].split("=")[0].strip() or "?"
        attrs = {p.split("=")[0].strip().lower() for p in parts[1:]}
        for flag in missing:
            if flag not in attrs:
                missing[flag].append(name)
    out: list[dict] = []
    spec = {
        "secure": ("medium", "Secure", "may be sent over unencrypted connections"),
        "httponly": ("medium", "HttpOnly", "are readable by page scripts"),
        "samesite": ("low", "SameSite", "are sent on cross-site requests"),
    }
    for flag, names in missing.items():
        if names:
            severity, label, effect = spec[flag]
            out.append(
                _finding(f"cookie-no-{flag}", severity, f"Cookies missing the {label} attribute",
                         f"Cookies {', '.join(sorted(set(names)))} {effect}.",
                         f"Set the {label} attribute on these cookies.")
            )  # fmt: skip
    return out


def _disclosure_findings(headers) -> list[dict]:
    leaks = []
    server = headers.get("server", "")
    if server and _VERSIONED.search(server):
        leaks.append(f"Server: {server}")
    if headers.get("x-powered-by"):
        leaks.append(f"X-Powered-By: {headers['x-powered-by']}")
    if not leaks:
        return []
    return [
        _finding("version-disclosure", "low", "Software details disclosed in headers",
                 "; ".join(leaks), "Remove or genericise Server and X-Powered-By headers.")
    ]  # fmt: skip


def _cors_findings(headers) -> list[dict]:
    if (
        headers.get("access-control-allow-origin", "").strip() == "*"
        and headers.get("access-control-allow-credentials", "").strip().lower() == "true"
    ):
        return [
            _finding("cors-wildcard-credentials", "medium", "Permissive CORS with credentials",
                     "Access-Control-Allow-Origin is * while credentials are allowed.",
                     "Reflect only specific trusted origins when allowing credentials.")
        ]  # fmt: skip
    return []


def _grade(findings: list[dict]) -> str:
    score = sum(_WEIGHT[f["severity"]] for f in findings)
    for limit, letter in ((0, "A"), (3, "B"), (8, "C"), (15, "D")):
        if score <= limit:
            return letter
    return "F"


async def _probe(ctx: SkillContext, url: str):
    try:
        return await safe_get(
            ctx, url, allow_private=True, max_bytes=65_536, follow=lambda _: False
        )
    except SkillError:
        return None


@skill(tags=["security"], timeout=180.0)
async def security_scan(url: str, ctx: SkillContext = None) -> dict:
    """Run a passive, non-intrusive security check of a website you own or may assess.

    Checks HTTPS, TLS certificate, security headers, cookies and a few commonly
    exposed files using ordinary GET requests only. Refuses hosts that are not authorised.

    Args:
        url: The site to check, for example https://example.com.
    """
    if "://" not in url:
        url = "https://" + url
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme not in ("http", "https") or not host:
        raise SkillError(f"'{url}' is not a valid http(s) URL.")
    allowed = ctx.config.get("security.authorized_targets", []) or []
    if isinstance(allowed, str):
        allowed = [allowed]
    if not _is_authorized(host, list(allowed)):
        raise SkillError(
            f"I only run security scans against sites you are authorised to test, and '{host}' "
            "is not on that list. If you own it or have permission, add it to jarvis.toml under "
            f'[security] authorized_targets = ["{host}"] (wildcards like "*.example.com" work).'
        )
    url = await check_url(ctx, url, allow_private=True)

    findings: list[dict] = []
    skipped: list[str] = []

    def guard(target: str) -> bool:
        target_host = urlsplit(target).hostname or ""
        if _is_authorized(target_host, list(allowed)):
            return True
        skipped.append(target_host)
        return False

    final, status, headers, body = await safe_get(
        ctx, url, allow_private=True, max_bytes=500_000, follow=guard
    )
    if skipped:
        findings.append(
            _finding("redirect-offsite", "info", "Redirect to a host outside the scan scope",
                     f"The site redirects to {skipped[0]}, which is not authorised, so it was "
                     "not followed.")
        )  # fmt: skip
    final_parts = urlsplit(final)
    https = final_parts.scheme == "https"
    origin = f"{final_parts.scheme}://{final_parts.netloc}"

    if not https:
        findings.append(
            _finding("no-https", "high", "Site served over plain HTTP",
                     f"{final} is not encrypted.", "Serve the site over HTTPS only.")
        )  # fmt: skip
    elif final_parts.port in (None, 443):
        bracketed = f"[{host}]" if ":" in host else host
        plain = await _probe(ctx, f"http://{bracketed}/")
        if plain is None:
            findings.append(_finding("http-unreachable", "info", "Plain HTTP not reachable", ""))
        elif plain[1] in _REDIRECTS and plain[2].get("location", "").startswith("https://"):
            findings.append(_finding("http-redirects", "info", "HTTP redirects to HTTPS", ""))
        else:
            findings.append(
                _finding("no-http-redirect", "medium", "HTTP does not redirect to HTTPS",
                         "Requests to http:// are answered without redirecting to https://.",
                         "Redirect all HTTP traffic to HTTPS.")
            )  # fmt: skip

    if https and "http_transport" not in ctx.services:
        findings += await _tls_findings(final_parts.hostname or host, final_parts.port or 443)

    findings += _header_findings(headers, https)
    findings += _cookie_findings(headers)
    findings += _disclosure_findings(headers)
    findings += _cors_findings(headers)

    if https and "html" in headers.get("content-type", "").lower():
        resources = _ResourceParser()
        resources.feed(body.decode("utf-8", errors="replace"))
        if resources.insecure:
            findings.append(
                _finding("mixed-content", "medium", "Mixed content on an HTTPS page",
                         "Insecure references: " + "; ".join(resources.insecure[:5]),
                         "Load all scripts, images, styles and form targets over HTTPS.")
            )  # fmt: skip

    sec = await _probe(ctx, f"{origin}/.well-known/security.txt")
    if sec and sec[1] == 200:
        findings.append(_finding("security-txt", "info", "security.txt present", ""))
    else:
        findings.append(
            _finding("no-security-txt", "info", "No security.txt found",
                     "/.well-known/security.txt is missing.",
                     "Publish a security.txt so researchers know how to report issues.")
        )  # fmt: skip

    git = await _probe(ctx, f"{origin}/.git/HEAD")
    if git and git[1] == 200 and git[3].lstrip().startswith(b"ref:"):
        findings.append(
            _finding("exposed-git", "high", "Git repository exposed",
                     "/.git/HEAD is publicly readable, which can leak source code and secrets.",
                     "Block access to /.git and remove it from the web root.")
        )  # fmt: skip
    env = await _probe(ctx, f"{origin}/.env")
    if env and env[1] == 200 and _looks_like_env(env[2].get("content-type", ""), env[3]):
        findings.append(
            _finding("exposed-env", "high", "Environment file exposed",
                     "/.env is publicly readable and appears to contain configuration values.",
                     "Block access to dotfiles, move secrets out of the web root and rotate them.")
        )  # fmt: skip
    status_page = await _probe(ctx, f"{origin}/server-status")
    if status_page and status_page[1] == 200 and b"Apache Server Status" in status_page[3]:
        findings.append(
            _finding("exposed-server-status", "medium", "Apache server-status page exposed",
                     "/server-status is publicly readable.",
                     "Restrict mod_status to trusted addresses.")
        )  # fmt: skip

    findings.sort(key=lambda f: _ORDER[f["severity"]])
    summary = {level: sum(f["severity"] == level for f in findings) for level in _ORDER}
    return {
        "target": final,
        "scanned_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "http_status": status,
        "grade": _grade(findings),
        "summary": summary,
        "findings": findings,
        "note": "Passive checks only; not a substitute for a full penetration test.",
    }


def _looks_like_env(content_type: str, body: bytes) -> bool:
    if "html" in content_type.lower():
        return False
    text = body.decode("utf-8", errors="replace").strip()
    if not text or text.startswith("<"):
        return False
    lines = [
        ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")
    ]
    return bool(lines) and all(_ENV_LINE.match(ln) for ln in lines[:20])
