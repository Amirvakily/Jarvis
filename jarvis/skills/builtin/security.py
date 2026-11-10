"""Passive website security posture check for sites the user is authorised to assess."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import socket
import ssl
import time
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import urlsplit

from jarvis._http import httpx
from jarvis.skills.authorization import (
    concrete_hosts,
    is_authorized,
    load_targets,
    require_authorized,
)
from jarvis.skills.base import SkillContext, SkillError, skill
from jarvis.skills.builtin.web import safe_get
from jarvis.skills.util import check_url, http_client

_REDIRECTS = {301, 302, 303, 307, 308}
_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}
_WEIGHT = {"high": 10, "medium": 4, "low": 1, "info": 0}
_ENV_LINE = re.compile(r"^(export\s+)?[A-Za-z_][A-Za-z0-9_.]*\s*=")
_VERSIONED = re.compile(r"\d+\.\d+|/\d")


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
    require_authorized(ctx, host)
    allowed = load_targets(ctx.config)
    url = await check_url(ctx, url, allow_private=True)

    findings: list[dict] = []
    skipped: list[str] = []

    def guard(target: str) -> bool:
        target_host = urlsplit(target).hostname or ""
        if is_authorized(target_host, allowed):
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


# -- shared helpers for the checks below ---------------------------------------


def _host_arg(value: str) -> str:
    """Pull a bare host name out of a host, host:port or URL argument."""
    text = (value or "").strip()
    try:
        host = urlsplit(text if "://" in text else "//" + text).hostname
    except ValueError:
        host = None
    host = (host or "").lower().rstrip(".")
    if not host:
        raise SkillError(f"'{value}' is not a valid host name.")
    return host


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _summarise(findings: list[dict]) -> dict:
    return {level: sum(f["severity"] == level for f in findings) for level in _ORDER}


# -- DNS and email security ----------------------------------------------------

_DNS_TYPES = {"MX": 15, "TXT": 16, "CAA": 257}
_QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _txt_value(data: str) -> str:
    chunks = _QUOTED.findall(data)
    return "".join(chunks) if chunks else data.strip()


async def _doh(client, url: str, name: str, rtype: str) -> tuple[list[str], bool]:
    """One DNS-over-HTTPS JSON lookup. Returns (answer data strings, AD flag)."""
    try:
        response = await client.get(
            url, params={"name": name, "type": rtype}, headers={"accept": "application/dns-json"}
        )
    except httpx.HTTPError as exc:
        raise SkillError(f"DNS lookup of {name} failed: {exc or type(exc).__name__}") from exc
    if response.status_code != 200:
        raise SkillError(f"DNS resolver answered HTTP {response.status_code} for {name}.")
    try:
        data = response.json()
    except ValueError as exc:
        raise SkillError("DNS resolver returned something that is not JSON.") from exc
    if not isinstance(data, dict) or data.get("Status", 0) not in (0, 3):  # 3 = no such name
        raise SkillError(f"DNS lookup of {name} failed (status {data.get('Status')}).")
    wanted = _DNS_TYPES[rtype]
    answers = [
        str(a.get("data", ""))
        for a in data.get("Answer") or []
        if isinstance(a, dict) and a.get("type", wanted) == wanted
    ]
    return answers, bool(data.get("AD"))


def _spf_findings(spf: list[str], has_mx: bool) -> list[dict]:
    if not spf:
        return [
            _finding("spf-missing", "medium" if has_mx else "low", "No SPF record",
                     "Without SPF, anyone can send mail that claims to come from this domain "
                     "and receivers have no way to tell." if has_mx else
                     "The domain has no MX records but still has no SPF record saying it "
                     "sends no mail.",
                     "Publish a TXT record such as 'v=spf1 include:<your mail provider> -all' "
                     "(or 'v=spf1 -all' if the domain sends no mail)."),
        ]  # fmt: skip
    out: list[dict] = []
    if len(spf) > 1:
        out.append(
            _finding("spf-multiple", "medium", "More than one SPF record",
                     f"{len(spf)} records start with v=spf1. Receivers treat this as an error "
                     "and SPF fails.",
                     "Merge them into a single v=spf1 TXT record.")
        )  # fmt: skip
    record = spf[0]
    match = re.search(r"(?:^|\s)([+\-~?]?)all(?:\s|$)", record, flags=re.IGNORECASE)
    qualifier = None if match is None else (match.group(1) or "+")
    if qualifier == "+":
        out.append(
            _finding("spf-allow-all", "high", "SPF allows any sender (+all)",
                     "The record ends in +all, so every server on the internet is authorised "
                     "to send as this domain.",
                     "Change the ending to -all (or ~all while testing).")
        )  # fmt: skip
    elif qualifier == "?":
        out.append(
            _finding("spf-neutral", "medium", "SPF gives no verdict (?all)",
                     "The record ends in ?all, which tells receivers not to judge unlisted "
                     "senders.",
                     "End the record with -all once your legitimate senders are listed.")
        )  # fmt: skip
    elif qualifier == "~":
        out.append(
            _finding("spf-softfail", "low", "SPF uses soft fail (~all)",
                     "Unlisted senders are marked suspicious rather than rejected.",
                     "Move to -all once you are sure every legitimate sender is listed.")
        )  # fmt: skip
    elif qualifier == "-":
        out.append(_finding("spf-ok", "info", "SPF is strict (-all)", record[:120]))
    elif "redirect=" not in record.lower():
        out.append(
            _finding("spf-no-all", "low", "SPF record has no 'all' mechanism",
                     "Without a closing all mechanism the default result is neutral.",
                     "End the record with -all.")
        )  # fmt: skip
    return out


def _dmarc_findings(dmarc: list[str]) -> list[dict]:
    if not dmarc:
        return [
            _finding("dmarc-missing", "medium", "No DMARC record",
                     "Receivers have no instructions for mail that fails SPF/DKIM, and you get "
                     "no reports about spoofing of your domain.",
                     "Publish a TXT record at _dmarc.<domain>, starting with "
                     "'v=DMARC1; p=none; rua=mailto:you@<domain>' and tightening to "
                     "quarantine or reject.")
        ]  # fmt: skip
    tags = {}
    for part in dmarc[0].split(";"):
        key, sep, value = part.partition("=")
        if sep:
            tags[key.strip().lower()] = value.strip().lower()
    policy = tags.get("p", "")
    if policy in ("quarantine", "reject"):
        return [_finding("dmarc-ok", "info", f"DMARC enforces p={policy}", dmarc[0][:120])]
    return [
        _finding("dmarc-monitor-only", "low", "DMARC is monitoring only (p=none)",
                 "Spoofed mail is reported but still delivered.",
                 "Once the reports look clean, move to p=quarantine and then p=reject.")
    ]  # fmt: skip


@skill(tags=["security"], timeout=90.0)
async def dns_security_check(domain: str, ctx: SkillContext = None) -> dict:
    """Check a domain's email-spoofing protection (SPF, DMARC) and DNS hygiene (MX, CAA, DNSSEC).

    Uses DNS-over-HTTPS lookups only. Refuses domains that are not authorised.

    Args:
        domain: The domain to check, for example example.com.
    """
    host = _host_arg(domain)
    require_authorized(ctx, host)
    url = str(ctx.config.get("security.doh_url", "https://cloudflare-dns.com/dns-query"))
    if not url.startswith(("https://", "http://")):
        raise SkillError("security.doh_url must be an http(s) URL.")
    async with http_client(ctx) as client:
        mx, _ = await _doh(client, url, host, "MX")
        txt, ad = await _doh(client, url, host, "TXT")
        dmarc_txt, _ = await _doh(client, url, f"_dmarc.{host}", "TXT")
        caa, _ = await _doh(client, url, host, "CAA")
    spf = [v for v in map(_txt_value, txt) if v.lower().startswith("v=spf1")]
    dmarc = [v for v in map(_txt_value, dmarc_txt) if v.lower().startswith("v=dmarc1")]

    findings: list[dict] = []
    if not mx:
        findings.append(_finding("mx-none", "info", "No MX records", "The domain does not "
                                 "receive mail."))  # fmt: skip
    findings += _spf_findings(spf, bool(mx))
    findings += _dmarc_findings(dmarc)
    if caa:
        findings.append(_finding("caa-present", "info", "CAA records restrict certificate issuers",
                                 "; ".join(caa)[:200]))  # fmt: skip
    else:
        findings.append(
            _finding("caa-missing", "info", "No CAA records",
                     "Any certificate authority may issue certificates for this domain.",
                     "Add CAA records naming the authorities you use.")
        )  # fmt: skip
    if ad:
        findings.append(_finding("dnssec-ok", "info", "DNSSEC validated", ""))
    else:
        findings.append(
            _finding("dnssec-off", "info", "DNSSEC not validated",
                     "The resolver did not report an authenticated answer.",
                     "Enable DNSSEC at your registrar and DNS host if they support it.")
        )  # fmt: skip
    findings.sort(key=lambda f: _ORDER[f["severity"]])
    return {
        "domain": host,
        "records": {"mx": mx, "spf": spf, "dmarc": dmarc, "caa": caa},
        "dnssec": ad,
        "findings": findings,
    }


# -- TLS ----------------------------------------------------------------------


def _name_part(rdns, key: str) -> str:
    for rdn in rdns or ():
        for k, v in rdn:
            if k == key:
                return v
    return ""


def _tls_handshake(host: str, port: int) -> dict:
    """Blocking handshake with full verification; falls back to an unverified one to
    still report the protocol when the certificate is rejected."""
    error: str | None = None
    context = ssl.create_default_context()
    try:
        sock = socket.create_connection((host, port), timeout=8)
        with sock, context.wrap_socket(sock, server_hostname=host) as tls:
            cert = tls.getpeercert() or {}
            cipher = tls.cipher()
            version = tls.version()
        verified = True
    except ssl.SSLCertVerificationError as exc:
        error = str(exc.verify_message or exc)
        cert, verified = {}, False
        loose = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        loose.check_hostname = False
        loose.verify_mode = ssl.CERT_NONE
        with socket.create_connection((host, port), timeout=8) as sock:
            with loose.wrap_socket(sock, server_hostname=host) as tls:
                cipher, version = tls.cipher(), tls.version()
    sans = [v for k, v in cert.get("subjectAltName", ()) if k == "DNS"]
    not_after = cert.get("notAfter")
    return {
        "verified": verified,
        "verify_error": error,
        "subject_cn": _name_part(cert.get("subject"), "commonName"),
        "issuer": _name_part(cert.get("issuer"), "organizationName")
        or _name_part(cert.get("issuer"), "commonName"),
        "sans": sans,
        "not_after": ssl.cert_time_to_seconds(not_after) if not_after else None,
        "version": version,
        "cipher": cipher[0] if cipher else None,
    }


def _legacy_tls_accepted(host: str, port: int) -> bool | None:
    """True if the server completes a TLS 1.0/1.1 handshake, False if it refuses,
    None if this machine's OpenSSL cannot make the attempt."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE  # this probe only looks at the protocol, nothing is sent
    try:
        context.minimum_version = ssl.TLSVersion.TLSv1
        context.maximum_version = ssl.TLSVersion.TLSv1_1
        context.set_ciphers("ALL:@SECLEVEL=0")
    except (ValueError, ssl.SSLError):
        return None
    try:
        with socket.create_connection((host, port), timeout=8) as sock:
            with context.wrap_socket(sock, server_hostname=host):
                return True
    except ssl.SSLError as exc:
        text = str(exc).upper()
        if "NO_PROTOCOLS_AVAILABLE" in text or ("UNSUPPORTED_PROTOCOL" in text and "ALERT" not in text):
            return None
        return False
    except OSError:
        return None


@skill(tags=["security"], timeout=60.0)
async def tls_check(host: str, port: int = 443, ctx: SkillContext = None) -> dict:
    """Inspect a site's TLS certificate and protocol: issuer, expiry, names, and whether
    legacy TLS 1.0/1.1 is still accepted. Refuses hosts that are not authorised.

    Args:
        host: The host name to check, for example example.com.
        port: TCP port of the TLS service.
    """
    name = _host_arg(host)
    require_authorized(ctx, name)
    port = int(port)
    if not 1 <= port <= 65535:
        raise SkillError("port must be between 1 and 65535.")
    findings: list[dict] = []
    try:
        info = await asyncio.to_thread(_tls_handshake, name, port)
    except (OSError, ssl.SSLError) as exc:
        raise SkillError(f"Could not complete a TLS handshake with {name}:{port}: {exc}") from exc

    days = None
    if info.get("not_after"):
        days = int((info["not_after"] - time.time()) // 86400)
    error = (info.get("verify_error") or "").lower()
    if not info.get("verified", True):
        if "expired" in error:
            findings.append(
                _finding("cert-expired", "high", "TLS certificate has expired", info["verify_error"],
                         "Renew the certificate and automate renewal.")
            )  # fmt: skip
        elif "match" in error or "hostname" in error or "ip address" in error:
            findings.append(
                _finding("cert-hostname-mismatch", "high", "Certificate does not match the host name",
                         info["verify_error"],
                         "Install a certificate that lists this host name.")
            )  # fmt: skip
        else:
            findings.append(
                _finding("cert-untrusted", "high", "Certificate chain is not trusted",
                         info["verify_error"] or "verification failed",
                         "Install a certificate from a trusted CA, including intermediates.")
            )  # fmt: skip
    if days is not None and info.get("verified", True):
        if days < 0:
            findings.append(
                _finding("cert-expired", "high", "TLS certificate has expired",
                         f"Expired {-days} days ago.", "Renew the certificate.")
            )  # fmt: skip
        elif days < 14:
            findings.append(
                _finding("cert-expiring", "high", "TLS certificate expires very soon",
                         f"{days} days left.", "Renew the certificate now.")
            )  # fmt: skip
        elif days < 30:
            findings.append(
                _finding("cert-expiring", "medium", "TLS certificate expires soon",
                         f"{days} days left.", "Renew the certificate and automate renewal.")
            )  # fmt: skip

    legacy = await asyncio.to_thread(_legacy_tls_accepted, name, port)
    if legacy is True:
        findings.append(
            _finding("legacy-tls", "medium", "Legacy TLS 1.0/1.1 still accepted",
                     "The server completed a handshake using TLS 1.0 or 1.1, which have known "
                     "weaknesses and are deprecated.",
                     "Disable TLS 1.0 and 1.1 and require TLS 1.2 or newer.")
        )  # fmt: skip
    elif legacy is None:
        findings.append(
            _finding("legacy-tls-untested", "info", "Legacy TLS could not be tested",
                     "This machine's TLS library cannot offer TLS 1.0/1.1, so the check was "
                     "skipped.")
        )  # fmt: skip
    if not findings:
        findings.append(_finding("tls-ok", "info", "TLS certificate and protocol look healthy", ""))
    findings.sort(key=lambda f: _ORDER[f["severity"]])
    return {
        "host": name,
        "port": port,
        "subject": info.get("subject_cn"),
        "issuer": info.get("issuer"),
        "alt_names": list(info.get("sans") or [])[:10],
        "days_until_expiry": days,
        "protocol": info.get("version"),
        "cipher": info.get("cipher"),
        "certificate_trusted": bool(info.get("verified", True)),
        "legacy_tls_accepted": legacy,
        "findings": findings,
    }


# -- exposed services -----------------------------------------------------------

# The only ports ever probed. Deliberately fixed: no ranges, no model-chosen ports.
_SERVICE_PORTS = {
    21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 445: "SMB", 1433: "MSSQL",
    3306: "MySQL", 3389: "RDP", 5432: "PostgreSQL", 5900: "VNC", 6379: "Redis",
    9200: "Elasticsearch", 11211: "Memcached", 27017: "MongoDB",
}  # fmt: skip
_DATA_PORTS = {1433, 3306, 5432, 6379, 9200, 11211, 27017}
_RISKY_PORTS = {21, 23, 445, 3389, 5900}


async def _port_open(host: str, port: int, timeout: float) -> bool:
    """TCP connect and close. Nothing is sent and no banner is read."""
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
    except (OSError, TimeoutError):
        return False
    writer.close()
    try:
        await writer.wait_closed()
    except OSError:
        pass
    return True


def _port_finding(port: int, service: str) -> dict:
    fid = f"port-{port}-open"
    if port in _DATA_PORTS:
        return _finding(fid, "high", f"{service} (port {port}) reachable from the internet",
                        f"{service} accepts connections from outside.",
                        f"{service} should not be reachable from the internet; bind it to "
                        "localhost or firewall it.")  # fmt: skip
    if port in _RISKY_PORTS:
        return _finding(fid, "medium", f"{service} (port {port}) reachable from the internet",
                        f"{service} is a frequent target of automated attacks.",
                        f"Close port {port} or restrict it to a VPN or trusted addresses.")  # fmt: skip
    return _finding(fid, "info", f"{service} (port {port}) reachable",
                    f"{service} accepts connections.",
                    "Expected for a mail or admin host; make sure it is patched and uses "
                    "strong authentication.")  # fmt: skip


@skill(tags=["security"], timeout=60.0)
async def exposed_services_check(host: str, ctx: SkillContext = None) -> dict:
    """Check whether risky network services (databases, remote desktop, telnet, ...) are
    reachable on a host. Plain TCP connects to a fixed list of ports only; nothing is
    sent. Refuses hosts that are not authorised.

    Args:
        host: The host name or IP to check.
    """
    name = _host_arg(host)
    require_authorized(ctx, name)
    ports = sorted(_SERVICE_PORTS)
    results = await asyncio.gather(*(_port_open(name, p, 2.0) for p in ports))
    open_ports = [
        {"port": p, "service": _SERVICE_PORTS[p]} for p, ok in zip(ports, results, strict=True) if ok
    ]
    findings = [_port_finding(o["port"], o["service"]) for o in open_ports]
    if not findings:
        findings.append(
            _finding("no-exposed-services", "info", "None of the checked services are reachable",
                     f"Checked {len(ports)} common service ports.")
        )  # fmt: skip
    findings.sort(key=lambda f: _ORDER[f["severity"]])
    return {"host": name, "checked_ports": len(ports), "open_ports": open_ports, "findings": findings}


# -- full audit -----------------------------------------------------------------


def _safe_dirname(host: str) -> str:
    return re.sub(r"[^a-z0-9.-]", "-", host.lower()).strip(".") or "site"


def _titles(findings: list[dict]) -> list[str]:
    # Informational notes come and go with the weather; track what needs attention.
    return [f["title"] for f in findings if f["severity"] != "info"]


def _markdown_report(host: str, result: dict, full: list[dict], day: str) -> str:
    summary = result["summary"]
    lines = [
        f"# Security report: {host}",
        "",
        f"Date: {day}  ",
        f"Grade: **{result['grade']}**",
        "",
        "| High | Medium | Low | Info |",
        "|---|---|---|---|",
        f"| {summary['high']} | {summary['medium']} | {summary['low']} | {summary['info']} |",
        "",
    ]
    changes = result["changes"]
    if changes:
        lines += [f"## Changes since last run (previous grade {changes['previous_grade']})", ""]
        lines += [f"- New: {t}" for t in changes["new"]] or ["- Nothing new."]
        lines += [f"- Resolved: {t}" for t in changes["resolved"]]
        lines.append("")
    for level in _ORDER:
        group = [f for f in full if f["severity"] == level]
        if not group:
            continue
        lines += [f"## {level.capitalize()} ({len(group)})", ""]
        for f in group:
            lines.append(f"### {f['title']}")
            if f.get("detail"):
                lines.append(f["detail"])
            if f.get("recommendation"):
                lines.append(f"\n**Fix:** {f['recommendation']}")
            lines.append("")
    if result["checks_failed"]:
        lines += ["## Checks that could not run", ""]
        lines += [f"- {c['check']}: {c['error']}" for c in result["checks_failed"]]
        lines.append("")
    lines.append("Passive checks only; this is not a substitute for a penetration test.")
    return "\n".join(lines) + "\n"


async def _attempt(name: str, call):
    try:
        return name, await call, None
    except Exception as exc:  # one failing check must not sink the audit
        return name, None, str(exc) or type(exc).__name__


@skill(tags=["security"], timeout=300.0)
async def security_audit(url: str, ctx: SkillContext = None) -> dict:
    """Run the full passive security check of one site you own: web headers and exposed
    files, TLS, DNS/email protection and exposed network services. Saves a report and
    compares with the previous run. Refuses sites that are not authorised.

    Args:
        url: The site to audit, for example https://example.com.
    """
    if "://" not in url:
        url = "https://" + url
    parts = urlsplit(url)
    host = _host_arg(url)
    if parts.scheme not in ("http", "https"):
        raise SkillError(f"'{url}' is not a valid http(s) URL.")
    require_authorized(ctx, host)

    domain = host[4:] if host.startswith("www.") else host
    notes: list[dict] = []
    jobs = [("web", security_scan(url, ctx=ctx))]
    if _is_ip(host) or host == "localhost":
        pass
    elif is_authorized(domain, load_targets(ctx.config)):
        jobs.append(("dns", dns_security_check(domain, ctx=ctx)))
    else:
        notes.append(
            _finding("dns:skipped", "info", "DNS check skipped",
                     f"{domain} is not on the authorised list, so its DNS records were not "
                     "checked.", f"Run `jarvis security allow {domain}` to include them.")
        )  # fmt: skip
    if parts.scheme == "https":
        jobs.append(("tls", tls_check(host, parts.port or 443, ctx=ctx)))
    jobs.append(("ports", exposed_services_check(host, ctx=ctx)))

    outcomes = await asyncio.gather(*(_attempt(n, c) for n, c in jobs))
    results = {n: r for n, r, _ in outcomes if r is not None}
    failed = [{"check": n, "error": e} for n, _, e in outcomes if e]

    findings: list[dict] = []
    for check, result in results.items():
        for f in result["findings"]:
            merged = dict(f, check=check)
            if check != "web":
                merged["id"] = f"{check}:{f['id']}"
            findings.append(merged)
    findings += notes
    findings.sort(key=lambda f: _ORDER[f["severity"]])
    grade, summary = _grade(findings), _summarise(findings)
    if any(c["check"] == "web" for c in failed):
        grade = "?"  # without the web check a letter would be misleading

    folder = f"security/{_safe_dirname(host)}"
    latest = ctx.resolve_path(f"{folder}/latest.json")
    previous = None
    try:
        previous = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    changes = None
    carried: list[dict] = []
    if isinstance(previous, dict) and isinstance(previous.get("findings"), list):
        # a check that failed today says nothing about its old findings: keep them as-is
        carried = [
            dict(f, stale=True) for f in previous["findings"]
            if f.get("check") in {c["check"] for c in failed}
        ]
        before, now = _titles(previous["findings"]), _titles(findings + carried)
        changes = {
            "new": [t for t in now if t not in before],
            "resolved": [t for t in before if t not in now],
            "previous_grade": previous.get("grade"),
        }

    now_utc = datetime.now(UTC)
    day = now_utc.date().isoformat()
    report_rel = f"{folder}/{day}.md"
    result = {
        "target": url,
        "grade": grade,
        "summary": summary,
        "changes": changes,
        "report": report_rel,
        "findings": [
            {"severity": f["severity"], "title": f["title"], "recommendation": f["recommendation"]}
            for f in findings[:8]
        ],
        "checks_failed": failed,
    }
    full = {
        **result,
        "host": host,
        "scanned_at": now_utc.isoformat(timespec="seconds"),
        "findings": findings + carried,
        "checks": {n: {k: v for k, v in r.items() if k != "findings"} for n, r in results.items()},
    }
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(json.dumps(full, indent=2, default=str), encoding="utf-8")
    ctx.resolve_path(report_rel).write_text(
        _markdown_report(host, result, findings, day), encoding="utf-8"
    )
    return result


_MAX_SITES = 10
_GRADE_RANK = {"F": 0, "D": 1, "C": 2, "B": 3, "A": 4}


@skill(tags=["security"], timeout=900.0)
async def security_scan_all(ctx: SkillContext = None) -> dict:
    """Run the full security audit for every authorised website (up to 10), worst grade
    first. Use it for requests like "check our websites". Wildcard entries are skipped.
    """
    hosts = concrete_hosts(load_targets(ctx.config))
    if not hosts:
        return {
            "sites": [],
            "note": "No authorised sites yet. Authorise one first with "
            "`jarvis security allow <host>`.",
        }
    chosen = hosts[:_MAX_SITES]
    gate = asyncio.Semaphore(3)

    async def one(host: str) -> dict:
        target = f"https://[{host}]" if ":" in host else f"https://{host}"
        async with gate:
            try:
                audit = await security_audit(target, ctx=ctx)
            except Exception as exc:
                return {"host": host, "grade": "?", "error": str(exc) or type(exc).__name__}
        changes = audit["changes"]
        return {
            "host": host,
            "grade": audit["grade"],
            "summary": audit["summary"],
            # on a first run everything that needs attention counts as new
            "new_findings": len(changes["new"]) if changes else sum(
                audit["summary"][level] for level in ("high", "medium", "low")
            ),
            "resolved": len(changes["resolved"]) if changes else 0,
            "first_run": changes is None,
            "report": audit["report"],
            "checks_failed": [c["check"] for c in audit["checks_failed"]],
        }

    sites = list(await asyncio.gather(*(one(h) for h in chosen)))
    sites.sort(key=lambda s: _GRADE_RANK.get(s["grade"], -1))
    out: dict = {"sites": sites}
    if len(hosts) > _MAX_SITES:
        out["note"] = (
            f"{len(hosts)} sites are authorised; only the first {_MAX_SITES} were checked."
        )
    return out
