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


# -- authorisation store, new skills, audit, CLI ---------------------------------

import json  # noqa: E402
import time  # noqa: E402

from jarvis.skills import authorization as auth  # noqa: E402
from jarvis.skills.builtin import security as sec  # noqa: E402
from jarvis.skills.builtin.security import (  # noqa: E402
    dns_security_check,
    exposed_services_check,
    security_audit,
    security_scan_all,
    tls_check,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Example.com", "example.com"),
        ("*.example.com", "*.example.com"),
        ("a-b.example.co.uk", "a-b.example.co.uk"),
        ("10.0.0.1", "10.0.0.1"),
        ("::1", "::1"),
        ("2001:DB8::1", "2001:db8::1"),
    ],
)
def test_validation_accepts(value, expected):
    assert auth.normalize_pattern(value) == expected


@pytest.mark.parametrize(
    "value",
    ["*", "*.com", "com", "localhost", "http://x.com", "x.com/path", "x.com:8080", "",
     "-a.com", "a..com", "*.*.com", "a b.com", "*.10.0.0.1", "user@x.com"],
)  # fmt: skip
def test_validation_rejects(value):
    with pytest.raises(ValueError):
        auth.normalize_pattern(value)


def test_add_remove_and_union(config):
    config.set("security.authorized_targets", ["Example.com", "a.org"])
    assert auth.add_target(config, "A.org") == "a.org"
    auth.add_target(config, "*.b.net")
    assert auth.load_targets(config) == ["example.com", "a.org", "*.b.net"]
    assert auth.added_targets(config) == ["a.org", "*.b.net"]
    assert auth.remove_target(config, "a.org") is True
    assert auth.remove_target(config, "a.org") is False
    assert auth.remove_target(config, "example.com") is False  # config entries untouched
    assert auth.load_targets(config) == ["example.com", "a.org", "*.b.net"]
    assert auth.concrete_hosts(auth.load_targets(config)) == ["example.com", "a.org"]
    with pytest.raises(ValueError):
        auth.add_target(config, "*.com")
    auth.store_path(config).write_text("not json")
    assert auth.load_targets(config) == ["example.com", "a.org"]


def test_is_authorized_rules():
    assert auth.is_authorized("localhost", [])
    assert auth.is_authorized("www.x.com", ["*.x.com"])
    assert not auth.is_authorized("x.com", ["*.x.com"])
    assert not auth.is_authorized("evilx.com", ["*.x.com"])


def test_no_skill_can_authorise():
    from jarvis.skills.registry import SkillRegistry

    registry = SkillRegistry()
    registry.reload()
    names = [s.name for s in registry.all()]
    assert "security_audit" in names
    assert not [n for n in names if "allow" in n or "authori" in n]
    assert not [n for n in vars(auth) if hasattr(getattr(auth, n), "__skill__")]
    for s in registry.all():
        assert s.fn.__module__ != auth.__name__


class Probes:
    """Patched network probes that record whether they were called."""

    def __init__(self, monkeypatch, tls=None, legacy=False, open_ports=()):
        self.calls = []
        self.probed = []
        info = tls or {
            "verified": True, "verify_error": None, "subject_cn": "example.com",
            "issuer": "Test CA", "sans": [f"h{i}.example.com" for i in range(15)],
            "not_after": time.time() + 90 * 86400 + 3600, "version": "TLSv1.3",
            "cipher": "TLS_AES_256_GCM_SHA384",
        }  # fmt: skip

        def handshake(host, port):
            self.calls.append(("tls", host, port))
            if isinstance(info, Exception):
                raise info
            return info

        def legacy_probe(host, port):
            self.calls.append(("legacy", host, port))
            return legacy

        async def port_open(host, port, timeout):
            self.calls.append(("port", host, port))
            self.probed.append(port)
            return port in open_ports

        monkeypatch.setattr(sec, "_tls_handshake", handshake)
        monkeypatch.setattr(sec, "_legacy_tls_accepted", legacy_probe)
        monkeypatch.setattr(sec, "_port_open", port_open)


def dns_handler(records, ad=False, calls=None):
    """records: {(name, type): [data, ...]}"""
    codes = {"MX": 15, "TXT": 16, "CAA": 257}

    def handler(request):
        if calls is not None:
            calls.append(request)
        assert request.headers["accept"] == "application/dns-json"
        name, rtype = request.url.params["name"], request.url.params["type"]
        data = records.get((name, rtype), [])
        body = {"Status": 0, "AD": ad}
        if data:
            body["Answer"] = [
                {"name": name, "type": codes[rtype], "TTL": 60, "data": d} for d in data
            ]
        return httpx.Response(200, json=body)

    return handler


def fid(result):
    return {f["id"]: f for f in result["findings"]}


async def test_new_skills_refuse_unauthorised_before_any_io(ctx, monkeypatch):
    probes = Probes(monkeypatch)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200)

    ctx.services["http_transport"] = httpx.MockTransport(handler)
    for coro in (
        dns_security_check("example.com", ctx=ctx),
        tls_check("example.com", ctx=ctx),
        exposed_services_check("example.com", ctx=ctx),
        security_audit("https://example.com", ctx=ctx),
    ):
        with pytest.raises(SkillError, match="jarvis security allow"):
            await coro
    assert calls == [] and probes.calls == []


async def test_dns_verdicts(ctx):
    d = "example.com"
    spf = {
        "-all": '"v=spf1 include:_spf.x.com -all"',
        "~all": '"v=spf1 ~all"',
        "?all": '"v=spf1 ?all"',
        "+all": '"v=spf1 +all"',
    }
    expect = {"-all": ("spf-ok", "info"), "~all": ("spf-softfail", "low"),
              "?all": ("spf-neutral", "medium"), "+all": ("spf-allow-all", "high")}  # fmt: skip
    for key, txt in spf.items():
        recs = {(d, "MX"): ["10 mx.example.com."], (d, "TXT"): [txt],
                ("_dmarc." + d, "TXT"): ['"v=DMARC1; p=reject"']}  # fmt: skip
        scan_ctx(ctx, dns_handler(recs, ad=True))
        r = await dns_security_check(d, ctx=ctx)
        f = fid(r)
        assert f[expect[key][0]]["severity"] == expect[key][1]
        assert f["dmarc-ok"]["severity"] == "info" and r["dnssec"] is True
        assert "dnssec-ok" in f and "caa-missing" in f

    # missing everything: SPF medium with MX, low without; DMARC medium
    scan_ctx(ctx, dns_handler({(d, "MX"): ["10 mx."]}))
    f = fid(await dns_security_check(d, ctx=ctx))
    assert f["spf-missing"]["severity"] == "medium" and f["dmarc-missing"]["severity"] == "medium"
    assert "dnssec-off" in f
    scan_ctx(ctx, dns_handler({}))
    r = await dns_security_check(d, ctx=ctx)
    assert fid(r)["spf-missing"]["severity"] == "low" and "mx-none" in fid(r)


async def test_dns_multichunk_multiple_spf_and_dmarc_none(ctx):
    d = "example.com"
    recs = {
        (d, "TXT"): ['"v=spf1 include:a.com " "include:b.com -all"', '"v=spf1 -all"', '"other"'],
        ("_dmarc." + d, "TXT"): ['"v=DMARC1; " "p=none"'],
        (d, "CAA"): ['0 issue "letsencrypt.org"'],
    }
    scan_ctx(ctx, dns_handler(recs))
    r = await dns_security_check(d, ctx=ctx)
    f = fid(r)
    assert r["records"]["spf"][0] == "v=spf1 include:a.com include:b.com -all"
    assert f["spf-multiple"]["severity"] == "medium"
    assert f["dmarc-monitor-only"]["severity"] == "low"
    assert "caa-present" in f and r["records"]["dmarc"] == ["v=DMARC1; p=none"]


async def test_dns_uses_configured_resolver(ctx):
    seen = []
    ctx.config.set("security.doh_url", "https://doh.test/q")
    scan_ctx(ctx, dns_handler({}, calls=seen))
    await dns_security_check("example.com", ctx=ctx)
    assert seen and all(str(r.url).startswith("https://doh.test/q") for r in seen)


async def test_tls_findings(ctx, monkeypatch):
    ctx.config.set("security.authorized_targets", ["example.com"])
    Probes(monkeypatch)
    r = await tls_check("example.com", ctx=ctx)
    assert len(r["alt_names"]) == 10 and r["protocol"] == "TLSv1.3"
    assert 89 <= r["days_until_expiry"] <= 90 and r["legacy_tls_accepted"] is False
    assert "tls-ok" in fid(r)

    base = {"verified": True, "verify_error": None, "subject_cn": "x", "issuer": "y", "sans": [],
            "version": "TLSv1.2", "cipher": "c"}  # fmt: skip
    day = 86400
    cases = [
        (dict(base, not_after=time.time() - 3 * day), "cert-expired", "high"),
        (dict(base, not_after=time.time() + 5 * day), "cert-expiring", "high"),
        (dict(base, not_after=time.time() + 20 * day), "cert-expiring", "medium"),
        (dict(base, not_after=None, verified=False, verify_error="Hostname mismatch"),
         "cert-hostname-mismatch", "high"),
        (dict(base, not_after=None, verified=False, verify_error="self-signed certificate"),
         "cert-untrusted", "high"),
    ]  # fmt: skip
    for info, expected_id, severity in cases:
        Probes(monkeypatch, tls=info)
        f = fid(await tls_check("example.com", ctx=ctx))
        assert f[expected_id]["severity"] == severity

    Probes(monkeypatch, legacy=True)
    assert fid(await tls_check("example.com", ctx=ctx))["legacy-tls"]["severity"] == "medium"
    Probes(monkeypatch, legacy=None)
    r = await tls_check("example.com", ctx=ctx)
    assert r["legacy_tls_accepted"] is None and "legacy-tls-untested" in fid(r)
    assert "legacy-tls" not in fid(r)


async def test_tls_port_validation(ctx, monkeypatch):
    ctx.config.set("security.authorized_targets", ["example.com"])
    probes = Probes(monkeypatch)
    for bad in (0, 70000):
        with pytest.raises(SkillError, match="port"):
            await tls_check("example.com", bad, ctx=ctx)
    assert probes.calls == []


async def test_exposed_services(ctx, monkeypatch):
    ctx.config.set("security.authorized_targets", ["example.com"])
    probes = Probes(monkeypatch, open_ports={22, 25, 3306, 6379, 3389, 23})
    r = await exposed_services_check("example.com", ctx=ctx)
    sev = {o["port"]: fid(r)[f"port-{o['port']}-open"]["severity"] for o in r["open_ports"]}
    assert sev == {22: "info", 23: "medium", 25: "info", 3306: "high", 3389: "medium", 6379: "high"}
    assert {o["port"]: o["service"] for o in r["open_ports"]}[6379] == "Redis"
    assert sorted(probes.probed) == [21, 22, 23, 25, 445, 1433, 3306, 3389, 5432, 5900, 6379,
                                     9200, 11211, 27017]  # fmt: skip
    assert "firewall" in fid(r)["port-6379-open"]["recommendation"]
    Probes(monkeypatch)
    assert "no-exposed-services" in fid(await exposed_services_check("example.com", ctx=ctx))


def audit_ctx(ctx, monkeypatch, headers=HARDENED, records=None, **probe_kw):
    ctx.config.set("security.authorized_targets", ["example.com", "*.example.com"])
    site = make_site(headers)
    dns = dns_handler(records or {})

    def handler(request):
        return dns(request) if request.url.host == "doh.test" else site(request)

    ctx.config.set("security.doh_url", "https://doh.test/q")
    ctx.services["http_transport"] = httpx.MockTransport(handler)
    return Probes(monkeypatch, **probe_kw)


async def test_audit_writes_reports_and_tracks_changes(ctx, monkeypatch):
    audit_ctx(ctx, monkeypatch, headers={"content-type": "text/html"}, open_ports={6379})
    first = await security_audit("https://www.example.com", ctx=ctx)
    assert first["changes"] is None and first["grade"] == "F"
    assert len(first["findings"]) <= 8 and first["checks_failed"] == []
    assert first["findings"][0]["severity"] == "high"
    assert (ctx.workspace / first["report"]).read_text().startswith("# Security report")
    assert first["report"].startswith("security/www.example.com/") and first["report"].endswith(".md")
    latest = json.loads((ctx.workspace / "security/www.example.com/latest.json").read_text())
    assert latest["grade"] == "F" and len(latest["findings"]) > 8
    assert "hunter2" not in json.dumps(latest)

    audit_ctx(ctx, monkeypatch, headers=HARDENED, open_ports=set(),
              records={("example.com", "TXT"): ['"v=spf1 -all"'],
                       ("_dmarc.example.com", "TXT"): ['"v=DMARC1; p=reject"']})  # fmt: skip
    second = await security_audit("https://www.example.com", ctx=ctx)
    assert second["changes"]["previous_grade"] == "F"
    assert any("Redis" in t for t in second["changes"]["resolved"])
    assert second["grade"] < first["grade"]
    third = await security_audit("https://www.example.com", ctx=ctx)
    assert third["changes"]["new"] == [] and third["changes"]["resolved"] == []


async def test_audit_tolerates_failing_check(ctx, monkeypatch):
    audit_ctx(ctx, monkeypatch, tls=OSError("handshake boom"))
    result = await security_audit("https://example.com", ctx=ctx)
    assert [c["check"] for c in result["checks_failed"]] == ["tls"]
    assert "boom" in result["checks_failed"][0]["error"]
    assert result["grade"]
    assert "Checks that could not run" in (ctx.workspace / result["report"]).read_text()


async def test_audit_skips_dns_when_domain_unauthorised(ctx, monkeypatch):
    ctx.config.set("security.authorized_targets", ["www.example.com"])
    ctx.services["http_transport"] = httpx.MockTransport(make_site(HARDENED))
    Probes(monkeypatch)
    result = await security_audit("https://www.example.com", ctx=ctx)
    assert result["checks_failed"] == []
    full = json.loads((ctx.workspace / "security/www.example.com/latest.json").read_text())
    assert "dns:skipped" in {f["id"] for f in full["findings"]}


async def test_scan_all(ctx, monkeypatch):
    empty = await security_scan_all(ctx=ctx)
    assert empty["sites"] == [] and "jarvis security allow" in empty["note"]

    ctx.config.set("security.authorized_targets", ["good.com", "*.wild.com", "bad.com", "ugly.com"])
    Probes(monkeypatch)

    def handler(request):
        host = request.url.host
        if host == "bad.com":
            raise httpx.ConnectError("down")
        if host == "ugly.com":
            return httpx.Response(200, text="x", headers={"content-type": "text/html"})
        if host == "good.com":
            return make_site(HARDENED)(request)
        return httpx.Response(200, json={"Status": 0})  # DoH resolver

    ctx.services["http_transport"] = httpx.MockTransport(handler)
    out = await security_scan_all(ctx=ctx)
    hosts = [s["host"] for s in out["sites"]]
    assert sorted(hosts) == ["bad.com", "good.com", "ugly.com"] and "wild.com" not in repr(hosts)
    grades = [s["grade"] for s in out["sites"]]
    assert grades[0] == "?" and grades[1:] == sorted(grades[1:], reverse=True)
    assert out["sites"][-1]["host"] == "good.com" and out["sites"][0]["host"] == "bad.com"
    bad = next(s for s in out["sites"] if s["host"] == "bad.com")
    assert "web" in bad["checks_failed"]

    ctx.config.set("security.authorized_targets", [f"h{i}.com" for i in range(12)])
    out = await security_scan_all(ctx=ctx)
    assert len(out["sites"]) == 10 and "10" in out["note"]


def test_cli_security_commands(tmp_path, capsys):
    from jarvis.cli import main

    cfg = tmp_path / "jarvis.toml"
    cfg.write_text(
        f'[assistant]\ndata_dir = "{tmp_path / "d"}"\nworkspace = "{tmp_path / "w"}"\n'
        '[security]\nauthorized_targets = ["Config.org"]\n'
    )
    assert main(["-c", str(cfg), "security", "allow", "example.com"]) == 0
    assert "own or have written permission" in capsys.readouterr().out
    assert main(["-c", str(cfg), "security", "allow", "*.com"]) == 2
    assert main(["-c", str(cfg), "security", "targets"]) == 0
    out = capsys.readouterr().out
    assert "config.org" in out and "jarvis.toml" in out and "example.com" in out
    assert main(["-c", str(cfg), "security", "remove", "example.com"]) == 0
    assert main(["-c", str(cfg), "security", "remove", "example.com"]) == 1
    capsys.readouterr()
    assert main(["-c", str(cfg), "security", "targets"]) == 0
    assert "example.com" not in capsys.readouterr().out
