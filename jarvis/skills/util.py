"""Helpers shared by skills that touch the network."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import Any
from urllib.parse import urlsplit

from jarvis._http import httpx
from jarvis.skills.base import SkillContext, SkillError

USER_AGENT = "Jarvis/0.1 (+local assistant)"


def http_client(ctx: SkillContext, **kwargs: Any) -> httpx.AsyncClient:
    """An AsyncClient with sane defaults. Tests inject a transport through
    ``ctx.services["http_transport"]`` so no skill ever needs real network."""
    kwargs.setdefault("timeout", float(ctx.config.get("web.timeout_seconds", 20)))
    kwargs.setdefault("follow_redirects", True)
    headers = {"User-Agent": USER_AGENT, **kwargs.pop("headers", {})}
    transport = ctx.services.get("http_transport")
    return httpx.AsyncClient(transport=transport, headers=headers, **kwargs)


def is_private_address(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


async def resolve_host(host: str) -> list[str]:
    if _is_ip(host):
        return [host]
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SkillError(f"Cannot resolve host '{host}'.") from exc
    return sorted({info[4][0] for info in infos})


async def check_url(ctx: SkillContext, url: str, *, allow_private: bool | None = None) -> str:
    """Validate a model-supplied URL before fetching it.

    Only http(s) is accepted, and unless ``web.allow_private_hosts`` is set
    the host must not resolve to a loopback, private or link-local address:
    a web page must not be able to steer the assistant into your LAN or
    cloud metadata endpoints. Returns the normalised URL.
    """
    if "://" not in url:
        url = "https://" + url
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise SkillError(f"'{url}' is not a valid http(s) URL.")
    if allow_private is None:
        allow_private = bool(ctx.config.get("web.allow_private_hosts", False))
    if not allow_private:
        host = parts.hostname
        if host == "localhost" or host.endswith(".localhost"):
            addresses = ["127.0.0.1"]
        elif "http_transport" in ctx.services and not _is_ip(host):
            addresses = []  # injected transports (tests) never touch the network
        else:
            addresses = await resolve_host(host)
        for address in addresses:
            if is_private_address(address):
                raise SkillError(
                    f"Refusing to access '{parts.hostname}': it resolves to a private or "
                    "local address. Set web.allow_private_hosts = true to allow this."
                )
    return url
