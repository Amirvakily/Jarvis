"""The allow-list every security skill goes through.

Targets come from two places: ``[security] authorized_targets`` in jarvis.toml
and a small JSON file in the data directory that ``jarvis security allow``
edits. Only a human at the command line can change either: no skill adds or
removes targets, so the language model can never widen its own scope.
"""

from __future__ import annotations

import ipaddress
import json
import re
from pathlib import Path

from jarvis.config import Config
from jarvis.skills.base import SkillContext, SkillError

LOOPBACK = {"localhost", "127.0.0.1", "::1"}
_LABEL = r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
_HOSTNAME = re.compile(rf"^{_LABEL}(?:\.{_LABEL})+$")
_MAX_HOST = 253


def store_path(config: Config) -> Path:
    return config.data_dir / "authorized_targets.json"


def _dedupe(items) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        text = str(item).strip().lower().rstrip(".")
        if text:
            seen.setdefault(text, None)
    return list(seen)


def config_targets(config: Config) -> list[str]:
    raw = config.get("security.authorized_targets", []) or []
    if isinstance(raw, str):
        raw = [raw]
    return _dedupe(raw) if isinstance(raw, list) else []


def added_targets(config: Config) -> list[str]:
    try:
        data = json.loads(store_path(config).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return _dedupe(data) if isinstance(data, list) else []


def load_targets(config: Config) -> list[str]:
    """Config entries followed by added ones, lower-cased and de-duplicated."""
    return _dedupe([*config_targets(config), *added_targets(config)])


def normalize_pattern(pattern: str) -> str:
    """Validate and normalise a host, ``*.host`` wildcard or IP literal."""
    text = str(pattern).strip().lower()
    if not text:
        raise ValueError("Give a host name such as example.com.")
    if "://" in text or "/" in text or "?" in text or "#" in text or "@" in text:
        raise ValueError(
            f"'{pattern}' looks like a URL. Give just the host name, for example example.com."
        )
    if any(ch.isspace() for ch in text):
        raise ValueError(f"'{pattern}' contains spaces.")
    try:
        return str(ipaddress.ip_address(text))
    except ValueError:
        pass
    if ":" in text:
        raise ValueError(f"'{pattern}' includes a port or is not a valid address; use the host only.")
    wildcard = text.startswith("*.")
    name = text[2:] if wildcard else text
    if wildcard and ":" not in name and re.fullmatch(r"[0-9.]+", name):
        raise ValueError(f"'{pattern}': wildcards cannot be used with IP addresses.")
    if "*" in name:
        raise ValueError(f"'{pattern}': a wildcard is only allowed as a leading '*.'.")
    if len(name) > _MAX_HOST or not _HOSTNAME.match(name):
        raise ValueError(
            f"'{pattern}' is not a valid host name. Use something like example.com, "
            "*.example.com, or an IP address."
        )
    return text


def _write(config: Config, items: list[str]) -> None:
    path = store_path(config)
    path.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")


def add_target(config: Config, pattern: str) -> str:
    """Add a target to the data-directory store. Returns the normalised entry."""
    entry = normalize_pattern(pattern)
    items = added_targets(config)
    if entry not in items:
        items.append(entry)
        _write(config, items)
    return entry


def remove_target(config: Config, pattern: str) -> bool:
    """Remove an added target. Entries from jarvis.toml are not touched."""
    entry = str(pattern).strip().lower().rstrip(".")
    items = added_targets(config)
    if entry not in items:
        return False
    _write(config, [i for i in items if i != entry])
    return True


def is_authorized(host: str, patterns: list[str]) -> bool:
    host = host.strip().lower().rstrip(".")
    if host in LOOPBACK:
        return True
    for raw in patterns:
        pattern = str(raw).strip().lower().rstrip(".")
        if pattern.startswith("*."):
            if host.endswith(pattern[1:]) and host != pattern[2:]:
                return True
        elif pattern and host == pattern:
            return True
    return False


def require_authorized(ctx: SkillContext, host: str) -> None:
    """Raise unless ``host`` is on the allow-list. Call before any network activity."""
    if not is_authorized(host, load_targets(ctx.config)):
        raise SkillError(
            f"Security checks only run against sites you have authorised, and '{host}' is not "
            f"on that list. If you own it or have written permission to test it, you can add "
            f"it yourself with `jarvis security allow {host}` (or list it under "
            "[security] authorized_targets in jarvis.toml). I cannot add sites for you."
        )


def concrete_hosts(patterns: list[str]) -> list[str]:
    return [p for p in patterns if not p.startswith("*.")]
