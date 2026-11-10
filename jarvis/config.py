"""Configuration: built-in defaults deep-merged with an optional TOML file.

Secrets never live in the config file. Profiles name the environment variable
that holds a key (``api_key_env``) and the value is read at call time.
"""

from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "assistant": {
        "name": "Jarvis",
        "user_name": "",
        "data_dir": "~/.jarvis",
        "workspace": "./workspace",
        # Conversation turns kept verbatim before the oldest are dropped.
        "max_history_messages": 60,
        "max_tool_rounds": 8,
    },
    "llm": {
        # Name of the profile used for new requests. Switch at runtime from
        # the UI, by voice ("switch to the cloud"), or with /backend in the CLI.
        "active": "local",
        # Profile tried when the active one is unreachable ("" disables).
        "fallback": "",
        "profiles": {
            "local": {
                "provider": "ollama",
                "model": "llama3.1:8b",
                "base_url": "http://localhost:11434",
                "num_ctx": 8192,
                "keep_alive": "10m",
                "memory_mb": 5500,
            },
            "claude": {
                "provider": "anthropic",
                "model": "claude-opus-5-5",
                # Thinking depth. "low" keeps a spoken conversation snappy;
                # raise it for agents that do heavy analysis.
                "effort": "low",
                "max_tokens": 16000,
                # Retry on another Claude model if a safety classifier
                # declines a request (server-side, one round trip).
                "refusal_fallback": True,
            },
            "openai": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
            },
        },
    },
    "resources": {
        # Soft ceiling for locally loaded components (LLM, STT, TTS models).
        "memory_budget_mb": 8000,
        "idle_unload_seconds": 600,
    },
    "voice": {
        # auto   - use the PC microphone and speakers when the speech engines
        #          are installed, otherwise let the window's browser engine do it
        # native - PC microphone/speakers only     browser - window only
        # off    - text only
        "mode": "auto",
        "language": "en",
        # Talk over the assistant to interrupt it. Best with headphones; with
        # open speakers the assistant can occasionally hear itself.
        "barge_in": True,
        # Optional: only react to speech containing this word ("jarvis").
        "wake_word": "",
        "follow_up_seconds": 20,
        "input_device": None,
        "output_device": None,
        "stt": {
            # auto | faster-whisper (local) | openai (cloud/compatible) | browser | mock
            "provider": "auto",
            "model": "base.en",
            "device": "auto",
            "compute_type": "int8",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "memory_mb": 500,
        },
        "tts": {
            # auto | piper (local neural) | system (OS voice) | openai | browser | mock
            "provider": "auto",
            "model": "",
            "voice": "alloy",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "memory_mb": 150,
        },
    },
    "server": {"host": "127.0.0.1", "port": 8765, "token": ""},
    "skills": {"plugin_dirs": ["./plugins"], "disabled": []},
    "agents": {"dirs": ["./agents"]},
    "trading": {
        # auto | binance | yahoo | synthetic
        "data_source": "auto",
        "binance_url": "https://data-api.binance.vision",
        "watchlist": ["BTCUSDT", "ETHUSDT", "AAPL", "NVDA"],
    },
    "publish": {
        "output_dir": "./workspace/published",
        "webhook_url": "",
        "wordpress_url": "",
        "wordpress_user": "",
        "wordpress_password_env": "WORDPRESS_APP_PASSWORD",
    },
    "security": {
        # Hosts you own or are authorised to assess. The scanner refuses
        # everything else (loopback is always allowed).
        "authorized_targets": [],
        # DNS-over-HTTPS resolver for the DNS/email check (it sees the domain names).
        "doh_url": "https://cloudflare-dns.com/dns-query",
    },
    "web": {"allow_private_hosts": False, "timeout_seconds": 20},
}

SEARCH_PATHS = ("./jarvis.toml", "~/.config/jarvis/jarvis.toml")


def _merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


class Config:
    """Nested settings with dotted-path access: ``cfg.get("llm.active")``."""

    def __init__(self, data: dict[str, Any] | None = None, path: Path | None = None):
        self.data = _merge(copy.deepcopy(DEFAULTS), data or {})
        self.path = path

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> Config:
        candidates = [path] if path else [os.environ.get("JARVIS_CONFIG"), *SEARCH_PATHS]
        for candidate in candidates:
            if not candidate:
                continue
            file = Path(candidate).expanduser()
            if file.is_file():
                with file.open("rb") as fh:
                    return cls(tomllib.load(fh), file)
            if path:
                raise FileNotFoundError(f"Config file not found: {file}")
        return cls()

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        node = self.data
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value

    # -- resolved paths ---------------------------------------------------
    @property
    def data_dir(self) -> Path:
        path = Path(self.get("assistant.data_dir")).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def workspace(self) -> Path:
        path = Path(self.get("assistant.workspace")).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def secret(self, env_name: str | None) -> str | None:
        return os.environ.get(env_name) if env_name else None
