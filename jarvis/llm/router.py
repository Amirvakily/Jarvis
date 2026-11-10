"""The hybrid switch: one place that decides which backend serves a request.

Profiles are named in the config (``local``, ``claude``, ``openai``, or any
you add). Exactly one is active at a time and it can be changed at runtime.
Providers are built lazily, so a profile you never use costs nothing, and
local ones are loaded/unloaded through the ResourceManager.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from jarvis.config import Config
from jarvis.core.types import LLMError, LLMResponse, Message, ToolSpec
from jarvis.llm.base import LLMProvider, StreamEvent
from jarvis.resources import ResourceManager

log = logging.getLogger(__name__)


def _build(profile: str, settings: dict[str, Any], config: Config) -> LLMProvider:
    kind = settings.get("provider", "")
    if kind == "ollama":
        from jarvis.llm.ollama import OllamaProvider

        return OllamaProvider(profile, settings, config)
    if kind == "anthropic":
        from jarvis.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(profile, settings, config)
    if kind in ("openai", "openai-compatible", "custom"):
        from jarvis.llm.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(profile, settings, config)
    if kind == "llamacpp":
        from jarvis.llm.llamacpp import LlamaCppProvider

        return LlamaCppProvider(profile, settings, config)
    if kind == "mock":
        from jarvis.llm.mock import MockProvider

        return MockProvider(profile, settings, config)
    raise LLMError(f"Profile '{profile}' has unknown provider '{kind}'.")


class LLMRouter:
    def __init__(self, config: Config, resources: ResourceManager | None = None):
        self.config = config
        self.resources = resources or ResourceManager()
        self._providers: dict[str, LLMProvider] = {}
        self._limits: dict[str, asyncio.Semaphore] = {}
        self._listeners: list[Callable[[str], Any]] = []
        self.active: str = config.get("llm.active", "local")
        if self.active not in self.profiles:
            raise LLMError(
                f"llm.active is '{self.active}' but no such profile exists "
                f"(have: {', '.join(self.profiles) or 'none'})."
            )

    @property
    def profiles(self) -> dict[str, dict[str, Any]]:
        return self.config.get("llm.profiles", {})

    def on_switch(self, listener: Callable[[str], Any]) -> None:
        self._listeners.append(listener)

    def add_provider(self, provider: LLMProvider) -> None:
        """Inject a ready-made provider (tests, embedding Jarvis in other code)."""
        self.profiles.setdefault(provider.profile, provider.settings)
        self._providers[provider.profile] = provider
        self._register_resource(provider)

    def provider(self, profile: str | None = None) -> LLMProvider:
        name = profile or self.active
        if name not in self._providers:
            if name not in self.profiles:
                raise LLMError(f"Unknown backend profile '{name}'.")
            self._providers[name] = _build(name, self.profiles[name], self.config)
            self._register_resource(self._providers[name])
        return self._providers[name]

    def _register_resource(self, provider: LLMProvider) -> None:
        if provider.is_local:
            self.resources.register(
                f"llm:{provider.profile}",
                provider.load,
                provider.unload,
                memory_mb=provider.memory_mb,
                kind="llm",
            )

    def switch(self, profile: str) -> dict[str, Any]:
        """Make ``profile`` the active backend. Accepts a profile name or the
        words ``local`` / ``cloud`` to pick the first matching profile."""
        name = self._resolve(profile)
        self.active = name
        for listener in self._listeners:
            listener(name)
        return self.provider(name).describe()

    def _resolve(self, wanted: str) -> str:
        wanted = wanted.strip().lower()
        if wanted in self.profiles:
            return wanted
        local_kinds = ("ollama", "llamacpp")
        for name, settings in self.profiles.items():
            is_local = settings.get("provider") in local_kinds or settings.get("local")
            if (wanted in ("local", "offline") and is_local) or (
                wanted in ("cloud", "api", "online", "remote") and not is_local
                and settings.get("provider") != "mock"
            ):
                return name
            if wanted in (settings.get("provider"), str(settings.get("model", "")).lower()):
                return name
        raise LLMError(f"No backend matches '{wanted}'. Profiles: {', '.join(self.profiles)}.")

    def describe(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "fallback": self.config.get("llm.fallback") or None,
            "profiles": [
                {
                    "name": name,
                    "provider": settings.get("provider"),
                    "model": settings.get("model") or settings.get("model_path", ""),
                    "local": settings.get("provider") in ("ollama", "llamacpp")
                    or bool(settings.get("local")),
                }
                for name, settings in self.profiles.items()
            ],
        }

    async def stream(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        profile: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream from ``profile`` (default: the active one). If it cannot be
        reached before producing output, try the configured fallback once."""
        primary = profile or self.active
        fallback = self.config.get("llm.fallback") or None
        candidates = [primary] + ([fallback] if fallback and fallback != primary else [])
        for index, name in enumerate(candidates):
            emitted = False
            try:
                async for event in self._stream_one(name, system, messages, tools, max_tokens):
                    emitted = True
                    yield event
                return
            except LLMError as exc:
                is_last = index == len(candidates) - 1
                if emitted or is_last or not exc.retryable:
                    raise
                log.warning("backend %s failed (%s); falling back to %s", name, exc, candidates[index + 1])

    async def _stream_one(self, name, system, messages, tools, max_tokens):
        provider = self.provider(name)
        limit = self._limits.setdefault(name, asyncio.Semaphore(provider.max_concurrency))
        resource = f"llm:{name}"
        async with limit:
            managed = resource in self.resources
            if managed:
                await self.resources.acquire(resource)
            try:
                async for event in provider.stream(
                    system=system, messages=messages, tools=tools, max_tokens=max_tokens
                ):
                    yield event
            finally:
                if managed:
                    self.resources.release(resource)

    async def complete(self, system: str, prompt: str, profile: str | None = None) -> str:
        """One-shot text completion without tools."""
        text = ""
        async for event in self.stream(
            system=system, messages=[Message("user", prompt)], tools=[], profile=profile
        ):
            if isinstance(event, LLMResponse):
                text = event.message.content
        return text

    async def health(self) -> dict[str, dict[str, Any]]:
        report = {}
        for name in self.profiles:
            try:
                report[name] = await self.provider(name).health()
            except LLMError as exc:
                report[name] = {"ok": False, "detail": str(exc)}
        return report

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()
