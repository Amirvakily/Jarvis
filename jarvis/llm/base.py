"""Backend interface. Every provider turns the neutral conversation into its
own wire format and streams back text deltas followed by one LLMResponse."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

from jarvis.config import Config
from jarvis.core.types import LLMResponse, Message, TextDelta, ToolSpec

StreamEvent = TextDelta | LLMResponse


class LLMProvider:
    #: Local providers hold model weights in memory and are managed by the
    #: ResourceManager (loaded on demand, unloaded when idle).
    is_local = False
    default_concurrency = 8

    def __init__(self, profile: str, settings: dict[str, Any], config: Config):
        self.profile = profile
        self.settings = settings
        self.config = config
        self.model: str = settings.get("model", "")

    @property
    def memory_mb(self) -> int:
        return int(self.settings.get("memory_mb", 0))

    @property
    def max_concurrency(self) -> int:
        return int(self.settings.get("max_concurrency", self.default_concurrency))

    def stream(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError

    async def load(self) -> Any:
        """Bring the model into memory. Returns a handle for :meth:`unload`."""
        return self

    async def unload(self, handle: Any = None) -> None:
        """Free the model's memory."""

    async def health(self) -> dict[str, Any]:
        return {"ok": True, "detail": ""}

    async def aclose(self) -> None:
        pass

    def describe(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "provider": self.settings.get("provider"),
            "model": self.model,
            "local": self.is_local,
        }


def new_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:16]
