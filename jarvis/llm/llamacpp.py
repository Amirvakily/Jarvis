"""Run a GGUF model in-process with llama-cpp-python (no Ollama needed).

    [llm.profiles.gguf]
    provider = "llamacpp"
    model_path = "~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    n_ctx = 8192
    n_gpu_layers = -1      # offload everything that fits; 0 = CPU only
    memory_mb = 5500

The weights are only read from disk when a request needs this profile and
are dropped again by the ResourceManager when idle or when memory is tight.
"""

from __future__ import annotations

import asyncio
import gc
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from jarvis.core.types import LLMError, TextDelta
from jarvis.llm.base import LLMProvider, StreamEvent
from jarvis.llm.openai_compat import ChunkAccumulator, to_openai_messages, to_openai_tools


class LlamaCppProvider(LLMProvider):
    is_local = True
    default_concurrency = 1  # one llama context, one generation at a time

    def __init__(self, profile, settings, config):
        super().__init__(profile, settings, config)
        self.model_path = Path(settings.get("model_path", "")).expanduser()
        self.model = settings.get("model") or self.model_path.name
        self._llm: Any = None

    def _load_sync(self) -> Any:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise LLMError(
                f"{self.profile}: llama-cpp-python is not installed. "
                "Run: pip install 'jarvis-assistant[llamacpp]'"
            ) from exc
        if not self.model_path.is_file():
            raise LLMError(f"{self.profile}: model file not found: {self.model_path}")
        return Llama(
            model_path=str(self.model_path),
            n_ctx=int(self.settings.get("n_ctx", 8192)),
            n_gpu_layers=int(self.settings.get("n_gpu_layers", -1)),
            n_threads=self.settings.get("n_threads"),
            # A chat format that understands OpenAI-style tool definitions.
            chat_format=self.settings.get("chat_format", "chatml-function-calling"),
            verbose=False,
        )

    async def load(self) -> Any:
        self._llm = await asyncio.to_thread(self._load_sync)
        return self._llm

    async def unload(self, handle: Any = None) -> None:
        llm, self._llm = self._llm, None
        if llm is not None and hasattr(llm, "close"):
            llm.close()
        del llm, handle
        gc.collect()

    async def stream(
        self, *, system, messages, tools, max_tokens=None
    ) -> AsyncIterator[StreamEvent]:
        if self._llm is None:
            await self.load()
        llm = self._llm
        kwargs: dict[str, Any] = {
            "messages": to_openai_messages(system, messages),
            "stream": True,
            "max_tokens": max_tokens or self.settings.get("max_tokens"),
        }
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"
        if "temperature" in self.settings:
            kwargs["temperature"] = self.settings["temperature"]

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()
        done = object()

        def worker() -> None:
            try:
                for chunk in llm.create_chat_completion(**kwargs):
                    if stop.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
                loop.call_soon_threadsafe(queue.put_nowait, done)
            except Exception as exc:  # surfaced on the event loop below
                loop.call_soon_threadsafe(queue.put_nowait, exc)

        thread = threading.Thread(target=worker, daemon=True, name=f"llama-{self.profile}")
        thread.start()
        acc = ChunkAccumulator()
        try:
            while True:
                item = await queue.get()
                if item is done:
                    break
                if isinstance(item, Exception):
                    raise LLMError(f"{self.profile}: llama.cpp failed: {item}") from item
                text = acc.feed(item)
                if text:
                    yield TextDelta(text)
        finally:
            stop.set()  # also reached on cancellation (barge-in)
        yield acc.finish(self.model)

    async def health(self) -> dict[str, Any]:
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            return {"ok": False, "detail": "llama-cpp-python not installed"}
        if not self.model_path.is_file():
            return {"ok": False, "detail": f"model file not found: {self.model_path}"}
        return {"ok": True, "detail": ""}
