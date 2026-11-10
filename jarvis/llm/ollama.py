"""Local models through Ollama (https://ollama.com).

The default local profile runs ``llama3.1:8b``, which fits a 16 GB laptop.
Loading and unloading are explicit so the ResourceManager can decide what
stays in memory: ``load`` warms the model, ``unload`` evicts it at once
(``keep_alive: 0``) instead of waiting for Ollama's own timer.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from jarvis._http import httpx
from jarvis.core.types import LLMError, LLMResponse, Message, TextDelta, ToolCall, ToolSpec
from jarvis.llm.base import LLMProvider, StreamEvent, new_call_id


def to_ollama_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
    for msg in messages:
        if msg.role == "tool":
            out.append({"role": "tool", "content": msg.content, "tool_name": msg.name or ""})
        elif msg.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {"function": {"name": call.name, "arguments": call.arguments}}
                    for call in msg.tool_calls
                ]
            out.append(entry)
        else:
            out.append({"role": "user", "content": msg.content})
    return out


def to_ollama_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


class OllamaProvider(LLMProvider):
    is_local = True
    default_concurrency = 2

    def __init__(self, profile, settings, config, transport: Any = None):
        super().__init__(profile, settings, config)
        self.base_url = settings.get("base_url", "http://localhost:11434").rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(float(settings.get("timeout", 600)), connect=5.0),
            transport=transport,
        )

    def _options(self) -> dict[str, Any]:
        options = {}
        for key in ("num_ctx", "temperature", "num_predict", "num_gpu", "num_thread"):
            if key in self.settings:
                options[key] = self.settings[key]
        return options

    async def stream(
        self, *, system, messages, tools, max_tokens=None
    ) -> AsyncIterator[StreamEvent]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": to_ollama_messages(system, messages),
            "stream": True,
            "keep_alive": self.settings.get("keep_alive", "10m"),
            "options": self._options(),
        }
        if tools:
            payload["tools"] = to_ollama_tools(tools)
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        text: list[str] = []
        calls: list[ToolCall] = []
        usage: dict[str, int] = {}
        done_reason = "stop"
        try:
            async with self._client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode(errors="replace")[:500]
                    hint = ""
                    if response.status_code == 404:
                        hint = f" Pull the model first: ollama pull {self.model}"
                    raise LLMError(
                        f"{self.profile}: Ollama returned HTTP {response.status_code}: {body}.{hint}",
                        retryable=response.status_code >= 500,
                    )
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise LLMError(f"{self.profile}: unreadable response from Ollama") from exc
                    if chunk.get("error"):
                        raise LLMError(f"{self.profile}: Ollama error: {chunk['error']}")
                    message = chunk.get("message") or {}
                    for call in message.get("tool_calls") or []:
                        function = call.get("function") or {}
                        arguments = function.get("arguments") or {}
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                arguments = {"INVALID_JSON": arguments}
                        calls.append(ToolCall(new_call_id(), function.get("name", ""), arguments))
                    piece = message.get("content") or ""
                    if piece:
                        text.append(piece)
                        yield TextDelta(piece)
                    if chunk.get("done"):
                        done_reason = chunk.get("done_reason", "stop")
                        usage = {
                            "input_tokens": chunk.get("prompt_eval_count", 0),
                            "output_tokens": chunk.get("eval_count", 0),
                        }
        except httpx.HTTPError as exc:
            raise LLMError(
                f"{self.profile}: cannot reach Ollama at {self.base_url}. "
                f"Is it running? ({type(exc).__name__})",
                retryable=True,
            ) from exc

        stop = "tool_use" if calls else ("max_tokens" if done_reason == "length" else "end")
        yield LLMResponse(Message("assistant", "".join(text), tool_calls=calls), stop, usage, self.model)

    async def _keep_alive(self, value: Any) -> None:
        try:
            response = await self._client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "keep_alive": value},
                timeout=180.0,
            )
        except httpx.HTTPError as exc:
            raise LLMError(
                f"{self.profile}: cannot reach Ollama at {self.base_url}. Is it running?",
                retryable=True,
            ) from exc
        if response.status_code == 404:
            raise LLMError(
                f"{self.profile}: model '{self.model}' is not installed. "
                f"Run: ollama pull {self.model}"
            )

    async def load(self) -> Any:
        await self._keep_alive(self.settings.get("keep_alive", "10m"))
        return self

    async def unload(self, handle: Any = None) -> None:
        await self._keep_alive(0)

    async def health(self) -> dict[str, Any]:
        try:
            response = await self._client.get(f"{self.base_url}/api/tags", timeout=3.0)
            names = [m.get("name", "") for m in response.json().get("models", [])]
        except (httpx.HTTPError, ValueError):
            return {"ok": False, "detail": f"Ollama is not reachable at {self.base_url}"}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names and self.model not in names:
            return {"ok": False, "detail": f"model missing; run: ollama pull {self.model}"}
        return {"ok": True, "detail": ""}

    async def aclose(self) -> None:
        await self._client.aclose()
