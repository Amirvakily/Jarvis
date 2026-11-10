"""OpenAI-compatible chat completions.

Covers OpenAI itself and any server speaking the same protocol: llama.cpp's
``llama-server``, LM Studio, vLLM, Groq, OpenRouter, a company gateway...
Point ``base_url`` at it and, if it needs one, name the key's environment
variable in ``api_key_env``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from jarvis._http import httpx
from jarvis.core.types import LLMError, LLMResponse, Message, TextDelta, ToolCall, ToolSpec
from jarvis.llm.base import LLMProvider, StreamEvent, new_call_id


def to_openai_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
    for msg in messages:
        if msg.role == "tool":
            out.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content})
        elif msg.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": msg.content or None}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    }
                    for call in msg.tool_calls
                ]
            elif not msg.content:
                entry["content"] = ""
            out.append(entry)
        else:
            out.append({"role": "user", "content": msg.content})
    return out


def to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


class ChunkAccumulator:
    """Folds OpenAI-format stream chunks into one assistant message."""

    def __init__(self) -> None:
        self.text: list[str] = []
        self.calls: dict[int, dict[str, str]] = {}
        self.finish_reason: str | None = None
        self.usage: dict[str, int] = {}

    def feed(self, chunk: dict[str, Any]) -> str:
        """Consume a chunk; return any new text."""
        if chunk.get("usage"):
            usage = chunk["usage"]
            self.usage = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            }
        choices = chunk.get("choices") or []
        if not choices:
            return ""
        choice = choices[0]
        if choice.get("finish_reason"):
            self.finish_reason = choice["finish_reason"]
        delta = choice.get("delta") or {}
        for part in delta.get("tool_calls") or []:
            slot = self.calls.setdefault(part.get("index", 0), {"id": "", "name": "", "args": ""})
            function = part.get("function") or {}
            slot["id"] = part.get("id") or slot["id"]
            slot["name"] += function.get("name") or ""
            slot["args"] += function.get("arguments") or ""
        text = delta.get("content") or ""
        if text:
            self.text.append(text)
        return text

    def finish(self, model: str) -> LLMResponse:
        calls: list[ToolCall] = []
        for _, slot in sorted(self.calls.items()):
            try:
                arguments = json.loads(slot["args"]) if slot["args"].strip() else {}
            except json.JSONDecodeError:
                # Handed to the skill validator, which reports it to the model.
                arguments = {"INVALID_JSON": slot["args"]}
            calls.append(ToolCall(slot["id"] or new_call_id(), slot["name"], arguments))
        if calls:
            stop = "tool_use"
        elif self.finish_reason == "length":
            stop = "max_tokens"
        elif self.finish_reason == "content_filter":
            stop = "refusal"
        else:
            stop = "end"
        message = Message("assistant", "".join(self.text), tool_calls=calls)
        return LLMResponse(message, stop, self.usage, model)


class OpenAICompatProvider(LLMProvider):
    def __init__(self, profile, settings, config, transport: Any = None):
        super().__init__(profile, settings, config)
        self.base_url = settings.get("base_url", "https://api.openai.com/v1").rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(float(settings.get("timeout", 300)), connect=10.0),
            transport=transport,
        )

    @property
    def is_local(self) -> bool:  # type: ignore[override]
        return bool(self.settings.get("local", False))

    def _headers(self) -> dict[str, str]:
        key = self.config.secret(self.settings.get("api_key_env"))
        return {"Authorization": f"Bearer {key}"} if key else {}

    async def stream(
        self, *, system, messages, tools, max_tokens=None
    ) -> AsyncIterator[StreamEvent]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(system, messages),
            "stream": True,
        }
        if tools:
            payload["tools"] = to_openai_tools(tools)
        limit = max_tokens or self.settings.get("max_tokens")
        if limit:
            payload[self.settings.get("max_tokens_param", "max_tokens")] = limit
        if "temperature" in self.settings:
            payload["temperature"] = self.settings["temperature"]
        if self.settings.get("stream_usage", True):
            payload["stream_options"] = {"include_usage": True}

        acc = ChunkAccumulator()
        try:
            async with self._client.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload, headers=self._headers()
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode(errors="replace")[:500]
                    raise LLMError(
                        f"{self.profile}: HTTP {response.status_code} from {self.base_url}: {body}",
                        retryable=response.status_code in (408, 429) or response.status_code >= 500,
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue  # keep-alive comments and partial lines from proxies
                    if isinstance(chunk, dict) and chunk.get("error"):
                        raise LLMError(f"{self.profile}: {chunk['error']}")
                    text = acc.feed(chunk)
                    if text:
                        yield TextDelta(text)
        except httpx.HTTPError as exc:
            raise LLMError(
                f"{self.profile}: cannot reach {self.base_url} ({type(exc).__name__}: {exc})",
                retryable=True,
            ) from exc
        yield acc.finish(self.model)

    async def health(self) -> dict[str, Any]:
        if self.settings.get("api_key_env") and not self._headers():
            return {"ok": False, "detail": f"{self.settings['api_key_env']} is not set"}
        try:
            response = await self._client.get(
                f"{self.base_url}/models", headers=self._headers(), timeout=5.0
            )
            ok = response.status_code < 400
            return {"ok": ok, "detail": "" if ok else f"HTTP {response.status_code}"}
        except httpx.HTTPError as exc:
            return {"ok": False, "detail": f"unreachable: {exc}"}

    async def aclose(self) -> None:
        await self._client.aclose()
