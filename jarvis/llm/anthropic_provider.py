"""Claude through the official Anthropic SDK.

Credentials are resolved by the SDK (``ANTHROPIC_API_KEY``,
``ANTHROPIC_AUTH_TOKEN`` or an ``ant auth login`` profile); nothing is read
from the config file unless a profile names its own ``api_key_env``.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from jarvis.core.types import LLMError, LLMResponse, Message, TextDelta, ToolCall, ToolSpec
from jarvis.llm.base import LLMProvider, StreamEvent

FALLBACK_BETA = "server-side-fallback-2026-07-01"
REFUSAL_TEXT = "I'm not able to help with that request."


def _block_to_param(block: Any) -> dict[str, Any]:
    """Serialise a response content block so it can be replayed verbatim."""
    kind = block.type
    if kind == "text":
        return {"type": "text", "text": block.text}
    if kind == "thinking":
        return {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
    if kind == "redacted_thinking":
        return {"type": "redacted_thinking", "data": block.data}
    if kind == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return block.model_dump(mode="json", exclude_none=True)


def to_anthropic_messages(messages: list[Message], *, native: bool = True) -> list[dict[str, Any]]:
    """Convert neutral history to Messages API shape.

    With ``native`` the exact content blocks Claude produced (kept under
    ``meta["anthropic"]``) are replayed, which preserves thinking blocks.
    Without it, turns are rebuilt from plain text and tool calls only.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "tool":
            result = {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id,
                "content": msg.content or "(no output)",
            }
            if msg.is_error:
                result["is_error"] = True
            # All results of one assistant turn must share a single user message.
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(result)
            else:
                out.append({"role": "user", "content": [result]})
        elif msg.role == "assistant":
            blocks = msg.meta.get("anthropic") if native else None
            if not blocks:
                blocks = [{"type": "text", "text": msg.content}] if msg.content.strip() else []
                blocks += [
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                    for c in msg.tool_calls
                ]
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        elif msg.content.strip():
            out.append({"role": "user", "content": msg.content})
    return out


def to_anthropic_tools(tools: list[ToolSpec], eager: bool) -> list[dict[str, Any]]:
    out = []
    for tool in tools:
        entry: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }
        if eager:
            # Stream tool inputs as they are generated. Inputs are validated
            # against the skill schema before anything runs (skills/base.py).
            entry["eager_input_streaming"] = True
        out.append(entry)
    return out


class AnthropicProvider(LLMProvider):
    def __init__(self, profile, settings, config, client: Any = None):
        super().__init__(profile, settings, config)
        self.model = settings.get("model", "claude-opus-5-5")
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            kwargs: dict[str, Any] = {}
            key = self.config.secret(self.settings.get("api_key_env"))
            if key:
                kwargs["api_key"] = key
            if self.settings.get("base_url"):
                kwargs["base_url"] = self.settings["base_url"]
            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    def _params(self, system, messages, tools, max_tokens, native: bool) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or int(self.settings.get("max_tokens", 16000)),
            "messages": to_anthropic_messages(messages, native=native),
            "cache_control": {"type": "ephemeral"},
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = to_anthropic_tools(
                tools, bool(self.settings.get("eager_tool_streaming", True))
            )
        if self.settings.get("thinking", True):
            params["thinking"] = {"type": "adaptive"}
        if self.settings.get("effort"):
            params["output_config"] = {"effort": self.settings["effort"]}
        if self.settings.get("refusal_fallback", True):
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    async def stream(
        self, *, system, messages, tools, max_tokens=None
    ) -> AsyncIterator[StreamEvent]:
        import anthropic

        client = self._get_client()
        has_native = any(m.meta.get("anthropic") for m in messages)
        native, json_retries, emitted = True, 0, False
        while True:
            params = self._params(system, messages, tools, max_tokens, native)
            api = client.beta.messages if "betas" in params else client.messages
            try:
                async with api.stream(**params) as stream:
                    async for event in stream:
                        if event.type == "text":
                            emitted = True
                            yield TextDelta(event.text)
                    final = await stream.get_final_message()
                break
            except ValueError as exc:
                # The SDK could not parse a streamed tool input at all.
                json_retries += 1
                if emitted or json_retries > 2:
                    raise LLMError(f"{self.profile}: malformed tool input from model") from exc
            except anthropic.BadRequestError as exc:
                if native and has_native and not emitted:
                    # Replayed thinking blocks no longer match this conversation
                    # (edited history or a model change). Resend without them.
                    native = False
                    continue
                raise LLMError(f"{self.profile}: request rejected: {exc.message}") from exc
            except anthropic.AuthenticationError as exc:
                raise LLMError(
                    f"{self.profile}: Anthropic credentials were rejected. "
                    "Set ANTHROPIC_API_KEY or run `ant auth login`."
                ) from exc
            except anthropic.PermissionDeniedError as exc:
                raise LLMError(f"{self.profile}: permission denied: {exc.message}") from exc
            except anthropic.NotFoundError as exc:
                raise LLMError(f"{self.profile}: model '{self.model}' not found") from exc
            except anthropic.RateLimitError as exc:
                raise LLMError(f"{self.profile}: rate limited by Anthropic", retryable=True) from exc
            except anthropic.APIStatusError as exc:
                raise LLMError(
                    f"{self.profile}: Anthropic API error {exc.status_code}: {exc.message}",
                    retryable=exc.status_code >= 500,
                ) from exc
            except anthropic.APIConnectionError as exc:
                raise LLMError(
                    f"{self.profile}: cannot reach the Anthropic API", retryable=True
                ) from exc
            except anthropic.AnthropicError as exc:
                raise LLMError(
                    f"{self.profile}: {exc}. Set ANTHROPIC_API_KEY or run `ant auth login`."
                ) from exc

        usage = {
            "input_tokens": getattr(final.usage, "input_tokens", 0) or 0,
            "output_tokens": getattr(final.usage, "output_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(final.usage, "cache_read_input_tokens", 0) or 0,
        }
        served_by = getattr(final, "model", self.model) or self.model

        # Check the stop reason before trusting any content.
        if final.stop_reason == "refusal":
            text = "" if emitted else REFUSAL_TEXT
            if not emitted:
                yield TextDelta(text)
            yield LLMResponse(Message("assistant", text), "refusal", usage, served_by)
            return

        text = "".join(b.text for b in final.content if b.type == "text")
        calls = [
            ToolCall(b.id, b.name, b.input if isinstance(b.input, dict) else {})
            for b in final.content
            if b.type == "tool_use"
        ]
        if final.stop_reason == "max_tokens" and calls:
            # A cut-off tool input can parse as a valid partial object: never run it.
            yield LLMResponse(Message("assistant", text), "max_tokens", usage, served_by)
            return

        message = Message("assistant", text, tool_calls=calls)
        message.meta["anthropic"] = [_block_to_param(b) for b in final.content]
        stop = "tool_use" if calls else ("max_tokens" if final.stop_reason == "max_tokens" else "end")
        yield LLMResponse(message, stop, usage, served_by)

    async def health(self) -> dict[str, Any]:
        named = self.settings.get("api_key_env")
        if named and self.config.secret(named):
            return {"ok": True, "detail": ""}
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            return {"ok": True, "detail": ""}
        if (Path.home() / ".config" / "anthropic").is_dir():
            return {"ok": True, "detail": "using `ant auth login` profile"}
        return {"ok": False, "detail": "no credentials: set ANTHROPIC_API_KEY or run `ant auth login`"}

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
