"""Offline backends: a scripted one for tests and a rule-based one for demos.

``provider = "mock"`` lets you try the whole stack (voice loop, agents,
skills, UI) with no model installed and no API key:

* ``!skill_name {"arg": 1}`` calls a skill directly.
* ``@agent task`` delegates to an agent (when delegation is available).
* anything else is echoed back.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Callable
from typing import Any

from jarvis.core.types import LLMError, LLMResponse, Message, TextDelta, ToolCall
from jarvis.llm.base import LLMProvider, StreamEvent, new_call_id

Scripted = str | Message | Exception | Callable[[list[Message]], "str | Message"]


class MockProvider(LLMProvider):
    def __init__(self, profile="mock", settings=None, config=None, script: list[Scripted] | None = None):
        super().__init__(profile, settings or {"provider": "mock", "model": "mock"}, config)  # type: ignore[arg-type]
        self.model = self.settings.get("model", "mock")
        self.script = list(script) if script is not None else None
        self.calls: list[dict[str, Any]] = []
        self.delay = float(self.settings.get("delay", 0.0))

    @property
    def is_local(self) -> bool:  # type: ignore[override]
        return bool(self.settings.get("local", False))

    async def stream(
        self, *, system, messages, tools, max_tokens=None
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append({"system": system, "messages": list(messages), "tools": list(tools)})
        if self.script is not None:
            if not self.script:
                raise LLMError("mock script exhausted")
            item = self.script.pop(0)
            if callable(item):
                item = item(messages)
            if isinstance(item, Exception):
                raise item
            message = Message("assistant", item) if isinstance(item, str) else item
        else:
            message = self._rule_based(messages, {t.name for t in tools})

        for piece in re.findall(r"\S+\s*", message.content):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield TextDelta(piece)
        stop = "tool_use" if message.tool_calls else "end"
        yield LLMResponse(message, stop, {"input_tokens": 0, "output_tokens": 0}, self.model)

    def _rule_based(self, messages: list[Message], tool_names: set[str]) -> Message:
        last = messages[-1] if messages else Message("user", "")
        if last.role == "tool":
            results = []
            for msg in reversed(messages):
                if msg.role != "tool":
                    break
                results.insert(0, f"{msg.name} returned: {msg.content[:400]}")
            return Message("assistant", " ".join(results))
        text = last.content.strip()
        direct = re.match(r"^!(\w+)\s*(\{.*\})?\s*$", text, flags=re.DOTALL)
        if direct and direct.group(1) in tool_names:
            try:
                arguments = json.loads(direct.group(2) or "{}")
            except json.JSONDecodeError:
                arguments = {}
            return Message("assistant", "", [ToolCall(new_call_id(), direct.group(1), arguments)])
        delegate = re.match(r"^@(\w+)\s+(.+)$", text, flags=re.DOTALL)
        if delegate and "delegate" in tool_names:
            arguments = {"agent": delegate.group(1), "task": delegate.group(2)}
            return Message("assistant", "", [ToolCall(new_call_id(), "delegate", arguments)])
        return Message("assistant", f"You said: {text}" if text else "I'm listening.")
