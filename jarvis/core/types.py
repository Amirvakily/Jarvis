"""Provider-neutral conversation types shared by every backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class Message:
    """One conversation entry.

    ``role`` is ``user``, ``assistant`` or ``tool``. ``meta`` carries data
    that only one backend understands (for example Anthropic content blocks
    under ``meta["anthropic"]``) plus bookkeeping such as ``epoch``.
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    is_error: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        if self.is_error:
            data["is_error"] = True
        if self.meta:
            data["meta"] = self.meta
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Message:
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            tool_calls=[ToolCall(**call) for call in data.get("tool_calls", [])],
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
            is_error=data.get("is_error", False),
            meta=data.get("meta", {}),
        )


@dataclass
class ToolSpec:
    """A tool as advertised to a model."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class TextDelta:
    text: str


@dataclass
class LLMResponse:
    """Final event of a provider stream: the complete assistant message."""

    message: Message
    stop_reason: str = "end"  # end | tool_use | max_tokens | refusal
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""


class LLMError(RuntimeError):
    """A backend failed. ``retryable`` marks failures worth trying elsewhere."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable
