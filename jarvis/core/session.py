"""Conversation state that survives restarts."""

from __future__ import annotations

import json
import re
from pathlib import Path

from jarvis.core.types import Message


class Session:
    def __init__(self, session_id: str = "main", path: Path | None = None):
        self.id = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id)[:64] or "main"
        self.path = path
        self.messages: list[Message] = []
        # Bumped whenever earlier context changes (trimmed history, different
        # system prompt, tools or backend). Provider-specific payloads from an
        # older epoch are not replayed; see Engine._view.
        self.epoch = 0
        self.fingerprint: str | None = None

    @classmethod
    def load(cls, session_id: str, directory: Path) -> Session:
        session = cls(session_id)
        session.path = directory / f"{session.id}.json"
        if session.path.is_file():
            try:
                data = json.loads(session.path.read_text(encoding="utf-8"))
                session.messages = [Message.from_dict(m) for m in data.get("messages", [])]
                session.epoch = int(data.get("epoch", 0)) + 1  # restart = new epoch
            except (ValueError, KeyError, TypeError):
                session.messages = []
        return session

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"epoch": self.epoch, "messages": [m.to_dict() for m in self.messages]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, message: Message) -> Message:
        self.messages.append(message)
        return message

    def clear(self) -> None:
        self.messages.clear()
        self.epoch += 1
        self.save()

    def trim(self, max_messages: int) -> bool:
        """Drop the oldest whole turns once the history exceeds ``max_messages``.

        Cuts only at a user message so a tool call is never separated from
        its result.
        """
        if len(self.messages) <= max_messages:
            return False
        start = len(self.messages) - max_messages
        for index in range(start, len(self.messages)):
            if self.messages[index].role == "user":
                del self.messages[:index]
                self.epoch += 1
                return True
        return False

    def transcript(self, limit: int = 50) -> list[dict]:
        """User/assistant text only, for display."""
        rows = [
            {"role": m.role, "text": m.content, "interrupted": bool(m.meta.get("interrupted"))}
            for m in self.messages
            if m.role in ("user", "assistant") and m.content.strip()
        ]
        return rows[-limit:]
