"""The reasoning loop shared by the orchestrator and every specialist agent.

One turn = stream the model, run any skills it asked for (concurrently),
feed the results back, repeat until it answers in plain text. The loop is
backend-agnostic and safe to cancel at any point: an interrupted turn
always leaves the session in a state every provider accepts.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from jarvis.agents.base import AgentSpec
from jarvis.config import Config
from jarvis.core.session import Session
from jarvis.core.types import LLMError, LLMResponse, Message, TextDelta, ToolCall, ToolSpec
from jarvis.llm.router import LLMRouter
from jarvis.skills.base import SkillContext
from jarvis.skills.registry import SkillRegistry

log = logging.getLogger(__name__)

Emit = Callable[[dict], Awaitable[None]]
INTERRUPTED = "Interrupted by the user before this finished."


async def _noop(_: dict) -> None:
    return None


class Engine:
    def __init__(self, config: Config, router: LLMRouter, skills: SkillRegistry):
        self.config = config
        self.router = router
        self.skills = skills

    async def run(
        self,
        *,
        agent: AgentSpec,
        system: str | Callable[[str], str],
        session: Session,
        user_text: str,
        emit: Emit | None = None,
        services: dict[str, Any] | None = None,
    ) -> str:
        """Run one user turn for ``agent`` and return its final text.

        ``system`` may be a callable taking the backend profile name, for
        prompts that mention which backend is in use.
        """
        emit = emit or _noop
        max_rounds = agent.max_rounds or int(self.config.get("assistant.max_tool_rounds", 8))
        session.trim(int(self.config.get("assistant.max_history_messages", 60)))
        checkpoint = len(session.messages)
        session.add(Message("user", user_text))
        ctx = SkillContext(self.config, agent=agent.name, emit=emit, services=services or {})
        final_text = ""
        try:
            for round_index in range(max_rounds + 1):
                # The last round offers no tools, forcing a spoken wrap-up
                # instead of an endless tool loop.
                tools = self.skills.specs(agent.skills) if round_index < max_rounds else []
                profile = self._profile_for(agent)
                prompt = system(profile) if callable(system) else system
                response = await self._model_step(session, profile, prompt, tools, emit, agent.name)
                final_text = response.message.content
                calls = response.message.tool_calls
                if not calls or response.stop_reason != "tool_use":
                    break
                await self._run_tools(session, calls, ctx, emit)
        except LLMError:
            # Leave no half-finished turn behind; the caller reports the error.
            del session.messages[checkpoint:]
            raise
        return final_text

    def _profile_for(self, agent: AgentSpec) -> str:
        if agent.llm_profile and agent.llm_profile in self.router.profiles:
            return agent.llm_profile
        return self.router.active

    def _view(self, session: Session, profile: str, system: str, tools: list[ToolSpec]) -> list[Message]:
        """History as sent to the model.

        Provider-specific payloads (Claude's thinking blocks) are only valid
        while everything before them is unchanged. When the backend, system
        prompt, tool set or earlier history changes, the epoch moves on and
        older turns are sent as plain text and tool calls instead.
        """
        material = json.dumps(
            [profile, system, sorted((t.name, json.dumps(t.parameters, sort_keys=True)) for t in tools)]
        )
        fingerprint = hashlib.sha256(material.encode()).hexdigest()
        if session.fingerprint != fingerprint:
            session.fingerprint = fingerprint
            session.epoch += 1
        view = []
        for message in session.messages:
            if "anthropic" in message.meta and message.meta.get("epoch") != session.epoch:
                meta = {k: v for k, v in message.meta.items() if k != "anthropic"}
                message = dataclasses.replace(message, meta=meta)
            view.append(message)
        return view

    async def _model_step(
        self, session: Session, profile: str, system: str, tools: list[ToolSpec], emit: Emit, agent: str
    ) -> LLMResponse:
        view = self._view(session, profile, system, tools)
        pieces: list[str] = []
        response: LLMResponse | None = None
        try:
            async for event in self.router.stream(
                system=system, messages=view, tools=tools, profile=profile
            ):
                if isinstance(event, TextDelta):
                    pieces.append(event.text)
                    await emit({"type": "text_delta", "text": event.text, "agent": agent})
                else:
                    response = event
        except asyncio.CancelledError:
            heard = "".join(pieces).strip()
            if heard:
                meta = {"interrupted": True, "epoch": session.epoch}
                session.add(Message("assistant", heard, meta=meta))
            raise
        if response is None:
            raise LLMError(f"Backend '{profile}' ended the stream without a response.")
        message = response.message
        message.meta["epoch"] = session.epoch
        if message.content.strip() or message.tool_calls:
            session.add(message)
        if response.usage:
            await emit({"type": "usage", "agent": agent, "model": response.model, **response.usage})
        return response

    async def _run_tools(
        self, session: Session, calls: list[ToolCall], ctx: SkillContext, emit: Emit
    ) -> None:
        async def run_one(call: ToolCall) -> Message:
            await emit(
                {
                    "type": "tool_start",
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                    "agent": ctx.agent,
                }
            )
            text, is_error = await self.skills.execute(call.name, call.arguments, ctx)
            await emit(
                {
                    "type": "tool_end",
                    "id": call.id,
                    "name": call.name,
                    "ok": not is_error,
                    "preview": text[:300],
                    "agent": ctx.agent,
                }
            )
            return Message("tool", text, tool_call_id=call.id, name=call.name, is_error=is_error)

        try:
            results = await asyncio.gather(*(run_one(call) for call in calls))
        except asyncio.CancelledError:
            # Every tool call needs a result or the next request is invalid.
            for call in calls:
                session.add(
                    Message("tool", INTERRUPTED, tool_call_id=call.id, name=call.name, is_error=True)
                )
            raise
        for result in results:
            session.add(result)
