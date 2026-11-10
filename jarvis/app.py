"""The assembled assistant: backends, skills, agents, sessions and tasks."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

from jarvis.agents.base import AgentRegistry
from jarvis.config import Config
from jarvis.core import prompts
from jarvis.core.engine import Emit, Engine
from jarvis.core.session import Session
from jarvis.llm.base import LLMProvider
from jarvis.llm.router import LLMRouter
from jarvis.resources import ResourceManager
from jarvis.skills.base import SkillError
from jarvis.skills.registry import SkillRegistry
from jarvis.tasks import TaskManager

log = logging.getLogger(__name__)

MAX_DELEGATION_DEPTH = 1


class Jarvis:
    def __init__(self, config: Config | None = None, providers: list[LLMProvider] | None = None):
        self.config = config or Config.load()
        self.resources = ResourceManager(
            int(self.config.get("resources.memory_budget_mb", 8000)),
            float(self.config.get("resources.idle_unload_seconds", 600)),
        )
        self.router = LLMRouter(self.config, self.resources)
        for provider in providers or []:
            self.router.add_provider(provider)
        # The backend chosen last time (by voice, UI or CLI) survives restarts.
        self._state_file = self.config.data_dir / "state.json"
        remembered = self._load_state().get("active_backend")
        if remembered in self.router.profiles:
            self.router.active = remembered
        self.skills = SkillRegistry(
            self.config.get("skills.plugin_dirs", []), self.config.get("skills.disabled", [])
        ).load_all()
        self.agents = AgentRegistry(self.config.get("agents.dirs", []))
        self.engine = Engine(self.config, self.router, self.skills)
        self.tasks = TaskManager(self._run_background, self.broadcast)
        self._sessions: dict[str, Session] = {}
        self._conversations: dict[str, Any] = {}
        self._listeners: list[Callable[[dict], Any]] = []
        self.router.on_switch(self._on_backend_switch)

    def _load_state(self) -> dict:
        try:
            return json.loads(self._state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _on_backend_switch(self, name: str) -> None:
        state = {**self._load_state(), "active_backend": name}
        self._state_file.write_text(json.dumps(state), encoding="utf-8")
        self.broadcast({"type": "backend", **self.router.describe()})

    # -- events ---------------------------------------------------------------
    def subscribe(self, listener: Callable[[dict], Any]) -> Callable[[], None]:
        """Receive app-wide events (backend switches, background task updates)."""
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener) if listener in self._listeners else None

    def broadcast(self, event: dict) -> None:
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:
                log.exception("event listener failed")

    # -- conversations ----------------------------------------------------------
    def session(self, session_id: str = "main") -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session.load(session_id, self.config.data_dir / "sessions")
        return self._sessions[session_id]

    def conversation(self, session_id: str = "main"):
        from jarvis.conversation import Conversation

        if session_id not in self._conversations:
            self._conversations[session_id] = Conversation(self, self.session(session_id))
        return self._conversations[session_id]

    def services(self, depth: int = 0) -> dict[str, Any]:
        return {"jarvis": self, "llm": self.router, "depth": depth}

    async def chat(
        self, text: str, *, session: Session | None = None, emit: Emit | None = None, voice: bool = False
    ) -> str:
        """Run one orchestrator turn and return the reply."""

        def system(backend: str) -> str:
            return prompts.core_prompt(
                self.config, self.agents.specialists(), voice=voice, backend=backend
            )

        return await self.engine.run(
            agent=self.agents.core,
            system=system,
            session=session or self.session(),
            user_text=text,
            emit=emit,
            services=self.services(depth=0),
        )

    # -- agents -----------------------------------------------------------------
    async def run_agent(
        self, name: str, task: str, *, emit: Emit | None = None, depth: int = 1
    ) -> str:
        """Run a specialist on ``task`` in a fresh context and return its report."""
        agent = self.agents.get(name)
        if agent is None or agent is self.agents.core:
            names = ", ".join(a.name for a in self.agents.specialists())
            raise SkillError(f"No agent named '{name}'. Available agents: {names}.")
        if depth > MAX_DELEGATION_DEPTH:
            raise SkillError("Specialist agents cannot delegate further.")

        async def relay(event: dict) -> None:
            if emit is None:
                return
            if event["type"] == "text_delta":  # shown as activity, never spoken
                event = {**event, "type": "agent_text"}
            await emit(event)

        if emit:
            await emit({"type": "agent_start", "agent": agent.name, "task": task})
        ok = False
        try:
            report = await self.engine.run(
                agent=agent,
                system=prompts.specialist_prompt(agent),
                session=Session(f"{agent.name}-run"),
                user_text=task,
                emit=relay,
                services=self.services(depth=depth),
            )
            ok = True
            return report or "(the agent returned no report)"
        finally:
            if emit:
                await asyncio.shield(emit({"type": "agent_end", "agent": agent.name, "ok": ok}))

    async def run_agents(self, jobs: list[tuple[str, str]], emit: Emit | None = None) -> list[dict]:
        """Run several agents concurrently; one failing does not stop the rest."""

        async def one(name: str, task: str) -> dict:
            try:
                return {"agent": name, "task": task, "ok": True, "report": await self.run_agent(name, task, emit=emit)}
            except Exception as exc:
                return {"agent": name, "task": task, "ok": False, "report": str(exc)}

        return list(await asyncio.gather(*(one(name, task) for name, task in jobs)))

    async def _run_background(self, agent: str, task: str) -> str:
        return await self.run_agent(agent, task)

    # -- long-term notes ----------------------------------------------------------
    def notes(self) -> dict[str, str]:
        path = self.config.data_dir / "memory.json"
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except ValueError:
            return {}

    def save_notes(self, notes: dict[str, str]) -> None:
        path = self.config.data_dir / "memory.json"
        path.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")

    # -- introspection ------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        return {
            "assistant": self.config.get("assistant.name", "Jarvis"),
            "backend": self.router.describe(),
            "agents": [a.describe() for a in self.agents.all()],
            "skills": [
                {"name": s.name, "description": s.description, "tags": s.tags, "source": s.source}
                for s in self.skills.all()
            ],
            "skill_errors": self.skills.errors,
            "resources": self.resources.status(),
            "tasks": [t.describe() for t in self.tasks.list()],
        }

    async def aclose(self) -> None:
        await self.tasks.shutdown()
        for session in self._sessions.values():
            session.save()
        await self.resources.shutdown()
        await self.router.aclose()
