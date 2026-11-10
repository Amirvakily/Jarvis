"""Core skills: delegation, background work, backend switching, memory."""

from __future__ import annotations

import datetime as dt
import platform

from jarvis.core.types import LLMError
from jarvis.skills.base import SkillContext, SkillError, skill


def _app(ctx: SkillContext):
    app = ctx.services.get("jarvis")
    if app is None:
        raise SkillError("This skill needs a running Jarvis instance.")
    return app


@skill(tags=["core"], timeout=900)
async def delegate(agent: str, task: str, ctx: SkillContext) -> str:
    """Hand a task to a specialist agent and wait for its report.

    The agent works in its own context and cannot see this conversation, so
    put everything it needs into the task. To run several agents at the same
    time, call this skill once per agent in the same step.

    Args:
        agent: Name of the specialist, for example "trading" or "content".
        task: Complete, self-contained description of what to do and what to report back.
    """
    depth = int(ctx.services.get("depth", 0)) + 1
    return await _app(ctx).run_agent(agent, task, emit=ctx.emit, depth=depth)


@skill(tags=["core"])
async def start_background_task(
    agent: str, task: str, ctx: SkillContext, repeat_every_minutes: float = 0
) -> dict:
    """Start an agent on a task in the background and return immediately.

    Use this for long jobs the user should not have to wait for, and for
    recurring monitoring. The user is told when a result arrives.

    Args:
        agent: Name of the specialist agent.
        task: Complete, self-contained description of the job.
        repeat_every_minutes: Re-run the task on this interval; 0 runs it once.
    """
    app = _app(ctx)
    if app.agents.get(agent) is None or agent == "jarvis":
        names = ", ".join(a.name for a in app.agents.specialists())
        raise SkillError(f"No agent named '{agent}'. Available agents: {names}.")
    try:
        job = app.tasks.start(agent, task, repeat_every_minutes)
    except RuntimeError as exc:
        raise SkillError(str(exc)) from exc
    return {"task_id": job.id, "status": "started", "agent": agent}


@skill(tags=["core"])
def list_background_tasks(ctx: SkillContext) -> list[dict]:
    """List background tasks with their status and latest result."""
    return [job.describe() for job in _app(ctx).tasks.list()]


@skill(tags=["core"])
def cancel_background_task(task_id: str, ctx: SkillContext) -> str:
    """Stop a running or recurring background task.

    Args:
        task_id: The id returned by start_background_task, for example "t2".
    """
    if not _app(ctx).tasks.cancel(task_id):
        raise SkillError(f"No running task with id '{task_id}'.")
    return f"Task {task_id} cancelled."


@skill(tags=["core"])
def switch_backend(target: str, ctx: SkillContext) -> dict:
    """Switch the language model backend that powers the assistant.

    Use when the user asks to go local/offline, to use the cloud, or names a
    specific provider or model. Takes effect from the next step.

    Args:
        target: A profile name, or "local" / "cloud" to pick the first matching profile.
    """
    router = _app(ctx).router
    try:
        info = router.switch(target)
    except LLMError as exc:
        raise SkillError(str(exc)) from exc
    return {"active": router.active, **info}


@skill(tags=["core"])
async def system_status(ctx: SkillContext) -> dict:
    """Report which backend is active, what is loaded in memory, and the available agents and skills."""
    app = _app(ctx)
    return {
        "backend": app.router.describe(),
        "backend_health": await app.router.health(),
        "memory": app.resources.status(),
        "agents": [a.name for a in app.agents.specialists()],
        "skills": len(app.skills.all()),
        "background_tasks": len([t for t in app.tasks.list() if t.status in ("running", "waiting")]),
        "platform": f"{platform.system()} {platform.release()}",
    }


@skill(tags=["core"])
def reload_skills(ctx: SkillContext) -> dict:
    """Reload skill plugins and agent definitions from disk after they were added or edited."""
    app = _app(ctx)
    report = app.skills.reload()
    app.agents.reload()
    report["agents"] = len(app.agents.all())
    return report


@skill(tags=["core", "util"])
def get_current_time() -> dict:
    """Get the current local date, time and timezone."""
    now = dt.datetime.now().astimezone()
    return {
        "date": now.strftime("%A, %d %B %Y"),
        "time": now.strftime("%H:%M"),
        "timezone": now.tzname(),
        "iso": now.isoformat(timespec="seconds"),
    }


@skill(tags=["core", "memory"])
def remember(key: str, value: str, ctx: SkillContext) -> str:
    """Save a fact for future conversations (a preference, a name, a recurring detail).

    Args:
        key: Short label for the fact, for example "favourite_exchange".
        value: The fact to store. An empty string forgets the key.
    """
    app = _app(ctx)
    notes = app.notes()
    key = key.strip().lower()[:80]
    if not key:
        raise SkillError("A key is required.")
    if value.strip():
        notes[key] = value.strip()[:2000]
    else:
        notes.pop(key, None)
    app.save_notes(notes)
    return f"Saved '{key}'." if value.strip() else f"Forgot '{key}'."


@skill(tags=["core", "memory"])
def recall(ctx: SkillContext, query: str = "") -> dict:
    """Look up facts saved earlier with remember.

    Args:
        query: Word to search for in keys and values. Empty returns everything.
    """
    notes = _app(ctx).notes()
    needle = query.strip().lower()
    return {k: v for k, v in notes.items() if needle in k or needle in v.lower()} if needle else notes
