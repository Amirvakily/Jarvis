"""Background agent runs: work that continues while the conversation goes on."""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

Runner = Callable[[str, str], Awaitable[str]]
Listener = Callable[[dict], Any]

MAX_TASKS = 12
MIN_INTERVAL_MINUTES = 1.0


@dataclass
class BackgroundTask:
    id: str
    agent: str
    task: str
    interval_minutes: float = 0.0
    status: str = "running"  # running | waiting | done | failed | cancelled
    created: float = field(default_factory=time.time)
    finished: float | None = None
    runs: int = 0
    result: str = ""
    handle: asyncio.Task | None = field(default=None, repr=False)

    def describe(self, full: bool = False) -> dict:
        return {
            "id": self.id,
            "agent": self.agent,
            "task": self.task,
            "status": self.status,
            "recurring_minutes": self.interval_minutes or None,
            "runs": self.runs,
            "result": self.result if full else self.result[:400],
        }


class TaskManager:
    def __init__(self, runner: Runner, notify: Listener):
        self._runner = runner
        self._notify = notify
        self._tasks: dict[str, BackgroundTask] = {}
        self._ids = itertools.count(1)

    def start(self, agent: str, task: str, interval_minutes: float = 0.0) -> BackgroundTask:
        active = [t for t in self._tasks.values() if t.status in ("running", "waiting")]
        if len(active) >= MAX_TASKS:
            raise RuntimeError(f"Too many background tasks ({MAX_TASKS}); cancel one first.")
        if interval_minutes and interval_minutes < MIN_INTERVAL_MINUTES:
            interval_minutes = MIN_INTERVAL_MINUTES
        job = BackgroundTask(f"t{next(self._ids)}", agent, task, interval_minutes)
        self._tasks[job.id] = job
        job.handle = asyncio.create_task(self._drive(job), name=f"jarvis-task-{job.id}")
        return job

    async def _drive(self, job: BackgroundTask) -> None:
        try:
            while True:
                job.status = "running"
                job.result = await self._runner(job.agent, job.task)
                job.runs += 1
                if not job.interval_minutes:
                    job.status = "done"
                    break
                job.status = "waiting"
                self._announce(job)
                await asyncio.sleep(job.interval_minutes * 60)
        except asyncio.CancelledError:
            job.status = "cancelled"
            raise
        except Exception as exc:
            job.status, job.result = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            if job.status != "waiting":
                job.finished = time.time()
                self._announce(job)

    def _announce(self, job: BackgroundTask) -> None:
        self._notify({"type": "task_update", "task": job.describe(full=True)})

    def cancel(self, task_id: str) -> bool:
        job = self._tasks.get(task_id)
        if not job or not job.handle or job.handle.done():
            return False
        job.handle.cancel()
        return True

    def get(self, task_id: str) -> BackgroundTask | None:
        return self._tasks.get(task_id)

    def list(self) -> list[BackgroundTask]:
        return list(self._tasks.values())

    async def shutdown(self) -> None:
        handles = [t.handle for t in self._tasks.values() if t.handle and not t.handle.done()]
        for handle in handles:
            handle.cancel()
        await asyncio.gather(*handles, return_exceptions=True)
