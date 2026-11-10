"""On-demand loading of heavy local components.

A laptop cannot keep an 8B LLM, a Whisper model and a TTS voice resident
all the time. Every heavy component registers here with a loader, an
unloader and a memory estimate. Components load the first time a task
needs them, are evicted least-recently-used first when the budget would be
exceeded, and are unloaded after sitting idle.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

Loader = Callable[[], Any | Awaitable[Any]]
Unloader = Callable[[Any], Any | Awaitable[Any]]


@dataclass
class Resource:
    name: str
    kind: str
    memory_mb: int
    loader: Loader
    unloader: Unloader | None = None
    handle: Any = None
    loaded: bool = False
    users: int = 0
    last_used: float = 0.0
    load_count: int = 0


async def _call(fn: Callable, *args: Any) -> Any:
    result = fn(*args)
    if inspect.isawaitable(result):
        result = await result
    return result


class ResourceManager:
    def __init__(self, memory_budget_mb: int = 8000, idle_unload_seconds: float = 600):
        self.memory_budget_mb = memory_budget_mb
        self.idle_unload_seconds = idle_unload_seconds
        self._resources: dict[str, Resource] = {}
        self._lock = asyncio.Lock()

    def register(
        self,
        name: str,
        loader: Loader,
        unloader: Unloader | None = None,
        *,
        memory_mb: int = 0,
        kind: str = "component",
    ) -> None:
        if name not in self._resources:
            self._resources[name] = Resource(name, kind, memory_mb, loader, unloader)

    def __contains__(self, name: str) -> bool:
        return name in self._resources

    @property
    def used_mb(self) -> int:
        return sum(r.memory_mb for r in self._resources.values() if r.loaded)

    async def acquire(self, name: str) -> Any:
        """Load ``name`` if needed and pin it until :meth:`release`."""
        async with self._lock:
            res = self._resources[name]
            if not res.loaded:
                await self._make_room(res)
                log.info("loading %s (%s, ~%d MB)", res.name, res.kind, res.memory_mb)
                res.handle = await _call(res.loader)
                res.loaded = True
                res.load_count += 1
            res.users += 1
            res.last_used = time.monotonic()
            return res.handle

    def release(self, name: str) -> None:
        res = self._resources.get(name)
        if res and res.users > 0:
            res.users -= 1
            res.last_used = time.monotonic()

    @contextlib.asynccontextmanager
    async def use(self, name: str):
        handle = await self.acquire(name)
        try:
            yield handle
        finally:
            self.release(name)

    async def _make_room(self, incoming: Resource) -> None:
        idle = sorted(
            (r for r in self._resources.values() if r.loaded and r.users == 0),
            key=lambda r: r.last_used,
        )
        while self.used_mb + incoming.memory_mb > self.memory_budget_mb and idle:
            await self._unload(idle.pop(0), reason="memory budget")

    async def _unload(self, res: Resource, reason: str) -> None:
        log.info("unloading %s (%s)", res.name, reason)
        try:
            if res.unloader:
                await _call(res.unloader, res.handle)
        except Exception:  # an unload failure must not break the caller
            log.warning("unloading %s failed", res.name, exc_info=True)
        res.handle, res.loaded = None, False

    async def unload(self, name: str) -> bool:
        async with self._lock:
            res = self._resources.get(name)
            if not res or not res.loaded or res.users:
                return False
            await self._unload(res, reason="requested")
            return True

    async def sweep(self) -> list[str]:
        """Unload everything idle for longer than the configured TTL."""
        now, unloaded = time.monotonic(), []
        async with self._lock:
            for res in self._resources.values():
                idle_for = now - res.last_used
                if res.loaded and res.users == 0 and idle_for > self.idle_unload_seconds:
                    await self._unload(res, reason=f"idle {idle_for:.0f}s")
                    unloaded.append(res.name)
        return unloaded

    async def run_sweeper(self, interval: float = 30.0) -> None:
        while True:
            await asyncio.sleep(interval)
            await self.sweep()

    async def shutdown(self) -> None:
        async with self._lock:
            for res in self._resources.values():
                if res.loaded:
                    await self._unload(res, reason="shutdown")

    def status(self) -> dict:
        return {
            "budget_mb": self.memory_budget_mb,
            "used_mb": self.used_mb,
            "components": [
                {
                    "name": r.name,
                    "kind": r.kind,
                    "memory_mb": r.memory_mb,
                    "loaded": r.loaded,
                    "in_use": r.users > 0,
                }
                for r in self._resources.values()
            ],
        }
