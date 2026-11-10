"""ResourceManager: lazy loading, pinning, LRU eviction and idle sweeping."""

from __future__ import annotations

import asyncio

import pytest

from jarvis import resources as resources_module
from jarvis.resources import ResourceManager


class Recorder:
    def __init__(self):
        self.events: list[str] = []

    def loader(self, name, *, is_async=False):
        if is_async:

            async def load():
                await asyncio.sleep(0)
                self.events.append(f"load:{name}")
                return f"handle-{name}"

        else:

            def load():
                self.events.append(f"load:{name}")
                return f"handle-{name}"

        return load

    def unloader(self, name, *, is_async=False):
        if is_async:

            async def unload(handle):
                await asyncio.sleep(0)
                self.events.append(f"unload:{name}:{handle}")

        else:

            def unload(handle):
                self.events.append(f"unload:{name}:{handle}")

        return unload


@pytest.fixture
def rec():
    return Recorder()


@pytest.fixture
def clock(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(resources_module.time, "monotonic", lambda: now["t"])
    return now


def add(manager, rec, name, mb=100, **kw):
    manager.register(name, rec.loader(name, **kw), rec.unloader(name, **kw), memory_mb=mb)


def loaded(manager):
    return {c["name"] for c in manager.status()["components"] if c["loaded"]}


async def test_lazy_load_on_first_acquire_only(rec):
    manager = ResourceManager()
    add(manager, rec, "stt")
    assert rec.events == [] and manager.used_mb == 0
    assert await manager.acquire("stt") == "handle-stt"
    assert await manager.acquire("stt") == "handle-stt"
    assert rec.events == ["load:stt"]
    assert manager.used_mb == 100


async def test_register_is_idempotent(rec):
    manager = ResourceManager()
    add(manager, rec, "a", mb=10)
    add(manager, rec, "a", mb=999)
    assert "a" in manager and "b" not in manager
    assert manager.status()["components"][0]["memory_mb"] == 10


async def test_use_context_pins_and_releases(rec):
    manager = ResourceManager()
    add(manager, rec, "tts")
    async with manager.use("tts") as handle:
        assert handle == "handle-tts"
        assert manager.status()["components"][0]["in_use"] is True
    assert manager.status()["components"][0]["in_use"] is False
    assert manager.status()["components"][0]["loaded"] is True


async def test_use_releases_on_exception(rec):
    manager = ResourceManager()
    add(manager, rec, "tts")
    with pytest.raises(RuntimeError):
        async with manager.use("tts"):
            raise RuntimeError("x")
    assert manager.status()["components"][0]["in_use"] is False


async def test_release_never_goes_negative(rec):
    manager = ResourceManager()
    add(manager, rec, "a")
    manager.release("a")
    manager.release("missing")
    await manager.acquire("a")
    manager.release("a")
    manager.release("a")
    assert manager._resources["a"].users == 0


async def test_lru_eviction_when_over_budget(rec, clock):
    manager = ResourceManager(memory_budget_mb=250)
    for name in ("a", "b", "c"):
        add(manager, rec, name)
    for name in ("a", "b"):
        async with manager.use(name):
            clock["t"] += 1
    # Touch "a" again so "b" is the least recently used.
    async with manager.use("a"):
        clock["t"] += 1
    async with manager.use("c"):
        pass
    assert loaded(manager) == {"a", "c"}
    assert "unload:b:handle-b" in rec.events
    assert manager.used_mb <= 250


async def test_eviction_never_touches_resources_in_use(rec, clock):
    manager = ResourceManager(memory_budget_mb=150)
    add(manager, rec, "a")
    add(manager, rec, "b")
    await manager.acquire("a")  # pinned
    await manager.acquire("b")  # cannot evict "a": loads over budget
    assert loaded(manager) == {"a", "b"}
    assert not any(e.startswith("unload") for e in rec.events)


async def test_eviction_skips_pinned_but_evicts_idle(rec, clock):
    manager = ResourceManager(memory_budget_mb=200)
    for name in ("pinned", "idle", "new"):
        add(manager, rec, name)
    await manager.acquire("pinned")
    clock["t"] += 1
    async with manager.use("idle"):
        clock["t"] += 1
    await manager.acquire("new")
    assert loaded(manager) == {"pinned", "new"}


async def test_sweep_unloads_only_idle_past_ttl(rec, clock):
    manager = ResourceManager(idle_unload_seconds=60)
    for name in ("old", "fresh", "busy", "never"):
        add(manager, rec, name)
    async with manager.use("old"):
        pass
    await manager.acquire("busy")
    clock["t"] += 100
    async with manager.use("fresh"):
        pass
    clock["t"] += 10
    assert await manager.sweep() == ["old"]
    assert loaded(manager) == {"fresh", "busy"}
    clock["t"] += 100
    assert await manager.sweep() == ["fresh"]  # "busy" is pinned
    assert loaded(manager) == {"busy"}


async def test_sweep_with_tiny_ttl_real_clock(rec):
    manager = ResourceManager(idle_unload_seconds=0.01)
    add(manager, rec, "a")
    async with manager.use("a"):
        pass
    await asyncio.sleep(0.05)
    assert await manager.sweep() == ["a"]
    assert manager.used_mb == 0


async def test_reload_after_unload(rec):
    manager = ResourceManager()
    add(manager, rec, "a")
    await manager.acquire("a")
    manager.release("a")
    assert await manager.unload("a") is True
    await manager.acquire("a")
    assert rec.events == ["load:a", "unload:a:handle-a", "load:a"]
    assert manager._resources["a"].load_count == 2


async def test_unload_refuses_while_in_use_or_unloaded(rec):
    manager = ResourceManager()
    add(manager, rec, "a")
    assert await manager.unload("a") is False  # not loaded
    assert await manager.unload("missing") is False
    await manager.acquire("a")
    assert await manager.unload("a") is False
    assert loaded(manager) == {"a"}
    manager.release("a")
    assert await manager.unload("a") is True
    assert loaded(manager) == set()


@pytest.mark.parametrize("is_async", [False, True])
async def test_sync_and_async_loaders_and_unloaders(rec, is_async):
    manager = ResourceManager()
    add(manager, rec, "a", is_async=is_async)
    async with manager.use("a") as handle:
        assert handle == "handle-a"
    assert await manager.unload("a") is True
    assert rec.events == ["load:a", "unload:a:handle-a"]


async def test_resource_without_unloader(rec):
    manager = ResourceManager()
    manager.register("a", rec.loader("a"), memory_mb=5)
    await manager.acquire("a")
    manager.release("a")
    assert await manager.unload("a") is True
    assert loaded(manager) == set()


async def test_shutdown_unloads_everything(rec):
    manager = ResourceManager()
    for name in ("a", "b", "c"):
        add(manager, rec, name)
    await manager.acquire("a")
    await manager.acquire("b")  # even pinned ones are unloaded on shutdown
    await manager.shutdown()
    assert loaded(manager) == set()
    assert sorted(e for e in rec.events if e.startswith("unload")) == [
        "unload:a:handle-a",
        "unload:b:handle-b",
    ]


async def test_failing_unloader_does_not_break_manager(rec):
    manager = ResourceManager(memory_budget_mb=150)

    def bad_unload(handle):
        raise RuntimeError("cannot free")

    manager.register("bad", rec.loader("bad"), bad_unload, memory_mb=100)
    add(manager, rec, "good")
    async with manager.use("bad"):
        pass
    async with manager.use("good"):  # eviction of "bad" raises internally
        pass
    assert loaded(manager) == {"good"}
    assert await manager.unload("good") is True
    await manager.acquire("bad")
    manager.release("bad")
    await manager.shutdown()  # must not raise
    assert loaded(manager) == set()


async def test_status_reports_budget_and_usage(rec):
    manager = ResourceManager(memory_budget_mb=500)
    manager.register("a", rec.loader("a"), memory_mb=120, kind="llm")
    await manager.acquire("a")
    status = manager.status()
    assert status["budget_mb"] == 500 and status["used_mb"] == 120
    assert status["components"] == [
        {"name": "a", "kind": "llm", "memory_mb": 120, "loaded": True, "in_use": True}
    ]
