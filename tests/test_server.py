"""The real ASGI app, served in-process by uvicorn and driven over HTTP and WebSocket."""

import asyncio
import contextlib
import json
import socket
from dataclasses import dataclass

import pytest
import uvicorn
import websockets
from websockets.exceptions import InvalidHandshake

from jarvis._http import httpx
from jarvis.app import Jarvis
from jarvis.llm.mock import MockProvider
from jarvis.server.app import create_app, is_loopback, same_origin

TIMEOUT = 8.0
EVIL = "http://evil.example"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.asynccontextmanager
async def serve(app: Jarvis, token=None):
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(app, token=token),
            host="127.0.0.1",
            port=port,
            log_level="warning",
            lifespan="off",
            timeout_graceful_shutdown=2,
        )
    )
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(1000):
            if server.started:
                break
            if task.done():
                task.result()
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("server did not start")
        yield port
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 15)


@dataclass
class Running:
    app: Jarvis
    provider: MockProvider
    port: int

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def ws(self, query: str = "", **kwargs):
        return websockets.connect(
            f"ws://127.0.0.1:{self.port}/ws{query}", proxy=None, open_timeout=TIMEOUT, **kwargs
        )

    def http(self, **kwargs):
        return httpx.AsyncClient(base_url=self.base, timeout=TIMEOUT, trust_env=False, **kwargs)


@pytest.fixture
async def start(config):
    config.set("voice.stt.provider", "mock")
    config.set("voice.tts.provider", "mock")
    config.set("llm.profiles.alt", {"provider": "mock", "model": "alt-model"})
    stack = contextlib.AsyncExitStack()

    async def _start(delay=0.0, token=None, script=None) -> Running:
        provider = MockProvider(
            "mock", {"provider": "mock", "model": "mock", "delay": delay}, config, script=script
        )
        app = Jarvis(config, providers=[provider])
        stack.push_async_callback(app.aclose)
        port = await stack.enter_async_context(serve(app, token))
        return Running(app, provider, port)

    yield _start
    await stack.aclose()


@pytest.fixture
async def server(start):
    return await start()


async def recv(ws):
    """Next message: a dict for text frames, bytes for binary frames."""
    raw = await asyncio.wait_for(ws.recv(), TIMEOUT)
    return raw if isinstance(raw, bytes) else json.loads(raw)


async def send(ws, **message):
    await ws.send(json.dumps(message))


async def until(ws, kind, collected=None):
    """Read messages until one of type ``kind``; returns every message read."""
    seen = collected if collected is not None else []
    while True:
        message = await recv(ws)
        seen.append(message)
        if isinstance(message, dict) and message.get("type") == kind:
            return seen


def types(messages):
    return [m["type"] for m in messages if isinstance(m, dict)]


async def connect_ready(server, query="", **kwargs):
    ws = await server.ws(query, **kwargs).__aenter__()
    ready = await recv(ws)
    assert ready["type"] == "ready"
    return ws, ready


# --------------------------------------------------------------------------- helpers


def test_helper_functions():
    assert is_loopback("127.0.0.1") and is_loopback("::1") and is_loopback("localhost")
    assert not is_loopback("0.0.0.0") and not is_loopback("192.168.1.5") and not is_loopback("example.com")
    assert same_origin(None, "127.0.0.1:8000")  # native clients send no Origin
    assert same_origin("http://127.0.0.1:8000", "127.0.0.1:8000")
    assert not same_origin(EVIL, "127.0.0.1:8000")
    assert not same_origin("http://127.0.0.1:8000", None)


# --------------------------------------------------------------------------- websocket protocol


async def test_ready_message_has_state_voice_and_transcript(server):
    async with server.ws() as ws:
        ready = await recv(ws)
    assert ready["type"] == "ready"
    assert ready["busy"] is False and ready["transcript"] == []
    assert ready["state"]["backend"]["active"] == "mock"
    assert {a["name"] for a in ready["state"]["agents"]} >= {"jarvis", "trading", "research"}
    voice = ready["voice"]
    assert voice["native"] is False
    assert voice["server_stt"] is True and voice["server_tts"] is True
    assert voice["barge_in"] is True


async def test_user_text_streams_a_turn(server):
    async with server.ws() as ws:
        await recv(ws)
        await send(ws, type="user_text", text="hello there friend")
        messages = await until(ws, "turn_end")
    kinds = types(messages)
    assert kinds[0] == "turn_start" and kinds[-1] == "turn_end"
    assert kinds.index("text_delta") < kinds.index("sentence") < kinds.index("turn_end")
    assert messages[0]["text"] == "hello there friend" and messages[0]["source"] == "text"
    deltas = "".join(m["text"] for m in messages if m["type"] == "text_delta")
    assert deltas == "You said: hello there friend"
    assert [m["text"] for m in messages if m["type"] == "sentence"] == ["You said: hello there friend"]
    end = messages[-1]
    assert end["interrupted"] is False and end["error"] is None and end["text"] == deltas
    # No speech was requested, so no audio is produced.
    assert "audio" not in kinds


async def test_server_side_speech_sends_audio_header_then_wav(server):
    async with server.ws() as ws:
        await recv(ws)
        await send(ws, type="hello", speech="server")
        await send(ws, type="user_text", text="speak to me")
        header = None
        while header is None:
            message = await recv(ws)
            if isinstance(message, dict) and message["type"] == "audio":
                header = message
        frame = await recv(ws)
    assert isinstance(frame, bytes) and frame[:4] == b"RIFF" and frame[8:12] == b"WAVE"
    assert header["text"] == "You said: speak to me"
    assert header["sample_rate"] == 16000 and header["bytes"] == len(frame)


async def test_server_speech_falls_back_to_browser_without_a_tts_engine(config, start):
    config.set("voice.tts.provider", "browser")  # no server-side TTS
    running = await start()
    async with running.ws() as ws:
        ready = await recv(ws)
        assert ready["voice"]["server_tts"] is False
        await send(ws, type="hello", speech="server")
        message = await recv(ws)
    assert message == {"type": "speech_mode", "mode": "browser", "reason": "no server TTS"}


async def test_audio_upload_is_transcribed_and_starts_a_turn(server):
    pcm = bytes(32000)  # one second of 16 kHz silence; the mock STT does not care
    async with server.ws() as ws:
        await recv(ws)
        await send(ws, type="audio_start", sample_rate=16000)
        for offset in range(0, len(pcm), 8000):
            await ws.send(pcm[offset : offset + 8000])
        await send(ws, type="audio_end")
        messages = await until(ws, "turn_end")
    kinds = types(messages)
    assert kinds.index("transcript") < kinds.index("turn_start")
    assert messages[kinds.index("transcript")]["text"] == "hello jarvis"
    start = messages[kinds.index("turn_start")]
    assert start["source"] == "voice" and start["text"] == "hello jarvis"
    assert messages[-1]["text"] == "You said: hello jarvis"


async def test_too_short_audio_is_ignored(server):
    async with server.ws() as ws:
        await recv(ws)
        await send(ws, type="audio_start", sample_rate=16000)
        await ws.send(bytes(100))
        await send(ws, type="audio_end")
        await asyncio.sleep(0.2)
        await send(ws, type="ping")
        messages = await until(ws, "pong")
    assert types(messages) == ["pong"]


async def test_interrupt_ends_the_turn_as_interrupted(start):
    server = await start(delay=0.1)
    async with server.ws() as ws:
        await recv(ws)
        await send(ws, type="user_text", text="one two three four five six seven eight nine ten")
        await until(ws, "text_delta")
        await send(ws, type="interrupt", heard="You said:")
        messages = await until(ws, "turn_end")
    end = messages[-1]
    assert end["interrupted"] is True
    assert server.app.conversation().session.messages[-1].content == "You said:"


async def test_new_user_text_interrupts_the_running_turn(start):
    server = await start(delay=0.1)
    async with server.ws() as ws:
        await recv(ws)
        await send(ws, type="user_text", text="one two three four five six seven eight nine ten")
        await until(ws, "text_delta")
        await send(ws, type="user_text", text="quick")
        messages = await until(ws, "turn_end")
        assert messages[-1]["interrupted"] is True
        messages = await until(ws, "turn_end")
    assert messages[-1]["interrupted"] is False and messages[-1]["text"] == "You said: quick"


async def test_set_backend_broadcasts_a_backend_event(server):
    async with server.ws() as ws:
        await recv(ws)
        await send(ws, type="set_backend", profile="alt")
        messages = await until(ws, "backend")
    event = messages[-1]
    assert event["active"] == "alt"
    assert {p["name"] for p in event["profiles"]} >= {"mock", "alt"}
    assert server.app.router.active == "alt"


async def test_set_backend_with_unknown_profile_reports_an_error(server):
    async with server.ws() as ws:
        await recv(ws)
        await send(ws, type="set_backend", profile="nope")
        messages = await until(ws, "error")
        assert "No backend matches" in messages[-1]["message"]
        await send(ws, type="ping")
        await until(ws, "pong")
    assert server.app.router.active == "mock"


async def test_reset_clears_the_transcript(server):
    ws, _ = await connect_ready(server)
    try:
        await send(ws, type="user_text", text="remember this")
        await until(ws, "turn_end")
        async with server.ws() as second:
            ready = await recv(second)
            assert [row["role"] for row in ready["transcript"]] == ["user", "assistant"]
            assert ready["transcript"][0]["text"] == "remember this"
        await send(ws, type="reset")
        await until(ws, "reset")
        async with server.ws() as third:
            assert (await recv(third))["transcript"] == []
    finally:
        await ws.close()
    assert server.app.session().messages == []


async def test_malformed_messages_report_an_error_and_keep_the_connection(server):
    async with server.ws() as ws:
        await recv(ws)
        for bad in ("this is not json {", json.dumps({"no_type": 1}), json.dumps([1, 2, 3]), "null"):
            await ws.send(bad)
            message = await recv(ws)
            assert message == {"type": "error", "message": "malformed message"}
        await send(ws, type="ping")
        assert (await recv(ws)) == {"type": "pong"}
        await send(ws, type="no_such_message_type")  # unknown types are ignored
        await send(ws, type="user_text", text="still works")
        messages = await until(ws, "turn_end")
    assert messages[-1]["text"] == "You said: still works"


async def test_oversized_text_is_truncated(server):
    async with server.ws() as ws:
        await recv(ws)
        await send(ws, type="user_text", text="x" * 20000)
        messages = await until(ws, "turn_end")
    assert len(messages[0]["text"]) == 8000


async def test_events_reach_every_connected_client(server):
    async with server.ws() as one, server.ws() as two:
        await recv(one)
        await recv(two)
        await send(one, type="user_text", text="broadcast me")
        for ws in (one, two):
            messages = await until(ws, "turn_end")
            assert messages[-1]["text"] == "You said: broadcast me"


async def test_disconnect_mid_turn_does_not_break_the_server(start):
    server = await start(delay=0.05)
    async with server.ws() as ws:
        await recv(ws)
        await send(ws, type="user_text", text="one two three four five six seven eight")
        await until(ws, "text_delta")
    # The turn keeps running without a listener; a new client can connect and use the app.
    await asyncio.wait_for(server.app.conversation().wait(), TIMEOUT)
    async with server.ws() as ws:
        ready = await recv(ws)
        assert [row["role"] for row in ready["transcript"]] == ["user", "assistant"]
        await send(ws, type="user_text", text="after")
        messages = await until(ws, "turn_end")
    assert messages[-1]["text"] == "You said: after"


# --------------------------------------------------------------------------- security


async def test_websocket_with_a_foreign_origin_is_refused(server):
    with pytest.raises(InvalidHandshake):
        async with server.ws(origin=EVIL):
            pass
    # The matching origin (what a browser on this page would send) is fine.
    async with server.ws(origin=server.base) as ws:
        assert (await recv(ws))["type"] == "ready"


async def test_rest_requests_with_a_foreign_origin_are_refused(server):
    async with server.http() as client:
        post = await client.post("/api/backend", json={"profile": "alt"}, headers={"Origin": EVIL})
        assert post.status_code == 403
        assert server.app.router.active == "mock"  # nothing was changed
        get = await client.get("/api/state", headers={"Origin": EVIL})
        assert get.status_code == 403
        ok = await client.post(
            "/api/backend", json={"profile": "alt"}, headers={"Origin": server.base}
        )
        assert ok.status_code == 200
        # Native clients send no Origin at all.
        assert (await client.get("/api/state")).status_code == 200


async def test_token_protects_rest_and_websocket(start):
    server = await start(token="secret")
    async with server.http() as client:
        assert (await client.get("/api/state")).status_code == 401
        assert (await client.get("/api/state", headers={"Authorization": "Bearer wrong"})).status_code == 401
        good = await client.get("/api/state", headers={"Authorization": "Bearer secret"})
        assert good.status_code == 200 and good.json()["backend"]["active"] == "mock"
        assert (await client.get("/api/state", params={"token": "secret"})).status_code == 200
        assert (await client.post("/api/backend", json={"profile": "alt"})).status_code == 401
        assert server.app.router.active == "mock"
        assert (await client.post("/api/skills/reload")).status_code == 401
        assert (await client.get("/api/tasks")).status_code == 401
        assert (await client.delete("/api/tasks/t1")).status_code == 401

    for query in ("", "?token=wrong"):
        with pytest.raises(InvalidHandshake):
            async with server.ws(query):
                pass
    async with server.ws("?token=secret") as ws:
        assert (await recv(ws))["type"] == "ready"


async def test_token_and_origin_are_both_required(start):
    server = await start(token="secret")
    async with server.http() as client:
        bad_origin = await client.get(
            "/api/state", headers={"Authorization": "Bearer secret", "Origin": EVIL}
        )
        assert bad_origin.status_code == 403
    with pytest.raises(InvalidHandshake):
        async with server.ws("?token=secret", origin=EVIL):
            pass


# --------------------------------------------------------------------------- REST


async def test_get_state(server):
    async with server.http() as client:
        response = await client.get("/api/state")
    assert response.status_code == 200
    state = response.json()
    assert state["assistant"] == "Jarvis"
    assert state["backend"]["active"] == "mock"
    assert {a["name"] for a in state["agents"]} >= {"jarvis", "trading", "content"}
    assert "delegate" in {s["name"] for s in state["skills"]}
    assert state["tasks"] == [] and state["skill_errors"] == {}
    assert state["voice"]["server_stt"] is True


async def test_post_backend_valid_and_invalid(server):
    async with server.http() as client:
        ok = await client.post("/api/backend", json={"profile": "alt"})
        assert ok.status_code == 200 and ok.json()["active"] == "alt"
        assert server.app.router.active == "alt"
        bad = await client.post("/api/backend", json={"profile": "nonexistent"})
        assert bad.status_code == 400 and "No backend matches" in bad.json()["detail"]
        assert server.app.router.active == "alt"
        missing = await client.post("/api/backend", json={})
        assert missing.status_code == 422


async def test_run_agents_endpoint_runs_several_agents(server):
    runs = [{"agent": "trading", "task": "alpha"}, {"agent": "research", "task": "beta"}]
    async with server.http() as client:
        response = await client.post("/api/agents/run", json={"runs": runs})
        assert response.status_code == 200
        results = response.json()["results"]
        assert [(r["agent"], r["ok"], r["report"]) for r in results] == [
            ("trading", True, "You said: alpha"),
            ("research", True, "You said: beta"),
        ]
        mixed = await client.post(
            "/api/agents/run",
            json={"runs": [{"agent": "trading", "task": "ok"}, {"agent": "ghost", "task": "x"}]},
        )
        results = mixed.json()["results"]
        assert results[0]["ok"] is True
        assert results[1]["ok"] is False and "No agent named 'ghost'" in results[1]["report"]
        for invalid in ([], [{"agent": "trading"}], [{"task": "x"}]):
            rejected = await client.post("/api/agents/run", json={"runs": invalid})
            assert rejected.status_code == 400


async def test_tasks_and_reload_endpoints(server):
    async with server.http() as client:
        assert (await client.get("/api/tasks")).json() == {"tasks": []}
        assert (await client.delete("/api/tasks/t99")).status_code == 404
        reload = await client.post("/api/skills/reload")
        assert reload.status_code == 200 and reload.json()["errors"] == {}
        health = await client.get("/api/health")
        assert health.json()["backends"]["mock"]["ok"] is True


async def test_index_serves_html_and_static_assets(server):
    async with server.http() as client:
        page = await client.get("/")
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        assert page.headers["cache-control"] == "no-store"
        assert "<html" in page.text.lower()
        script = await client.get("/static/app.js")
        assert script.status_code == 200
        assert (await client.get("/static/does-not-exist.js")).status_code == 404
        assert (await client.get("/openapi.json")).status_code == 404  # API docs are not exposed
