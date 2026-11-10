"""Local HTTP + WebSocket interface.

This is what the desktop window (and, optionally, a browser) talks to. It
listens on 127.0.0.1 by default. Because the assistant can run skills, the
socket is guarded: cross-origin pages are refused, and a token is required
whenever the server is reachable from other machines.

WebSocket protocol (JSON text frames unless noted)
--------------------------------------------------
client -> server
    {"type": "hello", "speech": "none" | "browser" | "server"}
    {"type": "user_text", "text": str, "spoken": bool}
    {"type": "audio_start", "sample_rate": int}  then binary PCM16 mono frames
    {"type": "audio_end"}                        -> transcribed and submitted
    {"type": "interrupt", "heard": str | null}
    {"type": "set_backend", "profile": str}
    {"type": "native_mute", "muted": bool}
    {"type": "reset"}
server -> client
    {"type": "ready", "state": {...}, "voice": {...}, "transcript": [...]}
    every Conversation event (turn_start, text_delta, sentence, tool_*, agent_*,
    turn_end, announcement, backend, task_update, voice_state, mic_level, ...)
    {"type": "audio", "turn_id", "text", "sample_rate", "bytes"} followed by one
    binary WAV frame, when the client asked for speech: "server"
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import ipaddress
import logging
import secrets
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import Body, Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from jarvis.app import Jarvis
from jarvis.core.types import LLMError
from jarvis.voice.audio import Audio
from jarvis.voice.speaker import SpeechQueue

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
MAX_UTTERANCE_BYTES = 16000 * 2 * 60  # one minute of 16 kHz PCM16
MAX_TEXT_CHARS = 8000


def is_loopback(host: str) -> bool:
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def same_origin(origin: str | None, host_header: str | None) -> bool:
    """Browsers attach Origin to cross-site requests. Anything that carries
    one must match the host it is talking to; native clients send none."""
    if not origin:
        return True
    return bool(host_header) and urlsplit(origin).netloc == host_header


def create_app(jarvis: Jarvis, native: Any = None, token: str | None = None) -> FastAPI:
    """Build the ASGI app. ``native`` is the NativeVoice loop when the PC's
    own microphone and speakers are in use."""
    config = jarvis.config
    token = token if token is not None else (config.get("server.token") or "")
    if not token and not is_loopback(str(config.get("server.host", "127.0.0.1"))):
        token = secrets.token_urlsafe(24)
        log.warning("server.host is not loopback; access token generated: %s", token)

    app = FastAPI(title="Jarvis", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.jarvis = jarvis
    app.state.native = native
    app.state.token = token
    app.state.stt = None
    app.state.tts = None

    def token_ok(supplied: str | None) -> bool:
        return not token or (supplied is not None and hmac.compare_digest(supplied, token))

    def guard(request: Request) -> None:
        if not same_origin(request.headers.get("origin"), request.headers.get("host")):
            raise HTTPException(403, "cross-origin request refused")
        supplied = request.query_params.get("token")
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            supplied = header[7:]
        if not token_ok(supplied):
            raise HTTPException(401, "missing or invalid token")

    def speech_engines() -> tuple[Any, Any]:
        """Server-side STT/TTS for browser clients, created on first use."""
        from jarvis.voice.stt import create_stt
        from jarvis.voice.tts import create_tts

        if app.state.stt is None:
            with contextlib.suppress(Exception):
                app.state.stt = create_stt(config, jarvis.resources)
        if app.state.tts is None:
            with contextlib.suppress(Exception):
                app.state.tts = create_tts(config, jarvis.resources)
        return app.state.stt, app.state.tts

    def voice_info() -> dict[str, Any]:
        native_voice = app.state.native
        stt, tts = (None, None) if native_voice else speech_engines()
        return {
            "mode": config.get("voice.mode", "auto"),
            "native": native_voice is not None,
            "native_muted": bool(native_voice and native_voice.muted),
            "native_state": native_voice.state if native_voice else None,
            "server_stt": stt is not None,
            "server_tts": tts is not None,
            "barge_in": bool(config.get("voice.barge_in", True)),
            "language": config.get("voice.language", "en"),
            "wake_word": config.get("voice.wake_word", ""),
        }

    # -- REST -------------------------------------------------------------------
    @app.get("/api/state", dependencies=[Depends(guard)])
    async def get_state() -> dict:
        return {**jarvis.state(), "voice": voice_info()}

    @app.get("/api/health", dependencies=[Depends(guard)])
    async def get_health() -> dict:
        return {"backends": await jarvis.router.health()}

    @app.post("/api/backend", dependencies=[Depends(guard)])
    async def set_backend(profile: str = Body(..., embed=True)) -> dict:
        try:
            jarvis.router.switch(profile)
        except LLMError as exc:
            raise HTTPException(400, str(exc)) from exc
        return jarvis.router.describe()

    @app.post("/api/agents/run", dependencies=[Depends(guard)])
    async def run_agents(runs: list[dict] = Body(..., embed=True)) -> dict:
        """Run several agents concurrently: {"runs": [{"agent": ..., "task": ...}]}."""
        jobs = [(str(r.get("agent", "")), str(r.get("task", ""))) for r in runs[:8]]
        if not jobs or any(not agent or not task for agent, task in jobs):
            raise HTTPException(400, "each run needs an 'agent' and a 'task'")
        return {"results": await jarvis.run_agents(jobs)}

    @app.get("/api/tasks", dependencies=[Depends(guard)])
    async def list_tasks() -> dict:
        return {"tasks": [t.describe(full=True) for t in jarvis.tasks.list()]}

    @app.delete("/api/tasks/{task_id}", dependencies=[Depends(guard)])
    async def cancel_task(task_id: str) -> dict:
        if not jarvis.tasks.cancel(task_id):
            raise HTTPException(404, "no such running task")
        return {"cancelled": task_id}

    @app.post("/api/skills/reload", dependencies=[Depends(guard)])
    async def reload_skills() -> dict:
        report = jarvis.skills.reload()
        jarvis.agents.reload()
        return report

    # -- WebSocket ----------------------------------------------------------------
    @app.websocket("/ws")
    async def websocket(ws: WebSocket) -> None:
        if not same_origin(ws.headers.get("origin"), ws.headers.get("host")) or not token_ok(
            ws.query_params.get("token")
        ):
            await ws.close(code=4403)
            return
        await ws.accept()
        await ClientConnection(app, ws, voice_info, speech_engines).run()

    # -- static UI ------------------------------------------------------------------
    @app.get("/")
    async def index() -> Any:
        page = STATIC / "index.html"
        if not page.is_file():
            return JSONResponse({"error": "UI assets are missing"}, status_code=500)
        return FileResponse(page, headers={"Cache-Control": "no-store"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


class ClientConnection:
    """One window/browser tab attached to the shared conversation."""

    def __init__(self, app: FastAPI, ws: WebSocket, voice_info, speech_engines):
        self.app = app
        self.ws = ws
        self.jarvis: Jarvis = app.state.jarvis
        self.conversation = self.jarvis.conversation("main")
        self._voice_info = voice_info
        self._speech_engines = speech_engines
        self.speech_mode = "none"
        self.outbox: asyncio.Queue[dict | tuple[dict, bytes]] = asyncio.Queue(maxsize=2000)
        self.speech: SpeechQueue | None = None
        self._audio = bytearray()
        self._audio_rate = 16000
        self._receiving_audio = False

    @property
    def native(self) -> Any:
        return self.app.state.native

    async def run(self) -> None:
        unsubscribe = self.conversation.subscribe(self._on_event)
        sender = asyncio.create_task(self._send_loop())
        try:
            self._queue(
                {
                    "type": "ready",
                    "state": self.jarvis.state(),
                    "voice": self._voice_info(),
                    "transcript": self.conversation.session.transcript(),
                    "busy": self.conversation.busy,
                }
            )
            while True:
                message = await self.ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    self._on_audio(message["bytes"])
                elif message.get("text"):
                    await self._on_message(message["text"])
        except WebSocketDisconnect:
            pass
        finally:
            unsubscribe()
            sender.cancel()
            if self.speech:
                await self.speech.aclose()

    # -- outbound ---------------------------------------------------------------
    def _queue(self, item: dict | tuple[dict, bytes]) -> None:
        with contextlib.suppress(asyncio.QueueFull):  # a stalled client loses events
            self.outbox.put_nowait(item)

    def _on_event(self, event: dict) -> None:
        self._queue(event)
        if self.speech_mode != "server" or self.speech is None or self.native is not None:
            return
        if event["type"] == "turn_start":
            self.speech.begin_turn()
        elif event["type"] == "sentence" or (
            event["type"] == "announcement" and not self.conversation.busy
        ):
            self.speech.say(event["text"])

    async def _send_audio(self, audio: Audio, text: str) -> None:
        wav = audio.to_wav()
        header = {"type": "audio", "text": text, "sample_rate": audio.sample_rate, "bytes": len(wav)}
        self._queue((header, wav))

    async def _send_loop(self) -> None:
        while True:
            item = await self.outbox.get()
            try:
                if isinstance(item, tuple):
                    await self.ws.send_json(item[0])
                    await self.ws.send_bytes(item[1])
                else:
                    await self.ws.send_json(item)
            except (WebSocketDisconnect, RuntimeError):
                return

    # -- inbound ----------------------------------------------------------------
    async def _on_message(self, raw: str) -> None:
        import json

        try:
            data = json.loads(raw)
            kind = data["type"]
        except (ValueError, KeyError, TypeError):
            return self._queue({"type": "error", "message": "malformed message"})

        if kind == "hello":
            await self._set_speech_mode(str(data.get("speech", "none")))
        elif kind == "user_text":
            text = str(data.get("text", ""))[:MAX_TEXT_CHARS]
            spoken = bool(data.get("spoken"))
            speaks = spoken or self.speech_mode != "none" or self._native_listening()
            if self.native is not None:
                await self.native.stop_speaking()
            elif self.speech is not None:
                await self.speech.cancel()
            await self.conversation.submit(text, source="voice" if spoken else "text", voice=speaks)
        elif kind == "interrupt":
            heard = data.get("heard")
            if self.native is not None:
                heard = await self.native.stop_speaking()
            elif self.speech is not None:
                await self.speech.cancel()
            await self.conversation.interrupt(heard if isinstance(heard, str) else None)
        elif kind == "audio_start":
            self._audio = bytearray()
            self._audio_rate = int(data.get("sample_rate", 16000))
            self._receiving_audio = True
        elif kind == "audio_end":
            self._receiving_audio = False
            pcm, self._audio = bytes(self._audio), bytearray()
            asyncio.create_task(self._transcribe(pcm, self._audio_rate))
        elif kind == "set_backend":
            try:
                self.jarvis.router.switch(str(data.get("profile", "")))
            except LLMError as exc:
                self._queue({"type": "error", "message": str(exc)})
        elif kind == "native_mute" and self.native is not None:
            self.native.set_muted(bool(data.get("muted")))
        elif kind == "reset":
            await self.conversation.reset()
        elif kind == "ping":
            self._queue({"type": "pong"})

    def _native_listening(self) -> bool:
        return self.native is not None and not self.native.muted

    async def _set_speech_mode(self, mode: str) -> None:
        self.speech_mode = mode if mode in ("none", "browser", "server") else "none"
        if self.speech_mode == "server" and self.native is None and self.speech is None:
            _, tts = self._speech_engines()
            if tts is None:
                self.speech_mode = "browser"
                self._queue({"type": "speech_mode", "mode": "browser", "reason": "no server TTS"})
            else:
                self.speech = SpeechQueue(tts, self._send_audio)

    def _on_audio(self, chunk: bytes) -> None:
        if self._receiving_audio and len(self._audio) + len(chunk) <= MAX_UTTERANCE_BYTES:
            self._audio.extend(chunk)

    async def _transcribe(self, pcm: bytes, sample_rate: int) -> None:
        stt, _ = self._speech_engines()
        if stt is None or len(pcm) < sample_rate // 5 * 2:
            return
        try:
            text = await stt.transcribe(
                Audio(pcm, sample_rate), self.jarvis.config.get("voice.language") or None
            )
        except Exception as exc:
            log.exception("transcription failed")
            return self._queue({"type": "error", "message": f"Speech recognition failed: {exc}"})
        if text.strip():
            await self.conversation.emit({"type": "transcript", "text": text.strip()})
            if self.speech is not None:
                await self.speech.cancel()
            await self.conversation.submit(text, source="voice")
