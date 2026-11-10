"""Ordered, interruptible speech output.

Sentences are synthesised ahead of playback (the next one renders while the
current one plays) but always played in order. ``cancel`` stops at once and
reports what was actually heard, so an interrupted reply can be recorded
truthfully in the conversation history.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable

from jarvis.voice.audio import Audio
from jarvis.voice.tts import TTS

log = logging.getLogger(__name__)

Player = Callable[[Audio, str], Awaitable[None]]
StateListener = Callable[[bool], None]


class SpeechQueue:
    def __init__(self, tts: TTS, play: Player, on_state: StateListener | None = None):
        self.tts = tts
        self._play = play
        self._on_state = on_state
        self._queue: asyncio.Queue[tuple[str, asyncio.Task]] = asyncio.Queue()
        self._synth_lock = asyncio.Lock()
        self._worker: asyncio.Task | None = None
        self._current: asyncio.Task | None = None
        self._heard: list[str] = []
        self._pending = 0
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def speaking(self) -> bool:
        return self._pending > 0

    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run(), name="jarvis-speech")

    def begin_turn(self) -> None:
        self._heard = []

    def say(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.start()
        self._set_pending(self._pending + 1)
        self._queue.put_nowait((text, asyncio.create_task(self._synthesize(text))))

    async def _synthesize(self, text: str) -> Audio | None:
        async with self._synth_lock:
            try:
                return await self.tts.synthesize(text)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("speech synthesis failed")
                return None

    async def _run(self) -> None:
        while True:
            text, synth = await self._queue.get()
            self._current = asyncio.create_task(self._speak(text, synth))
            with contextlib.suppress(asyncio.CancelledError):
                await self._current
            self._current = None
            self._set_pending(self._pending - 1)

    async def _speak(self, text: str, synth: asyncio.Task) -> None:
        try:
            audio = await synth
        except asyncio.CancelledError:
            synth.cancel()
            raise
        if audio is not None and audio.pcm:
            await self._play(audio, text)
            self._heard.append(text)

    def _set_pending(self, value: int) -> None:
        was_speaking = self._pending > 0
        self._pending = max(0, value)
        if self._pending:
            self._idle.clear()
        else:
            self._idle.set()
        if self._on_state and was_speaking != (self._pending > 0):
            self._on_state(self._pending > 0)

    async def cancel(self) -> str:
        """Stop speaking now. Returns the text that finished playing."""
        while not self._queue.empty():
            _, synth = self._queue.get_nowait()
            synth.cancel()
            self._set_pending(self._pending - 1)
        if self._current is not None:
            self._current.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._current
        heard, self._heard = " ".join(self._heard), []
        return heard

    async def wait_idle(self) -> None:
        await self._idle.wait()

    async def aclose(self) -> None:
        await self.cancel()
        if self._worker:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
