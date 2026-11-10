"""Hands-free voice on the PC itself: microphone in, speakers out.

The loop runs inside the Jarvis process and uses the computer's own audio
devices, so it works with the desktop window, in a bare terminal, or with no
interface at all. Pipeline:

    mic frames -> Endpointer (VAD) -> STT -> Conversation -> sentences -> TTS -> speakers

Barge-in: while Jarvis is talking the microphone stays open with a raised
threshold. Sustained speech stops playback immediately, cancels the reply in
progress, and the new utterance is handled as the next turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import threading
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Protocol

from jarvis.voice.audio import Audio, resample
from jarvis.voice.speaker import SpeechQueue
from jarvis.voice.stt import STT
from jarvis.voice.tts import TTS
from jarvis.voice.vad import Endpointer, Utterance

if TYPE_CHECKING:  # pragma: no cover
    from jarvis.conversation import Conversation

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_MS = 30


class AudioIO(Protocol):
    """Microphone and speaker access. The real one wraps sounddevice; tests
    supply a scripted fake."""

    sample_rate: int

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def frames(self) -> AsyncIterator[bytes]: ...
    async def play(self, audio: Audio) -> None: ...


class SoundDeviceIO:
    """PortAudio-backed audio via the ``sounddevice`` package."""

    sample_rate = SAMPLE_RATE

    def __init__(self, input_device: Any = None, output_device: Any = None):
        try:
            import sounddevice
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Microphone access needs the 'sounddevice' package (and PortAudio on Linux). "
                "Install with: pip install 'jarvis-assistant[voice]'"
            ) from exc
        self._sd = sounddevice
        self.input_device = input_device
        self.output_device = output_device
        self._stream: Any = None
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=400)
        self._capture_rate = SAMPLE_RATE
        self._stop_playback = threading.Event()

    async def start(self) -> None:
        loop = asyncio.get_running_loop()

        def push(data: bytes) -> None:
            if self._queue.full():  # the consumer stalled: drop the oldest audio
                with contextlib.suppress(asyncio.QueueEmpty):
                    self._queue.get_nowait()
            self._queue.put_nowait(data)

        def callback(indata, frames, time_info, status) -> None:  # audio thread
            loop.call_soon_threadsafe(push, bytes(indata))

        def open_stream(rate: int):
            return self._sd.RawInputStream(
                samplerate=rate,
                blocksize=int(rate * FRAME_MS / 1000),
                channels=1,
                dtype="int16",
                device=self.input_device,
                callback=callback,
            )

        try:
            self._stream = open_stream(SAMPLE_RATE)
        except self._sd.PortAudioError:
            # Some drivers refuse 16 kHz; capture natively and resample.
            info = self._sd.query_devices(self.input_device, "input")
            self._capture_rate = int(info["default_samplerate"])
            self._stream = open_stream(self._capture_rate)
        self._stream.start()

    async def stop(self) -> None:
        self._stop_playback.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def frames(self) -> AsyncIterator[bytes]:
        while True:
            frame = await self._queue.get()
            if self._capture_rate != SAMPLE_RATE:
                frame = resample(frame, self._capture_rate, SAMPLE_RATE)
            yield frame

    async def play(self, audio: Audio) -> None:
        self._stop_playback.clear()
        try:
            await asyncio.to_thread(self._play_blocking, audio)
        except asyncio.CancelledError:
            self._stop_playback.set()
            raise

    def _play_blocking(self, audio: Audio) -> None:
        pcm, rate = audio.pcm, audio.sample_rate
        try:
            stream = self._sd.RawOutputStream(
                samplerate=rate, channels=1, dtype="int16", device=self.output_device
            )
        except self._sd.PortAudioError:
            info = self._sd.query_devices(self.output_device, "output")
            target = int(info["default_samplerate"])
            pcm, rate = resample(pcm, rate, target), target
            stream = self._sd.RawOutputStream(
                samplerate=rate, channels=1, dtype="int16", device=self.output_device
            )
        step = int(rate * 0.04) * 2  # 40 ms slices keep interruption snappy
        with stream:
            for offset in range(0, len(pcm), step):
                if self._stop_playback.is_set():
                    stream.abort()
                    return
                stream.write(pcm[offset : offset + step])


class NativeVoice:
    def __init__(
        self,
        conversation: Conversation,
        stt: STT,
        tts: TTS,
        io: AudioIO,
        *,
        language: str | None = None,
        barge_in: bool = True,
        wake_word: str = "",
        follow_up_seconds: float = 20.0,
    ):
        self.conversation = conversation
        self.stt = stt
        self.io = io
        self.language = language
        self.barge_in = barge_in
        self.wake_word = wake_word.strip().lower()
        self.follow_up_seconds = follow_up_seconds
        self.muted = False
        self.state = "idle"
        self.endpointer = Endpointer(SAMPLE_RATE, FRAME_MS)
        self.speech = SpeechQueue(tts, self._play, self._on_speaking)
        self._started = False
        self._awake_until = 0.0
        self._last_level_at = 0.0
        self._jobs: set[asyncio.Task] = set()
        self._unsubscribe = conversation.subscribe(self._on_event)

    @classmethod
    def from_config(cls, conversation: Conversation, stt: STT, tts: TTS, io: AudioIO) -> NativeVoice:
        config = conversation.app.config
        return cls(
            conversation,
            stt,
            tts,
            io,
            language=config.get("voice.language") or None,
            barge_in=bool(config.get("voice.barge_in", True)),
            wake_word=config.get("voice.wake_word", "") or "",
            follow_up_seconds=float(config.get("voice.follow_up_seconds", 20)),
        )

    # -- output ---------------------------------------------------------------
    async def _play(self, audio: Audio, text: str) -> None:
        await self.io.play(audio)

    def _on_speaking(self, speaking: bool) -> None:
        if not speaking:
            self._awake_until = time.monotonic() + self.follow_up_seconds
        self._set_state("speaking" if speaking else self._resting_state())

    def _resting_state(self) -> str:
        return "thinking" if self.conversation.busy else "idle"

    def _on_event(self, event: dict) -> None:
        kind = event["type"]
        if kind == "turn_start":
            self.speech.begin_turn()
            self._set_state("thinking")
        elif kind == "sentence":
            self.speech.say(event["text"])
        elif kind == "announcement" and not self.conversation.busy and not self.speech.speaking:
            self.speech.say(event["text"])
        elif kind == "turn_end" and not self.speech.speaking:
            self._set_state("idle")
        elif kind == "stopped":
            self._spawn(self.speech.cancel())

    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self._spawn(self.conversation.emit({"type": "voice_state", "state": state}))

    def _spawn(self, coro) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._jobs.add(task)
        task.add_done_callback(self._jobs.discard)

    # -- input ----------------------------------------------------------------
    def set_muted(self, muted: bool) -> None:
        self.muted = muted
        self.endpointer.reset()
        self._spawn(self.conversation.emit({"type": "voice_muted", "muted": muted}))

    async def stop_speaking(self) -> str:
        """Silence the assistant; returns the part of the reply already heard."""
        return await self.speech.cancel()

    async def start(self) -> None:
        """Open the audio devices. Raises if there is no usable microphone."""
        if not self._started:
            await self.io.start()
            self._started = True

    async def run(self) -> None:
        """Listen until cancelled."""
        await self.start()
        self.speech.start()
        try:
            async for frame in self.io.frames():
                await self._on_frame(frame)
        finally:
            await self.speech.aclose()
            await self.io.stop()
            self._unsubscribe()

    async def _on_frame(self, frame: bytes) -> None:
        if self.muted:
            return
        speaking = self.speech.speaking
        if speaking and not self.barge_in:
            self.endpointer.reset()
            return
        self.endpointer.boost = 2.5 if speaking else 1.0
        result = self.endpointer.feed(frame)
        self._report_level()
        if result == "start":
            if speaking:
                heard = await self.speech.cancel()
                await self.conversation.interrupt(heard)
                await self.speech.cancel()  # a sentence may have slipped in meanwhile
            self._set_state("listening")
        elif result == "cancel":
            self._set_state(self._resting_state())
        elif isinstance(result, Utterance):
            self._set_state("transcribing")
            self._spawn(self._handle(result))

    def _report_level(self) -> None:
        now = time.monotonic()
        if now - self._last_level_at >= 0.12:
            self._last_level_at = now
            level = round(min(1.0, self.endpointer.level * 6), 3)
            self._spawn(self.conversation.emit({"type": "mic_level", "level": level}))

    async def _handle(self, utterance: Utterance) -> None:
        try:
            text = await self.stt.transcribe(Audio(utterance.pcm, utterance.sample_rate), self.language)
        except Exception as exc:
            log.exception("transcription failed")
            await self.conversation.emit({"type": "error", "message": f"Speech recognition failed: {exc}"})
            text = ""
        text = self._apply_wake_word(text.strip())
        if not text:
            self._set_state(self._resting_state())
            return
        self._awake_until = time.monotonic() + self.follow_up_seconds
        await self.conversation.emit({"type": "transcript", "text": text})
        await self.conversation.submit(text, source="voice")

    def _apply_wake_word(self, text: str) -> str:
        """With a wake word configured, ignore speech not addressed to the
        assistant, except for follow-ups shortly after it last spoke."""
        if not self.wake_word or not text:
            return text
        match = re.search(rf"\b{re.escape(self.wake_word)}\b[\s,.!?]*", text, flags=re.IGNORECASE)
        if match:
            remainder = (text[: match.start()] + text[match.end() :]).strip(" ,.")
            return remainder or text
        if time.monotonic() < self._awake_until or self.conversation.busy:
            return text
        return ""


def create_native_voice(conversation: Conversation, io: AudioIO | None = None) -> NativeVoice:
    """Build the native voice loop from config, or raise RuntimeError with a
    message that says what is missing."""
    from jarvis.voice.stt import create_stt
    from jarvis.voice.tts import create_tts

    app = conversation.app
    stt = create_stt(app.config, app.resources)
    if stt is None:
        raise RuntimeError(
            "No speech-to-text engine is available. Install the local one with "
            "pip install 'jarvis-assistant[voice]', or set voice.stt.provider = \"openai\"."
        )
    tts = create_tts(app.config, app.resources)
    if tts is None:
        raise RuntimeError(
            "No text-to-speech engine is available. Run `jarvis setup-voice` to download a "
            "local voice, or set voice.tts.provider to \"system\" or \"openai\"."
        )
    return NativeVoice.from_config(conversation, stt, tts, io or SoundDeviceIO(
        app.config.get("voice.input_device"), app.config.get("voice.output_device")
    ))
