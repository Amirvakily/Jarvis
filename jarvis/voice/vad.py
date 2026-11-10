"""Energy-based voice activity detection with an adaptive noise floor.

Decides when the user started and stopped talking. Dependency-free and
cheap enough to run on every 30 ms microphone frame.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from jarvis.voice.audio import rms


@dataclass
class Utterance:
    pcm: bytes
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.pcm) / 2 / self.sample_rate


class Endpointer:
    """Feed fixed-size PCM16 frames; get "start" when speech begins and an
    :class:`Utterance` when the speaker has been quiet for long enough."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        *,
        min_threshold: float = 0.012,
        noise_ratio: float = 3.0,
        start_ms: int = 150,
        end_silence_ms: int = 750,
        pre_roll_ms: int = 300,
        min_speech_ms: int = 250,
        max_seconds: float = 30.0,
    ):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.min_threshold = min_threshold
        self.noise_ratio = noise_ratio
        self.start_frames = max(1, start_ms // frame_ms)
        self.end_frames = max(1, end_silence_ms // frame_ms)
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.max_frames = int(max_seconds * 1000 / frame_ms)
        self._pre_roll: deque[bytes] = deque(maxlen=max(1, pre_roll_ms // frame_ms))
        self._noise = min_threshold / noise_ratio
        #: Raised while the assistant is talking so that its own voice coming
        #: back through the microphone does not count as the user speaking.
        self.boost = 1.0
        self.level = 0.0
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._frames: list[bytes] = []
        self._voiced_run = 0
        self._silent_run = 0
        self._voiced_total = 0
        self._pre_roll.clear()

    @property
    def active(self) -> bool:
        return self._active

    @property
    def threshold(self) -> float:
        return max(self.min_threshold, self._noise * self.noise_ratio) * self.boost

    def feed(self, frame: bytes) -> str | Utterance | None:
        level = self.level = rms(frame)
        voiced = level >= self.threshold
        if not self._active:
            if not voiced:
                # Track the room's background level while nobody is talking.
                self._noise += (level - self._noise) * 0.05
            self._pre_roll.append(frame)
            self._voiced_run = self._voiced_run + 1 if voiced else 0
            if self._voiced_run >= self.start_frames:
                self._active = True
                self._frames = list(self._pre_roll)
                self._voiced_total = self._voiced_run
                self._silent_run = 0
                return "start"
            return None

        self._frames.append(frame)
        if voiced:
            self._voiced_total += 1
            self._silent_run = 0
        else:
            self._silent_run += 1
        if self._silent_run >= self.end_frames or len(self._frames) >= self.max_frames:
            enough = self._voiced_total >= self.min_speech_frames
            utterance = Utterance(b"".join(self._frames), self.sample_rate)
            self.reset()
            return utterance if enough else "cancel"
        return None
