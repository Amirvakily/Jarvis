"""PCM audio container and helpers (mono, signed 16-bit little-endian)."""

from __future__ import annotations

import io
import sys
import wave
from array import array
from dataclasses import dataclass


def _samples(pcm: bytes) -> array:
    arr = array("h")
    arr.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if sys.byteorder == "big":
        arr.byteswap()
    return arr


def _to_bytes(arr: array) -> bytes:
    if sys.byteorder == "big":
        arr = array("h", arr)
        arr.byteswap()
    return arr.tobytes()


@dataclass
class Audio:
    pcm: bytes
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.pcm) / 2 / self.sample_rate if self.sample_rate else 0.0

    def to_wav(self) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(self.pcm)
        return buf.getvalue()

    @classmethod
    def from_wav(cls, data: bytes) -> Audio:
        try:
            with wave.open(io.BytesIO(data), "rb") as w:
                if w.getsampwidth() != 2:
                    raise ValueError("only 16-bit PCM WAV audio is supported")
                channels, rate = w.getnchannels(), w.getframerate()
                frames = w.readframes(w.getnframes())
        except (wave.Error, EOFError) as exc:
            raise ValueError(f"not a readable PCM WAV file: {exc}") from exc
        if channels > 1:
            arr = _samples(frames)
            usable = len(arr) - len(arr) % channels
            mono = array(
                "h",
                (sum(arr[i : i + channels]) // channels for i in range(0, usable, channels)),
            )
            frames = _to_bytes(mono)
        return cls(frames, rate)


def rms(pcm: bytes) -> float:
    """Root-mean-square level of 16-bit PCM, normalised to 0..1."""
    arr = _samples(pcm)
    if not arr:
        return 0.0
    total = sum(s * s for s in arr)
    return min(1.0, (total / len(arr)) ** 0.5 / 32768.0)


def resample(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resampler for mono 16-bit PCM."""
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("sample rates must be positive")
    if src_rate == dst_rate or not pcm:
        return pcm
    src = _samples(pcm)
    n_src = len(src)
    n_dst = max(1, round(n_src * dst_rate / src_rate))
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        x = np.frombuffer(_to_bytes(src), dtype="<i2").astype(np.float32)
        pos = np.arange(n_dst) * (src_rate / dst_rate)
        out = np.interp(pos, np.arange(n_src), x)
        return np.clip(np.rint(out), -32768, 32767).astype("<i2").tobytes()
    ratio = src_rate / dst_rate
    out_arr = array("h")
    last = n_src - 1
    for i in range(n_dst):
        pos = i * ratio
        j = int(pos)
        if j >= last:
            out_arr.append(src[last])
            continue
        frac = pos - j
        out_arr.append(int(round(src[j] + (src[j + 1] - src[j]) * frac)))
    return _to_bytes(out_arr)
