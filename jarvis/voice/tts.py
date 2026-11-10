"""Text-to-speech engines: Piper, OpenAI-compatible HTTP, the OS voice, and a mock."""

from __future__ import annotations

import asyncio
import gc
import importlib.util
import io
import math
import os
import re
import shutil
import sys
import tempfile
import wave
from array import array
from pathlib import Path
from typing import Any

from jarvis._http import httpx
from jarvis.config import Config
from jarvis.resources import ResourceManager
from jarvis.voice.audio import Audio

_VOICE_NAME = re.compile(r"^[a-z]{2}_[A-Z]{2}-[a-z0-9_]+-(x_low|low|medium|high)$")
_VOICE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


class TTS:
    name = "tts"

    async def synthesize(self, text: str) -> Audio:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


def find_piper_voice(config: Config) -> Path | None:
    configured = str(config.get("voice.tts.model") or "")
    if configured.endswith(".onnx") and Path(configured).expanduser().is_file():
        return Path(configured).expanduser()
    voices = config.data_dir / "voices"
    found = sorted(voices.glob("*.onnx")) if voices.is_dir() else []
    return found[0] if found else None


class PiperTTS(TTS):
    name = "piper"
    resource_name = "tts:piper"

    def __init__(self, config: Config, resources: ResourceManager):
        self.config = config
        self.resources = resources
        resources.register(
            self.resource_name,
            self._load,
            self._unload,
            memory_mb=int(config.get("voice.tts.memory_mb", 150)),
            kind="tts",
        )

    async def _load(self) -> Any:
        path = find_piper_voice(self.config)
        if path is None:
            raise RuntimeError("No Piper voice found: run `jarvis setup-voice` to download one")

        def build() -> Any:
            try:
                from piper import PiperVoice
            except ImportError as exc:
                raise RuntimeError(
                    "piper is not installed: pip install 'jarvis-assistant[voice]'"
                ) from exc
            return PiperVoice.load(str(path))

        return await asyncio.to_thread(build)

    @staticmethod
    def _unload(voice: Any) -> None:
        del voice
        gc.collect()

    @staticmethod
    def _render(voice: Any, text: str) -> Audio:
        if hasattr(voice, "synthesize_wav"):
            pcm, rate = bytearray(), 0
            for chunk in voice.synthesize(text):
                rate = chunk.sample_rate
                pcm += chunk.audio_int16_bytes
            return Audio(bytes(pcm), rate or int(voice.config.sample_rate))
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            voice.synthesize(text, w)
        return Audio.from_wav(buf.getvalue())

    async def synthesize(self, text: str) -> Audio:
        if not text.strip():
            return Audio(b"", 22050)
        async with self.resources.use(self.resource_name) as voice:
            return await asyncio.to_thread(self._render, voice, text)


class OpenAITTS(TTS):
    name = "openai"
    sample_rate = 24000

    def __init__(self, config: Config, transport: Any = None):
        self.config = config
        self._client = httpx.AsyncClient(transport=transport, timeout=60)

    async def synthesize(self, text: str) -> Audio:
        base = str(self.config.get("voice.tts.base_url")).rstrip("/")
        headers = {}
        key = self.config.secret(self.config.get("voice.tts.api_key_env"))
        if key:
            headers["Authorization"] = f"Bearer {key}"
        resp = await self._client.post(
            f"{base}/audio/speech",
            headers=headers,
            json={
                "model": self.config.get("voice.tts.model") or "tts-1",
                "voice": self.config.get("voice.tts.voice", "alloy"),
                "input": text,
                "response_format": "pcm",
            },
        )
        resp.raise_for_status()
        return Audio(resp.content, self.sample_rate)

    async def aclose(self) -> None:
        await self._client.aclose()


_WIN_SCRIPT = """\
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SetOutputToWaveFile($env:JARVIS_TTS_OUT)
$s.Speak($env:JARVIS_TTS_TEXT)
$s.Dispose()
"""


def _linux_binary() -> str | None:
    return shutil.which("espeak-ng") or shutil.which("espeak")


class SystemTTS(TTS):
    name = "system"

    def __init__(self, config: Config):
        self.config = config

    @staticmethod
    def available() -> bool:
        if sys.platform.startswith("win"):
            return shutil.which("powershell") is not None
        if sys.platform == "darwin":
            return shutil.which("say") is not None
        return _linux_binary() is not None

    async def _run(self, args: list[str], *, stdin: bytes | None = None, env: dict | None = None):
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        out, err = await proc.communicate(stdin)
        if proc.returncode != 0:
            detail = err.decode(errors="replace").strip()[:300]
            raise RuntimeError(f"{args[0]} failed (exit {proc.returncode}): {detail}")
        return out

    async def synthesize(self, text: str) -> Audio:
        if not self.available():
            raise RuntimeError(
                "No system voice available (needs PowerShell, `say`, or espeak-ng); "
                "install espeak-ng or set voice.tts.provider to piper or openai"
            )
        if not text.strip():
            return Audio(b"", 22050)
        if sys.platform.startswith("win"):
            return await self._windows(text)
        if sys.platform == "darwin":
            return await self._macos(text)
        out = await self._run([_linux_binary(), "--stdout", "--", text])
        return Audio.from_wav(out)

    async def _windows(self, text: str) -> Audio:
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            env = {**os.environ, "JARVIS_TTS_TEXT": text, "JARVIS_TTS_OUT": tmp}
            await self._run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", "-"],
                stdin=_WIN_SCRIPT.encode(),
                env=env,
            )
            return Audio.from_wav(Path(tmp).read_bytes())
        finally:
            Path(tmp).unlink(missing_ok=True)

    async def _macos(self, text: str) -> Audio:
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            await self._run(["say", "-o", tmp, "--data-format=LEI16@22050", "--", text])
            return Audio.from_wav(Path(tmp).read_bytes())
        finally:
            Path(tmp).unlink(missing_ok=True)


class MockTTS(TTS):
    name = "mock"
    sample_rate = 16000

    def __init__(self, config: Config):
        self.config = config

    async def synthesize(self, text: str) -> Audio:
        seconds = max(0.2, 0.05 * len(text.split()))
        n = int(seconds * self.sample_rate)
        tone = array(
            "h",
            (int(3000 * math.sin(2 * math.pi * 220 * i / self.sample_rate)) for i in range(n)),
        )
        if sys.byteorder == "big":
            tone.byteswap()
        return Audio(tone.tobytes(), self.sample_rate)


def create_tts(config: Config, resources: ResourceManager) -> TTS | None:
    provider = str(config.get("voice.tts.provider", "browser")).lower()
    if provider == "browser":
        return None
    if provider == "piper":
        return PiperTTS(config, resources)
    if provider == "openai":
        return OpenAITTS(config)
    if provider == "system":
        return SystemTTS(config)
    if provider == "mock":
        return MockTTS(config)
    if provider == "auto":
        if importlib.util.find_spec("piper") is not None and find_piper_voice(config):
            return PiperTTS(config, resources)
        if SystemTTS.available():
            return SystemTTS(config)
        if config.secret(config.get("voice.tts.api_key_env")):
            return OpenAITTS(config)
        return None
    raise ValueError(
        f"Unknown voice.tts.provider {provider!r}; use browser, piper, openai, system, mock or auto"
    )


async def download_piper_voice(config: Config, name: str = "en_US-lessac-medium") -> Path:
    """Fetch a Piper voice (model + config) from Hugging Face into ``<data_dir>/voices``."""
    if not _VOICE_NAME.match(name):
        raise ValueError(
            f"Invalid Piper voice name {name!r}; expected e.g. 'en_US-lessac-medium' "
            "(<lang>_<REGION>-<speaker>-<x_low|low|medium|high>)"
        )
    locale, speaker, quality = name.split("-")
    lang = locale.split("_")[0]
    base = f"{_VOICE_URL}/{lang}/{locale}/{speaker}/{quality}/{name}"
    dest = config.data_dir / "voices"
    dest.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
        for suffix in (".onnx", ".onnx.json"):
            target = dest / f"{name}{suffix}"
            if target.exists():
                continue
            part = target.with_name(target.name + ".part")
            try:
                async with client.stream("GET", base + suffix) as resp:
                    resp.raise_for_status()
                    with part.open("wb") as fh:
                        async for chunk in resp.aiter_bytes(1 << 16):
                            fh.write(chunk)
                part.replace(target)
            finally:
                part.unlink(missing_ok=True)
    return dest / f"{name}.onnx"
