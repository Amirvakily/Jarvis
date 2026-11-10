"""Speech-to-text engines: local faster-whisper, OpenAI-compatible HTTP, and a mock."""

from __future__ import annotations

import asyncio
import gc
import importlib.util
import re
from typing import Any

from jarvis._http import httpx
from jarvis.config import Config
from jarvis.resources import ResourceManager
from jarvis.voice.audio import Audio, resample

_LOCAL_MODEL = re.compile(r"^(tiny|base|small|medium|large|turbo|distil)([-._\w]*)$", re.I)


class STT:
    name = "stt"

    async def transcribe(self, audio: Audio, language: str | None = None) -> str:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


def _language(config: Config, language: str | None) -> str | None:
    return language or config.get("voice.language") or None


class FasterWhisperSTT(STT):
    name = "faster-whisper"
    resource_name = "stt:faster-whisper"

    def __init__(self, config: Config, resources: ResourceManager):
        self.config = config
        self.resources = resources
        resources.register(
            self.resource_name,
            self._load,
            self._unload,
            memory_mb=int(config.get("voice.stt.memory_mb", 500)),
            kind="stt",
        )

    async def _load(self) -> Any:
        def build() -> Any:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is not installed: pip install 'jarvis-assistant[voice]'"
                ) from exc
            return WhisperModel(
                self.config.get("voice.stt.model", "base.en"),
                device=self.config.get("voice.stt.device", "auto"),
                compute_type=self.config.get("voice.stt.compute_type", "int8"),
            )

        return await asyncio.to_thread(build)

    @staticmethod
    def _unload(model: Any) -> None:
        del model
        gc.collect()

    async def transcribe(self, audio: Audio, language: str | None = None) -> str:
        import numpy as np

        pcm = resample(audio.pcm, audio.sample_rate, 16000)
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if samples.size == 0:
            return ""
        lang = _language(self.config, language)

        async with self.resources.use(self.resource_name) as model:

            def run() -> str:
                segments, _info = model.transcribe(
                    samples, language=lang, vad_filter=True, beam_size=1
                )
                return " ".join(seg.text.strip() for seg in segments)

            text = await asyncio.to_thread(run)
        return " ".join(text.split())


class OpenAISTT(STT):
    name = "openai"

    def __init__(self, config: Config, transport: Any = None):
        self.config = config
        self._client = httpx.AsyncClient(transport=transport, timeout=60)

    def _model(self) -> str:
        configured = str(self.config.get("voice.stt.model") or "")
        if not configured or _LOCAL_MODEL.match(configured):
            return "whisper-1"
        return configured

    async def transcribe(self, audio: Audio, language: str | None = None) -> str:
        base = str(self.config.get("voice.stt.base_url")).rstrip("/")
        headers = {}
        key = self.config.secret(self.config.get("voice.stt.api_key_env"))
        if key:
            headers["Authorization"] = f"Bearer {key}"
        data = {"model": self._model()}
        lang = _language(self.config, language)
        if lang:
            data["language"] = lang
        resp = await self._client.post(
            f"{base}/audio/transcriptions",
            headers=headers,
            data=data,
            files={"file": ("speech.wav", audio.to_wav(), "audio/wav")},
        )
        resp.raise_for_status()
        return str(resp.json().get("text", "")).strip()

    async def aclose(self) -> None:
        await self._client.aclose()


class MockSTT(STT):
    name = "mock"

    def __init__(self, config: Config):
        self.text = str(config.get("voice.stt.mock_text", "hello jarvis"))

    async def transcribe(self, audio: Audio, language: str | None = None) -> str:
        return self.text.strip()


def create_stt(config: Config, resources: ResourceManager) -> STT | None:
    provider = str(config.get("voice.stt.provider", "browser")).lower()
    if provider == "browser":
        return None
    if provider == "faster-whisper":
        return FasterWhisperSTT(config, resources)
    if provider == "openai":
        return OpenAISTT(config)
    if provider == "mock":
        return MockSTT(config)
    if provider == "auto":
        if importlib.util.find_spec("faster_whisper") is not None:
            return FasterWhisperSTT(config, resources)
        if config.secret(config.get("voice.stt.api_key_env")):
            return OpenAISTT(config)
        return None
    raise ValueError(
        f"Unknown voice.stt.provider {provider!r}; use browser, faster-whisper, openai, mock or auto"
    )
