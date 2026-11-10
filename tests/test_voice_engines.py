import math
import os
import struct

import pytest

from jarvis._http import httpx
from jarvis.config import Config
from jarvis.resources import ResourceManager
from jarvis.voice.audio import Audio, resample, rms
from jarvis.voice.stt import FasterWhisperSTT, MockSTT, OpenAISTT, create_stt
from jarvis.voice.tts import (
    MockTTS,
    OpenAITTS,
    PiperTTS,
    SystemTTS,
    create_tts,
    download_piper_voice,
)


def tone(seconds=0.5, rate=16000, freq=440, amp=8000) -> Audio:
    n = int(seconds * rate)
    pcm = struct.pack(
        f"<{n}h", *(int(amp * math.sin(2 * math.pi * freq * i / rate)) for i in range(n))
    )
    return Audio(pcm, rate)


def cfg(tmp_path, **voice) -> Config:
    return Config({"assistant": {"data_dir": str(tmp_path / "data")}, "voice": voice})


def test_wav_round_trip():
    a = tone()
    b = Audio.from_wav(a.to_wav())
    assert b.pcm == a.pcm and b.sample_rate == 16000
    assert a.duration == pytest.approx(0.5)


def test_stereo_downmix():
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(struct.pack("<4h", 100, 300, -200, 200))
    a = Audio.from_wav(buf.getvalue())
    assert struct.unpack("<2h", a.pcm) == (200, 0)


def test_from_wav_rejects_8bit_and_garbage():
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(8000)
        w.writeframes(b"\x80" * 10)
    with pytest.raises(ValueError):
        Audio.from_wav(buf.getvalue())
    with pytest.raises(ValueError):
        Audio.from_wav(b"nope")


def test_rms():
    assert rms(b"\x00\x00" * 100) == 0.0
    assert rms(b"") == 0.0
    assert 0.16 < rms(tone().pcm) < 0.19


def test_resample_length_and_identity():
    a = tone(1.0, 24000)
    out = resample(a.pcm, 24000, 16000)
    assert len(out) // 2 == 16000
    assert resample(a.pcm, 24000, 24000) == a.pcm
    assert abs(rms(out) - rms(a.pcm)) < 0.02


def test_resample_pure_python(monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name, *args, **kw):
        if name == "numpy":
            raise ImportError
        return real(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", fake)
    out = resample(tone(1.0, 8000).pcm, 8000, 16000)
    assert len(out) // 2 == 16000


async def test_mock_engines(tmp_path):
    c = cfg(tmp_path, stt={"mock_text": " hi there "})
    assert await MockSTT(c).transcribe(tone()) == "hi there"
    assert await MockSTT(cfg(tmp_path)).transcribe(tone()) == "hello jarvis"
    short = await MockTTS(c).synthesize("hi")
    assert short.sample_rate == 16000 and short.duration == pytest.approx(0.2)
    longer = await MockTTS(c).synthesize(" ".join(["w"] * 10))
    assert longer.duration == pytest.approx(0.5)
    assert rms(longer.pcm) > 0.01


async def test_openai_stt(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["ctype"] = request.headers["content-type"]
        seen["body"] = request.content
        return httpx.Response(200, json={"text": "  hello world "})

    stt = OpenAISTT(cfg(tmp_path), transport=httpx.MockTransport(handler))
    assert await stt.transcribe(tone(), language="de") == "hello world"
    await stt.aclose()
    assert seen["url"] == "https://api.openai.com/v1/audio/transcriptions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["ctype"].startswith("multipart/form-data")
    body = seen["body"]
    assert b'filename="speech.wav"' in body and b"RIFF" in body
    assert b"whisper-1" in body and b"base.en" not in body
    assert b'name="language"' in body and b"de" in body


async def test_openai_stt_custom_model(tmp_path):
    seen = {}

    def handler(request):
        seen["body"] = request.content
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"text": "x"})

    c = cfg(tmp_path, stt={"model": "gpt-4o-transcribe", "base_url": "http://srv/v1/"})
    await OpenAISTT(c, transport=httpx.MockTransport(handler)).transcribe(tone())
    assert b"gpt-4o-transcribe" in seen["body"]
    assert seen["url"] == "http://srv/v1/audio/transcriptions"


async def test_openai_tts(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen = {}
    pcm = b"\x01\x00" * 2400

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["json"] = json.loads(request.content)
        return httpx.Response(200, content=pcm)

    tts = OpenAITTS(cfg(tmp_path), transport=httpx.MockTransport(handler))
    audio = await tts.synthesize("Hello")
    await tts.aclose()
    assert seen["url"] == "https://api.openai.com/v1/audio/speech"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["json"] == {
        "model": "tts-1",
        "voice": "alloy",
        "input": "Hello",
        "response_format": "pcm",
    }
    assert audio.pcm == pcm and audio.sample_rate == 24000


def test_create_stt_providers(tmp_path, monkeypatch):
    res = ResourceManager()
    # The default is "auto" (covered below); "browser" means the window handles speech.
    assert create_stt(cfg(tmp_path, stt={"provider": "browser"}), res) is None
    assert isinstance(create_stt(cfg(tmp_path, stt={"provider": "mock"}), res), MockSTT)
    assert isinstance(create_stt(cfg(tmp_path, stt={"provider": "openai"}), res), OpenAISTT)
    fw = create_stt(cfg(tmp_path, stt={"provider": "faster-whisper"}), res)
    assert isinstance(fw, FasterWhisperSTT) and "stt:faster-whisper" in res
    with pytest.raises(ValueError):
        create_stt(cfg(tmp_path, stt={"provider": "bogus"}), res)


def test_create_stt_auto(tmp_path, monkeypatch):
    import importlib.util

    c = cfg(tmp_path, stt={"provider": "auto"})
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert create_stt(c, ResourceManager()) is None
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert isinstance(create_stt(c, ResourceManager()), OpenAISTT)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert isinstance(create_stt(c, ResourceManager()), FasterWhisperSTT)


def test_create_tts_providers(tmp_path):
    res = ResourceManager()
    assert create_tts(cfg(tmp_path, tts={"provider": "browser"}), res) is None
    assert isinstance(create_tts(cfg(tmp_path, tts={"provider": "mock"}), res), MockTTS)
    assert isinstance(create_tts(cfg(tmp_path, tts={"provider": "openai"}), res), OpenAITTS)
    assert isinstance(create_tts(cfg(tmp_path, tts={"provider": "system"}), res), SystemTTS)
    assert isinstance(create_tts(cfg(tmp_path, tts={"provider": "piper"}), res), PiperTTS)
    assert "tts:piper" in res
    with pytest.raises(ValueError):
        create_tts(cfg(tmp_path, tts={"provider": "bogus"}), res)


def test_create_tts_auto(tmp_path, monkeypatch):
    import importlib.util

    from jarvis.voice import tts as tts_mod

    c = cfg(tmp_path, tts={"provider": "auto"})
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    voices = c.data_dir / "voices"
    voices.mkdir()
    (voices / "v.onnx").write_bytes(b"x")
    assert isinstance(create_tts(c, ResourceManager()), PiperTTS)
    (voices / "v.onnx").unlink()
    monkeypatch.setattr(tts_mod.SystemTTS, "available", staticmethod(lambda: True))
    assert isinstance(create_tts(c, ResourceManager()), SystemTTS)
    monkeypatch.setattr(tts_mod.SystemTTS, "available", staticmethod(lambda: False))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert create_tts(c, ResourceManager()) is None
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert isinstance(create_tts(c, ResourceManager()), OpenAITTS)


async def test_piper_without_voice_tells_user(tmp_path):
    tts = PiperTTS(cfg(tmp_path), ResourceManager())
    with pytest.raises(RuntimeError, match="setup-voice"):
        await tts.synthesize("hello")


def test_piper_render_handles_both_apis():
    import wave

    class New:
        def synthesize_wav(self, *a):  # marks the newer API
            raise AssertionError

        def synthesize(self, text):
            class Chunk:
                sample_rate = 22050
                audio_int16_bytes = b"\x01\x00" * 100

            yield Chunk()
            yield Chunk()

    class Old:
        def synthesize(self, text, wav):
            assert isinstance(wav, wave.Wave_write)
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x02\x00" * 50)

    new = PiperTTS._render(New(), "hi")
    assert new.sample_rate == 22050 and len(new.pcm) == 400
    old = PiperTTS._render(Old(), "hi")
    assert old.sample_rate == 16000 and len(old.pcm) == 100


@pytest.mark.parametrize(
    "name",
    ["", "bogus", "../etc-passwd", "en_US-lessac-huge", "en_us-lessac-medium", "en_US-L-low"],
)
async def test_download_rejects_bad_names(tmp_path, name):
    with pytest.raises(ValueError):
        await download_piper_voice(cfg(tmp_path), name)


async def test_download_skips_existing(tmp_path):
    c = cfg(tmp_path)
    voices = c.data_dir / "voices"
    voices.mkdir()
    for suffix in (".onnx", ".onnx.json"):
        (voices / f"en_US-lessac-medium{suffix}").write_bytes(b"x")
    path = await download_piper_voice(c)
    assert path == voices / "en_US-lessac-medium.onnx"


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("JARVIS_REAL_SPEECH") != "1", reason="set JARVIS_REAL_SPEECH=1")
async def test_real_round_trip(tmp_path):
    pytest.importorskip("faster_whisper")
    pytest.importorskip("piper")
    c = cfg(
        tmp_path,
        stt={"model": "tiny.en", "device": "cpu", "compute_type": "int8"},
        tts={"provider": "piper"},
    )
    await download_piper_voice(c)
    res = ResourceManager()
    audio = await PiperTTS(c, res).synthesize("The quick brown fox jumps over the lazy dog")
    assert audio.duration > 1
    text = await FasterWhisperSTT(c, res).transcribe(audio, "en")
    assert "quick brown fox" in text.lower()
    await res.shutdown()
