"""Tests for the ffmpeg-backed video skills. Requires ffmpeg on PATH."""

from __future__ import annotations

import asyncio
import shutil
import subprocess

import pytest

from jarvis.skills.base import SkillError
from jarvis.skills.builtin import video
from jarvis.skills.builtin.video import (
    parse_time,
    video_clip,
    video_concat,
    video_extract_audio,
    video_probe,
    video_thumbnail,
    video_transcode,
)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


@pytest.fixture
def sample(ctx):
    """A 3-second 320x240 clip with a 440 Hz tone, built in the workspace."""
    ctx.workspace.mkdir(parents=True, exist_ok=True)
    path = ctx.workspace / "sample.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=3:size=320x240:rate=15",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-shortest",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _run(coro):
    return asyncio.run(coro)


def _probe(ctx, rel: str) -> dict:
    return _run(video_probe(rel, ctx=ctx))


@pytest.mark.parametrize(
    "value, seconds",
    [
        ("12.5", 12.5),
        ("90", 90.0),
        ("0", 0.0),
        ("01:30", 90.0),
        ("00:01:30", 90.0),
        ("00:00:01.5", 1.5),
        ("1:02:03.25", 3723.25),
    ],
)
def test_parse_time_accepts(value, seconds):
    assert parse_time(value) == pytest.approx(seconds)


@pytest.mark.parametrize(
    "value",
    ["", "abc", "-1", "1:75", "1e3", "1:2:3:4", "1.2.3", "nan", "inf", "١٢", "12 ; rm"],
)
def test_parse_time_rejects(value):
    with pytest.raises(SkillError):
        parse_time(value)


def test_probe_reports_streams(ctx, sample):
    info = _probe(ctx, "sample.mp4")
    assert info["path"] == "sample.mp4"
    assert "mp4" in info["format"]
    assert info["duration_seconds"] == pytest.approx(3.0, abs=0.2)
    assert info["size_bytes"] > 0
    assert info["video"]["codec"] == "h264"
    assert (info["video"]["width"], info["video"]["height"]) == (320, 240)
    assert info["video"]["fps"] == pytest.approx(15.0)
    assert info["audio"]["codec"] == "aac"
    assert info["audio"]["channels"] == 1


def test_probe_missing_input(ctx):
    with pytest.raises(SkillError, match="does not exist"):
        _run(video_probe("missing.mp4", ctx=ctx))


def test_clip_stream_copy_default_name(ctx, sample):
    result = _run(video_clip("sample.mp4", start="0.5", duration="1", ctx=ctx))
    assert result["output"] == "sample_clip.mp4"
    assert result["duration_seconds"] == 1.0
    assert result["size_bytes"] > 0
    assert _probe(ctx, "sample_clip.mp4")["video"]["codec"] == "h264"


def test_clip_reencode_is_frame_accurate(ctx, sample):
    result = _run(
        video_clip(
            "sample.mp4",
            start="00:00:01",
            end="00:00:02",
            output="cut.mp4",
            reencode=True,
            ctx=ctx,
        )
    )
    assert result["output"] == "cut.mp4"
    assert _probe(ctx, "cut.mp4")["duration_seconds"] == pytest.approx(1.0, abs=0.1)


def test_clip_argument_rules(ctx, sample):
    with pytest.raises(SkillError, match="exactly one"):
        _run(video_clip("sample.mp4", "0", end="1", duration="1", ctx=ctx))
    with pytest.raises(SkillError, match="exactly one"):
        _run(video_clip("sample.mp4", "0", ctx=ctx))
    with pytest.raises(SkillError, match="after start"):
        _run(video_clip("sample.mp4", "2", end="1", ctx=ctx))
    with pytest.raises(SkillError):
        _run(video_clip("sample.mp4", "soon", end="1", ctx=ctx))


def test_clip_refuses_to_overwrite_input(ctx, sample):
    with pytest.raises(SkillError, match="different file"):
        _run(video_clip("sample.mp4", "0", duration="1", output="sample.mp4", ctx=ctx))


@pytest.mark.parametrize(
    "fmt, codec",
    [("mp3", "mp3"), ("wav", "pcm_s16le"), ("m4a", "aac")],
)
def test_extract_audio(ctx, sample, fmt, codec):
    result = _run(video_extract_audio("sample.mp4", format=fmt, ctx=ctx))
    assert result["output"] == f"sample_audio.{fmt}"
    info = _probe(ctx, result["output"])
    assert info["video"] is None
    assert info["audio"]["codec"] == codec


def test_thumbnail_writes_jpeg(ctx, sample):
    result = _run(video_thumbnail("sample.mp4", at="00:00:01", ctx=ctx))
    assert result["output"] == "sample_thumbnail.jpg"
    assert (ctx.workspace / "sample_thumbnail.jpg").read_bytes().startswith(b"\xff\xd8\xff")


def test_concat_joins_files_with_quotes_in_names(ctx, sample):
    shutil.copy(sample, ctx.workspace / "it's.mp4")
    result = _run(video_concat(["sample.mp4", "it's.mp4"], output="joined.mp4", ctx=ctx))
    assert result["output"] == "joined.mp4"
    assert result["inputs"] == 2
    assert _probe(ctx, "joined.mp4")["duration_seconds"] == pytest.approx(6.0, abs=0.3)


def test_concat_needs_two_inputs(ctx, sample):
    with pytest.raises(SkillError, match="at least two"):
        _run(video_concat(["sample.mp4"], output="joined.mp4", ctx=ctx))


def test_transcode_scales_height_keeping_aspect(ctx, sample):
    result = _run(video_transcode("sample.mp4", height=120, output="small.mp4", ctx=ctx))
    assert result["output"] == "small.mp4"
    info = _probe(ctx, "small.mp4")
    assert (info["video"]["width"], info["video"]["height"]) == (160, 120)


def test_transcode_to_gif_has_no_audio(ctx, sample):
    result = _run(video_transcode("sample.mp4", format="gif", ctx=ctx))
    assert result["output"] == "sample_transcode.gif"
    info = _probe(ctx, result["output"])
    assert info["video"]["codec"] == "gif"
    assert info["audio"] is None


def test_transcode_to_webm(ctx, sample):
    result = _run(video_transcode("sample.mp4", format="webm", height=96, ctx=ctx))
    assert result["output"] == "sample_transcode.webm"
    info = _probe(ctx, result["output"])
    assert "webm" in info["format"]
    assert info["video"]["height"] == 96


@pytest.mark.parametrize(
    "kwargs",
    [{"height": 15}, {"height": 4321}, {"crf": 52}, {"crf": -1}],
)
def test_transcode_rejects_out_of_range(ctx, sample, kwargs):
    with pytest.raises(SkillError):
        _run(video_transcode("sample.mp4", ctx=ctx, **kwargs))


def test_paths_outside_workspace_are_refused(ctx, sample):
    outside = ctx.workspace.parent / "outside.mp4"
    shutil.copy(sample, outside)
    calls = [
        lambda: video_probe("../outside.mp4", ctx=ctx),
        lambda: video_clip("../outside.mp4", "0", duration="1", ctx=ctx),
        lambda: video_extract_audio(str(outside), ctx=ctx),
        lambda: video_thumbnail("../outside.mp4", ctx=ctx),
        lambda: video_concat(["sample.mp4", "../outside.mp4"], output="joined.mp4", ctx=ctx),
        lambda: video_transcode("../outside.mp4", ctx=ctx),
        lambda: video_clip("sample.mp4", "0", duration="1", output="../escaped.mp4", ctx=ctx),
        lambda: video_concat(["sample.mp4", "sample.mp4"], output="../joined.mp4", ctx=ctx),
    ]
    for make_call in calls:
        with pytest.raises(SkillError, match="outside the workspace"):
            _run(make_call())
    assert not (ctx.workspace.parent / "escaped.mp4").exists()


def test_missing_input_is_skill_error(ctx):
    with pytest.raises(SkillError, match="does not exist"):
        _run(video_thumbnail("missing.mp4", ctx=ctx))


def test_missing_ffmpeg_gives_install_hint(ctx, sample, monkeypatch):
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    with pytest.raises(SkillError, match="winget install ffmpeg"):
        _run(video_thumbnail("sample.mp4", ctx=ctx))


def test_cancelling_a_skill_stops_it(ctx, sample):
    with pytest.raises(TimeoutError):
        _run(asyncio.wait_for(video_transcode("sample.mp4", ctx=ctx), timeout=0.01))
