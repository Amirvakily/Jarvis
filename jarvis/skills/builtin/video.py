"""Video and audio skills that wrap ffmpeg and ffprobe.

Processes start from argument lists, never through a shell, and every path is
resolved inside the workspace before it reaches ffmpeg.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from jarvis.skills.base import SkillContext, SkillError, skill

_SECONDS = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_CLOCK = re.compile(r"(?:[0-9]{1,3}:)?[0-5]?[0-9]:[0-5]?[0-9](?:\.[0-9]+)?")
_INSTALL_HINT = (
    "Install ffmpeg with: winget install ffmpeg (Windows), brew install ffmpeg (macOS), "
    "or apt install ffmpeg (Linux)."
)
_AUDIO_CODECS = {"mp3": "libmp3lame", "wav": "pcm_s16le", "m4a": "aac"}


def parse_time(value: str) -> float:
    """Convert seconds ("12.5", "90") or a clock time ("01:30", "00:01:30.25") to seconds."""
    text = value.strip() if isinstance(value, str) else ""
    if _SECONDS.fullmatch(text):
        return float(text)
    if _CLOCK.fullmatch(text):
        *head, minutes, seconds = text.split(":")
        hours = int(head[0]) if head else 0
        return hours * 3600 + int(minutes) * 60 + float(seconds)
    raise SkillError(
        f"'{value}' is not a valid time. Use seconds like 12.5 or a clock time like 00:01:30."
    )


@skill(tags=["video"], timeout=600)
async def video_probe(path: str, *, ctx: SkillContext) -> dict:
    """Describe a video or audio file: container, duration, and first video and audio streams.

    Use this before cutting, converting or extracting, to learn the codecs, size and length.

    Args:
        path: Media file path relative to the workspace.
    """
    source = _source(ctx, path)
    raw = await _run(
        _tool("ffprobe"),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(source),
    )
    try:
        info = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise SkillError("ffprobe returned output that could not be read.") from exc
    fmt = info.get("format") or {}
    streams = info.get("streams") or []
    video = next(
        (s for s in streams if s.get("codec_type") == "video" and not _is_cover_art(s)), None
    )
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    return {
        "path": _rel(source, ctx.workspace.resolve()),
        "format": fmt.get("format_name"),
        "duration_seconds": _as_float(fmt.get("duration")),
        "size_bytes": source.stat().st_size,
        "video": None
        if video is None
        else {
            "codec": video.get("codec_name"),
            "width": video.get("width"),
            "height": video.get("height"),
            "fps": _fps(video),
        },
        "audio": None
        if audio is None
        else {
            "codec": audio.get("codec_name"),
            "sample_rate": _as_int(audio.get("sample_rate")),
            "channels": audio.get("channels"),
        },
    }


@skill(tags=["video"], timeout=600)
async def video_clip(
    path: str,
    start: str,
    end: str | None = None,
    duration: str | None = None,
    output: str | None = None,
    reencode: bool = False,
    *,
    ctx: SkillContext,
) -> dict:
    """Cut one segment out of a video or audio file.

    Give either an end time or a duration. The default copies the streams without
    re-encoding, which is fast but may only cut on keyframes; set reencode to true for
    frame-accurate cuts.

    Args:
        path: Media file path relative to the workspace.
        start: Start time in seconds (12.5) or clock form (00:01:30).
        end: End time in the same formats. Use this or duration, not both.
        duration: Segment length in the same formats. Use this or end, not both.
        output: Output path relative to the workspace. Defaults to <name>_clip<ext>.
        reencode: Re-encode to H.264 video and AAC audio for frame-accurate cuts.
    """
    source = _source(ctx, path)
    if (end is None) == (duration is None):
        raise SkillError("Give exactly one of end or duration.")
    start_s = parse_time(start)
    if end is not None:
        length = parse_time(end) - start_s
        if length <= 0:
            raise SkillError("end must be after start.")
    else:
        length = parse_time(duration)
        if length <= 0:
            raise SkillError("duration must be greater than zero.")
    target = _target(ctx, [source], output, "clip", source.suffix)
    codecs = ["-c:v", "libx264", "-c:a", "aac"] if reencode else ["-c", "copy"]
    await _ffmpeg(
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(source),
        "-t",
        f"{length:.3f}",
        *codecs,
        str(target),
    )
    return {
        "output": _rel(target, ctx.workspace.resolve()),
        "duration_seconds": round(length, 3),
        "size_bytes": target.stat().st_size,
    }


@skill(tags=["video"], timeout=600)
async def video_extract_audio(
    path: str,
    format: Literal["mp3", "wav", "m4a"] = "mp3",
    output: str | None = None,
    *,
    ctx: SkillContext,
) -> dict:
    """Save the audio track of a video as its own file.

    Use this to get speech or music out of a video, for example before transcription.

    Args:
        path: Media file path relative to the workspace.
        format: Audio format: mp3, wav or m4a.
        output: Output path relative to the workspace. Defaults to <name>_audio.<format>.
    """
    if format not in _AUDIO_CODECS:
        raise SkillError("format must be one of mp3, wav or m4a.")
    source = _source(ctx, path)
    target = _target(ctx, [source], output, "audio", f".{format}")
    await _ffmpeg("-i", str(source), "-vn", "-c:a", _AUDIO_CODECS[format], str(target))
    return {
        "output": _rel(target, ctx.workspace.resolve()),
        "size_bytes": target.stat().st_size,
    }


@skill(tags=["video"], timeout=600)
async def video_thumbnail(
    path: str,
    at: str = "1",
    output: str | None = None,
    *,
    ctx: SkillContext,
) -> dict:
    """Save one frame of a video as a JPEG image.

    Use this to preview a video or pick a cover image.

    Args:
        path: Media file path relative to the workspace.
        at: Time of the frame, in seconds (12.5) or clock form (00:01:30).
        output: Output path relative to the workspace. Defaults to <name>_thumbnail.jpg.
    """
    source = _source(ctx, path)
    seconds = parse_time(at)
    target = _target(ctx, [source], output, "thumbnail", ".jpg")
    await _ffmpeg(
        "-ss",
        f"{seconds:.3f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(target),
    )
    if not target.exists():
        raise SkillError(f"No video frame was found at {at}.")
    return {
        "output": _rel(target, ctx.workspace.resolve()),
        "size_bytes": target.stat().st_size,
    }


@skill(tags=["video"], timeout=600)
async def video_concat(paths: list[str], output: str, *, ctx: SkillContext) -> dict:
    """Join two or more videos end to end into one file, without re-encoding.

    The files should share codecs and resolution, as ones from the same camera or
    source usually do. Inputs are joined in the order given.

    Args:
        paths: Two or more media file paths relative to the workspace, in playback order.
        output: Output path relative to the workspace.
    """
    if len(paths) < 2:
        raise SkillError("Give at least two files to join.")
    sources = [_source(ctx, p) for p in paths]
    for source in sources:
        if "\n" in str(source) or "\r" in str(source):
            raise SkillError(f"'{source.name}' has a line break in its name and cannot be joined.")
    target = _target(ctx, sources, output, "concat", "")
    with tempfile.TemporaryDirectory() as scratch:
        listing = Path(scratch) / "inputs.txt"
        lines = ["file '" + str(s).replace("'", "'\\''") + "'" for s in sources]
        listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
        await _ffmpeg(
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            str(target),
        )
    return {
        "output": _rel(target, ctx.workspace.resolve()),
        "inputs": len(sources),
        "size_bytes": target.stat().st_size,
    }


@skill(tags=["video"], timeout=600)
async def video_transcode(
    path: str,
    format: Literal["mp4", "webm", "mkv", "gif"] = "mp4",
    height: int | None = None,
    crf: int = 23,
    output: str | None = None,
    *,
    ctx: SkillContext,
) -> dict:
    """Re-encode a video into another format, optionally scaling it to a new height.

    Use this to shrink a video for sharing, change its container, or make a GIF. MP4 and
    MKV use H.264, WebM uses VP9 with Opus audio, and GIF is silent at 12 frames per second.

    Args:
        path: Media file path relative to the workspace.
        format: Output format: mp4, webm, mkv or gif.
        height: Target height in pixels, from 16 to 4320; width follows the aspect ratio.
            Odd heights are rounded up to the next even number.
        crf: Constant quality from 0 (best, largest) to 51 (smallest). Ignored for GIF.
        output: Output path relative to the workspace. Defaults to <name>_transcode.<format>.
    """
    source = _source(ctx, path)
    if height is not None and not 16 <= height <= 4320:
        raise SkillError("height must be between 16 and 4320 pixels.")
    if not 0 <= crf <= 51:
        raise SkillError("crf must be between 0 and 51.")
    filters: list[str] = []
    if format == "gif":
        filters.append("fps=12")
    if height is not None:
        filters.append(f"scale=-2:{height + height % 2}")
    target = _target(ctx, [source], output, "transcode", f".{format}")

    args = ["-i", str(source)]
    if filters:
        args += ["-vf", ",".join(filters)]
    if format == "gif":
        args += ["-an", "-f", "gif"]
    elif format == "webm":
        args += ["-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0", "-c:a", "libopus"]
    else:
        args += [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
        ]
    await _ffmpeg(*args, str(target))
    return {
        "output": _rel(target, ctx.workspace.resolve()),
        "format": format,
        "size_bytes": target.stat().st_size,
    }


# -- helpers ------------------------------------------------------------------


def _tool(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise SkillError(f"{name} is not installed. {_INSTALL_HINT}")
    return found


def _rel(path: Path, workspace: Path) -> str:
    return path.relative_to(workspace).as_posix()


def _source(ctx: SkillContext, path: str) -> Path:
    target = ctx.resolve_path(path, must_exist=True)
    if not target.is_file():
        raise SkillError(f"'{path}' is not a file.")
    return target


def _target(
    ctx: SkillContext,
    sources: list[Path],
    output: str | None,
    operation: str,
    suffix: str,
) -> Path:
    if output is None:
        first = sources[0]
        target = first.with_name(f"{first.stem}_{operation}{suffix}")
    else:
        target = ctx.resolve_path(output)
    if target.is_dir():
        raise SkillError(f"'{output}' is a folder; give a file name instead.")
    if target in sources:
        raise SkillError("The output must be a different file from the input.")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


async def _run(exe: str, *args: str) -> bytes:
    process = await asyncio.create_subprocess_exec(
        exe,
        *args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await process.communicate()
    except asyncio.CancelledError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        await process.wait()
        raise
    if process.returncode != 0:
        tail = stderr.decode("utf-8", errors="replace").strip()[-300:]
        raise SkillError(f"{Path(exe).stem} failed: {tail or 'no error output'}")
    return stdout


async def _ffmpeg(*args: str) -> None:
    await _run(_tool("ffmpeg"), "-hide_banner", "-nostdin", "-y", *args)


def _is_cover_art(stream: dict) -> bool:
    return bool((stream.get("disposition") or {}).get("attached_pic"))


def _fps(stream: dict) -> float | None:
    for key in ("avg_frame_rate", "r_frame_rate"):
        numerator, _, denominator = str(stream.get(key) or "").partition("/")
        try:
            top, bottom = float(numerator), float(denominator or 1)
        except ValueError:
            continue
        if top and bottom:
            return round(top / bottom, 3)
    return None


def _as_float(value: object) -> float | None:
    try:
        return round(float(value), 3)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> int | None:
    number = _as_float(value)
    return None if number is None else int(number)
