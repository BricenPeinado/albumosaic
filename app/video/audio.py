"""FFprobe and FFmpeg utilities for final H.264 MP4 assembly."""

from collections.abc import Sequence
from os import close as close_file_descriptor
from pathlib import Path
from shutil import which
from subprocess import CalledProcessError, CompletedProcess, run
from tempfile import mkstemp

from app.video.errors import FFmpegError

_MP4_AUDIO_COPY_CODECS = {"aac", "ac3", "alac", "eac3", "mp3"}


def ensure_ffmpeg_available() -> tuple[str, str]:
    """Return FFmpeg and FFprobe paths, or fail before expensive rendering."""
    return _require_binary("ffmpeg"), _require_binary("ffprobe")


def has_audio_stream(video_path: str | Path) -> bool:
    """Return whether FFprobe finds at least one audio stream."""
    return audio_codec_name(video_path) is not None


def audio_codec_name(video_path: str | Path) -> str | None:
    """Return the first audio stream's codec, or None for a silent video."""
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {path}")

    result = _run_command(
        [
            _require_binary("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ]
    )
    codec_name = result.stdout.strip()
    return codec_name or None


def video_codec_name(video_path: str | Path) -> str:
    """Return the first video stream's FFprobe codec name."""
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {path}")

    result = _run_command(
        [
            _require_binary("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ]
    )
    codec_name = result.stdout.strip()
    if not codec_name:
        raise FFmpegError(f"No video stream found in {path}")
    return codec_name


def finalize_h264_mp4(
    silent_video_path: str | Path,
    source_video_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Encode H.264 video and preserve source audio when one is present.

    MP4-compatible source audio is stream-copied. Other audio codecs are
    encoded as AAC because they cannot be retained in a broadly playable MP4.
    """
    silent_path = Path(silent_video_path)
    source_path = Path(source_video_path)
    destination_path = Path(output_path)
    if not silent_path.is_file():
        raise FileNotFoundError(f"Silent video does not exist: {silent_path}")
    if not source_path.is_file():
        raise FileNotFoundError(f"Source video does not exist: {source_path}")
    if destination_path.suffix.lower() != ".mp4":
        raise ValueError("Final video output must use the .mp4 extension")
    resolved_destination = destination_path.resolve()
    if resolved_destination in {silent_path.resolve(), source_path.resolve()}:
        raise ValueError("Final output path must differ from both input video paths")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_mp4_path(destination_path)

    try:
        source_audio_codec = audio_codec_name(source_path)
        if source_audio_codec is None:
            output_audio_codec = None
        elif source_audio_codec in _MP4_AUDIO_COPY_CODECS:
            output_audio_codec = "copy"
        else:
            output_audio_codec = "aac"
        _run_ffmpeg(
            _finalize_arguments(
                silent_path,
                source_path,
                temporary_path,
                audio_codec=output_audio_codec,
            )
        )

        temporary_path.replace(destination_path)
        return destination_path
    finally:
        temporary_path.unlink(missing_ok=True)


def restore_audio(
    silent_video_path: Path,
    source_video_path: Path,
    output_path: Path,
) -> Path:
    """Backward-compatible name for H.264 finalization and audio restoration."""
    return finalize_h264_mp4(silent_video_path, source_video_path, output_path)


def _finalize_arguments(
    silent_path: Path,
    source_path: Path,
    output_path: Path,
    *,
    audio_codec: str | None,
) -> list[str]:
    arguments = [
        "-y",
        "-v",
        "error",
        "-i",
        str(silent_path),
    ]
    if audio_codec is not None:
        arguments.extend(
            [
                "-i",
                str(source_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
            ]
        )
    else:
        arguments.extend(["-map", "0:v:0"])

    arguments.extend(
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    )
    if audio_codec is None:
        arguments.append("-an")
    elif audio_codec == "copy":
        arguments.extend(["-c:a", "copy"])
    else:
        arguments.extend(["-c:a", "aac", "-b:a", "192k"])

    arguments.append(str(output_path))
    return arguments


def _run_ffmpeg(arguments: Sequence[str]) -> CompletedProcess[str]:
    return _run_command([_require_binary("ffmpeg"), *arguments])


def _run_command(command: Sequence[str]) -> CompletedProcess[str]:
    try:
        return run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
        )
    except CalledProcessError as error:
        detail = (error.stderr or error.stdout or "unknown FFmpeg error").strip()
        raise FFmpegError(detail) from error
    except OSError as error:
        raise FFmpegError(f"Could not execute {command[0]}: {error}") from error


def _require_binary(name: str) -> str:
    executable = which(name)
    if executable is None:
        raise FFmpegError(f"Required executable is not available on PATH: {name}")
    return executable


def _temporary_mp4_path(destination_path: Path) -> Path:
    file_descriptor, temporary_name = mkstemp(
        prefix=".albumosaic-final-",
        suffix=".mp4",
        dir=destination_path.parent,
    )
    close_file_descriptor(file_descriptor)
    return Path(temporary_name)
