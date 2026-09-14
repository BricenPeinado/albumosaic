"""Tests for FFmpeg finalization and conditional audio handling."""

from pathlib import Path
from subprocess import CompletedProcess

import pytest

from app.video import audio


def create_placeholder_inputs(tmp_path: Path) -> tuple[Path, Path]:
    silent_path = tmp_path / "silent.mp4"
    source_path = tmp_path / "source.mp4"
    silent_path.write_bytes(b"silent")
    source_path.write_bytes(b"source")
    return silent_path, source_path


def test_finalizer_stream_copies_compatible_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    silent_path, source_path = create_placeholder_inputs(tmp_path)
    output_path = tmp_path / "output.mp4"
    commands: list[list[str]] = []

    monkeypatch.setattr(audio, "audio_codec_name", lambda _: "aac")

    def successful_run(arguments: list[str]) -> CompletedProcess[str]:
        commands.append(arguments)
        return CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(audio, "_run_ffmpeg", successful_run)

    audio.finalize_h264_mp4(silent_path, source_path, output_path)

    assert len(commands) == 1
    assert _option_value(commands[0], "-c:v") == "libx264"
    assert _option_value(commands[0], "-c:a") == "copy"


def test_finalizer_reencodes_incompatible_audio_as_aac(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    silent_path, source_path = create_placeholder_inputs(tmp_path)
    output_path = tmp_path / "output.mp4"
    commands: list[list[str]] = []

    monkeypatch.setattr(audio, "audio_codec_name", lambda _: "pcm_s16le")

    def successful_run(arguments: list[str]) -> CompletedProcess[str]:
        commands.append(arguments)
        return CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(audio, "_run_ffmpeg", successful_run)

    audio.finalize_h264_mp4(silent_path, source_path, output_path)

    assert len(commands) == 1
    assert _option_value(commands[0], "-c:a") == "aac"


def test_finalizer_keeps_no_audio_input_silent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    silent_path, source_path = create_placeholder_inputs(tmp_path)
    output_path = tmp_path / "output.mp4"
    commands: list[list[str]] = []

    monkeypatch.setattr(audio, "audio_codec_name", lambda _: None)

    def successful_run(arguments: list[str]) -> CompletedProcess[str]:
        commands.append(arguments)
        return CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(audio, "_run_ffmpeg", successful_run)

    audio.finalize_h264_mp4(silent_path, source_path, output_path)

    assert len(commands) == 1
    assert "-an" in commands[0]
    assert "-c:a" not in commands[0]


def test_ffmpeg_preflight_requires_both_executables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audio, "which", lambda name: f"/tools/{name}")

    assert audio.ensure_ffmpeg_available() == ("/tools/ffmpeg", "/tools/ffprobe")


def test_finalizer_rejects_overwriting_an_input(tmp_path: Path) -> None:
    silent_path, source_path = create_placeholder_inputs(tmp_path)

    with pytest.raises(ValueError, match="must differ"):
        audio.finalize_h264_mp4(silent_path, source_path, source_path)
    with pytest.raises(ValueError, match="must differ"):
        audio.finalize_h264_mp4(silent_path, source_path, silent_path)


def _option_value(arguments: list[str], option: str) -> str:
    index = arguments.index(option)
    return arguments[index + 1]
