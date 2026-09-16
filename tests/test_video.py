"""Integration tests for streaming silent mosaic video rendering."""

import logging
from pathlib import Path
from shutil import which

import cv2
import numpy as np
import pytest
from PIL import Image

import app.video.renderer as video_renderer_module
from app.mosaic.matcher import MatchMode
from app.mosaic.renderer import MosaicRenderPlan
from app.video import audio
from app.video.audio import audio_codec_name, has_audio_stream, video_codec_name
from app.video.errors import CorruptVideoError, UnsupportedCodecError, VideoWriteError
from app.video.reader import VideoReader, read_frames
from app.video.renderer import VideoRenderTiming, render_video
from app.video.writer import SilentVideoWriter


def create_test_video(
    path: Path,
    *,
    width: int = 32,
    height: int = 18,
    frames_per_second: float = 8.0,
    frame_count: int = 4,
) -> None:
    """Write a tiny MJPG fixture using the same OpenCV runtime as the app."""
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        frames_per_second,
        (width, height),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG encoder is unavailable")

    try:
        for index in range(frame_count):
            frame = np.full(
                (height, width, 3),
                fill_value=index * 50,
                dtype=np.uint8,
            )
            writer.write(frame)
    finally:
        writer.release()


def test_reader_reports_metadata_and_streams_frames(tmp_path: Path) -> None:
    input_path = tmp_path / "input.avi"
    create_test_video(input_path)

    metadata, frames = read_frames(input_path)

    assert metadata.width == 32
    assert metadata.height == 18
    assert metadata.frames_per_second == pytest.approx(8.0)
    assert metadata.frame_count == 4
    assert metadata.duration == pytest.approx(0.5)
    assert len(list(frames)) == 4


def test_render_video_preserves_geometry_timing_and_reports_progress(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    input_path = tmp_path / "input.avi"
    output_path = tmp_path / "mosaic.mp4"
    create_test_video(input_path)
    album_tiles = [
        Image.new("RGB", (20, 20), "black"),
        Image.new("RGB", (20, 20), "white"),
    ]
    progress: list[tuple[int, int]] = []
    timings: list[VideoRenderTiming] = []

    with caplog.at_level(logging.DEBUG, logger="app.video.renderer"):
        result = render_video(
            input_path,
            album_tiles,
            tile_count=2,
            output_path=output_path,
            progress_callback=lambda processed, total: progress.append(
                (processed, total)
            ),
            timing_callback=timings.append,
        )

    assert result == output_path
    assert output_path.is_file()
    assert has_audio_stream(output_path) is False
    assert audio_codec_name(output_path) is None
    assert video_codec_name(output_path) == "h264"
    assert not list(tmp_path.glob(".albumosaic-*"))
    assert progress == [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]
    assert len(timings) == 1
    assert timings[0].frame_count == 4
    assert timings[0].total_seconds > 0
    assert "target color calculation" in timings[0].format()
    assert "matching" in caplog.text
    assert "composition" in caplog.text
    assert "frame encoding" in caplog.text
    with VideoReader(output_path) as reader:
        assert reader.metadata.width == 32
        assert reader.metadata.height == 18
        assert reader.metadata.frames_per_second == pytest.approx(8.0)
        assert reader.metadata.frame_count == 4
        assert len(list(reader)) == 4


def test_render_video_preserves_existing_audio(tmp_path: Path) -> None:
    if which("ffmpeg") is None:
        pytest.skip("FFmpeg is unavailable")

    silent_source = tmp_path / "silent-source.avi"
    source_with_audio = tmp_path / "source-with-audio.mp4"
    output_path = tmp_path / "mosaic-with-audio.mp4"
    create_test_video(silent_source)
    audio._run_command(
        [
            audio._require_binary("ffmpeg"),
            "-y",
            "-v",
            "error",
            "-i",
            str(silent_source),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=0.5",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(source_with_audio),
        ]
    )

    render_video(
        source_with_audio,
        [
            Image.new("RGB", (20, 20), "black"),
            Image.new("RGB", (20, 20), "white"),
        ],
        tile_count=2,
        output_path=output_path,
    )

    assert has_audio_stream(output_path) is True
    assert audio_codec_name(output_path) == "aac"
    assert video_codec_name(output_path) == "h264"


def test_render_video_passes_match_mode_into_render_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.avi"
    output_path = tmp_path / "mosaic.mp4"
    create_test_video(input_path, frame_count=1)
    observed_modes: list[MatchMode] = []
    real_plan = MosaicRenderPlan

    def recording_plan(
        width: int,
        height: int,
        album_tiles: tuple,
        tile_count: int,
        match_mode: MatchMode,
    ) -> MosaicRenderPlan:
        observed_modes.append(match_mode)
        return real_plan(width, height, album_tiles, tile_count, match_mode)

    monkeypatch.setattr(video_renderer_module, "MosaicRenderPlan", recording_plan)

    render_video(
        input_path,
        [
            Image.new("RGB", (20, 20), "black"),
            Image.new("RGB", (20, 20), "white"),
        ],
        tile_count=2,
        output_path=output_path,
        match_mode=MatchMode.UNIQUE_PER_FRAME,
    )

    assert observed_modes == [MatchMode.UNIQUE_PER_FRAME]


def test_reader_rejects_corrupt_video(tmp_path: Path) -> None:
    corrupt_path = tmp_path / "corrupt.mp4"
    corrupt_path.write_bytes(b"not a video")

    with pytest.raises(CorruptVideoError, match=r"corrupt|codec"):
        VideoReader(corrupt_path)


def test_render_video_rejects_unsupported_output_extension(tmp_path: Path) -> None:
    input_path = tmp_path / "input.avi"
    create_test_video(input_path)

    with pytest.raises(UnsupportedCodecError, match="extension"):
        render_video(
            input_path,
            [Image.new("RGB", (10, 10), "black")],
            2,
            tmp_path / "output.unsupported",
        )


def test_writer_rejects_wrong_frame_dimensions(tmp_path: Path) -> None:
    with (
        SilentVideoWriter(tmp_path / "output.avi", 8.0, (32, 18)) as writer,
        pytest.raises(VideoWriteError, match="shape"),
    ):
        writer.write(np.zeros((10, 10, 3), dtype=np.uint8))


def test_render_video_does_not_overwrite_input(tmp_path: Path) -> None:
    input_path = tmp_path / "input.avi"
    create_test_video(input_path)

    with pytest.raises(ValueError, match="different"):
        render_video(
            input_path,
            [Image.new("RGB", (10, 10), "black")],
            2,
            input_path,
        )
