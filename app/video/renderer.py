"""Frame-by-frame mosaic rendering and H.264 MP4 finalization."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from os import close as close_file_descriptor
from pathlib import Path
from tempfile import mkstemp
from time import perf_counter

from PIL import Image

from app.mosaic.matcher import AlbumTile, AlbumTileCache
from app.mosaic.renderer import MosaicRenderPlan, validate_blend_alpha
from app.video.audio import ensure_ffmpeg_available, finalize_h264_mp4
from app.video.errors import UnsupportedCodecError
from app.video.reader import VideoReader
from app.video.writer import SilentVideoWriter, codec_for_output

ProgressCallback = Callable[[int, int], None]
FinalizingCallback = Callable[[], None]


@dataclass(frozen=True, slots=True)
class VideoRenderTiming:
    """Timing totals for the streamed silent-video rendering stage."""

    frame_count: int
    decode_seconds: float
    target_color_seconds: float
    matching_seconds: float
    composition_seconds: float
    encoding_seconds: float

    @property
    def total_seconds(self) -> float:
        """Return the sum of the five measured frame-pipeline stages."""
        return (
            self.decode_seconds
            + self.target_color_seconds
            + self.matching_seconds
            + self.composition_seconds
            + self.encoding_seconds
        )

    def format(self) -> str:
        """Return compact human-readable timing output."""
        values = (
            ("frame decode", self.decode_seconds),
            ("target color calculation", self.target_color_seconds),
            ("matching", self.matching_seconds),
            ("composition", self.composition_seconds),
            ("frame encoding", self.encoding_seconds),
        )
        lines = [f"Video render timing ({self.frame_count} frames):"]
        for label, seconds in values:
            per_frame_ms = seconds * 1000 / self.frame_count if self.frame_count else 0
            lines.append(f"  {label}: {seconds:.3f}s ({per_frame_ms:.2f} ms/frame)")
        lines.append(f"  measured total: {self.total_seconds:.3f}s")
        return "\n".join(lines)


TimingCallback = Callable[[VideoRenderTiming], None]


@dataclass(slots=True)
class _TimingAccumulator:
    decode_seconds: float = 0.0
    target_color_seconds: float = 0.0
    matching_seconds: float = 0.0
    composition_seconds: float = 0.0
    encoding_seconds: float = 0.0

    def snapshot(self, frame_count: int) -> VideoRenderTiming:
        return VideoRenderTiming(
            frame_count=frame_count,
            decode_seconds=self.decode_seconds,
            target_color_seconds=self.target_color_seconds,
            matching_seconds=self.matching_seconds,
            composition_seconds=self.composition_seconds,
            encoding_seconds=self.encoding_seconds,
        )


def render_video(
    input_path: str | Path,
    album_tiles: Sequence[AlbumTile | Image.Image],
    tile_count: int,
    output_path: str | Path,
    progress_callback: ProgressCallback | None = None,
    finalizing_callback: FinalizingCallback | None = None,
    timing_callback: TimingCallback | None = None,
    blend_alpha: float = 0.0,
) -> Path:
    """Stream an input video into an audio-preserving H.264 mosaic MP4.

    The progress callback receives ``(processed_frames, total_frames)`` after
    every encoded frame, plus an initial ``(0, total_frames)`` update. The
    finalizing callback runs immediately before FFmpeg restores the audio. The
    timing callback receives per-stage totals after silent-frame encoding.
    ``blend_alpha`` controls the original-frame contribution after composition.
    """
    source_path = Path(input_path)
    destination_path = Path(output_path)
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("Input and output video paths must be different")
    if not album_tiles:
        raise ValueError("At least one album tile is required")
    alpha = validate_blend_alpha(blend_alpha)
    if destination_path.suffix.lower() != ".mp4":
        raise UnsupportedCodecError("Final video output must use the .mp4 extension")

    codec_for_output(destination_path)
    ensure_ffmpeg_available()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_tiles = _prepare_album_tiles(album_tiles)
    temporary_path = _temporary_output_path(destination_path)
    timings = _TimingAccumulator()

    try:
        with VideoReader(source_path) as reader:
            metadata = reader.metadata
            render_plan = MosaicRenderPlan(
                metadata.width,
                metadata.height,
                prepared_tiles,
                tile_count,
            )
            if progress_callback is not None:
                progress_callback(0, metadata.frame_count)

            with SilentVideoWriter(
                temporary_path,
                metadata.frames_per_second,
                (metadata.width, metadata.height),
            ) as writer:
                while True:
                    started = perf_counter()
                    try:
                        frame = next(reader)
                    except StopIteration:
                        break
                    timings.decode_seconds += perf_counter() - started

                    started = perf_counter()
                    target_means = render_plan.target_cell_means(frame)
                    timings.target_color_seconds += perf_counter() - started

                    started = perf_counter()
                    matches = render_plan.match(target_means)
                    timings.matching_seconds += perf_counter() - started

                    started = perf_counter()
                    mosaic_bgr = render_plan.compose_bgr(matches)
                    mosaic_bgr = render_plan.blend_bgr(
                        mosaic_bgr,
                        frame,
                        alpha,
                    )
                    timings.composition_seconds += perf_counter() - started

                    started = perf_counter()
                    writer.write(mosaic_bgr)
                    timings.encoding_seconds += perf_counter() - started
                    if progress_callback is not None:
                        progress_callback(writer.frames_written, metadata.frame_count)

                if timing_callback is not None:
                    timing_callback(timings.snapshot(writer.frames_written))

        if finalizing_callback is not None:
            finalizing_callback()
        return finalize_h264_mp4(
            silent_video_path=temporary_path,
            source_video_path=source_path,
            output_path=destination_path,
        )
    finally:
        temporary_path.unlink(missing_ok=True)


def _prepare_album_tiles(
    album_tiles: Sequence[AlbumTile | Image.Image],
) -> tuple[AlbumTile, ...]:
    """Build raw-image features once and retain them for the complete render."""
    cache = AlbumTileCache()
    return tuple(
        item
        if isinstance(item, AlbumTile)
        else cache.get(identifier=f"album-{index}", image=item)
        for index, item in enumerate(album_tiles)
    )


def _temporary_output_path(destination_path: Path) -> Path:
    """Reserve an intermediate path with the destination's codec extension."""
    file_descriptor, temporary_name = mkstemp(
        prefix=".albumosaic-",
        suffix=destination_path.suffix,
        dir=destination_path.parent,
    )
    close_file_descriptor(file_descriptor)
    return Path(temporary_name)
