"""Streaming OpenCV encoding for intermediate silent videos."""

from __future__ import annotations

from collections.abc import Iterable
from math import isfinite
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np
from numpy.typing import NDArray

from app.video.errors import UnsupportedCodecError, VideoWriteError

VideoFrame = NDArray[np.uint8]

_OUTPUT_CODECS = {
    ".avi": "MJPG",
    ".m4v": "mp4v",
    ".mov": "mp4v",
    ".mp4": "mp4v",
}


def codec_for_output(output_path: str | Path) -> str:
    """Return the OpenCV codec associated with a supported file extension."""
    path = Path(output_path)
    try:
        return _OUTPUT_CODECS[path.suffix.lower()]
    except KeyError as error:
        supported = ", ".join(sorted(_OUTPUT_CODECS))
        raise UnsupportedCodecError(
            f"Unsupported output extension {path.suffix or '<none>'}; "
            f"expected one of: {supported}"
        ) from error


class SilentVideoWriter:
    """Write BGR frames incrementally while preserving size and frame rate."""

    def __init__(
        self,
        output_path: str | Path,
        frames_per_second: float,
        frame_size: tuple[int, int],
    ) -> None:
        self.path = Path(output_path)
        self.frames_per_second = frames_per_second
        self.frame_size = frame_size
        self.frames_written = 0
        self._closed = False

        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError("Output frame dimensions must be positive")
        if not isfinite(frames_per_second) or frames_per_second <= 0:
            raise ValueError("Output FPS must be positive and finite")

        codec = codec_for_output(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)  # type: ignore[attr-defined]
        self._writer = cv2.VideoWriter(
            str(self.path),
            fourcc,
            frames_per_second,
            frame_size,
        )
        if not self._writer.isOpened():
            self._writer.release()
            self._closed = True
            raise UnsupportedCodecError(
                f"OpenCV could not initialize codec {codec} for {self.path}"
            )

    def write(self, frame: VideoFrame) -> None:
        """Write one validated uint8 BGR frame."""
        width, height = self.frame_size
        if (
            frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape != (height, width, 3)
        ):
            raise VideoWriteError(
                f"Expected a uint8 BGR frame with shape {(height, width, 3)}, "
                f"received shape {frame.shape} and dtype {frame.dtype}"
            )
        if self._closed:
            raise VideoWriteError("Cannot write to a closed video writer")

        self._writer.write(frame)
        self.frames_written += 1

    def close(self) -> None:
        """Flush and release the underlying OpenCV writer."""
        if not self._closed:
            self._writer.release()
            self._closed = True

    def __enter__(self) -> SilentVideoWriter:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        self.close()


def write_silent_video(
    frames: Iterable[VideoFrame],
    output_path: str | Path,
    frames_per_second: float,
    frame_size: tuple[int, int],
) -> Path:
    """Stream BGR frames into a silent video and return its path."""
    path = Path(output_path)
    with SilentVideoWriter(path, frames_per_second, frame_size) as writer:
        for frame in frames:
            writer.write(frame)
        if writer.frames_written == 0:
            raise VideoWriteError("Cannot create a video without frames")
    return path
