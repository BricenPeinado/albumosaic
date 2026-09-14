"""Streaming OpenCV video metadata and frame decoding."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from types import TracebackType
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from app.video.errors import CorruptVideoError

VideoFrame = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Source properties required to preserve video geometry and timing."""

    width: int
    height: int
    frames_per_second: float
    frame_count: int
    duration: float


class VideoReader(Iterator[VideoFrame]):
    """Iterate over decoded BGR frames without buffering the whole video."""

    def __init__(self, video_path: str | Path) -> None:
        self.path = Path(video_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {self.path}")

        self._capture = cv2.VideoCapture(str(self.path))
        self._closed = False
        self._processed_frames = 0
        if not self._capture.isOpened():
            self._capture.release()
            self._closed = True
            raise CorruptVideoError(
                f"Could not open {self.path}; it may be corrupt or use an "
                "unsupported input codec"
            )

        try:
            self.metadata = _read_metadata(self._capture, self.path)
        except Exception:
            self.close()
            raise

    def __iter__(self) -> VideoReader:
        return self

    def __next__(self) -> VideoFrame:
        if self._closed:
            raise StopIteration

        decoded, frame = self._capture.read()
        if decoded:
            expected_shape = (
                self.metadata.height,
                self.metadata.width,
                3,
            )
            if (
                frame is None
                or frame.dtype != np.uint8
                or frame.shape != expected_shape
            ):
                self.close()
                raise CorruptVideoError(
                    f"Decoded an invalid frame from {self.path}; expected "
                    f"uint8 BGR data with shape {expected_shape}"
                )
            self._processed_frames += 1
            return cast(VideoFrame, frame)

        self.close()
        if self._processed_frames < self.metadata.frame_count:
            raise CorruptVideoError(
                f"Video ended after {self._processed_frames} of "
                f"{self.metadata.frame_count} reported frames: {self.path}"
            )
        raise StopIteration

    def close(self) -> None:
        """Release the underlying OpenCV capture."""
        if not self._closed:
            self._capture.release()
            self._closed = True

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        self.close()


def _read_metadata(capture: cv2.VideoCapture, path: Path) -> VideoMetadata:
    width_value = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height_value = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count_value = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    values = (width_value, height_value, fps, frame_count_value)
    if not all(isfinite(value) and value > 0 for value in values):
        raise CorruptVideoError(
            f"Video has invalid metadata or an unsupported codec: {path}"
        )

    width = round(width_value)
    height = round(height_value)
    frame_count = round(frame_count_value)
    return VideoMetadata(
        width=width,
        height=height,
        frames_per_second=float(fps),
        frame_count=frame_count,
        duration=frame_count / fps,
    )


def read_frames(
    video_path: str | Path,
) -> tuple[VideoMetadata, Iterator[VideoFrame]]:
    """Return validated metadata and a lazy, self-closing BGR frame iterator."""
    reader = VideoReader(video_path)

    def frame_iterator() -> Iterator[VideoFrame]:
        with reader:
            yield from reader

    return reader.metadata, frame_iterator()
