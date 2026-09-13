"""Source-video metadata and frame decoding."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

VideoFrame = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Properties required to preserve a source video's timing and size."""

    width: int
    height: int
    frames_per_second: float
    frame_count: int


def read_frames(video_path: Path) -> tuple[VideoMetadata, Iterator[VideoFrame]]:
    """Open a video and return its metadata and decoded frame iterator."""
    raise NotImplementedError("Video reading is not implemented yet")
