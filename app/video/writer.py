"""Intermediate video encoding."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

VideoFrame = NDArray[np.uint8]


def write_silent_video(
    frames: Iterable[VideoFrame],
    output_path: Path,
    frames_per_second: float,
    frame_size: tuple[int, int],
) -> Path:
    """Encode processed frames as an intermediate video without audio."""
    raise NotImplementedError("Video writing is not implemented yet")
