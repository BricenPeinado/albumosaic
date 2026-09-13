"""Mosaic frame composition."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from app.mosaic.grid import Grid

ImageArray = NDArray[np.uint8]


def render_mosaic(
    frame_size: tuple[int, int],
    artwork: Sequence[ImageArray],
    matches: Sequence[int],
    grid: Grid,
) -> ImageArray:
    """Compose one mosaic frame from the selected artwork."""
    raise NotImplementedError("Mosaic rendering is not implemented yet")
