"""Color-based matching between frame regions and album covers."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from app.mosaic.grid import Grid

ImageArray = NDArray[np.uint8]


def match_artwork(
    frame: ImageArray,
    artwork: Sequence[ImageArray],
    grid: Grid,
) -> tuple[int, ...]:
    """Return the artwork index selected for each grid cell."""
    raise NotImplementedError("Artwork matching is not implemented yet")
