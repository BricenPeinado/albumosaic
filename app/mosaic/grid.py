"""Grid calculations for mosaic frames."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Grid:
    """A rectangular tile grid."""

    columns: int
    rows: int

    @property
    def tile_count(self) -> int:
        """Return the total number of cells in the grid."""
        return self.columns * self.rows


def choose_grid(frame_size: tuple[int, int], target_tiles: int) -> Grid:
    """Choose a grid close to the requested count for a frame size."""
    raise NotImplementedError("Mosaic grid selection is not implemented yet")
