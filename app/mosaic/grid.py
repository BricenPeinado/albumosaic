"""Aspect-aware grid calculation for a requested visible tile count."""

from dataclasses import dataclass
from math import ceil, floor, log, sqrt

_COUNT_ERROR_WEIGHT = 2.0
MAX_RENDER_DENSITY = 3000
_MIN_CELL_SIDE = 8


@dataclass(frozen=True, slots=True)
class GridSpec:
    """A validated rectangular grid and its actual visible tile count."""

    rows: int
    columns: int
    tile_count: int

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.columns <= 0:
            raise ValueError("Grid rows and columns must be positive")
        if self.tile_count != self.rows * self.columns:
            raise ValueError("Grid tile count must equal rows multiplied by columns")


def calculate_grid(width: int, height: int, target_tile_count: int) -> GridSpec:
    """Return a grid balancing proximity to N with the video's aspect ratio.

    The target is approximate because many values of N, especially primes,
    cannot form a useful rectangular grid. Relative count error receives twice
    the weight of log aspect-ratio error. This avoids extreme exact-factor grids
    while keeping the visible tile count close to the slider value.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Video dimensions must be positive")
    if target_tile_count < 2:
        raise ValueError("Target tile count must be at least 2")

    aspect_ratio = width / height
    ideal_columns = sqrt(target_tile_count * aspect_ratio)
    ideal_rows = sqrt(target_tile_count / aspect_ratio)
    candidates: set[tuple[int, int]] = set()

    def nearby(value: float) -> set[int]:
        center = round(value)
        return {
            candidate
            for candidate in (floor(value), ceil(value), center - 1, center, center + 1)
            if candidate >= 1
        }

    def add_candidate(rows: int, columns: int) -> None:
        if rows <= height and columns <= width:
            candidates.add((rows, columns))

    max_rows = min(height, ceil(ideal_rows * 2) + 1)
    for rows in range(1, max_rows + 1):
        column_options = nearby(target_tile_count / rows) | nearby(rows * aspect_ratio)
        for columns in column_options:
            add_candidate(rows, columns)

    max_columns = min(width, ceil(ideal_columns * 2) + 1)
    for columns in range(1, max_columns + 1):
        row_options = nearby(target_tile_count / columns) | nearby(
            columns / aspect_ratio
        )
        for rows in row_options:
            add_candidate(rows, columns)

    if not candidates:
        candidates.add((1, 1))

    def score(dimensions: tuple[int, int]) -> tuple[float, float, float, int, int, int]:
        rows, columns = dimensions
        actual_tile_count = rows * columns
        count_error = abs(actual_tile_count - target_tile_count) / target_tile_count
        aspect_error = abs(log((columns / rows) / aspect_ratio))
        return (
            (_COUNT_ERROR_WEIGHT * count_error) + aspect_error,
            count_error,
            aspect_error,
            0 if actual_tile_count >= target_tile_count else 1,
            rows,
            columns,
        )

    rows, columns = min(candidates, key=score)
    return GridSpec(
        rows=rows,
        columns=columns,
        tile_count=rows * columns,
    )


def calculate_render_grid(width: int, height: int, target_tile_count: int) -> GridSpec:
    """Apply practical rendering limits without changing aspect-aware grid search."""
    if target_tile_count > MAX_RENDER_DENSITY:
        raise ValueError(f"Mosaic density cannot exceed {MAX_RENDER_DENSITY} tiles")
    adaptive_limit = maximum_render_density(width, height)
    if target_tile_count > adaptive_limit:
        raise ValueError(
            f"Mosaic density cannot exceed {adaptive_limit} tiles "
            f"at {width} × {height} pixels"
        )
    grid = calculate_grid(width, height, target_tile_count)
    # Keep legacy two/four-cell rendering usable on tiny images. Higher densities
    # must not turn each cover into an effectively sub-thumbnail patch.
    if target_tile_count > 4 and (
        grid.columns > width // _MIN_CELL_SIDE or grid.rows > height // _MIN_CELL_SIDE
    ):
        raise ValueError(
            f"Mosaic density is too high for {width} × {height} pixels; "
            f"cells must be at least {_MIN_CELL_SIDE} pixels wide and tall"
        )
    return grid


def maximum_render_density(width: int, height: int) -> int:
    """Return a resolution-based requested-cell ceiling under the backend cap."""
    if width <= 0 or height <= 0:
        raise ValueError("Video dimensions must be positive")
    capacity = (width // _MIN_CELL_SIDE) * (height // _MIN_CELL_SIDE)
    # Four low-density cells remain available for legacy tiny-image rendering.
    return min(MAX_RENDER_DENSITY, max(4, capacity))


def choose_grid(
    frame_size: tuple[int, int],
    target_tiles: int,
) -> GridSpec:
    """Compatibility wrapper accepting a Pillow-style width/height size."""
    width, height = frame_size
    return calculate_grid(width, height, target_tiles)
