"""Tests for aspect-aware mosaic grid calculation."""

import pytest

from app.mosaic.grid import (
    GridSpec,
    calculate_grid,
    calculate_render_grid,
    choose_grid,
    maximum_render_density,
)


def test_16_by_9_grid_favors_columns() -> None:
    grid = calculate_grid(width=1920, height=1080, target_tile_count=100)

    assert grid == GridSpec(rows=7, columns=14, tile_count=98)
    assert grid.columns > grid.rows


def test_9_by_16_grid_favors_rows() -> None:
    grid = calculate_grid(width=1080, height=1920, target_tile_count=100)

    assert grid == GridSpec(rows=14, columns=7, tile_count=98)
    assert grid.rows > grid.columns


def test_square_video_uses_square_grid_when_available() -> None:
    grid = calculate_grid(width=1000, height=1000, target_tile_count=100)

    assert grid == GridSpec(rows=10, columns=10, tile_count=100)


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (17, GridSpec(rows=3, columns=6, tile_count=18)),
        (97, GridSpec(rows=7, columns=14, tile_count=98)),
        (997, GridSpec(rows=24, columns=42, tile_count=1008)),
    ],
)
def test_prime_targets_avoid_extreme_exact_factor_grids(
    target: int,
    expected: GridSpec,
) -> None:
    grid = calculate_grid(width=1920, height=1080, target_tile_count=target)

    assert grid == expected
    assert grid.rows > 1
    assert grid.columns > grid.rows


def test_two_tiles_follow_landscape_orientation() -> None:
    assert calculate_grid(1920, 1080, 2) == GridSpec(
        rows=1,
        columns=2,
        tile_count=2,
    )


def test_large_target_remains_close_in_count_and_aspect() -> None:
    grid = calculate_grid(width=3840, height=2160, target_tile_count=10_000)

    assert grid == GridSpec(rows=75, columns=133, tile_count=9_975)
    assert abs(grid.tile_count - 10_000) / 10_000 < 0.01
    assert abs((grid.columns / grid.rows) - (16 / 9)) < 0.01


@pytest.mark.parametrize("dimensions", [(0, 100), (100, 0), (-1, 100)])
def test_grid_rejects_invalid_video_dimensions(
    dimensions: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="dimensions"):
        calculate_grid(*dimensions, target_tile_count=10)


@pytest.mark.parametrize("target", [0, 1, -1])
def test_grid_rejects_tile_counts_below_slider_minimum(target: int) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        calculate_grid(100, 100, target)


def test_grid_never_creates_subpixel_tiles() -> None:
    assert calculate_grid(1, 1, 100) == GridSpec(
        rows=1,
        columns=1,
        tile_count=1,
    )


def test_compatibility_wrapper_uses_width_height_order() -> None:
    assert choose_grid((1920, 1080), 2) == GridSpec(
        rows=1,
        columns=2,
        tile_count=2,
    )


def test_render_grid_density_uses_dimensions_not_album_count() -> None:
    grid = calculate_render_grid(320, 180, 200)

    assert grid == GridSpec(rows=11, columns=18, tile_count=198)


def test_render_grid_rejects_subthumbnail_cells_on_tiny_video() -> None:
    with pytest.raises(ValueError, match="cannot exceed 8 tiles"):
        calculate_render_grid(32, 18, 200)


def test_render_grid_accepts_density_above_500() -> None:
    grid = calculate_render_grid(640, 360, 1500)

    assert grid == GridSpec(rows=29, columns=52, tile_count=1508)


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [(1920, 1080, 3000), (640, 360, 3000), (320, 180, 880), (32, 18, 8)],
)
def test_adaptive_density_maximum_uses_video_resolution(
    width: int, height: int, expected: int
) -> None:
    assert maximum_render_density(width, height) == expected


def test_render_grid_limits_requested_density_to_backend_cap() -> None:
    with pytest.raises(ValueError, match="cannot exceed 3000"):
        calculate_render_grid(1920, 1080, 3001)


def test_render_grid_enforces_resolution_specific_cap() -> None:
    with pytest.raises(ValueError, match="cannot exceed 880"):
        calculate_render_grid(320, 180, 881)


def test_grid_spec_rejects_inconsistent_tile_count() -> None:
    with pytest.raises(ValueError, match="multiplied"):
        GridSpec(rows=2, columns=3, tile_count=5)
