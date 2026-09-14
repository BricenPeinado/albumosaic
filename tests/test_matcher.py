"""Tests for perceptual color matching and album-tile caching."""

import numpy as np
import pytest
from PIL import Image

from app.mosaic.grid import GridSpec
from app.mosaic.matcher import (
    AlbumLabIndex,
    AlbumTile,
    AlbumTileCache,
    MatchMode,
    calculate_cell_lab_means,
    calculate_cell_means,
    nearest_color_indices,
    rgb_to_lab,
)


def test_rgb_to_lab_matches_reference_primary_colors() -> None:
    colors = np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0]], dtype=np.uint8)

    converted = rgb_to_lab(colors)

    assert converted[0] == pytest.approx([100.0, 0.0, 0.0], abs=0.02)
    assert converted[1] == pytest.approx([0.0, 0.0, 0.0], abs=0.02)
    assert converted[2] == pytest.approx([53.24, 80.09, 67.20], abs=0.05)


def test_album_tile_contains_features_and_reuses_resized_image() -> None:
    tile = AlbumTile("red-album", Image.new("RGB", (20, 10), "red"))

    first = tile.resized((8, 8))
    second = tile.resized((8, 8))

    assert tile.identifier == "red-album"
    assert tile.image.size == (10, 10)
    assert tile.rgb_mean == pytest.approx([255.0, 0.0, 0.0])
    assert tile.lab_mean == pytest.approx([53.24, 80.09, 67.20], abs=0.05)
    assert first is second
    assert tile.resized_tile_cache[(8, 8)] is first


def test_album_tile_cache_reuses_calculated_features() -> None:
    image = Image.new("RGB", (20, 10), "red")
    cache = AlbumTileCache()

    first = cache.get("album-id", image)
    second = cache.get("album-id", image)

    assert first is second
    assert len(cache) == 1


def test_cell_lab_mean_is_mean_of_lab_pixels() -> None:
    image = Image.new("RGB", (2, 1), "black")
    image.putpixel((1, 0), (255, 255, 255))

    grid = GridSpec(rows=1, columns=1, tile_count=1)
    mean = calculate_cell_lab_means(image, grid)[0]

    assert mean == pytest.approx([50.0, 0.0, 0.0], abs=0.02)


def test_nearest_color_indices_matches_all_rows() -> None:
    targets = np.array([[1.0, 1.0, 1.0], [9.0, 9.0, 9.0]], dtype=np.float32)
    albums = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]], dtype=np.float32)

    matches = nearest_color_indices(targets, albums)

    assert matches.tolist() == [0, 1]


def test_batched_cell_reduction_matches_integral_image_reference() -> None:
    rng = np.random.default_rng(1234)
    values = rng.normal(size=(19, 31, 3)).astype(np.float32)
    grid = GridSpec(rows=4, columns=7, tile_count=28)
    integral = np.pad(
        values.cumsum(axis=0, dtype=np.float64).cumsum(axis=1, dtype=np.float64),
        ((1, 0), (1, 0), (0, 0)),
    )
    x_edges = np.linspace(0, 31, 8, dtype=np.intp)
    y_edges = np.linspace(0, 19, 5, dtype=np.intp)
    left, right = x_edges[:-1], x_edges[1:]
    top, bottom = y_edges[:-1], y_edges[1:]
    sums = (
        integral[bottom[:, None], right[None, :]]
        - integral[top[:, None], right[None, :]]
        - integral[bottom[:, None], left[None, :]]
        + integral[top[:, None], left[None, :]]
    )
    areas = (bottom - top)[:, None] * (right - left)[None, :]
    expected = (sums / areas[..., None]).reshape(-1, 3).astype(np.float32)

    assert np.array_equal(calculate_cell_means(values, grid), expected)


def test_album_lab_index_reuses_precomputed_vectors_and_norms() -> None:
    tiles = (
        AlbumTile("black", Image.new("RGB", (8, 8), "black")),
        AlbumTile("white", Image.new("RGB", (8, 8), "white")),
    )
    index = AlbumLabIndex.from_tiles(tiles)

    assert index.vectors.shape == (2, 3)
    assert index.squared_norms.shape == (2,)
    assert index.nearest(np.array([[99.0, 0.0, 0.0]], dtype=np.float32)).tolist() == [1]


def test_nearest_mode_remains_backward_compatible_and_allows_duplicates() -> None:
    index = AlbumLabIndex(
        vectors=np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float32),
        squared_norms=np.array([0.0, 10_000.0], dtype=np.float32),
    )
    targets = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)

    assert index.match(targets).tolist() == [0, 0]
    assert index.match(targets, MatchMode.NEAREST).tolist() == [0, 0]


def test_unique_mode_assigns_each_cell_a_different_album() -> None:
    index = AlbumLabIndex(
        vectors=np.array(
            [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        squared_norms=np.array([0.0, 2_500.0, 10_000.0], dtype=np.float32),
    )
    targets = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)

    matches = index.match(targets, MatchMode.UNIQUE_PER_FRAME)

    assert len(matches) == 2
    assert len(set(matches.tolist())) == 2


def test_unique_mode_uses_global_assignment_instead_of_row_greedy() -> None:
    index = AlbumLabIndex(
        vectors=np.array([[0.0, 0.0, 0.0], [-10.0, 0.0, 0.0]], dtype=np.float32),
        squared_norms=np.array([0.0, 100.0], dtype=np.float32),
    )
    targets = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

    matches = index.unique(targets)

    assert matches.tolist() == [1, 0]
    global_cost = 100.0 + 1.0
    row_greedy_cost = 0.0 + 121.0
    assert global_cost < row_greedy_cost


def test_unique_mode_uses_all_albums_before_overflow_repeats() -> None:
    vectors = np.array(
        [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    index = AlbumLabIndex(
        vectors=vectors,
        squared_norms=np.sum(vectors * vectors, axis=1, dtype=np.float32),
    )
    targets = np.array(
        [
            [0.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [80.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    matches = index.unique(targets)

    assert len(matches) == 5
    assert set(matches.tolist()) == {0, 1, 2}
    assert len(matches) - len(set(matches.tolist())) == 2
    assert np.array_equal(matches, index.unique(targets))


def test_unique_mode_supports_a_single_album() -> None:
    index = AlbumLabIndex(
        vectors=np.array([[10.0, 0.0, 0.0]], dtype=np.float32),
        squared_norms=np.array([100.0], dtype=np.float32),
    )
    targets = np.zeros((3, 3), dtype=np.float32)

    assert index.unique(targets).tolist() == [0, 0, 0]


def test_album_index_rejects_an_untyped_match_mode() -> None:
    index = AlbumLabIndex(
        vectors=np.array([[10.0, 0.0, 0.0]], dtype=np.float32),
        squared_norms=np.array([100.0], dtype=np.float32),
    )

    with pytest.raises(TypeError, match="MatchMode"):
        index.match(np.zeros((1, 3), dtype=np.float32), "unique")  # type: ignore[arg-type]


def test_rgb_to_lab_rejects_non_rgb_input() -> None:
    with pytest.raises(ValueError, match="three channels"):
        rgb_to_lab(np.zeros((2, 2), dtype=np.uint8))
