"""Tests for complete still-image mosaic rendering."""

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from app import network
from app.mosaic.matcher import AlbumTile, AlbumTileCache, MatchMode
from app.mosaic.renderer import MosaicRenderPlan, blend_uint8_frames, render_mosaic


def test_render_mosaic_matches_solid_regions_and_preserves_size() -> None:
    target = Image.new("RGB", (40, 20), "red")
    ImageDraw.Draw(target).rectangle((20, 0, 39, 19), fill="blue")
    album_images = [
        Image.new("RGB", (12, 20), "red"),
        Image.new("RGB", (20, 12), "blue"),
    ]

    result = render_mosaic(target, album_images, target_tile_count=2)

    assert result.mode == "RGB"
    assert result.size == target.size
    assert result.getpixel((5, 10)) == (255, 0, 0)
    assert result.getpixel((35, 10)) == (0, 0, 255)


def test_render_mosaic_reuses_explicit_tile_cache() -> None:
    target = Image.new("RGB", (20, 20), "white")
    albums = [Image.new("RGB", (10, 10), "white")]
    cache = AlbumTileCache()

    render_mosaic(target, albums, 4, tile_cache=cache)
    first_tile = cache.get("album-0", albums[0])
    render_mosaic(target, albums, 4, tile_cache=cache)

    assert cache.get("album-0", albums[0]) is first_tile
    assert len(cache) == 1


@pytest.mark.parametrize(("album_count", "requested"), [(8, 200), (3, 100)])
def test_many_cells_match_a_small_album_library_without_tile_duplication(
    album_count: int, requested: int
) -> None:
    images = [
        Image.new("RGB", (20, 20), (index * 25, index * 25, index * 25))
        for index in range(album_count)
    ]
    cache = AlbumTileCache()
    tiles = cache.get_many(images)
    plan = MosaicRenderPlan(320, 180, tiles, requested, MatchMode.NEAREST)
    target_means = np.zeros((plan.grid.tile_count, 3), dtype=np.float32)

    matches = plan.match(target_means)
    rendered = render_mosaic(
        Image.new("RGB", (320, 180), "black"),
        images,
        requested,
        tile_cache=cache,
    )

    assert len(matches) == plan.grid.tile_count
    assert set(matches.tolist()) <= set(range(album_count))
    assert len(set(matches.tolist())) < len(matches)
    assert len(plan.album_tiles) == album_count
    assert all(plan.album_tiles[index] is tiles[index] for index in range(album_count))
    assert plan.album_index.vectors.shape == (album_count, 3)
    assert len(cache) == album_count
    assert rendered.size == (320, 180)


def test_high_density_render_reuses_three_album_tiles() -> None:
    tiles = tuple(
        AlbumTile(str(index), Image.new("RGB", (24, 24), (index * 80, 0, 0)))
        for index in range(3)
    )
    plan = MosaicRenderPlan(640, 360, tiles, 1500, MatchMode.NEAREST)
    target_means = np.zeros((plan.grid.tile_count, 3), dtype=np.float32)
    matches = plan.match(target_means)

    result = plan.compose_bgr(matches)

    assert plan.grid.tile_count > 500
    assert len(matches) == plan.grid.tile_count
    assert set(matches.tolist()) <= {0, 1, 2}
    assert len(plan.album_tiles) == 3
    assert result.shape == (360, 640, 3)
    assert result.dtype == np.uint8


def test_unique_mode_fills_overflow_cells_without_reducing_density() -> None:
    tiles = tuple(
        AlbumTile(str(index), Image.new("RGB", (20, 20), (index * 30, 0, 0)))
        for index in range(8)
    )
    plan = MosaicRenderPlan(320, 180, tiles, 200, MatchMode.UNIQUE_PER_FRAME)
    target_means = np.zeros((plan.grid.tile_count, 3), dtype=np.float32)

    matches = plan.match(target_means)

    assert len(matches) == plan.grid.tile_count
    assert set(matches.tolist()) == set(range(8))
    assert len(set(matches.tolist())) < len(matches)
    assert plan.grid.tile_count > len(tiles)


def test_repeated_frame_rendering_uses_cached_features_and_no_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        network,
        "open_url",
        lambda request, timeout: pytest.fail("rendering attempted network I/O"),
    )
    tiles = tuple(
        AlbumTile(str(index), Image.new("RGB", (20, 20), (index * 30, 0, 0)))
        for index in range(8)
    )
    plan = MosaicRenderPlan(320, 180, tiles, 200)
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    first = plan.render_bgr(frame)
    cached_variants = tuple(len(tile.resized_tile_cache) for tile in tiles)
    first_bgr_variants = len(plan._bgr_tile_cache)

    assert np.array_equal(plan.render_bgr(frame), first)
    assert tuple(len(tile.resized_tile_cache) for tile in tiles) == cached_variants
    assert len(plan._bgr_tile_cache) == first_bgr_variants
    assert len(plan.album_tiles) == len(tiles)
    assert plan.album_index.vectors.shape == (len(tiles), 3)


def test_render_plan_keeps_match_mode_as_frame_invariant_configuration() -> None:
    tiles = tuple(
        AlbumTile(str(index), Image.new("RGB", (8, 8), value))
        for index, value in enumerate(("black", "gray", "white"))
    )
    plan = MosaicRenderPlan(
        20,
        10,
        tiles,
        2,
        MatchMode.UNIQUE_PER_FRAME,
    )
    target_means = np.zeros((plan.grid.tile_count, 3), dtype=np.float32)

    matches = plan.match(target_means)

    assert plan.match_mode is MatchMode.UNIQUE_PER_FRAME
    assert len(set(matches.tolist())) == plan.grid.tile_count


def test_still_renderer_supports_unique_mode_without_changing_blending() -> None:
    target = Image.new("RGB", (20, 20), (12, 34, 56))
    albums = [
        Image.new("RGB", (8, 8), color)
        for color in ((0, 0, 0), (64, 0, 0), (128, 0, 0), (255, 0, 0))
    ]

    mosaic = render_mosaic(
        target,
        albums,
        4,
        match_mode=MatchMode.UNIQUE_PER_FRAME,
    )
    original_only = render_mosaic(
        target,
        albums,
        4,
        match_mode=MatchMode.UNIQUE_PER_FRAME,
        blend_alpha=1.0,
    )

    sampled_colors = {
        mosaic.getpixel((5, 5)),
        mosaic.getpixel((15, 5)),
        mosaic.getpixel((5, 15)),
        mosaic.getpixel((15, 15)),
    }
    assert sampled_colors == {(0, 0, 0), (64, 0, 0), (128, 0, 0), (255, 0, 0)}
    assert original_only.tobytes() == target.tobytes()


def test_render_mosaic_requires_album_images() -> None:
    with pytest.raises(ValueError, match="album image"):
        render_mosaic(Image.new("RGB", (10, 10)), [], 2)


def test_video_render_plan_is_pixel_identical_to_pil_renderer() -> None:
    rng = np.random.default_rng(20260913)
    bgr_frame = rng.integers(0, 256, (37, 61, 3), dtype=np.uint8)
    tiles = tuple(
        AlbumTile(
            f"album-{index}",
            Image.fromarray(rng.integers(0, 256, (23, 23, 3), dtype=np.uint8)),
        )
        for index in range(9)
    )
    rgb_target = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
    expected_rgb = render_mosaic(rgb_target, tiles, 17)
    expected_bgr = cv2.cvtColor(np.asarray(expected_rgb), cv2.COLOR_RGB2BGR)

    plan = MosaicRenderPlan(61, 37, tiles, 17)
    actual_bgr = plan.render_bgr(bgr_frame)

    assert np.array_equal(actual_bgr, expected_bgr)
    first_cache_size = len(plan._bgr_tile_cache)
    assert np.array_equal(plan.render_bgr(bgr_frame), expected_bgr)
    assert len(plan._bgr_tile_cache) == first_cache_size


def test_blend_zero_returns_mosaic_unchanged() -> None:
    mosaic = np.full((3, 4, 3), 25, dtype=np.uint8)
    original = np.full((3, 4, 3), 225, dtype=np.uint8)

    result = blend_uint8_frames(mosaic, original, 0.0)

    assert np.array_equal(result, mosaic)


def test_blend_one_returns_original_frame() -> None:
    mosaic = np.full((3, 4, 3), 25, dtype=np.uint8)
    original = np.full((3, 4, 3), 225, dtype=np.uint8)

    result = blend_uint8_frames(mosaic, original, 1.0)

    assert np.array_equal(result, original)


def test_blend_half_mixes_frames_and_preserves_shape_and_dtype() -> None:
    mosaic = np.full((3, 4, 3), 10, dtype=np.uint8)
    original = np.full((3, 4, 3), 240, dtype=np.uint8)

    result = blend_uint8_frames(mosaic, original, 0.5)

    assert np.all(result == 125)
    assert result.shape == mosaic.shape
    assert result.dtype == np.uint8


@pytest.mark.parametrize("blend_alpha", [-0.01, 1.01])
def test_invalid_blend_alpha_raises_clear_error(blend_alpha: float) -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"between 0.0 and 1.0"):
        blend_uint8_frames(frame, frame, blend_alpha)


def test_render_mosaic_blends_only_after_album_composition() -> None:
    target = Image.new("RGB", (8, 8), (240, 240, 240))
    black_album = Image.new("RGB", (8, 8), (10, 10, 10))

    pure_mosaic = render_mosaic(target, [black_album], 2, blend_alpha=0.0)
    original_only = render_mosaic(target, [black_album], 2, blend_alpha=1.0)

    assert pure_mosaic.getpixel((4, 4)) == (10, 10, 10)
    assert original_only.getpixel((4, 4)) == (240, 240, 240)
