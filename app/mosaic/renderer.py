"""Mosaic frame composition."""

from collections.abc import Sequence
from math import isfinite

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app.mosaic.grid import GridSpec, calculate_grid
from app.mosaic.matcher import (
    AlbumLabIndex,
    AlbumTile,
    AlbumTileCache,
    ColorArray,
    MatchMode,
    RgbArray,
    calculate_cell_lab_means_array,
    match_artwork,
)

_DEFAULT_TILE_CACHE = AlbumTileCache()


class MosaicRenderPlan:
    """Frame-invariant grid, album features, and resized cover caches."""

    def __init__(
        self,
        width: int,
        height: int,
        album_tiles: Sequence[AlbumTile],
        target_tile_count: int,
        match_mode: MatchMode = MatchMode.NEAREST,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Target dimensions must be positive")
        if not album_tiles:
            raise ValueError("At least one album tile is required")
        self.width = width
        self.height = height
        self.album_tiles = tuple(album_tiles)
        if not isinstance(match_mode, MatchMode):
            raise TypeError("match_mode must be a MatchMode")
        self.match_mode = match_mode
        self.grid: GridSpec = calculate_grid(width, height, target_tile_count)
        self.album_index = AlbumLabIndex.from_tiles(self.album_tiles)
        self.x_edges = np.linspace(0, width, self.grid.columns + 1, dtype=np.intp)
        self.y_edges = np.linspace(0, height, self.grid.rows + 1, dtype=np.intp)
        self._bgr_tile_cache: dict[tuple[int, int, int], RgbArray] = {}

    def target_cell_means(self, bgr_frame: RgbArray) -> ColorArray:
        """Batch target-cell LAB means without copying BGR into a PIL image."""
        self._validate_frame(bgr_frame)
        return calculate_cell_lab_means_array(bgr_frame[..., ::-1], self.grid)

    def match(self, target_means: ColorArray) -> NDArray[np.intp]:
        """Match all target cells through the precomputed album LAB index."""
        return self.album_index.match(target_means, self.match_mode)

    def compose_bgr(self, matches: NDArray[np.intp]) -> RgbArray:
        """Compose a BGR output frame using persistent resized-tile caches."""
        if matches.shape != (self.grid.tile_count,):
            raise ValueError("Match count must equal the mosaic grid cell count")
        output = np.empty((self.height, self.width, 3), dtype=np.uint8)
        for cell_index, artwork_index_value in enumerate(matches):
            row, column = divmod(cell_index, self.grid.columns)
            left, right = self.x_edges[column : column + 2]
            top, bottom = self.y_edges[row : row + 2]
            artwork_index = int(artwork_index_value)
            tile = self._resized_bgr_tile(
                artwork_index,
                int(right - left),
                int(bottom - top),
            )
            output[int(top) : int(bottom), int(left) : int(right)] = tile
        return output

    def blend_bgr(
        self,
        mosaic_frame: RgbArray,
        original_frame: RgbArray,
        blend_alpha: float = 0.0,
    ) -> RgbArray:
        """Blend two same-sized BGR frames after mosaic composition."""
        self._validate_frame(mosaic_frame)
        self._validate_frame(original_frame)
        return blend_uint8_frames(mosaic_frame, original_frame, blend_alpha)

    def render_bgr(
        self,
        bgr_frame: RgbArray,
        blend_alpha: float = 0.0,
    ) -> RgbArray:
        """Render and optionally blend one OpenCV BGR frame."""
        validate_blend_alpha(blend_alpha)
        target_means = self.target_cell_means(bgr_frame)
        matches = self.match(target_means)
        mosaic_frame = self.compose_bgr(matches)
        return self.blend_bgr(mosaic_frame, bgr_frame, blend_alpha)

    def _resized_bgr_tile(self, index: int, width: int, height: int) -> RgbArray:
        key = (index, width, height)
        cached = self._bgr_tile_cache.get(key)
        if cached is None:
            rgb = np.asarray(
                self.album_tiles[index].resized((width, height)),
                dtype=np.uint8,
            )
            cached = np.ascontiguousarray(rgb[..., ::-1])
            self._bgr_tile_cache[key] = cached
        return cached

    def _validate_frame(self, frame: RgbArray) -> None:
        expected_shape = (self.height, self.width, 3)
        if frame.dtype != np.uint8 or frame.shape != expected_shape:
            raise ValueError(
                f"BGR frame must have shape {expected_shape} and uint8 data"
            )


def render_mosaic(
    target_image: Image.Image,
    album_images: Sequence[Image.Image | AlbumTile],
    target_tile_count: int,
    *,
    tile_cache: AlbumTileCache | None = None,
    blend_alpha: float = 0.0,
    match_mode: MatchMode = MatchMode.NEAREST,
) -> Image.Image:
    """Render a target-sized photomosaic from reusable album-cover images."""
    alpha = validate_blend_alpha(blend_alpha)
    if target_image.width <= 0 or target_image.height <= 0:
        raise ValueError("Target image must have positive dimensions")
    if not album_images:
        raise ValueError("At least one album image is required")

    grid = calculate_grid(
        width=target_image.width,
        height=target_image.height,
        target_tile_count=target_tile_count,
    )
    cache = tile_cache if tile_cache is not None else _DEFAULT_TILE_CACHE
    album_tiles = tuple(
        item
        if isinstance(item, AlbumTile)
        else cache.get(identifier=f"album-{index}", image=item)
        for index, item in enumerate(album_images)
    )
    matches = match_artwork(target_image, album_tiles, grid, match_mode)

    output = Image.new("RGB", target_image.size)
    x_edges = np.linspace(0, target_image.width, grid.columns + 1, dtype=int)
    y_edges = np.linspace(0, target_image.height, grid.rows + 1, dtype=int)

    for cell_index, artwork_index in enumerate(matches):
        row, column = divmod(cell_index, grid.columns)
        left, right = int(x_edges[column]), int(x_edges[column + 1])
        top, bottom = int(y_edges[row]), int(y_edges[row + 1])
        tile = album_tiles[artwork_index].resized((right - left, bottom - top))
        output.paste(tile, (left, top))

    if alpha == 0.0:
        return output
    mosaic_rgb = np.asarray(output, dtype=np.uint8)
    original_rgb = np.asarray(target_image.convert("RGB"), dtype=np.uint8)
    blended_rgb = blend_uint8_frames(mosaic_rgb, original_rgb, alpha)
    return Image.fromarray(blended_rgb)


def validate_blend_alpha(blend_alpha: float) -> float:
    """Return a finite blend alpha in the inclusive range from zero to one."""
    try:
        value = float(blend_alpha)
    except (TypeError, ValueError) as error:
        raise ValueError("blend_alpha must be between 0.0 and 1.0") from error
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("blend_alpha must be between 0.0 and 1.0")
    return value


def blend_uint8_frames(
    mosaic_frame: RgbArray,
    original_frame: RgbArray,
    blend_alpha: float = 0.0,
) -> RgbArray:
    """Blend same-format RGB or BGR uint8 frames without changing geometry."""
    alpha = validate_blend_alpha(blend_alpha)
    if mosaic_frame.shape != original_frame.shape:
        raise ValueError("Mosaic and original frames must have identical dimensions")
    if (
        mosaic_frame.ndim != 3
        or mosaic_frame.shape[2] != 3
        or mosaic_frame.dtype != np.uint8
        or original_frame.dtype != np.uint8
    ):
        raise ValueError(
            "Mosaic and original frames must be uint8 three-channel images"
        )
    if alpha == 0.0:
        return mosaic_frame
    if alpha == 1.0:
        return original_frame

    blended = (
        mosaic_frame.astype(np.float32) * (1.0 - alpha)
        + original_frame.astype(np.float32) * alpha
    )
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)
