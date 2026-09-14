"""Cached album-tile features and vectorized perceptual color matching."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from threading import RLock
from weakref import ReferenceType, ref

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from app.mosaic.grid import GridSpec

RgbArray = NDArray[np.uint8]
ColorArray = NDArray[np.float32]

_FEATURE_SAMPLE_SIZE = 64
_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)
_D65_WHITE_POINT = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
_SRGB_VALUES = np.arange(256, dtype=np.float32) / 255.0
_SRGB_TO_LINEAR_LUT = np.where(
    _SRGB_VALUES <= 0.04045,
    _SRGB_VALUES / 12.92,
    ((_SRGB_VALUES + 0.055) / 1.055) ** 2.4,
).astype(np.float32)


@dataclass(slots=True)
class AlbumTile:
    """An album cover with reusable image and color features."""

    identifier: str
    image: Image.Image
    resized_tile_cache: dict[tuple[int, int], Image.Image] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    rgb_mean: ColorArray = field(init=False, repr=False)
    lab_mean: ColorArray = field(init=False, repr=False)
    _cache_lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.image.width <= 0 or self.image.height <= 0:
            raise ValueError("Album images must have positive dimensions")

        oriented = ImageOps.exif_transpose(self.image).convert("RGB")
        side_length = min(oriented.size)
        self.image = ImageOps.fit(
            oriented,
            (side_length, side_length),
            method=Image.Resampling.LANCZOS,
        )
        sample = self.image.resize(
            (_FEATURE_SAMPLE_SIZE, _FEATURE_SAMPLE_SIZE),
            resample=Image.Resampling.BOX,
        )
        pixels = np.asarray(sample, dtype=np.uint8)
        self.rgb_mean = pixels.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        self.lab_mean = (
            rgb_to_lab(pixels).mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        )

    def resized(self, size: tuple[int, int]) -> Image.Image:
        """Return a cached cover resized to an output cell's dimensions."""
        width, height = size
        if width <= 0 or height <= 0:
            raise ValueError("Tile dimensions must be positive")

        with self._cache_lock:
            cached = self.resized_tile_cache.get(size)
            if cached is None:
                cached = self.image.resize(size, resample=Image.Resampling.LANCZOS)
                self.resized_tile_cache[size] = cached
            return cached


@dataclass(slots=True)
class _CacheEntry:
    image_reference: ReferenceType[Image.Image]
    tile: AlbumTile


class AlbumTileCache:
    """Cache AlbumTile instances by source-image identity across frames."""

    def __init__(self) -> None:
        self._entries: dict[int, _CacheEntry] = {}
        self._lock = RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def get(self, identifier: str, image: Image.Image) -> AlbumTile:
        """Return a cached tile, calculating its features only once."""
        image_key = id(image)
        with self._lock:
            entry = self._entries.get(image_key)
            if entry is not None and entry.image_reference() is image:
                return entry.tile

            tile = AlbumTile(identifier=identifier, image=image)

            def discard(expired: ReferenceType[Image.Image]) -> None:
                self._discard(image_key, expired)

            image_reference = ref(image, discard)
            self._entries[image_key] = _CacheEntry(image_reference, tile)
            return tile

    def get_many(
        self,
        images: Sequence[Image.Image],
    ) -> tuple[AlbumTile, ...]:
        """Return cached tiles in the same order as the supplied images."""
        return tuple(
            self.get(identifier=f"album-{index}", image=image)
            for index, image in enumerate(images)
        )

    def clear(self) -> None:
        """Remove all cached tiles and their resized-image caches."""
        with self._lock:
            self._entries.clear()

    def _discard(
        self,
        image_key: int,
        expired_reference: ReferenceType[Image.Image],
    ) -> None:
        with self._lock:
            entry = self._entries.get(image_key)
            if entry is not None and entry.image_reference is expired_reference:
                self._entries.pop(image_key, None)


@dataclass(frozen=True, slots=True)
class AlbumLabIndex:
    """Precomputed LAB vectors and norms used for every rendered frame."""

    vectors: ColorArray
    squared_norms: ColorArray

    @classmethod
    def from_tiles(cls, album_tiles: Sequence[AlbumTile]) -> "AlbumLabIndex":
        """Build one contiguous nearest-neighbor index for a tile collection."""
        if not album_tiles:
            raise ValueError("At least one album tile is required")
        vectors = np.ascontiguousarray(
            np.stack([tile.lab_mean for tile in album_tiles]),
            dtype=np.float32,
        )
        squared_norms = np.asarray(
            np.sum(vectors * vectors, axis=1, dtype=np.float32),
            dtype=np.float32,
        )
        return cls(vectors=vectors, squared_norms=squared_norms)

    def nearest(self, target_means: ColorArray) -> NDArray[np.intp]:
        """Return vectorized nearest-album indices for all target cells."""
        return nearest_color_indices(
            target_means,
            self.vectors,
            album_squared_norms=self.squared_norms,
        )


def rgb_to_lab(rgb: RgbArray) -> ColorArray:
    """Convert an RGB uint8 array to CIE L*a*b* using a D65 white point."""
    if rgb.shape[-1:] != (3,):
        raise ValueError("RGB input must have three channels")

    # uint8 sRGB has only 256 possible channel values. This exact lookup table
    # avoids evaluating the nonlinear transfer function for every frame pixel.
    linear = _SRGB_TO_LINEAR_LUT[rgb]
    xyz = (linear @ _RGB_TO_XYZ.T) / _D65_WHITE_POINT

    delta = 6 / 29
    transformed = np.where(
        xyz > delta**3,
        np.cbrt(xyz),
        (xyz / (3 * delta**2)) + (4 / 29),
    )
    x, y, z = np.moveaxis(transformed, -1, 0)
    return np.stack(
        ((116 * y) - 16, 500 * (x - y), 200 * (y - z)),
        axis=-1,
    ).astype(np.float32)


def calculate_cell_means(values: ColorArray, grid: GridSpec) -> ColorArray:
    """Calculate all rectangular cell means with batched NumPy reductions."""
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("Color values must have shape (height, width, 3)")

    height, width, _ = values.shape
    if grid.columns > width or grid.rows > height:
        raise ValueError("Grid dimensions cannot exceed image dimensions")

    x_edges = np.linspace(0, width, grid.columns + 1, dtype=np.intp)
    y_edges = np.linspace(0, height, grid.rows + 1, dtype=np.intp)
    left, right = x_edges[:-1], x_edges[1:]
    top, bottom = y_edges[:-1], y_edges[1:]
    row_sums = np.add.reduceat(values, top, axis=0, dtype=np.float64)
    sums = np.add.reduceat(row_sums, left, axis=1, dtype=np.float64)
    areas = (bottom - top)[:, None] * (right - left)[None, :]
    return (sums / areas[..., None]).reshape(-1, 3).astype(np.float32)


def calculate_cell_rgb_means(target_image: Image.Image, grid: GridSpec) -> ColorArray:
    """Calculate mean RGB values for all target cells."""
    rgb = np.asarray(target_image.convert("RGB"), dtype=np.uint8).astype(np.float32)
    return calculate_cell_means(rgb, grid)


def calculate_cell_lab_means(target_image: Image.Image, grid: GridSpec) -> ColorArray:
    """Convert target pixels to LAB, then calculate each cell's LAB mean."""
    rgb = np.asarray(target_image.convert("RGB"), dtype=np.uint8)
    return calculate_cell_lab_means_array(rgb, grid)


def calculate_cell_lab_means_array(rgb: RgbArray, grid: GridSpec) -> ColorArray:
    """Calculate LAB cell means directly from an RGB NumPy frame."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("RGB frame must have shape (height, width, 3) and uint8 data")
    return calculate_cell_means(rgb_to_lab(rgb), grid)


def nearest_color_indices(
    target_means: ColorArray,
    album_means: ColorArray,
    *,
    album_squared_norms: ColorArray | None = None,
) -> NDArray[np.intp]:
    """Match all cells at once using squared Euclidean matrix distances."""
    if target_means.ndim != 2 or target_means.shape[1] != 3:
        raise ValueError("Target means must have shape (cell_count, 3)")
    if album_means.ndim != 2 or album_means.shape[1] != 3 or not len(album_means):
        raise ValueError("Album means must have shape (album_count, 3)")

    if album_squared_norms is None:
        album_squared_norms = np.sum(
            album_means * album_means,
            axis=1,
            dtype=np.float32,
        )
    elif album_squared_norms.shape != (len(album_means),):
        raise ValueError("Album squared norms must match the album count")

    distances = (
        np.sum(target_means * target_means, axis=1)[:, None]
        + album_squared_norms[None, :]
        - (2 * target_means @ album_means.T)
    )
    return np.asarray(np.argmin(distances, axis=1), dtype=np.intp)


def match_artwork(
    target_image: Image.Image,
    album_tiles: Sequence[AlbumTile],
    grid: GridSpec,
) -> NDArray[np.intp]:
    """Match target-cell LAB means to cached album LAB means."""
    if not album_tiles:
        raise ValueError("At least one album tile is required")

    target_means = calculate_cell_lab_means(target_image, grid)
    return AlbumLabIndex.from_tiles(album_tiles).nearest(target_means)


def match_artwork_rgb(
    target_image: Image.Image,
    album_tiles: Sequence[AlbumTile],
    grid: GridSpec,
) -> NDArray[np.intp]:
    """Match in RGB as a retained baseline for tests and benchmarks."""
    if not album_tiles:
        raise ValueError("At least one album tile is required")

    target_means = calculate_cell_rgb_means(target_image, grid)
    album_means = np.stack([tile.rgb_mean for tile in album_tiles])
    return nearest_color_indices(target_means, album_means)
