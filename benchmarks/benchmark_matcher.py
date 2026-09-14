"""Compare the average-RGB baseline with the CIELAB mosaic matcher."""

from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from statistics import median
from time import perf_counter

import numpy as np
from PIL import Image

from app.mosaic.grid import calculate_grid
from app.mosaic.matcher import (
    AlbumTileCache,
    match_artwork,
    match_artwork_rgb,
)


def parse_args() -> Namespace:
    """Parse benchmark dimensions and workload size."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--albums", type=int, default=200)
    parser.add_argument("--tiles", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def timed_median(operation: Callable[[], object], repeats: int) -> float:
    """Return the median runtime after one unmeasured warm-up call."""
    operation()
    timings: list[float] = []
    for _ in range(repeats):
        started = perf_counter()
        operation()
        timings.append(perf_counter() - started)
    return median(timings)


def main() -> None:
    """Create a deterministic synthetic workload and print timing results."""
    args = parse_args()
    if min(args.width, args.height, args.albums, args.tiles, args.repeats) <= 0:
        raise SystemExit("All benchmark values must be positive")

    rng = np.random.default_rng(seed=20260912)
    target = Image.fromarray(
        rng.integers(0, 256, (args.height, args.width, 3), dtype=np.uint8),
    )
    album_images = [
        Image.fromarray(
            rng.integers(0, 256, (128, 128, 3), dtype=np.uint8),
        )
        for _ in range(args.albums)
    ]
    grid = calculate_grid(target.width, target.height, args.tiles)

    cache = AlbumTileCache()
    started = perf_counter()
    album_tiles = cache.get_many(album_images)
    feature_build_time = perf_counter() - started

    rgb_time = timed_median(
        lambda: match_artwork_rgb(target, album_tiles, grid),
        args.repeats,
    )
    lab_time = timed_median(
        lambda: match_artwork(target, album_tiles, grid),
        args.repeats,
    )
    rgb_matches = match_artwork_rgb(target, album_tiles, grid)
    lab_matches = match_artwork(target, album_tiles, grid)
    agreement = float(np.mean(rgb_matches == lab_matches))

    print(
        f"workload: {args.width}x{args.height}, {len(album_tiles)} albums, "
        f"{grid.tile_count} cells, {args.repeats} timed repeats"
    )
    print(f"one-time cached album feature build: {feature_build_time * 1000:.2f} ms")
    print(f"RGB baseline median: {rgb_time * 1000:.2f} ms")
    print(f"CIELAB matcher median: {lab_time * 1000:.2f} ms")
    print(f"CIELAB/RGB runtime ratio: {lab_time / rgb_time:.2f}x")
    print(f"same selected album: {agreement:.1%} of cells")


if __name__ == "__main__":
    main()
