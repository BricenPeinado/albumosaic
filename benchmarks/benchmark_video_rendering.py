"""Profile the legacy and optimized frame pipelines on identical video data."""

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from app.mosaic.grid import GridSpec, calculate_grid
from app.mosaic.matcher import AlbumTile, nearest_color_indices
from app.mosaic.renderer import MosaicRenderPlan
from app.video.reader import VideoReader
from app.video.writer import SilentVideoWriter

_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)
_D65_WHITE_POINT = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)


@dataclass(slots=True)
class StageTimes:
    """Benchmark timing buckets matching the production timing output."""

    decode: float = 0.0
    target_color: float = 0.0
    matching: float = 0.0
    composition: float = 0.0
    encoding: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.decode
            + self.target_color
            + self.matching
            + self.composition
            + self.encoding
        )


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--albums", type=int, default=80)
    parser.add_argument("--tiles", type=int, default=96)
    return parser.parse_args()


def _legacy_cell_means(rgb: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Retain the pre-optimization integral-image implementation for profiling."""
    values = _legacy_rgb_to_lab(rgb)
    integral = np.pad(
        values.cumsum(axis=0, dtype=np.float64).cumsum(axis=1, dtype=np.float64),
        ((1, 0), (1, 0), (0, 0)),
    )
    x_edges = np.linspace(0, rgb.shape[1], grid.columns + 1, dtype=np.intp)
    y_edges = np.linspace(0, rgb.shape[0], grid.rows + 1, dtype=np.intp)
    left, right = x_edges[:-1], x_edges[1:]
    top, bottom = y_edges[:-1], y_edges[1:]
    sums = (
        integral[bottom[:, None], right[None, :]]
        - integral[top[:, None], right[None, :]]
        - integral[bottom[:, None], left[None, :]]
        + integral[top[:, None], left[None, :]]
    )
    areas = (bottom - top)[:, None] * (right - left)[None, :]
    return (sums / areas[..., None]).reshape(-1, 3).astype(np.float32)


def _legacy_rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Retain the original per-pixel nonlinear sRGB conversion."""
    srgb = rgb.astype(np.float32) / 255.0
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
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


def _legacy_compose(
    size: tuple[int, int],
    grid: GridSpec,
    tiles: tuple[AlbumTile, ...],
    matches: np.ndarray,
) -> np.ndarray:
    output = Image.new("RGB", size)
    x_edges = np.linspace(0, size[0], grid.columns + 1, dtype=np.intp)
    y_edges = np.linspace(0, size[1], grid.rows + 1, dtype=np.intp)
    for cell_index, artwork_index in enumerate(matches):
        row, column = divmod(cell_index, grid.columns)
        left, right = x_edges[column : column + 2]
        top, bottom = y_edges[row : row + 2]
        tile = tiles[int(artwork_index)].resized((int(right - left), int(bottom - top)))
        output.paste(tile, (int(left), int(top)))
    return cv2.cvtColor(np.asarray(output), cv2.COLOR_RGB2BGR)


def _profile_legacy(
    input_path: Path,
    output_path: Path,
    tiles: tuple[AlbumTile, ...],
    target_tiles: int,
) -> StageTimes:
    times = StageTimes()
    with VideoReader(input_path) as reader:
        metadata = reader.metadata
        with SilentVideoWriter(
            output_path,
            metadata.frames_per_second,
            (metadata.width, metadata.height),
        ) as writer:
            while True:
                started = perf_counter()
                try:
                    frame = next(reader)
                except StopIteration:
                    break
                times.decode += perf_counter() - started

                started = perf_counter()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                grid = calculate_grid(metadata.width, metadata.height, target_tiles)
                target_means = _legacy_cell_means(rgb, grid)
                times.target_color += perf_counter() - started

                started = perf_counter()
                album_means = np.stack([tile.lab_mean for tile in tiles])
                matches = nearest_color_indices(target_means, album_means)
                times.matching += perf_counter() - started

                started = perf_counter()
                output = _legacy_compose(
                    (metadata.width, metadata.height), grid, tiles, matches
                )
                times.composition += perf_counter() - started

                started = perf_counter()
                writer.write(output)
                times.encoding += perf_counter() - started
    return times


def _profile_optimized(
    input_path: Path,
    output_path: Path,
    tiles: tuple[AlbumTile, ...],
    target_tiles: int,
) -> StageTimes:
    times = StageTimes()
    with VideoReader(input_path) as reader:
        metadata = reader.metadata
        plan = MosaicRenderPlan(metadata.width, metadata.height, tiles, target_tiles)
        with SilentVideoWriter(
            output_path,
            metadata.frames_per_second,
            (metadata.width, metadata.height),
        ) as writer:
            while True:
                started = perf_counter()
                try:
                    frame = next(reader)
                except StopIteration:
                    break
                times.decode += perf_counter() - started

                started = perf_counter()
                target_means = plan.target_cell_means(frame)
                times.target_color += perf_counter() - started

                started = perf_counter()
                matches = plan.match(target_means)
                times.matching += perf_counter() - started

                started = perf_counter()
                output = plan.compose_bgr(matches)
                times.composition += perf_counter() - started

                started = perf_counter()
                writer.write(output)
                times.encoding += perf_counter() - started
    return times


def _write_source(path: Path, args: Namespace, rng: np.random.Generator) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        24.0,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV MJPG encoder is unavailable")
    try:
        for _ in range(args.frames):
            writer.write(
                rng.integers(
                    0,
                    256,
                    (args.height, args.width, 3),
                    dtype=np.uint8,
                )
            )
    finally:
        writer.release()


def _verify_equivalent_output(
    input_path: Path,
    tiles: tuple[AlbumTile, ...],
    target_tiles: int,
) -> None:
    """Fail the benchmark if the optimized frame changes any output pixel."""
    with VideoReader(input_path) as reader:
        frame = next(reader)
        metadata = reader.metadata
    grid = calculate_grid(metadata.width, metadata.height, target_tiles)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    target_means = _legacy_cell_means(rgb, grid)
    album_means = np.stack([tile.lab_mean for tile in tiles])
    matches = nearest_color_indices(target_means, album_means)
    legacy = _legacy_compose((metadata.width, metadata.height), grid, tiles, matches)
    optimized = MosaicRenderPlan(
        metadata.width,
        metadata.height,
        tiles,
        target_tiles,
    ).render_bgr(frame)
    if not np.array_equal(legacy, optimized):
        raise RuntimeError("Optimized output differs from the legacy renderer")


def _print_results(name: str, times: StageTimes, frames: int) -> None:
    print(f"{name}: {times.total:.3f}s measured total")
    for label, seconds in (
        ("frame decode", times.decode),
        ("target color calculation", times.target_color),
        ("matching", times.matching),
        ("composition", times.composition),
        ("frame encoding", times.encoding),
    ):
        print(f"  {label}: {seconds:.3f}s ({seconds * 1000 / frames:.2f} ms/frame)")


def main() -> None:
    args = parse_args()
    if min(args.width, args.height, args.frames, args.albums, args.tiles) <= 0:
        raise SystemExit("All benchmark values must be positive")
    rng = np.random.default_rng(20260913)
    tiles = tuple(
        AlbumTile(
            f"album-{index}",
            Image.fromarray(rng.integers(0, 256, (192, 192, 3), dtype=np.uint8)),
        )
        for index in range(args.albums)
    )

    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        input_path = root / "input.avi"
        _write_source(input_path, args, rng)
        _verify_equivalent_output(input_path, tiles, args.tiles)

        legacy = _profile_legacy(input_path, root / "legacy.avi", tiles, args.tiles)
        optimized = _profile_optimized(
            input_path, root / "optimized.avi", tiles, args.tiles
        )

    print(
        f"workload: {args.width}x{args.height}, {args.frames} frames, "
        f"{args.albums} albums, target {args.tiles} tiles"
    )
    print("pixel equivalence: verified")
    _print_results("before (legacy)", legacy, args.frames)
    _print_results("after (optimized)", optimized, args.frames)
    print(f"speedup: {legacy.total / optimized.total:.2f}x")


if __name__ == "__main__":
    main()
