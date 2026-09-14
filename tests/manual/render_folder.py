"""Render a still-image mosaic using arbitrary local cover images."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from PIL import Image, ImageOps

from app.mosaic.matcher import AlbumTile
from app.mosaic.renderer import render_mosaic

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> Namespace:
    """Parse manual-renderer command-line arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="Target JPEG or PNG image")
    parser.add_argument("album_folder", type=Path, help="Folder of cover images")
    parser.add_argument(
        "--tiles",
        type=int,
        default=100,
        help="Approximate number of tiles (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/manual_mosaic.png"),
        help="Output JPEG or PNG path",
    )
    return parser.parse_args()


def load_image(path: Path) -> Image.Image:
    """Load an oriented RGB image without retaining an open file handle."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def main() -> None:
    """Load local images and write one mosaic preview."""
    args = parse_args()
    album_paths = sorted(
        path
        for path in args.album_folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not album_paths:
        raise SystemExit(f"No JPEG or PNG images found in {args.album_folder}")

    target = load_image(args.target)
    albums = [
        AlbumTile(identifier=path.stem, image=load_image(path)) for path in album_paths
    ]
    result = render_mosaic(target, albums, args.tiles)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(f"Saved {result.size[0]}x{result.size[1]} mosaic to {args.output}")


if __name__ == "__main__":
    main()
