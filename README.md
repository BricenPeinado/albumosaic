# Albumosaic

Albumosaic is an open-source Python application for turning a video into a
photomosaic whose tiles are album covers from a Spotify playlist.

This repository contains a Gradio application, a still-image mosaic renderer,
frame-by-frame video rendering with FFmpeg audio restoration, provider-neutral
playlist ingestion, and a validated artwork cache. Local Exportify CSV
ingestion is available. Direct Spotify ingestion is intentionally not yet
configured and the application does not scrape Spotify pages.

> **Project status:** Albumosaic is pre-release software. Still-image and video
> rendering, Exportify CSV ingestion, caching, and the Gradio workflow are
> implemented. The default UI validates Spotify playlist URLs but cannot fetch
> their contents until a live `PlaylistSource` provider is configured.

## Intended pipeline

1. Accept a public Spotify playlist URL and a source video.
2. Resolve the playlist's unique albums and cache their cover artwork.
3. Decode the source video frame by frame with OpenCV.
4. Divide each frame into a grid and match each region to an album cover using
   NumPy-based color comparisons.
5. Render the matched covers into photomosaic frames with Pillow and NumPy.
6. Encode the frames as MP4, then use FFmpeg to restore the source audio.
7. Show progress and expose the result for preview and download in Gradio.

The tile-count control remains disabled until playlist resolution succeeds,
then ranges from `2` to the number of unique albums. Changing the source video
or density refreshes an aspect-aware grid estimate.

## Architecture

```text
albumosaic/
├── app/
│   ├── main.py              # Application entry point
│   ├── workflow.py          # Provider-neutral application orchestration
│   ├── playlist/
│   │   ├── source.py        # Provider-neutral ingestion interface
│   │   ├── exportify.py     # Local Exportify CSV source
│   │   ├── spotify.py       # Pure Spotify playlist URL validation
│   │   ├── parser.py        # Provider-neutral ingestion entry point
│   │   ├── models.py        # Playlist and album domain models
│   │   └── artwork.py       # Artwork download and cache management
│   ├── mosaic/
│   │   ├── matcher.py       # Frame-region to artwork matching
│   │   ├── grid.py          # Mosaic grid calculations
│   │   └── renderer.py      # Mosaic frame composition
│   ├── video/
│   │   ├── reader.py        # Video metadata and frame decoding
│   │   ├── writer.py        # Silent video encoding
│   │   ├── renderer.py      # Streaming video mosaic orchestration
│   │   ├── errors.py        # Video processing errors
│   │   └── audio.py         # FFmpeg audio restoration
│   └── ui/
│       ├── app.py           # Declarative Gradio layout and event bindings
│       └── controller.py    # Thin Gradio-to-workflow adapter
├── cache/
│   ├── artwork/             # Validated square RGB PNG files (ignored)
│   └── metadata/            # Cache provenance and hashes (ignored)
├── output/                  # Generated videos (ignored)
├── tests/
├── requirements.txt
└── pytest.ini
```

The modules are split by responsibility so Spotify access, image matching, and
video I/O can be developed and tested independently. Business orchestration is
implemented in `app.workflow`; the Gradio layout contains no playlist, image,
or video processing logic.

`Playlist` preserves its track sequence but exposes only deduplicated albums.
Album IDs are authoritative when present; otherwise identity falls back to a
Unicode-normalized, case-insensitive artist and album-title key.

## Playlist ingestion

Playlist providers implement the `PlaylistSource.resolve_playlist(...)`
interface. The first implementation reads an Exportify CSV entirely locally:

```python
from app.playlist.parser import resolve_playlist

playlist = resolve_playlist("my-playlist.csv")
print(playlist.unique_album_count)
```

The parser reads track and album URIs, names, artists, artwork URLs, release
dates, track numbering and duration, preview URLs, explicit/popularity fields,
ISRC, and added metadata when those columns are present. It makes no Spotify
requests.

Spotify playlist URLs can be validated independently with
`parse_spotify_playlist_url(...)`. This extracts a base-62 playlist ID and
returns a canonical `SpotifyPlaylistReference`; it does not fetch or scrape the
URL.

## Artwork cache

`ArtworkCache` accepts unique `Album` objects and returns local artwork paths.
It uses stable album-identity keys, validates HTTP responses and image content,
stores original-resolution square artwork as RGB PNG, and writes JSON metadata
atomically. Valid cached images are reused without another request.

Downloads use configurable timeouts and retries, a 20 MB response limit, and a
bounded thread pool with four workers by default. Playlist parsing remains
independent from artwork retrieval.

The renderer center-crops album artwork to square reusable tiles, calculates
mean CIELAB color features, and uses perceptual nearest-color matching for each
target cell. `AlbumTileCache` retains normalized covers, RGB/LAB means, and
resized tile variants so video frames can reuse them.

## Requirements

- Python 3.11 or newer
- FFmpeg available on `PATH`

FFmpeg is a system dependency and is not installed by `pip`.

## Development setup

Install FFmpeg first. For example, use `brew install ffmpeg` on macOS or
`sudo apt-get install ffmpeg` on Ubuntu, then verify both required executables:

```bash
ffmpeg -version
ffprobe -version
```

Create the environment and install runtime dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m app.main --help
python -m app.main --host 127.0.0.1 --port 7860
```

Open the displayed local URL to use the interface. The page resolves playlists
through an injected `PlaylistSource`, updates density from the unique-album
count, previews the calculated grid, streams all six generation stages, and
exposes the completed MP4 for preview and download.

The default source validates Spotify playlist URLs but deliberately stops
before making a network request. Configure a live `PlaylistSource` when direct
Spotify access is added; the workflow and UI do not require changes.

Install development tools and run the same release gates as CI with:

```bash
python -m pip install -r requirements-dev.txt
ruff check .
ruff format --check .
mypy app
pytest --cov=app --cov-report=term-missing --cov-fail-under=80
```

To exercise the still-image renderer with local files, put arbitrary JPEG or
PNG covers in a folder and run:

```bash
python -m tests.manual.render_folder target.jpg album_covers/ \
  --tiles 100 --output output/manual_mosaic.png
```

The tile count is approximate because a rectangular grid may need a nearby
cell count to preserve the target image's aspect ratio.

## Matcher benchmark

Run the deterministic RGB-baseline versus CIELAB benchmark with:

```bash
python -m benchmarks.benchmark_matcher
```

The benchmark reports the one-time cached album-feature cost separately from
warm per-frame matching. Workload dimensions, album count, tile count, and
repeat count are configurable through command-line options.

## Video rendering

`app.video.renderer.render_video` accepts an input video, album-cover images or
prepared `AlbumTile` objects, a target tile count, and an MP4 output path. It
reads and writes one frame at a time, preserves the source dimensions and FPS,
and reports progress through a `(processed_frames, total_frames)` callback.

The renderer builds an intermediate silent video, then uses FFmpeg to produce a
broadly compatible H.264/yuv420p MP4. Existing MP4-compatible audio is copied
without re-encoding; incompatible audio is encoded as AAC. Silent sources
remain silent. Temporary files are removed after success or failure.

Frame-invariant render plans precompute the grid and stacked album LAB vectors,
retain resized BGR tile arrays by album and cell dimensions, and perform
target-cell reduction and nearest-neighbor matching in vectorized NumPy. The
video path works directly with OpenCV BGR arrays, avoiding PIL conversions for
each target and output frame. Frames continue to stream directly between the
decoder and video writer; no individual frame images are written to disk.

Pass a callback to `render_video(..., timing_callback=...)` to receive a
`VideoRenderTiming` summary containing decode, target-color calculation,
matching, composition, and encoding totals plus per-frame formatting.

### Original-video blend

Albumosaic can mix the original frame into the completed mosaic after album
matching and full-size composition. Pass `blend_alpha` to the video or still
image renderer:

```python
render_video(
    input_path,
    album_tiles,
    tile_count=96,
    output_path="output/mosaic.mp4",
    blend_alpha=0.15,
)
```

The value ranges from `0.0` to `1.0`; the UI exposes the intentionally narrower
0–50% range. As a practical guide:

- **0%** gives the strongest album-cover appearance.
- **10–20%** is often a good balance between mosaic texture and recognition.
- **30%+** increasingly resembles the original video.

The original frame is never blended before target analysis, so changing this
setting does not affect which album cover is selected for any cell. The default
is 0%, preserving existing output.

To reproduce the before/after profile on deterministic synthetic video data:

```bash
python -m benchmarks.benchmark_video_rendering
```

The benchmark retains the previous integral-image/PIL frame path locally for
comparison and prints the same five timing categories for both implementations.

## User interface

The UI includes a Spotify playlist field, drag-and-drop video upload,
playlist-driven mosaic density, an aspect-aware grid estimate, a Generate
control, six-stage progress with percentage and frame counts, and MP4 preview
and download outputs.

## Privacy and generated files

Videos, playlist exports, cached artwork, Gradio uploads, and generated output
are excluded by `.gitignore`. Review staged files before every commit; users
remain responsible for having permission to process and share their source
media and artwork.

## License and attribution

Albumosaic source code is released under the [MIT License](LICENSE). Runtime
and development dependencies retain their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Spotify is a trademark of
Spotify AB, and this project is not affiliated with or endorsed by Spotify.
