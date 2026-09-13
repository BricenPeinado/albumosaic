# Albumosaic

Albumosaic is an open-source Python application for turning a video into a
photomosaic whose tiles are album covers from a Spotify playlist.

This repository currently contains the project foundation and a minimal Gradio
interface. Spotify ingestion, artwork downloading, mosaic rendering, and final
video encoding are intentionally left as typed placeholders for later stages.

## Intended pipeline

1. Accept a public Spotify playlist URL and a source video.
2. Resolve the playlist's unique albums and cache their cover artwork.
3. Decode the source video frame by frame with OpenCV.
4. Divide each frame into a grid and match each region to an album cover using
   NumPy-based color comparisons.
5. Render the matched covers into photomosaic frames with Pillow and NumPy.
6. Encode the frames as MP4, then use FFmpeg to restore the source audio.
7. Show progress and expose the result for preview and download in Gradio.

The tile-count control will ultimately range from `2` to the number of unique
albums in the loaded playlist. Until playlist ingestion is implemented, the UI
shows the control in a disabled placeholder state.

## Architecture

```text
albumosaic/
├── app/
│   ├── main.py              # Application entry point
│   ├── playlist/
│   │   ├── parser.py        # Spotify playlist ingestion
│   │   ├── models.py        # Playlist and album domain models
│   │   └── artwork.py       # Artwork download and cache management
│   ├── mosaic/
│   │   ├── matcher.py       # Frame-region to artwork matching
│   │   ├── grid.py          # Mosaic grid calculations
│   │   └── renderer.py      # Mosaic frame composition
│   ├── video/
│   │   ├── reader.py        # Video metadata and frame decoding
│   │   ├── writer.py        # Silent video encoding
│   │   └── audio.py         # FFmpeg audio restoration
│   └── ui/
│       └── app.py           # Gradio interface
├── cache/                   # Downloaded album artwork (ignored)
├── output/                  # Generated videos (ignored)
├── tests/
├── requirements.txt
└── pytest.ini
```

The modules are split by responsibility so Spotify access, image matching, and
video I/O can be developed and tested independently. The UI should orchestrate
these services rather than contain processing logic.

## Requirements

- Python 3.11 or newer
- FFmpeg available on `PATH`

FFmpeg is a system dependency and is not installed by `pip`.

## Development setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m app.main
```

Open the local URL printed by Gradio. The controls render, but generation is a
placeholder at this stage.

Run the tests with:

```bash
pytest
```

## Planned UI

- Spotify playlist URL input
- Video upload input
- Album-cover tile-count slider
- Generate button
- Progress information
- Generated MP4 preview and download
# albumosaic
music library -> mosaic video
