"""Declarative Gradio interface for Albumosaic."""

from typing import cast

import gradio as gr

from app.ui.controller import AlbumosaicUIController, format_blend_percentage
from app.workflow import AlbumosaicWorkflow

CSS = """
.gradio-container { max-width: 1080px !important; margin: 0 auto !important; }
.albumosaic-shell { padding-top: 2.25rem; }
.albumosaic-kicker { color: #f59e0b; font-size: .78rem; font-weight: 700;
  letter-spacing: .16em; text-transform: uppercase; margin-bottom: .4rem; }
.albumosaic-title h1 { font-size: clamp(2.5rem, 7vw, 5.25rem) !important;
  line-height: .95 !important; letter-spacing: -.055em !important; margin: 0 !important; }
.albumosaic-title p { color: #a1a1aa; font-size: 1.08rem; margin-top: 1rem; }
.albumosaic-card { border: 1px solid rgba(255,255,255,.10) !important;
  border-radius: 18px !important; padding: 1.1rem !important;
  background: rgba(24,24,27,.72) !important; }
.albumosaic-summary { min-height: 2.1rem; color: #d4d4d8; }
.albumosaic-grid { color: #fbbf24; }
.albumosaic-button { min-height: 3.2rem; font-weight: 700 !important; }
.albumosaic-progress input[type=range] { accent-color: #f59e0b; }
.albumosaic-output { min-height: 260px; }
footer { display: none !important; }
"""

THEME = gr.themes.Base(primary_hue="orange", neutral_hue="zinc")


def build_interface(workflow: AlbumosaicWorkflow | None = None) -> gr.Blocks:
    """Build the Albumosaic interface without launching it."""
    active_workflow = workflow or AlbumosaicWorkflow()
    controller = AlbumosaicUIController(active_workflow)

    with gr.Blocks(title="Albumosaic") as interface:
        playlist_state = gr.State(value=None)

        with gr.Column(elem_classes="albumosaic-shell"):
            gr.HTML('<div class="albumosaic-kicker">Video photomosaics</div>')
            gr.Markdown(
                "# Albumosaic\nTurn your Spotify playlist into the pixels of a video.",
                elem_classes="albumosaic-title",
            )

            with gr.Group(elem_classes="albumosaic-card"):
                connect_spotify = gr.Button("Connect Spotify")
                spotify_status = gr.Markdown(active_workflow.playlist_connection_status)
                playlist_url = gr.Textbox(
                    label="Spotify playlist",
                    placeholder="https://open.spotify.com/playlist/…",
                    info="Playlist details are resolved when you leave this field.",
                )
                playlist_summary = gr.Markdown(
                    "Paste a Spotify playlist URL to begin.",
                    elem_classes="albumosaic-summary",
                )
                exportify_csv = gr.File(
                    label="Or use an Exportify CSV",
                    file_types=[".csv"],
                    type="filepath",
                )

                source_video = gr.Video(
                    label="Video",
                    sources=["upload"],
                    format="mp4",
                )

                density = gr.Slider(
                    minimum=2,
                    maximum=3,
                    value=2,
                    step=1,
                    label="Mosaic density",
                    info="Number of album-cover tiles visible in each frame.",
                    interactive=False,
                )
                unique_per_frame = gr.Checkbox(
                    value=False,
                    label="Don't repeat albums in the same frame",
                    info=(
                        "Try to use every album only once per frame. This can "
                        "increase variety but may make the mosaic slightly less "
                        "color-accurate."
                    ),
                )
                grid_summary = gr.Markdown(
                    "Mosaic grid: resolve a playlist and add a video.",
                    elem_classes="albumosaic-grid",
                )

                original_blend = gr.Slider(
                    minimum=0,
                    maximum=50,
                    value=0,
                    step=1,
                    label="Original video blend",
                    info=(
                        "Blend some of the original video into the mosaic. "
                        "Higher percentages make the source video easier to recognize."
                    ),
                )
                blend_summary = gr.Markdown("Selected blend: **0%**")

                generate = gr.Button(
                    "Generate Mosaic",
                    variant="primary",
                    interactive=False,
                    elem_classes="albumosaic-button",
                )

            with gr.Group(elem_classes="albumosaic-card"):
                gr.Markdown("## Progress")
                stage = gr.Markdown("Ready when you are.")
                percentage = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    label="Overall progress (%)",
                    interactive=False,
                    elem_classes="albumosaic-progress",
                )
                frames = gr.Markdown("Frames processed: —")

            with gr.Group(elem_classes="albumosaic-card"):
                gr.Markdown("## Output")
                result_video = gr.Video(
                    label="Mosaic preview",
                    format="mp4",
                    interactive=False,
                    elem_classes="albumosaic-output",
                )
                download = gr.File(label="Download MP4", interactive=False)

        resolution_outputs = [
            playlist_state,
            playlist_summary,
            density,
            grid_summary,
            generate,
            stage,
            percentage,
            frames,
        ]
        playlist_url.change(
            fn=controller.resolve_playlist,
            inputs=[playlist_url, source_video, density, unique_per_frame],
            outputs=resolution_outputs,
        )
        exportify_csv.change(
            fn=controller.resolve_exportify,
            inputs=[exportify_csv, source_video, density, unique_per_frame],
            outputs=resolution_outputs,
        )
        connect_spotify.click(
            fn=controller.connect_spotify,
            outputs=spotify_status,
        )
        source_video.change(
            fn=controller.update_readiness,
            inputs=[source_video, density, playlist_state, unique_per_frame],
            outputs=[grid_summary, generate],
        )
        density.change(
            fn=controller.update_readiness,
            inputs=[source_video, density, playlist_state, unique_per_frame],
            outputs=[grid_summary, generate],
        )
        unique_per_frame.change(
            fn=controller.update_readiness,
            inputs=[source_video, density, playlist_state, unique_per_frame],
            outputs=[grid_summary, generate],
        )
        original_blend.change(
            fn=format_blend_percentage,
            inputs=original_blend,
            outputs=blend_summary,
        )
        generate.click(
            fn=controller.generate,
            inputs=[
                playlist_state,
                source_video,
                density,
                original_blend,
                unique_per_frame,
            ],
            outputs=[stage, percentage, frames, result_video, download, generate],
        )

    return cast(gr.Blocks, interface)
